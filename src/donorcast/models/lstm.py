"""Global PyTorch LSTM forecasting model for DonorCast (Task 3.4).

Implements M3:
- Global LSTM in PyTorch (CPU only).
- Input: Past 56 days of daily donations (scaled per series) plus categorical embeddings
  for facility (22 sites) and blood group (4 groups).
- Target Calendar Features: 17 calendar features for the 14 target days (t+1..t+14)
  concatenated right before the final linear output layer.
- Output: 14 direct multi-horizon values.
- Same training origins and splits as LightGBM/SARIMA. Early stopping on the last 6 months
  of training data (2022-07-01 to 2022-12-31).
- Fixed seeds for deterministic execution.
- Model saving to models/lstm/v001/ containing model.pt and config.json.
- Evaluated on validation split via evaluate.py, writing reports/results_lstm_val.md.
"""

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from donorcast.calendar import build_calendar_dataframe, load_facility_state_mapping
from donorcast.config import (
    CALENDAR_PROCESSED_FILE,
    DATA_PROCESSED_DIR,
    EVAL_ORIGIN_WEEKDAY,
    FACILITY_STATE_FILE,
    HORIZON,
    PROJECT_ROOT,
    REPORTS_DIR,
    SEED,
    TRAIN_END,
    TRAIN_ORIGIN_STEP,
    TRAIN_START,
    VAL_END,
    VAL_START,
)
from donorcast.evaluate import evaluate
from donorcast.features import CALENDAR_FEATURE_COLS, get_all_eval_origins, get_all_train_origins

LOOKBACK = 56
CAL_DIM = len(CALENDAR_FEATURE_COLS)


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SeriesScaler:
    """Computes and stores per-series (facility, group) mean and standard deviation."""

    def __init__(self):
        self.means: dict[tuple[str, str], float] = {}
        self.stds: dict[tuple[str, str], float] = {}

    def fit(self, long_df: pd.DataFrame, train_end: str = TRAIN_END) -> None:
        """Fit scaler parameters strictly on training data <= train_end."""
        train_df = long_df[long_df["date"] <= train_end]
        grouped = train_df.groupby(["facility", "group"], observed=False)["donations"]
        for (fac, grp), s in grouped:
            m = float(s.mean())
            std = float(s.std())
            self.means[(str(fac), str(grp))] = m
            self.stds[(str(fac), str(grp))] = std if std > 1e-5 else 1.0

    def transform(self, values: np.ndarray, fac: str, grp: str) -> np.ndarray:
        """Standardize raw donation array."""
        m = self.means.get((fac, grp), 0.0)
        std = self.stds.get((fac, grp), 1.0)
        return (values - m) / std

    def inverse_transform(self, scaled_values: np.ndarray, fac: str, grp: str) -> np.ndarray:
        """Unscale predictions back to donation counts, clipping at >= 0."""
        m = self.means.get((fac, grp), 0.0)
        std = self.stds.get((fac, grp), 1.0)
        raw = scaled_values * std + m
        return np.maximum(0.0, raw)


class CalendarScaler:
    """Normalizes continuous/integer calendar features."""

    def __init__(self):
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None

    def fit(self, cal_matrix: np.ndarray) -> None:
        self.means = np.mean(cal_matrix, axis=0, keepdims=True)
        self.stds = np.std(cal_matrix, axis=0, keepdims=True)
        self.stds = np.where(self.stds > 1e-5, self.stds, 1.0)

    def transform(self, cal_matrix: np.ndarray) -> np.ndarray:
        if self.means is None or self.stds is None:
            return cal_matrix
        return (cal_matrix - self.means) / self.stds


class DonorLSTM(nn.Module):
    """Global PyTorch LSTM direct multi-horizon forecasting network."""

    def __init__(
        self,
        num_facilities: int = 22,
        num_groups: int = 4,
        embed_fac_dim: int = 8,
        embed_grp_dim: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        cal_dim: int = CAL_DIM,
    ):
        super().__init__()
        self.facility_embed = nn.Embedding(num_facilities, embed_fac_dim)
        self.group_embed = nn.Embedding(num_groups, embed_grp_dim)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        in_dim = hidden_size + embed_fac_dim + embed_grp_dim + (horizon * cal_dim)
        self.head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        fac_idx: torch.Tensor,
        grp_idx: torch.Tensor,
        cal_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        x_seq: (batch_size, 56, 1)
        fac_idx: (batch_size,)
        grp_idx: (batch_size,)
        cal_feats: (batch_size, 14 * CAL_DIM)
        """
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]  # (batch_size, hidden_size)

        f_emb = self.facility_embed(fac_idx)  # (batch_size, embed_fac_dim)
        g_emb = self.group_embed(grp_idx)  # (batch_size, embed_grp_dim)

        cat_vec = torch.cat([h_last, f_emb, g_emb, cal_feats], dim=-1)
        out = self.head(cat_vec)  # (batch_size, horizon)
        return out


class DonorDataset(Dataset):
    """PyTorch Dataset for LSTM training and evaluation."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
    ):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        return {
            "x_seq": torch.tensor(s["x_seq"], dtype=torch.float32).unsqueeze(-1),  # (56, 1)
            "fac_idx": torch.tensor(s["fac_idx"], dtype=torch.long),
            "grp_idx": torch.tensor(s["grp_idx"], dtype=torch.long),
            "cal_feats": torch.tensor(s["cal_feats"], dtype=torch.float32),  # (14 * CAL_DIM,)
            "y_target": torch.tensor(s["y_target"], dtype=torch.float32),  # (14,)
        }


def prepare_lstm_data(
    long_df: pd.DataFrame | None = None,
    calendar_df: pd.DataFrame | None = None,
    facility_state_df: pd.DataFrame | None = None,
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str], np.ndarray],
    dict[str, int],
    dict[str, int],
    list[str],
    dict[tuple[str, str], dict[str, np.ndarray]],
    SeriesScaler,
    CalendarScaler,
]:
    """Load and index time series and calendar structures for sequence building."""
    if long_df is None:
        long_path = DATA_PROCESSED_DIR / "long.parquet"
        if not long_path.exists():
            from donorcast.clean import clean_data

            clean_data()
        long_df = pd.read_parquet(long_path)

    if calendar_df is None:
        cal_path = CALENDAR_PROCESSED_FILE
        if not cal_path.exists():
            calendar_df = build_calendar_dataframe()
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            calendar_df.to_parquet(cal_path, index=False)
        else:
            calendar_df = pd.read_parquet(cal_path)

    if facility_state_df is None:
        facility_state_df = load_facility_state_mapping(FACILITY_STATE_FILE)

    fac_to_state = dict(
        zip(facility_state_df["facility"], facility_state_df["state"], strict=False)
    )

    facilities = sorted(long_df["facility"].unique())
    groups = sorted(long_df["group"].unique())
    fac_to_idx = {f: i for i, f in enumerate(facilities)}
    grp_to_idx = {g: i for i, g in enumerate(groups)}

    all_dates = sorted(long_df["date"].unique())

    # Fit series scaler
    scaler = SeriesScaler()
    scaler.fit(long_df, train_end=TRAIN_END)

    # Build matrix of daily donations per (facility, group)
    series_matrices: dict[tuple[str, str], np.ndarray] = {}
    scaled_series_matrices: dict[tuple[str, str], np.ndarray] = {}

    for fac in facilities:
        for grp in groups:
            sub = long_df[(long_df["facility"] == fac) & (long_df["group"] == grp)].sort_values(
                "date"
            )
            arr = sub["donations"].values.astype(np.float32)
            series_matrices[(fac, grp)] = arr
            scaled_series_matrices[(fac, grp)] = scaler.transform(arr, fac, grp)

    # Build calendar feature matrix per state indexed by date
    cal_dict: dict[tuple[str, str], np.ndarray] = {}
    raw_cal_rows = []
    for row in calendar_df.itertuples():
        arr = np.array([getattr(row, col) for col in CALENDAR_FEATURE_COLS], dtype=np.float32)
        cal_dict[(row.state, row.date)] = arr
        raw_cal_rows.append(arr)

    raw_cal_mat = np.vstack(raw_cal_rows)
    cal_scaler = CalendarScaler()
    cal_scaler.fit(raw_cal_mat)

    # Pre-scale calendar feature lookup
    scaled_cal_dict: dict[tuple[str, str], np.ndarray] = {}
    for key, arr in cal_dict.items():
        scaled_cal_dict[key] = cal_scaler.transform(arr.reshape(1, -1)).flatten()

    facility_cal_matrices: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for fac in facilities:
        st = fac_to_state[fac]
        for grp in groups:
            # Construct array of shape (n_dates, CAL_DIM) for fast lookup
            cal_arr = np.zeros((len(all_dates), CAL_DIM), dtype=np.float32)
            for i, d in enumerate(all_dates):
                if (st, d) in scaled_cal_dict:
                    cal_arr[i] = scaled_cal_dict[(st, d)]
            facility_cal_matrices[(fac, grp)] = {"cal": cal_arr}

    return (
        series_matrices,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
        scaler,
        cal_scaler,
    )


def build_samples_for_origins(
    origin_dates: list[str],
    scaled_series_matrices: dict[tuple[str, str], np.ndarray],
    fac_to_idx: dict[str, int],
    grp_to_idx: dict[str, int],
    all_dates: list[str],
    facility_cal_matrices: dict[tuple[str, str], dict[str, np.ndarray]],
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> list[dict[str, Any]]:
    """Construct dataset samples for specified origin dates."""
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    samples = []

    for origin in origin_dates:
        if origin not in date_to_idx:
            continue
        t_idx = date_to_idx[origin]
        if t_idx < lookback - 1 or t_idx + horizon >= len(all_dates):
            continue

        start_seq = t_idx - lookback + 1
        end_seq = t_idx + 1

        target_start = t_idx + 1
        target_end = t_idx + horizon + 1

        for (fac, grp), y_scaled_arr in scaled_series_matrices.items():
            x_seq = y_scaled_arr[start_seq:end_seq]
            y_target = y_scaled_arr[target_start:target_end]

            cal_mat = facility_cal_matrices[(fac, grp)]["cal"][target_start:target_end]
            cal_feats = cal_mat.flatten()

            samples.append(
                {
                    "facility": fac,
                    "group": grp,
                    "origin_date": origin,
                    "fac_idx": fac_to_idx[fac],
                    "grp_idx": grp_to_idx[grp],
                    "x_seq": x_seq,
                    "cal_feats": cal_feats,
                    "y_target": y_target,
                }
            )

    return samples


def train_lstm_model(
    train_samples: list[dict[str, Any]],
    es_samples: list[dict[str, Any]],
    num_facilities: int = 22,
    num_groups: int = 4,
    hidden_size: int = 64,
    batch_size: int = 256,
    max_epochs: int = 40,
    lr: float = 0.001,
    patience: int = 10,
    seed: int = SEED,
) -> tuple[DonorLSTM, int, float]:
    """Train DonorLSTM with early stopping on validation training slice."""
    set_seed(seed)

    train_ds = DonorDataset(train_samples)
    es_ds = DonorDataset(es_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    es_loader = DataLoader(es_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = DonorLSTM(
        num_facilities=num_facilities,
        num_groups=num_groups,
        hidden_size=hidden_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_es_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    print(f"Training LSTM model (CPU only, batch_size={batch_size}, hidden_size={hidden_size})...")

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_seq = batch["x_seq"].to(device)
            fac_idx = batch["fac_idx"].to(device)
            grp_idx = batch["grp_idx"].to(device)
            cal_feats = batch["cal_feats"].to(device)
            y_target = batch["y_target"].to(device)

            optimizer.zero_grad()
            preds = model(x_seq, fac_idx, grp_idx, cal_feats)
            loss = criterion(preds, y_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_seq)

        train_loss /= len(train_ds)

        # Early stopping evaluation
        model.eval()
        es_loss = 0.0
        with torch.no_grad():
            for batch in es_loader:
                x_seq = batch["x_seq"].to(device)
                fac_idx = batch["fac_idx"].to(device)
                grp_idx = batch["grp_idx"].to(device)
                cal_feats = batch["cal_feats"].to(device)
                y_target = batch["y_target"].to(device)

                preds = model(x_seq, fac_idx, grp_idx, cal_feats)
                loss = criterion(preds, y_target)
                es_loss += loss.item() * len(x_seq)

        es_loss /= len(es_ds)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:02d}/{max_epochs}: Train MSE = {train_loss:.5f}, ES MSE = {es_loss:.5f}"
            )

        if es_loss < best_es_loss:
            best_es_loss = es_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"LSTM training finished. Best epoch: {best_epoch} with ES MSE = {best_es_loss:.5f}")
    return model, best_epoch, best_es_loss


def predict_lstm(
    model: DonorLSTM,
    val_samples: list[dict[str, Any]],
    scaler: SeriesScaler,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Generate predictions for validation samples and return schema matching evaluate.py."""
    if not val_samples:
        return pd.DataFrame(
            columns=[
                "facility",
                "group",
                "origin_date",
                "horizon",
                "target_date",
                "target",
                "prediction",
            ]
        )

    val_ds = DonorDataset(val_samples)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in val_loader:
            x_seq = batch["x_seq"].to(device)
            fac_idx = batch["fac_idx"].to(device)
            grp_idx = batch["grp_idx"].to(device)
            cal_feats = batch["cal_feats"].to(device)

            preds = model(x_seq, fac_idx, grp_idx, cal_feats)
            all_preds.append(preds.numpy())

    preds_mat = np.vstack(all_preds)  # (N_samples, 14)

    rows = []
    for idx, s in enumerate(val_samples):
        fac = s["facility"]
        grp = s["group"]
        orig = s["origin_date"]
        orig_dt = datetime.date.fromisoformat(orig)

        scaled_pred_row = preds_mat[idx]
        unscaled_preds = scaler.inverse_transform(scaled_pred_row, fac, grp)

        unscaled_targets = scaler.inverse_transform(s["y_target"], fac, grp)

        for h in range(1, HORIZON + 1):
            target_dt = orig_dt + datetime.timedelta(days=h)
            rows.append(
                {
                    "facility": fac,
                    "group": grp,
                    "origin_date": orig,
                    "horizon": h,
                    "target_date": target_dt.isoformat(),
                    "target": float(unscaled_targets[h - 1]),
                    "prediction": float(unscaled_preds[h - 1]),
                }
            )

    return pd.DataFrame(rows)


def save_lstm_model_version(
    model: DonorLSTM,
    scaler: SeriesScaler,
    best_epoch: int,
    best_es_loss: float,
    val_wape: float,
    models_dir: Path = PROJECT_ROOT / "models" / "lstm",
) -> Path:
    """Save model checkpoint, scaler state, and config metadata to versioned folder."""
    models_dir.mkdir(parents=True, exist_ok=True)

    existing_versions = []
    for d in models_dir.iterdir():
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
            existing_versions.append(int(d.name[1:]))

    next_idx = max(existing_versions) + 1 if existing_versions else 1
    version_str = f"v{next_idx:03d}"
    version_dir = models_dir / version_str
    version_dir.mkdir(parents=True, exist_ok=False)

    # Save PyTorch checkpoint
    checkpoint = {
        "state_dict": model.state_dict(),
        "series_means": scaler.means,
        "series_stds": scaler.stds,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
    }
    torch.save(checkpoint, version_dir / "model.pt")

    # Save metadata
    config = {
        "model_name": "lstm",
        "version": version_str,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "hidden_size": 64,
        "num_layers": 1,
        "embed_fac_dim": 8,
        "embed_grp_dim": 8,
        "best_epoch": best_epoch,
        "best_es_loss": float(best_es_loss),
        "validation_wape": float(val_wape),
        "dataset_version": "v2026-09-22",
        "saved_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with open(version_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved LSTM model version {version_str} to {version_dir}")
    return version_dir


def run_lstm_evaluation(
    split: str = "val",
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Main orchestration function for LSTM model training and validation evaluation (Task 3.4).

    Steps:
    1. Load data and index time series matrices & calendar features.
    2. Build training samples (origins <= 2022-06-30 with step=7), early stopping samples (2022-07-01 to 2022-12-31),
       and validation samples (weekly Mondays 2023-01-01 to 2024-12-31).
    3. Train global PyTorch LSTM model with early stopping.
    4. Save versioned model to models/lstm/v001/.
    5. Generate predictions for validation split and evaluate via evaluate.py.
    6. Write reports/results_lstm_val.md.
    """
    start_time = datetime.datetime.now(datetime.UTC)
    set_seed(SEED)

    (
        _series_matrices,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
        scaler,
        _cal_scaler,
    ) = prepare_lstm_data()

    # Define training and validation origin sets
    train_origins = get_all_train_origins(
        train_start=TRAIN_START, train_end="2022-06-30", step=TRAIN_ORIGIN_STEP
    )
    es_origins = get_all_train_origins(
        train_start="2022-07-01", train_end=TRAIN_END, step=TRAIN_ORIGIN_STEP
    )
    val_origins = get_all_eval_origins(
        start_date=VAL_START, end_date=VAL_END, weekday=EVAL_ORIGIN_WEEKDAY
    )

    print(
        f"Building sequence samples: {len(train_origins)} train origins, "
        f"{len(es_origins)} early stopping origins, {len(val_origins)} validation origins..."
    )

    train_samples = build_samples_for_origins(
        train_origins,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )
    es_samples = build_samples_for_origins(
        es_origins,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )
    val_samples = build_samples_for_origins(
        val_origins,
        scaled_series_matrices,
        fac_to_idx,
        grp_to_idx,
        all_dates,
        facility_cal_matrices,
    )

    print(
        f"Dataset samples built: {len(train_samples):,} train, {len(es_samples):,} ES, {len(val_samples):,} val."
    )

    # Train model
    model, best_epoch, best_es_loss = train_lstm_model(
        train_samples,
        es_samples,
        num_facilities=len(fac_to_idx),
        num_groups=len(grp_to_idx),
        hidden_size=64,
        batch_size=256,
        max_epochs=40,
        lr=0.001,
        patience=10,
        seed=SEED,
    )

    # Generate predictions on validation set
    print(f"Generating predictions for validation split ({len(val_samples):,} series-origins)...")
    preds_val = predict_lstm(model, val_samples, scaler, batch_size=256)

    # Compute validation WAPE for metadata
    total_abs_err = np.sum(np.abs(preds_val["target"] - preds_val["prediction"]))
    total_actual = np.sum(preds_val["target"])
    val_wape = float(total_abs_err / total_actual) if total_actual > 0 else float("nan")

    # Save model version
    version_dir = save_lstm_model_version(
        model,
        scaler,
        best_epoch,
        best_es_loss,
        val_wape,
    )

    # Run evaluation harness
    print(f"Evaluating LSTM model on '{split}' split via evaluate.py...")
    eval_res = evaluate(preds_val, split=split, model_name="lstm", reports_dir=reports_dir)

    elapsed = (datetime.datetime.now(datetime.UTC) - start_time).total_seconds()
    print(f"Task 3.4 completed in {elapsed / 60:.2f} minutes.")

    return {
        "version_dir": str(version_dir),
        "validation_wape": val_wape,
        "elapsed_seconds": elapsed,
        "evaluation_summary": eval_res,
    }


if __name__ == "__main__":
    run_lstm_evaluation()
