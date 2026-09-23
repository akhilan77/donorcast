# DonorCast: Forecasting Blood Donation Shortfalls

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Package Manager: uv](https://img.shields.io/badge/uv-fast%20python-purple.svg)](https://astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Test Suite: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

**DonorCast** is an operational early-warning forecasting and alert system for blood donation shortfalls. Built on **20.7 years** of open data (2006 to 2026) from the Ministry of Health Malaysia (National Blood Centre / Pusat Darah Negara), DonorCast forecasts daily donations across **88 distinct time series** (22 hospital collection centres × 4 ABO blood groups) 1 to 14 days ahead and identifies upcoming 7-day shortfall windows.

Platelets last only 5 days and red blood cells 42 days. When donation volume dips during festivals, school vacations, or long weekends, hospital blood banks risk critical stockouts. DonorCast flags impending deficits with **7-day lead time** and translates tree model feature attributions into **plain-English operational reasons**, giving blood bank organizers time to schedule mobile drives and recall donors *before* inventories run dry.

---

## Key Results & Selected Model

The winning model is a global **LightGBM Tweedie Regressor** paired with pinball-loss **Quantile Models (p10 / p90)**:
- **Primary Metric (WAPE_7D)**: **16.9%** on held-out test data (beating the strong M0b 4-week weekday baseline of **21.3%** by 4.5 percentage points).
- **Daily WAPE (Horizons 1–14)**: **37.9%** (vs 45.2% baseline).
- **Shortfall Classification**: **62.0% Precision**, **48.2% Recall**, **0.54 F1 Score** on predicting 7-day deficits (< 0.8× historical 3-year median).
- **Prediction Interval**: **81.2% empirical coverage** on held-out data for the nominal 80% p10–p90 band.
- **Explainability**: SHAP explanations explain every flagged alert with its top 3 actionable drivers.

---

## Invariants & Design Principles

All pipelines strictly enforce six core architectural invariants:
1. `RAW_DATA_FROZEN`: Data in `data/raw/` matches immutable SHA-256 hashes (`hashes.json`).
2. `NO_SAME_DAY_BREAKDOWNS`: Breakdown features (mobile vs centre, student share, regular donor mix) are lagged only.
3. `FEATURES_AS_OF_ORIGIN`: Forecast features for origin date $t$ use only data dated $\le t$.
4. `TIME_ORDERED_SPLITS`: Chronological splits: Train (2006–2022) $\to$ Validation (2023–2024) $\to$ Test (2025–2026). No random shuffling.
5. `TEST_SET_TOUCHED_ONCE`: Final evaluation on held-out test data is strictly guarded and logged in `reports/final_run.json`.
6. `BASELINE_MUST_BE_BEATEN`: The selected model must beat the M0b moving average baseline on validation WAPE.

---

## Windows Setup Guide

### 1. Prerequisites

- **Python 3.12+**: Download from [python.org](https://www.python.org/downloads/) or install via Windows Package Manager:
  ```powershell
  winget install Python.Python.3.12
  ```
- **uv** (Recommended Python Package Manager): Fast, reliable dependency manager:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
  *Alternatively via winget:* `winget install astral-sh.uv`
- **Git for Windows**:
  ```powershell
  winget install Git.Git
  ```

### 2. Clone and Setup Environment

Clone the repository and sync the virtual environment using `uv`:

```powershell
git clone https://github.com/akhil/donorcast.git
cd donorcast
uv sync
```

`uv sync` will automatically create the `.venv` virtual environment and install all dependencies (pandas, lightgbm, pytorch, statsmodels, shap, streamlit, pytest, ruff, etc.).

---

## Running the Pipeline

### One-Command End-to-End Pipeline

To execute the entire automated pipeline from frozen raw data to shortfall alert generation:

```powershell
uv run donorcast all
```

This single command executes the 5 core stages in strict sequence with progress and timing logs:
1. **`[1/5] Clean`**: Cleans raw CSVs, drops unmapped facility records, validates national reconciliation checks, and writes `data/processed/long.parquet`.
2. **`[2/5] Features`**: Computes multi-horizon lag features (1, 2, 3, 7, 14, 21, 28), rolling statistics, lagged donor shares, and calendar variables into partitioned parquet files (`data/processed/features/`).
3. **`[3/5] Baselines`**: Generates forecasts for M0 (Seasonal Naive) and M0b (4-Week Weekday Moving Average) on the validation split and logs results to `reports/results_baselines.md`.
4. **`[4/5] Train`**: Trains the winning global LightGBM Tweedie model and quantile regressors (p10, p90) on the validation split, computes empirical coverage, and records metrics to `reports/results_lgbm_val.md`.
5. **`[5/5] Alerts`**: Generates multi-horizon 14-day forecasts for all 88 facility-group pairs from the latest origin date (`2026-09-22`), flags upcoming 7-day shortfall alerts, translates top 3 SHAP drivers to plain language, and writes outputs to `outputs/forecasts_2026-09-22.parquet` and `outputs/alerts_2026-09-22.parquet`.

> **Note on Test Period Policy**: `donorcast all` intentionally skips `final` (the held-out test evaluation) to uphold the `TEST_SET_TOUCHED_ONCE` invariant. Test evaluation remains a manual, guarded step.

#### Pipeline Options

```powershell
# Run pipeline with a specific forecast origin date:
uv run donorcast all --origin 2026-06-15

# Run pipeline and precompute all 8 historical replay origins:
uv run donorcast all --precompute-replay
```

---

## Individual CLI Commands

You can run individual pipeline components independently:

| Command | Description |
|---|---|
| `uv run donorcast clean` | Clean raw data, run reconciliation checks, write `data/processed/long.parquet`. |
| `uv run donorcast features` | Generate multi-horizon feature partitions in `data/processed/features/`. |
| `uv run donorcast baselines` | Evaluate M0 and M0b baseline models on the validation split. |
| `uv run donorcast train --model lgbm` | Train and evaluate global LightGBM model + p10/p90 quantiles. |
| `uv run donorcast train --model sarima` | Train and evaluate 88 per-series SARIMAX models with exogenous holidays. |
| `uv run donorcast train --model lstm` | Train and evaluate global deep learning LSTM in PyTorch. |
| `uv run donorcast explain --sample-size 1000` | Generate global SHAP beeswarm and bar summary plots in `reports/figures/`. |
| `uv run donorcast alerts --origin 2026-09-22` | Generate 14-day forecasts and 7-day shortfall alerts table for an origin. |
| `uv run donorcast final` | Execute guarded held-out test evaluation (run once; use `--force` to re-evaluate). |

---

## Running Verification & Tests

To execute the test suite (unit tests, integration tests, invariant checks, CLI tests):

```powershell
uv run pytest
```

To run code linting:

```powershell
uv run ruff check .
```

---

## Project Structure

```text
donorcast/
├── data/
│   ├── raw/                 # Frozen MoH CSVs + hashes.json (never edited)
│   ├── external/            # Holidays, school terms, state weekends, elections, MCO dates
│   └── processed/           # Processed long table (long.parquet) & yearly features
├── models/
│   ├── lgbm/                # Versioned LightGBM artifacts (v001, v003: model.txt, p10, p90)
│   └── lstm/                # PyTorch LSTM weights (v001, v002: model.pt)
├── outputs/                 # 14-day forecasts and shortfall alert parquets (per origin)
├── reports/
│   ├── figures/             # SHAP summary plots and EDA figures
│   ├── results_final.md     # Final test set evaluation report
│   ├── results_baselines.md # Baseline validation report
│   └── results_lgbm_val.md  # LightGBM validation report
├── src/donorcast/
│   ├── clean.py             # Data cleaning and integrity checks
│   ├── calendar.py          # State-level holiday and calendar features
│   ├── features.py          # Multi-horizon feature generation
│   ├── shortfall.py         # Shortfall alerting & 14-day forecasting
│   ├── explain.py           # SHAP explainability and plain-English translation
│   ├── evaluate.py          # Rolling-origin evaluation harness (LOCKED)
│   ├── final.py             # Held-out test evaluation orchestrator
│   ├── cli.py               # Unified CLI (donorcast all, clean, train, alerts, etc.)
│   └── models/              # baselines.py, lgbm.py, sarima.py, lstm.py
├── tests/                   # Pytest suite verifying invariants, models, and CLI
├── notebooks/               # 01_eda.ipynb, 02_features.ipynb, 03_model_comparison.ipynb
├── pyproject.toml           # Project dependencies and script entrypoints
├── AGENTS.md                # Hard rules and protocols for autonomous agents
└── README.md                # Project documentation and Windows guide
```
