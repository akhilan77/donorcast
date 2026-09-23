# DonorCast Task Tracking

Tracking progress across project roadmap defined in `docs/DONORCAST_PLAN.md`.

| Task | Description | Status | Deliverables & Artifacts |
|---|---|---|---|
| **Task 1.1** | Project setup & frozen raw data verification | **COMPLETED** | `pyproject.toml`, `data/raw/` CSVs + SHA-256 `hashes.json`, `test_raw_data_frozen` |
| **Task 1.2** | Data cleaning & reconciliation | **COMPLETED** | `src/donorcast/clean.py`, `data/processed/long.parquet`, `reports/cleaning_log.md` |
| **Task 1.3** | Exploratory Data Analysis (EDA) | **COMPLETED** | `notebooks/01_eda.ipynb`, `reports/figures/` (EDA figures) |
| **Task 2.1** | External calendar generation | **COMPLETED** | `src/donorcast/calendar.py`, `data/external/` datasets (elections, holidays, school terms) |
| **Task 2.2** | Multi-horizon feature engineering | **COMPLETED** | `src/donorcast/features.py`, `data/processed/features/features_*.parquet` |
| **Task 2.3** | Evaluation harness & invariant enforcement | **COMPLETED** | `src/donorcast/evaluate.py`, `tests/test_evaluate.py`, `tests/test_invariants.py` |
| **Task 2.4** | Baseline forecasting models (M0, M0b) | **COMPLETED** | `src/donorcast/models/baselines.py`, `reports/results_baselines.md` |
| **Task 3.1** | Classical SARIMAX forecasting | **COMPLETED** | `src/donorcast/models/sarima.py`, `reports/results_sarima_val.md` |
| **Task 3.2** | Global LightGBM Tweedie regression | **COMPLETED** | `src/donorcast/models/lgbm.py`, `models/lgbm/v001`, `reports/results_lgbm_val.md` |
| **Task 3.3** | LightGBM quantile regression (p10/p90) | **COMPLETED** | `models/lgbm/v001/model_p10.txt`, `model_p90.txt`, coverage validation |
| **Task 3.4** | Deep learning LSTM forecasting | **COMPLETED** | `src/donorcast/models/lstm.py`, `models/lstm/v001`, `reports/results_lstm_val.md` |
| **Task 3.5** | Model comparison & selection | **COMPLETED** | `notebooks/03_model_comparison.ipynb`, LightGBM selected as winning model |
| **Task 4.1** | Final evaluation on held-out test split | **COMPLETED** | `src/donorcast/final.py`, `reports/final_run.json`, `reports/results_final.md` |
| **Task 4.2** | Explainability & SHAP reason translation | **COMPLETED** | `src/donorcast/explain.py`, `feature_labels.yaml`, SHAP figures in `reports/figures/` |
| **Task 4.3** | 14-day forecasts & shortfall alert system | **COMPLETED** | `src/donorcast/shortfall.py`, `outputs/alerts_*.parquet`, `outputs/forecasts_*.parquet` |
| **Task 4.4** | End-to-end pipeline CLI & Windows guide | **COMPLETED** | `donorcast all`, progress logging & timings, `README.md`, fresh clone validation |
| **Task 4.5** | Interactive Streamlit dashboard | **UPCOMING** | Next milestone (Streamlit app with 4 pages) |
