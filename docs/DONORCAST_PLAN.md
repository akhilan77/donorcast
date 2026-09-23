# DonorCast — Forecasting Blood Donation Shortfalls (v3, working name)

**Problem.** Blood centres need a steady daily supply of donations, and supply dips around holidays, festivals, and school breaks. Platelets last only 5 days and red cells 42, so a dip of even a week hurts patients. Planners usually react after stock has already fallen.

**Goal.** Forecast daily blood donations for each collection facility and each blood group 1–14 days ahead. Flag upcoming **shortfall weeks**: weeks when donations of a group are predicted to fall well below normal. The flag gives planners time to schedule mobile drives and recall donors *before* stock runs low.

**Why this counts as a real problem.** Malaysian officials have said demand often outstrips supply around festive seasons and holidays. The health ministry has asked campaign organisers to keep running drives during Ramadan because stock is expected to drop. A 14-day early warning turns that general knowledge into a specific, per-facility, per-group signal.

---

## 1. Dataset

**Source:** Ministry of Health Malaysia, National Blood Centre (Pusat Darah Negara), open data. The data comes from BBISv2, the blood bank information system used by the 22 main collection sites under MoH, which together cover the majority of donations in Malaysia.
**Where:** GitHub `MoH-Malaysia/data-darah-public` (also on data.moh.gov.my). Free, no request needed.
**Licence:** check the data catalogue page and cite the source as "Ministry of Health Malaysia / National Blood Centre" in the report.

### Files used (checked on 23 Sep 2026)

| File | Raw rows | Size | Span | Role |
|---|---|---|---|---|
| `donations_facility.csv` | 219,530 | 15 MB | 2006-01-01 → 2026-09-22 | **Main modelling data** |
| `donations_state.csv` | 105,980 | 6.6 MB | 2006-01-01 → 2026-09-22 | Validation of totals, national-level EDA |
| `newdonors_facility.csv` | 211,960 | 11 MB | 2006-01-01 → 2026-09-22 | Feature: new-donor age mix |

**How much data after cleaning:**

- 22 facilities × 7,570 days = **166,540 facility-days** (every facility covers every day of the 20.7 years)
- × 4 blood groups = **666,160 facility-group-days**. This is the modelling table: 88 time series, each 7,570 days long.
- National daily donations: mean 1,304, median 1,195, max 5,543.

**Freeze the data.** The files update every day. Download once, save a copy in `data/raw/` with its SHA-256 hash, and fix the cutoff at **2026-09-22**. Every result in the report comes from that frozen copy (dataset version `v2026-09-22`).

### Data-quality findings (from profiling; each goes in the report)

| Finding | Detail | Action |
|---|---|---|
| Blank facility names | 52,990 rows (24% of the file) have no `hospital`. They average 0.5 donations and total ~0.2% of national volume. | Drop them and document it. |
| Totals reconcile | The sum of the 22 facilities equals the national total exactly on every day. | Record as a validation check. |
| Blood groups slightly short | In 283 rows (0.13%), A+B+O+AB is 1–3 lower than `daily` (likely unknown group). | Keep; targets use the group columns. |
| Other breakdowns reconcile | Location, donation type, donor type, and social group all sum exactly to `daily`. | Record as a validation check. |
| Intermittent small sites | Share of zero-donation days: 1% at Pusat Darah Negara vs 67% at Hospital Duchess of Kent. | Use a count-aware loss (Tweedie) and report small sites separately. |
| Strong weekly cycle | National mean: Sunday 2,099, Monday 867, Friday 895. | Day-of-week is a core feature. The Friday dip likely reflects states with Friday weekends; test it. |
| Group shares | O is 41.8% of donations, AB 6.1%. | AB series are small and noisy; evaluate per group. |
| Apheresis is rare | Apheresis platelets are ~1% of donations. | Not a forecasting target; used only as a lagged share feature. |
| National zero days | A handful of days show 0 donations nationally. | Inspect: closure or data gap; mark as missing, not zero, if it's a gap. |
| 2020 not lower than 2019 | 2020 totals 560,777 vs 497,590 in 2019 in this data. | Don't assume a COVID dip; let the data show the effect of the lockdown dates. |

---

## 2. Every column and how it is used

**The key leakage rule.** Every breakdown column is a slice of the same day's total. Knowing today's mobile or student count means already knowing today's donations. These columns may **only** be used as *lagged* features (values from before the forecast date), never same-day.

### `donations_facility.csv`

| Column | Meaning | Used as | How |
|---|---|---|---|
| `date` | Day of record | Index | Builds all calendar features; defines time-ordered splits |
| `hospital` | Collection facility (22) | Series ID + categorical feature | Lets one global model learn facility-specific levels |
| `daily` | Total donations that day | Consistency check; secondary target | Validates the group columns; total-donation model for comparison |
| `blood_a`, `blood_b`, `blood_o`, `blood_ab` | Donations by ABO group | **Primary targets** (reshaped to long: `group`, `donations`) | Forecast per facility × group; lags 1–28 days and rolling means become features |
| `location_centre`, `location_mobile` | Donated at the centre vs at a mobile drive | Lagged feature | Mobile share over the last 7/28 days: captures drive activity |
| `type_wholeblood`, `type_apheresis_platelet`, `type_apheresis_plasma`, `type_other` | Donation type | Lagged feature (share) / EDA | Apheresis share over 28 days; mainly descriptive |
| `social_civilian`, `social_student`, `social_policearmy` | Donor group | Lagged feature | Student share over 28 days: captures school and university terms |
| `donations_new`, `donations_regular`, `donations_irregular` | New, regular, lapsed donors | Lagged feature | Regular-donor share and new-donor trend: donor-base health |

### `newdonors_facility.csv`

| Column | Used as | How |
|---|---|---|
| `17-24` … `60-64`, `other`, `total` | Lagged feature | Share of new donors aged 17–24 over the last 28 days (young donors come in via campus drives) |

### `donations_state.csv`

Not used for model training. Used for: the national-total reconciliation check, national EDA charts, and a state-level map in the dashboard.

### External calendar data (built by us, documented in the report)

| Feature | Source | Why |
|---|---|---|
| Public holidays (national + state) | Python `holidays` package (Malaysia, with state subdivisions) | Holiday dips |
| Ramadan, Hari Raya Aidilfitri, Aidiladha | Hijri calendar conversion (`hijri-converter`) | Moves ~11 days earlier each year, so a naive "same week last year" model gets it wrong |
| Chinese New Year, Deepavali | `holidays` package | Festival dips |
| School holiday periods | Ministry of Education calendars, entered by hand as a CSV | Student donors disappear |
| State weekend (Fri–Sat vs Sat–Sun) | Hand-entered, with change dates (verify each state's history) | Explains the Friday pattern |
| COVID movement-control periods | Hand-entered start/end dates | Lets the model separate the lockdown shock from normal patterns |
| Days to / days since each holiday | Derived | Donations often dip before a holiday and recover after |

---

## 3. Targets

1. **Regression (primary):** `donations[facility, group, day]` for horizons h = 1…14 days, forecast from an origin day t using only data up to t.
2. **Shortfall flag (derived, for users):** for the next 7 days, `sum(forecast) < SHORTFALL_RATIO × typical`. Here `typical` is the median 7-day total for the same facility, group, and week-of-year over the previous 3 years, computed only from data before t. `SHORTFALL_RATIO` defaults to 0.8 and is configurable. The flag is evaluated against what actually happened, using the same rule.

---

## 4. Models and how the best one is selected

| # | Model | Why it's included | Notes |
|---|---|---|---|
| M0 | **Seasonal naive**: same weekday last week | The bar every model must clear | |
| M0b | **Weekday moving average**: mean of the same weekday over the last 4 weeks | Stronger simple baseline | |
| M1 | **SARIMA** (per series, weekly seasonality, holiday dummies as exogenous inputs) | The classical statistics answer | 88 separate models; captures each series' own cycle |
| M2 | **LightGBM**, one global model with a Tweedie objective | Expected best: handles many series, calendar effects, zeros, and interactions | One model across all facilities and groups; direct multi-horizon |
| M3 | **LSTM** (PyTorch), global, same features | The deep learning comparison | Expected to lose on tabular data of this size; explaining *why* is part of the analysis |

**Selection rule.** The winner is the model with the lowest WAPE on the validation period, across horizons 1–14, that also beats M0b. Ties are broken by shortfall-flag recall and then by simplicity. The test period is used **once**, at the end.

**Metrics:**
- **WAPE** (weighted absolute percentage error) as the headline metric.
- **MASE** (error relative to the seasonal-naive baseline). It's scale-free and works with zero days, unlike MAPE, which breaks on zeros. Explain this choice in the report.
- **Pinball loss** for p10/p90 prediction intervals (LightGBM quantile models).
- **Shortfall flag:** precision, recall, and lead time.
- Everything is also broken down by blood group, by facility size, and around holidays.

**Splits (time-ordered, never random):**

| Period | Dates | Use |
|---|---|---|
| Train | 2006-01-01 → 2022-12-31 | Fitting (early years may be dropped if EDA shows a regime change) |
| Validation | 2023-01-01 → 2024-12-31 | Rolling-origin evaluation with weekly origins; tuning and model selection |
| Test | 2025-01-01 → 2026-09-22 | Final held-out evaluation, run once |

**Explainability:** SHAP on LightGBM, both global (which features matter overall) and per alert ("Ramadan starts in 3 days; student share down").

---

## 5. How a user uses it

The dashboard (Streamlit) has four pages:

1. **Forecast:** pick a facility and blood group, then see the next 14 days with a p10–p90 band against the last 8 weeks of actual donations.
2. **Shortfall alerts:** every facility × group flagged for the coming week, sorted by severity, with the top 3 reasons in plain words.
3. **Model comparison:** baseline vs SARIMA vs LightGBM vs LSTM on the test period, by horizon and by group.
4. **Data explorer:** national trends, weekly cycle, holiday effects, and a state map.

Example alert:
> ⚠ **Hospital Melaka, group O**: next 7 days forecast 310 donations vs typical 420 (−26%)
> Why: Hari Raya in 4 days · school holidays · mobile-drive share down over the last 4 weeks
> Suggested action: schedule an extra mobile drive and message regular O donors this week.

---

## 6. Project structure and stack

```
data/raw/            frozen CSVs + SHA-256 (never edited)
data/external/       holidays, school terms, weekends, MCO dates (hand-built CSVs, with sources)
data/processed/      cleaned long table (parquet)
notebooks/           01_eda.ipynb, 02_features.ipynb, 03_results.ipynb
src/donorcast/       clean.py, calendar.py, features.py, models/, evaluate.py (LOCKED), shortfall.py
app/                 streamlit_app.py
reports/             figures, results tables, final report
tests/
```

Stack: Python 3.12, uv, pandas, statsmodels (SARIMA), lightgbm, pytorch (LSTM), shap, holidays, hijri-converter, streamlit, pytest.

### Invariants (each has a test)

| Invariant | Meaning |
|---|---|
| `RAW_DATA_FROZEN` | Files in `data/raw/` match their recorded hashes. |
| `NO_SAME_DAY_BREAKDOWNS` | No feature uses a same-day breakdown column. |
| `FEATURES_AS_OF_ORIGIN` | Features for origin t use only rows dated ≤ t. The test changes all rows after t and asserts the features are unchanged. |
| `TIME_ORDERED_SPLITS` | No random splits anywhere. |
| `TEST_SET_TOUCHED_ONCE` | The test-period evaluation runs only via `make final`, which records a timestamp. |
| `BASELINE_MUST_BE_BEATEN` | The selected model beats M0b on validation WAPE. |

---

## 7. `AGENTS.md` (commit on day 1)

```md
# AGENTS.md
Project: forecast daily blood donations per facility and blood group (Malaysia MoH open data).
Read docs/DONORCAST_PLAN.md before every task.

## Hard rules
- Never edit data/raw/ or src/donorcast/evaluate.py unless the task says so.
- Never use a same-day breakdown column as a feature; lagged only.
- Features are computed as of the forecast origin. No random splits.
- Never run the test-period evaluation unless the task says "final".
- All thresholds and dates live in src/donorcast/config.py.
- Never weaken or delete a test for an invariant in plan section 6.

## Protocol
1. Plan first: list files and approach. Wait for approval.
2. Change only what the task asks. Put other issues under "Follow-ups".
3. Before finishing: ruff, pytest. Notebooks must run top to bottom.
4. Final message: files changed, how to verify, follow-ups.
```

---

## 8. Steps (4 weeks)

★ = checkpoint: bring the output back to review before continuing.

### Week 1 — Data
**Task 1.1 — Setup.** *"Create the project structure in plan section 6 with uv, ruff, pytest, and AGENTS.md. Add a download script that fetches the three CSVs, saves them to data/raw/ with SHA-256 hashes, and a test for RAW_DATA_FROZEN. Plan first."*

**Task 1.2 — Cleaning.** *"Implement src/donorcast/clean.py: drop blank-hospital rows, reshape blood groups to long format (facility, date, group, donations), join lagged-ready breakdown columns, and run the reconciliation checks from plan section 1 as tests. Output data/processed/long.parquet. Plan first."*

**Task 1.3 — EDA notebook.** *"notebooks/01_eda.ipynb: national trend, weekly cycle, yearly seasonality, holiday windows, per-facility zero shares, group shares, the national zero days, and 2019 vs 2020. Save figures to reports/figures/."*

★ **Checkpoint A:** bring the EDA figures. We decide whether to drop early years and how to treat the national zero days.

### Week 2 — Calendar, features, evaluation
**Task 2.1 — Calendar.** *"Build src/donorcast/calendar.py from data/external/ CSVs plus the holidays and hijri-converter packages: all features in the plan section 2 calendar table, per state."*

**Task 2.2 — Features.** *"src/donorcast/features.py: lags 1, 2, 3, 7, 14, 21, 28; rolling mean and std (7, 28); lagged breakdown shares (mobile, student, regular, apheresis, new donors aged 17–24); calendar joins; facility and group as categoricals. Enforce NO_SAME_DAY_BREAKDOWNS and FEATURES_AS_OF_ORIGIN with tests. Plan first."*

**Task 2.3 — Evaluation harness (locked afterwards).** *"src/donorcast/evaluate.py: rolling-origin backtest with weekly origins, horizons 1–14; WAPE, MASE, pinball loss; shortfall flag and its precision, recall, and lead time; breakdowns by group, facility size, and holiday windows. Tests for TIME_ORDERED_SPLITS and TEST_SET_TOUCHED_ONCE. Plan first."*

**Task 2.4 — Baselines.** *"Implement M0 and M0b and run them on validation. Write reports/results_baselines.md."*

★ **Checkpoint B:** bring the baseline results. This sets the bar.

### Week 3 — Models
**Task 3.1 — SARIMA.** *"M1: SARIMA per series with weekly seasonality and holiday exogenous variables; parallelise across series; evaluate on validation via evaluate.py."*

**Task 3.2 — LightGBM.** *"M2: global LightGBM, Tweedie objective, direct multi-horizon; quantile models for p10/p90; tune on validation only (time-ordered). Save models with dataset and feature versions."*

**Task 3.3 — LSTM.** *"M3: global LSTM in PyTorch with the same features, sequence length 56 days; early stopping on validation. Keep it simple and reproducible (fixed seeds)."*

★ **Checkpoint C:** bring the validation comparison table. We pick the winner by the section 4 rule, and check for leakage if anything looks too good.

### Week 4 — Final, explain, ship
**Task 4.1 — Final evaluation.** *"final: retrain the selected model and the baselines on train+validation, evaluate once on the test period, write reports/results_final.md."*

**Task 4.2 — SHAP and alerts.** *"Global SHAP summary; per-alert top-3 reasons translated to plain language; src/donorcast/shortfall.py produces the weekly alert table."*

**Task 4.3 — Dashboard.** *"Streamlit app with the four pages in plan section 5. Read only from saved outputs; no training inside the app."*

**Task 4.4 — Report.** Write it yourself, using the agent for figures and tables only. Sections: problem, dataset (section 1 numbers), column usage (section 2), cleaning, EDA, features, models, evaluation design, results, limitations, future work.

★ **Checkpoint D:** bring the final results and the report draft for review.

---

## 9. Limitations to state honestly

- Donations are **supply**, not stock or demand. The system predicts when supply dips, not when patients go without blood.
- The data is from Malaysia. The method transfers to India, but there is no public historical blood-supply data for India to prove it. That gap is worth stating as a finding.
- Small facilities are intermittent, so their forecasts are less reliable.
- Hand-built calendars (school terms, state weekends, lockdown dates) can contain errors; their sources are listed.

**Future work:** apply the same pipeline to India by collecting daily e-RaktKosh stock snapshots going forward (the earlier RaktWatch plan).

---

## 10. Questions to be ready for in the viva

- Why these metrics, and why not MAPE? (Zero days.)
- Why time-ordered splits and not random ones? (Leakage from the future.)
- Why can't same-day mobile or student counts be used? (They're slices of the target.)
- Why did LightGBM beat the LSTM (or not)? (Data size, tabular features, the handling of calendar effects.)
- How much better than the baseline is the chosen model, and does that matter in practice? (Lead time and shortfall recall, not just WAPE.)
- What does the model get wrong? (Show the worst weeks and explain them.)
