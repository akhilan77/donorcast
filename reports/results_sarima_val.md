# SARIMAX (M1) Model Evaluation Report (VAL Split)

- **Model**: `M1_sarimax` (SARIMAX with weekly seasonality $s=7$ and 4 exogenous calendar dummies)
- **Order Specification**: `SARIMAX(1, 0, 1)x(1, 0, 1, 7)`
- **Exogenous Variables**: `is_public_holiday, is_school_holiday, is_ramadan, is_mco` (Facility's State)
- **Estimation Strategy**: Fit once on last 3 years of train (`2020-01-01` to `2022-12-31`), rolling state update via `extend(refit=False)` across all validation origins
- **Total Runtime (Fit + State Updating + 14D Forecast)**: **49.08 seconds** (0.82 minutes)
- **Evaluation Period**: `2023-01-01` to `2024-12-31` (104 weekly Monday origins)
- **Forecast Horizons**: 1 to 14 days ahead

---

## 1. Order Selection Grid Search (5 Representative Series)

Evaluated across representative facilities (Pusat Darah Negara, Melaka, Sultanah Bahiyah, Duchess of Kent, Queen Elizabeth II):

| Order | Seasonal Order | Mean AIC | Mean BIC | Avg Fit Time (s) | Converged |
|---|---|---|---|---|---|
| `(1, 0, 1)` | `(0, 1, 1, 7)` | 8471.6 | 8511.5 | 1.48s | 5/5 |
| `(1, 1, 1)` | `(0, 1, 1, 7)` | 8484.4 | 8524.2 | 1.67s | 5/5 |
| `(1, 0, 1)` | `(1, 0, 1, 7)` | 8581.4 | 8626.3 | 1.23s | 5/5 |
| `(2, 0, 1)` | `(1, 0, 1, 7)` | 8683.2 | 8733.1 | 1.49s | 5/5 |
| `(2, 0, 0)` | `(1, 0, 0, 7)` | 8847.0 | 8886.9 | 0.86s | 5/5 |
| `(1, 0, 0)` | `(1, 0, 0, 7)` | 8874.5 | 8909.5 | 0.59s | 5/5 |

---

## 2. Headline Regression Metrics Comparison

| Model | Description | WAPE_7D (Primary) | Daily WAPE | MASE | Rows Evaluated |
|---|---|---|---|---|---|
| **M0** | Seasonal Naive (last week) | 25.7% | 55.7% | 1.10 | 127,688 |
| **M0b** | Weekday Moving Avg (4-wk) | 21.5% | 45.7% | 0.90 | 127,688 |
| **M1 (SARIMAX)** | SARIMAX(1,0,1)x(1,0,1)7 + Exog | **20.3%** | **41.5%** | **0.82** | 127,688 |

> **Validation Comparison vs M0b Baseline**: SARIMAX WAPE_7D is **20.3%** vs M0b **21.5%** (beats M0b by 1.2%).

---

## 3. Shortfall Classification Metrics (7-Day Ahead)

- **Actual Shortfall Prevalence**: **21.6%** (1,979 of 9,152 windows)

| Model | Prevalence | Precision | Recall | F1 Score | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| **M0** | 21.6% | 35.9% | 37.6% | 0.37 | 745 | 1328 | 1234 | 5845 |
| **M0b** | 21.6% | 45.6% | 28.2% | 0.35 | 558 | 666 | 1421 | 6507 |
| **M1 (SARIMAX)** | 21.6% | 51.7% | 24.8% | 0.33 | 490 | 457 | 1489 | 6716 |

### Shortfall Metrics by Facility Tier

| Tier | Model | Precision | Recall | F1 Score | Total Windows |
|---|---|---|---|---|---|
| **top_5** (Top 5 High-Volume Sites) | M1 (SARIMAX) | 43.7% | 22.5% | 0.30 | 2,080 |
| **middle** (Middle 10 Sites) | M1 (SARIMAX) | 49.6% | 24.9% | 0.33 | 4,160 |
| **bottom_7** (Bottom 7 Small Sites) | M1 (SARIMAX) | 57.3% | 25.6% | 0.35 | 2,912 |

---

## 4. Breakdown by Horizon (h = 1..14)

| Horizon | SARIMAX WAPE | SARIMAX MASE |
|---|---|---|
| Day 01 | 42.3% | 0.67 |
| Day 02 | 42.6% | 0.82 |
| Day 03 | 43.3% | 0.80 |
| Day 04 | 50.2% | 0.59 |
| Day 05 | 39.7% | 1.00 |
| Day 06 | 34.1% | 1.24 |
| Day 07 | 49.4% | 0.63 |
| Day 08 | 42.3% | 0.67 |
| Day 09 | 43.1% | 0.83 |
| Day 10 | 44.3% | 0.81 |
| Day 11 | 50.6% | 0.59 |
| Day 12 | 39.7% | 1.00 |
| Day 13 | 34.2% | 1.24 |
| Day 14 | 50.0% | 0.63 |

---

## 5. Breakdown by Blood Group

| Group | SARIMAX WAPE_7D | SARIMAX Daily WAPE | SARIMAX MASE |
|---|---|---|---|
| **A** | 20.0% | 41.4% | 0.82 |
| **B** | 20.5% | 42.0% | 0.83 |
| **O** | 19.0% | 39.1% | 0.82 |
| **AB** | 29.9% | 58.3% | 0.82 |

---

## 6. Breakdown by Facility Tier

| Tier | Description | SARIMAX WAPE_7D | SARIMAX Daily WAPE | SARIMAX MASE |
|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 16.9% | 32.1% | 0.86 |
| **middle** | Middle 10 Sites | 21.5% | 46.9% | 0.79 |
| **bottom_7** | Bottom 7 Small Sites | 34.9% | 74.8% | 0.85 |

---

## 7. Holiday Window Performance

| Window | SARIMAX WAPE_7D | SARIMAX Daily WAPE | SARIMAX MASE |
|---|---|---|---|
| **Public Holiday** | 24.3% | 70.8% | 1.37 |
| **Non-Holiday** | 18.7% | 39.6% | 0.79 |
