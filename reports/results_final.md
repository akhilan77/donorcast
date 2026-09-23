# Final Evaluation Report: Test Split (2025-01-01 to 2026-09-22)

- **Evaluation Split**: `test` (Held-Out Final Evaluation)
- **Evaluated at**: `2026-09-23T11:05:33.154651+00:00`
- **Final Model Version**: `v003`
- **Total Rows Evaluated**: 109,208
- **Evaluation Frequency**: Weekly (every Monday origin, 89 origins)
- **Forecast Horizons**: 1 to 14 days ahead

---

## 1. Main Headline Metrics Comparison

| Model | Description | WAPE_7D (Primary) | Daily WAPE | MASE | Pinball (p10) | Pinball (p90) | Rows Evaluated |
|---|---|---|---|---|---|---|---|
| **M0** | Seasonal Naive (last week) | **24.4%** | 54.8% | 1.11 | N/A | N/A | 109,208 |
| **M0b** | Weekday Moving Avg (4-wk) | **21.3%** | 45.2% | 0.92 | N/A | N/A | 109,208 |
| **M2 (LightGBM)** | Global Tweedie Regressor | **16.9%** | **37.9%** | **0.79** | 1.45 | 2.00 | 109,208 |

> **Invariant Check (`BASELINE_MUST_BE_BEATEN`)**: **PASSED**.
> LightGBM achieved **16.9%** WAPE_7D vs M0b baseline **21.3%** (an improvement of **4.5 percentage points**).

---

## 2. Shortfall Classification Performance (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 7,832
- **Actual Shortfall Windows**: 1,841 (Prevalence: **23.5%**)

| Model | Prevalence | Precision | Recall | F1 Score | TP | FP | FN | TN | Total Windows |
|---|---|---|---|---|---|---|---|---|---|
| **M0** | 23.5% | 40.5% | 41.9% | 0.41 | 772 | 1136 | 1069 | 4855 | 7,832 |
| **M0b** | 23.5% | 47.5% | 28.7% | 0.36 | 529 | 585 | 1312 | 5406 | 7,832 |
| **M2 (LightGBM)** | **23.5%** | **62.0%** | **48.2%** | **0.54** | 887 | 543 | 954 | 5448 | 7,832 |

### Shortfall Metrics by Facility Tier

| Tier | Model | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** (Top 5 High-Volume Sites) | M0 | 19.1% | 37.0% | 35.9% | 0.36 | 1,780 |
| | M0b | 19.1% | 44.6% | 25.6% | 0.33 | 1,780 |
| | **LightGBM** | **19.1%** | **69.9%** | **54.7%** | **0.61** | 1,780 |
| **middle** (Middle 10 Sites) | M0 | 19.9% | 37.6% | 38.1% | 0.38 | 3,560 |
| | M0b | 19.9% | 42.9% | 21.7% | 0.29 | 3,560 |
| | **LightGBM** | **19.9%** | **58.7%** | **49.1%** | **0.53** | 3,560 |
| **bottom_7** (Bottom 7 Small Sites) | M0 | 31.8% | 44.2% | 48.0% | 0.46 | 2,492 |
| | M0b | 31.8% | 51.4% | 36.4% | 0.43 | 2,492 |
| | **LightGBM** | **31.8%** | **61.8%** | **44.6%** | **0.52** | 2,492 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | M0 WAPE | M0 MASE | M0b WAPE | M0b MASE | LightGBM WAPE | LightGBM MASE | Best Model |
|---|---|---|---|---|---|---|---|
| Day 01 | 57.2% | 1.00 | 47.6% | 0.83 | **40.1%** | **0.74** | **LightGBM** |
| Day 02 | 52.2% | 1.06 | 44.3% | 0.90 | **37.6%** | **0.77** | **LightGBM** |
| Day 03 | 53.7% | 1.05 | 45.2% | 0.87 | **38.1%** | **0.76** | **LightGBM** |
| Day 04 | 60.5% | 0.76 | 52.7% | 0.66 | **44.5%** | **0.57** | **LightGBM** |
| Day 05 | 53.1% | 1.37 | 44.2% | 1.14 | **37.3%** | **0.98** | **LightGBM** |
| Day 06 | 46.8% | 1.66 | 37.8% | 1.35 | **31.8%** | **1.16** | **LightGBM** |
| Day 07 | 65.1% | 0.78 | 53.6% | 0.64 | **43.8%** | **0.55** | **LightGBM** |
| Day 08 | 58.6% | 1.02 | 48.5% | 0.85 | **40.5%** | **0.75** | **LightGBM** |
| Day 09 | 55.4% | 1.10 | 45.3% | 0.91 | **37.8%** | **0.77** | **LightGBM** |
| Day 10 | 56.2% | 1.07 | 46.3% | 0.88 | **38.4%** | **0.77** | **LightGBM** |
| Day 11 | 63.5% | 0.78 | 53.4% | 0.66 | **44.4%** | **0.58** | **LightGBM** |
| Day 12 | 54.2% | 1.39 | 44.7% | 1.15 | **37.5%** | **0.98** | **LightGBM** |
| Day 13 | 48.7% | 1.70 | 38.2% | 1.36 | **31.9%** | **1.16** | **LightGBM** |
| Day 14 | 65.4% | 0.78 | 53.8% | 0.64 | **43.8%** | **0.55** | **LightGBM** |

---

## 4. Breakdown by Blood Group

| Group | M0 WAPE_7D | M0b WAPE_7D | LightGBM WAPE_7D | LightGBM Daily WAPE | LightGBM MASE | Best (7D) |
|---|---|---|---|---|---|---|
| **A** | 23.9% | 21.0% | **16.6%** | 37.7% | 0.79 | **LightGBM** |
| **B** | 24.4% | 21.3% | **16.5%** | 37.7% | 0.78 | **LightGBM** |
| **O** | 23.0% | 20.0% | **16.1%** | 35.8% | 0.79 | **LightGBM** |
| **AB** | 36.7% | 32.7% | **25.6%** | 55.1% | 0.81 | **LightGBM** |

---

## 5. Breakdown by Facility Tier

| Tier | Description | M0 WAPE_7D | M0b WAPE_7D | LightGBM WAPE_7D | LightGBM Daily WAPE | LightGBM MASE | Best (7D) |
|---|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 19.0% | 17.3% | **12.6%** | 28.0% | 0.80 | **LightGBM** |
| **middle** | Middle 10 Sites | 27.2% | 23.3% | **19.6%** | 44.2% | 0.79 | **LightGBM** |
| **bottom_7** | Bottom 7 Small Sites | 43.9% | 36.3% | **30.4%** | 69.6% | 0.80 | **LightGBM** |

---

## 6. Holiday Window Performance

| Window | M0 WAPE_7D | M0b WAPE_7D | LightGBM WAPE_7D | LightGBM Daily WAPE | LightGBM MASE | Best (7D) |
|---|---|---|---|---|---|---|
| **Public Holiday** | 28.3% | 25.4% | **17.1%** | 48.1% | 0.94 | **LightGBM** |
| **Non-Holiday** | 22.8% | 19.7% | **16.8%** | 37.1% | 0.78 | **LightGBM** |

---

## 7. Prediction Interval Coverage (p10–p90)

- **Target Coverage**: **80.0%** (p10 to p90 interval)
- **Empirical Coverage (Test Split)**: **81.2%**
- **Pinball Loss (p10)**: **1.45**
- **Pinball Loss (p90)**: **2.00**
- **Calibration Assessment**: The empirical coverage on held-out test data closely confirms that the p10–p90 prediction intervals remain well-calibrated (achieving 81.2% vs 80.0% nominal coverage).
