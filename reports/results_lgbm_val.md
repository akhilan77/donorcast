# Evaluation Report: lgbm (VAL Split)

- **Model**: `lgbm`
- **Split**: `val`
- **Evaluated at**: `2026-09-23T09:10:18.742180+00:00`
- **Total Rows Evaluated**: 127,688

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **18.9%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **39.6%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **0.79** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | 1.41 | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | 2.08 | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 9,152
- **Actual Shortfall Windows**: 1,979 (Prevalence: **21.6%**)
- **Predicted Shortfall Windows**: 1,509

| Metric | Value |
|---|---|
| **Prevalence** | **21.6%** |
| **Precision** | **55.1%** |
| **Recall** | **42.0%** |
| **F1 Score** | **0.48** |
| True Positives (TP) | 832 |
| False Positives (FP) | 677 |
| False Negatives (FN) | 1147 |
| True Negatives (TN) | 6496 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 17.1% | 53.0% | 32.7% | 0.40 | 2,080 |
| **middle** | Middle 10 Sites | 17.2% | 50.8% | 45.7% | 0.48 | 4,160 |
| **bottom_7** | Bottom 7 Small Sites | 31.2% | 60.2% | 42.8% | 0.50 | 2,912 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 40.8% | 0.66 | 9,240 |
| Day 02 | 40.8% | 0.79 | 9,152 |
| Day 03 | 41.5% | 0.76 | 9,152 |
| Day 04 | 48.1% | 0.55 | 9,152 |
| Day 05 | 38.2% | 0.97 | 9,152 |
| Day 06 | 32.7% | 1.21 | 9,152 |
| Day 07 | 45.8% | 0.58 | 9,152 |
| Day 08 | 40.7% | 0.66 | 9,152 |
| Day 09 | 40.9% | 0.79 | 9,064 |
| Day 10 | 41.8% | 0.77 | 9,064 |
| Day 11 | 48.2% | 0.55 | 9,064 |
| Day 12 | 38.6% | 0.97 | 9,064 |
| Day 13 | 32.6% | 1.21 | 9,064 |
| Day 14 | 46.3% | 0.58 | 9,064 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 18.6% | 39.4% | 0.78 | 31,922 |
| **B** | 18.6% | 39.7% | 0.78 | 31,922 |
| **O** | 17.9% | 37.4% | 0.78 | 31,922 |
| **AB** | 28.8% | 57.3% | 0.81 | 31,922 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 15.4% | 30.5% | 0.82 | 29,020 |
| **middle** | Mid-size hospitals | 21.0% | 45.5% | 0.77 | 58,040 |
| **bottom_7** | Small / intermittent hospitals | 30.6% | 69.7% | 0.80 | 40,628 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 19.3% | 51.5% | 1.01 | 7,136 |
| **Non-Holiday** | 18.7% | 38.9% | 0.78 | 120,552 |

---

## 7. Quantile Prediction Band & Empirical Coverage (p10–p90)

- **Target Coverage**: **80.0%** (p10 to p90 interval)
- **Empirical Coverage (Validation)**: **81.2%**
- **Pinball Loss (p10)**: **1.41**
- **Pinball Loss (p90)**: **2.08**
- **Interpretation**: The LightGBM quantile models (p10 and p90) achieve an empirical coverage of 81.2% on the validation split, closely matching the target 80.0% prediction interval.
