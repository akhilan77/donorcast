# Evaluation Report: M0_seasonal_naive (VAL Split)

- **Model**: `M0_seasonal_naive`
- **Split**: `val`
- **Evaluated at**: `2026-09-23T08:25:14.088807+00:00`
- **Total Rows Evaluated**: 127,688

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **25.7%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **55.7%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **1.10** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | N/A | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | N/A | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 9,152
- **Actual Shortfall Windows**: 1,979 (Prevalence: **21.6%**)
- **Predicted Shortfall Windows**: 2,073

| Metric | Value |
|---|---|
| **Prevalence** | **21.6%** |
| **Precision** | **35.9%** |
| **Recall** | **37.6%** |
| **F1 Score** | **0.37** |
| True Positives (TP) | 745 |
| False Positives (FP) | 1328 |
| False Negatives (FN) | 1234 |
| True Negatives (TN) | 5845 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 17.1% | 25.4% | 28.7% | 0.27 | 2,080 |
| **middle** | Middle 10 Sites | 17.2% | 33.0% | 34.9% | 0.34 | 4,160 |
| **bottom_7** | Bottom 7 Small Sites | 31.2% | 43.0% | 43.3% | 0.43 | 2,912 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 53.8% | 0.85 | 9,240 |
| Day 02 | 57.8% | 1.09 | 9,152 |
| Day 03 | 58.8% | 1.06 | 9,152 |
| Day 04 | 61.1% | 0.71 | 9,152 |
| Day 05 | 54.1% | 1.34 | 9,152 |
| Day 06 | 46.5% | 1.72 | 9,152 |
| Day 07 | 64.3% | 0.82 | 9,152 |
| Day 08 | 55.1% | 0.85 | 9,152 |
| Day 09 | 59.4% | 1.12 | 9,064 |
| Day 10 | 60.1% | 1.08 | 9,064 |
| Day 11 | 63.6% | 0.73 | 9,064 |
| Day 12 | 56.1% | 1.39 | 9,064 |
| Day 13 | 47.4% | 1.73 | 9,064 |
| Day 14 | 66.9% | 0.84 | 9,064 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 25.1% | 55.7% | 1.10 | 31,922 |
| **B** | 25.5% | 56.0% | 1.10 | 31,922 |
| **O** | 24.3% | 52.6% | 1.09 | 31,922 |
| **AB** | 39.0% | 78.8% | 1.09 | 31,922 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 20.4% | 43.1% | 1.16 | 29,020 |
| **middle** | Mid-size hospitals | 28.4% | 64.1% | 1.07 | 58,040 |
| **bottom_7** | Small / intermittent hospitals | 45.2% | 96.3% | 1.08 | 40,628 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 28.8% | 81.8% | 1.60 | 7,136 |
| **Non-Holiday** | 24.5% | 54.0% | 1.07 | 120,552 |
