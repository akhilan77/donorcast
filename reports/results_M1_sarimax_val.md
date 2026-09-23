# Evaluation Report: M1_sarimax (VAL Split)

- **Model**: `M1_sarimax`
- **Split**: `val`
- **Evaluated at**: `2026-09-23T08:24:21.962340+00:00`
- **Total Rows Evaluated**: 127,688

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **20.3%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **41.5%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **0.82** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | N/A | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | N/A | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 9,152
- **Actual Shortfall Windows**: 1,979 (Prevalence: **21.6%**)
- **Predicted Shortfall Windows**: 947

| Metric | Value |
|---|---|
| **Prevalence** | **21.6%** |
| **Precision** | **51.7%** |
| **Recall** | **24.8%** |
| **F1 Score** | **0.33** |
| True Positives (TP) | 490 |
| False Positives (FP) | 457 |
| False Negatives (FN) | 1489 |
| True Negatives (TN) | 6716 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 17.1% | 43.7% | 22.5% | 0.30 | 2,080 |
| **middle** | Middle 10 Sites | 17.2% | 49.6% | 24.9% | 0.33 | 4,160 |
| **bottom_7** | Bottom 7 Small Sites | 31.2% | 57.3% | 25.6% | 0.35 | 2,912 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 42.3% | 0.67 | 9,240 |
| Day 02 | 42.6% | 0.82 | 9,152 |
| Day 03 | 43.3% | 0.80 | 9,152 |
| Day 04 | 50.2% | 0.59 | 9,152 |
| Day 05 | 39.7% | 1.00 | 9,152 |
| Day 06 | 34.1% | 1.24 | 9,152 |
| Day 07 | 49.4% | 0.63 | 9,152 |
| Day 08 | 42.3% | 0.67 | 9,152 |
| Day 09 | 43.1% | 0.83 | 9,064 |
| Day 10 | 44.3% | 0.81 | 9,064 |
| Day 11 | 50.6% | 0.59 | 9,064 |
| Day 12 | 39.7% | 1.00 | 9,064 |
| Day 13 | 34.2% | 1.24 | 9,064 |
| Day 14 | 50.0% | 0.63 | 9,064 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 20.0% | 41.4% | 0.82 | 31,922 |
| **B** | 20.5% | 42.0% | 0.83 | 31,922 |
| **O** | 19.0% | 39.1% | 0.82 | 31,922 |
| **AB** | 29.9% | 58.3% | 0.82 | 31,922 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 16.9% | 32.1% | 0.86 | 29,020 |
| **middle** | Mid-size hospitals | 21.5% | 46.9% | 0.79 | 58,040 |
| **bottom_7** | Small / intermittent hospitals | 34.9% | 74.8% | 0.85 | 40,628 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 24.3% | 70.8% | 1.37 | 7,136 |
| **Non-Holiday** | 18.7% | 39.6% | 0.79 | 120,552 |
