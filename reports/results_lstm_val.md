# Evaluation Report: lstm (VAL Split)

- **Model**: `lstm`
- **Split**: `val`
- **Evaluated at**: `2026-09-23T09:42:01.995305+00:00`
- **Total Rows Evaluated**: 127,688

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **19.4%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **43.9%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **0.87** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | N/A | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | N/A | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 9,152
- **Actual Shortfall Windows**: 1,983 (Prevalence: **21.7%**)
- **Predicted Shortfall Windows**: 1,308

| Metric | Value |
|---|---|
| **Prevalence** | **21.7%** |
| **Precision** | **54.9%** |
| **Recall** | **36.2%** |
| **F1 Score** | **0.44** |
| True Positives (TP) | 718 |
| False Positives (FP) | 590 |
| False Negatives (FN) | 1265 |
| True Negatives (TN) | 6579 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 17.1% | 47.1% | 24.8% | 0.32 | 2,080 |
| **middle** | Middle 10 Sites | 17.2% | 51.5% | 38.4% | 0.44 | 4,160 |
| **bottom_7** | Bottom 7 Small Sites | 31.3% | 60.5% | 38.9% | 0.47 | 2,912 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 46.6% | 0.78 | 9,240 |
| Day 02 | 42.3% | 0.83 | 9,152 |
| Day 03 | 44.7% | 0.84 | 9,152 |
| Day 04 | 54.3% | 0.64 | 9,152 |
| Day 05 | 42.5% | 1.03 | 9,152 |
| Day 06 | 34.3% | 1.27 | 9,152 |
| Day 07 | 55.6% | 0.72 | 9,152 |
| Day 08 | 45.0% | 0.76 | 9,152 |
| Day 09 | 44.4% | 0.87 | 9,064 |
| Day 10 | 45.4% | 0.85 | 9,064 |
| Day 11 | 56.6% | 0.66 | 9,064 |
| Day 12 | 44.4% | 1.04 | 9,064 |
| Day 13 | 34.6% | 1.27 | 9,064 |
| Day 14 | 54.9% | 0.70 | 9,064 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 19.1% | 43.6% | 0.87 | 31,922 |
| **B** | 19.1% | 44.1% | 0.88 | 31,922 |
| **O** | 18.3% | 41.7% | 0.88 | 31,922 |
| **AB** | 30.1% | 60.8% | 0.86 | 31,922 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 16.3% | 33.7% | 0.92 | 29,020 |
| **middle** | Mid-size hospitals | 20.6% | 50.4% | 0.85 | 58,040 |
| **bottom_7** | Small / intermittent hospitals | 32.3% | 77.7% | 0.88 | 40,628 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 20.9% | 60.2% | 1.14 | 7,136 |
| **Non-Holiday** | 18.8% | 42.8% | 0.86 | 120,552 |
