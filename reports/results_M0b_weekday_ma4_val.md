# Evaluation Report: M0b_weekday_ma4 (VAL Split)

- **Model**: `M0b_weekday_ma4`
- **Split**: `val`
- **Evaluated at**: `2026-09-23T08:26:41.940047+00:00`
- **Total Rows Evaluated**: 127,688

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **21.5%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **45.7%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **0.90** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | N/A | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | N/A | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 9,152
- **Actual Shortfall Windows**: 1,979 (Prevalence: **21.6%**)
- **Predicted Shortfall Windows**: 1,224

| Metric | Value |
|---|---|
| **Prevalence** | **21.6%** |
| **Precision** | **45.6%** |
| **Recall** | **28.2%** |
| **F1 Score** | **0.35** |
| True Positives (TP) | 558 |
| False Positives (FP) | 666 |
| False Negatives (FN) | 1421 |
| True Negatives (TN) | 6507 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 17.1% | 38.9% | 22.8% | 0.29 | 2,080 |
| **middle** | Middle 10 Sites | 17.2% | 44.4% | 26.0% | 0.33 | 4,160 |
| **bottom_7** | Bottom 7 Small Sites | 31.2% | 48.7% | 32.0% | 0.39 | 2,912 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 44.9% | 0.71 | 9,240 |
| Day 02 | 48.0% | 0.91 | 9,152 |
| Day 03 | 47.8% | 0.87 | 9,152 |
| Day 04 | 50.5% | 0.59 | 9,152 |
| Day 05 | 44.7% | 1.11 | 9,152 |
| Day 06 | 37.2% | 1.38 | 9,152 |
| Day 07 | 55.6% | 0.70 | 9,152 |
| Day 08 | 45.4% | 0.72 | 9,152 |
| Day 09 | 49.0% | 0.93 | 9,064 |
| Day 10 | 48.5% | 0.88 | 9,064 |
| Day 11 | 52.1% | 0.60 | 9,064 |
| Day 12 | 45.4% | 1.13 | 9,064 |
| Day 13 | 37.4% | 1.37 | 9,064 |
| Day 14 | 56.2% | 0.70 | 9,064 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 21.1% | 45.3% | 0.89 | 31,922 |
| **B** | 21.4% | 45.9% | 0.90 | 31,922 |
| **O** | 20.1% | 43.1% | 0.89 | 31,922 |
| **AB** | 34.4% | 65.5% | 0.91 | 31,922 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 17.7% | 35.3% | 0.94 | 29,020 |
| **middle** | Mid-size hospitals | 23.4% | 52.3% | 0.88 | 58,040 |
| **bottom_7** | Small / intermittent hospitals | 35.9% | 79.6% | 0.90 | 40,628 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 24.6% | 74.9% | 1.44 | 7,136 |
| **Non-Holiday** | 20.3% | 43.8% | 0.87 | 120,552 |
