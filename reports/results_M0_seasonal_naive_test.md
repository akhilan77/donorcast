# Evaluation Report: M0_seasonal_naive (TEST Split)

- **Model**: `M0_seasonal_naive`
- **Split**: `test`
- **Evaluated at**: `2026-09-23T10:58:05.550058+00:00`
- **Total Rows Evaluated**: 109,208

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **24.4%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **54.8%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **1.11** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | N/A | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | N/A | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 7,832
- **Actual Shortfall Windows**: 1,841 (Prevalence: **23.5%**)
- **Predicted Shortfall Windows**: 1,908

| Metric | Value |
|---|---|
| **Prevalence** | **23.5%** |
| **Precision** | **40.5%** |
| **Recall** | **41.9%** |
| **F1 Score** | **0.41** |
| True Positives (TP) | 772 |
| False Positives (FP) | 1136 |
| False Negatives (FN) | 1069 |
| True Negatives (TN) | 4855 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 19.1% | 37.0% | 35.9% | 0.36 | 1,780 |
| **middle** | Middle 10 Sites | 19.9% | 37.6% | 38.1% | 0.38 | 3,560 |
| **bottom_7** | Bottom 7 Small Sites | 31.8% | 44.2% | 48.0% | 0.46 | 2,492 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 57.2% | 1.00 | 7,920 |
| Day 02 | 52.2% | 1.06 | 7,832 |
| Day 03 | 53.7% | 1.05 | 7,832 |
| Day 04 | 60.5% | 0.76 | 7,832 |
| Day 05 | 53.1% | 1.37 | 7,832 |
| Day 06 | 46.8% | 1.66 | 7,832 |
| Day 07 | 65.1% | 0.78 | 7,832 |
| Day 08 | 58.6% | 1.02 | 7,832 |
| Day 09 | 55.4% | 1.10 | 7,744 |
| Day 10 | 56.2% | 1.07 | 7,744 |
| Day 11 | 63.5% | 0.78 | 7,744 |
| Day 12 | 54.2% | 1.39 | 7,744 |
| Day 13 | 48.7% | 1.70 | 7,744 |
| Day 14 | 65.4% | 0.78 | 7,744 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 23.9% | 54.6% | 1.11 | 27,302 |
| **B** | 24.4% | 54.7% | 1.10 | 27,302 |
| **O** | 23.0% | 52.1% | 1.12 | 27,302 |
| **AB** | 36.7% | 77.7% | 1.10 | 27,302 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 19.0% | 42.5% | 1.16 | 24,820 |
| **middle** | Mid-size hospitals | 27.2% | 62.6% | 1.11 | 49,640 |
| **bottom_7** | Small / intermittent hospitals | 43.9% | 95.0% | 1.07 | 34,748 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 28.3% | 75.6% | 1.43 | 6,936 |
| **Non-Holiday** | 22.8% | 53.4% | 1.09 | 102,272 |
