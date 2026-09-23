# Evaluation Report: lgbm (TEST Split)

- **Model**: `lgbm`
- **Split**: `test`
- **Evaluated at**: `2026-09-23T11:04:21.772018+00:00`
- **Total Rows Evaluated**: 109,208

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **16.9%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **37.9%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **0.79** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | 1.45 | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | 2.00 | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 7,832
- **Actual Shortfall Windows**: 1,841 (Prevalence: **23.5%**)
- **Predicted Shortfall Windows**: 1,430

| Metric | Value |
|---|---|
| **Prevalence** | **23.5%** |
| **Precision** | **62.0%** |
| **Recall** | **48.2%** |
| **F1 Score** | **0.54** |
| True Positives (TP) | 887 |
| False Positives (FP) | 543 |
| False Negatives (FN) | 954 |
| True Negatives (TN) | 5448 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 19.1% | 69.9% | 54.7% | 0.61 | 1,780 |
| **middle** | Middle 10 Sites | 19.9% | 58.7% | 49.1% | 0.53 | 3,560 |
| **bottom_7** | Bottom 7 Small Sites | 31.8% | 61.8% | 44.6% | 0.52 | 2,492 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 40.1% | 0.74 | 7,920 |
| Day 02 | 37.6% | 0.77 | 7,832 |
| Day 03 | 38.1% | 0.76 | 7,832 |
| Day 04 | 44.5% | 0.57 | 7,832 |
| Day 05 | 37.3% | 0.98 | 7,832 |
| Day 06 | 31.8% | 1.16 | 7,832 |
| Day 07 | 43.8% | 0.55 | 7,832 |
| Day 08 | 40.5% | 0.75 | 7,832 |
| Day 09 | 37.8% | 0.77 | 7,744 |
| Day 10 | 38.4% | 0.77 | 7,744 |
| Day 11 | 44.4% | 0.58 | 7,744 |
| Day 12 | 37.5% | 0.98 | 7,744 |
| Day 13 | 31.9% | 1.16 | 7,744 |
| Day 14 | 43.8% | 0.55 | 7,744 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 16.6% | 37.7% | 0.79 | 27,302 |
| **B** | 16.5% | 37.7% | 0.78 | 27,302 |
| **O** | 16.1% | 35.8% | 0.79 | 27,302 |
| **AB** | 25.6% | 55.1% | 0.81 | 27,302 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 12.6% | 28.0% | 0.80 | 24,820 |
| **middle** | Mid-size hospitals | 19.6% | 44.2% | 0.79 | 49,640 |
| **bottom_7** | Small / intermittent hospitals | 30.4% | 69.6% | 0.80 | 34,748 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 17.1% | 48.1% | 0.94 | 6,936 |
| **Non-Holiday** | 16.8% | 37.1% | 0.78 | 102,272 |
