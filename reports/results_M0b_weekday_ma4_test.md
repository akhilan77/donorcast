# Evaluation Report: M0b_weekday_ma4 (TEST Split)

- **Model**: `M0b_weekday_ma4`
- **Split**: `test`
- **Evaluated at**: `2026-09-23T10:59:07.812286+00:00`
- **Total Rows Evaluated**: 109,208

---

## 1. Headline Regression Metrics

| Metric | Value | Description |
|---|---|---|
| **WAPE_7D (Primary)** | **21.3%** | 7-day cumulative sum WAPE |
| **WAPE (Daily)** | **45.2%** | Daily Weighted Absolute Percentage Error |
| **MASE** | **0.92** | Mean Absolute Scaled Error (vs 2020–2022 seasonal naive 7) |
| **Pinball Loss (p10)** | N/A | Quantile loss at 10th percentile |
| **Pinball Loss (p90)** | N/A | Quantile loss at 90th percentile |

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Shortfall Definition**: 7-day predicted sum < 0.8 × historical 3-year median for ISO week
- **Evaluated Windows**: 7,832
- **Actual Shortfall Windows**: 1,841 (Prevalence: **23.5%**)
- **Predicted Shortfall Windows**: 1,114

| Metric | Value |
|---|---|
| **Prevalence** | **23.5%** |
| **Precision** | **47.5%** |
| **Recall** | **28.7%** |
| **F1 Score** | **0.36** |
| True Positives (TP) | 529 |
| False Positives (FP) | 585 |
| False Negatives (FN) | 1312 |
| True Negatives (TN) | 5406 |

### Shortfall Metrics by Facility Tier

| Tier | Description | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 19.1% | 44.6% | 25.6% | 0.33 | 1,780 |
| **middle** | Middle 10 Sites | 19.9% | 42.9% | 21.7% | 0.29 | 3,560 |
| **bottom_7** | Bottom 7 Small Sites | 31.8% | 51.4% | 36.4% | 0.43 | 2,492 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | WAPE | MASE | Count |
|---|---|---|---|
| Day 01 | 47.6% | 0.83 | 7,920 |
| Day 02 | 44.3% | 0.90 | 7,832 |
| Day 03 | 45.2% | 0.87 | 7,832 |
| Day 04 | 52.7% | 0.66 | 7,832 |
| Day 05 | 44.2% | 1.14 | 7,832 |
| Day 06 | 37.8% | 1.35 | 7,832 |
| Day 07 | 53.6% | 0.64 | 7,832 |
| Day 08 | 48.5% | 0.85 | 7,832 |
| Day 09 | 45.3% | 0.91 | 7,744 |
| Day 10 | 46.3% | 0.88 | 7,744 |
| Day 11 | 53.4% | 0.66 | 7,744 |
| Day 12 | 44.7% | 1.15 | 7,744 |
| Day 13 | 38.2% | 1.36 | 7,744 |
| Day 14 | 53.8% | 0.64 | 7,744 |

---

## 4. Breakdown by Blood Group

| Group | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **A** | 21.0% | 45.3% | 0.92 | 27,302 |
| **B** | 21.3% | 45.3% | 0.91 | 27,302 |
| **O** | 20.0% | 42.7% | 0.92 | 27,302 |
| **AB** | 32.7% | 64.2% | 0.92 | 27,302 |

---

## 5. Breakdown by Facility Tier

| Tier | Description | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|---|
| **top_5** | PDN, Penang, JB, Ipoh, Melaka | 17.3% | 35.4% | 0.96 | 24,820 |
| **middle** | Mid-size hospitals | 23.3% | 50.9% | 0.90 | 49,640 |
| **bottom_7** | Small / intermittent hospitals | 36.3% | 79.6% | 0.90 | 34,748 |

---

## 6. Holiday Window Slices

| Window | WAPE_7D | Daily WAPE | MASE | Count |
|---|---|---|---|---|
| **Public Holiday** | 25.4% | 69.3% | 1.30 | 6,936 |
| **Non-Holiday** | 19.7% | 43.6% | 0.89 | 102,272 |
