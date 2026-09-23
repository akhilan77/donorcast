# Baseline Models Evaluation Report (VAL Split)

- **Evaluation Period**: `2023-01-01` to `2024-12-31` (Validation Split)
- **Evaluation Frequency**: Weekly (every Monday origin)
- **Forecast Horizons**: 1 to 14 days ahead
- **Primary Metric**: **WAPE_7D** (7-day cumulative sum error)
- **Models Evaluated**:
  - `M0`: **Seasonal Naive** (most recent same weekday $\le t$)
  - `M0b`: **Weekday Moving Average** (mean of last 4 same weekdays $\le t$)

---

## 1. Headline Regression Metrics Comparison

| Model | Description | WAPE_7D (Primary) | Daily WAPE | MASE | Rows Evaluated |
|---|---|---|---|---|---|
| **M0** | Seasonal Naive (last week) | **25.7%** | **55.7%** | **1.10** | 127,688 |
| **M0b** | Weekday Moving Avg (4-wk) | **21.5%** | **45.7%** | **0.90** | 127,688 |

> **Winner on Validation**: **`M0b`** (Lower WAPE_7D: 21.5% vs 25.7%)

---

## 2. Shortfall Classification Metrics (7-Day Ahead)

- **Actual Shortfall Prevalence**: **21.6%** (1,979 of 9,152 windows)

| Model | Prevalence | Precision | Recall | F1 Score | TP | FP | FN | TN | Total Windows |
|---|---|---|---|---|---|---|---|---|---|
| **M0** | 21.6% | 35.9% | 37.6% | 0.37 | 745 | 1328 | 1234 | 5845 | 9,152 |
| **M0b** | 21.6% | 45.6% | 28.2% | 0.35 | 558 | 666 | 1421 | 6507 | 9,152 |

### Shortfall Metrics by Facility Tier

| Tier | Model | Prevalence | Precision | Recall | F1 Score | Windows |
|---|---|---|---|---|---|---|
| **top_5** (Top 5 High-Volume Sites) | M0 | 17.1% | 25.4% | 28.7% | 0.27 | 2,080 |
| | **M0b** | 17.1% | 38.9% | 22.8% | 0.29 | 2,080 |
| **middle** (Middle 10 Sites) | M0 | 17.2% | 33.0% | 34.9% | 0.34 | 4,160 |
| | **M0b** | 17.2% | 44.4% | 26.0% | 0.33 | 4,160 |
| **bottom_7** (Bottom 7 Small Sites) | M0 | 31.2% | 43.0% | 43.3% | 0.43 | 2,912 |
| | **M0b** | 31.2% | 48.7% | 32.0% | 0.39 | 2,912 |

---

## 3. Breakdown by Horizon (h = 1..14)

| Horizon | M0 WAPE | M0 MASE | M0b WAPE | M0b MASE | Better Model |
|---|---|---|---|---|---|
| Day 01 | 53.8% | 0.85 | 44.9% | 0.71 | **M0b** |
| Day 02 | 57.8% | 1.09 | 48.0% | 0.91 | **M0b** |
| Day 03 | 58.8% | 1.06 | 47.8% | 0.87 | **M0b** |
| Day 04 | 61.1% | 0.71 | 50.5% | 0.59 | **M0b** |
| Day 05 | 54.1% | 1.34 | 44.7% | 1.11 | **M0b** |
| Day 06 | 46.5% | 1.72 | 37.2% | 1.38 | **M0b** |
| Day 07 | 64.3% | 0.82 | 55.6% | 0.70 | **M0b** |
| Day 08 | 55.1% | 0.85 | 45.4% | 0.72 | **M0b** |
| Day 09 | 59.4% | 1.12 | 49.0% | 0.93 | **M0b** |
| Day 10 | 60.1% | 1.08 | 48.5% | 0.88 | **M0b** |
| Day 11 | 63.6% | 0.73 | 52.1% | 0.60 | **M0b** |
| Day 12 | 56.1% | 1.39 | 45.4% | 1.13 | **M0b** |
| Day 13 | 47.4% | 1.73 | 37.4% | 1.37 | **M0b** |
| Day 14 | 66.9% | 0.84 | 56.2% | 0.70 | **M0b** |

---

## 4. Breakdown by Blood Group

| Group | M0 WAPE_7D | M0 Daily WAPE | M0 MASE | M0b WAPE_7D | M0b Daily WAPE | M0b MASE | Better (7D) |
|---|---|---|---|---|---|---|---|
| **A** | 25.1% | 55.7% | 1.10 | 21.1% | 45.3% | 0.89 | **M0b** |
| **B** | 25.5% | 56.0% | 1.10 | 21.4% | 45.9% | 0.90 | **M0b** |
| **O** | 24.3% | 52.6% | 1.09 | 20.1% | 43.1% | 0.89 | **M0b** |
| **AB** | 39.0% | 78.8% | 1.09 | 34.4% | 65.5% | 0.91 | **M0b** |

---

## 5. Breakdown by Facility Tier

| Tier | Description | M0 WAPE_7D | M0 Daily WAPE | M0 MASE | M0b WAPE_7D | M0b Daily WAPE | M0b MASE | Better (7D) |
|---|---|---|---|---|---|---|---|---|
| **top_5** | Top 5 High-Volume Sites | 20.4% | 43.1% | 1.16 | 17.7% | 35.3% | 0.94 | **M0b** |
| **middle** | Middle 10 Sites | 28.4% | 64.1% | 1.07 | 23.4% | 52.3% | 0.88 | **M0b** |
| **bottom_7** | Bottom 7 Small Sites | 45.2% | 96.3% | 1.08 | 35.9% | 79.6% | 0.90 | **M0b** |

---

## 6. Holiday Window Performance

| Window | M0 WAPE_7D | M0 Daily WAPE | M0 MASE | M0b WAPE_7D | M0b Daily WAPE | M0b MASE | Better (7D) |
|---|---|---|---|---|---|---|---|
| **Public Holiday** | 28.8% | 81.8% | 1.60 | 24.6% | 74.9% | 1.44 | **M0b** |
| **Non-Holiday** | 24.5% | 54.0% | 1.07 | 20.3% | 43.8% | 0.87 | **M0b** |
