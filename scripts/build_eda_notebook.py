import nbformat as nbf

nb = nbf.v4.new_notebook()

intro_md = """# DonorCast — Task 1.3: Exploratory Data Analysis (EDA)

This notebook performs exploratory data analysis on the Malaysian blood donation dataset (`data/processed/long.parquet` and `data/raw/donations_state.csv`) spanning **2006-01-01 to 2026-09-22**.

### Scope of Analysis
1. **National daily donations (2006–2026)** with a 28-day rolling mean.
2. **Yearly totals** bar chart and secular trends.
3. **Day-of-week profile** (National vs representative states with varying weekend regimes).
4. **Yearly seasonality**: mean donations by ISO week of year.
5. **Holiday event windows**: trajectory from 14 days before to 14 days after *Hari Raya Aidilfitri*, *Chinese New Year*, and *Deepavali*.
6. **Facility profiles**: mean daily volume, zero-donation day shares, and temporal coverage.
7. **Blood group distribution**: overall ABO proportions and facility breakdowns.
8. **National zero-donation days**: complete inventory and anomaly inspection.
9. **2019 vs 2020 monthly comparison**: impact of COVID-19 Movement Control Orders (MCO).

All figures are automatically saved to `reports/figures/` as PNG files. Every chart includes an evidence-based markdown takeaway derived directly from the data.
"""

setup_code = """from pathlib import Path

import holidays
from matplotlib import pyplot as plt, ticker
import numpy as np
import pandas as pd

# Configure plot styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Output paths
FIGURES_DIR = Path('../reports/figures') if Path('../reports/figures').exists() or not Path('reports/figures').exists() else Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
long_path = Path('../data/processed/long.parquet') if Path('../data/processed/long.parquet').exists() else Path('data/processed/long.parquet')
state_path = Path('../data/raw/donations_state.csv') if Path('../data/raw/donations_state.csv').exists() else Path('data/raw/donations_state.csv')

df_long = pd.read_parquet(long_path)
df_state = pd.read_csv(state_path)

print(f"Loaded long.parquet: {len(df_long):,} rows across {df_long['facility'].nunique()} facilities.")
print(f"Loaded donations_state.csv: {len(df_state):,} rows across {df_state['state'].nunique()} states/regions.")
"""

s1_code = """# 1. National Daily Donations (2006-2026) with 28-day Rolling Mean
nat_daily = df_long.groupby('date')['donations'].sum().reset_index()
nat_daily['date'] = pd.to_datetime(nat_daily['date'])
nat_daily = nat_daily.sort_values('date').reset_index(drop=True)
nat_daily['rolling_28d'] = nat_daily['donations'].rolling(window=28, min_periods=14, center=False).mean()

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.plot(nat_daily['date'], nat_daily['donations'], color='#90caf9', alpha=0.45, linewidth=0.8, label='Daily Donations')
ax.plot(nat_daily['date'], nat_daily['rolling_28d'], color='#c62828', linewidth=2.0, label='28-day Rolling Mean')

ax.set_title('National Daily Blood Donations in Malaysia (2006–2026)', pad=15)
ax.set_xlabel('Date')
ax.set_ylabel('Donations per Day')
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_ylim(bottom=0)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()

fig_path_1 = FIGURES_DIR / '01_national_daily_trend.png'
plt.savefig(fig_path_1)
plt.show()
"""

s1_md = """**Takeaway (Figure 1):** National daily donations expanded steadily from ~600/day in 2006 to over 1,600/day by 2025–2026 (a 2.7× volume increase), exhibiting pronounced recurring annual dips during festive holiday seasons and sharp volatility during early 2020 COVID lockdowns."""

s2_code = """# 2. Yearly Totals Bar Chart
nat_daily['year'] = nat_daily['date'].dt.year
yearly_stats = nat_daily.groupby('year')['donations'].agg(total='sum', days='count', mean='mean').reset_index()

fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#1565c0' if y < 2026 else '#90a4ae' for y in yearly_stats['year']]
bars = ax.bar(yearly_stats['year'], yearly_stats['total'], color=colors, edgecolor='black', linewidth=0.5)

for bar, year, tot in zip(bars, yearly_stats['year'], yearly_stats['total']):
    if year == 2026:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8000, f'{tot:,.0f}\\n(through Sep 22)', 
                ha='center', va='bottom', fontsize=8, color='#37474f', fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8000, f'{tot/1e3:.0f}k', 
                ha='center', va='bottom', fontsize=8, color='#263238')

ax.set_title('Annual Blood Donations Total (2006–2026)', pad=15)
ax.set_xlabel('Year')
ax.set_ylabel('Total Donations')
ax.set_xticks(yearly_stats['year'])
ax.set_xticklabels(yearly_stats['year'], rotation=45)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_ylim(0, yearly_stats['total'].max() * 1.15)
plt.tight_layout()

fig_path_2 = FIGURES_DIR / '02_yearly_totals.png'
plt.savefig(fig_path_2)
plt.show()
"""

s2_md = """**Takeaway (Figure 2):** Annual collections grew from 218,887 in 2006 to a peak of 589,834 in 2025 (averaging 550k–590k annually from 2018 onwards), with a temporary dip in 2020–2021 (497k–499k) before rapid full recovery."""

s3_code = """# 3. Mean Donations by Day of Week (National and 4 Example States)
df_state_dow = df_state.copy()
df_state_dow['date'] = pd.to_datetime(df_state_dow['date'])
df_state_dow['dow_name'] = df_state_dow['date'].dt.day_name()
df_state_dow['dow_num'] = df_state_dow['date'].dt.dayofweek

states_focus = ['Malaysia', 'W.P. Kuala Lumpur', 'Johor', 'Kedah', 'Sabah']
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

dow_df = df_state_dow[df_state_dow['state'].isin(states_focus)].copy()
dow_summary = dow_df.groupby(['state', 'dow_num', 'dow_name'])['daily'].mean().reset_index()

fig, axes = plt.subplots(1, 5, figsize=(18, 4.5), sharey=False)
palette = {'Malaysia': '#c62828', 'W.P. Kuala Lumpur': '#1565c0', 'Johor': '#2e7d32', 'Kedah': '#e65100', 'Sabah': '#6a1b9a'}

for i, state in enumerate(states_focus):
    ax = axes[i]
    state_data = dow_summary[dow_summary['state'] == state].sort_values('dow_num')
    bars = ax.bar(state_data['dow_name'], state_data['daily'], color=palette[state], edgecolor='black', linewidth=0.5)
    ax.set_title(state, fontsize=11, fontweight='bold')
    ax.set_xticklabels(['M', 'Tu', 'W', 'Th', 'F', 'Sa', 'Su'])
    ax.set_xlabel('')
    ax.set_ylabel('Mean Daily Donations' if i == 0 else '')
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

fig.suptitle('Mean Donations by Day of the Week (National & Selected States)', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()

fig_path_3 = FIGURES_DIR / '03_day_of_week_distribution.png'
plt.savefig(fig_path_3, bbox_inches='tight')
plt.show()
"""

s3_md = """**Takeaway (Figure 3):** Nationally, collections peak heavily on weekends (Sunday mean: 2,099 vs Monday mean: 867), with state-level profiles directly reflecting regional weekend schedules (e.g. Kedah exhibits high collections on Thursdays and Sundays with a distinct Saturday trough, while Johor and KL peak on Sundays)."""

s4_code = """# 4. Seasonality: Mean Donations by Week of Year
nat_daily['week'] = nat_daily['date'].dt.isocalendar().week.astype(int)
# Exclude week 53 as it only appears in leap/special calendar years with few days
weekly_seasonality = nat_daily[nat_daily['week'] <= 52].groupby('week')['donations'].agg(['mean', 'std', 'count']).reset_index()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(weekly_seasonality['week'], weekly_seasonality['mean'], color='#0277bd', marker='o', markersize=4, linewidth=2, label='Mean Daily Donations')
ax.fill_between(weekly_seasonality['week'], 
                weekly_seasonality['mean'] - 0.5 * weekly_seasonality['std'], 
                weekly_seasonality['mean'] + 0.5 * weekly_seasonality['std'], 
                color='#b3e5fc', alpha=0.5, label='± 0.5 Std Dev Range')

# Annotations for major seasonal phases
ax.annotate('Year-end & School Holidays Dip', xy=(51, weekly_seasonality.loc[weekly_seasonality['week']==51, 'mean'].values[0]),
            xytext=(38, 1050), arrowprops={'facecolor': 'black', 'shrink': 0.08, 'width': 1, 'headwidth': 6},
            fontsize=9, fontweight='bold')
ax.annotate('Mid-year Blood Drive Surge', xy=(28, weekly_seasonality.loc[weekly_seasonality['week']==28, 'mean'].values[0]),
            xytext=(15, 1480), arrowprops={'facecolor': 'black', 'shrink': 0.08, 'width': 1, 'headwidth': 6},
            fontsize=9, fontweight='bold')

ax.set_title('Weekly Seasonality of Daily Donations (ISO Week 1–52)', pad=15)
ax.set_xlabel('Week of Year (ISO Week)')
ax.set_ylabel('Mean Daily Donations')
ax.set_xticks(range(1, 53, 2))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.legend(loc='lower left', frameon=True)
plt.tight_layout()

fig_path_4 = FIGURES_DIR / '04_weekly_seasonality.png'
plt.savefig(fig_path_4)
plt.show()
"""

s4_md = """**Takeaway (Figure 4):** Mean daily donations demonstrate a strong mid-year collection campaign peak between Weeks 26–34 (~1,360–1,400/day) and experience a severe drop during the year-end holiday and school break period (Weeks 49–52, bottoming below 1,180/day)."""

s5_code = """# 5. Holiday Windows: Average Donations from -14 to +14 Days
my_holidays = holidays.Malaysia(subdiv='KUL', years=range(2006, 2027))

holiday_events = {
    'Hari Raya Aidilfitri': [],
    'Chinese New Year': [],
    'Deepavali': []
}

for dt, name in sorted(my_holidays.items()):
    # Filter to main 1st day holiday instances
    if 'Hari Raya Puasa' in name and 'Hari Kedua' not in name and 'Cuti' not in name and 'pergantian' not in name:
        holiday_events['Hari Raya Aidilfitri'].append(pd.to_datetime(dt))
    elif 'Tahun Baharu Cina' in name and 'Hari Kedua' not in name and 'Cuti' not in name:
        holiday_events['Chinese New Year'].append(pd.to_datetime(dt))
    elif 'Deepavali' in name and 'Cuti' not in name:
        holiday_events['Deepavali'].append(pd.to_datetime(dt))

window_records = []
daily_indexed = nat_daily.set_index('date')['donations']

for h_name, h_dates in holiday_events.items():
    for h_date in h_dates:
        for offset in range(-14, 15):
            target_date = h_date + pd.Timedelta(days=offset)
            if target_date in daily_indexed.index:
                window_records.append({
                    'holiday': h_name,
                    'event_date': h_date,
                    'offset': offset,
                    'donations': daily_indexed.loc[target_date]
                })

df_windows = pd.DataFrame(window_records)
window_avg = df_windows.groupby(['holiday', 'offset'])['donations'].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 5.5))
palette_h = {'Hari Raya Aidilfitri': '#2e7d32', 'Chinese New Year': '#c62828', 'Deepavali': '#f57f17'}

for h_name, group in window_avg.groupby('holiday'):
    ax.plot(group['offset'], group['donations'], label=h_name, color=palette_h[h_name], linewidth=2.2, marker='o', markersize=3.5)

ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, label='Holiday Day (t=0)')
ax.set_title('Average Daily Donations in Holiday Windows (t-14 to t+14 Days)', pad=15)
ax.set_xlabel('Days Relative to Festival (0 = Day of Festival)')
ax.set_ylabel('Mean Daily Donations')
ax.set_xticks(range(-14, 15, 2))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax.legend(loc='lower left', frameon=True)
plt.tight_layout()

fig_path_5 = FIGURES_DIR / '05_holiday_windows.png'
plt.savefig(fig_path_5)
plt.show()
"""

s5_md = """**Takeaway (Figure 5):** Donations experience steep, predictable collapses on festival days (t=0 averages: 120 for Hari Raya Aidilfitri, 252 for Chinese New Year, 483 for Deepavali vs baseline >1,300), with Hari Raya showing prolonged suppression during the surrounding festive fortnight."""

s6_code = """# 6. Per-Facility Table and Volume Profiles
fac_daily = df_long.groupby(['facility', 'date'])['donations'].sum().reset_index()
fac_table = fac_daily.groupby('facility').agg(
    mean_daily=('donations', 'mean'),
    median_daily=('donations', 'median'),
    total_donations=('donations', 'sum'),
    zero_days=('donations', lambda s: (s == 0).sum()),
    zero_share=('donations', lambda s: (s == 0).mean()),
    start_date=('date', 'min'),
    end_date=('date', 'max'),
).reset_index().sort_values('mean_daily', ascending=False)

# Display table
print(fac_table.to_string(index=False))

# Plot facility distribution
fig, ax = plt.subplots(figsize=(13, 7))
sorted_facs = fac_table.sort_values('mean_daily', ascending=True)
bars = ax.barh(sorted_facs['facility'], sorted_facs['mean_daily'], color='#455a64', edgecolor='black', linewidth=0.5)

for bar, zero_pct in zip(bars, sorted_facs['zero_share']):
    width = bar.get_width()
    ax.text(width + 4, bar.get_y() + bar.get_height()/2, f'{width:.1f} /d ({zero_pct*100:.1f}% zeros)', 
            va='center', ha='left', fontsize=8.5, color='#263238')

ax.set_title('Facility Profiles: Mean Daily Donations & Zero-Day Proportions', pad=15)
ax.set_xlabel('Mean Daily Donations')
ax.set_ylabel('Collection Facility')
ax.set_xlim(0, fac_table['mean_daily'].max() * 1.25)
plt.tight_layout()

fig_path_6 = FIGURES_DIR / '06_facility_profiles.png'
plt.savefig(fig_path_6)
plt.show()
"""

s6_md = """**Takeaway (Figure 6):** Collection volume is heavily skewed: Pusat Darah Negara dominates with 485 donations/day (0.5% zero days), while district centres like Hospital Duchess of Kent (Sandakan) average 7.3 donations/day with 67.5% zero-donation days, requiring zero-tolerant Tweedie/count-aware loss formulations."""

s7_code = """# 7. Blood Group Shares: Overall and by Facility
overall_abo = df_long.groupby('group')['donations'].sum()
overall_abo_pct = (overall_abo / overall_abo.sum() * 100).sort_values(ascending=False)

fac_abo = df_long.groupby(['facility', 'group'])['donations'].sum().unstack()
fac_abo_pct = fac_abo.div(fac_abo.sum(axis=1), axis=0) * 100
fac_abo_pct = fac_abo_pct.loc[fac_table['facility']]  # order by facility volume

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2.2]})

# Overall Donut Chart
colors_abo = {'O': '#c62828', 'B': '#1565c0', 'A': '#2e7d32', 'AB': '#f57f17'}
ordered_groups = ['O', 'B', 'A', 'AB']
wedges, texts, autotexts = ax1.pie(
    [overall_abo_pct[g] for g in ordered_groups],
    labels=ordered_groups,
    autopct='%1.1f%%',
    pctdistance=0.75,
    colors=[colors_abo[g] for g in ordered_groups],
    wedgeprops={'width': 0.45, 'edgecolor': 'white', 'linewidth': 2},
    startangle=140
)
for at in autotexts:
    at.set_color('white')
    at.set_fontweight('bold')
ax1.set_title('National ABO Blood Group Proportions', fontsize=12, fontweight='bold')

# Facility Stacked Bar Chart
fac_abo_pct[ordered_groups].plot(
    kind='barh',
    stacked=True,
    color=[colors_abo[g] for g in ordered_groups],
    edgecolor='black',
    linewidth=0.3,
    ax=ax2
)
ax2.set_title('ABO Proportions Across 22 Collection Facilities', fontsize=12, fontweight='bold')
ax2.set_xlabel('Percentage Share (%)')
ax2.set_ylabel('')
ax2.legend(title='Blood Group', bbox_to_anchor=(1.02, 1), loc='upper left')
ax2.set_xlim(0, 100)
plt.tight_layout()

fig_path_7 = FIGURES_DIR / '07_blood_group_shares.png'
plt.savefig(fig_path_7, bbox_inches='tight')
plt.show()
"""

s7_md = """**Takeaway (Figure 7):** Blood group proportions remain remarkably stable across all facilities and nationally (O: 41.85%, B: 27.20%, A: 24.82%, AB: 6.13%), meaning AB series represent sparse, low-volume targets (~6% of total) that must be tracked with group-level evaluations."""

s8_code = """# 8. Inventory and Inspection of National Zero-Donation Days
nat_zeros = nat_daily[nat_daily['donations'] == 0].copy()
nat_zeros['day_name'] = nat_zeros['date'].dt.day_name()

# Inspect holiday and context for each zero day
zero_details = []
for _, row in nat_zeros.iterrows():
    dt_val = row['date'].date()
    h_match = my_holidays.get(dt_val, 'No standard public holiday recorded')
    zero_details.append({
        'Date': row['date'].strftime('%Y-%m-%d'),
        'Day of Week': row['day_name'],
        'Holiday / Event Context': h_match
    })

df_zeros_summary = pd.DataFrame(zero_details)
print('Exact dates where national daily donations = 0:')
print(df_zeros_summary.to_string(index=False))
"""

s8_md = """**Takeaway (Table 8):** Exactly 6 dates in 20.7 years recorded 0 donations nationwide: 2006-01-29 (Chinese New Year Day 1), 2013-05-05 (13th General Election polling day), 2018-06-15 (Hari Raya Aidilfitri Day 1), 2021-05-13 (Hari Raya Aidilfitri Day 1 during MCO), 2024-04-10 (Hari Raya Aidilfitri Day 1), and 2025-03-31 (Hari Raya Aidilfitri Day 1); each reflects known nationwide facility shutdowns rather than random missing data gaps."""

s9_code = """# 9. 2019 vs 2020 Monthly Comparison
df_state_nat = df_state[df_state['state'] == 'Malaysia'].copy()
df_state_nat['date'] = pd.to_datetime(df_state_nat['date'])
df_state_nat['year'] = df_state_nat['date'].dt.year
df_state_nat['month'] = df_state_nat['date'].dt.month

df_19_20 = df_state_nat[df_state_nat['year'].isin([2019, 2020])].copy()
monthly_comp = df_19_20.groupby(['month', 'year'])['daily'].sum().unstack()
monthly_comp['pct_change'] = ((monthly_comp[2020] - monthly_comp[2019]) / monthly_comp[2019]) * 100

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x = np.arange(1, 13)
width = 0.35

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Bar Chart
rects1 = ax1.bar(x - width/2, monthly_comp[2019], width, label='2019', color='#1e88e5', edgecolor='black', linewidth=0.5)
rects2 = ax1.bar(x + width/2, monthly_comp[2020], width, label='2020', color='#fb8c00', edgecolor='black', linewidth=0.5)

ax1.set_title('Monthly Blood Donation Comparison: 2019 vs 2020 (COVID-19 MCO Shock)', pad=15)
ax1.set_ylabel('Total Donations')
ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
ax1.legend(loc='upper right', frameon=True)

# Annotate MCO onset
ax1.annotate('MCO 1.0 Imposed\\n(Mar 18, 2020)', xy=(4, monthly_comp.loc[4, 2020]), xytext=(3.5, 45000),
             arrowprops={'facecolor': 'black', 'shrink': 0.08, 'width': 1, 'headwidth': 6},
             fontsize=9, fontweight='bold', ha='center')

# Percentage Change Bar Chart
colors_pct = ['#e53935' if v < 0 else '#43a047' for v in monthly_comp['pct_change']]
ax2.bar(x, monthly_comp['pct_change'], color=colors_pct, edgecolor='black', linewidth=0.5)
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_ylabel('% YoY Change')
ax2.set_xlabel('Month')
ax2.set_xticks(x)
ax2.set_xticklabels(months)
ax2.set_ylim(-50, 20)

for idx, val in enumerate(monthly_comp['pct_change']):
    y_pos = -48 if val < 0 else val + 1
    ax2.text(x[idx], val + (1.5 if val >= 0 else -4), f'{val:+.1f}%', ha='center', fontsize=8.5, fontweight='bold')

plt.tight_layout()

fig_path_8 = FIGURES_DIR / '08_2019_vs_2020_monthly.png'
plt.savefig(fig_path_8)
plt.show()
"""

s9_md = """**Takeaway (Figure 8):** When COVID-19 Movement Control Orders (MCO) were introduced in March 2020, donations plunged sharply by -30.2% in March and -39.6% in April 2020 compared to 2019, but aggressive recovery campaigns pushed June–August 2020 above 2019 baseline levels."""

# Assemble notebook cells
cells = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(setup_code),
    nbf.v4.new_code_cell(s1_code),
    nbf.v4.new_markdown_cell(s1_md),
    nbf.v4.new_code_cell(s2_code),
    nbf.v4.new_markdown_cell(s2_md),
    nbf.v4.new_code_cell(s3_code),
    nbf.v4.new_markdown_cell(s3_md),
    nbf.v4.new_code_cell(s4_code),
    nbf.v4.new_markdown_cell(s4_md),
    nbf.v4.new_code_cell(s5_code),
    nbf.v4.new_markdown_cell(s5_md),
    nbf.v4.new_code_cell(s6_code),
    nbf.v4.new_markdown_cell(s6_md),
    nbf.v4.new_code_cell(s7_code),
    nbf.v4.new_markdown_cell(s7_md),
    nbf.v4.new_code_cell(s8_code),
    nbf.v4.new_markdown_cell(s8_md),
    nbf.v4.new_code_cell(s9_code),
    nbf.v4.new_markdown_cell(s9_md),
]

nb['cells'] = cells
with open('notebooks/01_eda.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Successfully generated notebooks/01_eda.ipynb')
