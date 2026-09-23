# External Calendar Data (`data/external/`)

This directory contains hand-built, domain-specific calendar datasets used to construct feature sets for forecasting blood donations across Malaysia. Every row includes a verifiable `source` entry citing the official government circular, statutory gazette, or authority announcement.

---

## 1. `school_holidays.csv`

- **Purpose**: Tracks major Malaysian Ministry of Education (Kementerian Pendidikan Malaysia - KPM) school vacation periods from 2006 to 2026. Student donations and campus mobile drives experience major declines during these windows.
- **Columns**:
  - `start_date` (`YYYY-MM-DD`): First day of the school break.
  - `end_date` (`YYYY-MM-DD`): Last day of the school break (inclusive).
  - `description` (`str`): Type of school break (e.g., `Term 1 Break`, `Mid-Year Break`, `Term 2 Break`, `Year-End Break`).
  - `verified` (`bool`): `true` if directly cross-referenced against published KPM Surat Pekeliling Ikhtisas / official press releases; `false` if estimated/interpolated based on historic calendar patterns.
  - `source` (`str`): Primary reference document name or circular number.
- **Notes & Gaps**:
  - Years **2008–2026** are verified against published KPM Surat Pekeliling Ikhtisas (SPI) / Surat Siaran circulars.
  - Years **2006–2007** use standard term week alignments and are flagged with `verified=false` due to archived primary circular document unavailability.
  - During the COVID-19 pandemic (2020–2024), KPM adjusted the academic calendar (shifting the academic year start from January to March). These revised schedules are reflected with their specific amendment circulars.

---

## 2. `state_weekends.csv`

- **Purpose**: Defines whether a state follows a Friday–Saturday (`FRI_SAT`) or Saturday–Sunday (`SAT_SUN`) official rest weekend over time.
- **Columns**:
  - `state` (`str`): State name matching the official naming in `donations_state.csv` and hospital locations.
  - `weekend_type` (`str`): `FRI_SAT` or `SAT_SUN`.
  - `valid_from` (`YYYY-MM-DD`): Effective start date of the weekend policy.
  - `valid_to` (`YYYY-MM-DD`): Effective end date of the weekend policy.
  - `source` (`str`): State gazette, royal decree, or federal circular reference.
- **Historical Transitions**:
  - **Johor**:
    - Prior to 2014: `SAT_SUN` (2006-01-01 to 2013-12-31).
    - 2014–2024: `FRI_SAT` (effective 2014-01-01 by decree of the Sultan of Johor; gazetted in Warta Kerajaan Johor J.P.U. 45/2013).
    - 2025 onwards: `SAT_SUN` (effective 2025-01-01 by decree of the Regent of Johor, Tunku Mahkota Ismail).
  - **Kedah, Kelantan, Terengganu**:
    - Consistently follow `FRI_SAT` across the 2006–2026 dataset period.
  - **All other States & Federal Territories** (Melaka, Negeri Sembilan, Pahang, Perak, Perlis, Pulau Pinang, Sabah, Sarawak, Selangor, W.P. Kuala Lumpur, W.P. Labuan, W.P. Putrajaya):
    - Consistently follow `SAT_SUN` under Federal Civil Service 5-Day Work Week Circular JPA/PEN/229/6/1.

---

## 3. `mco_periods.csv`

- **Purpose**: Documents all phases and major national lockdown periods under the Movement Control Order (Perintah Kawalan Pergerakan - PKP) and the National Recovery Plan (Pelan Pemulihan Negara - PPN) during COVID-19.
- **Columns**:
  - `start_date` (`YYYY-MM-DD`): Start date of the phase.
  - `end_date` (`YYYY-MM-DD`): End date of the phase (inclusive).
  - `phase` (`str`): Official designation (e.g., `MCO Phase 1`, `Conditional MCO (CMCO / PKPB)`, `Recovery MCO (RMCO / PKPP)`, `Full MCO (FMCO)`).
  - `source` (`str`): Legal gazettes under the Prevention and Control of Infectious Diseases Act 1988 [Act 342] and official Prime Minister's Department / MKN announcements.

---

## 4. `general_elections.csv`

- **Purpose**: Polling days for Malaysian Parliamentary General Elections (Pilihan Raya Umum - PRU). Polling days cause notable disruptions to civic routines and mobile donation campaigns.
- **Columns**:
  - `date` (`YYYY-MM-DD`): Polling day date.
  - `election` (`str`): Election identifier (`GE12`, `GE13`, `GE14`, `GE15`).
  - `source` (`str`): Suruhanjaya Pilihan Raya Malaysia (SPR - Election Commission of Malaysia) official gazettes.
- **Coverage**:
  - `2008-03-08`: GE12 (12th General Election)
  - `2013-05-05`: GE13 (13th General Election)
  - `2018-05-09`: GE14 (14th General Election)
  - `2022-11-19`: GE15 (15th General Election)
