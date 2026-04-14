# Data Sources

## pmi.csv (required, not included)

This file is not included in the repository as ISM Manufacturing PMI
data is proprietary.

**Format expected** (see `pmi_schema.csv`):
- Column `Date`: first calendar day of each month, YYYY-MM-DD
- Column `PMI`: ISM Manufacturing PMI reading, float in range [30, 70]
- Monthly frequency, no gaps

**Where to get it:**

| Source | Notes |
|--------|-------|
| [ISM website](https://www.ismworld.org) | Official source, manual download |
| [Quandl/Nasdaq Data Link: ISM/MAN_PMI](https://data.nasdaq.com) | Paid, clean historical series back to 1948 |
| * [FRED: MANEMP proxy](https://fred.stlouisfed.org) | Free. FRED does not carry ISM PMI directly but carries related series, such as CFNAI or INDPRO |

Place the file at `data/pmi.csv` before running `scripts/run_model.py`.