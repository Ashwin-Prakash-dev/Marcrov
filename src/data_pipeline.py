import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')


# ════════════════════════════════════
# Decay-fill utility
# ════════════════════════════════════

def decay_fill(
    monthly_series: pd.Series,
    daily_index: pd.DatetimeIndex,
    halflife_days: int = 10,
    max_staleness_days: int = None,
    release_index: pd.DatetimeIndex = None,
) -> pd.Series:
    """
    Forward-fill a sparse series onto daily_index, then apply exponential
    decay toward the long-run mean of the source series.

    Parameters
    ----------
    monthly_series : pd.Series
        Sparse source data (monthly PMI, weekly Jobless Claims).
    daily_index : pd.DatetimeIndex
        Target dense daily index (aligned to market-open days).
    halflife_days : int
        Half-life in calendar days for the exponential decay.
        After this many days the gap between the observed value and the
        long-run mean is halved.
    max_staleness_days : int or None
        If set, values forward-filled beyond this many days are replaced
        with NaN. Prevents indefinite propagation of a suspended release.
    release_index : pd.DatetimeIndex or None

    Returns
    -------
    pd.Series
        Dense daily series on daily_index with decay applied.
    """
    filled        = monthly_series.reindex(daily_index, method='ffill')
    long_run_mean = monthly_series.mean()

    if release_index is not None:
        release_points = pd.Series(
            daily_index.isin(release_index), index=daily_index
        )
    else:
        release_points = (filled != filled.shift(1))

    release_groups     = release_points.cumsum()
    days_since_release = filled.groupby(release_groups).cumcount()

    if max_staleness_days is not None:
        stale_mask         = days_since_release > max_staleness_days
        filled[stale_mask] = np.nan

    decay_weight = np.exp(-np.log(2) / halflife_days * days_since_release)
    return long_run_mean + (filled - long_run_mean) * decay_weight


# ══════════════════════════════════
# PMI loader
# ══════════════════════════════════
def load_pmi(
    filepath: str,
    daily_index: pd.DatetimeIndex,
    halflife_days: int = 10,
    max_staleness_days: int = 45,
) -> pd.Series:
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"PMI data file not found at '{filepath}'.\n"
            f"See data/README.md for instructions on obtaining this file."
        )
    
    pmi = pd.read_csv(filepath, parse_dates=['Date'])
    pmi = pmi.set_index('Date')
    pmi.index = pd.to_datetime(pmi.index).tz_localize(None)
    pmi = pmi.rename(columns={'PMI': 'ISM_PMI'})

    assert pmi['ISM_PMI'].max() < 100, (
        "PMI values should be in the 0-100 range. Check your CSV."
    )

    pmi_decayed = decay_fill(
        pmi['ISM_PMI'],
        daily_index,
        halflife_days=halflife_days,
        max_staleness_days=max_staleness_days,
        release_index=None,
    )

    print(
        f"  ISM PMI range: {pmi['ISM_PMI'].min():.1f}-{pmi['ISM_PMI'].max():.1f}"
        f"  |  halflife={halflife_days}d  |  max_staleness={max_staleness_days}d"
    )

    pmi_decayed.name = 'ISM_PMI'
    return pmi_decayed


# ════════════════════════════════════════════
# Main dataset builder
# ════════════════════════════════════════════
def fetch_macro_dataset(
    start_date: str = '2006-01-01',
    end_date: str = '2026-01-01',
    pmi_filepath: str = 'data/pmi.csv',
    ism_halflife: int = 10,
    claims_halflife: int = 5,
) -> pd.DataFrame:
   
    api_key = os.environ.get('FRED_API_KEY', '')
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found. "
            "Add it to your .env file or set it as an environment variable.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    # ── Market data ───────────────────────────────────────────────────────────
    print(f"Fetching market data {start_date} -> {end_date}...")
    tickers     = ['^VIX', 'DX-Y.NYB', 'SPY']
    market_data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    market_data.columns = ['DXY', 'SPY', 'VIX']
    market_data.index   = pd.to_datetime(market_data.index).tz_localize(None)

    # ── FRED data ─────────────────────────────────────────────────────────────
    print("Fetching FRED macro series (T10Y2Y, BAMLH0A0HYM2, ICSA)...")
    fred     = Fred(api_key=api_key)
    fred_map = {
        'T10Y2Y':       'Yield_Curve_Spread',
        'BAMLH0A0HYM2': 'Credit_Spread',
        'ICSA':         'Jobless_Claims',
    }

    econ_frames = {}
    for series_id, col_name in fred_map.items():
        s = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
        s.name                = col_name
        s.index               = pd.to_datetime(s.index).tz_localize(None)
        econ_frames[col_name] = s

    econ_data = pd.DataFrame(econ_frames)

    # ── Merge and clean ───────────────────────────────────────────────────────
    macro_df = pd.concat([market_data, econ_data], axis=1).sort_index()
    macro_df.ffill(inplace=True)
    macro_df = macro_df.loc[start_date:end_date]
    macro_df.dropna(inplace=True)
    daily_index = macro_df.index

    # ── PMI ───────────────────────────────────────────────────────────────────
    print("Loading ISM PMI (decay-weighted)...")
    pmi_series          = load_pmi(pmi_filepath, daily_index, halflife_days=ism_halflife)
    macro_df['ISM_PMI'] = pmi_series

    # ── Jobless Claims ────────────────────────────────────────────────────────
    print("Applying decay fill to Jobless Claims (calendar-based release detection)...")
    claims_raw       = econ_data['Jobless_Claims'].dropna()
    claims_raw.index = pd.to_datetime(claims_raw.index).tz_localize(None)
    macro_df['Jobless_Claims'] = decay_fill(
        claims_raw,
        daily_index,
        halflife_days=claims_halflife,
        release_index=claims_raw.index,
    )

    # ── Derived features ──────────────────────────────────────────────────────
    macro_df['Log_Jobless_Claims']     = np.log(macro_df['Jobless_Claims'].clip(lower=1))
    macro_df['Log_Jobless_Claims_Chg'] = macro_df['Log_Jobless_Claims'].diff(5)
    macro_df['ISM_PMI_Chg']            = macro_df['ISM_PMI'].diff(21)
    macro_df['SPY_LogRet']             = np.log(macro_df['SPY'] / macro_df['SPY'].shift(1))
    macro_df.dropna(inplace=True)

    print(
        f"  Dataset shape: {macro_df.shape} | "
        f"{macro_df.index.min().date()} -> {macro_df.index.max().date()}"
    )
    return macro_df
