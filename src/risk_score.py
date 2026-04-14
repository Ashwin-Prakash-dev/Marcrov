"""
Two risk scores are produced:

Risk_Score_Static
    Weighted sum of state posteriors using VIX-mean weights derived from
    the training period

Risk_Score  (recalibrated)
    At each OOS day t, state weights are recomputed from a rolling window
    of [t - CALIB_WINDOW, t - FWD_H] observations using Spearman rank
    correlations between each state's posterior and the forward SPY return.
    States that historically preceded negative forward returns receive higher
    risk weight

"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ══════════════════════════════════════════════════════════════════════════
# Static risk score
# ══════════════════════════════════════════════════════════════════════════

def calculate_static_risk_score(
    gamma_all: np.ndarray,
    vix_weights: np.ndarray,
) -> np.ndarray:
    """
    Dot-product risk score using fixed VIX-mean weights

    Parameters
    ----------
    gamma_all : np.ndarray, shape (T, K)
        Smoothed state posteriors for the full dataset.
    vix_weights : np.ndarray, shape (K,)
        Per-state weights proportional to mean VIX in training period.
        Must sum to 1 and be ordered low-risk → high-risk.

    Returns
    -------
    np.ndarray, shape (T,)
        Values in [0, 1].
    """
    score = gamma_all @ vix_weights
    return np.clip(score, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# Rolling Spearman-recalibrated risk score
# ══════════════════════════════════════════════════════════════════════════

def calculate_recalibrated_risk_score(
    gamma_all: np.ndarray,
    spy_logret: pd.Series,
    full_index: pd.DatetimeIndex,
    oos_start: pd.Timestamp,
    vix_weights: np.ndarray,
    fwd_h: int = 60,
    calib_window: int = 504,
    min_obs: int = 60,
) -> tuple:
    """
    Builds the rolling Spearman-recalibrated risk score for the full dataset.

    In-sample rows use the static VIX weights unchanged.  OOS rows use
    Spearman rank correlations from a rolling calibration window.

    Parameters
    ----------
    gamma_all : np.ndarray, shape (T, K)
        Full-dataset smoothed posteriors (train + OOS)
    spy_logret : pd.Series, shape (T,)
        Daily log return of SPY, indexed by full_index
    full_index : pd.DatetimeIndex
        Date index corresponding to gamma_all rows.
    oos_start : pd.Timestamp
        First OOS date (e.g. pd.Timestamp('2018-01-01'))
        Rows before this date use static weights.
    vix_weights : np.ndarray, shape (K,)
        Fallback weights (used in-sample and when window is too small)
    fwd_h : int
        Forward horizon in days for SPY return (default 60 -> around 3 months)
    calib_window : int
        Maximum lookback in days for calibration (default 504 -> around 2 years)
    min_obs : int
        Minimum non-NaN observations required to compute Spearman.
        Falls back to vix_weights if threshold not met

    Returns
    -------
    risk_score : np.ndarray, shape (T,)
        Recalibrated risk score, values in [0, 1]
    recalib_weights : np.ndarray, shape (T, K)
        Per-day state weights used to produce risk_score
    """
    T, K = gamma_all.shape

    # Forward SPY log-return over fwd_h days 
    spy_fwd = spy_logret.rolling(fwd_h).sum().shift(-fwd_h).values

    recalib_weights = np.full((T, K), np.nan)

    for i in range(T):
        t = full_index[i]
        if t < oos_start:
            recalib_weights[i] = vix_weights
            continue

        start_i = max(0, i - calib_window)
        end_i   = i - fwd_h   # last index included = i - fwd_h - 1 

        if end_i - start_i < min_obs:
            recalib_weights[i] = vix_weights
            continue

        w_idx   = slice(start_i, end_i)
        fwd_ret = spy_fwd[w_idx]
        valid   = ~np.isnan(fwd_ret)

        if valid.sum() < min_obs:
            recalib_weights[i] = vix_weights
            continue

        # For each state: Spearman p between posterior and forward return.
        # Higher posterior → more negative future SPY return = higher risk.
        # We negate p so that a strong negative correlation maps to a large positive risk weight
 
        rhos = np.zeros(K)
        for k in range(K):
            pk      = gamma_all[w_idx, k][valid]
            fr      = fwd_ret[valid]
            rho, _  = spearmanr(pk, fr)
            rhos[k] = rho if not np.isnan(rho) else 0.0

        risk_rhos  = -rhos                    # invert: risk high when SPY low
        risk_rhos -= risk_rhos.min()          # shift to non-negative
        total      = risk_rhos.sum()
        recalib_weights[i] = risk_rhos / total if total > 1e-9 else vix_weights

    risk_score = np.einsum('ij,ij->i', gamma_all, recalib_weights)
    risk_score = np.clip(risk_score, 0.0, 1.0)

    # to check if no NaNs in OOS window
    oos_mask = np.array([t >= oos_start for t in full_index])
    assert not np.isnan(risk_score[oos_mask]).any(), \
        "Risk_Score has unexpected NaNs in OOS window — check vix_weights fallback."

    return risk_score, recalib_weights


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def smooth_signal(signal: pd.Series, window: int = 10) -> pd.Series:
    """
    Applies a rolling mean to smooth out daily signal noise.

    Used in the backtest (run_backtest.py) before mapping the risk score
    to portfolio weights.

    Parameters
    ----------
    signal : pd.Series
    window : int
        Rolling window in trading days (default 10 ≈ 2 weeks).

    Returns
    -------
    pd.Series, same index as signal.
    """
    return signal.rolling(window, min_periods=1).mean()


def derive_vix_weights(
    gamma_train: np.ndarray,
    vix_train: np.ndarray,
    n_states: int,
) -> np.ndarray:
    """
    Computes per-state VIX-weighted risk scores from training posteriors.

    The weights are proportional to the posterior-weighted mean VIX for
    each state, normalised to sum to 1.  These are used as both the static
    risk score weights and as the fallback for the recalibrated score.

    Parameters
    ----------
    gamma_train : np.ndarray, shape (T_train, K)
        Training-period smoothed posteriors (already state-ordered).
    vix_train : np.ndarray, shape (T_train,)
        VIX values aligned to training rows.
    n_states : int

    Returns
    -------
    np.ndarray, shape (K,)
        Weights ordered low-risk (index 0) → high-risk (index K-1).
    """
    means = np.array([
        float((vix_train * gamma_train[:, k]).sum() /
              (gamma_train[:, k].sum() or 1e-300))
        for k in range(n_states)
    ])
    means /= means.sum()
    return means
