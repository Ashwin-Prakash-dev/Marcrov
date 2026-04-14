import pandas as pd
import numpy as np

def apply_lazy_rebalance(target_series: pd.Series, buffer: float) -> pd.Series:
    """
    Apply a hysteresis buffer to a target weight series to eliminate
    micro-turnover from small daily fluctuations in the risk signal.

    A rebalance only occurs when the target weight differs from the current
    executed weight by at least `buffer`. 

    Parameters
    ----------
    target_series : pd.Series
        Desired portfolio weight on each day.
    buffer : float
        Minimum absolute deviation required to trigger a rebalance.
        Calibrated to 0.15 (15%) for the Apex v2 strategy.

    Returns
    -------
    pd.Series
        Executed weight series (same index as target_series).
    """
    executed = pd.Series(index=target_series.index, dtype=float)
    current  = target_series.iloc[0]

    for i in range(len(target_series)):
        if abs(target_series.iloc[i] - current) >= buffer:
            current = target_series.iloc[i]
        executed.iloc[i] = current

    return executed


def calculate_strategy_returns(
    sw: pd.Series,
    dw: pd.Series,
    spy_ret: pd.Series,
    def_ret: pd.Series,
    rf: pd.Series,
    tc_bps: float,
) -> tuple:
    """
    Compute net daily returns after transaction costs.

    Parameters
    ----------
    sw : pd.Series
        Daily executed SPY weight (from apply_lazy_rebalance).
    dw : pd.Series
        Daily executed defensive leg weight (SHY in v2, TLT in v1).
    spy_ret : pd.Series
        SPY daily return.
    def_ret : pd.Series
        Defensive ETF daily return.
    rf : pd.Series
        Daily risk-free rate (TB3MS / 252).
    tc_bps : float
        One-way transaction cost in basis points (e.g. 5.0 = 0.05%).

    Returns
    -------
    net : pd.Series
        Net daily returns.
    turnover : float
        Mean daily two-way turnover expressed as a percentage.
    tc_drag : float
        Total transaction cost drag over the period, as a percentage.
    """
    sw      = sw.fillna(0)
    dw      = dw.fillna(0).clip(0, 1)
    cash_w  = (1.0 - sw - dw).clip(0, 1)   # no leverage

    gross   = sw * spy_ret + dw * def_ret + cash_w * rf

    tc_decimal = tc_bps / 10_000
    d_spy      = sw.diff().fillna(0)
    d_def      = dw.diff().fillna(0)
    total_tc   = (d_spy.abs() + d_def.abs()) * tc_decimal

    net       = gross - total_tc
    turnover  = (d_spy.abs() + d_def.abs()).mean() * 100
    tc_drag   = total_tc.sum() * 100

    return net, turnover, tc_drag


def perf_metrics(returns: pd.Series, rf_series: pd.Series) -> dict:
    """
    Compute standard risk-adjusted performance metrics.

    Parameters
    ----------
    returns : pd.Series
        Daily net return series.
    rf_series : pd.Series
        Daily risk-free rate (aligned to returns index).

    Returns
    -------
    dict with keys:
        Ann_Ret  — annualised arithmetic return
        Ann_Vol  — annualised volatility
        Sharpe   — annualised Sharpe ratio (excess return / vol)
        Max_DD   — maximum drawdown (negative number)
        Calmar   — Ann_Ret / |Max_DD|
    """
    ann    = returns.mean() * 252
    vol    = returns.std() * np.sqrt(252)
    sharpe = (ann - rf_series.mean() * 252) / vol
    cum    = (1 + returns).cumprod()
    mdd    = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann / abs(mdd)

    return {
        'Ann_Ret': ann,
        'Ann_Vol': vol,
        'Sharpe':  sharpe,
        'Max_DD':  mdd,
        'Calmar':  calmar,
    }
