import pandas as pd
import numpy as np

def apply_lazy_rebalance(target_series: pd.Series, buffer: float) -> pd.Series:
    """Applies a hysteresis buffer to eliminate micro-turnover."""
    executed = pd.Series(index=target_series.index, dtype=float)
    current = target_series.iloc[0]
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
    tc_bps: float
) -> tuple[pd.Series, float, float]:
    """Calculates net returns minus transaction costs."""
    sw = sw.fillna(0)
    dw = dw.fillna(0).clip(0, 1)
    cash_w = (1.0 - sw - dw).clip(0, 1)
    
    gross = (sw * spy_ret) + (dw * def_ret) + (cash_w * rf)
    
    tc_decimal = tc_bps / 10_000
    d_spy, d_def = sw.diff().fillna(0), dw.diff().fillna(0)
    total_tc = (d_spy.abs() + d_def.abs()) * tc_decimal
    
    net = gross - total_tc
    turnover = (d_spy.abs() + d_def.abs()).mean() * 100
    tc_drag = total_tc.sum() * 100
    
    return net, turnover, tc_drag

def perf_metrics(returns: pd.Series, rf_series: pd.Series) -> dict:
    """Calculates standard risk-adjusted metrics."""
    ann = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = (ann - rf_series.mean() * 252) / vol
    cum = (1 + returns).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann / abs(mdd)
    return {'Ann_Ret': ann, 'Ann_Vol': vol, 'Sharpe': sharpe, 'Max_DD': mdd, 'Calmar': calmar}