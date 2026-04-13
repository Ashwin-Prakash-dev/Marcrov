import pandas as pd
import numpy as np
from src.backtest import apply_lazy_rebalance, calculate_strategy_returns

def test_apply_lazy_rebalance():
    """Tests if the 15% hysteresis buffer correctly kills micro-turnover."""
    # Create a target series that drifts slightly, then makes a big jump
    dates = pd.date_range("2024-01-01", periods=5)
    target = pd.Series([0.50, 0.52, 0.54, 0.70, 0.71], index=dates)
    
    buffer = 0.15
    executed = apply_lazy_rebalance(target, buffer)
    
    # Assertions
    assert executed.iloc[0] == 0.50, "Initial weight should match target"
    assert executed.iloc[1] == 0.50, "0.02 drift should be ignored by 0.15 buffer"
    assert executed.iloc[2] == 0.50, "0.04 drift should be ignored"
    assert executed.iloc[3] == 0.70, "0.20 jump exceeds buffer, should rebalance"
    assert executed.iloc[4] == 0.70, "0.01 drift after rebalance should be ignored"

def test_calculate_strategy_returns_costs():
    """Tests if the transaction cost drag is calculated correctly."""
    dates = pd.date_range("2024-01-01", periods=3)
    
    # 0% returns, so the only change in net should be the TC drag
    spy_ret = pd.Series([0.0, 0.0, 0.0], index=dates)
    def_ret = pd.Series([0.0, 0.0, 0.0], index=dates)
    rf = pd.Series([0.0, 0.0, 0.0], index=dates)
    
    # We change weights by 100% on day 2. 
    # 1.0 (SPY change) + 1.0 (DEF change) = 2.0 total turnover
    sw = pd.Series([1.0, 0.0, 0.0], index=dates)
    dw = pd.Series([0.0, 1.0, 1.0], index=dates)
    
    tc_bps = 5.0 # 5 bps = 0.0005
    
    net, turnover, tc_drag = calculate_strategy_returns(sw, dw, spy_ret, def_ret, rf, tc_bps)
    
    # Turnover on day 2 is 2.0. Mean over 3 days = 2.0 / 3 = 0.666 * 100 = 66.66%
    assert round(turnover, 2) == 66.67, "Turnover calculation is incorrect"
    
    # Total TC = 2.0 turnover * 0.0005 cost = 0.001 (0.10%)
    assert round(tc_drag, 2) == 0.10, "Transaction cost drag calculation is incorrect"