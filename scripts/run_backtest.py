import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add root directory to path to allow src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline import fetch_market_data
from src.backtest import apply_lazy_rebalance, calculate_strategy_returns, perf_metrics

# CONFIG
SIGNAL_FILE   = 'outputs/regime_signals.csv'
OOS_START     = '2018-01-01'
TC_BPS        = 5.0
REBAL_BUFFER  = 0.15
SMOOTH_WINDOW = 10

def main():
    print("Loading signals and fetching market data...")
    signals = pd.read_csv(SIGNAL_FILE, parse_dates=['Date'], index_col='Date')
    prices, rf_daily = fetch_market_data('2006-01-01', signals.index.max().strftime('%Y-%m-%d'))

    idx = prices.index.intersection(signals.index)
    idx = idx[idx >= OOS_START]

    spy, shy = prices['SPY'].reindex(idx), prices['SHY'].reindex(idx)
    rs_raw = signals['Risk_Score'].reindex(idx).clip(0, 1)
    rf = rf_daily.reindex(idx).fillna(rf_daily.mean())

    spy_ret, shy_ret = spy.pct_change().fillna(0), shy.pct_change().fillna(0)

    print(f"Applying Apex v2 logic (Smooth: {SMOOTH_WINDOW}d, Buffer: {REBAL_BUFFER})...")
    rs_smooth = rs_raw.rolling(SMOOTH_WINDOW, min_periods=1).mean()
    
    t_spy = np.clip(0.95 - (0.55 * rs_smooth), 0.40, 0.95)
    t_shy = np.clip(1.0 - t_spy, 0.05, 0.60)

    exec_spy = apply_lazy_rebalance(t_spy, REBAL_BUFFER)
    exec_shy = apply_lazy_rebalance(t_shy, REBAL_BUFFER)

    net_ret, turnover, tc_drag = calculate_strategy_returns(exec_spy, exec_shy, spy_ret, shy_ret, rf, TC_BPS)
    spy_bh = spy_ret.reindex(idx).fillna(0)

    print("\n" + "="*60)
    print("APEX v2 OOS PERFORMANCE")
    print("="*60)
    print(f"Turnover: {turnover:.2f}%/day | TC Drag: {tc_drag:.2f}%\n")
    
    v2_stats = perf_metrics(net_ret, rf)
    spy_stats = perf_metrics(spy_bh, rf)

    print(f"{'Metric':<15} {'Apex v2':<15} {'SPY B&H':<15}")
    print("-" * 45)
    for k in v2_stats.keys():
        v2_val = f"{v2_stats[k]*100:.1f}%" if 'Ret' in k or 'Vol' in k or 'DD' in k else f"{v2_stats[k]:.2f}"
        spy_val = f"{spy_stats[k]*100:.1f}%" if 'Ret' in k or 'Vol' in k or 'DD' in k else f"{spy_stats[k]:.2f}"
        print(f"{k:<15} {v2_val:<15} {spy_val:<15}")

if __name__ == "__main__":
    main()