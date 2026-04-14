
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest import (
    apply_lazy_rebalance,
    calculate_strategy_returns,
    perf_metrics,
)

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

SIGNAL_FILE      = 'outputs/regime_signals.csv'
OOS_START        = '2018-01-01'

TC_BPS           = 5.0
REBAL_BUFFER     = 0.15
SMOOTH_WINDOW    = 10
DISCRETE_MODE    = False
MIN_HOLD_DAYS    = 20
USE_STATIC_SCORE = False    # True -> use Risk_Score_Static instead of Risk_Score

BG    = '#070b14'
PANEL = '#0c1120'
GRID  = '#182035'
TEXT  = '#aab8cc'

COLOR_MAP = {
    'HMM Apex v2 (No Lev + SHY)': '#00e5a0',
    'HMM Apex v1 (Lev + TLT)':    '#e74c3c',
    '60/40 B&H (SPY/TLT)':        '#4a9eff',
    'SPY B&H':                     '#aab8cc',
}


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Validate API key ────────
    api_key = os.environ.get('FRED_API_KEY', '')
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found. "
            "Add it to your .env file.\n"
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    # ── Load signals ────────────
    print("Loading regime signals...")
    signals = pd.read_csv(SIGNAL_FILE, parse_dates=['Date'], index_col='Date')
    signals.index = pd.to_datetime(signals.index).tz_localize(None)

    score_col = 'Risk_Score_Static' if USE_STATIC_SCORE else 'Risk_Score'
    if score_col not in signals.columns:
        raise ValueError(
            f"Column '{score_col}' not found in {SIGNAL_FILE}. "
            f"Run run_model.py first."
        )

    # ── Market data ──────────────
    print("Fetching market data (SPY, TLT, SHY)...")
    start = '2006-01-01'
    end   = signals.index.max().strftime('%Y-%m-%d')

    prices = yf.download(['SPY', 'TLT', 'SHY'], start=start, end=end, progress=False)['Close']
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # ── Risk-free rate from FRED ─────────────────
    print("Fetching TB3MS risk-free rate...")
    fred  = Fred(api_key=api_key)
    tb3ms = fred.get_series('TB3MS', observation_start=start, observation_end=end)
    tb3ms.index = pd.to_datetime(tb3ms.index).tz_localize(None)
    rf_daily = (tb3ms / 100 / 252).reindex(prices.index).ffill()

    # ── Align to OOS window ──────────────────────
    idx = prices.index.intersection(signals.index)
    idx = idx[idx >= OOS_START]

    spy = prices['SPY'].reindex(idx)
    tlt = prices['TLT'].reindex(idx)
    shy = prices['SHY'].reindex(idx)
    rs  = signals[score_col].reindex(idx).clip(0, 1)
    rf  = rf_daily.reindex(idx).fillna(rf_daily.mean())

    spy_ret = spy.pct_change().fillna(0)
    tlt_ret = tlt.pct_change().fillna(0)
    shy_ret = shy.pct_change().fillna(0)

    # ── Signal smoothing ──────────────────
    rs_smooth = rs.rolling(SMOOTH_WINDOW, min_periods=1).mean()

    print(f"\nRisk_Score stats (raw vs smoothed):")
    print(f"  Raw    — mean={rs.mean():.3f}  std={rs.std():.3f}  "
          f"daily_change_mean={rs.diff().abs().mean():.4f}")
    print(f"  Smooth — mean={rs_smooth.mean():.3f}  std={rs_smooth.std():.3f}  "
          f"daily_change_mean={rs_smooth.diff().abs().mean():.4f}")

    # ── Allocation engines ───────────────

    def continuous_allocation(rs_signal):
        t_spy = np.clip(0.95 - (0.55 * rs_signal), 0.40, 0.95)
        t_shy = np.clip(1.0 - t_spy, 0.05, 0.60)
        t_spy = apply_lazy_rebalance(t_spy, REBAL_BUFFER)
        t_shy = apply_lazy_rebalance(t_shy, REBAL_BUFFER)
        return t_spy, t_shy

    def discrete_allocation(rs_signal, min_hold=MIN_HOLD_DAYS):
        regime_map = {0: (0.90, 0.10), 1: (0.60, 0.40), 2: (0.30, 0.70)}
        raw_regime = pd.cut(
            rs_signal, bins=[-0.01, 0.33, 0.66, 1.01], labels=[0, 1, 2]
        ).astype(int)
        locked_regime = raw_regime.copy()
        last_switch, current = 0, raw_regime.iloc[0]
        for i in range(1, len(raw_regime)):
            if raw_regime.iloc[i] != current and (i - last_switch) >= min_hold:
                current     = raw_regime.iloc[i]
                last_switch = i
            locked_regime.iloc[i] = current
        t_spy = locked_regime.map({k: v[0] for k, v in regime_map.items()})
        t_shy = locked_regime.map({k: v[1] for k, v in regime_map.items()})
        return t_spy, t_shy

    if DISCRETE_MODE:
        apex_spy_w, apex_shy_w = discrete_allocation(rs_smooth)
        v2_label = 'HMM Apex v2 (Discrete+SHY)'
    else:
        apex_spy_w, apex_shy_w = continuous_allocation(rs_smooth)
        v2_label = 'HMM Apex v2 (No Lev + SHY)'

    # v1 reconstruction for comparison
    target_spy_v1 = np.clip(1.30 - (1.10 * rs), 0.20, 1.30)
    target_tlt_v1 = np.clip(1.0 - target_spy_v1, 0.0, 0.80)
    apex_spy_v1   = apply_lazy_rebalance(target_spy_v1, 0.05)
    apex_tlt_v1   = apply_lazy_rebalance(target_tlt_v1, 0.05)

    # 60/40 monthly rebalance
    month_first  = idx.to_series().groupby(idx.to_period('M')).first()
    is_rebal_day = pd.Series(idx.isin(month_first.values), index=idx)
    spy_w_6040   = pd.Series(np.nan, index=idx)
    tlt_w_6040   = pd.Series(np.nan, index=idx)
    spy_w_6040.iloc[0], tlt_w_6040.iloc[0] = 0.60, 0.40

    for i in range(1, len(idx)):
        if is_rebal_day.iloc[i]:
            spy_w_6040.iloc[i], tlt_w_6040.iloc[i] = 0.60, 0.40
        else:
            ps, pt   = spy_w_6040.iloc[i - 1], tlt_w_6040.iloc[i - 1]
            pc       = max(0.0, 1.0 - ps - pt)
            port_val = 1.0 + ps * spy_ret.iloc[i] + pt * tlt_ret.iloc[i] + pc * rf.iloc[i]
            spy_w_6040.iloc[i] = ps * (1 + spy_ret.iloc[i]) / port_val
            tlt_w_6040.iloc[i] = pt * (1 + tlt_ret.iloc[i]) / port_val

    # ── Returns with transaction costs ────────
    MARGIN_SPREAD = 0.015

    def strategy_returns_ext(sw, dw, def_ret, rf_series, name='', allow_leverage=False):
        sw     = sw.fillna(0)
        dw     = dw.fillna(0).clip(0, 1)
        cash_w = 1.0 - sw - dw

        if allow_leverage:
            borrow_rate = rf_series + (MARGIN_SPREAD / 252)
            cash_return = np.where(cash_w >= 0, cash_w * rf_series, cash_w * borrow_rate)
        else:
            cash_w      = cash_w.clip(0, 1)
            cash_return = cash_w * rf_series

        tc       = TC_BPS / 10_000
        gross    = sw * spy_ret + dw * def_ret + cash_return
        d_spy    = sw.diff().fillna(0)
        d_def    = dw.diff().fillna(0)
        total_tc = (d_spy.abs() + d_def.abs()) * tc
        net      = gross - total_tc
        turnover = (d_spy.abs() + d_def.abs()).mean() * 100
        tc_drag  = total_tc.sum() * 100
        print(f"  {name:<35} turnover={turnover:.2f}%/day   TC drag={tc_drag:.2f}%")
        return net

    print("\nComputing OOS strategy returns...")
    ret = pd.DataFrame(index=idx)
    ret[v2_label]                   = strategy_returns_ext(apex_spy_w, apex_shy_w, shy_ret, rf, v2_label)
    ret['HMM Apex v1 (Lev + TLT)'] = strategy_returns_ext(apex_spy_v1, apex_tlt_v1, tlt_ret, rf, 'HMM Apex v1 (Lev+TLT)', allow_leverage=True)
    ret['60/40 B&H (SPY/TLT)']     = strategy_returns_ext(spy_w_6040, tlt_w_6040, tlt_ret, rf, '60/40 B&H (SPY/TLT)')
    ret['SPY B&H']                  = spy_ret.reindex(idx).fillna(0)

    # ── Performance table ───────────────────────────
    print("\n" + "=" * 82)
    print(f"APEX v2 OOS PERFORMANCE SUMMARY  (2018-Present, 5bps TC)")
    print("=" * 82)
    print(f"{'Strategy':<35} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>8} "
          f"{'Max DD':>8} {'Calmar':>8}")
    print("-" * 82)

    for col in ret.columns:
        s = perf_metrics(ret[col], rf)
        print(
            f"{col:<35} "
            f"{s['Ann_Ret']*100:>7.1f}%  "
            f"{s['Ann_Vol']*100:>7.1f}%  "
            f"{s['Sharpe']:>8.2f}  "
            f"{s['Max_DD']*100:>7.1f}%  "
            f"{s['Calmar']:>8.2f}"
        )

    # ── Chart ─────────────────────────────────────────────────────────────────
    cum = (1 + ret).cumprod()
    fig = plt.figure(figsize=(20, 18), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.35, height_ratios=[2.5, 1.2, 1.0, 1.0])

    ax_cum = fig.add_subplot(gs[0])
    ax_dd  = fig.add_subplot(gs[1])
    ax_w   = fig.add_subplot(gs[2])
    ax_rs  = fig.add_subplot(gs[3])

    for ax in fig.get_axes():
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

    local_colors = dict(COLOR_MAP)
    local_colors[v2_label] = '#00e5a0'

    for col in ret.columns:
        lw = 2.5 if 'v2' in col else 1.5
        ls = '--' if ('v1' in col or 'B&H' in col) else '-'
        c  = local_colors.get(col, '#ffffff')
        ax_cum.plot(cum.index, cum[col], color=c, lw=lw, ls=ls, label=col)
        ax_cum.annotate(
            f'{cum[col].iloc[-1]:.2f}x',
            xy=(cum.index[-1], cum[col].iloc[-1]),
            xytext=(6, 0), textcoords='offset points',
            color=c, fontsize=9, va='center',
        )

    ax_cum.set_title('Apex v2 vs v1: OOS Cumulative Returns (2018-Present)',
                     fontsize=14, fontweight='bold', color='white', pad=10)
    ax_cum.legend(fontsize=10, facecolor='#0a0f1e', labelcolor=TEXT, loc='upper left')
    ax_cum.grid(True, alpha=0.07, color=GRID)
    ax_cum.set_ylabel('Wealth ($1 start)', color=TEXT)

    for col in ret.columns:
        c  = local_colors.get(col, '#ffffff')
        dd = (cum[col] - cum[col].cummax()) / cum[col].cummax()
        ax_dd.plot(dd.index, dd * 100, color=c, lw=2.0 if 'v2' in col else 1.0, label=col)
    ax_dd.set_title('Drawdown Comparison (%)', fontsize=11, color='white')
    ax_dd.grid(True, alpha=0.07, color=GRID)
    ax_dd.set_ylabel('Drawdown (%)', color=TEXT)
    ax_dd.legend(fontsize=8, facecolor='#0a0f1e', labelcolor=TEXT, loc='lower left')

    ax_w.fill_between(idx, 0, apex_spy_w, color='#00e5a0', alpha=0.75, label='SPY Weight')
    ax_w.fill_between(idx, apex_spy_w, apex_spy_w + apex_shy_w,
                      color='#f39c12', alpha=0.75, label='SHY Weight')
    ax_w.fill_between(idx, apex_spy_w + apex_shy_w, 1.0,
                      color='#aab8cc', alpha=0.4, label='Cash')
    ax_w.axhline(1.0, color='white', lw=0.8, ls='--')
    ax_w.set_title('v2 Allocation: SPY / SHY / Cash (15% Buffer, 10d Smooth)',
                   fontsize=11, color='white')
    ax_w.legend(fontsize=9, facecolor='#0a0f1e', labelcolor=TEXT, loc='lower left')
    ax_w.grid(True, alpha=0.07, color=GRID)
    ax_w.set_ylabel('Weight', color=TEXT)
    ax_w.set_ylim(0, 1.05)

    ax_rs.plot(idx, rs,        color='#e74c3c', lw=0.8, alpha=0.5, label='Raw Risk_Score')
    ax_rs.plot(idx, rs_smooth, color='#f39c12', lw=1.5, label=f'Smoothed ({SMOOTH_WINDOW}d)')
    ax_rs.axhline(0.33, color='white', lw=0.6, ls=':', alpha=0.5)
    ax_rs.axhline(0.66, color='white', lw=0.6, ls=':', alpha=0.5)
    ax_rs.set_title('HMM Risk Score: Raw vs Smoothed', fontsize=11, color='white')
    ax_rs.legend(fontsize=9, facecolor='#0a0f1e', labelcolor=TEXT)
    ax_rs.grid(True, alpha=0.07, color=GRID)
    ax_rs.set_ylabel('Risk Score', color=TEXT)
    ax_rs.set_ylim(-0.05, 1.05)

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/regime_backtest_apex_v2.png', dpi=155,
                bbox_inches='tight', facecolor=BG)
    print("\nSaved -> outputs/regime_backtest_apex_v2.png")


if __name__ == '__main__':
    main()
