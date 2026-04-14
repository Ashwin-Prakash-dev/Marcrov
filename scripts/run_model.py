"""
Pipeline
--------
1.  Fetch raw macro data (VIX, DXY, SPY, T10Y2Y, BAMLH0A0HYM2, ICSA) + PMI
2.  Engineer 12-feature matrix (6 z-scores + 6 momentum diffs)
3.  Train/OOS split at 2018-01-01 — scaler and PCA fitted on train ONLY
4.  BIC scan over 2–7 states, then override to 4 (domain prior)
5.  Fit 4-state HMM with sticky Dirichlet prior (κ=1000)
6.  Order states by posterior-weighted VIX mean (training period only)
7.  Compute Risk_Score_Static (VIX-weight dot product)
8.  Compute Risk_Score (rolling Spearman recalibration, OOS only)
9.  Export regime_signals.csv which is then used by run_backtest.py

Data leakage controls
---------------------
- StandardScaler.fit() and PCA.fit() called on X_train only
- State ordering uses gamma_train weights (not gamma_all)
- Recalibrated score uses [t - CALIB_WINDOW, t - FWD_H] window — the last
  included index's forward return is fully realised at decision time t
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline import fetch_macro_dataset
from src.features import (
    build_feature_matrix, fit_scaler_pca, apply_scaler_pca,
    print_pca_loadings, FEATURE_COLS,
)
from src.hmm import GaussianHMM
from src.risk_score import (
    calculate_static_risk_score,
    calculate_recalibrated_risk_score,
    derive_vix_weights,
)

# ─── CONFIGURATION ───────────────────────────
TRAIN_START     = '2006-01-01'
TRAIN_END       = '2017-12-31'
OOS_START       = pd.Timestamp('2018-01-01')
PMI_FILEPATH    = 'data/pmi.csv'
OUTPUT_FILE     = 'outputs/regime_signals.csv'
CHART_FILE      = 'outputs/regime_v3.png'

# HMM
FIXED_N_STATES  = 4            # Calm / Elevated / Stressed / Crisis
STICKY_KAPPA    = 1000.0       # strong persistence prior for daily data
COVAR_REG       = 5e-3         # ridge regularisation for covariance matrices
PCA_VARIANCE    = 0.95         # retain 95% of variance

# Risk score recalibration
FWD_H           = 60           # forward SPY return horizon (days)
CALIB_WINDOW    = 504          # rolling calibration lookback (days ≈ 2 years)

# Plotting
PALETTE = ['#26a65b', '#f1c40f', '#e67e22', '#e74c3c',
           '#9b59b6', '#1abc9c', '#e91e63']

_STATE_NAME_MAP = {
    4: ['Calm', 'Elevated', 'Stressed', 'Crisis'],
    3: ['Low-Risk', 'Mid-Risk', 'High-Risk'],
    2: ['Low-Risk', 'High-Risk'],
}

assert FIXED_N_STATES in _STATE_NAME_MAP, (
    f"Add state name list for FIXED_N_STATES={FIXED_N_STATES} "
    f"to _STATE_NAME_MAP (currently covers: {list(_STATE_NAME_MAP)})"
)

BG, PANEL, GRID, TEXT = '#070b14', '#0c1120', '#182035', '#aab8cc'


# ─── HELPER ───────────────────────────────────────────────────────────────────

def fmt_date(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    raw = fetch_macro_dataset(
        start_date=TRAIN_START,
        pmi_filepath=PMI_FILEPATH,
    )

    # ── 2. Feature engineering ────────────────────────────────────────────────
    df = build_feature_matrix(raw)

    # ── 3. Train / OOS split ──────────────────────────────────────────────────
    train_df = df[df.index <  OOS_START]
    test_df  = df[df.index >= OOS_START]

    print(f"Train: {len(train_df)} days  |  OOS: {len(test_df)} days")

    X_train = train_df[FEATURE_COLS].values
    X_test  = test_df [FEATURE_COLS].values
    X_all   = df      [FEATURE_COLS].values

    # Scaler and PCA fitted only on training data
    scaler, pca, Xtr_p = fit_scaler_pca(X_train, n_components=PCA_VARIANCE)
    Xte_p = apply_scaler_pca(X_test, scaler, pca)
    Xal_p = apply_scaler_pca(X_all,  scaler, pca)

    print_pca_loadings(pca, FEATURE_COLS)
    n_pca = pca.n_components_

    # ── 4. BIC scan (reference only — domain override applied) ────────────────
    print(f"\nBIC reference scan (train {TRAIN_START}–{TRAIN_END}, "
          f"sticky_kappa={STICKY_KAPPA})...")
    print(f"  (Model will use FIXED_N_STATES={FIXED_N_STATES} regardless of BIC)")

    bic_scores, models = {}, {}
    for n in range(2, 8):
        print(f"  {n} states...", end=' ', flush=True)
        m = GaussianHMM(
            n_states=n, n_iter=300, random_state=42,
            covar_reg=COVAR_REG, sticky_kappa=STICKY_KAPPA,
        )
        m.fit(Xtr_p)
        bic_scores[n] = m.bic(Xtr_p)
        models[n] = m
        print(f"BIC={bic_scores[n]:.1f}")

    best_n_bic = min(bic_scores, key=bic_scores.get)
    print(f"\n  BIC would select {best_n_bic} states — "
          f"overridden to {FIXED_N_STATES} (domain prior)")
    print("  BIC scores: " +
          "  ".join([f"{n}st={bic_scores[n]:.0f}" for n in sorted(bic_scores)]))

    model = models[FIXED_N_STATES]
    del models

    model.log_transition_diagnostics()

    # ── 5. Posteriors ─────────────────────────────────────────────────────────
    gamma_train = model.predict_proba(Xtr_p)
    gamma_test  = model.predict_proba(Xte_p)
    gamma_all   = model.predict_proba(Xal_p)

    # ── 6. State ordering ─
    vix_train = train_df['Z_VIX'].values   

    # Sort states by weighted-mean VIX (low → high risk) using train weights
    state_order = model.sort_states_by_feature_mean(gamma_train, vix_train)

    # Reorder gamma matrices so column 0 = Calm, column K-1 = Crisis
    gamma_train = gamma_train[:, state_order]
    gamma_test  = gamma_test [:, state_order]
    gamma_all   = gamma_all  [:, state_order]

    # Build display metadata
    train_stats = {}
    for rank, s in enumerate(range(FIXED_N_STATES)):
        w = gamma_train[:, s]
        train_stats[s] = {
            'VIX_mean':    float((train_df['Z_VIX'].values      * w).sum() / (w.sum() or 1e-300)),
            'CS_mean':     float((train_df['Z_CS'].values        * w).sum() / (w.sum() or 1e-300)),
            'ISM_mean':    float((train_df['Z_ISM'].values       * w).sum() / (w.sum() or 1e-300)),
            'Claims_mean': float((train_df['Z_Claims'].values    * w).sum() / (w.sum() or 1e-300)),
        }

    STATE_COLORS = {k: PALETTE[k] for k in range(FIXED_N_STATES)}
    STATE_LABELS = {}
    for k in range(FIXED_N_STATES):
        name = _STATE_NAME_MAP[FIXED_N_STATES][k]
        s    = train_stats[k]
        STATE_LABELS[k] = (
            f'State {k+1}: {name}  '
            f"(μVIX={s['VIX_mean']:.1f}, μCS={s['CS_mean']:.1f}, "
            f"μISM={s['ISM_mean']:.1f})"
        )

    print("\nState characteristics (train period):")
    for k in range(FIXED_N_STATES):
        print(f"  {STATE_LABELS[k]}")

    # Viterbi for run-length diagnostics
    viterbi_train = model.predict(Xtr_p)
    # Remap viterbi states to sorted order
    remap = {old: new for new, old in enumerate(state_order)}
    viterbi_train = np.vectorize(remap.get)(viterbi_train)

    for k in range(FIXED_N_STATES):
        runs, cnt = [], 0
        for v in viterbi_train:
            if v == k:
                cnt += 1
            elif cnt:
                runs.append(cnt)
                cnt = 0
        if cnt:
            runs.append(cnt)
        med = np.median(runs) if runs else 0
        lbl = STATE_LABELS[k].split(':')[1].strip().split('(')[0].strip()
        print(f"  {lbl}: median run = {med:.0f} days  (n_episodes={len(runs)})")

    # ── 7. Static risk score ──────────────────────────────────────────────────
    vix_weights = derive_vix_weights(
        gamma_train,
        train_df['Z_VIX'].values,
        FIXED_N_STATES,
    )
    risk_static = calculate_static_risk_score(gamma_all, vix_weights)

    # ── 8. Recalibrated risk score ────────────────────────────────────────────
    print(f"\nBuilding recalibrated risk score "
          f"(FWD_H={FWD_H}d, CALIB_WINDOW={CALIB_WINDOW}d)...")

    risk_recalib, recalib_weights = calculate_recalibrated_risk_score(
        gamma_all     = gamma_all,
        spy_logret    = raw['SPY_LogRet'].reindex(df.index),
        full_index    = df.index,
        oos_start     = OOS_START,
        vix_weights   = vix_weights,
        fwd_h         = FWD_H,
        calib_window  = CALIB_WINDOW,
    )

    oos_mask = df.index >= OOS_START
    print(
        f"  OOS risk score range: "
        f"{risk_recalib[oos_mask].min():.3f} – "
        f"{risk_recalib[oos_mask].max():.3f}"
    )

    # Dominant state (Viterbi on full dataset)
    viterbi_all = model.predict(Xal_p)
    viterbi_all = np.vectorize(remap.get)(viterbi_all)

    # ── 9. Attach to df and export ────────────────────────────────────────────
    for k in range(FIXED_N_STATES):
        df[f'P{k}'] = gamma_all[:, k]

    df['Dom_State']         = viterbi_all
    df['Risk_Score']        = risk_recalib
    df['Risk_Score_Static'] = risk_static

    export = df[['Risk_Score', 'Risk_Score_Static', 'Dom_State']].copy()
    export.index.name = 'Date'
    export.to_csv(OUTPUT_FILE)
    print(f"\nSuccess! Regime signals saved → {OUTPUT_FILE}")

    # ── 10. Chart ─────────────────────────────────────────────────────────────
    print("Generating regime chart...")
    _plot_regimes(
        df, raw, train_df, test_df,
        STATE_LABELS, STATE_COLORS, FIXED_N_STATES,
        bic_scores, best_n_bic,
        n_pca, risk_recalib, risk_static,
        oos_mask,
    )
    print(f"Chart saved → {CHART_FILE}")


# ─── PLOTTING (self-contained) ────────────────────────────────────────────────

def _plot_regimes(
    df, raw, train_df, test_df,
    STATE_LABELS, STATE_COLORS, K,
    bic_scores, best_n_bic,
    n_pca, risk_recalib, risk_static,
    oos_mask,
):
    fig = plt.figure(figsize=(24, 18), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.4, wspace=0.3,
        height_ratios=[2.0, 1.5, 1.4],
    )

    ax_main = fig.add_subplot(gs[0, :])
    ax_prob = fig.add_subplot(gs[1, :])
    ax_rs   = fig.add_subplot(gs[2, 0])
    ax_bic  = fig.add_subplot(gs[2, 1])

    for ax in fig.get_axes():
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

    # Panel 1: VIX with regime shading
    vix = raw['VIX'].reindex(df.index)
    dom = df['Dom_State']
    for s in range(K):
        mask = dom == s
        blk  = (mask != mask.shift(1)).cumsum()
        for _, block in df[mask].groupby(blk[mask]):
            ax_main.axvspan(block.index.min(), block.index.max(),
                            color=STATE_COLORS[s], alpha=0.20, linewidth=0)
    ax_main.plot(df.index, vix, color='#dce3ec', lw=1.0, zorder=3)
    ax_main.axvline(pd.Timestamp('2018-01-01'), color='white', lw=1.3, ls=':', alpha=0.55)

    for lbl, dt in {
        'Lehman\n2008': '2008-09-15', 'EU Crisis\n2011': '2011-08-05',
        'COVID\n2020': '2020-03-23',  'Fed Hikes\n2022': '2022-03-16',
    }.items():
        x = pd.Timestamp(dt)
        ax_main.axvline(x, color='white', lw=0.6, ls='--', alpha=0.25)
        ax_main.text(x, vix.max() * 1.01, lbl, color='#667799', fontsize=7,
                     ha='center', va='bottom', fontfamily='monospace')

    legend_items = [plt.Line2D([0], [0], color='#dce3ec', lw=1.2, label='VIX')]
    for s in range(K):
        legend_items.append(mpatches.Patch(
            color=STATE_COLORS[s], alpha=0.65, label=STATE_LABELS[s]
        ))
    ax_main.legend(handles=legend_items, loc='upper right', fontsize=8.5,
                   facecolor='#0a0f1e', labelcolor=TEXT, framealpha=0.93)
    ax_main.set_ylabel('VIX', color=TEXT, fontsize=10)
    ax_main.set_title(
        f'Unsupervised Gaussian HMM  ·  {K}-State (Domain Override)  ·  '
        f'PCA({n_pca})+Sticky(κ={STICKY_KAPPA:.0f})  ·  Hard OOS Split 2018',
        fontsize=13, fontweight='bold', color='white', pad=14, fontfamily='monospace',
    )
    ax_main.grid(True, alpha=0.07, color=GRID)
    fmt_date(ax_main)

    # Panel 2: Stacked posterior ribbon
    base = np.zeros(len(df))
    for s in range(K):
        vals = df[f'P{s}'].values
        ax_prob.fill_between(df.index, base, base + vals,
                             color=STATE_COLORS[s], alpha=0.82, linewidth=0)
        base += vals
    ax_prob.axvline(pd.Timestamp('2018-01-01'), color='white', lw=1.3, ls=':', alpha=0.55)
    ax_prob.set_ylabel('P(State | data)', color=TEXT, fontsize=10)
    ax_prob.set_ylim(0, 1)
    ax_prob.set_title('Continuous Posterior Probability Ribbon  (no hard labels)',
                      fontsize=10, color=TEXT, pad=6)
    ax_prob.grid(True, alpha=0.07, color=GRID)
    fmt_date(ax_prob)
    legend2 = [mpatches.Patch(color=STATE_COLORS[s], alpha=0.75,
                               label=STATE_LABELS[s].split('(')[0].strip())
               for s in range(K)]
    ax_prob.legend(handles=legend2, loc='upper left', fontsize=8, ncol=K,
                   facecolor='#0a0f1e', labelcolor=TEXT, framealpha=0.93)

    # Panel 3: Risk score (OOS only)
    oos_idx = df.index[oos_mask]
    ax_rs.plot(oos_idx, risk_static[oos_mask], color='#4a9eff', lw=0.8,
               alpha=0.40, ls='--', label='Static (VIX-weighted)')
    ax_rs.fill_between(oos_idx, risk_recalib[oos_mask],
                       color='#f1c40f', alpha=0.18, linewidth=0)
    ax_rs.plot(oos_idx, risk_recalib[oos_mask], color='#f1c40f', lw=1.2,
               label='Recalibrated (rolling Spearman)')
    ax_rs.legend(fontsize=8, facecolor='#0a0f1e', labelcolor=TEXT,
                 framealpha=0.9, loc='upper left')
    ax_rs.set_ylabel('Risk Score', color=TEXT, fontsize=9)
    ax_rs.set_title('Risk Score — OOS 2018+  |  Recalibrated vs Static',
                    fontsize=9, color=TEXT, pad=6)
    ax_rs.grid(True, alpha=0.07, color=GRID)
    ax_rs.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax_rs.xaxis.set_major_locator(mdates.YearLocator())

    # Panel 4: BIC curve
    ns   = list(bic_scores.keys())
    bics = [bic_scores[n] for n in ns]
    ax_bic.plot(ns, bics, 'o-', color='#4a9eff', markersize=6, lw=1.8)
    ax_bic.axvline(FIXED_N_STATES, color='#e74c3c', lw=1.5, ls='--', alpha=0.8,
                   label=f'Domain override: {FIXED_N_STATES} states')
    ax_bic.axvline(best_n_bic, color='#f1c40f', lw=1.0, ls=':', alpha=0.7,
                   label=f'BIC min: {best_n_bic} states')
    ax_bic.set_xlabel('States', color=TEXT, fontsize=9)
    ax_bic.set_ylabel('BIC', color=TEXT, fontsize=9)
    ax_bic.set_title('BIC State Selection\n(train set, corrected)', fontsize=9,
                     color=TEXT, pad=6)
    ax_bic.legend(fontsize=8, facecolor='#0a0f1e', labelcolor=TEXT, framealpha=0.9)
    ax_bic.grid(True, alpha=0.07, color=GRID)
    ax_bic.set_xticks(ns)

    plt.savefig(CHART_FILE, dpi=155, bbox_inches='tight', facecolor=BG)
    plt.close()


if __name__ == '__main__':
    main()
