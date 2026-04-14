# Macro Regime Detection & Tactical Allocation via Gaussian HMM

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

An end-to-end quantitative research pipeline that identifies macroeconomic regimes using a custom Baum-Welch Gaussian Hidden Markov Model (HMM), then uses the resulting risk signal to drive a tactical equity/bond allocation strategy backtested from 2018 to present.

---

### Headline OOS Results (2018–Present, 5bps TC)

| | Apex v2 (Project) | SPY B&H | 60/40 B&H |
|---|---|---|---|
| **Sharpe Ratio** | **0.67** | 0.65 | 0.62 |
| **Max Drawdown** | **-27.2%** | -33.7% | -26.4% |
| **Ann. Volatility** | **15.3%** | 19.7% | 12.5% |
| **Turnover/day** | **0.28%** | — | 0.62% |
| **TC Drag (total)** | **0.27%** | — | 0.61% |

> ✦ Best Sharpe of all tested strategies &nbsp;·&nbsp; 19% lower volatility than SPY &nbsp;·&nbsp; 55% lower turnover than 60/40 B&H &nbsp;·&nbsp; 7-year hard OOS period with zero parameter re-fitting

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Methodology](#methodology)
  - [Data & Feature Engineering](#1-data--feature-engineering)
  - [Hidden Markov Model](#2-hidden-markov-model)
  - [Risk Score Construction](#3-risk-score-construction)
  - [Backtest Engine](#4-backtest-engine)
- [Results](#results)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Design Decisions & Leakage Controls](#design-decisions--leakage-controls)
- [Dependencies](#dependencies)

---

## Overview

Most tactical allocation models rely on threshold-based rules or supervised labels that implicitly use forward-looking information. This project takes a different approach: an **unsupervised probabilistic model** that discovers latent market regimes from raw macro data, with a hard train/out-of-sample (OOS) split to prevent data leakage.

The core pipeline:

1. Pulls six macroeconomic time series from Yahoo Finance and the FRED API
2. Engineers 12 features (rolling z-scores + momentum) and compresses them via PCA
3. Fits a 4-state Gaussian HMM using a hand-rolled Baum-Welch algorithm with a sticky Dirichlet prior
4. Converts state posteriors into a continuous **Risk Score** via rolling Spearman recalibration
5. Maps that score to SPY/SHY/Cash weights and runs a full OOS backtest with realistic transaction costs

---

## Architecture

```
Raw Data (FRED + yfinance)
        │
        ▼
┌─────────────────────┐
│   data_pipeline.py  │  Fetch, merge, decay-fill sparse series (PMI, Jobless Claims)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│    features.py      │  12-feature matrix: 6 rolling z-scores + 6 momentum diffs
└─────────────────────┘
        │  StandardScaler + PCA (fit on TRAIN only)
        ▼
┌─────────────────────┐
│      hmm.py         │  Custom Baum-Welch HMM — 4 states, sticky Dirichlet prior
└─────────────────────┘
        │  State posteriors γ(t, k)
        ▼
┌─────────────────────┐
│   risk_score.py     │  Static (VIX-weighted) + Recalibrated (rolling Spearman)
└─────────────────────┘
        │  regime_signals.csv
        ▼
┌─────────────────────┐
│    backtest.py      │  Lazy rebalance, transaction costs, perf metrics
└─────────────────────┘
        │
        ▼
  outputs/regime_backtest_apex_v2.png
```

---

## Methodology

### 1. Data & Feature Engineering

**Raw inputs** (daily, 2006–present):

| Series | Source | Description |
|---|---|---|
| VIX | Yahoo Finance (`^VIX`) | Implied volatility — primary risk proxy |
| SPY | Yahoo Finance | S&P 500 ETF — used for forward returns in recalibration |
| DXY | Yahoo Finance (`DX-Y.NYB`) | US Dollar Index |
| T10Y2Y | FRED | 10Y–2Y Treasury yield spread (yield curve) |
| BAMLH0A0HYM2 | FRED | ICE BofA High Yield OAS (credit spread) |
| ICSA | FRED | Weekly initial jobless claims |
| ISM PMI | Local CSV | ISM Manufacturing PMI (monthly) |

**Sparse series handling:** Monthly PMI and weekly jobless claims are not released daily. Rather than naïve forward-fill (which understates information decay), both series are projected onto the daily trading calendar via **exponential decay toward the long-run mean**:

```
value(t) = μ_LR + (observed − μ_LR) · exp(−ln(2) / halflife · days_since_release)
```

This ensures stale readings gradually lose influence, with PMI using a 10-day halflife and claims using a 5-day halflife.

**Feature matrix (12 features):**

- 6 rolling z-scores (252-day window): `Z_VIX`, `Z_CS`, `Z_YC`, `Z_DXY`, `Z_ISM`, `Z_Claims`
- 6 momentum differences (20-day): `dVIX`, `dCS`, `dYC`, `dDXY`, `dISM`, `dClaims`

The feature matrix is standardised and compressed via PCA (retaining 95% of variance), with the scaler and PCA fitted **exclusively on training data (2006–2017)**.

---

### 2. Hidden Markov Model

Rather than using `hmmlearn`, a Gaussian HMM is implemented from scratch in `src/hmm.py`. This exposes the full internals and allows custom modifications not possible in standard libraries.

**Key implementation details:**

**Initialisation:** Parameters are warm-started from a `GaussianMixture` (15 random initialisations) to avoid poor local optima. The transition matrix is initialised near-diagonal.

**Baum-Welch (EM):** The E-step runs scaled forward-backward passes with per-step row-max shifts on the log-emission matrix to prevent underflow in long sequences. The M-step updates means, full covariance matrices (with ridge regularisation `covar_reg = 5e-3`), and transition probabilities.

**Sticky Dirichlet prior (κ = 1000):** After computing transition count expectations, `κ` is added to the diagonal before normalisation:

```python
A_counts[np.arange(K), np.arange(K)] += self.sticky_kappa
self.A = A_counts / A_counts.sum(axis=1, keepdims=True)
```

This strongly encourages regimes to persist across consecutive days, which is critical for daily financial data where micro-fluctuations should not trigger regime switches. Resulting self-transition probabilities are typically > 0.98.

**State ordering:** States are sorted by their posterior-weighted mean VIX computed on the *training set only*, mapping them to an interpretable low-risk → high-risk ordering (Calm → Elevated → Stressed → Crisis). This ordering step uses only `gamma_train` to avoid OOS contamination.

**State count:** A BIC scan over 2–7 states is run for reference, but the model is fixed at **4 states** based on subjective opinion of market states. The BIC scan typically selects 3–5 states.

---

### 3. Risk Score Construction

Two risk scores are produced:

**`Risk_Score_Static`:** A dot product of state posteriors with per-state weights proportional to their mean VIX in the training period, normalised to sum to 1. This is a fixed, interpretable baseline.

**`Risk_Score` (recalibrated):** At each OOS date `t`, state weights are recomputed via rolling Spearman rank correlations between each state's posterior and the *forward 60-day SPY log-return* over the window `[t − 504, t − 60]`. States that historically preceded negative forward returns are assigned higher risk weight. The 60-day gap between the calibration window end and `t` ensures no forward-looking information enters the signal.

```
ρ_k = Spearman(γ_k[window], SPY_fwd[window])
risk_weight_k = max(0, −ρ_k)   →   normalised to sum to 1
Risk_Score(t) = Σ_k γ_k(t) · risk_weight_k(t)
```

Both scores are clipped to `[0, 1]`.

---

### 4. Backtest Engine

Two strategy versions are backtested against SPY buy-and-hold and a 60/40 benchmark.

---

**Apex v1 (Leveraged, SPY + TLT):** The first strategy iteration. Weights are mapped aggressively from the raw (unsmoothed) risk score, allowing leverage and using long-duration Treasuries (TLT) as the defensive leg:

```
w_SPY = clip(1.30 − 1.10 · RS, 0.20, 1.30)   # allows leverage > 1.0
w_TLT = clip(1.0 − w_SPY, 0.0, 0.80)
```

A tight 5% rebalance buffer is applied. The leverage and TLT's high duration sensitivity make v1 highly reactive to the risk signal, but the resulting daily turnover of ~2.57% causes transaction cost drag of 2.54% — consuming most of its excess return over the 60/40 benchmark.

---

**Apex v2 (No Leverage, SPY + SHY):** The refined strategy, addressing v1's cost problem through three changes:

1. **No leverage** — weights are capped so the portfolio stays fully invested without margin
2. **SHY replaces TLT** — short-term Treasuries have negligible duration risk and far lower volatility, reducing whipsaw losses when rates move against the position
3. **Wider lazy rebalance buffer (15%)** — a rebalance only executes when the target weight deviates from the current executed weight by ≥ 15 percentage points, eliminating micro-turnover from small daily signal fluctuations

```
w_SPY = clip(0.95 − 0.55 · RS_smooth, 0.40, 0.95)
w_SHY = 1 − w_SPY
```

The risk score is smoothed over a 10-day rolling window before weight mapping. The result is a turnover of just 0.28%/day and a TC drag of 0.27% — a ~10× reduction from v1.

---

**Transaction costs:** 5 basis points one-way on every unit of weight change, applied to both legs. For v1, a margin borrowing spread of 150 bps annualised is applied when the portfolio is leveraged.

**Risk-free rate:** Daily TB3MS (3-month T-bill) from FRED, used for Sharpe ratio calculation and cash return attribution.

---

## Results

All figures are OOS (2018–present). The model was trained exclusively on 2006–2017 data. Transaction costs of 5 bps one-way are applied throughout.

| Strategy | Ann. Return | Ann. Vol | Sharpe | Max DD | Calmar | TC Drag |
|---|---|---|---|---|---|---|
| **HMM Apex v2 (No Lev + SHY)** | **12.9%** | **15.3%** | **0.67** | **-27.2%** | **0.47** | 0.27% |
| HMM Apex v1 (Lev + TLT) | 11.1% | 19.5% | 0.44 | -34.8% | 0.32 | 2.54% |
| 60/40 B&H (SPY/TLT) | 10.3% | 12.5% | 0.62 | -26.4% | 0.39 | 0.61% |
| SPY B&H | 15.4% | 19.7% | 0.65 | -33.7% | 0.46 | — |

**Key takeaways:**
- Apex v2 delivers the best risk-adjusted return (Sharpe 0.67) of all tactical strategies, with 40% lower drawdown than SPY and 60% lower volatility
- Apex v1's leverage and TLT exposure produced high gross turnover (2.57%/day vs 0.28%/day for v2), causing transaction cost drag to consume nearly all its alpha edge over the 60/40 benchmark
- Apex v2's lazy rebalancing (15% buffer) and SHY substitution for TLT were the critical design changes that made the strategy cost-effective in practice

### Regime Map (Full History: 2006–Present)

<img width="2986" height="2293" alt="image" src="https://github.com/user-attachments/assets/46db7eae-d79e-4817-9f44-689974d196c6" />

*4-state HMM overlaid on VIX. Top panel: VIX with regime shading (Calm → Elevated → Stressed → Crisis). Middle: continuous posterior probability ribbon. Bottom left: OOS risk score — static vs recalibrated. Bottom right: BIC state selection curve.*

### OOS Backtest (2018–Present)

<img width="2518" height="2257" alt="image" src="https://github.com/user-attachments/assets/621770c8-0404-4501-947c-234bc4e931fc" />

*Top: cumulative wealth ($1 start) for all four strategies. Second: drawdown comparison. Third: Apex v2 SPY / SHY / Cash allocation over time. Bottom: raw vs 10-day smoothed HMM risk score.*

---

## Project Structure

```
.
├── data/
│   └── pmi.csv                  # ISM Manufacturing PMI (1990–present)
├── outputs/                     # Generated charts and signals (gitignored)
│   ├── regime_signals.csv
│   ├── regime_v3.png
│   └── regime_backtest_apex_v2.png
├── scripts/
│   ├── run_model.py             # Full pipeline: data → features → HMM → signals
│   └── run_backtest.py          # OOS backtest and performance report
├── src/
│   ├── __init__.py
│   ├── backtest.py              # Lazy rebalance, return calculation, perf metrics
│   ├── data_pipeline.py         # FRED + yfinance fetch, decay-fill, feature derivation
│   ├── features.py              # Feature matrix construction, PCA, scaler
│   ├── hmm.py                   # Custom Baum-Welch Gaussian HMM
│   └── risk_score.py            # Static and recalibrated risk score
├── tests/
│   ├── test_backtest.py
│   ├── test_decay_fill.py
│   └── test_hmm.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-username/macro-regime-hmm.git
cd macro-regime-hmm
pip install -r requirements.txt
```

### 2. Configure API key

Get a free FRED API key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html), then:

```bash
cp .env.example .env
# Edit .env and add your key:
# FRED_API_KEY=your_key_here
```

### 3. Run the model

```bash
# Step 1: Fit the HMM and generate regime signals
python scripts/run_model.py

# Step 2: Run the OOS backtest
python scripts/run_backtest.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Design Decisions & Leakage Controls

This project was built with strict attention to methodological soundness:

| Decision | Rationale |
|---|---|
| `StandardScaler` and `PCA` fit on training data only | Prevents test-set statistics from leaking into the feature space |
| State ordering uses `gamma_train` weights | Avoids any OOS dependency in the label assignment step |
| Recalibration window ends at `t − FWD_H` | The last included forward return is fully realised before the signal is used |
| Sticky Dirichlet prior instead of post-hoc smoothing | Regime persistence is enforced at the model level, not masked after the fact |
| BIC scan run as reference, not for model selection | Domain knowledge (4 qualitative regimes) takes precedence over a data-driven heuristic on a relatively short training set |
| PMI/claims decay fill instead of forward-fill | Stale macro releases should carry diminishing weight as new information would be available in practice |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `yfinance` | Market data (SPY, VIX, DXY) |
| `fredapi` | FRED macro series |
| `scikit-learn` | `StandardScaler`, `PCA`, `GaussianMixture` (init only) |
| `scipy` | `multivariate_normal`, `spearmanr` |
| `matplotlib` | Charting |
| `python-dotenv` | API key management |
| `pytest` | Unit tests |

> `hmmlearn` is listed in `requirements.txt` but not used — the HMM is implemented from scratch.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
