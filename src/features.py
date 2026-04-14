"""
PCA
---
Applied on the standardised feature matrix (StandardScaler).
Components are selected by explained variance threshold (default 95%).
Scaler and PCA are ALWAYS fitted on training data only and then applied
via .transform() on the full dataset — the caller is responsible for
enforcing this split.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Canonical feature names — used by run_model.py for column selection
FEATURE_COLS = [
    'Z_VIX', 'Z_CS', 'Z_YC', 'Z_DXY',
    'dVIX',  'dCS',  'dYC',  'dDXY',
    'Z_ISM', 'Z_Claims', 'dISM', 'dClaims',
]


def _roll_z(s: pd.Series, window: int = 252, min_periods: int = 126) -> pd.Series:
    """
    Parameters
    ----------
    s : pd.Series
        Input series (should be log-transformed where appropriate).
    window : int
        Rolling window in trading days (default 252 ≈ 1 year).
    min_periods : int
        Minimum observations required to compute a valid z-score.

    Returns
    -------
    pd.Series
        Same index as input; NaN where insufficient history.
    """
    m  = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - m) / sd.replace(0, np.nan)


def build_feature_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """
    columns 
    ---------------------------------------------------------------
    VIX, Credit_Spread, Yield_Curve_Spread, DXY,
    ISM_PMI, Log_Jobless_Claims, ISM_PMI_Chg, Log_Jobless_Claims_Chg

    Returns
    -------
    pd.DataFrame
        Columns: FEATURE_COLS (12 features).  Rows with NaN dropped.
    """
    df = pd.DataFrame(index=raw.index)

    # ── Level z-scores (log-transformed where appropriate) ──────────────
    df['Z_VIX']    = _roll_z(np.log(raw['VIX']))
    df['Z_CS']     = _roll_z(np.log(raw['Credit_Spread']))
    df['Z_YC']     = _roll_z(raw['Yield_Curve_Spread'])
    df['Z_DXY']    = _roll_z(raw['DXY'])
    df['Z_ISM']    = _roll_z(raw['ISM_PMI'])
    df['Z_Claims'] = _roll_z(raw['Log_Jobless_Claims'])

    # ── Momentum (20-day difference) ────────────────────────────────────
    df['dVIX']    = np.log(raw['VIX']).diff(20)
    df['dCS']     = np.log(raw['Credit_Spread']).diff(20)
    df['dYC']     = raw['Yield_Curve_Spread'].diff(20)
    df['dDXY']    = raw['DXY'].diff(20)
    df['dISM']    = raw['ISM_PMI_Chg']            # 21-day diff, pre-computed
    df['dClaims'] = raw['Log_Jobless_Claims_Chg'] # 5-day diff, pre-computed

    df.dropna(inplace=True)

    print(
        f"Feature matrix: {df.shape} | "
        f"{df.index.min().date()} → {df.index.max().date()}"
    )
    return df[FEATURE_COLS]


def fit_scaler_pca(
    X_train: np.ndarray,
    n_components: float = 0.95,
    random_state: int = 42,
) -> tuple:
    """
    Parameters
    ----------
    X_train : np.ndarray, shape (T_train, n_features)
        Raw feature values for the training period.
    n_components : float or int
        If float in (0, 1): fraction of variance to retain (e.g. 0.95).
        If int: fixed number of components.
    random_state : int

    Returns
    -------
    scaler : StandardScaler (fitted)
    pca    : PCA (fitted)
    Xtr_p  : np.ndarray — PCA-transformed training data
    """
    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)

    pca   = PCA(n_components=n_components, random_state=random_state)
    Xtr_p = pca.fit_transform(Xtr_s)

    n_kept = pca.n_components_
    var    = pca.explained_variance_ratio_.sum() * 100
    print(
        f"\nPCA: {X_train.shape[1]} features → {n_kept} components "
        f"({var:.1f}% variance retained)"
    )
    print(
        "  Per-component variance: "
        + "  ".join([
            f"PC{i + 1}={v * 100:.1f}%"
            for i, v in enumerate(pca.explained_variance_ratio_)
        ])
    )
    return scaler, pca, Xtr_p


def apply_scaler_pca(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    """
    Parameters
    ----------
    X : np.ndarray, shape (T, n_features)
    scaler : StandardScaler fitted on training data
    pca    : PCA fitted on training data

    Returns
    -------
    np.ndarray, shape (T, n_components)
    """
    return pca.transform(scaler.transform(X))


def print_pca_loadings(pca: PCA, feature_names: list, top_k: int = 2) -> None:
    """
    Parameters
    ----------
    pca : PCA (fitted)
    feature_names : list of str
    top_k : int
        Number of top features to display per component.
    """
    n_components = pca.n_components_
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f'PC{i + 1}' for i in range(n_components)],
    )
    print("\nPCA loadings (top-2 features per component):")
    for col in loadings.columns:
        top = loadings[col].abs().nlargest(top_k).index.tolist()
        signs = [f"{f}({loadings.loc[f, col]:+.2f})" for f in top]
        print(f"  {col}: {', '.join(signs)}")
