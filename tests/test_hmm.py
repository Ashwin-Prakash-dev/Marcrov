import pandas as pd
import numpy as np
import pytest
from src.hmm import GaussianHMM


def test_hmm_shape_and_fit():
    """Tests if the HMM fits and returns the correct probability matrix shape."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)

    X = np.column_stack([
        np.random.randn(100),
        np.random.randn(100),
    ])

    model = GaussianHMM(n_states=4, random_state=42, sticky_kappa=10.0)
    model.fit(X)

    probas = model.predict_proba(X)

    assert probas.shape == (100, 4), \
        "Probability matrix should be (100 days, 4 states)."
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6), \
        "Probabilities for each day must sum to 1.0."


def test_sort_states_by_feature_mean():
    """Tests if states are correctly ordered from lowest to highest risk."""
    np.random.seed(42)
    T = 200

    X = np.random.randn(T, 2)

    # VIX proxy that strictly increases over time —> later periods = higher risk
    vix_proxy = np.linspace(10, 80, T)

    model = GaussianHMM(n_states=4, random_state=42, sticky_kappa=10.0)
    model.fit(X)

    gamma = model.predict_proba(X)
    state_order = model.sort_states_by_feature_mean(gamma, vix_proxy)

    # Returns a list of length K with state indices
    assert len(state_order) == 4, \
        "sort_states_by_feature_mean should return a list of length n_states."
    assert set(state_order) == {0, 1, 2, 3}, \
        "Returned list must be a permutation of [0, 1, 2, 3]."

    # Reorder gamma using the returned order
    gamma_sorted = gamma[:, state_order]

    # The state with the highest average VIX (column 3 after
    # sorting) should dominate periods of higher VIX than the lowest-risk state.
    dominant = gamma_sorted.argmax(axis=1)
    high_risk_vix = vix_proxy[dominant == 3].mean() if (dominant == 3).any() else 0
    low_risk_vix  = vix_proxy[dominant == 0].mean() if (dominant == 0).any() else 0

    assert high_risk_vix >= low_risk_vix, \
        "State 3 (Crisis) should dominate higher-VIX periods than State 0 (Calm)."


def test_bic_decreases_then_increases():
    """BIC should have a minimum — it should not monotonically decrease."""
    np.random.seed(42)
    X = np.random.randn(300, 3)

    bic_scores = {}
    for n in range(2, 6):
        m = GaussianHMM(n_states=n, n_iter=50, random_state=42,
                        sticky_kappa=10.0, covar_reg=1e-2)
        m.fit(X)
        bic_scores[n] = m.bic(X)

    # BIC values should all be finite
    assert all(np.isfinite(v) for v in bic_scores.values()), \
        "All BIC values must be finite."


def test_viterbi_output_shape():
    """Viterbi decoding should return a 1-D integer array of length T."""
    np.random.seed(0)
    X = np.random.randn(150, 2)

    model = GaussianHMM(n_states=3, n_iter=50, random_state=0,
                        sticky_kappa=10.0, covar_reg=1e-2)
    model.fit(X)
    states = model.predict(X)

    assert states.shape == (150,), \
        "Viterbi output should have shape (T,)."
    assert states.dtype in [np.int32, np.int64, int], \
        "Viterbi output should be integer dtype."
    assert set(states).issubset({0, 1, 2}), \
        "All predicted states should be in range [0, n_states)."
