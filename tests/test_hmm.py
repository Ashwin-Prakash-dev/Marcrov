import pandas as pd
import numpy as np
import pytest
from src.hmm import MacroRegimeHMM

def test_hmm_shape_and_fit():
    """ Tests if the HMM wrapper fits and returns the correct probability matrix shape """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    
    # Mock 2-component PCA data
    X = pd.DataFrame({
        'PC1': np.random.randn(100), 
        'PC2': np.random.randn(100)
    }, index=dates)
    
    model = MacroRegimeHMM(n_states=4, random_state=42)
    model.fit(X)
    
    probas = model.predict_proba(X)
    
    assert probas.shape == (100, 4), "Probability matrix should be 100 days x 4 states."
    assert np.allclose(probas.sum(axis=1), 1.0), "Probabilities for each day must sum to 1.0"

def test_sort_states_by_risk():
    """ Tests if the model correctly orders states from lowest risk to highest risk """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100)
    
    X = pd.DataFrame({'PC1': np.random.randn(100)}, index=dates)
    
    # fake VIX proxy that strictly increases over time
    vix_proxy = pd.Series(np.linspace(10, 50, 100), index=dates)
    
    model = MacroRegimeHMM(n_states=4, random_state=42)
    model.fit(X)
    raw_probas = model.predict_proba(X)
    
    # sorting logic
    sorted_probas = model.sort_states_by_risk(raw_probas, vix_proxy)
    # column names
    expected_cols = ['Regime_0', 'Regime_1', 'Regime_2', 'Regime_3']
    assert list(sorted_probas.columns) == expected_cols, "Columns were not renamed correctly."
    
    # 2. Mathematical Proof: 
    # If the sorting worked, the average VIX when Regime_3 is dominant 
    # must be strictly greater than the average VIX when Regime_0 is dominant
    dominant_states = sorted_probas.idxmax(axis=1)
    
    vix_mean_regime_0 = vix_proxy[dominant_states == 'Regime_0'].mean()
    vix_mean_regime_3 = vix_proxy[dominant_states == 'Regime_3'].mean()
    
    assert vix_mean_regime_3 > vix_mean_regime_0, "Regime 3 (Crisis) did not capture the highest VIX periods."