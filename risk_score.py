import pandas as pd
import numpy as np

def calculate_continuous_risk_score(sorted_probas: pd.DataFrame) -> pd.Series:
    """
    Translates sorted regime probabilities into a continuous 0 to 1 risk score.
    Assumes columns are strictly ordered: Regime_0 (Calm) to Regime_3 (Crisis).
    """
    # Assign a linear severity weight to each state
    # Regime 0 (Calm) = 0.0, Regime 1 = 0.33, Regime 2 = 0.66, Regime 3 (Crisis) = 1.0
    n_states = sorted_probas.shape[1]
    weights = np.linspace(0, 1, n_states)
    
    # Calculate expected value (dot product of probabilities and weights)
    risk_score = sorted_probas.dot(weights)
    
    # Ensure strict 0 to 1 bounding
    return risk_score.clip(0, 1).rename('Risk_Score')

def smooth_signal(signal: pd.Series, window: int = 10) -> pd.Series:
    """Applies a rolling mean to smooth out daily signal noise."""
    return signal.rolling(window=window, min_periods=1).mean()