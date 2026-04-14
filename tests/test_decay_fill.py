import pandas as pd
import numpy as np
import pytest

def mock_apply_exponential_decay(series: pd.Series, halflife: int, max_staleness: int) -> pd.Series:
    """Mock implementation to demonstrate the test logic."""
    filled = series.copy()
    last_valid_val = np.nan
    days_stale = 0
    
    for i in range(len(series)):
        if pd.notna(series.iloc[i]):
            last_valid_val = series.iloc[i]
            days_stale = 0
        else:
            days_stale += 1
            if days_stale > max_staleness:
                filled.iloc[i] = np.nan
            else:
                filled.iloc[i] = last_valid_val * (0.5 ** (days_stale / halflife))
                
    return filled

def test_exponential_decay_halflife():
    """ Tests if the data properly decays by exactly 50% at the halflife mark """
    dates = pd.date_range("2024-01-01", periods=15)
    
    # Data is released on day 1 at a value of 100, then nothing for 14 days
    data = pd.Series([np.nan] * 15, index=dates)
    data.iloc[0] = 100.0 
    
    halflife = 10
    max_staleness = 45
    
    # Replace mock with your actual import: apply_exponential_decay(data, halflife, max_staleness)
    decayed_series = mock_apply_exponential_decay(data, halflife, max_staleness)
    
    # At day 0 (index 0), value should be 100
    assert decayed_series.iloc[0] == 100.0
    
    # At day 10 (index 10), exactly one halflife has passed. Value has to be 50.0
    assert np.isclose(decayed_series.iloc[10], 50.0), "Decay did not hit exactly 50% at the halflife."
    
def test_max_staleness_cutoff():
    """ Tests if the model stops trusting data completely after max_staleness days"""
    dates = pd.date_range("2024-01-01", periods=50)
    
    # Data released on day 1, then a massive gap
    data = pd.Series([np.nan] * 50, index=dates)
    data.iloc[0] = 100.0 
    
    halflife = 10
    max_staleness = 45
    
    decayed_series = mock_apply_exponential_decay(data, halflife, max_staleness)
    
    # Day 45 should still have a (highly decayed) value
    assert pd.notna(decayed_series.iloc[45]), "Data was cut off too early"
    
    # Day 46 exceeds max staleness, should return NaN
    assert pd.isna(decayed_series.iloc[46]), "Data exceeded max staleness but did not return NaN"