import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM

class MacroRegimeHMM:
    """Wrapper for the Gaussian Hidden Markov Model tailored for Macro Regime detection."""
    
    def __init__(self, n_states: int = 4, random_state: int = 42):
        # We force the domain prior of 4 states (Calm, Elevated, Stressed, Crisis)
        self.n_states = n_states
        self.model = GaussianHMM(
            n_components=n_states, 
            covariance_type="full", 
            n_iter=1000, 
            random_state=random_state,
            tol=1e-4
        )
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame):
        """Fits the HMM to the PCA feature matrix."""
        print(f"Fitting {self.n_states}-state HMM...")
        self.model.fit(X.values)
        self.is_fitted = True
        
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predicts the probability of being in each state for every timestamp."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        
        probas = self.model.predict_proba(X.values)
        return pd.DataFrame(probas, index=X.index, columns=[f'State_{i}' for i in range(self.n_states)])
        
    def sort_states_by_risk(self, probas: pd.DataFrame, risk_proxy: pd.Series) -> pd.DataFrame:
        """
        HMM states are output in random order. This aligns them so State 0 is 'Calm' 
        and State 3 is 'Crisis' by checking the mean of a risk proxy (like VIX) 
        when each state is the dominant probability.
        """
        dominant_states = probas.idxmax(axis=1)
        state_risk_means = {}
        
        for state in probas.columns:
            state_mask = dominant_states == state
            state_risk_means[state] = risk_proxy[state_mask].mean()
            
        # Sort states by their average risk proxy value
        sorted_states = sorted(state_risk_means, key=state_risk_means.get)
        
        # Rename columns to reflect sorted risk (0 = Lowest Risk, 3 = Highest Risk)
        rename_map = {old_name: f'Regime_{i}' for i, old_name in enumerate(sorted_states)}
        return probas.rename(columns=rename_map)
    