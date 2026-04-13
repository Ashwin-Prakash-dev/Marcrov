"""
run_model.py — End-to-End HMM Training & Signal Generation
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline import fetch_market_data
from src.features import calculate_momentum_and_zscores, apply_pca
from src.hmm import MacroRegimeHMM
from src.risk_score import calculate_continuous_risk_score

# ─── CONFIGURATION ───────────────────────────────────────────────────────
TRAIN_START = '2006-01-01'
TRAIN_END   = '2017-12-31'  # Strict separation to avoid data leakage
OUTPUT_FILE = 'outputs/regime_signals.csv'

def main():
    # 1. Fetch raw data (In production, you'd load your PMI and Credit Spread data here too)
    # Using SPY Volatility and TLT as proxy features for this example
    print("Fetching raw macroeconomic data...")
    prices, _ = fetch_market_data(TRAIN_START, '2025-01-01')
    
    raw_features = pd.DataFrame({
        'VIX_Proxy': prices['SPY'].pct_change().rolling(21).std() * 100,
        'Duration_Proxy': prices['TLT'].pct_change()
    }).dropna()

    # 2. Feature Engineering & PCA
    print("Calculating rolling Z-scores and momentum...")
    engineered_features = calculate_momentum_and_zscores(raw_features, window=60)
    
    # (Optional: limit to 2 components since we only have 2 base features in this mock)
    pca_features, pca_model = apply_pca(engineered_features, n_components=2)
    
    # 3. Train/Test Split (Preventing Data Leakage)
    print(f"Splitting data: Train ({TRAIN_START} to {TRAIN_END})...")
    train_idx = pca_features.index[pca_features.index <= TRAIN_END]
    X_train = pca_features.reindex(train_idx)
    
    # 4. Train HMM
    hmm = MacroRegimeHMM(n_states=4)
    hmm.fit(X_train)
    
    # 5. Generate Predictions for the entire dataset (Train + OOS)
    print("Generating regime probabilities for full timeline...")
    raw_probas = hmm.predict_proba(pca_features)
    
    # Align states by risk severity using VIX_Proxy as the guide
    sorted_probas = hmm.sort_states_by_risk(raw_probas, raw_features['VIX_Proxy'])
    
    # 6. Calculate Risk Score
    print("Calculating continuous Risk Score...")
    risk_score = calculate_continuous_risk_score(sorted_probas)
    
    # 7. Save to CSV
    output_df = pd.DataFrame({'Risk_Score': risk_score})
    
    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output_df.to_csv(OUTPUT_FILE)
    print(f"\nSuccess! Regime signals saved to -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()