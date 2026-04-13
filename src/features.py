import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def calculate_momentum_and_zscores(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Calculates rolling Z-scores and momentum (differencing) for macro features.
    
    Args:
        df (pd.DataFrame): Raw macroeconomic data (VIX, Credit Spreads, Yield Curve, etc.)
        window (int): Rolling window for Z-score calculation.
        
    Returns:
        pd.DataFrame: Engineered feature matrix.
    """
    features = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        # Momentum (Daily Difference)
        features[f'd_{col}'] = df[col].diff()
        
        # Rolling Z-Score
        roll_mean = df[col].rolling(window=window).mean()
        roll_std = df[col].rolling(window=window).std()
        features[f'Z_{col}'] = (df[col] - roll_mean) / (roll_std + 1e-8)
        
    return features.dropna()

def apply_pca(features: pd.DataFrame, n_components: int = 10) -> tuple[pd.DataFrame, PCA]:
    """
    Applies Principal Component Analysis (PCA) to orthogonalize the feature set.
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_features)
    
    pca_df = pd.DataFrame(
        principal_components, 
        index=features.index, 
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Print variance retained (to match your logs)
    variance_retained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"PCA: {features.shape[1]} features → {n_components} components ({variance_retained:.1f}% variance retained)")
    
    return pca_df, pca