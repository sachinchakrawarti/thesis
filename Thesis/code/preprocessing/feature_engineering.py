from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact set of market-centric features for regime-aware modeling."""
    out = df.copy()

    price = out['close'].astype(float)
    return_series = price.pct_change().fillna(0.0)

    out['close_return'] = return_series
    out['rolling_mean_5'] = price.rolling(5).mean().fillna(method='bfill')
    out['rolling_std_5'] = price.rolling(5).std().fillna(0.0)
    out['momentum_5'] = return_series.rolling(5).mean().fillna(0.0)
    out['momentum_10'] = return_series.rolling(10).mean().fillna(0.0)
    out['volatility_ratio'] = out['rolling_std_5'] / (price.abs().replace(0, np.nan).fillna(1.0))
    out['trend_strength'] = price.diff().rolling(10).mean().fillna(0.0)
    out['price_position'] = (price - price.rolling(20).mean()) / price.rolling(20).std().replace(0, 1)
    out['market_pressure'] = out['close_return'] * out['volume_ratio'].fillna(1.0)
    out['risk_score'] = out['rolling_std_5'] * out['volatility'].fillna(0.0)

    return out.fillna(0.0)


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric features for stable transformer training."""
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaled = scaler.fit_transform(df[numeric_cols])
    normalized = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)
    return pd.concat([normalized, df.drop(columns=numeric_cols)], axis=1)
