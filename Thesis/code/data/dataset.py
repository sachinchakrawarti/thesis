from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from code.preprocessing.feature_engineering import engineer_features, normalize_features


def build_synthetic_market_data(length: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trend = np.linspace(-0.5, 0.5, length)
    noise = rng.normal(0, 1, length)
    returns = trend + 0.1 * noise
    values = np.cumsum(returns)

    columns = [
        'close',
        'return',
        'volatility',
        'momentum',
        'rsi',
        'macd',
        'volume_ratio',
        'corr_signal',
        'trend_strength',
        'risk_score',
        'liquidity',
        'market_pressure'
    ]

    data = pd.DataFrame(
        np.column_stack([
            values,
            returns,
            np.abs(noise),
            np.roll(returns, 1),
            np.clip(noise, -1, 1),
            np.linspace(-1, 1, length),
            rng.uniform(0.9, 1.1, length),
            rng.uniform(-0.2, 0.2, length),
            trend,
            np.abs(noise) * 0.5,
            rng.uniform(0.5, 1.5, length),
            rng.uniform(-1, 1, length)
        ]),
        columns=columns
    )
    return engineer_features(data)


def prepare_data(window_size: int = 60):
    raw_data = build_synthetic_market_data()
    processed = normalize_features(raw_data)

    target = processed['close_return'].shift(-1).fillna(0.0)
    regimes = np.where(processed['close_return'] > 0.12, 3,
                       np.where(processed['close_return'] < -0.12, 1, 2))

    processed['target'] = target
    processed['regime'] = regimes

    feature_columns = [col for col in processed.columns if col not in {'target', 'regime'}]
    features = processed[feature_columns].to_numpy(dtype=np.float32)
    labels = processed['regime'].to_numpy(dtype=np.int64)
    y = processed['target'].to_numpy(dtype=np.float32)

    windows = []
    window_labels = []
    window_targets = []
    for i in range(len(features) - window_size + 1):
        windows.append(features[i:i + window_size])
        window_labels.append(labels[i + window_size - 1])
        window_targets.append(y[i + window_size - 1])

    X = torch.tensor(np.array(windows), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(window_targets), dtype=torch.float32)
    regime_label = torch.tensor(np.array(window_labels), dtype=torch.long)

    x_train, x_rest, y_train, y_rest, r_train, r_rest = train_test_split(
        X, y_tensor, regime_label, test_size=0.3, random_state=42, shuffle=False
    )
    x_val, x_test, y_val, y_test, r_val, r_test = train_test_split(
        x_rest, y_rest, r_rest, test_size=0.5, random_state=42, shuffle=False
    )

    return {
        'train': (x_train, y_train, r_train),
        'val': (x_val, y_val, r_val),
        'test': (x_test, y_test, r_test)
    }
