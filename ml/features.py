"""Feature engineering utilities for timeseries modeling."""

from __future__ import annotations

import numpy as np


def build_supervised_dataset(closes: np.ndarray, lookback: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding-window features with next-return binary labels."""
    if closes.size <= lookback + 1:
        raise ValueError("Not enough points for selected lookback.")
    returns = np.diff(np.log(closes))
    x, y = [], []
    for i in range(lookback, returns.size):
        x.append(returns[i - lookback : i])
        y.append(1.0 if returns[i] > 0 else 0.0)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    return x_arr, y_arr


def split_train_test(x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8):
    """Chronological split for train/test."""
    cut = int(len(x) * train_ratio)
    return x[:cut], y[:cut], x[cut:], y[cut:]
