"""Loss and metrics for binary classification."""

from __future__ import annotations

import numpy as np


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss with clipping for numerical stability."""

    @staticmethod
    def value(y_hat: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))

    @staticmethod
    def accuracy(y_hat: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> float:
        pred = (y_hat >= threshold).astype(np.float64)
        return float((pred == y).mean())
