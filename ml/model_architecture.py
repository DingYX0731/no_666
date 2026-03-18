"""Model architecture: single-hidden-layer MLP binary classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModelConfig:
    """Configuration persisted with checkpoint."""

    input_dim: int
    hidden_dim: int
    seed: int = 42


class SingleHiddenLayerMLP:
    """Simple MLP with ReLU hidden layer and sigmoid output."""

    def __init__(self, config: ModelConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.w1 = rng.normal(0, 0.1, size=(config.input_dim, config.hidden_dim))
        self.b1 = np.zeros((1, config.hidden_dim))
        self.w2 = rng.normal(0, 0.1, size=(config.hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

        # Cached tensors for backward pass.
        self.z1: np.ndarray | None = None
        self.a1: np.ndarray | None = None
        self.z2: np.ndarray | None = None
        self.y_hat: np.ndarray | None = None

    @staticmethod
    def _relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning probability."""
        self.z1 = x @ self.w1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.y_hat = self._sigmoid(self.z2)
        return self.y_hat

    def backward(self, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        """Backward pass gradients for current batch."""
        if self.y_hat is None or self.a1 is None or self.z1 is None:
            raise RuntimeError("Call forward before backward.")
        n = x.shape[0]
        dz2 = (self.y_hat - y) / n
        dw2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        da1 = dz2 @ self.w2.T
        dz1 = da1 * (self.z1 > 0)
        dw1 = x.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        return {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}

    def apply_grads(self, grads: dict[str, np.ndarray], lr: float) -> None:
        """Gradient descent update."""
        self.w1 -= lr * grads["dw1"]
        self.b1 -= lr * grads["db1"]
        self.w2 -= lr * grads["dw2"]
        self.b2 -= lr * grads["db2"]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Predict probability without caching grads."""
        z1 = x @ self.w1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.w2 + self.b2
        return self._sigmoid(z2)

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels with threshold."""
        return (self.predict_proba(x) >= threshold).astype(np.float64)

    def save(self, ckpt_path: str | Path, *, feature_mean: np.ndarray, feature_std: np.ndarray) -> None:
        """Save model weights and feature normalization stats."""
        path = Path(ckpt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            seed=self.config.seed,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    @classmethod
    def load(cls, ckpt_path: str | Path) -> tuple["SingleHiddenLayerMLP", np.ndarray, np.ndarray]:
        """Load model and normalization stats from checkpoint."""
        obj = np.load(Path(ckpt_path), allow_pickle=False)
        config = ModelConfig(
            input_dim=int(obj["input_dim"]),
            hidden_dim=int(obj["hidden_dim"]),
            seed=int(obj["seed"]),
        )
        model = cls(config)
        model.w1 = obj["w1"]
        model.b1 = obj["b1"]
        model.w2 = obj["w2"]
        model.b2 = obj["b2"]
        feature_mean = obj["feature_mean"]
        feature_std = obj["feature_std"]
        return model, feature_mean, feature_std
