"""Trainer abstraction for the MLP demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .loss import BinaryCrossEntropyLoss
from .model_architecture import ModelConfig, SingleHiddenLayerMLP


@dataclass
class TrainerConfig:
    """Trainer hyperparameters."""

    hidden_dim: int = 16
    lr: float = 0.01
    epochs: int = 300
    seed: int = 42


@dataclass
class TrainReport:
    """Training report metrics."""

    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    samples: int
    features: int


class MLPTrainer:
    """Training workflow for SingleHiddenLayerMLP."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.loss_fn = BinaryCrossEntropyLoss()

    @staticmethod
    def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fit z-score stats on train and apply to train/test."""
        mu = train_x.mean(axis=0, keepdims=True)
        sigma = train_x.std(axis=0, keepdims=True) + 1e-8
        return (train_x - mu) / sigma, (test_x - mu) / sigma, mu, sigma

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
    ) -> tuple[SingleHiddenLayerMLP, TrainReport, np.ndarray, np.ndarray]:
        """Train model and return checkpoint-ready artifacts."""
        train_x, test_x, feature_mean, feature_std = self.standardize(train_x, test_x)

        model = SingleHiddenLayerMLP(
            ModelConfig(input_dim=train_x.shape[1], hidden_dim=self.config.hidden_dim, seed=self.config.seed)
        )
        for _ in range(self.config.epochs):
            y_hat = model.forward(train_x)
            _ = self.loss_fn.value(y_hat, train_y)
            grads = model.backward(train_x, train_y)
            model.apply_grads(grads, lr=self.config.lr)

        train_pred = model.predict_proba(train_x)
        test_pred = model.predict_proba(test_x)
        report = TrainReport(
            train_loss=self.loss_fn.value(train_pred, train_y),
            train_acc=self.loss_fn.accuracy(train_pred, train_y),
            test_loss=self.loss_fn.value(test_pred, test_y),
            test_acc=self.loss_fn.accuracy(test_pred, test_y),
            samples=len(train_x) + len(test_x),
            features=train_x.shape[1],
        )
        return model, report, feature_mean, feature_std
