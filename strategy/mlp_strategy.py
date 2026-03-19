"""MLP checkpoint strategy class."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ml.model_architecture import SingleHiddenLayerMLP

from .base import BaseStrategy


class MLPCheckpointStrategy(BaseStrategy):
    """MLP strategy that loads model and scaling stats from checkpoint once."""

    def __init__(
        self,
        ckpt_path: str,
        threshold_buy: float = 0.55,
        threshold_sell: float = 0.45,
    ):
        super().__init__(name="mlp")
        if threshold_sell >= threshold_buy:
            raise ValueError("threshold_sell must be smaller than threshold_buy")
        self.ckpt_path = ckpt_path
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.model, self.feature_mean, self.feature_std = SingleHiddenLayerMLP.load(ckpt_path)
        self.lookback = int(self.model.config.input_dim)

    @property
    def required_prices(self) -> int:
        # Returns size is len(prices)-1, so we need lookback + 1 closes.
        return self.lookback + 1

    def generate_signal(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> str:
        if len(prices) < self.required_prices:
            return "HOLD"

        returns = np.diff(np.log(np.asarray(prices, dtype=np.float64)))
        feat = returns[-self.lookback :].reshape(1, -1)
        feat = (feat - self.feature_mean) / self.feature_std
        prob_up = float(self.model.predict_proba(feat)[0, 0])

        if prob_up >= self.threshold_buy and position_coin <= 0:
            return "BUY"
        if prob_up <= self.threshold_sell and position_coin > 0:
            return "SELL"
        return "HOLD"
