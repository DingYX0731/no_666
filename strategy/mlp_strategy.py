"""MLP checkpoint strategy class."""

from __future__ import annotations

import json
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

    @staticmethod
    def _mlp_three_way_probs(p: float) -> tuple[float, float, float]:
        """Map binary P(up) to (hold, buy, sell) that sum to 1."""
        p = float(np.clip(p, 1e-8, 1.0 - 1e-8))
        w = 2.0 * abs(p - 0.5)
        h = max(0.0, 1.0 - w)
        b = w * p
        s = w * (1.0 - p)
        t = h + b + s
        return h / t, b / t, s / t

    def evaluate_step(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        if len(prices) < self.required_prices:
            return "HOLD", {
                "conf_hold": 1.0,
                "conf_buy": 0.0,
                "conf_sell": 0.0,
                "input_summary": "{}",
            }

        returns = np.diff(np.log(np.asarray(prices, dtype=np.float64)))
        feat = returns[-self.lookback :].reshape(1, -1)
        feat_norm = (feat - self.feature_mean) / self.feature_std
        prob_up = float(self.model.predict_proba(feat_norm)[0, 0])
        ch, cb, cs = self._mlp_three_way_probs(prob_up)

        signal = self.generate_signal(prices, position_coin, **kwargs)
        flat = feat_norm.reshape(-1)
        summary = json.dumps(
            {
                "input_dim": int(self.model.config.input_dim),
                "prob_up": round(prob_up, 6),
                "feat_head": [round(float(x), 6) for x in flat[:12]],
                "feat_tail": [round(float(x), 6) for x in flat[-12:]] if flat.size > 12 else [],
                "threshold_buy": self.threshold_buy,
                "threshold_sell": self.threshold_sell,
            },
            separators=(",", ":"),
        )
        return signal, {
            "conf_hold": ch,
            "conf_buy": cb,
            "conf_sell": cs,
            "input_summary": summary,
        }
