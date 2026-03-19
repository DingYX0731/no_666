"""Stable-Baselines3 PPO strategy: one saved agent per product (pair)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from drl.io_utils import load_drl_meta, pair_to_slug

from .base import BaseStrategy


class DRLSb3Strategy(BaseStrategy):
    """Load PPO zip + meta json produced by `run_train_drl.py` per pair."""

    def __init__(
        self,
        pair: str,
        model_dir: str = "checkpoints/drl",
        deterministic: bool = True,
    ):
        super().__init__(name="drl")
        self.pair = pair.strip().upper()
        self.model_dir = Path(model_dir)
        self.deterministic = deterministic
        slug = pair_to_slug(self.pair)
        self._model_path = self.model_dir / f"{slug}_ppo.zip"
        self._meta_path = self.model_dir / f"{slug}_meta.json"
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"DRL model not found for {self.pair}: {self._model_path}. "
                f"Train with: python run_train_drl.py --symbol {self.pair}"
            )
        if not self._meta_path.is_file():
            raise FileNotFoundError(f"DRL meta not found: {self._meta_path}")
        self._meta = load_drl_meta(self._meta_path)
        self.lookback = int(self._meta["lookback"])
        try:
            from stable_baselines3 import PPO as PPOCls
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DRL strategy requires: pip install -r requirements-drl.txt"
            ) from exc
        self._model = PPOCls.load(str(self._model_path))

    @property
    def required_prices(self) -> int:
        return self.lookback + 1

    def _build_obs(
        self,
        prices: Sequence[float],
        position_coin: float,
        quote_free: float,
        last_price: float,
    ) -> np.ndarray:
        arr = np.asarray(prices, dtype=np.float64)
        t = arr.size - 1
        window = arr[t - self.lookback : t + 1]
        prev = window[:-1]
        nxt = window[1:]
        log_ret = np.log(nxt / np.clip(prev, 1e-12, None)).astype(np.float32)
        eq = max(float(quote_free) + float(position_coin) * float(last_price), 1e-12)
        cash_ratio = np.array([float(quote_free) / eq], dtype=np.float32)
        pos_ratio = np.array(
            [(float(position_coin) * float(last_price)) / eq], dtype=np.float32
        )
        return np.concatenate([log_ret, cash_ratio, pos_ratio]).astype(np.float32)

    def generate_signal(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> str:
        quote_free = float(kwargs.get("quote_free", 0.0))
        last_price = float(kwargs.get("last_price", 0.0))
        if len(prices) < self.required_prices or last_price <= 0:
            return "HOLD"

        obs = self._build_obs(prices, position_coin, quote_free, last_price)
        action, _ = self._model.predict(obs, deterministic=self.deterministic)
        a = int(action) if not isinstance(action, (list, np.ndarray)) else int(action[0])

        if a == 1 and position_coin <= 0 and quote_free > 0:
            return "BUY"
        if a == 2 and position_coin > 0:
            return "SELL"
        return "HOLD"
