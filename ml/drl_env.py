"""Single-asset crypto Gymnasium env inspired by FinRL CryptoEnv.

Supports two observation modes:
- Close-price log-returns only (legacy, feature_dim=1)
- Pre-computed multi-factor feature matrix (feature_dim >= 1)
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CryptoSingleAssetEnv(gym.Env):
    """One-asset spot simulator on close-price series with optional features.

    Optional reward shaping (training only) to reduce buy-and-hold collapse
    and encourage round-trips that match full-position backtest:

    - ``holding_penalty_per_step``: subtracted each step while holding base asset
      (encourages eventual SELL).
    - ``sell_execution_bonus``: added when a SELL action actually executes
      (makes closing a position slightly more attractive vs HOLD).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        lookback: int = 20,
        initial_cash: float = 10_000.0,
        buy_cost_pct: float = 0.0008,
        sell_cost_pct: float = 0.0008,
        buy_fraction: float = 1.0,
        sell_fraction: float = 1.0,
        features: np.ndarray | None = None,
        holding_penalty_per_step: float = 0.0,
        holding_penalty_growth: float = 0.0,
        sell_execution_bonus: float = 0.0,
        invalid_action_penalty: float = 0.0,
        realized_pnl_reward_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        if self.prices.size < lookback + 3:
            raise ValueError("Need at least lookback+3 price points for stepping.")
        self.lookback = lookback
        self.initial_cash = float(initial_cash)
        self.buy_cost_pct = float(buy_cost_pct)
        self.sell_cost_pct = float(sell_cost_pct)
        self.buy_fraction = float(buy_fraction)
        self.sell_fraction = float(sell_fraction)
        self.holding_penalty_per_step = float(holding_penalty_per_step)
        self.holding_penalty_growth = float(holding_penalty_growth)
        self.sell_execution_bonus = float(sell_execution_bonus)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.realized_pnl_reward_scale = float(realized_pnl_reward_scale)

        if features is not None:
            self.features = np.asarray(features, dtype=np.float64)
            if self.features.shape[0] != self.prices.size:
                raise ValueError(
                    f"features rows ({self.features.shape[0]}) != prices length ({self.prices.size})"
                )
            self.feature_dim = self.features.shape[1]
        else:
            self.features = None
            self.feature_dim = 1

        self.action_space = spaces.Discrete(3)
        obs_dim = lookback * self.feature_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._t = 0
        self._cash = 0.0
        self._base = 0.0
        self._avg_entry_price = 0.0
        self._position_age = 0

    def _equity(self, t: int) -> float:
        return self._cash + self._base * self.prices[t]

    def _obs(self, t: int) -> np.ndarray:
        if self.features is not None:
            seq = self.features[t - self.lookback : t]  # [lookback, F]
        else:
            window = self.prices[t - self.lookback : t + 1]
            prev = window[:-1]
            nxt = window[1:]
            log_ret = np.log(nxt / np.clip(prev, 1e-12, None))
            seq = log_ret.reshape(-1, 1)  # [lookback, 1]

        eq = max(self._equity(t), 1e-12)
        cash_ratio = np.array([self._cash / eq], dtype=np.float64)
        pos_value = self._base * self.prices[t]
        pos_ratio = np.array([pos_value / eq], dtype=np.float64)
        return np.concatenate([seq.flatten(), cash_ratio, pos_ratio]).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._t = self.lookback
        self._cash = self.initial_cash
        self._base = 0.0
        self._avg_entry_price = 0.0
        self._position_age = 0
        return self._obs(self._t), {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        t = self._t
        price = self.prices[t]
        equity_before = self._equity(t)
        sold_this_step = False
        invalid_action = False
        realized_pnl_pct = 0.0

        if action == 1 and self._cash > 0 and price > 0:
            spend = self._cash * self.buy_fraction
            fill_price = price * (1.0 + self.buy_cost_pct)
            qty = spend / fill_price
            prev_base = self._base
            self._cash -= spend
            self._base += qty
            if qty > 0:
                if prev_base <= 1e-12:
                    self._avg_entry_price = float(fill_price)
                else:
                    prev_cost = prev_base * self._avg_entry_price
                    add_cost = qty * fill_price
                    self._avg_entry_price = float((prev_cost + add_cost) / max(self._base, 1e-12))
        elif action == 1:
            invalid_action = True
        elif action == 2 and self._base > 0 and price > 0:
            sell_qty = self._base * self.sell_fraction
            fill_price = price * (1.0 - self.sell_cost_pct)
            proceeds = sell_qty * fill_price
            self._base -= sell_qty
            self._cash += proceeds
            sold_this_step = sell_qty > 0
            if sold_this_step and self._avg_entry_price > 1e-12:
                realized_pnl_pct = (fill_price - self._avg_entry_price) / self._avg_entry_price
            if self._base <= 1e-12:
                self._base = 0.0
                self._avg_entry_price = 0.0
                self._position_age = 0
        elif action == 2:
            invalid_action = True

        self._t += 1
        terminated = self._t >= len(self.prices) - 1
        t2 = min(self._t, len(self.prices) - 1)
        equity_after = self._equity(t2)
        reward = float((equity_after - equity_before) / max(equity_before, 1e-12))

        # Encourage closing positions and discourage policy collapse.
        if self.holding_penalty_per_step > 0 and self._base > 1e-12:
            self._position_age += 1
            reward -= self.holding_penalty_per_step * (
                1.0 + self.holding_penalty_growth * float(self._position_age)
            )
        elif self._base <= 1e-12:
            self._position_age = 0
        if self.sell_execution_bonus > 0 and sold_this_step:
            reward += self.sell_execution_bonus
        if self.invalid_action_penalty > 0 and invalid_action:
            reward -= self.invalid_action_penalty
        if self.realized_pnl_reward_scale != 0.0 and sold_this_step:
            reward += self.realized_pnl_reward_scale * float(realized_pnl_pct)

        obs = self._obs(t2)
        return obs, reward, terminated, False, {}
