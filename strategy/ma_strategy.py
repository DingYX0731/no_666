"""Moving-average crossover strategy class."""

from __future__ import annotations

from statistics import mean
from typing import Any, Sequence

import json

from .base import BaseStrategy


class MovingAverageCrossStrategy(BaseStrategy):
    """MA crossover strategy implemented via BaseStrategy."""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        super().__init__(name="ma")
        if short_window <= 0 or long_window <= 0:
            raise ValueError("short_window and long_window must be positive")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window

    @property
    def required_prices(self) -> int:
        return self.long_window + 1

    def generate_signal(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> str:
        if len(prices) < self.required_prices:
            return "HOLD"

        prev_prices = prices[:-1]
        curr_prices = prices
        prev_short = mean(prev_prices[-self.short_window :])
        prev_long = mean(prev_prices[-self.long_window :])
        curr_short = mean(curr_prices[-self.short_window :])
        curr_long = mean(curr_prices[-self.long_window :])

        crossed_up = prev_short <= prev_long and curr_short > curr_long
        crossed_down = prev_short >= prev_long and curr_short < curr_long

        if crossed_up and position_coin <= 0:
            return "BUY"
        if crossed_down and position_coin > 0:
            return "SELL"
        return "HOLD"

    def evaluate_step(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        if len(prices) < self.required_prices:
            return "HOLD", self._ma_diag_empty()

        prev_prices = prices[:-1]
        curr_prices = prices
        prev_short = mean(prev_prices[-self.short_window :])
        prev_long = mean(prev_prices[-self.long_window :])
        curr_short = mean(curr_prices[-self.short_window :])
        curr_long = mean(curr_prices[-self.long_window :])

        crossed_up = prev_short <= prev_long and curr_short > curr_long
        crossed_down = prev_short >= prev_long and curr_short < curr_long

        raw_buy = crossed_up
        raw_sell = crossed_down
        if raw_buy and not raw_sell:
            ch, cb, cs = 0.0, 1.0, 0.0
        elif raw_sell and not raw_buy:
            ch, cb, cs = 0.0, 0.0, 1.0
        elif raw_buy and raw_sell:
            ch, cb, cs = 1.0, 0.0, 0.0
        else:
            ch, cb, cs = 1.0, 0.0, 0.0

        signal = self.generate_signal(prices, position_coin, **kwargs)
        summary = json.dumps(
            {
                "short_window": self.short_window,
                "long_window": self.long_window,
                "curr_short": round(curr_short, 6),
                "curr_long": round(curr_long, 6),
                "prev_short": round(prev_short, 6),
                "prev_long": round(prev_long, 6),
                "crossed_up": crossed_up,
                "crossed_down": crossed_down,
                "last_prices_tail": [round(float(x), 6) for x in prices[-5:]],
            },
            separators=(",", ":"),
        )
        return signal, {
            "conf_hold": ch,
            "conf_buy": cb,
            "conf_sell": cs,
            "input_summary": summary,
        }

    @staticmethod
    def _ma_diag_empty() -> dict[str, Any]:
        return {
            "conf_hold": 1.0,
            "conf_buy": 0.0,
            "conf_sell": 0.0,
            "input_summary": "{}",
        }
