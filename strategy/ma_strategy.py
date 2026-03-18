"""Moving-average crossover strategy class."""

from __future__ import annotations

from statistics import mean
from typing import Sequence

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

    def generate_signal(self, prices: Sequence[float], position_coin: float) -> str:
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
