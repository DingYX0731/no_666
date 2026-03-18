"""Backward-compatible MA signal wrapper."""

from typing import Sequence

from .ma_strategy import MovingAverageCrossStrategy

def generate_signal(
    prices: Sequence[float],
    position_coin: float,
    short_window: int = 5,
    long_window: int = 20,
) -> str:
    """Generate BUY/SELL/HOLD using MA strategy class."""
    strategy = MovingAverageCrossStrategy(short_window=short_window, long_window=long_window)
    return strategy.generate_signal(prices=prices, position_coin=position_coin)
