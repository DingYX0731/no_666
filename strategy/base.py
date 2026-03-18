"""Base abstractions for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class BaseStrategy(ABC):
    """Common contract for all strategies."""

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def required_prices(self) -> int:
        """Minimum history length needed to produce non-HOLD signals."""

    @abstractmethod
    def generate_signal(self, prices: Sequence[float], position_coin: float) -> str:
        """Return one of BUY/SELL/HOLD."""
