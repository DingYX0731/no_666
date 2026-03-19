"""Base abstractions for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class BaseStrategy(ABC):
    """Common contract for all strategies."""

    def __init__(self, name: str):
        self.name = name

    @property
    @abstractmethod
    def required_prices(self) -> int:
        """Minimum history length needed to produce non-HOLD signals."""

    @abstractmethod
    def generate_signal(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> str:
        """Return one of BUY/SELL/HOLD.

        Optional kwargs (used by some strategies, e.g. DRL):
        - quote_free: free quote balance
        - last_price: latest close / last price for the pair
        """
