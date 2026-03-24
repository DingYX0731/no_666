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

    def evaluate_step(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Return (signal, diagnostics) for logging / visualization.

        ``diagnostics`` should include when available:
        - ``conf_hold``, ``conf_buy``, ``conf_sell``: floats (ideally sum to ~1)
        - ``input_summary``: short human-readable or JSON string of model inputs
        """
        sig = self.generate_signal(prices, position_coin, **kwargs)
        return sig, {
            "conf_hold": float("nan"),
            "conf_buy": float("nan"),
            "conf_sell": float("nan"),
            "input_summary": "",
        }
