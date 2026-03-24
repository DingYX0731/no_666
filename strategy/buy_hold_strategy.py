"""Buy once with available quote, then hold indefinitely."""

from __future__ import annotations

import json
from typing import Any, Sequence

from .base import BaseStrategy


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy-and-hold baseline strategy.

    In this project, "BUY" means: trader_engine will place a BUY order using RiskManager
    (potentially dust sizing + reserve constraints), then the strategy should stop buying.
    """

    def __init__(
        self,
        mode: str = "all",
        use_dust: bool = False,
        buy_once_per_pair: bool = True,
        cash_reserve_ratio: float = 0.7,
        dust_multiplier: float = 1.1,
    ) -> None:
        super().__init__(name="buy_hold")
        mode = str(mode).strip().lower()
        if mode not in {"single", "all"}:
            raise ValueError("mode must be 'single' or 'all'")
        if cash_reserve_ratio < 0 or cash_reserve_ratio >= 1:
            raise ValueError("cash_reserve_ratio must be in [0, 1)")
        if dust_multiplier <= 0:
            raise ValueError("dust_multiplier must be > 0")

        self.mode = mode
        self.use_dust = bool(use_dust)
        self.buy_once_per_pair = bool(buy_once_per_pair)
        self.cash_reserve_ratio = float(cash_reserve_ratio)
        self.dust_multiplier = float(dust_multiplier)

        # Per-strategy-instance guard (trader_engine creates one instance per pair).
        self.has_bought_once = False

    @property
    def required_prices(self) -> int:
        return 1

    def generate_signal(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> str:
        quote_free = float(kwargs.get("quote_free", 0.0))
        if self.buy_once_per_pair and self.has_bought_once:
            return "HOLD"

        if position_coin <= 1e-12 and quote_free > 1e-12:
            # Mark as bought-once at signal time.
            # trader_engine will still do best-effort order execution + logs.
            self.has_bought_once = True
            return "BUY"
        return "HOLD"

    def order_sizing_hints(self) -> dict[str, Any]:
        """Hints consumed by RiskManager for BUY sizing."""
        return {
            "use_dust": self.use_dust,
            "dust_multiplier": self.dust_multiplier,
            "cash_reserve_ratio": self.cash_reserve_ratio,
        }

    def evaluate_step(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        signal = self.generate_signal(prices, position_coin, **kwargs)
        quote_free = float(kwargs.get("quote_free", 0.0))
        if signal == "BUY":
            ch, cb, cs = 0.0, 1.0, 0.0
        else:
            ch, cb, cs = 1.0, 0.0, 0.0
        summary = json.dumps(
            {
                "position_coin": round(float(position_coin), 8),
                "quote_free": round(quote_free, 6),
                "last_price": round(float(kwargs.get("last_price", prices[-1] if prices else 0.0)), 8),
                "mode": self.mode,
                "use_dust": self.use_dust,
                "buy_once_per_pair": self.buy_once_per_pair,
                "has_bought_once": self.has_bought_once,
            },
            separators=(",", ":"),
        )
        return signal, {
            "conf_hold": ch,
            "conf_buy": cb,
            "conf_sell": cs,
            "input_summary": summary,
        }
