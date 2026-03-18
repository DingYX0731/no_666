"""Risk manager implementation used by trading runtime."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class RiskManager:
    """Risk guardrails used by the live trader loop."""

    max_position_usd: float
    max_daily_loss_pct: float
    min_notional_usd: float
    max_consecutive_errors: int

    def can_trade(
        self,
        equity_now: float,
        day_start_equity: float,
        consecutive_errors: int,
    ) -> Tuple[bool, str]:
        """Return whether trading is allowed with a reason string."""
        if consecutive_errors >= self.max_consecutive_errors:
            return False, "Too many consecutive API errors."

        if day_start_equity > 0:
            drawdown = (day_start_equity - equity_now) / day_start_equity
            if drawdown >= self.max_daily_loss_pct:
                return False, f"Daily drawdown limit reached ({drawdown:.2%})."

        return True, "OK"

    def calc_order_quantity(
        self,
        signal: str,
        price: float,
        quote_free: float,
        base_free: float,
    ) -> float:
        """Compute safe order quantity under risk constraints."""
        if signal == "BUY":
            # Cap buy notional by available quote balance and configured max position.
            buy_budget = min(quote_free, self.max_position_usd)
            qty = buy_budget / price if price > 0 else 0.0
            if qty * price < self.min_notional_usd:
                return 0.0
            return max(qty, 0.0)

        if signal == "SELL":
            # Sell available base asset position.
            qty = max(base_free, 0.0)
            if qty * price < self.min_notional_usd:
                return 0.0
            return qty

        return 0.0
