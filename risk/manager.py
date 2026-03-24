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
        *,
        use_dust: bool = False,
        dust_multiplier: float = 1.1,
        cash_reserve_ratio: float = 0.7,
    ) -> float:
        """Compute safe order quantity under risk constraints."""
        if signal == "BUY":
            if price <= 0:
                return 0.0

            if use_dust:
                # Dust sizing: aim near min_notional_usd, but never spend too much cash.
                target_notional = self.min_notional_usd * float(dust_multiplier)
                # Keep cash reserve to avoid buying out all quote.
                max_spend_by_reserve = quote_free * (1.0 - float(cash_reserve_ratio))
                # Also respect max_position_usd cap.
                max_spend_by_cap = float(self.max_position_usd)
                spend = min(target_notional, max_spend_by_reserve, max_spend_by_cap)
            else:
                # Default behavior: cap buy notional by available quote balance and configured max position.
                spend = min(quote_free, float(self.max_position_usd))

            if spend < self.min_notional_usd:
                return 0.0

            qty = spend / price
            return max(float(qty), 0.0)

        if signal == "SELL":
            # Sell available base asset position.
            qty = max(base_free, 0.0)
            if qty * price < self.min_notional_usd:
                return 0.0
            return qty

        return 0.0
