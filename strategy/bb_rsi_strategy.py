"""Bollinger Band + RSI mean reversion strategy (rule-based)."""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from .base import BaseStrategy


def _sma(x: np.ndarray) -> float:
    return float(np.mean(x))


def _std_sample(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.std(x, ddof=1))


def _ema_last(prices: np.ndarray, span: int) -> float:
    """EMA at last point; alpha = 2/(span+1)."""
    if prices.size == 0:
        return float("nan")
    if prices.size == 1:
        return float(prices[-1])
    alpha = 2.0 / (float(span) + 1.0)
    ema = float(prices[0])
    for p in prices[1:]:
        ema = alpha * float(p) + (1.0 - alpha) * ema
    return ema


def _rsi_last(prices: np.ndarray, window: int) -> float:
    """RSI from last `window` price changes (simple average gains/losses)."""
    if prices.size < window + 1:
        return float("nan")
    deltas = np.diff(prices.astype(np.float64))
    tail = deltas[-window:]
    gains = np.maximum(tail, 0.0)
    losses = np.maximum(-tail, 0.0)
    avg_g = float(np.mean(gains))
    avg_l = float(np.mean(losses))
    if avg_l <= 1e-12:
        return 100.0 if avg_g > 1e-12 else 50.0
    rs = avg_g / avg_l
    return float(100.0 - 100.0 / (1.0 + rs))


class BollingerRSIStrategy(BaseStrategy):
    """Mean reversion: lower BB + RSI oversold -> BUY; upper BB + RSI overbought -> SELL."""

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_window: int = 14,
        rsi_buy_threshold: float = 35.0,
        rsi_sell_threshold: float = 65.0,
        rsi_extreme_buy: float = 25.0,
        rsi_extreme_sell: float = 75.0,
        trend_window: int = 50,
        use_trend_filter: bool = True,
        stop_loss_pct: float = 0.03,
        trailing_stop: bool = True,
        cooldown_steps: int = 3,
    ) -> None:
        super().__init__(name="bb_rsi")
        if bb_window <= 1 or rsi_window <= 1 or trend_window <= 1:
            raise ValueError("bb_window, rsi_window, trend_window must be > 1")
        if bb_std <= 0 or stop_loss_pct < 0 or stop_loss_pct >= 1:
            raise ValueError("Invalid bb_std or stop_loss_pct")
        if cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")

        self.bb_window = int(bb_window)
        self.bb_std = float(bb_std)
        self.rsi_window = int(rsi_window)
        self.rsi_buy_threshold = float(rsi_buy_threshold)
        self.rsi_sell_threshold = float(rsi_sell_threshold)
        self.rsi_extreme_buy = float(rsi_extreme_buy)
        self.rsi_extreme_sell = float(rsi_extreme_sell)
        self.trend_window = int(trend_window)
        self.use_trend_filter = bool(use_trend_filter)
        self.stop_loss_pct = float(stop_loss_pct)
        self.trailing_stop = bool(trailing_stop)
        self.cooldown_steps = int(cooldown_steps)

        self._step_count = 0
        self._last_trade_step = -1
        self._entry_price: float | None = None
        self._peak_since_entry: float | None = None

    @property
    def required_prices(self) -> int:
        return max(self.bb_window, self.rsi_window + 1, self.trend_window) + 1

    def _indicators(self, prices: Sequence[float]) -> dict[str, float]:
        arr = np.asarray(prices, dtype=np.float64)
        tail_bb = arr[-self.bb_window :]
        mid = _sma(tail_bb)
        sd = _std_sample(tail_bb)
        upper = mid + self.bb_std * sd
        lower = mid - self.bb_std * sd
        rsi = _rsi_last(arr, self.rsi_window)
        trend_slice = arr[-self.trend_window :]
        ema = _ema_last(trend_slice, self.trend_window)
        last = float(arr[-1])
        return {
            "last": last,
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "rsi": rsi,
            "ema": ema,
        }

    def _buy_cooldown_active(self) -> bool:
        if self._last_trade_step < 0:
            return False
        return (self._step_count - self._last_trade_step) < self.cooldown_steps

    def _mark_trade(self) -> None:
        self._last_trade_step = self._step_count

    def generate_signal(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> str:
        self._step_count += 1
        last_price = float(kwargs.get("last_price", prices[-1]))

        if len(prices) < self.required_prices:
            return "HOLD"

        ind = self._indicators(prices)
        last = ind["last"]
        upper = ind["bb_upper"]
        lower = ind["bb_lower"]
        rsi = ind["rsi"]
        ema = ind["ema"]

        in_position = position_coin > 1e-12

        # Sync peak / entry when flat (e.g. external reset or first bar flat)
        if not in_position:
            self._entry_price = None
            self._peak_since_entry = None

        # Trailing / stop from peak since entry
        if in_position and self.trailing_stop and self._peak_since_entry is not None:
            self._peak_since_entry = max(self._peak_since_entry, last_price)
            floor = self._peak_since_entry * (1.0 - self.stop_loss_pct)
            if last_price <= floor:
                self._mark_trade()
                self._entry_price = None
                self._peak_since_entry = None
                return "SELL"

        # Initialize peak on first bar we observe a position without prior state
        if in_position and self._peak_since_entry is None:
            self._peak_since_entry = last_price
            if self._entry_price is None:
                self._entry_price = last_price

        # BB + RSI exits (not blocked by buy cooldown)
        if in_position:
            trend_ok_sell = (
                not self.use_trend_filter
                or last < ema
                or rsi > self.rsi_extreme_sell
            )
            if (
                last > upper
                and rsi > self.rsi_sell_threshold
                and trend_ok_sell
            ):
                self._mark_trade()
                self._entry_price = None
                self._peak_since_entry = None
                return "SELL"

        # Entries: respect cooldown after any trade
        if self._buy_cooldown_active():
            return "HOLD"

        if not in_position:
            trend_ok_buy = (
                not self.use_trend_filter
                or last > ema
                or rsi < self.rsi_extreme_buy
            )
            if (
                last < lower
                and rsi < self.rsi_buy_threshold
                and trend_ok_buy
            ):
                self._mark_trade()
                self._entry_price = last_price
                self._peak_since_entry = last_price
                return "BUY"

        return "HOLD"

    def evaluate_step(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        if len(prices) < self.required_prices:
            # Keep _step_count aligned with generate_signal() (increments once per bar).
            self._step_count += 1
            return "HOLD", self._diag_empty()

        # Snapshot indicators before generate_signal mutates counters/state for this bar
        ind = self._indicators(prices)
        signal = self.generate_signal(prices, position_coin, **kwargs)

        raw_buy = (
            ind["last"] < ind["bb_lower"]
            and ind["rsi"] < self.rsi_buy_threshold
            and (
                not self.use_trend_filter
                or ind["last"] > ind["ema"]
                or ind["rsi"] < self.rsi_extreme_buy
            )
        )
        raw_sell = (
            ind["last"] > ind["bb_upper"]
            and ind["rsi"] > self.rsi_sell_threshold
            and (
                not self.use_trend_filter
                or ind["last"] < ind["ema"]
                or ind["rsi"] > self.rsi_extreme_sell
            )
        )

        if signal == "BUY":
            ch, cb, cs = 0.0, 1.0, 0.0
        elif signal == "SELL":
            ch, cb, cs = 0.0, 0.0, 1.0
        elif raw_buy and raw_sell:
            ch, cb, cs = 1.0, 0.0, 0.0
        else:
            ch, cb, cs = 1.0, 0.0, 0.0

        summary = json.dumps(
            {
                "bb_window": self.bb_window,
                "bb_std": self.bb_std,
                "rsi_window": self.rsi_window,
                "trend_window": self.trend_window,
                "last": round(ind["last"], 8),
                "bb_mid": round(ind["bb_mid"], 8),
                "bb_upper": round(ind["bb_upper"], 8),
                "bb_lower": round(ind["bb_lower"], 8),
                "rsi": round(ind["rsi"], 4),
                "ema": round(ind["ema"], 8),
                "raw_buy": raw_buy,
                "raw_sell": raw_sell,
                "signal": signal,
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
    def _diag_empty() -> dict[str, Any]:
        return {
            "conf_hold": 1.0,
            "conf_buy": 0.0,
            "conf_sell": 0.0,
            "input_summary": "{}",
        }
