"""Trading engine supporting one/many/all tradable pairs."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict

from config import Settings
from risk import RiskManager
from strategy import BaseStrategy, build_strategy
from trade.client import RoostooClient
from trade.logging_utils import setup_run_logger


@dataclass
class PairState:
    """State tracked independently per trading pair."""

    prices: Deque[float]
    day_start_equity: float = 0.0
    consecutive_errors: int = 0
    amount_precision: int = 6


def _round_qty(qty: float, amount_precision: int) -> float:
    """Round order quantity by pair precision."""
    if amount_precision < 0:
        return qty
    return round(qty, amount_precision)


def _read_amount_precision(exchange_info: Dict, pair: str) -> int:
    """Extract amount precision for pair from exchangeInfo."""
    pair_info = exchange_info.get("TradePairs", {}).get(pair, {})
    return int(pair_info.get("AmountPrecision", 6))


def _resolve_target_pairs(client: RoostooClient, symbols_arg: str) -> list[str]:
    """Resolve target symbols from CLI arg and validate tradability."""
    exchange_info = client.get_exchange_info()
    trade_pairs = exchange_info.get("TradePairs", {})
    tradable = sorted([p for p, v in trade_pairs.items() if v.get("CanTrade", False)])
    tradable_set = set(tradable)

    arg = symbols_arg.strip()
    if arg.lower() == "all":
        if not tradable:
            raise ValueError("No tradable pairs found in exchangeInfo.")
        return tradable

    requested = [x.strip().upper() for x in arg.split(",") if x.strip()]
    if not requested:
        raise ValueError("No symbols provided. Use --symbols BTC/USD or --symbols all")

    invalid = [s for s in requested if s not in tradable_set]
    if invalid:
        raise ValueError(f"Invalid or non-tradable pairs: {invalid}.")
    return requested


def run(
    once: bool = False,
    symbols: str = "BTC/USD",
    poll_seconds: int | None = None,
    strategy_name: str = "ma",
    strategy_config: str = "",
    run_name: str | None = None,
) -> None:
    """Run one-shot or continuous auto-trading loop over selected symbols."""
    settings = Settings.from_env()
    if poll_seconds is not None:
        settings.poll_seconds = poll_seconds
    settings.validate()

    logger, log_path = setup_run_logger(run_name=run_name)
    client = RoostooClient(settings.base_url, settings.api_key, settings.api_secret)
    risk = RiskManager(
        max_position_usd=settings.max_position_usd,
        max_daily_loss_pct=settings.max_daily_loss_pct,
        min_notional_usd=settings.min_notional_usd,
        max_consecutive_errors=settings.max_consecutive_errors,
    )

    target_pairs = _resolve_target_pairs(client, symbols)
    exchange_info = client.get_exchange_info()
    states: dict[str, PairState] = {}
    strategies: dict[str, BaseStrategy] = {}
    for pair in target_pairs:
        strategy = build_strategy(strategy_name=strategy_name, strategy_config=strategy_config)
        strategies[pair] = strategy
        states[pair] = PairState(
            prices=deque(maxlen=max(strategy.required_prices + 5, settings.long_window + 5)),
            day_start_equity=0.0,
            consecutive_errors=0,
            amount_precision=_read_amount_precision(exchange_info, pair),
        )

    logger.info(
        "Starting trader. pairs=%s dry_run=%s strategy=%s strategy_config=%s",
        ",".join(target_pairs),
        settings.dry_run,
        strategy_name,
        strategy_config or f"configs/strategies/{strategy_name}.yaml",
    )
    logger.info("Run log file: %s", log_path)

    while True:
        for pair in target_pairs:
            state = states[pair]
            try:
                ticker = client.get_ticker(pair)
                price = client.parse_last_price(ticker, pair)
                state.prices.append(price)

                balance = client.get_balance()
                base_free, quote_free = client.parse_wallet(balance, pair)
                equity_now = quote_free + base_free * price
                if state.day_start_equity <= 0:
                    state.day_start_equity = equity_now

                ok_to_trade, reason = risk.can_trade(
                    equity_now=equity_now,
                    day_start_equity=state.day_start_equity,
                    consecutive_errors=state.consecutive_errors,
                )
                signal = strategies[pair].generate_signal(
                    prices=list(state.prices),
                    position_coin=base_free,
                )

                logger.info(
                    "pair=%s price=%.6f signal=%s base=%.6f quote=%.2f equity=%.2f risk=%s",
                    pair,
                    price,
                    signal,
                    base_free,
                    quote_free,
                    equity_now,
                    reason,
                )

                if ok_to_trade and signal in {"BUY", "SELL"}:
                    qty = risk.calc_order_quantity(
                        signal=signal,
                        price=price,
                        quote_free=quote_free,
                        base_free=base_free,
                    )
                    qty = _round_qty(qty, state.amount_precision)
                    if qty > 0:
                        if settings.dry_run:
                            logger.info("[DRY_RUN] %s %s qty=%s @ market", signal, pair, qty)
                        else:
                            resp = client.place_order(
                                pair=pair,
                                side=signal,
                                quantity=qty,
                                order_type="MARKET",
                            )
                            logger.info("pair=%s ORDER response=%s", pair, resp)
                    else:
                        logger.info("pair=%s signal=%s but qty <= 0, skip.", pair, signal)

                state.consecutive_errors = 0
            except Exception as exc:  # pylint: disable=broad-except
                state.consecutive_errors += 1
                logger.error("pair=%s loop error #%s: %s", pair, state.consecutive_errors, exc)
                if state.consecutive_errors >= settings.max_consecutive_errors:
                    logger.error("pair=%s reached max consecutive errors, removing from loop.", pair)
                    target_pairs = [p for p in target_pairs if p != pair]

        if not target_pairs:
            logger.error("No active pairs remaining. Stop trader.")
            break

        if once:
            break

        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("sleep=%ss (utc=%s)", settings.poll_seconds, now)
        time.sleep(settings.poll_seconds)
