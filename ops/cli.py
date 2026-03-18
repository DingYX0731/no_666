"""Unified operations CLI for Roostoo account/trading actions."""

from __future__ import annotations

import argparse
import json
from typing import Any

from config import Settings
from trade.client import RoostooClient


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _build_client() -> RoostooClient:
    s = Settings.from_env()
    return RoostooClient(s.base_url, s.api_key, s.api_secret)


def _cmd_server_time(_: argparse.Namespace) -> None:
    client = _build_client()
    _print_json(client.get_server_time())


def _cmd_products(args: argparse.Namespace) -> None:
    client = _build_client()
    info = client.get_exchange_info()
    trade_pairs = info.get("TradePairs", {})
    if args.detail:
        _print_json(trade_pairs)
        return
    pairs = sorted(trade_pairs.keys())
    print(f"Tradable product count: {len(pairs)}")
    for p in pairs:
        print(p)


def _cmd_ticker(args: argparse.Namespace) -> None:
    client = _build_client()
    pair = args.pair or None
    resp = client.get_ticker(pair)
    if args.raw:
        _print_json(resp)
        return
    data = resp.get("Data", {})
    if pair:
        v = data.get(pair, {})
        print(f"Pair: {pair}")
        print(f"LastPrice: {v.get('LastPrice')}")
        print(f"MaxBid: {v.get('MaxBid')}")
        print(f"MinAsk: {v.get('MinAsk')}")
        return
    for k in sorted(data.keys()):
        v = data.get(k, {})
        print(f"{k} | LastPrice={v.get('LastPrice')} | Bid={v.get('MaxBid')} | Ask={v.get('MinAsk')}")


def _cmd_balance(args: argparse.Namespace) -> None:
    client = _build_client()
    pair = args.pair or "BTC/USD"
    resp = client.get_balance()
    if args.raw:
        _print_json(resp)
        return
    base_free, quote_free = client.parse_wallet(resp, pair)
    ticker = client.get_ticker(pair)
    last_price = client.parse_last_price(ticker, pair)
    equity = quote_free + base_free * last_price
    print(f"Pair: {pair}")
    print(f"Base Free: {base_free}")
    print(f"Quote Free: {quote_free}")
    print(f"Last Price: {last_price}")
    print(f"Estimated Equity: {equity}")


def _cmd_pending_count(_: argparse.Namespace) -> None:
    client = _build_client()
    _print_json(client.get_pending_count())


def _cmd_orders(args: argparse.Namespace) -> None:
    client = _build_client()
    order_id = args.order_id or None
    pair = args.pair or "BTC/USD"
    if order_id:
        resp = client.query_order(order_id=order_id)
    else:
        resp = client.query_order(pair=pair, pending_only=args.pending_only)
    if args.raw:
        _print_json(resp)
        return
    orders = resp.get("OrderMatched", [])
    print(f"Matched Orders: {len(orders)}")
    for x in orders:
        print(
            f"OrderID={x.get('OrderID')} Pair={x.get('Pair')} Side={x.get('Side')} "
            f"Type={x.get('Type')} Status={x.get('Status')} Price={x.get('Price')} Qty={x.get('Quantity')}"
        )


def _cmd_place_order(args: argparse.Namespace) -> None:
    client = _build_client()
    pair = args.pair or "BTC/USD"
    side = args.side.upper()
    order_type = args.type.upper()
    if order_type == "LIMIT" and args.price is None:
        raise ValueError("LIMIT order requires --price")
    if not args.force:
        print(
            "[SIMULATION] Add --force to execute order:",
            {"pair": pair, "side": side, "type": order_type, "quantity": args.quantity, "price": args.price},
        )
        return
    resp = client.place_order(
        pair=pair,
        side=side,
        quantity=args.quantity,
        order_type=order_type,
        price=args.price,
    )
    _print_json(resp)


def _cmd_cancel_order(args: argparse.Namespace) -> None:
    client = _build_client()
    order_id = args.order_id or None
    pair = args.pair or "BTC/USD"
    if not args.force:
        print("[SIMULATION] Add --force to execute cancel:", {"order_id": order_id, "pair": pair})
        return
    if order_id:
        resp = client.cancel_order(order_id=order_id)
    else:
        resp = client.cancel_order(pair=pair)
    _print_json(resp)


def build_parser() -> argparse.ArgumentParser:
    """Build operations parser with subcommands."""
    parser = argparse.ArgumentParser(description="Operational commands for Roostoo")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("server-time", help="Check server time")
    p.set_defaults(func=_cmd_server_time)

    p = sub.add_parser("products", help="List tradable products")
    p.add_argument("--detail", action="store_true", help="Print full TradePairs details")
    p.set_defaults(func=_cmd_products)

    p = sub.add_parser("ticker", help="Fetch market ticker")
    p.add_argument("--pair", default="", help="Trading pair, e.g. BTC/USD")
    p.add_argument("--raw", action="store_true", help="Print raw API response")
    p.set_defaults(func=_cmd_ticker)

    p = sub.add_parser("balance", help="Fetch wallet balance and estimated equity")
    p.add_argument("--pair", default="", help="Pair for equity estimate")
    p.add_argument("--raw", action="store_true", help="Print raw API response")
    p.set_defaults(func=_cmd_balance)

    p = sub.add_parser("pending-count", help="Show total pending orders")
    p.set_defaults(func=_cmd_pending_count)

    p = sub.add_parser("orders", help="Query order history or pending orders")
    p.add_argument("--order-id", default="", help="Query by order id")
    p.add_argument("--pair", default="", help="Query by pair")
    p.add_argument("--pending-only", action="store_true", help="Only pending orders")
    p.add_argument("--raw", action="store_true", help="Print raw API response")
    p.set_defaults(func=_cmd_orders)

    p = sub.add_parser("place-order", help="Place an order")
    p.add_argument("--pair", default="", help="Trading pair")
    p.add_argument("--side", required=True, choices=["BUY", "SELL"], help="Order side")
    p.add_argument("--type", default="MARKET", choices=["MARKET", "LIMIT"], help="Order type")
    p.add_argument("--quantity", required=True, type=float, help="Order quantity")
    p.add_argument("--price", type=float, default=None, help="Limit order price")
    p.add_argument("--force", action="store_true", help="Execute live order")
    p.set_defaults(func=_cmd_place_order)

    p = sub.add_parser("cancel-order", help="Cancel pending orders")
    p.add_argument("--order-id", default="", help="Cancel by order id")
    p.add_argument("--pair", default="", help="Cancel by pair")
    p.add_argument("--force", action="store_true", help="Execute cancellation")
    p.set_defaults(func=_cmd_cancel_order)

    return parser


def main() -> None:
    """Run ops CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
