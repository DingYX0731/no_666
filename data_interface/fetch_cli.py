"""Internal CLI implementation for Binance archive fetching."""

from __future__ import annotations

import argparse
import json

from data.binance_public_data import (
    AGG_TRADES_COLUMNS,
    KLINES_COLUMNS,
    TRADES_COLUMNS,
    BinancePublicDataClient,
)


def _dataset_columns(dataset: str) -> list[str]:
    if dataset == "klines":
        return KLINES_COLUMNS
    if dataset == "aggTrades":
        return AGG_TRADES_COLUMNS
    if dataset == "trades":
        return TRADES_COLUMNS
    return []


def main() -> None:
    """Fetch Binance public historical archives."""
    parser = argparse.ArgumentParser(description="Fetch Binance public historical archives")
    parser.add_argument("--symbol", required=True, help="BTC/USD, BTCUSDT, or BTC")
    parser.add_argument("--dataset", required=True, choices=["klines", "aggTrades", "trades"])
    parser.add_argument("--frequency", default="daily", choices=["daily", "monthly"])
    parser.add_argument("--interval", default="", help="Required for klines. e.g. 1m, 1h, 1d")
    parser.add_argument("--market", default="spot", choices=["spot", "um", "cm"])
    parser.add_argument("--start-date", default="", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0, help="Number of periods")
    parser.add_argument("--quote-asset", default="USDT")
    parser.add_argument("--cache-dir", default="data_cache/binance_public_data")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--preview-rows", type=int, default=0)
    args = parser.parse_args()

    client = BinancePublicDataClient(cache_dir=args.cache_dir)
    summary = client.fetch_history(
        symbol=args.symbol,
        dataset=args.dataset,
        frequency=args.frequency,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        limit=args.limit or None,
        interval=args.interval or None,
        market=args.market,
        quote_asset=args.quote_asset,
        extract=not args.no_extract,
        skip_missing=not args.strict,
    )
    print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))

    if args.preview_rows > 0 and summary.extracted_csv_files:
        rows = client.iter_csv_rows(
            summary.extracted_csv_files,
            columns=_dataset_columns(args.dataset),
            max_rows=args.preview_rows,
        )
        print("\n# Preview Rows")
        print(json.dumps(rows, indent=2, ensure_ascii=False))
