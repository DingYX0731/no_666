import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from data.binance_public_data import KLINES_COLUMNS, BinancePublicDataClient
from strategy import BaseStrategy, build_strategy


@dataclass
class BacktestResult:
    """Container for backtest performance summary metrics."""
    start_equity: float
    end_equity: float
    return_pct: float
    max_drawdown_pct: float
    trades: int


def read_prices_from_csv(csv_path: Path, price_col: str = "close") -> List[float]:
    """Load price series from CSV file."""
    prices: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if price_col not in row:
                raise ValueError(f"CSV does not contain column: {price_col}")
            prices.append(float(row[price_col]))
    if not prices:
        raise ValueError("No prices loaded from CSV.")
    return prices


def synthetic_prices(n: int = 500, seed: int = 7) -> List[float]:
    """Generate synthetic random-walk prices for quick dry testing."""
    random.seed(seed)
    prices = [100.0]
    for _ in range(n - 1):
        drift = 0.0003
        shock = random.uniform(-0.01, 0.01)
        prices.append(max(1.0, prices[-1] * (1.0 + drift + shock)))
    return prices


def read_prices_from_binance(
    symbol: str,
    interval: str,
    frequency: str,
    start_date: str,
    end_date: str,
    limit: int,
    market: str = "spot",
    quote_asset: str = "USDT",
    cache_dir: str = "data_cache/binance_public_data",
) -> List[float]:
    """Fetch klines from Binance public data and return sorted close prices."""
    client = BinancePublicDataClient(cache_dir=cache_dir)
    summary = client.fetch_history(
        symbol=symbol,
        dataset="klines",
        interval=interval,
        frequency=frequency,
        start_date=start_date or None,
        end_date=end_date or None,
        limit=limit or None,
        market=market,
        quote_asset=quote_asset,
        extract=True,
        skip_missing=True,
    )
    if not summary.extracted_csv_files:
        raise ValueError("No Binance kline files found for the given parameters.")

    rows = client.iter_csv_rows(summary.extracted_csv_files, columns=KLINES_COLUMNS)
    if not rows:
        raise ValueError("No rows loaded from Binance kline files.")

    rows.sort(key=lambda x: int(x["open_time"]))
    closes = [float(row["close"]) for row in rows]
    if not closes:
        raise ValueError("No close prices extracted from Binance kline rows.")
    return closes


def run_backtest(
    prices: Sequence[float],
    strategy: BaseStrategy,
    fee_rate: float = 0.0008,
    initial_quote: float = 10000.0,
) -> BacktestResult:
    """Run backtest with any strategy implementing BaseStrategy."""
    quote = initial_quote
    base = 0.0
    trades = 0
    equity_curve: List[float] = []

    history: List[float] = []
    for price in prices:
        history.append(price)
        signal = strategy.generate_signal(
            history,
            base,
            quote_free=quote,
            last_price=price,
        )

        if signal == "BUY" and quote > 0:
            raw_qty = quote / price
            qty = raw_qty * (1.0 - fee_rate)
            base += qty
            quote = 0.0
            trades += 1
        elif signal == "SELL" and base > 0:
            raw_quote = base * price
            quote += raw_quote * (1.0 - fee_rate)
            base = 0.0
            trades += 1

        equity = quote + base * price
        equity_curve.append(equity)

    end_equity = equity_curve[-1]
    peak = equity_curve[0]
    max_drawdown = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd

    return BacktestResult(
        start_equity=initial_quote,
        end_equity=end_equity,
        return_pct=(end_equity - initial_quote) / initial_quote,
        max_drawdown_pct=max_drawdown,
        trades=trades,
    )


def main() -> None:
    """CLI entry for quick local backtest."""
    parser = argparse.ArgumentParser(description="Backtest runner with pluggable strategy")
    parser.add_argument(
        "--data-source",
        type=str,
        default="synthetic",
        choices=["synthetic", "csv", "binance"],
        help="Backtest data source",
    )
    parser.add_argument("--csv", type=str, default="", help="CSV path with close column")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name")
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Symbol for binance source")
    parser.add_argument("--interval", type=str, default="1h", help="Kline interval for binance source")
    parser.add_argument("--frequency", type=str, default="daily", choices=["daily", "monthly"])
    parser.add_argument("--market", type=str, default="spot", choices=["spot", "um", "cm"])
    parser.add_argument("--start-date", type=str, default="", help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default="", help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0, help="Period count when date range is omitted")
    parser.add_argument("--quote-asset", type=str, default="USDT")
    parser.add_argument("--cache-dir", type=str, default="data_cache/binance_public_data")
    parser.add_argument("--strategy", type=str, default="ma", help="Strategy name")
    parser.add_argument("--strategy-config", type=str, default="", help="Strategy yaml path")
    args = parser.parse_args()

    if args.data_source == "csv":
        if not args.csv:
            raise ValueError("--csv is required when --data-source=csv")
        prices = read_prices_from_csv(Path(args.csv), args.price_col)
    elif args.data_source == "binance":
        prices = read_prices_from_binance(
            symbol=args.symbol,
            interval=args.interval,
            frequency=args.frequency,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
            market=args.market,
            quote_asset=args.quote_asset,
            cache_dir=args.cache_dir,
        )
    else:
        prices = synthetic_prices(500)

    pair_label = args.symbol.strip().upper()
    strategy = build_strategy(
        strategy_name=args.strategy,
        strategy_config=args.strategy_config,
        pair=pair_label,
    )
    result = run_backtest(prices=prices, strategy=strategy)

    print("=== Backtest Result ===")
    print(f"Strategy     : {strategy.name}")
    print(f"Need Prices  : {strategy.required_prices}")
    print(f"Start Equity : {result.start_equity:.2f}")
    print(f"End Equity   : {result.end_equity:.2f}")
    print(f"Return       : {result.return_pct:.2%}")
    print(f"Max Drawdown : {result.max_drawdown_pct:.2%}")
    print(f"Trades       : {result.trades}")


if __name__ == "__main__":
    main()
