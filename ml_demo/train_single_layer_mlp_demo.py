"""Trainable demo with checkpoint export for strategy deployment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

# Allow running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.binance_public_data import KLINES_COLUMNS, BinancePublicDataClient
from ml.features import build_supervised_dataset, split_train_test
from ml.trainer import MLPTrainer, TrainerConfig


def load_close_prices(
    symbol: str,
    interval: str,
    frequency: str,
    start_date: str,
    end_date: str,
    limit: int,
    market: str,
    quote_asset: str,
    cache_dir: str,
) -> np.ndarray:
    """Load sorted close prices from Binance archives."""
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
        raise ValueError("No kline files found for training.")
    rows = client.iter_csv_rows(summary.extracted_csv_files, columns=KLINES_COLUMNS)
    rows.sort(key=lambda x: int(x["open_time"]))
    closes = np.array([float(r["close"]) for r in rows], dtype=np.float64)
    if closes.size < 50:
        raise ValueError("Too few data points; increase date range.")
    return closes


def main() -> None:
    """CLI training demo with checkpoint save path."""
    parser = argparse.ArgumentParser(description="Train single-hidden-layer MLP and save checkpoint")
    parser.add_argument("--symbol", type=str, default="BTC/USD")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--frequency", type=str, default="daily", choices=["daily", "monthly"])
    parser.add_argument("--market", type=str, default="spot", choices=["spot", "um", "cm"])
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2024-03-31")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quote-asset", type=str, default="USDT")
    parser.add_argument("--cache-dir", type=str, default="data_cache/binance_public_data")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/mlp/single_layer_mlp.npz")
    args = parser.parse_args()

    closes = load_close_prices(
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

    x, y = build_supervised_dataset(closes, lookback=args.lookback)
    train_x, train_y, test_x, test_y = split_train_test(x, y)

    trainer = MLPTrainer(
        TrainerConfig(hidden_dim=args.hidden_dim, lr=args.lr, epochs=args.epochs, seed=args.seed)
    )
    model, report, feature_mean, feature_std = trainer.fit(train_x, train_y, test_x, test_y)
    model.save(args.ckpt_path, feature_mean=feature_mean, feature_std=feature_std)

    print("=== Single-Layer MLP Training ===")
    print(f"Checkpoint : {args.ckpt_path}")
    print(f"Samples    : {report.samples}")
    print(f"Features   : {report.features}")
    print(f"TrainLoss  : {report.train_loss:.6f}")
    print(f"TrainAcc   : {report.train_acc:.4f}")
    print(f"TestLoss   : {report.test_loss:.6f}")
    print(f"TestAcc    : {report.test_acc:.4f}")


if __name__ == "__main__":
    main()
