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
from ml.config_loader import get_block, load_yaml_config
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
    """Train MLP from yaml config."""
    parser = argparse.ArgumentParser(description="Train single-hidden-layer MLP from yaml config")
    parser.add_argument("--config", type=str, default="configs/ml/mlp_train.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_cfg = get_block(cfg, "data")
    feat_cfg = get_block(cfg, "features")
    train_cfg = get_block(cfg, "train")
    model_cfg = get_block(cfg, "model")
    loss_cfg = get_block(cfg, "loss")
    out_cfg = get_block(cfg, "out")

    symbol = str(data_cfg.get("symbol", "BTC/USD"))
    interval = str(data_cfg.get("interval", "1h"))
    frequency = str(data_cfg.get("frequency", "daily"))
    market = str(data_cfg.get("market", "spot"))
    start_date = str(data_cfg.get("start_date", "2024-01-01"))
    end_date = str(data_cfg.get("end_date", "2024-03-31"))
    limit = int(data_cfg.get("limit", 0))
    quote_asset = str(data_cfg.get("quote_asset", "USDT"))
    cache_dir = str(data_cfg.get("cache_dir", "data_cache/binance_public_data"))

    lookback = int(feat_cfg.get("lookback", 20))
    train_ratio = float(feat_cfg.get("train_ratio", 0.8))

    hidden_dim = int(model_cfg.get("hidden_dim", 16))
    lr = float(train_cfg.get("lr", 0.01))
    epochs = int(train_cfg.get("epochs", 300))
    seed = int(train_cfg.get("seed", 42))
    loss_name = str(loss_cfg.get("name", "bce"))
    bce_eps = float(loss_cfg.get("eps", 1e-8))
    eval_threshold = float(loss_cfg.get("eval_threshold", 0.5))
    ckpt_path = str(out_cfg.get("ckpt_path", "checkpoints/mlp/single_layer_mlp.npz"))

    closes = load_close_prices(
        symbol=symbol,
        interval=interval,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        market=market,
        quote_asset=quote_asset,
        cache_dir=cache_dir,
    )

    x, y = build_supervised_dataset(closes, lookback=lookback)
    train_x, train_y, test_x, test_y = split_train_test(x, y, train_ratio=train_ratio)

    trainer = MLPTrainer(
        TrainerConfig(
            hidden_dim=hidden_dim,
            lr=lr,
            epochs=epochs,
            seed=seed,
            loss_name=loss_name,
            bce_eps=bce_eps,
            eval_threshold=eval_threshold,
        )
    )
    model, report, feature_mean, feature_std = trainer.fit(train_x, train_y, test_x, test_y)
    model.save(ckpt_path, feature_mean=feature_mean, feature_std=feature_std)

    print("=== Single-Layer MLP Training ===")
    print(f"Config     : {args.config}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Samples    : {report.samples}")
    print(f"Features   : {report.features}")
    print(f"TrainLoss  : {report.train_loss:.6f}")
    print(f"TrainAcc   : {report.train_acc:.4f}")
    print(f"TestLoss   : {report.test_loss:.6f}")
    print(f"TestAcc    : {report.test_acc:.4f}")


if __name__ == "__main__":
    main()
