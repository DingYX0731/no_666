"""Trainable demo with checkpoint export for strategy deployment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.market_dataset import (
    build_market_feature_dataset,
    build_supervised_pairs,
    split_supervised_pairs,
)
from ml.config_loader import get_block, load_yaml_config
from ml.trainer import MLPTrainer, TrainerConfig
from trade.logging_utils import setup_training_logger


class _Tee:
    """Write to multiple streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return default


def main() -> None:
    """Train MLP from yaml config."""
    parser = argparse.ArgumentParser(description="Train single-hidden-layer MLP from yaml config")
    parser.add_argument("--config", type=str, default="configs/ml/mlp_train.yaml")
    args = parser.parse_args()

    logger, log_path = setup_training_logger(run_prefix="mlp")
    logger.info("Log file: %s", log_path)
    logger.info("Config: %s", args.config)

    log_file = log_path.open("a", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_file)
    try:
        _run_training(args, logger, log_path)
    finally:
        sys.stdout = old_stdout
        log_file.close()


def _run_training(args, logger, log_path) -> None:
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
    use_agg_trades = _as_bool(data_cfg.get("use_agg_trades", True), True)
    use_trades = _as_bool(data_cfg.get("use_trades", True), True)
    vol_window = int(data_cfg.get("vol_window", 20))

    lookback = int(feat_cfg.get("lookback", 20))
    horizon = int(feat_cfg.get("horizon", 1))
    label_mode = str(feat_cfg.get("label_mode", "binary")).lower()
    train_ratio = float(feat_cfg.get("train_ratio", 0.8))
    split_method = str(feat_cfg.get("split_method", "chronological")).lower()

    hidden_dim = int(model_cfg.get("hidden_dim", 16))
    lr = float(train_cfg.get("lr", 0.01))
    epochs = int(train_cfg.get("epochs", 300))
    seed = int(train_cfg.get("seed", 42))
    loss_name = str(loss_cfg.get("name", "bce"))
    bce_eps = float(loss_cfg.get("eps", 1e-8))
    eval_threshold = float(loss_cfg.get("eval_threshold", 0.5))
    ckpt_path = str(out_cfg.get("ckpt_path", "checkpoints/mlp/single_layer_mlp.npz"))

    dataset = build_market_feature_dataset(
        symbol=symbol,
        interval=interval,
        frequency=frequency,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        market=market,
        quote_asset=quote_asset,
        cache_dir=cache_dir,
        use_agg_trades=use_agg_trades,
        use_trades=use_trades,
        vol_window=vol_window,
    )

    pairs = build_supervised_pairs(
        dataset,
        lookback=lookback,
        horizon=horizon,
        label_mode="binary" if label_mode not in {"binary", "regression"} else label_mode,
    )
    train_x, train_y, test_x, test_y = split_supervised_pairs(
        pairs,
        train_ratio=train_ratio,
        method="random" if split_method == "random" else "chronological",
        seed=seed,
    )
    # MLP expects 2D vectors; flatten [lookback, feature_dim] windows.
    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)

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
    print(f"Features   : {','.join(dataset.feature_names)}")
    print(f"InputShape : [{pairs.lookback}, {dataset.features.shape[1]}] -> flattened")
    print(f"LabelMode  : {pairs.label_mode} (horizon={pairs.horizon})")
    print(f"Split      : {split_method} (train_ratio={train_ratio})")
    print(f"Samples    : {report.samples}")
    print(f"FlatDim    : {report.features}")
    print(f"TrainLoss  : {report.train_loss:.6f}")
    print(f"TrainAcc   : {report.train_acc:.4f}")
    print(f"TestLoss   : {report.test_loss:.6f}")
    print(f"TestAcc    : {report.test_acc:.4f}")


if __name__ == "__main__":
    main()
