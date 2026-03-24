"""Train a per-symbol DRL (PPO) agent with custom MLP+LSTM extractor."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config_loader import get_block, load_yaml_config
from ml.drl_trainer import train_ppo_for_symbol
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
    """CLI entry for DRL training demo (yaml-driven)."""
    parser = argparse.ArgumentParser(description="Train PPO DRL agent from yaml config")
    parser.add_argument("--config", type=str, default="configs/ml/drl_train.yaml")
    args = parser.parse_args()

    logger, log_path = setup_training_logger(run_prefix="drl")
    logger.info("Log file: %s", log_path)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    file_handler.setLevel(logging.INFO)
    root = logging.getLogger()
    root.addHandler(file_handler)

    log_file = log_path.open("a", encoding="utf-8")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_file)
    try:
        _run_training(args, logger, log_path)
    finally:
        sys.stdout = old_stdout
        log_file.close()
        root.removeHandler(file_handler)


def _run_training(args, logger, log_path) -> None:
    logger.info("Config: %s", args.config)

    cfg = load_yaml_config(args.config)
    data_cfg = get_block(cfg, "data")
    env_cfg = get_block(cfg, "env")
    val_cfg = get_block(cfg, "validation")
    train_cfg = get_block(cfg, "train")
    ppo_cfg = get_block(cfg, "ppo")
    arch_cfg = get_block(cfg, "architecture")
    out_cfg = get_block(cfg, "out")

    model_path, meta_path = train_ppo_for_symbol(
        symbol=str(data_cfg.get("symbol", "BTC/USD")),
        interval=str(data_cfg.get("interval", "1h")),
        frequency=str(data_cfg.get("frequency", "daily")),
        start_date=str(data_cfg.get("start_date", "2024-01-01")),
        end_date=str(data_cfg.get("end_date", "2024-03-31")),
        limit=int(data_cfg.get("limit", 0)),
        market=str(data_cfg.get("market", "spot")),
        quote_asset=str(data_cfg.get("quote_asset", "USDT")),
        cache_dir=str(data_cfg.get("cache_dir", "data_cache/binance_public_data")),
        use_agg_trades=_as_bool(data_cfg.get("use_agg_trades", False), False),
        use_trades=_as_bool(data_cfg.get("use_trades", False), False),
        vol_window=int(data_cfg.get("vol_window", 20)),
        lookback=int(env_cfg.get("lookback", 20)),
        timesteps=int(train_cfg.get("timesteps", 200_000)),
        seed=int(train_cfg.get("seed", 42)),
        device=str(train_cfg.get("device", "auto")),
        out_dir=str(out_cfg.get("out_dir", "checkpoints/drl")),
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 512)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        buy_cost_pct=float(env_cfg.get("buy_cost_pct", 0.0008)),
        sell_cost_pct=float(env_cfg.get("sell_cost_pct", 0.0008)),
        buy_fraction=float(env_cfg.get("buy_fraction", 1.0)),
        sell_fraction=float(env_cfg.get("sell_fraction", 1.0)),
        initial_cash=float(env_cfg.get("initial_cash", 10_000.0)),
        holding_penalty_per_step=float(env_cfg.get("holding_penalty_per_step", 0.0)),
        holding_penalty_growth=float(env_cfg.get("holding_penalty_growth", 0.0)),
        sell_execution_bonus=float(env_cfg.get("sell_execution_bonus", 0.0)),
        invalid_action_penalty=float(env_cfg.get("invalid_action_penalty", 0.0)),
        realized_pnl_reward_scale=float(env_cfg.get("realized_pnl_reward_scale", 0.0)),
        lstm_hidden_size=int(arch_cfg.get("lstm_hidden_size", 64)),
        lstm_layers=int(arch_cfg.get("lstm_layers", 1)),
        lstm_dropout=float(arch_cfg.get("lstm_dropout", 0.0)),
        seq_mlp_hidden_dims=str(arch_cfg.get("seq_mlp_hidden_dims", "64")),
        account_mlp_hidden_dims=str(arch_cfg.get("account_mlp_hidden_dims", "16")),
        fusion_hidden_dims=str(arch_cfg.get("fusion_hidden_dims", "64")),
        policy_hidden_dims=str(arch_cfg.get("policy_hidden_dims", "64,64")),
        validation_enabled=bool(val_cfg.get("enabled", False)),
        validation_eval_every_steps=int(val_cfg.get("eval_every_steps", 0)),
        validation_eval_episodes=int(val_cfg.get("eval_episodes", 1)),
        validation_window_start=float(val_cfg.get("window_start", 0.8)),
        validation_window_end=float(val_cfg.get("window_end", 0.9)),
        validation_save_best_by=str(val_cfg.get("save_best_by", "return_pct")),
        validation_early_stop_patience_evals=int(
            val_cfg.get("early_stop_patience_evals", 5)
        ),
        validation_store_best_on_validation_step=bool(
            val_cfg.get("store_best_on_validation_step", True)
        ),
        validation_log_dir=str(log_path.parent),
    )
    print(f"Config     : {args.config}")
    print(f"Saved model: {model_path}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
