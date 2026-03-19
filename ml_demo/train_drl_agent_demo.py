"""Train a per-symbol DRL (PPO) agent with custom MLP+LSTM extractor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.config_loader import get_block, load_yaml_config
from ml.drl_trainer import train_ppo_for_symbol


def main() -> None:
    """CLI entry for DRL training demo (yaml-driven)."""
    parser = argparse.ArgumentParser(description="Train PPO DRL agent from yaml config")
    parser.add_argument("--config", type=str, default="configs/ml/drl_train.yaml")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    data_cfg = get_block(cfg, "data")
    env_cfg = get_block(cfg, "env")
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
        lookback=int(env_cfg.get("lookback", 20)),
        timesteps=int(train_cfg.get("timesteps", 50_000)),
        seed=int(train_cfg.get("seed", 42)),
        out_dir=str(out_cfg.get("out_dir", "checkpoints/drl")),
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 2048)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        buy_cost_pct=float(env_cfg.get("buy_cost_pct", 0.0008)),
        sell_cost_pct=float(env_cfg.get("sell_cost_pct", 0.0008)),
        buy_fraction=float(env_cfg.get("buy_fraction", 0.25)),
        sell_fraction=float(env_cfg.get("sell_fraction", 0.5)),
        initial_cash=float(env_cfg.get("initial_cash", 10_000.0)),
        lstm_hidden_size=int(arch_cfg.get("lstm_hidden_size", 64)),
        lstm_layers=int(arch_cfg.get("lstm_layers", 1)),
        lstm_dropout=float(arch_cfg.get("lstm_dropout", 0.0)),
        seq_mlp_hidden_dims=str(arch_cfg.get("seq_mlp_hidden_dims", "64")),
        account_mlp_hidden_dims=str(arch_cfg.get("account_mlp_hidden_dims", "16")),
        fusion_hidden_dims=str(arch_cfg.get("fusion_hidden_dims", "64")),
        policy_hidden_dims=str(arch_cfg.get("policy_hidden_dims", "64,64")),
    )
    print(f"Config     : {args.config}")
    print(f"Saved model: {model_path}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()

