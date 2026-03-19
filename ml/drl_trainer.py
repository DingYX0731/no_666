"""DRL training pipeline (per-product PPO agent) within ml package."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.binance_public_data import KLINES_COLUMNS, BinancePublicDataClient

from .drl_env import CryptoSingleAssetEnv
from .drl_utils import pair_to_slug, save_drl_meta


def load_closes(
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
    """Load sorted close prices through BinancePublicDataClient."""
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
        raise ValueError("No kline files for DRL training.")
    rows = client.iter_csv_rows(summary.extracted_csv_files, columns=KLINES_COLUMNS)
    rows.sort(key=lambda x: int(x["open_time"]))
    return np.array([float(r["close"]) for r in rows], dtype=np.float64)


def train_ppo_for_symbol(
    *,
    symbol: str,
    interval: str,
    frequency: str,
    start_date: str,
    end_date: str,
    limit: int,
    market: str,
    quote_asset: str,
    cache_dir: str,
    lookback: int,
    timesteps: int,
    seed: int,
    out_dir: str,
    # PPO/env hyper-params
    learning_rate: float,
    n_steps: int,
    batch_size: int,
    gamma: float,
    gae_lambda: float,
    ent_coef: float,
    clip_range: float,
    buy_cost_pct: float,
    sell_cost_pct: float,
    buy_fraction: float,
    sell_fraction: float,
    initial_cash: float,
    # Custom extractor hyper-params
    lstm_hidden_size: int,
    lstm_layers: int,
    lstm_dropout: float,
    seq_mlp_hidden_dims: str,
    account_mlp_hidden_dims: str,
    fusion_hidden_dims: str,
    policy_hidden_dims: str,
) -> tuple[Path, Path]:
    """Train PPO agent for a single trading pair and save artifacts."""
    from .drl_model_architecture import MlpLstmFeatureExtractor, parse_hidden_dims

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install DRL deps: pip install -r requirements-drl.txt"
        ) from exc

    closes = load_closes(
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
    env = CryptoSingleAssetEnv(
        prices=closes,
        lookback=lookback,
        initial_cash=initial_cash,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        buy_fraction=buy_fraction,
        sell_fraction=sell_fraction,
    )

    seq_dims = parse_hidden_dims(seq_mlp_hidden_dims, [64])
    acc_dims = parse_hidden_dims(account_mlp_hidden_dims, [16])
    fusion_dims = parse_hidden_dims(fusion_hidden_dims, [64])
    head_dims = parse_hidden_dims(policy_hidden_dims, [64, 64])

    policy_kwargs = dict(
        features_extractor_class=MlpLstmFeatureExtractor,
        features_extractor_kwargs=dict(
            sequence_len=lookback,
            lstm_hidden_size=lstm_hidden_size,
            lstm_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            seq_mlp_hidden_dims=seq_dims,
            account_mlp_hidden_dims=acc_dims,
            fusion_hidden_dims=fusion_dims,
        ),
        net_arch=dict(pi=head_dims, vf=head_dims),
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=timesteps)

    slug = pair_to_slug(symbol)
    out = Path(out_dir)
    model_path = out / f"{slug}_ppo.zip"
    meta_path = out / f"{slug}_meta.json"
    model.save(str(model_path))
    save_drl_meta(
        meta_path,
        {
            "pair": symbol.strip().upper(),
            "lookback": lookback,
            "obs_dim": lookback + 2,
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "model_path": str(model_path.name),
            "timesteps": timesteps,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "buy_cost_pct": buy_cost_pct,
            "sell_cost_pct": sell_cost_pct,
            "buy_fraction": buy_fraction,
            "sell_fraction": sell_fraction,
            "initial_cash": initial_cash,
            "extractor": {
                "name": "MlpLstmFeatureExtractor",
                "lstm_hidden_size": lstm_hidden_size,
                "lstm_layers": lstm_layers,
                "lstm_dropout": lstm_dropout,
                "seq_mlp_hidden_dims": seq_dims,
                "account_mlp_hidden_dims": acc_dims,
                "fusion_hidden_dims": fusion_dims,
            },
            "policy_head_hidden_dims": head_dims,
            "ppo": {
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "ent_coef": ent_coef,
                "clip_range": clip_range,
                "seed": seed,
            },
        },
    )
    return model_path, meta_path

