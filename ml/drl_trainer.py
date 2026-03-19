"""DRL training pipeline (per-product PPO agent) within ml package.

Uses the data.market_dataset feature pipeline so the DRL agent sees the
same multi-factor features as the MLP model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data.market_dataset import build_market_feature_dataset

from .drl_env import CryptoSingleAssetEnv
from .drl_utils import pair_to_slug, save_drl_meta


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
    use_agg_trades: bool,
    use_trades: bool,
    vol_window: int,
    lookback: int,
    timesteps: int,
    seed: int,
    device: str,
    out_dir: str,
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
    feature_dim = dataset.features.shape[1]
    print(f"[drl] feature_dim={feature_dim}, features={dataset.feature_names}")
    print(f"[drl] rows={dataset.features.shape[0]}, lookback={lookback}")
    print(f"[drl] device={device}")

    env = CryptoSingleAssetEnv(
        prices=dataset.closes,
        lookback=lookback,
        initial_cash=initial_cash,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        buy_fraction=buy_fraction,
        sell_fraction=sell_fraction,
        features=dataset.features,
    )

    seq_dims = parse_hidden_dims(seq_mlp_hidden_dims, [64])
    acc_dims = parse_hidden_dims(account_mlp_hidden_dims, [16])
    fusion_dims = parse_hidden_dims(fusion_hidden_dims, [64])
    head_dims = parse_hidden_dims(policy_hidden_dims, [64, 64])

    policy_kwargs = dict(
        features_extractor_class=MlpLstmFeatureExtractor,
        features_extractor_kwargs=dict(
            sequence_len=lookback,
            feature_dim=feature_dim,
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
        device=device,
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
            "feature_dim": feature_dim,
            "feature_names": dataset.feature_names,
            "obs_dim": lookback * feature_dim + 2,
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
            "use_agg_trades": use_agg_trades,
            "use_trades": use_trades,
            "vol_window": vol_window,
            "extractor": {
                "name": "MlpLstmFeatureExtractor",
                "feature_dim": feature_dim,
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
                "device": device,
            },
        },
    )
    return model_path, meta_path
