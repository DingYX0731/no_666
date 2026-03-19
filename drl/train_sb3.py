"""Train a per-symbol PPO agent on Binance kline history (FinRL-style pipeline)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.binance_public_data import KLINES_COLUMNS, BinancePublicDataClient
from drl.crypto_env import CryptoSingleAssetEnv
from drl.io_utils import pair_to_slug, save_drl_meta


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agent per trading pair (DRL)")
    parser.add_argument("--symbol", type=str, required=True, help="e.g. BTC/USD")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--frequency", type=str, default="daily", choices=["daily", "monthly"])
    parser.add_argument("--market", type=str, default="spot", choices=["spot", "um", "cm"])
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    parser.add_argument("--end-date", type=str, default="2024-03-31")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quote-asset", type=str, default="USDT")
    parser.add_argument("--cache-dir", type=str, default="data_cache/binance_public_data")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/drl",
        help="Directory for model zip and meta json",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Install DRL deps: pip install -r requirements-drl.txt"
        ) from exc

    closes = load_closes(
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
    env = CryptoSingleAssetEnv(prices=closes, lookback=args.lookback)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=None,
    )
    model.learn(total_timesteps=args.timesteps)

    slug = pair_to_slug(args.symbol)
    out_dir = Path(args.out_dir)
    model_path = out_dir / f"{slug}_ppo.zip"
    meta_path = out_dir / f"{slug}_meta.json"
    model.save(str(model_path))
    save_drl_meta(
        meta_path,
        {
            "pair": args.symbol.strip().upper(),
            "lookback": args.lookback,
            "obs_dim": args.lookback + 2,
            "algorithm": "PPO",
            "policy": "MlpPolicy",
            "model_path": str(model_path.name),
            "timesteps": args.timesteps,
            "interval": args.interval,
            "start_date": args.start_date,
            "end_date": args.end_date,
        },
    )
    print(f"Saved model: {model_path}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
