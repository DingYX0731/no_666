# DRL strategy and FinRL reference

## FinRL upstream (local clone)

The [FinRL](https://github.com/AI4Finance-Foundation/FinRL) repository is cloned next to this project for study:

- `../FinRL` (sibling of `no_666` under `Ding/`)

Relevant FinRL modules for **cryptocurrency** trading:

- `finrl/meta/env_cryptocurrency_trading/env_multiple_crypto.py` — multi-asset `CryptoEnv` (cash, positions, fees, portfolio reward)
- `finrl/meta/env_cryptocurrency_trading/env_btc_ccxt.py` — single-asset style env with technical features
- `finrl/meta/data_processors/processor_ccxt.py` — CCXT data pipeline

## What we implemented in `no_666`

We do **not** vendor FinRL as a dependency. Instead:

1. **`drl/crypto_env.py`** — Gymnasium env with the same *ideas* as FinRL crypto envs: single asset, fees, discrete actions, portfolio return as step reward.
2. **`drl/train_sb3.py`** + **`run_train_drl.py`** — train **one PPO agent per symbol** using Binance public klines (same data path as the rest of `no_666`).
3. **Artifacts** (per pair, filesystem-safe slug e.g. `BTC_USD`):
   - `checkpoints/drl/<PAIR>_ppo.zip`
   - `checkpoints/drl/<PAIR>_meta.json`
4. **`strategy/drl_strategy.py`** — loads the pair-specific zip + meta and maps policy actions to `BUY` / `SELL` / `HOLD` in the live/backtest loop.

## Dependencies

- Base: `gymnasium` (in `requirements.txt`)
- Training + inference: `pip install -r requirements-drl.txt` (PyTorch + Stable-Baselines3)

## Commands

Train for one product:

```bash
pip install -r requirements.txt
pip install -r requirements-drl.txt
python run_train_drl.py --symbol BTC/USD --start-date 2024-01-01 --end-date 2024-02-01 --timesteps 50000
```

Live / backtest:

```bash
python run_trader.py --symbols BTC/USD --strategy drl
python run_backtest.py --data-source binance --symbol BTC/USD --strategy drl --start-date 2024-01-01 --end-date 2024-02-01
```

Multi-pair live trading loads **one checkpoint per pair** automatically from `checkpoints/drl/`.
