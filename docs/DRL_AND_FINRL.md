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

1. **`ml/drl_env.py`** — Gymnasium env with the same *ideas* as FinRL crypto envs: single asset, fees, discrete actions, portfolio return as step reward.
2. **`ml/drl_model_architecture.py`** — custom **MLP + LSTM** feature extractor for PPO policy.
3. **`ml/drl_trainer.py`** + **`ml_demo/train_drl_agent_demo.py`** + **`run_train_drl.py`** — train **one PPO agent per symbol** using Binance public klines.
   Training hyperparameters are maintained in `configs/ml/drl_train.yaml`.
3. **Artifacts** (per pair, filesystem-safe slug e.g. `BTC_USD`):
   - `checkpoints/drl/<PAIR>_ppo.zip`
   - `checkpoints/drl/<PAIR>_meta.json`
4. **`strategy/drl_strategy.py`** — loads pair-specific zip + meta and maps policy actions to `BUY` / `SELL` / `HOLD` in live/backtest.

## Dependencies

- Base: `gymnasium` (in `requirements.txt`)
- Training + inference: `pip install -r requirements-drl.txt` (PyTorch + Stable-Baselines3)

## Commands

Train for one product:

```bash
pip install -r requirements.txt
pip install -r requirements-drl.txt
python run_train_drl.py --config configs/ml/drl_train.yaml
```

Training logs are written to `logs/training/YYYYMMDD/drl_<HHMMSS>/train.log`.

Live / backtest:

```bash
python run_trader.py --symbols BTC/USD --strategy drl
python run_backtest.py --data-source binance --symbol BTC/USD --strategy drl --start-date 2024-01-01 --end-date 2024-02-01
```

Multi-pair live trading loads **one checkpoint per pair** automatically from `checkpoints/drl/`.

## Training log output

When running `run_train_drl.py`, Stable-Baselines3 prints periodic progress. Key fields:

| Field | Meaning |
|-------|---------|
| `time/total_timesteps` | Steps completed so far. Compare to `train.timesteps` in config (e.g. 200000) to see progress. |
| `time/fps` | Steps per second. Higher is faster (GPU usually gives higher fps). |
| `time/iterations` | Number of PPO update cycles. Each cycle = `n_steps` env steps (e.g. 512). |
| `rollout/ep_len_mean` | Mean episode length. For our env, ≈ number of bars per episode. |
| `rollout/ep_rew_mean` | Mean episode return. Higher = better policy; should trend up over training. |
| `train/loss` | Combined PPO loss (policy + value + entropy). Can be positive or negative; both are normal. |
| `train/policy_gradient_loss` | Policy gradient term. Small positive/negative swings are expected. |
| `train/value_loss` | Value function MSE. Usually small positive. |
| `train/entropy_loss` | Entropy bonus (negative). Keeps exploration; `ent_coef` scales it. |
| `train/approx_kl` | Approximate KL divergence. Keep &lt; 0.1; too large means policy changed too fast. |
| `train/clip_fraction` | Fraction of samples clipped. Occasional &gt; 0 is fine; near 1.0 often means too large step. |

**Progress**: `total_timesteps` is the main progress indicator. Training finishes when it reaches `train.timesteps`.

## Data pipeline (same as MLP)

DRL training uses `data.market_dataset.build_market_feature_dataset` — the same kline-derived features as MLP (log returns, volatility, volume, taker ratios, etc.). Configure `data.use_agg_trades` / `data.use_trades` in `configs/ml/drl_train.yaml` (default: false to avoid OOM).

## Backtest: no trades

If DRL backtest shows 0 trades:

1. **Train vs backtest mismatch** — Old checkpoints may have been trained with `buy_fraction=0.25`, `sell_fraction=0.5` (partial trades) while backtest uses all-in. Retrain with `env.buy_fraction: 1.0` and `env.sell_fraction: 1.0` in `configs/ml/drl_train.yaml`.
2. **Insufficient training** — Use at least 200k timesteps and `ent_coef: 0.01` for exploration.
3. **Data source** — Multi-feature models (`feature_dim` > 1) require `--data-source binance`; synthetic data has no features and will always HOLD.

## Device (GPU)

Set `train.device: cuda` in `configs/ml/drl_train.yaml` to force GPU, or `auto` to auto-detect. Strategy loads models with `device: auto` by default (uses CUDA if available).
