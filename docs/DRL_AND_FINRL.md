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

1. **`ml/drl_env.py`** — Gymnasium env with the same *ideas* as FinRL crypto envs: single asset, fees, discrete actions, portfolio return as step reward. Optional **reward shaping** (`holding_penalty_per_step`, `sell_execution_bonus` in `configs/ml/drl_train.yaml`) discourages endless buy-and-hold and nudges the policy to **sell and re-enter**, matching full-position backtest turnover. These terms apply **only during training**; live/backtest still use the learned policy without extra reward.
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
python run_backtest.py \
  --data-source binance \
  --symbol BTC/USD \
  --interval 1h \
  --frequency daily \
  --market spot \
  --quote-asset USDT \
  --start-date 2024-01-01 \
  --end-date 2024-02-01 \
  --strategy drl \
  --strategy-config configs/strategies/drl.yaml \
  --save-artifacts
```

Multi-pair live trading loads **one checkpoint per pair** automatically from `checkpoints/drl/`.

Auto-tune DRL:

This runs a closed loop `sample config -> train -> backtest -> evaluate -> promote best` using:

- training: `ml/drl_trainer.py::train_ppo_for_symbol`
- backtest: `backtest.py::run_backtest` (produces per-trial `steps.csv` + `backtest_chart.png`)
- promotion: copies `<PAIR>_ppo.zip` + `<PAIR>_meta.json` into `checkpoints/drl/` only if constraints are met

```bash
python -m ml.auto_tune_drl --config configs/ml/drl_autotune.yaml
```

Artifacts:

- per-trial: `logs/tuning/YYYYMMDD/drl_auto_tune/trial_XXXX/seed_YYYY/`
  - `metrics.json`
  - `trial_config.yaml`
  - `steps.csv`
  - `backtest_chart.png`
- summary:
  - `logs/tuning/YYYYMMDD/drl_auto_tune/leaderboard.csv`
  - `logs/tuning/YYYYMMDD/drl_auto_tune/best_summary.json`

## Training validation (loss/reward curves)

In `configs/ml/drl_train.yaml`, you can enable periodic validation under the `validation:` block.
During training, `ml/drl_trainer.py` will:

- run a deterministic validation episode every `validation.eval_every_steps`
- track `train/loss`, `rollout/ep_rew_mean` (when available) and validation metrics
- save curves and a best-by-validation summary into the training log folder

Generated files under `logs/training/YYYYMMDD/drl_<HHMMSS>/`:

- `train_loss_curve.png`, `train_loss_curve.csv`
- `train_reward_curve.png`, `train_reward_curve.csv`
- `validation_metrics_curve.png`, `validation_metrics_curve.csv`
- `best_by_validation_summary.json`

Validation equity/drawdown is computed from the environment internal cash/base state,
so it is not distorted by reward shaping.

## Auto-tune pre-screen notes

`configs/ml/drl_autotune.yaml` now includes a `validation:` section to enable a fast pre-screen
on a sub-window inside the backtest range.

- If the pre-screen fails, the full backtest is skipped and the trial seed `metrics.json`
  includes:
  - `stage: validation_pre_screen`
  - `passed_validation: false`
  - `failure_reasons: [...]`
- If the pre-screen passes, the full backtest runs and `score_final` blends validation and backtest ordering
  (see `objective.validation_weight`).

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
