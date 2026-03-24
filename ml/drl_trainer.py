"""DRL training pipeline (per-product PPO agent) within ml package.

Uses the data.market_dataset feature pipeline so the DRL agent sees the
same multi-factor features as the MLP model.
"""

from __future__ import annotations

import json
import shutil
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
    holding_penalty_per_step: float,
    holding_penalty_growth: float,
    sell_execution_bonus: float,
    invalid_action_penalty: float,
    realized_pnl_reward_scale: float,
    lstm_hidden_size: int,
    lstm_layers: int,
    lstm_dropout: float,
    seq_mlp_hidden_dims: str,
    account_mlp_hidden_dims: str,
    fusion_hidden_dims: str,
    policy_hidden_dims: str,
    # Validation loop (optional)
    validation_enabled: bool = False,
    validation_eval_every_steps: int = 0,
    validation_eval_episodes: int = 1,
    validation_window_start: float = 0.8,
    validation_window_end: float = 0.9,
    validation_save_best_by: str = "return_pct",
    validation_early_stop_patience_evals: int = 5,
    validation_store_best_on_validation_step: bool = True,
    validation_log_dir: str | None = None,
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
    if holding_penalty_per_step > 0 or sell_execution_bonus > 0:
        print(
            f"[drl] reward_shaping holding_penalty={holding_penalty_per_step} "
            f"growth={holding_penalty_growth} "
            f"sell_bonus={sell_execution_bonus} "
            f"invalid_penalty={invalid_action_penalty} "
            f"realized_pnl_scale={realized_pnl_reward_scale}"
        )

    # Create train/validation env slices to avoid leakage.
    train_prices = dataset.closes
    train_features = dataset.features
    val_prices = None
    val_features = None
    train_slice_end: int | None = None
    val_slice_start: int | None = None
    val_slice_end: int | None = None
    if validation_enabled:
        n = int(dataset.closes.shape[0])
        ws = float(validation_window_start)
        we = float(validation_window_end)
        if ws < 0 or ws > 1 or we < 0 or we > 1:
            raise ValueError("validation.window_start/end must be in [0,1].")
        if we <= ws:
            raise ValueError("validation.window_end must be > validation.window_start.")

        val_start_raw = int(ws * n)
        val_end_raw = int(we * n)
        # Keep `lookback` extra rows so env's internal history for the first
        # validation step has enough context.
        val_slice_start = max(0, val_start_raw - lookback)
        val_slice_end = min(n, max(val_end_raw, val_slice_start + lookback + 3))
        train_slice_end = val_slice_start

        if train_slice_end < lookback + 3:
            # Not enough data for training slice; fall back to full train.
            train_prices = dataset.closes
            train_features = dataset.features
            val_prices = dataset.closes
            val_features = dataset.features
        else:
            train_prices = dataset.closes[:train_slice_end]
            train_features = dataset.features[:train_slice_end]
            val_prices = dataset.closes[val_slice_start:val_slice_end]
            val_features = dataset.features[val_slice_start:val_slice_end]

    env = CryptoSingleAssetEnv(
        prices=train_prices,
        lookback=lookback,
        initial_cash=initial_cash,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        buy_fraction=buy_fraction,
        sell_fraction=sell_fraction,
        features=train_features,
        holding_penalty_per_step=holding_penalty_per_step,
        holding_penalty_growth=holding_penalty_growth,
        sell_execution_bonus=sell_execution_bonus,
        invalid_action_penalty=invalid_action_penalty,
        realized_pnl_reward_scale=realized_pnl_reward_scale,
    )

    val_env = None
    if validation_enabled and val_prices is not None and val_features is not None:
        val_env = CryptoSingleAssetEnv(
            prices=val_prices,
            lookback=lookback,
            initial_cash=initial_cash,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            buy_fraction=buy_fraction,
            sell_fraction=sell_fraction,
            features=val_features,
            holding_penalty_per_step=holding_penalty_per_step,
            holding_penalty_growth=holding_penalty_growth,
            sell_execution_bonus=sell_execution_bonus,
            invalid_action_penalty=invalid_action_penalty,
            realized_pnl_reward_scale=realized_pnl_reward_scale,
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
    best_model_path = None
    if validation_enabled:
        log_dir = Path(validation_log_dir) if validation_log_dir else Path(out_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = log_dir / "best_model_by_validation.zip"

    # Local callback implementation to avoid adding new modules.
    callback = None
    if validation_enabled and val_env is not None and validation_eval_every_steps > 0:
        from stable_baselines3.common.callbacks import BaseCallback
        import math

        class _ValidationCallback(BaseCallback):
            def __init__(self) -> None:
                super().__init__()
                self.last_eval_steps = 0
                self.eval_count = 0
                self.best_metric = -1e18
                self.no_improve = 0
                self.loss_curve: list[tuple[int, float]] = []
                self.reward_curve: list[tuple[int, float]] = []
                self.val_curve: list[dict[str, Any]] = []

            def _eval_once(self) -> dict[str, Any]:
                obs, _ = val_env.reset()
                # Reward is shaped; do NOT reconstruct equity from reward.
                equity = float(val_env._equity(val_env._t))
                peak = equity
                max_dd = 0.0
                trades = 0
                buy_trades = 0
                sell_trades = 0
                action_counts = [0, 0, 0]  # [HOLD, BUY, SELL]
                step_count = 0
                done = False
                while not done:
                    cash_ratio = float(obs[-2])
                    pos_ratio = float(obs[-1])
                    action, _state = self.model.predict(obs, deterministic=True)
                    a = int(action)
                    if a in (0, 1, 2):
                        action_counts[a] += 1
                    executed_buy = a == 1 and cash_ratio > 1e-12
                    executed_sell = a == 2 and pos_ratio > 1e-12
                    if executed_buy:
                        buy_trades += 1
                        trades += 1
                    if executed_sell:
                        sell_trades += 1
                        trades += 1

                    obs, reward, terminated, truncated, _info = val_env.step(a)
                    t2 = min(int(val_env._t), int(len(val_env.prices) - 1))
                    equity = float(val_env._equity(t2))
                    peak = max(peak, equity)
                    dd = (peak - equity) / peak if peak > 0 else 0.0
                    max_dd = max(max_dd, dd)
                    step_count += 1
                    done = bool(terminated or truncated)

                return_pct = equity / float(val_env.initial_cash) - 1.0
                # score uses the same default style as auto_tune to keep behavior consistent.
                score = return_pct - max_dd + 0.1 * math.log1p(max(trades, 0))
                if step_count > 0:
                    hold_ratio = float(action_counts[0]) / float(step_count)
                    buy_ratio = float(action_counts[1]) / float(step_count)
                    sell_ratio = float(action_counts[2]) / float(step_count)
                    effective_trading_rate = float(trades) / float(step_count)
                else:
                    hold_ratio = buy_ratio = sell_ratio = effective_trading_rate = 0.0
                return {
                    "val_return_pct": float(return_pct),
                    "val_max_drawdown_pct": float(max_dd),
                    "val_trades": int(trades),
                    "val_buy_trades": int(buy_trades),
                    "val_sell_trades": int(sell_trades),
                    "val_score": float(score),
                    "val_hold_ratio": float(hold_ratio),
                    "val_buy_ratio": float(buy_ratio),
                    "val_sell_ratio": float(sell_ratio),
                    "val_effective_trading_rate": float(effective_trading_rate),
                    "val_steps": int(step_count),
                }

            def _on_step(self) -> bool:
                # Collect training curves if SB3 exposes them.
                try:
                    ntv = getattr(self.model.logger, "name_to_value", {})
                    if "train/loss" in ntv:
                        self.loss_curve.append((int(self.num_timesteps), float(ntv["train/loss"])))
                    if "rollout/ep_rew_mean" in ntv:
                        self.reward_curve.append(
                            (int(self.num_timesteps), float(ntv["rollout/ep_rew_mean"]))
                        )
                except Exception:
                    pass

                if int(self.num_timesteps) - int(self.last_eval_steps) >= validation_eval_every_steps:
                    self.last_eval_steps = int(self.num_timesteps)
                    self.eval_count += 1

                    # Multi-episode eval (usually 1 because fixed window is deterministic).
                    vals = [self._eval_once() for _ in range(max(1, validation_eval_episodes))]
                    # Aggregate.
                    avg_return = float(sum(x["val_return_pct"] for x in vals) / len(vals))
                    avg_max_dd = float(max(x["val_max_drawdown_pct"] for x in vals))
                    avg_trades = int(sum(x["val_trades"] for x in vals) / len(vals))
                    avg_score = float(sum(x["val_score"] for x in vals) / len(vals))
                    avg_hold_ratio = float(sum(x["val_hold_ratio"] for x in vals) / len(vals))
                    avg_buy_ratio = float(sum(x["val_buy_ratio"] for x in vals) / len(vals))
                    avg_sell_ratio = float(sum(x["val_sell_ratio"] for x in vals) / len(vals))
                    avg_effective_trading_rate = float(
                        sum(x["val_effective_trading_rate"] for x in vals) / len(vals)
                    )

                    if validation_save_best_by == "return_pct":
                        metric = avg_return
                    else:
                        metric = avg_score

                    self.val_curve.append(
                        {
                            "num_timesteps": int(self.num_timesteps),
                            "val_return_pct": avg_return,
                            "val_max_drawdown_pct": avg_max_dd,
                            "val_trades": avg_trades,
                            "val_score": avg_score,
                            "val_hold_ratio": avg_hold_ratio,
                            "val_buy_ratio": avg_buy_ratio,
                            "val_sell_ratio": avg_sell_ratio,
                            "val_effective_trading_rate": avg_effective_trading_rate,
                            "metric": float(metric),
                        }
                    )

                    if metric > self.best_metric:
                        self.best_metric = metric
                        self.no_improve = 0
                        if best_model_path is not None:
                            self.model.save(str(best_model_path))
                    else:
                        self.no_improve += 1

                    if self.no_improve >= validation_early_stop_patience_evals:
                        return False

                return True

        callback = _ValidationCallback()

    model.learn(total_timesteps=timesteps, callback=callback)

    # Plot curves and/or overwrite the final model with the best-by-validation model.
    if validation_enabled and callback is not None:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            log_dir = Path(validation_log_dir) if validation_log_dir else Path(out_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Training curves
            if getattr(callback, "loss_curve", None):
                xs = [x for x, _ in callback.loss_curve]
                ys = [y for _, y in callback.loss_curve]
                try:
                    import csv as _csv

                    with (log_dir / "train_loss_curve.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as f:
                        w = _csv.DictWriter(f, fieldnames=["num_timesteps", "loss"])
                        w.writeheader()
                        for (step, loss) in callback.loss_curve:
                            w.writerow({"num_timesteps": step, "loss": loss})
                except Exception:
                    pass
                plt.figure(figsize=(12, 4))
                plt.plot(xs, ys, linewidth=1.0)
                plt.xlabel("num_timesteps")
                plt.ylabel("train/loss")
                plt.title("Training loss curve")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(log_dir / "train_loss_curve.png", dpi=150)
                plt.close()

            if getattr(callback, "reward_curve", None):
                xs = [x for x, _ in callback.reward_curve]
                ys = [y for _, y in callback.reward_curve]
                try:
                    import csv as _csv

                    with (log_dir / "train_reward_curve.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as f:
                        w = _csv.DictWriter(f, fieldnames=["num_timesteps", "ep_rew_mean"])
                        w.writeheader()
                        for (step, r) in callback.reward_curve:
                            w.writerow(
                                {"num_timesteps": step, "ep_rew_mean": r}
                            )
                except Exception:
                    pass
                plt.figure(figsize=(12, 4))
                plt.plot(xs, ys, linewidth=1.0)
                plt.xlabel("num_timesteps")
                plt.ylabel("rollout/ep_rew_mean")
                plt.title("Training reward curve")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(log_dir / "train_reward_curve.png", dpi=150)
                plt.close()

            if getattr(callback, "val_curve", None):
                # Validation metrics curve
                xs = [x["num_timesteps"] for x in callback.val_curve]
                rets = [x["val_return_pct"] for x in callback.val_curve]
                dds = [x["val_max_drawdown_pct"] for x in callback.val_curve]
                trades = [x["val_trades"] for x in callback.val_curve]

                try:
                    import csv as _csv

                    with (log_dir / "validation_metrics_curve.csv").open(
                        "w", encoding="utf-8", newline=""
                    ) as f:
                        w = _csv.DictWriter(
                            f,
                            fieldnames=[
                                "num_timesteps",
                                "val_return_pct",
                                "val_max_drawdown_pct",
                                "val_trades",
                                "val_score",
                                "val_hold_ratio",
                                "val_buy_ratio",
                                "val_sell_ratio",
                                "val_effective_trading_rate",
                                "metric",
                            ],
                        )
                        w.writeheader()
                        for row in callback.val_curve:
                            w.writerow(row)
                except Exception:
                    pass

                fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                axs[0].plot(xs, rets, label="val_return_pct")
                axs[0].set_ylabel("return_pct")
                axs[0].grid(True, alpha=0.3)
                axs[0].legend(loc="best")

                axs[1].plot(xs, dds, label="val_max_drawdown_pct")
                axs[1].plot(xs, trades, label="val_trades")
                axs[1].set_ylabel("drawdown / trades")
                axs[1].grid(True, alpha=0.3)
                axs[1].legend(loc="best")

                plt.xlabel("num_timesteps")
                plt.suptitle("Validation metrics")
                plt.tight_layout()
                plt.savefig(log_dir / "validation_metrics_curve.png", dpi=150)
                plt.close(fig)

            # Save best-by-validation json summary.
            try:
                val_summary_path = log_dir / "best_by_validation_summary.json"
                if callback.val_curve:
                    best = max(callback.val_curve, key=lambda x: x["metric"])
                else:
                    best = {}
                val_summary_path.write_text(
                    json.dumps({"best": best, "val_curve": callback.val_curve}, indent=2, default=str),
                    encoding="utf-8",
                )
            except Exception:
                pass
        except Exception:
            # Curves are non-critical; training should still be saved.
            pass

    slug = pair_to_slug(symbol)
    out = Path(out_dir)
    model_path = out / f"{slug}_ppo.zip"
    meta_path = out / f"{slug}_meta.json"
    # If validation is enabled, overwrite exported artifacts with the best-by-validation model.
    best_extra: dict[str, Any] = {}
    if validation_enabled and callback is not None and getattr(callback, "val_curve", None):
        try:
            best_row = max(callback.val_curve, key=lambda x: x.get("metric", x.get("val_score", -1e18)))
            best_extra = {
                "best_validation": best_row,
            }
        except Exception:
            best_extra = {}

    if validation_enabled and best_model_path is not None and Path(best_model_path).is_file():
        shutil.copy2(str(best_model_path), str(model_path))
    else:
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
            "holding_penalty_per_step": holding_penalty_per_step,
            "holding_penalty_growth": holding_penalty_growth,
            "sell_execution_bonus": sell_execution_bonus,
            "invalid_action_penalty": invalid_action_penalty,
            "realized_pnl_reward_scale": realized_pnl_reward_scale,
            "use_agg_trades": use_agg_trades,
            "use_trades": use_trades,
            "vol_window": vol_window,
            "validation": {
                "enabled": validation_enabled,
                "best_extra": best_extra,
                "slices": {
                    "train_slice_end": train_slice_end,
                    "val_slice_start": val_slice_start,
                    "val_slice_end": val_slice_end,
                },
            },
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
