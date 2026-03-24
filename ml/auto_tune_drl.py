"""DRL auto-tune runner.

This script implements a closed loop:
  sample config -> train PPO -> backtest -> evaluate -> promote best model.

It is designed to reuse the existing project building blocks:
  - `ml/drl_trainer.py::train_ppo_for_symbol`
  - `backtest.py::run_backtest`
  - `strategy/drl_strategy.py::DRLSb3Strategy`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.config_loader import get_block, load_yaml_config
from ml.drl_trainer import train_ppo_for_symbol
from ml.drl_utils import pair_to_slug

# Project root import path (matches run_train_drl.py and other scripts).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest import run_backtest  # noqa: E402
from data.market_dataset import build_market_feature_dataset  # noqa: E402
from strategy.drl_strategy import DRLSb3Strategy  # noqa: E402

DEBUG_LOG_PATH = Path("/userhome/cs5/u3664760/.cursor/debug-0a05a9.log")


def _append_debug_ndjson(
    *,
    runId: str,
    hypothesisId: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    # Keep this debug logging best-effort; never block the tuning process.
    try:
        DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessionId": "0a05a9",
            "runId": runId,
            "hypothesisId": hypothesisId,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass

def _set_by_dotted_path(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur: Any = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _sample_search_space(rng: random.Random, search_space: dict[str, Any]) -> dict[str, Any]:
    """Sample one set of overrides from `search_space`.

    Supported shapes:
      - `param.path: [v1, v2, ...]`  -> random choice from list
      - (optional extension) `param.path: {min, max}` -> uniform float
    """
    overrides: dict[str, Any] = {}
    for k, spec in search_space.items():
        if isinstance(spec, list) and spec:
            overrides[k] = rng.choice(spec)
        elif isinstance(spec, dict) and "min" in spec and "max" in spec:
            lo = float(spec["min"])
            hi = float(spec["max"])
            overrides[k] = rng.uniform(lo, hi)
        else:
            raise ValueError(f"Unsupported search_space spec for '{k}': {spec!r}")
    return overrides


def _make_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class Constraints:
    min_trades: int
    min_return_pct: float
    max_drawdown_pct: float
    min_sell_trades: int = 0
    max_single_action_ratio: float | None = None
    min_effective_trading_rate: float = 0.0

    def passed(
        self,
        *,
        trades: int,
        sell_trades: int = 0,
        return_pct: float,
        max_drawdown_pct: float,
        action_ratios: list[float] | None = None,
        effective_trading_rate: float = 0.0,
    ) -> bool:
        ok = (
            trades >= self.min_trades
            and return_pct >= self.min_return_pct
            and max_drawdown_pct <= self.max_drawdown_pct
        )
        if not ok:
            return False
        if self.min_sell_trades and sell_trades < self.min_sell_trades:
            return False
        if self.max_single_action_ratio is not None and action_ratios:
            if max(action_ratios) > float(self.max_single_action_ratio):
                return False
        if self.min_effective_trading_rate and effective_trading_rate < self.min_effective_trading_rate:
            return False
        return True


def _compute_objective(
    *, return_pct: float, max_drawdown_pct: float, trades: int, alpha: float, beta: float
) -> float:
    # All are already in "fraction" units (e.g. 0.05 means 5%).
    # log1p for trades prevents letting a single huge trades value dominate.
    return return_pct - alpha * max_drawdown_pct + beta * math.log1p(max(trades, 0))


def _prepare_backtest_data(backtest_cfg: dict[str, Any]) -> tuple[list[float], Any, list[int] | None]:
    data_source = str(backtest_cfg.get("data_source", "binance")).lower()
    if data_source != "binance":
        raise ValueError(
            "auto_tune_drl currently supports backtest.data_source=binance only. "
            f"Got: {data_source}"
        )

    symbol = str(backtest_cfg["symbol"])
    interval = str(backtest_cfg.get("interval", "1h"))
    frequency = str(backtest_cfg.get("frequency", "daily"))
    market = str(backtest_cfg.get("market", "spot"))
    quote_asset = str(backtest_cfg.get("quote_asset", "USDT"))
    start_date = str(backtest_cfg.get("start_date", ""))
    end_date = str(backtest_cfg.get("end_date", ""))
    limit = int(backtest_cfg.get("limit", 0))
    cache_dir = str(backtest_cfg.get("cache_dir", "data_cache/binance_public_data"))

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
        use_agg_trades=False,
        use_trades=False,
    )
    prices = dataset.closes.tolist()
    features = dataset.features
    time_ms_list = [int(x) for x in dataset.open_times.tolist()]
    return prices, features, time_ms_list


def _eval_strategy_on_prices_window(
    *,
    strategy: Any,
    prices: list[float],
    features: Any,
    start_idx: int,
    end_idx: int,
    initial_cash: float,
    buy_cost_pct: float,
    sell_cost_pct: float,
    buy_fraction: float,
    sell_fraction: float,
    holding_penalty_per_step: float,
    holding_penalty_growth: float,
    sell_execution_bonus: float,
    invalid_action_penalty: float,
    realized_pnl_reward_scale: float,
) -> dict[str, Any]:
    """Fast pre-screen evaluation on a price window.

    Uses environment internal state to compute equity/drawdown (reward shaping
    makes reward != pure equity return ratio).
    """
    from ml.drl_env import CryptoSingleAssetEnv

    if end_idx <= start_idx:
        raise ValueError("Invalid window: end_idx must be > start_idx")

    lookback = int(strategy.lookback)
    prices_slice = np.asarray(prices[start_idx:end_idx], dtype=np.float64)
    features_slice = None
    if features is not None:
        features_slice = np.asarray(features[start_idx:end_idx], dtype=np.float64)

    env = CryptoSingleAssetEnv(
        prices=prices_slice,
        lookback=lookback,
        initial_cash=initial_cash,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        buy_fraction=buy_fraction,
        sell_fraction=sell_fraction,
        features=features_slice if strategy.feature_dim > 1 else None,
        holding_penalty_per_step=holding_penalty_per_step,
        holding_penalty_growth=holding_penalty_growth,
        sell_execution_bonus=sell_execution_bonus,
        invalid_action_penalty=invalid_action_penalty,
        realized_pnl_reward_scale=realized_pnl_reward_scale,
    )

    obs, _ = env.reset()
    equity = float(env._equity(env._t))
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

        action, _state = strategy._model.predict(obs, deterministic=True)
        # Convert SB3 discrete action into a stable python int.
        # Sometimes SB3 returns a 0-d numpy array (shape=()) for discrete actions.
        a_arr = np.asarray(action)
        if a_arr.ndim == 0:
            a = int(a_arr.item())
        else:
            a = int(a_arr.reshape(-1)[0])

        # #region agent log
        if step_count == 0:
            _append_debug_ndjson(
                runId="pre-fix-or-verification",
                hypothesisId="H2_action_conversion_0d_array",
                location="auto_tune_drl.py:_eval_strategy_on_prices_window",
                message="action conversion",
                data={
                    "action_type": str(type(action)),
                    "action_ndim": int(a_arr.ndim),
                    "action_shape": tuple(getattr(a_arr, "shape", ())),
                    "a": a,
                },
            )
        # #endregion

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

        obs, reward, terminated, truncated, _info = env.step(a)
        t2 = min(int(env._t), int(len(env.prices) - 1))
        equity = float(env._equity(t2))
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        step_count += 1
        done = bool(terminated or truncated)

    return_pct = equity / float(env.initial_cash) - 1.0
    effective_trading_rate = float(trades) / float(step_count) if step_count > 0 else 0.0
    if step_count > 0:
        hold_ratio = float(action_counts[0]) / float(step_count)
        buy_ratio = float(action_counts[1]) / float(step_count)
        sell_ratio = float(action_counts[2]) / float(step_count)
    else:
        hold_ratio = buy_ratio = sell_ratio = 0.0

    return {
        "val_return_pct": float(return_pct),
        "val_max_drawdown_pct": float(max_dd),
        "val_trades": int(trades),
        "val_buy_trades": int(buy_trades),
        "val_sell_trades": int(sell_trades),
        "val_hold_ratio": float(hold_ratio),
        "val_buy_ratio": float(buy_ratio),
        "val_sell_ratio": float(sell_ratio),
        "val_effective_trading_rate": float(effective_trading_rate),
        "val_steps": int(step_count),
    }


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PyYAML may parse "YYYY-MM-DD" into `datetime.date`. json can't serialize it by default.
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _to_serializable(obj: Any) -> Any:
    """Convert nested objects into JSON/YAML friendly primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    return str(obj)


def _save_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_serializable(obj)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _copy_with_backup(src: Path, dst: Path, backup_existing: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and backup_existing:
        backup_dir = dst.parent / "_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        dst_bak = backup_dir / f"{dst.name}.{ts}.bak"
        shutil.copy2(dst, dst_bak)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="DRL auto-tune: train -> backtest -> promote best")
    parser.add_argument("--config", type=str, default="configs/ml/drl_autotune.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Only sample and prepare directories, do not train/backtest.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    baseline_path = str(cfg.get("baseline_train_config", "configs/ml/drl_train.yaml"))
    baseline = load_yaml_config(baseline_path)

    search_space = get_block(cfg, "search_space")
    if not search_space:
        raise ValueError("drl_autotune.yaml missing search_space")

    constraints_block = get_block(cfg, "constraints")
    constraints = Constraints(
        min_trades=int(constraints_block.get("min_trades", 0)),
        min_sell_trades=int(constraints_block.get("min_sell_trades", 0)),
        min_return_pct=float(constraints_block.get("min_return_pct", 0.0)),
        max_drawdown_pct=float(constraints_block.get("max_drawdown_pct", 1e9)),
        max_single_action_ratio=(
            float(constraints_block["max_single_action_ratio"])
            if "max_single_action_ratio" in constraints_block
            else None
        ),
        min_effective_trading_rate=float(
            constraints_block.get("min_effective_trading_rate", 0.0)
        ),
    )

    objective_block = get_block(cfg, "objective")
    alpha = float(objective_block.get("alpha", 1.0))
    beta = float(objective_block.get("beta", 0.1))
    validation_weight = float(objective_block.get("validation_weight", 0.0))

    budget = get_block(cfg, "budget")
    max_trials = int(budget.get("max_trials", 10))
    patience = int(budget.get("patience", 3))
    seed_list = budget.get("seed_list") or [budget.get("seed", 42)]
    seed_list = [int(x) for x in seed_list]

    backtest_cfg = get_block(cfg, "backtest")
    validation_cfg = get_block(cfg, "validation")
    artifact_cfg = get_block(cfg, "artifact")
    promotion_cfg = get_block(cfg, "promotion")

    base_dir = Path(str(artifact_cfg.get("base_dir", "logs/tuning")))
    run_name = str(artifact_cfg.get("run_name", "drl_auto_tune"))
    trial_models_subdir = str(artifact_cfg.get("trial_models_subdir", "models"))

    bt_log_steps = bool(backtest_cfg.get("log_steps", True))
    bt_log_chart = bool(backtest_cfg.get("log_chart", True))

    promote_enabled = bool(promotion_cfg.get("enabled", True))
    target_model_dir = Path(str(promotion_cfg.get("target_model_dir", "checkpoints/drl")))
    backup_existing = bool(promotion_cfg.get("backup_existing", True))

    val_pre_screener_enabled = bool(validation_cfg.get("enabled", False))
    val_pre_screen_window_start = float(validation_cfg.get("window_start", 0.0))
    val_pre_screen_window_end = float(validation_cfg.get("window_end", 0.5))

    # Backtest data prepared once for the whole tuning run.
    backtest_prices, backtest_features, backtest_time_ms = _prepare_backtest_data(backtest_cfg)

    rng = random.Random(int(budget.get("seed", 42)))

    run_root = base_dir / time.strftime("%Y%m%d", time.gmtime()) / run_name
    _make_dir(run_root)

    leaderboard_rows: list[dict[str, Any]] = []

    best: dict[str, Any] | None = None
    best_score = -1e18
    no_improve = 0

    # If baseline config data.symbol mismatches backtest symbol, we still allow it,
    # but the promoted model naming will use backtest.symbol (pair_to_slug).
    backtest_symbol = str(backtest_cfg["symbol"]).strip().upper()
    baseline_symbol = str(get_block(baseline, "data").get("symbol", backtest_cfg["symbol"])).strip().upper()

    if baseline_symbol != backtest_symbol:
        print(
            f"[auto_tune] WARNING: baseline_train_config.data.symbol={baseline_symbol} "
            f"!= backtest.symbol={backtest_symbol}. Backtest uses {backtest_symbol}.",
            flush=True,
        )

    for trial_idx in range(max_trials):
        overrides = _sample_search_space(rng, search_space)
        trial_seed = seed_list[0] if len(seed_list) == 1 else None

        trial_dir = run_root / f"trial_{trial_idx:04d}"
        seed_dirs: dict[int, Path] = {}
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        _make_dir(trial_dir)

        # Persist sampled overrides for reproducibility.
        _save_json(trial_dir / "sampled_overrides.json", overrides)

        # Apply overrides to a copy of baseline train config.
        trial_train_cfg = copy.deepcopy(baseline)
        for dotted_key, v in overrides.items():
            _set_by_dotted_path(trial_train_cfg, dotted_key, v)

        # We'll train each seed and backtest each seed (if seed_list > 1).
        seed_results: list[dict[str, Any]] = []
        seed_scores: list[float] = []

        for seed in seed_list:
            seed_dir = trial_dir / f"seed_{seed}"
            seed_dirs[seed] = seed_dir
            model_dir = seed_dir / trial_models_subdir
            _make_dir(model_dir)

            # Extract blocks from trial train config.
            data_cfg = get_block(trial_train_cfg, "data")
            env_cfg = get_block(trial_train_cfg, "env")
            arch_cfg = get_block(trial_train_cfg, "architecture")
            ppo_cfg = get_block(trial_train_cfg, "ppo")
            train_cfg = get_block(trial_train_cfg, "train")
            out_cfg = get_block(trial_train_cfg, "out")

            symbol = str(data_cfg.get("symbol", backtest_cfg["symbol"]))
            interval = str(data_cfg.get("interval", "1h"))
            frequency = str(data_cfg.get("frequency", "daily"))
            start_date = str(data_cfg.get("start_date", ""))
            end_date = str(data_cfg.get("end_date", ""))
            limit = int(data_cfg.get("limit", 0))
            market = str(data_cfg.get("market", "spot"))
            quote_asset = str(data_cfg.get("quote_asset", "USDT"))
            cache_dir = str(data_cfg.get("cache_dir", "data_cache/binance_public_data"))
            use_agg_trades = bool(data_cfg.get("use_agg_trades", False))
            use_trades = bool(data_cfg.get("use_trades", False))
            vol_window = int(data_cfg.get("vol_window", 20))

            # Env / reward.
            lookback = int(env_cfg.get("lookback", 20))
            initial_cash = float(env_cfg.get("initial_cash", 10_000.0))
            buy_cost_pct = float(env_cfg.get("buy_cost_pct", 0.0008))
            sell_cost_pct = float(env_cfg.get("sell_cost_pct", 0.0008))
            buy_fraction = float(env_cfg.get("buy_fraction", 1.0))
            sell_fraction = float(env_cfg.get("sell_fraction", 1.0))
            holding_penalty_per_step = float(env_cfg.get("holding_penalty_per_step", 0.0))
            holding_penalty_growth = float(env_cfg.get("holding_penalty_growth", 0.0))
            sell_execution_bonus = float(env_cfg.get("sell_execution_bonus", 0.0))
            invalid_action_penalty = float(env_cfg.get("invalid_action_penalty", 0.0))
            realized_pnl_reward_scale = float(env_cfg.get("realized_pnl_reward_scale", 0.0))

            # Architecture.
            lstm_hidden_size = int(arch_cfg.get("lstm_hidden_size", 64))
            lstm_layers = int(arch_cfg.get("lstm_layers", 1))
            lstm_dropout = float(arch_cfg.get("lstm_dropout", 0.0))
            seq_mlp_hidden_dims = str(arch_cfg.get("seq_mlp_hidden_dims", "64"))
            account_mlp_hidden_dims = str(arch_cfg.get("account_mlp_hidden_dims", "16"))
            fusion_hidden_dims = str(arch_cfg.get("fusion_hidden_dims", "64"))
            policy_hidden_dims = str(arch_cfg.get("policy_hidden_dims", "64,64"))

            # PPO.
            timesteps = int(train_cfg.get("timesteps", 200_000))
            device = str(train_cfg.get("device", "auto"))
            learning_rate = float(ppo_cfg.get("learning_rate", 3e-4))
            n_steps = int(ppo_cfg.get("n_steps", 512))
            batch_size = int(ppo_cfg.get("batch_size", 64))
            gamma = float(ppo_cfg.get("gamma", 0.99))
            gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
            ent_coef = float(ppo_cfg.get("ent_coef", 0.01))
            clip_range = float(ppo_cfg.get("clip_range", 0.2))

            # Save a full snapshot for this seed.
            trial_seed_snapshot = {
                "trial_idx": trial_idx,
                "seed": seed,
                "seed_dir": str(seed_dir),
                "overrides": overrides,
                "train_cfg_snapshot": trial_train_cfg,
            }
            _save_yaml(seed_dir / "trial_config.yaml", trial_seed_snapshot)
            _save_json(seed_dir / "trial_config.json", trial_seed_snapshot)

            if args.dry_run:
                print(f"[auto_tune] Dry-run: skip training for trial={trial_idx}, seed={seed}", flush=True)
                seed_results.append(
                    {
                        "seed": seed,
                        "return_pct": float("nan"),
                        "max_drawdown_pct": float("nan"),
                        "trades": 0,
                        "score": float("-inf"),
                        "passed": False,
                    }
                )
                continue

            print(
                f"[auto_tune] Trial {trial_idx}/{max_trials-1} | seed={seed} | "
                f"timesteps={timesteps} lr={learning_rate} holding_penalty={holding_penalty_per_step}",
                flush=True,
            )

            # Train to a seed-specific models dir (so we can promote later).
            _, _ = train_ppo_for_symbol(
                symbol=str(symbol),
                interval=str(interval),
                frequency=str(frequency),
                start_date=str(start_date),
                end_date=str(end_date),
                limit=int(limit),
                market=str(market),
                quote_asset=str(quote_asset),
                cache_dir=str(cache_dir),
                use_agg_trades=use_agg_trades,
                use_trades=use_trades,
                vol_window=vol_window,
                lookback=lookback,
                timesteps=timesteps,
                seed=seed,
                device=device,
                out_dir=str(model_dir),
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                clip_range=clip_range,
                buy_cost_pct=buy_cost_pct,
                sell_cost_pct=sell_cost_pct,
                buy_fraction=buy_fraction,
                sell_fraction=sell_fraction,
                initial_cash=initial_cash,
                holding_penalty_per_step=holding_penalty_per_step,
                holding_penalty_growth=holding_penalty_growth,
                sell_execution_bonus=sell_execution_bonus,
                invalid_action_penalty=invalid_action_penalty,
                realized_pnl_reward_scale=realized_pnl_reward_scale,
                lstm_hidden_size=lstm_hidden_size,
                lstm_layers=lstm_layers,
                lstm_dropout=lstm_dropout,
                seq_mlp_hidden_dims=seq_mlp_hidden_dims,
                account_mlp_hidden_dims=account_mlp_hidden_dims,
                fusion_hidden_dims=fusion_hidden_dims,
                policy_hidden_dims=policy_hidden_dims,
            )

            # Backtest using the trained model.
            strategy = DRLSb3Strategy(
                pair=str(backtest_cfg["symbol"]),
                model_dir=str(model_dir),
                deterministic=True,
                device="cpu",
            )

            # Stage A: fast validation pre-screen on a sub-window.
            seed_val_metrics: dict[str, Any] = {}
            passed_validation = True
            score_val = float("-inf")

            if val_pre_screener_enabled:
                n_bars = len(backtest_prices)
                lookback = int(strategy.lookback)
                start_raw = int(val_pre_screen_window_start * n_bars)
                end_raw = int(val_pre_screen_window_end * n_bars)
                val_start_idx = max(0, start_raw - lookback)
                val_end_idx = min(n_bars, max(end_raw, val_start_idx + lookback + 3))

                seed_val_metrics = _eval_strategy_on_prices_window(
                    strategy=strategy,
                    prices=backtest_prices,
                    features=backtest_features,
                    start_idx=val_start_idx,
                    end_idx=val_end_idx,
                    initial_cash=initial_cash,
                    buy_cost_pct=buy_cost_pct,
                    sell_cost_pct=sell_cost_pct,
                    buy_fraction=buy_fraction,
                    sell_fraction=sell_fraction,
                    holding_penalty_per_step=holding_penalty_per_step,
                    holding_penalty_growth=holding_penalty_growth,
                    sell_execution_bonus=sell_execution_bonus,
                    invalid_action_penalty=invalid_action_penalty,
                    realized_pnl_reward_scale=realized_pnl_reward_scale,
                )

                val_trades = int(seed_val_metrics["val_trades"])
                val_sell_trades = int(seed_val_metrics.get("val_sell_trades", 0))
                val_return_pct = float(seed_val_metrics["val_return_pct"])
                val_max_dd = float(seed_val_metrics["val_max_drawdown_pct"])
                action_ratios = [
                    float(seed_val_metrics.get("val_hold_ratio", 0.0)),
                    float(seed_val_metrics.get("val_buy_ratio", 0.0)),
                    float(seed_val_metrics.get("val_sell_ratio", 0.0)),
                ]
                effective_rate = float(seed_val_metrics.get("val_effective_trading_rate", 0.0))

                passed_validation = constraints.passed(
                    trades=val_trades,
                    sell_trades=val_sell_trades,
                    return_pct=val_return_pct,
                    max_drawdown_pct=val_max_dd,
                    action_ratios=action_ratios,
                    effective_trading_rate=effective_rate,
                )

                score_val = _compute_objective(
                    return_pct=val_return_pct,
                    max_drawdown_pct=val_max_dd,
                    trades=val_trades,
                    alpha=alpha,
                    beta=beta,
                )

                if not passed_validation:
                    failure_reasons: list[str] = []
                    if val_trades < constraints.min_trades:
                        failure_reasons.append(
                            f"val_trades {val_trades} < min_trades {constraints.min_trades}"
                        )
                    if val_return_pct < constraints.min_return_pct:
                        failure_reasons.append(
                            f"val_return {val_return_pct:.6f} < min_return {constraints.min_return_pct:.6f}"
                        )
                    if val_max_dd > constraints.max_drawdown_pct:
                        failure_reasons.append(
                            f"val_max_dd {val_max_dd:.6f} > max_drawdown {constraints.max_drawdown_pct:.6f}"
                        )
                    if constraints.min_sell_trades and val_sell_trades < constraints.min_sell_trades:
                        failure_reasons.append(
                            f"val_sell_trades {val_sell_trades} < min_sell_trades {constraints.min_sell_trades}"
                        )
                    if constraints.max_single_action_ratio is not None:
                        if max(action_ratios) > float(constraints.max_single_action_ratio):
                            failure_reasons.append(
                                f"max_single_action_ratio {max(action_ratios):.3f} > {constraints.max_single_action_ratio:.3f}"
                            )
                    if constraints.min_effective_trading_rate and effective_rate < constraints.min_effective_trading_rate:
                        failure_reasons.append(
                            f"effective_trading_rate {effective_rate:.6f} < min_effective_trading_rate {constraints.min_effective_trading_rate:.6f}"
                        )

                    metrics = {
                        "seed": seed,
                        "stage": "validation_pre_screen",
                        "passed_validation": False,
                        "failure_reasons": failure_reasons,
                        "val_return_pct": val_return_pct,
                        "val_max_drawdown_pct": val_max_dd,
                        "val_trades": val_trades,
                        "val_sell_trades": val_sell_trades,
                        "val_hold_ratio": float(seed_val_metrics.get("val_hold_ratio", 0.0)),
                        "val_buy_ratio": float(seed_val_metrics.get("val_buy_ratio", 0.0)),
                        "val_sell_ratio": float(seed_val_metrics.get("val_sell_ratio", 0.0)),
                        "val_effective_trading_rate": float(seed_val_metrics.get("val_effective_trading_rate", 0.0)),
                        "score_val": score_val,
                        "return_pct": float("nan"),
                        "max_drawdown_pct": float("nan"),
                        "trades": 0,
                        "score_backtest": float("-inf"),
                        "score_final": float("-inf"),
                        "passed": False,
                    }
                    _save_json(seed_dir / "metrics.json", metrics)
                    seed_results.append(metrics)
                    seed_scores.append(float("-inf"))
                    continue

            # Stage B: full backtest only if passed validation pre-screen.
            step_log_path = seed_dir / "steps.csv" if bt_log_steps else None
            plot_path = seed_dir / "backtest_chart.png" if bt_log_chart else None

            result = run_backtest(
                prices=backtest_prices,
                strategy=strategy,
                fee_rate=buy_cost_pct,
                initial_quote=initial_cash,
                features=backtest_features,
                step_log_path=step_log_path,
                plot_path=plot_path,
                time_ms=backtest_time_ms,
            )

            return_pct = float(result.return_pct)
            max_dd = float(result.max_drawdown_pct)
            trades = int(result.trades)

            score_backtest = _compute_objective(
                return_pct=return_pct,
                max_drawdown_pct=max_dd,
                trades=trades,
                alpha=alpha,
                beta=beta,
            )
            if val_pre_screener_enabled:
                seed_score = (1.0 - validation_weight) * score_backtest + validation_weight * score_val
            else:
                seed_score = score_backtest

            passed = constraints.passed(
                trades=trades,
                return_pct=return_pct,
                max_drawdown_pct=max_dd,
            )

            metrics = {
                "seed": seed,
                "stage": "backtest",
                "passed_validation": bool(passed_validation),
                "val_return_pct": float(seed_val_metrics.get("val_return_pct", float("nan"))),
                "val_max_drawdown_pct": float(seed_val_metrics.get("val_max_drawdown_pct", float("nan"))),
                "val_trades": int(seed_val_metrics.get("val_trades", 0)),
                "val_sell_trades": int(seed_val_metrics.get("val_sell_trades", 0)),
                "score_val": float(score_val),
                "return_pct": return_pct,
                "max_drawdown_pct": max_dd,
                "trades": trades,
                "score_backtest": float(score_backtest),
                "score_final": float(seed_score),
                "score": float(seed_score),
                "passed": bool(passed),
                "start_equity": result.start_equity,
                "end_equity": result.end_equity,
            }
            _save_json(seed_dir / "metrics.json", metrics)

            seed_results.append(metrics)
            seed_scores.append(float(seed_score))

        # Aggregate seeds into one trial metric.
        valid_seed_results = [x for x in seed_results if isinstance(x.get("return_pct"), (int, float)) and not math.isnan(x.get("return_pct"))]
        if not valid_seed_results:
            # In dry-run or training failures.
            trial_return = float("nan")
            trial_max_dd = float("nan")
            trial_trades = 0
            trial_score = float("-inf")
            passed = False
            best_seed_score = float("-inf")
            best_seed_dir = None
        else:
            trial_return = float(sum(x["return_pct"] for x in valid_seed_results) / len(valid_seed_results))
            trial_max_dd = float(max(x["max_drawdown_pct"] for x in valid_seed_results))
            trial_trades = int(sum(x["trades"] for x in valid_seed_results) / len(valid_seed_results))
            trial_score = float(
                sum(float(x.get("score_final", x.get("score", float("-inf")))) for x in valid_seed_results)
                / len(valid_seed_results)
            )
            passed = constraints.passed(
                trades=trial_trades,
                return_pct=trial_return,
                max_drawdown_pct=trial_max_dd,
            )

            # Choose best seed to promote (highest per-seed score).
            best_seed_idx = int(
                max(
                    range(len(valid_seed_results)),
                    key=lambda i: valid_seed_results[i].get("score_final", valid_seed_results[i]["score"]),
                )
            )
            best_seed = valid_seed_results[best_seed_idx]
            best_seed_score = float(best_seed.get("score_final", best_seed["score"]))
            best_seed_dir = seed_dirs[int(best_seed["seed"])]

        trial_row = {
            "trial_idx": trial_idx,
            "trial_score": trial_score,
            "trial_return_pct": trial_return,
            "trial_max_drawdown_pct": trial_max_dd,
            "trial_trades": trial_trades,
            "trial_passed": passed,
            "best_seed_score": best_seed_score,
            "best_seed_dir": str(best_seed_dir) if best_seed_dir else "",
        }
        leaderboard_rows.append(trial_row)

        improved = trial_score > best_score
        if improved:
            best_score = trial_score
            best = {
                "trial_idx": trial_idx,
                "trial_score": trial_score,
                "trial_return_pct": trial_return,
                "trial_max_drawdown_pct": trial_max_dd,
                "trial_trades": trial_trades,
                "trial_passed": passed,
                "best_seed_dir": str(best_seed_dir) if best_seed_dir else "",
                "overrides": overrides,
            }
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"[auto_tune] Trial {trial_idx} done. passed={passed} "
            f"return={trial_return:.4f} max_dd={trial_max_dd:.4f} trades={trial_trades} score={trial_score:.6f}",
            flush=True,
        )

        if no_improve >= patience:
            print(f"[auto_tune] Early stop: no improvement for {patience} trials.", flush=True)
            break

    # Save final leaderboard.
    leaderboard_path = run_root / "leaderboard.csv"
    import csv as _csv

    fieldnames = [
        "trial_idx",
        "trial_score",
        "trial_return_pct",
        "trial_max_drawdown_pct",
        "trial_trades",
        "trial_passed",
        "best_seed_score",
        "best_seed_dir",
    ]
    with leaderboard_path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in leaderboard_rows:
            w.writerow(row)

    best_path = run_root / "best_summary.json"
    _save_json(best_path, best or {})

    print(f"[auto_tune] Leaderboard: {leaderboard_path}", flush=True)
    print(f"[auto_tune] Best summary: {best_path}", flush=True)

    # Promote best model.
    if not best or not promote_enabled or not best.get("trial_passed", False):
        print("[auto_tune] Promotion skipped (no best model passed constraints).", flush=True)
        return

    # Determine slug and copy best seed's models.
    best_seed_dir = Path(str(best["best_seed_dir"]))
    best_model_dir = best_seed_dir / trial_models_subdir

    pair = str(backtest_cfg["symbol"])
    slug = pair_to_slug(pair)
    src_model_zip = best_model_dir / f"{slug}_ppo.zip"
    src_meta = best_model_dir / f"{slug}_meta.json"
    if not src_model_zip.exists() or not src_meta.exists():
        raise FileNotFoundError(
            f"Best seed model artifacts not found under {best_model_dir}. "
            f"Expected: {src_model_zip.name} and {src_meta.name}"
        )

    _copy_with_backup(src_model_zip, target_model_dir / src_model_zip.name, backup_existing=backup_existing)
    _copy_with_backup(src_meta, target_model_dir / src_meta.name, backup_existing=backup_existing)

    print(
        f"[auto_tune] Promoted best model to {target_model_dir} "
        f"({slug}_ppo.zip + {slug}_meta.json).",
        flush=True,
    )


if __name__ == "__main__":
    main()

