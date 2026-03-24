"""Stable-Baselines3 PPO strategy: one saved agent per product (pair).

Supports both:
- Legacy close-price-only mode (feature_dim=1)
- Multi-factor feature mode (feature_dim>1, features passed via kwargs)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ml.drl_utils import load_drl_meta, pair_to_slug

from .base import BaseStrategy


class DRLSb3Strategy(BaseStrategy):
    """Load PPO zip + meta json produced by `run_train_drl.py` per pair."""

    def __init__(
        self,
        pair: str,
        model_dir: str = "checkpoints/drl",
        deterministic: bool = True,
        device: str = "auto",
    ):
        super().__init__(name="drl")
        self.pair = pair.strip().upper()
        self.model_dir = Path(model_dir)
        self.deterministic = deterministic
        self.device = device
        slug = pair_to_slug(self.pair)
        self._model_path = self.model_dir / f"{slug}_ppo.zip"
        self._meta_path = self.model_dir / f"{slug}_meta.json"
        if not self._model_path.is_file():
            raise FileNotFoundError(
                f"DRL model not found for {self.pair}: {self._model_path}. "
                f"Train with: python run_train_drl.py --symbol {self.pair}"
            )
        if not self._meta_path.is_file():
            raise FileNotFoundError(f"DRL meta not found: {self._meta_path}")
        self._meta = load_drl_meta(self._meta_path)
        self.lookback = int(self._meta["lookback"])
        self.feature_dim = int(self._meta.get("feature_dim", 1))
        try:
            from stable_baselines3 import PPO as PPOCls
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DRL strategy requires: pip install -r requirements-drl.txt"
            ) from exc
        from ml import drl_model_architecture as _drl_arch  # noqa: F401

        load_device = self.device
        if self.device == "auto":
            try:
                import torch

                load_device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                load_device = "cpu"
        self._model = PPOCls.load(str(self._model_path), device=load_device)

    @property
    def required_prices(self) -> int:
        return self.lookback + 1

    def _build_obs_from_features(
        self,
        features: np.ndarray,
        step: int,
        position_coin: float,
        quote_free: float,
        last_price: float,
    ) -> np.ndarray:
        """Build observation from precomputed feature matrix [N, F]."""
        start = max(0, step - self.lookback)
        end = step
        window = features[start:end]  # [<=lookback, F]
        if window.shape[0] < self.lookback:
            pad = np.zeros((self.lookback - window.shape[0], self.feature_dim), dtype=np.float64)
            window = np.vstack([pad, window])
        eq = max(float(quote_free) + float(position_coin) * float(last_price), 1e-12)
        cash_ratio = np.array([float(quote_free) / eq], dtype=np.float32)
        pos_ratio = np.array(
            [(float(position_coin) * float(last_price)) / eq], dtype=np.float32
        )
        return np.concatenate([window.flatten().astype(np.float32), cash_ratio, pos_ratio])

    def _build_obs_from_prices(
        self,
        prices: Sequence[float],
        position_coin: float,
        quote_free: float,
        last_price: float,
    ) -> np.ndarray:
        """Build observation from close prices only (legacy mode)."""
        arr = np.asarray(prices, dtype=np.float64)
        t = arr.size - 1
        window = arr[t - self.lookback : t + 1]
        prev = window[:-1]
        nxt = window[1:]
        log_ret = np.log(nxt / np.clip(prev, 1e-12, None)).astype(np.float32)
        eq = max(float(quote_free) + float(position_coin) * float(last_price), 1e-12)
        cash_ratio = np.array([float(quote_free) / eq], dtype=np.float32)
        pos_ratio = np.array(
            [(float(position_coin) * float(last_price)) / eq], dtype=np.float32
        )
        return np.concatenate([log_ret, cash_ratio, pos_ratio]).astype(np.float32)

    def _build_obs(
        self,
        prices: Sequence[float],
        position_coin: float,
        quote_free: float,
        last_price: float,
        **kwargs: Any,
    ) -> np.ndarray | None:
        if len(prices) < self.required_prices or last_price <= 0:
            return None
        features = kwargs.get("features")
        step = kwargs.get("step")
        if self.feature_dim > 1 and features is not None and step is not None:
            return self._build_obs_from_features(
                features, int(step), position_coin, quote_free, last_price
            )
        if self.feature_dim > 1:
            import warnings

            warnings.warn(
                f"DRL model expects feature_dim={self.feature_dim} but no features provided. "
                "Use --data-source binance for backtest, or retrain with feature_dim=1. Returning HOLD.",
                UserWarning,
                stacklevel=1,
            )
            return None
        return self._build_obs_from_prices(prices, position_coin, quote_free, last_price)

    def _policy_obs_probs_action(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
        """Observation, action probabilities [HOLD,BUY,SELL], raw discrete action."""
        quote_free = float(kwargs.get("quote_free", 0.0))
        last_price = float(kwargs.get("last_price", 0.0))
        # Do not forward quote_free / last_price in **kwargs — _build_obs takes them positionally.
        rest = {k: v for k, v in kwargs.items() if k not in ("quote_free", "last_price")}
        obs = self._build_obs(prices, position_coin, quote_free, last_price, **rest)
        if obs is None:
            return None, None, None
        import torch

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._model.device).unsqueeze(0)
        with torch.no_grad():
            dist = self._model.policy.get_distribution(obs_t)
            probs_t = dist.distribution.probs.float().squeeze(0).cpu().numpy()
            if self.deterministic:
                a = int(np.argmax(probs_t))
            else:
                a = int(dist.sample().item())
        return obs, probs_t.reshape(-1), a

    def generate_signal(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> str:
        quote_free = float(kwargs.get("quote_free", 0.0))
        obs, _probs, a = self._policy_obs_probs_action(prices, position_coin, **kwargs)
        if obs is None:
            return "HOLD"

        if a == 1 and position_coin <= 0 and quote_free > 0:
            return "BUY"
        if a == 2 and position_coin > 0:
            return "SELL"
        return "HOLD"

    def evaluate_step(
        self,
        prices: Sequence[float],
        position_coin: float,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        quote_free = float(kwargs.get("quote_free", 0.0))
        obs, probs, a = self._policy_obs_probs_action(prices, position_coin, **kwargs)
        if obs is None or probs is None:
            return "HOLD", {
                "conf_hold": float("nan"),
                "conf_buy": float("nan"),
                "conf_sell": float("nan"),
                "input_summary": "{}",
            }

        ph, pb, ps = float(probs[0]), float(probs[1]), float(probs[2])
        signal = "HOLD"
        if a == 1 and position_coin <= 0 and quote_free > 0:
            signal = "BUY"
        elif a == 2 and position_coin > 0:
            signal = "SELL"

        flat = obs.reshape(-1)
        summary = json.dumps(
            {
                "raw_action": a,
                "obs_dim": int(flat.size),
                "obs_head": [round(float(x), 6) for x in flat[:16]],
                "obs_tail": [round(float(x), 6) for x in flat[-8:]] if flat.size > 16 else [],
                "policy_prob_hold": round(ph, 6),
                "policy_prob_buy": round(pb, 6),
                "policy_prob_sell": round(ps, 6),
            },
            separators=(",", ":"),
        )
        return signal, {
            "conf_hold": ph,
            "conf_buy": pb,
            "conf_sell": ps,
            "input_summary": summary,
        }
