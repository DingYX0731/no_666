"""Custom DRL network blocks: MLP + LSTM feature extractor for SB3.

Supports variable feature_dim per timestep:
- feature_dim=1: legacy mode (single log-return per step)
- feature_dim>1: multi-factor mode (kline-derived features per step)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def parse_hidden_dims(spec: str | Sequence[int], default: list[int]) -> list[int]:
    """Parse hidden dimensions from csv string or list."""
    if isinstance(spec, str):
        if not spec.strip():
            return default
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    dims = [int(x) for x in spec]
    return dims or default


class MlpLstmFeatureExtractor(BaseFeaturesExtractor):
    """Feature extractor combining sequence LSTM and account MLP."""

    def __init__(
        self,
        observation_space,
        sequence_len: int,
        feature_dim: int = 1,
        lstm_hidden_size: int = 64,
        lstm_layers: int = 1,
        lstm_dropout: float = 0.0,
        seq_mlp_hidden_dims: Sequence[int] = (64,),
        account_mlp_hidden_dims: Sequence[int] = (16,),
        fusion_hidden_dims: Sequence[int] = (64,),
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        self.sequence_len = int(sequence_len)
        self.feature_dim = int(feature_dim)
        self.seq_flat = self.sequence_len * self.feature_dim
        total = int(observation_space.shape[0])
        self.account_dim = total - self.seq_flat
        if self.account_dim <= 0:
            raise ValueError(
                f"Invalid observation shape for sequence/account split: "
                f"total={total}, seq_flat={self.seq_flat}"
            )

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        seq_layers: list[nn.Module] = []
        in_dim = lstm_hidden_size
        for h in seq_mlp_hidden_dims:
            seq_layers.append(nn.Linear(in_dim, int(h)))
            seq_layers.append(nn.ReLU())
            in_dim = int(h)
        self.seq_mlp = nn.Sequential(*seq_layers) if seq_layers else nn.Identity()
        seq_out_dim = in_dim

        acc_layers: list[nn.Module] = []
        acc_in = self.account_dim
        for h in account_mlp_hidden_dims:
            acc_layers.append(nn.Linear(acc_in, int(h)))
            acc_layers.append(nn.ReLU())
            acc_in = int(h)
        self.acc_mlp = nn.Sequential(*acc_layers) if acc_layers else nn.Identity()
        acc_out_dim = acc_in

        fusion_layers: list[nn.Module] = []
        fusion_in = seq_out_dim + acc_out_dim
        for h in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(fusion_in, int(h)))
            fusion_layers.append(nn.ReLU())
            fusion_in = int(h)
        self.fusion = nn.Sequential(*fusion_layers) if fusion_layers else nn.Identity()

        self._features_dim = fusion_in

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        seq_flat = observations[:, : self.seq_flat]
        acc = observations[:, self.seq_flat :]

        seq = seq_flat.view(-1, self.sequence_len, self.feature_dim)
        lstm_out, _ = self.lstm(seq)
        seq_last = lstm_out[:, -1, :]
        seq_feat = self.seq_mlp(seq_last)
        acc_feat = self.acc_mlp(acc)

        merged = torch.cat([seq_feat, acc_feat], dim=1)
        return self.fusion(merged)
