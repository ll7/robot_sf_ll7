"""Mamba / state-space feature extractor primitive for Robot SF observations.

This extractor is the first implementation slice for issue #4014. It provides a
CPU-safe sequence encoder that can run in CI and an optional exact ``mamba_ssm``
backend when that dependency is installed. The default standard-PPO contract is
the same as the existing LSTM extractor: it encodes an ordered sequence inside a
single observation, not hidden state across environment steps.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import import_module
from itertools import pairwise
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS

if TYPE_CHECKING:
    from collections.abc import Sequence

MAMBA_TEMPORAL_HISTORY_KEY = "temporal_history"
MambaBackend = Literal["auto", "mamba_ssm", "torch_ssm_lite"]
SequenceSource = Literal["rays", "temporal_history"]


@dataclass(frozen=True)
class MambaFeatureExtractorConfig:
    """Serializable default parameters for the issue #4014 Mamba extractor."""

    backend: MambaBackend = "auto"
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    num_layers: int = 1
    dropout_rate: float = 0.0
    sequence_source: SequenceSource = "rays"
    drive_hidden_dims: tuple[int, ...] = (32, 16)
    fail_if_exact_backend_missing: bool = False


class _TorchSSMLiteBlock(nn.Module):
    """Small PyTorch-only sequence block used when exact Mamba is unavailable."""

    def __init__(self, d_model: int, d_conv: int, expand: int, dropout_rate: float) -> None:
        super().__init__()
        hidden_dim = d_model * expand
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=d_conv,
            padding=max(d_conv - 1, 0),
            groups=d_model,
        )
        self.update_gate = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, d_model),
        )
        self.value_gate = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence: th.Tensor) -> th.Tensor:
        """Encode ``(batch, sequence, d_model)`` tensors with a gated residual update.

        Returns:
            Tensor with the same shape as ``sequence``.
        """
        residual = sequence
        convolved = self.depthwise_conv(sequence.transpose(1, 2)).transpose(1, 2)
        convolved = convolved[:, : sequence.shape[1], :].contiguous()
        gate = th.sigmoid(self.update_gate(convolved))
        values = th.tanh(self.value_gate(convolved))
        return self.norm(residual + gate * values)


def _load_mamba_ssm_class() -> type[nn.Module] | None:
    """Return the optional exact Mamba class without making it a hard dependency."""
    if importlib.util.find_spec("mamba_ssm") is None:
        return None

    try:
        return import_module("mamba_ssm").Mamba
    except ImportError:
        return None


class MambaFeatureExtractor(BaseFeaturesExtractor):
    """State-space sequence encoder for ray or bounded temporal-history observations."""

    def __init__(  # noqa: PLR0913
        self,
        observation_space: spaces.Dict,
        backend: MambaBackend = "auto",
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_layers: int = 1,
        dropout_rate: float = 0.0,
        sequence_source: SequenceSource = "rays",
        drive_hidden_dims: Sequence[int] | None = None,
        fail_if_exact_backend_missing: bool = False,
    ) -> None:
        """Initialize Mamba/SSM and drive-state branches."""
        if backend not in {"auto", "mamba_ssm", "torch_ssm_lite"}:
            msg = "backend must be one of: auto, mamba_ssm, torch_ssm_lite"
            raise ValueError(msg)
        if sequence_source not in {"rays", "temporal_history"}:
            msg = "sequence_source must be one of: rays, temporal_history"
            raise ValueError(msg)
        if d_model <= 0 or d_state <= 0 or d_conv <= 0 or expand <= 0 or num_layers <= 0:
            msg = "d_model, d_state, d_conv, expand, and num_layers must be positive"
            raise ValueError(msg)

        drive_hidden_dims = tuple(drive_hidden_dims if drive_hidden_dims is not None else (32, 16))
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("observation_space must be gymnasium.spaces.Dict")
        if OBS_DRIVE_STATE not in observation_space.spaces:
            msg = f"observation_space must include '{OBS_DRIVE_STATE}'"
            raise KeyError(msg)

        sequence_key = OBS_RAYS if sequence_source == "rays" else MAMBA_TEMPORAL_HISTORY_KEY
        if sequence_key not in observation_space.spaces:
            msg = f"observation_space must include '{sequence_key}' for sequence_source={sequence_source!r}"
            raise KeyError(msg)

        sequence_space = cast("spaces.Box", observation_space.spaces[sequence_key])
        drive_space = cast("spaces.Box", observation_space.spaces[OBS_DRIVE_STATE])
        sequence_input_dim = (
            1 if sequence_source == "rays" else int(np.prod(sequence_space.shape[1:]))
        )
        drive_input_dim = int(np.prod(drive_space.shape))
        drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim
        features_dim = d_model + drive_output_dim
        super().__init__(observation_space, features_dim=features_dim)

        self.sequence_key = sequence_key
        self.sequence_source = sequence_source
        self.d_state = d_state
        self.sequence_projection = nn.Linear(sequence_input_dim, d_model)
        self.sequence_layers, self.backend_name, self.backend_exact = self._build_sequence_layers(
            backend=backend,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            fail_if_exact_backend_missing=fail_if_exact_backend_missing,
        )
        self.sequence_dropout = nn.Dropout(dropout_rate)
        self.sequence_norm = nn.LayerNorm(d_model)

        drive_layers: list[nn.Module] = [nn.Flatten()]
        for in_dim, out_dim in pairwise([drive_input_dim, *drive_hidden_dims]):
            drive_layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        self.drive_mlp = nn.Sequential(*drive_layers)

    def _build_sequence_layers(
        self,
        *,
        backend: MambaBackend,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_layers: int,
        dropout_rate: float,
        fail_if_exact_backend_missing: bool,
    ) -> tuple[nn.ModuleList, str, bool]:
        mamba_cls = _load_mamba_ssm_class() if backend in {"auto", "mamba_ssm"} else None
        if mamba_cls is not None:
            layers = nn.ModuleList(
                [
                    mamba_cls(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
                    for _ in range(num_layers)
                ]
            )
            return layers, "mamba_ssm", True
        if backend == "mamba_ssm" or fail_if_exact_backend_missing:
            msg = (
                "mamba_ssm backend requested but unavailable. Install the `mamba-ssm` "
                "and `causal-conv1d` packages in the active environment."
            )
            raise ImportError(msg)
        layers = nn.ModuleList(
            [
                _TorchSSMLiteBlock(
                    d_model=d_model,
                    d_conv=d_conv,
                    expand=expand,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_layers)
            ]
        )
        return layers, "torch_ssm_lite", False

    def forward(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        """Extract feature tensors from ray/temporal sequence and drive-state inputs.

        Returns:
            Concatenated sequence and drive-state features.
        """
        sequence = self._prepare_sequence(obs[self.sequence_key])
        sequence_features = self.sequence_projection(sequence)
        for layer in self.sequence_layers:
            sequence_features = layer(sequence_features)
        sequence_features = self.sequence_norm(self.sequence_dropout(sequence_features))
        sequence_features = sequence_features.mean(dim=1)
        drive_features = self.drive_mlp(obs[OBS_DRIVE_STATE])
        return th.cat([sequence_features, drive_features], dim=1)

    def _prepare_sequence(self, values: th.Tensor) -> th.Tensor:
        """Normalize observation tensors into ``(batch, sequence, feature)`` form.

        Returns:
            Tensor ready for the sequence projection layer.
        """
        if values.ndim < 3:
            msg = (
                f"{self.sequence_source} observations must have shape "
                "(batch, sequence, features...)"
            )
            raise ValueError(msg)
        if self.sequence_source == "rays":
            return values.reshape(values.shape[0], -1, 1)
        if values.ndim < 3:
            msg = "temporal_history observations must have shape (batch, history, features...)"
            raise ValueError(msg)
        return values.reshape(values.shape[0], values.shape[1], -1)
