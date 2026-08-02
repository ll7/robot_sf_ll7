"""
Attention-based feature extractor for robot environments.

This extractor uses self-attention mechanisms to process LiDAR rays,
allowing the model to focus on the most relevant rays for decision making.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as th
    from gymnasium import spaces


def _init_classes() -> dict[str, Any]:
    from typing import cast  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch as th  # noqa: PLC0415
    from stable_baselines3.common.torch_layers import (  # noqa: PLC0415
        BaseFeaturesExtractor,
    )
    from torch import nn  # noqa: PLC0415

    from robot_sf.sensor.sensor_fusion import (  # noqa: PLC0415
        OBS_DRIVE_STATE,
        OBS_RAYS,
    )

    class MultiHeadAttention(nn.Module):
        """Multi-head attention module for processing sequential data."""

        def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
            super().__init__()
            self.embed_dim = embed_dim
            self.num_heads = num_heads

            if num_heads <= 0:
                raise ValueError(f"num_heads must be positive, got {num_heads}")
            if embed_dim % num_heads != 0:
                raise ValueError(
                    f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}",
                )
            self.head_dim = embed_dim // num_heads

            self.query = nn.Linear(embed_dim, embed_dim)
            self.key = nn.Linear(embed_dim, embed_dim)
            self.value = nn.Linear(embed_dim, embed_dim)

            self.dropout = nn.Dropout(dropout)
            self.output_proj = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: th.Tensor) -> th.Tensor:
            batch_size, seq_len, embed_dim = x.shape

            Q = (
                self.query(x)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = (
                self.value(x)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

            scores = th.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attention_weights = th.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)

            attended = th.matmul(attention_weights, V)

            attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            output = self.output_proj(attended)

            return output

    class AttentionFeatureExtractor(BaseFeaturesExtractor):
        """Attention-based feature extractor for robot sensor data."""

        def __init__(
            self,
            observation_space: spaces.Dict,
            embed_dim: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            dropout_rate: float = 0.1,
            drive_hidden_dims: list[int] | None = None,
        ):
            if drive_hidden_dims is None:
                drive_hidden_dims = [32, 16]
            rays_space = cast("spaces.Box", observation_space.spaces[OBS_RAYS])
            drive_state_space = cast("spaces.Box", observation_space.spaces[OBS_DRIVE_STATE])

            num_timesteps, _num_rays = rays_space.shape
            drive_input_dim = int(np.prod(drive_state_space.shape))
            drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim

            total_features = embed_dim + drive_output_dim

            super().__init__(observation_space, features_dim=total_features)

            self.ray_embedding = nn.Sequential(
                nn.Linear(num_timesteps, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            )

            self.attention_layers = nn.ModuleList(
                [MultiHeadAttention(embed_dim, num_heads, dropout_rate) for _ in range(num_layers)]
            )

            self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

            self.global_pool = nn.AdaptiveAvgPool1d(1)

            drive_layers = []
            drive_dims = [drive_input_dim] + drive_hidden_dims

            for i in range(len(drive_dims) - 1):
                drive_layers.extend(
                    [
                        nn.Linear(drive_dims[i], drive_dims[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                    ]
                )

            self.drive_state_extractor = nn.Sequential(nn.Flatten(), *drive_layers)

        def forward(self, obs: dict) -> th.Tensor:
            rays = obs[OBS_RAYS]
            rays_transposed = rays.transpose(1, 2)
            ray_embeddings = self.ray_embedding(rays_transposed)

            attended_rays = ray_embeddings
            for attention, layer_norm in zip(self.attention_layers, self.layer_norms, strict=False):
                attended_rays = layer_norm(attended_rays + attention(attended_rays))

            attended_rays = attended_rays.transpose(1, 2)
            ray_features = self.global_pool(attended_rays).squeeze(-1)

            drive_features = self.drive_state_extractor(obs[OBS_DRIVE_STATE])

            return th.cat([ray_features, drive_features], dim=1)

    return {
        "MultiHeadAttention": MultiHeadAttention,
        "AttentionFeatureExtractor": AttentionFeatureExtractor,
    }


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {"MultiHeadAttention", "AttentionFeatureExtractor"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        global _cache
        if _cache is None:
            _cache = _init_classes()
            for lazy_name, value in _cache.items():
                value.__qualname__ = lazy_name
                globals()[lazy_name] = value
        return _cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List lazy exports without importing optional ML dependencies.

    Returns:
        All module globals and deferred export names.
    """
    return sorted(set(globals()) | _LAZY_NAMES)
