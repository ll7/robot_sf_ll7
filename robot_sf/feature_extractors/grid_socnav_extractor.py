"""Feature extractor for SocNav + occupancy grid observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

if TYPE_CHECKING:
    from gymnasium import spaces

_DEFAULT_PEDESTRIAN_SLOT_KEYS = ("pedestrians_positions", "pedestrians_velocities")
_DEFAULT_PEDESTRIAN_COUNT_KEY = "pedestrians_count"


class PedestrianAttentionHead(nn.Module):
    """Masked multi-head self-attention over variable-length pedestrian sequences.

    Each pedestrian slot is projected to d_model, then passed through one layer of
    multi-head self-attention. Padding slots (beyond the live count) are masked out.
    The output is mean-pooled over valid slots and projected to output_dim.
    """

    def __init__(
        self,
        slot_input_dim: int,
        d_model: int = 64,
        num_heads: int = 4,
        output_dim: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        """Initialize the pedestrian attention head."""
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        super().__init__()
        self.input_proj = nn.Linear(slot_input_dim, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        """Dimension of the output feature vector."""
        return self._output_dim

    def forward(
        self,
        slot_feats: th.Tensor,
        count: th.Tensor | None,
    ) -> th.Tensor:
        """Compute attention-pooled pedestrian features.

        Args:
            slot_feats: (batch, max_peds, slot_input_dim) pedestrian slot tensor.
            count: (batch, 1) or (batch,) float tensor with live pedestrian counts,
                or None to treat all slots as valid.

        Returns:
            th.Tensor: (batch, output_dim) pooled feature vector.
        """
        batch_size, max_peds, _ = slot_feats.shape
        x = self.input_proj(slot_feats)  # (batch, max_peds, d_model)

        key_padding_mask: th.Tensor | None = None
        if count is not None:
            count_int = count.view(-1).long()  # (batch,)
            idx = th.arange(max_peds, device=slot_feats.device).unsqueeze(0)  # (1, max_peds)
            key_padding_mask = idx >= count_int.unsqueeze(1)  # (batch, max_peds); True=ignore

        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm(x + attn_out)

        # Masked mean-pool over valid slots
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)  # (batch, max_peds, 1)
        else:
            valid = th.ones(batch_size, max_peds, 1, device=slot_feats.device)
        valid_count = valid.sum(dim=1).clamp(min=1.0)  # (batch, 1)
        pooled = (x * valid).sum(dim=1) / valid_count  # (batch, d_model)

        return self.output_proj(pooled)  # (batch, output_dim)


class GridSocNavExtractor(BaseFeaturesExtractor):
    """CNN+MLP extractor for occupancy grid plus flattened SocNav inputs.

    Optionally replaces the flat-MLP treatment of pedestrian slot arrays with a
    masked multi-head self-attention head (``use_pedestrian_attention=True``).
    """

    def __init__(  # noqa: PLR0913,C901
        self,
        observation_space: spaces.Dict,
        *,
        grid_key: str = "occupancy_grid",
        exclude_prefixes: tuple[str, ...] = ("occupancy_grid_meta_",),
        include_socnav_keys: list[str] | tuple[str, ...] | None = None,
        exclude_socnav_keys: list[str] | tuple[str, ...] | None = None,
        privileged_state_key: str | None = None,
        include_privileged_state: bool = False,
        grid_channels: list[int] | None = None,
        grid_kernel_sizes: list[int] | None = None,
        socnav_hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
        include_ego_goal_vector: bool = True,
        use_pedestrian_attention: bool = False,
        pedestrian_slot_keys: list[str] | tuple[str, ...] | None = None,
        pedestrian_count_key: str = _DEFAULT_PEDESTRIAN_COUNT_KEY,
        attn_d_model: int = 64,
        attn_num_heads: int = 4,
        attn_output_dim: int = 64,
    ) -> None:
        """Initialize the grid + SocNav feature extractor."""
        if grid_channels is None:
            grid_channels = [32, 64, 64]
        if grid_kernel_sizes is None:
            grid_kernel_sizes = [5, 3, 3]
        if socnav_hidden_dims is None:
            socnav_hidden_dims = [128, 128]
        if len(grid_channels) != len(grid_kernel_sizes):
            raise ValueError(
                "'grid_channels' and 'grid_kernel_sizes' must have the same length, "
                f"got {len(grid_channels)} and {len(grid_kernel_sizes)}."
            )

        if grid_key not in observation_space.spaces:
            raise ValueError(f"GridSocNavExtractor requires '{grid_key}' in observation space.")

        grid_space = cast("spaces.Box", observation_space.spaces[grid_key])
        if len(grid_space.shape) != 3:
            raise ValueError("Grid observation must be shaped (C, H, W).")

        exclude_key_set = set(exclude_socnav_keys or ())
        socnav_keys = [
            key
            for key in observation_space.spaces.keys()
            if key != grid_key
            and not any(key.startswith(prefix) for prefix in exclude_prefixes)
            and key not in exclude_key_set
        ]
        if include_socnav_keys is not None:
            include_key_set = set(include_socnav_keys)
            unknown_keys = include_key_set.difference(observation_space.spaces.keys())
            if unknown_keys:
                raise ValueError(
                    "GridSocNavExtractor include_socnav_keys contains unknown observation keys: "
                    f"{sorted(unknown_keys)}"
                )
            socnav_keys = [key for key in socnav_keys if key in include_key_set]

        if privileged_state_key is not None:
            if include_privileged_state:
                if privileged_state_key not in observation_space.spaces:
                    raise ValueError(
                        "GridSocNavExtractor expected privileged_state_key to exist in observation space: "
                        f"{privileged_state_key}"
                    )
                socnav_keys.append(privileged_state_key)
            else:
                socnav_keys = [key for key in socnav_keys if key != privileged_state_key]

        # --- pedestrian attention setup ---
        self._pedestrian_attn: PedestrianAttentionHead | None = None
        self._pedestrian_slot_keys: tuple[str, ...] = ()
        self._pedestrian_count_key: str | None = None
        attn_out_dim = 0

        if use_pedestrian_attention:
            resolved_slot_keys = list(pedestrian_slot_keys or _DEFAULT_PEDESTRIAN_SLOT_KEYS)
            resolved_slot_keys = [k for k in resolved_slot_keys if k in observation_space.spaces]
            if not resolved_slot_keys:
                raise ValueError(
                    "use_pedestrian_attention=True but none of the pedestrian slot keys "
                    f"({list(pedestrian_slot_keys or _DEFAULT_PEDESTRIAN_SLOT_KEYS)}) "
                    "are present in the observation space."
                )
            self._pedestrian_slot_keys = tuple(resolved_slot_keys)
            # Remove slot keys from the flat MLP path
            slot_key_set = set(self._pedestrian_slot_keys)
            socnav_keys = [k for k in socnav_keys if k not in slot_key_set]

            if pedestrian_count_key in observation_space.spaces:
                self._pedestrian_count_key = pedestrian_count_key

            # Compute slot input dim: sum of per-slot feature dims (all but first axis)
            slot_input_dim = int(
                sum(
                    int(np.prod(observation_space.spaces[k].shape[1:]))
                    for k in self._pedestrian_slot_keys
                )
            )
            attn_out_dim = attn_output_dim

        self._socnav_keys = tuple(sorted(socnav_keys))
        self._grid_key = grid_key

        self._goal_vector_enabled = include_ego_goal_vector and all(
            key in observation_space.spaces
            for key in ("robot_position", "robot_heading", "goal_next")
        )

        socnav_input_dim = int(
            sum(np.prod(observation_space.spaces[key].shape) for key in self._socnav_keys)
        )
        if self._goal_vector_enabled:
            socnav_input_dim += 2

        grid_extractor = self._build_grid_cnn(
            in_channels=grid_space.shape[0],
            channels=grid_channels,
            kernels=grid_kernel_sizes,
            dropout_rate=dropout_rate,
        )
        grid_output_dim = self._infer_grid_output_dim(grid_extractor, grid_space.shape)

        socnav_output_dim = socnav_hidden_dims[-1] if socnav_hidden_dims else socnav_input_dim
        total_features = int(grid_output_dim + socnav_output_dim + attn_out_dim)

        super().__init__(observation_space, features_dim=total_features)

        self.grid_extractor = grid_extractor
        self.socnav_mlp = self._build_socnav_mlp(
            input_dim=socnav_input_dim,
            hidden_dims=socnav_hidden_dims,
            dropout_rate=dropout_rate,
        )
        if use_pedestrian_attention:
            self._pedestrian_attn = PedestrianAttentionHead(
                slot_input_dim=slot_input_dim,
                d_model=attn_d_model,
                num_heads=attn_num_heads,
                output_dim=attn_output_dim,
                dropout_rate=dropout_rate,
            )

    @staticmethod
    def _build_grid_cnn(
        *,
        in_channels: int,
        channels: list[int],
        kernels: list[int],
        dropout_rate: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        for idx, (out_ch, kernel) in enumerate(zip(channels, kernels, strict=False)):
            conv = nn.Conv2d(
                in_channels if idx == 0 else channels[idx - 1],
                out_ch,
                kernel_size=kernel,
                stride=2,
                padding=kernel // 2,
            )
            layers.extend([conv, nn.ReLU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_socnav_mlp(
        *,
        input_dim: int,
        hidden_dims: list[int],
        dropout_rate: float,
    ) -> nn.Sequential:
        if not hidden_dims:
            return nn.Sequential(nn.Identity())
        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for idx in range(len(dims) - 1):
            layers.extend(
                [nn.Linear(dims[idx], dims[idx + 1]), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
        return nn.Sequential(*layers)

    @staticmethod
    def _infer_grid_output_dim(grid_extractor: nn.Sequential, grid_shape: tuple[int, ...]) -> int:
        sample = th.zeros((1, *grid_shape), dtype=th.float32)
        with th.no_grad():
            out = grid_extractor(sample)
        return int(out.shape[1])

    def forward(self, obs: dict) -> th.Tensor:
        """Compute combined grid and SocNav feature vector.

        Returns:
            th.Tensor: Concatenated feature tensor for each batch element.
        """
        grid_obs = obs[self._grid_key]
        grid_features = self.grid_extractor(grid_obs)

        socnav_parts = [obs[key].view(obs[key].shape[0], -1) for key in self._socnav_keys]
        if self._goal_vector_enabled:
            robot_pos = obs["robot_position"]
            goal_next = obs["goal_next"]
            heading = obs["robot_heading"].view(-1)
            dx = goal_next[:, 0] - robot_pos[:, 0]
            dy = goal_next[:, 1] - robot_pos[:, 1]
            cos_h = th.cos(heading)
            sin_h = th.sin(heading)
            ego_x = cos_h * dx + sin_h * dy
            ego_y = -sin_h * dx + cos_h * dy
            goal_vec = th.stack([ego_x, ego_y], dim=1)
            socnav_parts.append(goal_vec)

        socnav_concat = (
            th.cat(socnav_parts, dim=1)
            if socnav_parts
            else th.zeros((grid_features.shape[0], 0), device=grid_features.device)
        )
        socnav_features = self.socnav_mlp(socnav_concat)

        parts = [grid_features, socnav_features]

        if self._pedestrian_attn is not None:
            # Stack slot arrays along the feature axis: (batch, max_peds, slot_dim)
            slot_tensors = [obs[k] for k in self._pedestrian_slot_keys]
            slot_feats = th.cat(slot_tensors, dim=-1)
            count = obs[self._pedestrian_count_key] if self._pedestrian_count_key else None
            parts.append(self._pedestrian_attn(slot_feats, count))

        return th.cat(parts, dim=1)
