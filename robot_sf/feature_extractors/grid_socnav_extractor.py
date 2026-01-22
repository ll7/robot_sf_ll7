"""Feature extractor for SocNav + occupancy grid observations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

if TYPE_CHECKING:
    from gymnasium import spaces


class GridSocNavExtractor(BaseFeaturesExtractor):
    """CNN+MLP extractor for occupancy grid plus flattened SocNav inputs."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        grid_key: str = "occupancy_grid",
        exclude_prefixes: tuple[str, ...] = ("occupancy_grid_meta_",),
        grid_channels: list[int] | None = None,
        grid_kernel_sizes: list[int] | None = None,
        socnav_hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.1,
        include_ego_goal_vector: bool = True,
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

        socnav_keys = [
            key
            for key in observation_space.spaces.keys()
            if key != grid_key and not any(key.startswith(prefix) for prefix in exclude_prefixes)
        ]
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
        total_features = int(grid_output_dim + socnav_output_dim)

        super().__init__(observation_space, features_dim=total_features)

        self.grid_extractor = grid_extractor
        self.socnav_mlp = self._build_socnav_mlp(
            input_dim=socnav_input_dim,
            hidden_dims=socnav_hidden_dims,
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

        return th.cat([grid_features, socnav_features], dim=1)
