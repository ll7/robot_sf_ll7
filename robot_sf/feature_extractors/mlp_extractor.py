"""
MLP-based feature extractor for robot environments.

This extractor uses simple Multi-Layer Perceptrons (MLPs) to process both
LiDAR rays and drive state, providing a lightweight alternative to the
convolutional approach of the original DynamicsExtractor.
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

    class MLPFeatureExtractor(BaseFeaturesExtractor):
        """MLP-based feature extractor for robot sensor data."""

        def __init__(
            self,
            observation_space: spaces.Dict,
            ray_hidden_dims: list[int] | None = None,
            drive_hidden_dims: list[int] | None = None,
            dropout_rate: float = 0.1,
        ):
            if ray_hidden_dims is None:
                ray_hidden_dims = [128, 64]
            if drive_hidden_dims is None:
                drive_hidden_dims = [32, 16]
            rays_space = cast("spaces.Box", observation_space.spaces[OBS_RAYS])
            drive_state_space = cast("spaces.Box", observation_space.spaces[OBS_DRIVE_STATE])

            ray_input_dim = int(np.prod(rays_space.shape))
            drive_input_dim = int(np.prod(drive_state_space.shape))
            ray_output_dim = ray_hidden_dims[-1] if ray_hidden_dims else ray_input_dim
            drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim

            total_features = ray_output_dim + drive_output_dim

            super().__init__(observation_space, features_dim=total_features)

            ray_layers = []
            ray_dims = [ray_input_dim] + ray_hidden_dims

            for i in range(len(ray_dims) - 1):
                ray_layers.extend(
                    [nn.Linear(ray_dims[i], ray_dims[i + 1]), nn.ReLU(), nn.Dropout(dropout_rate)]
                )

            self.ray_extractor = nn.Sequential(nn.Flatten(), *ray_layers)

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
            ray_features = self.ray_extractor(obs[OBS_RAYS])
            drive_features = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
            return th.cat([ray_features, drive_features], dim=1)

    return {"MLPFeatureExtractor": MLPFeatureExtractor}


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {"MLPFeatureExtractor"}


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
