"""Legacy feature extractor entrypoint kept for compatibility.

The original ``DynamicsExtractor`` class is preserved here to avoid breaking
historical imports, training configs, and serialized Stable-Baselines3 policy
artifacts. The canonical modern feature-extractor namespace is now
``robot_sf.feature_extractors``.
"""
# WARNING: don't move this script or else loading trained SB3 policies might not work

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as th
    from gymnasium import spaces


def _init_classes() -> dict[str, Any]:
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

    class DynamicsExtractor(BaseFeaturesExtractor):
        """A class used to extract features from the dynamics of the environment."""

        def __init__(
            self,
            observation_space: spaces.Dict,
            use_ray_conv: bool = True,
            num_filters: list[int] | None = None,
            kernel_sizes: list[int] | None = None,
            dropout_rates: list[float] | None = None,
        ):
            if dropout_rates is None:
                dropout_rates = [0.3, 0.3, 0.3, 0.3]
            if kernel_sizes is None:
                kernel_sizes = [3, 3, 3, 3]
            if num_filters is None:
                num_filters = [64, 16, 16, 16]
            rays_space: spaces.Box = observation_space.spaces[OBS_RAYS]  # type: ignore[assignment]
            drive_state_space: spaces.Box = observation_space.spaces[OBS_DRIVE_STATE]  # type: ignore[assignment]

            drive_state_features = np.prod(drive_state_space.shape)
            num_rays = rays_space.shape[1]
            ray_features = (
                num_filters[3] * (num_rays // 16) if use_ray_conv else np.prod(rays_space.shape)
            )

            total_features = ray_features + drive_state_features

            super().__init__(observation_space, features_dim=total_features)

            def padding(kernel_size: int):
                if kernel_size % 2 == 0:
                    raise ValueError("kernel size must be odd!")
                return int((kernel_size - 1) / 2)

            def conv_block(
                in_channels: int, out_channels: int, kernel_size: int, dropout_rate: float
            ):
                return [
                    nn.Conv1d(in_channels, out_channels, kernel_size, 2, padding(kernel_size)),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                ]

            if use_ray_conv:
                in_channels = [rays_space.shape[0]] + num_filters[:-1]
                out_channels = num_filters
                args_of_blocks = zip(
                    in_channels,
                    out_channels,
                    kernel_sizes,
                    dropout_rates,
                    strict=False,
                )
                layers = [layer for args in args_of_blocks for layer in conv_block(*args)] + [
                    nn.Flatten(),
                ]
                self.ray_extractor = nn.Sequential(*layers)
            else:
                self.ray_extractor = nn.Sequential(nn.Flatten())

            self.drive_state_extractor = nn.Sequential(nn.Flatten())

        def forward(self, obs: dict) -> th.Tensor:
            """Extract features from observation.

            Returns:
                th.Tensor: Extracted feature tensor.
            """
            ray_x = self.ray_extractor(obs[OBS_RAYS])
            drive_x = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
            return th.cat([ray_x, drive_x], dim=1)

    return {"DynamicsExtractor": DynamicsExtractor}


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {"DynamicsExtractor"}


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
