"""
Lightweight CNN-based feature extractor for robot environments.

This extractor uses a simplified convolutional approach with fewer parameters
than the original DynamicsExtractor while still leveraging spatial relationships
in the LiDAR data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as th
    from gymnasium import spaces
    from torch import nn


def _init_classes() -> dict[str, Any]:  # noqa: C901
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

    class LightweightCNNExtractor(BaseFeaturesExtractor):
        """Lightweight CNN-based feature extractor for robot sensor data."""

        def __init__(
            self,
            observation_space: spaces.Dict,
            num_filters: list[int] | None = None,
            kernel_sizes: list[int] | None = None,
            dropout_rate: float = 0.1,
            drive_hidden_dims: list[int] | None = None,
            record_feature_stats: bool = False,
        ):
            if num_filters is None:
                num_filters = [32, 16]
            if kernel_sizes is None:
                kernel_sizes = [5, 3]
            if drive_hidden_dims is None:
                drive_hidden_dims = [32, 16]
            rays_space = cast("spaces.Box", observation_space.spaces[OBS_RAYS])
            drive_state_space = cast("spaces.Box", observation_space.spaces[OBS_DRIVE_STATE])

            drive_input_dim = int(np.prod(drive_state_space.shape))
            drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim

            ray_features = num_filters[-1] * (rays_space.shape[1] // (2 ** len(num_filters)))

            total_features = ray_features + drive_output_dim

            super().__init__(observation_space, features_dim=total_features)

            def conv_block(
                in_channels: int, out_channels: int, kernel_size: int
            ) -> list[nn.Module]:
                padding = kernel_size // 2
                return [
                    nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout_rate),
                ]

            ray_layers = []
            in_channels = [rays_space.shape[0]] + num_filters[:-1]

            for i, (in_ch, out_ch, kernel_size) in enumerate(
                zip(in_channels, num_filters, kernel_sizes, strict=False)
            ):
                ray_layers.extend(conv_block(in_ch, out_ch, kernel_size))

            ray_layers.extend(
                [nn.AdaptiveAvgPool1d(rays_space.shape[1] // (2 ** len(num_filters))), nn.Flatten()]
            )

            self.ray_extractor = nn.Sequential(*ray_layers)
            self._record_feature_stats = bool(record_feature_stats)
            self._latest_feature_stats: dict[str, float] = {}

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

        def latest_feature_stats(self) -> dict[str, float]:
            return self._latest_feature_stats.copy()

        def forward(self, obs: dict) -> th.Tensor:
            ray_features = self.ray_extractor(obs[OBS_RAYS])
            drive_features = self.drive_state_extractor(obs[OBS_DRIVE_STATE])
            combined_features = th.cat([ray_features, drive_features], dim=1)

            if self._record_feature_stats:
                with th.no_grad():
                    stats_values = th.stack(
                        [
                            ray_features.abs().mean(),
                            ray_features.std(unbiased=False),
                            drive_features.abs().mean(),
                            drive_features.std(unbiased=False),
                            combined_features.abs().mean(),
                            combined_features.std(unbiased=False),
                            combined_features.abs().max(),
                        ]
                    ).tolist()
                    self._latest_feature_stats = {
                        "ray_mean_abs": float(stats_values[0]),
                        "ray_std": float(stats_values[1]),
                        "drive_mean_abs": float(stats_values[2]),
                        "drive_std": float(stats_values[3]),
                        "combined_mean_abs": float(stats_values[4]),
                        "combined_std": float(stats_values[5]),
                        "combined_max_abs": float(stats_values[6]),
                    }

            return combined_features

    return {"LightweightCNNExtractor": LightweightCNNExtractor}


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {"LightweightCNNExtractor"}


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
