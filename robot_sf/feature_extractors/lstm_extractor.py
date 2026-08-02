"""LSTM-based feature extractor for robot sensor data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch as th
    from gymnasium import spaces
    from torch import nn


def _init_classes() -> dict[str, Any]:
    from itertools import pairwise  # noqa: PLC0415
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

    class LSTMFeatureExtractor(BaseFeaturesExtractor):
        """LSTM-based feature extractor for ray / drive-state observations."""

        def __init__(
            self,
            observation_space: spaces.Dict,
            hidden_size: int = 64,
            num_layers: int = 1,
            lstm_dropout: float = 0.0,
            drive_hidden_dims: list[int] | None = None,
            bidirectional: bool = False,
        ) -> None:
            if drive_hidden_dims is None:
                drive_hidden_dims = [32, 16]

            rays_space = cast("spaces.Box", observation_space.spaces[OBS_RAYS])
            drive_space = cast("spaces.Box", observation_space.spaces[OBS_DRIVE_STATE])

            ray_seq_len = int(np.prod(rays_space.shape))
            drive_input_dim = int(np.prod(drive_space.shape))
            drive_output_dim = drive_hidden_dims[-1] if drive_hidden_dims else drive_input_dim
            directions = 2 if bidirectional else 1
            lstm_out_dim = hidden_size * directions

            features_dim = lstm_out_dim + drive_output_dim
            super().__init__(observation_space, features_dim=features_dim)

            self._ray_seq_len = ray_seq_len

            self.ray_lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

            drive_layers: list[nn.Module] = [nn.Flatten()]
            drive_dims = [drive_input_dim] + drive_hidden_dims
            for in_dim, out_dim in pairwise(drive_dims):
                drive_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            self.drive_mlp = nn.Sequential(*drive_layers)

        def forward(self, obs: dict) -> th.Tensor:
            rays = obs[OBS_RAYS]
            rays_seq = rays.reshape(rays.shape[0], self._ray_seq_len, 1)
            _, (h_n, _) = self.ray_lstm(rays_seq)
            if self.ray_lstm.bidirectional:
                lstm_features = th.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                lstm_features = h_n[-1]

            drive_features = self.drive_mlp(obs[OBS_DRIVE_STATE])
            return th.cat([lstm_features, drive_features], dim=1)

    return {"LSTMFeatureExtractor": LSTMFeatureExtractor}


_cache: dict[str, Any] | None = None
_LAZY_NAMES = {"LSTMFeatureExtractor"}


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
