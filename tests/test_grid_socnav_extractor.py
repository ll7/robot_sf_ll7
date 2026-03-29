"""Tests for the grid + SocNav feature extractor."""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces

from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor


def _make_obs_dict(space: spaces.Dict, batch: int = 2) -> dict:
    obs = {}
    for key, subspace in space.spaces.items():
        sample = subspace.sample()
        tensor = th.as_tensor(sample, dtype=th.float32)
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        tensor = tensor.unsqueeze(0)
        obs[key] = tensor.repeat(batch, *([1] * (tensor.ndim - 1)))
    return obs


def test_grid_socnav_extractor_forward_shape() -> None:
    """GridSocNavExtractor produces a feature vector per batch entry."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "pedestrians_positions": spaces.Box(low=0.0, high=10.0, shape=(4, 2), dtype=np.float32),
            "occupancy_grid_meta_resolution": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        }
    )
    extractor = GridSocNavExtractor(obs_space)
    obs = _make_obs_dict(obs_space, batch=2)
    features = extractor(obs)
    assert features.shape[0] == 2
    assert features.shape[1] == extractor.features_dim


def test_grid_socnav_extractor_supports_explicit_socnav_key_subset() -> None:
    """The extractor should allow config-driven SocNav input ablations."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_speed": spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32),
            "pedestrians_positions": spaces.Box(low=0.0, high=10.0, shape=(4, 2), dtype=np.float32),
            "pedestrians_velocities": spaces.Box(
                low=-5.0, high=5.0, shape=(4, 2), dtype=np.float32
            ),
        }
    )
    extractor = GridSocNavExtractor(
        obs_space,
        include_socnav_keys=["robot_speed"],
        include_ego_goal_vector=True,
    )

    assert extractor._socnav_keys == ("robot_speed",)

    obs = _make_obs_dict(obs_space, batch=2)
    features = extractor(obs)
    assert features.shape[0] == 2
    assert features.shape[1] == extractor.features_dim


def test_grid_socnav_extractor_rejects_unknown_included_socnav_keys() -> None:
    """Misconfigured include lists should fail closed at construction time."""
    obs_space = spaces.Dict(
        {
            "occupancy_grid": spaces.Box(low=0.0, high=1.0, shape=(3, 32, 32), dtype=np.float32),
            "robot_position": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            "robot_heading": spaces.Box(low=-3.14, high=3.14, shape=(1,), dtype=np.float32),
            "goal_next": spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
        }
    )

    try:
        GridSocNavExtractor(obs_space, include_socnav_keys=["missing_key"])
    except ValueError as exc:
        assert "unknown observation keys" in str(exc)
    else:
        raise AssertionError("Expected GridSocNavExtractor to reject unknown include keys.")
