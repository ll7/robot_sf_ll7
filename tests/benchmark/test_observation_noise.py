"""Tests for benchmark observation-noise injection."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.observation_noise import (
    apply_observation_noise,
    make_observation_noise_rng,
    normalize_observation_noise_spec,
    observation_noise_hash,
)


def _sample_obs() -> dict[str, object]:
    return {
        "robot": {"position": [1.0, 2.0], "heading": [0.0]},
        "goal": {"current": [5.0, 2.0]},
        "pedestrians": {
            "positions": [[2.0, 2.0], [3.0, 2.0]],
            "velocities": [[0.1, 0.0], [0.0, 0.1]],
            "radius": [0.35, 0.35],
            "count": 2,
        },
        "lidar": [1.0, 2.0, 3.0],
    }


def test_observation_noise_default_is_disabled_noop() -> None:
    """Absent observation-noise config should preserve planner inputs."""
    spec = normalize_observation_noise_spec(None)
    obs = _sample_obs()
    rng = make_observation_noise_rng(spec, seed=1, scenario_id="s1")

    noisy, stats = apply_observation_noise(obs, spec, rng)

    assert noisy is obs
    assert spec["enabled"] is False
    assert spec["profile"] == "none"
    assert observation_noise_hash(spec)
    assert all(value == 0 for value in stats.values())


def test_observation_noise_applies_lidar_pedestrian_and_pose_noise() -> None:
    """Non-zero noise settings should perturb only the planner-facing observation copy."""
    spec = normalize_observation_noise_spec(
        {
            "profile": "unit",
            "pose_noise_std_m": 0.1,
            "heading_noise_std_rad": 0.1,
            "lidar_dropout_prob": 1.0,
            "lidar_dropout_value": -1.0,
            "pedestrian_false_negative_prob": 1.0,
            "pedestrian_false_positive_prob": 1.0,
            "pedestrian_false_positive_radius_m": 0.0,
        }
    )
    obs = _sample_obs()
    rng = make_observation_noise_rng(spec, seed=1, scenario_id="s1")

    noisy, stats = apply_observation_noise(obs, spec, rng)

    assert noisy is not obs
    assert obs["robot"]["position"] == [1.0, 2.0]
    assert noisy["robot"]["position"] != [1.0, 2.0]
    assert noisy["lidar"] == [-1.0, -1.0, -1.0]
    assert noisy["pedestrians"]["count"] == 1
    assert noisy["pedestrians"]["velocities"] == [[0.0, 0.0]]
    assert stats["pose_noise_applied"] == 1
    assert stats["heading_noise_applied"] == 1
    assert stats["lidar_values_dropped"] == 3
    assert stats["pedestrians_removed"] == 2
    assert stats["pedestrians_added"] == 1
    assert stats["steps_with_noise"] == 1


def test_observation_noise_applies_map_runner_top_level_pose_keys() -> None:
    """Map-runner SOCNAV observations expose top-level robot pose fields."""
    spec = normalize_observation_noise_spec(
        {
            "profile": "top_level_pose",
            "pose_noise_std_m": 0.1,
            "heading_noise_std_rad": 0.1,
            "seed": 7,
        }
    )
    obs = {
        "robot_position": np.array([1.0, 2.0], dtype=np.float32),
        "robot_heading": np.array([0.0], dtype=np.float32),
    }
    rng = make_observation_noise_rng(spec, seed=1, scenario_id="s1")

    noisy, stats = apply_observation_noise(obs, spec, rng)

    assert noisy is not obs
    assert noisy["robot_position"] != [1.0, 2.0]
    assert noisy["robot_heading"] != [0.0]
    assert stats["pose_noise_applied"] == 1
    assert stats["heading_noise_applied"] == 1
    assert stats["steps_with_noise"] == 1


def test_observation_noise_validates_probability_ranges() -> None:
    """Probability-like noise fields should reject invalid values."""
    with pytest.raises(ValueError, match="lidar_dropout_prob"):
        normalize_observation_noise_spec({"lidar_dropout_prob": 1.5})


def test_observation_noise_rng_is_repeatable() -> None:
    """The same scenario seed and profile should produce repeatable corruptions."""
    spec = normalize_observation_noise_spec({"profile": "unit", "pose_noise_std_m": 0.1})
    first_rng = make_observation_noise_rng(spec, seed=9, scenario_id="s1")
    second_rng = make_observation_noise_rng(spec, seed=9, scenario_id="s1")

    first, _ = apply_observation_noise(_sample_obs(), spec, first_rng)
    second, _ = apply_observation_noise(_sample_obs(), spec, second_rng)

    np.testing.assert_allclose(first["robot"]["position"], second["robot"]["position"])
