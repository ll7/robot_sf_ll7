"""Tests for the analytic proxemic costmap layer."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.proxemic_costmap import (
    ProxemicCostmapConfig,
    config_hash,
    proxemic_cost_at_points,
)


def _enabled_config(**overrides: float | str | bool) -> ProxemicCostmapConfig:
    values = {
        "enabled": True,
        "personal_radius": 0.45,
        "social_radius": 1.2,
        "personal_weight": 1.0,
        "social_weight": 0.5,
        "velocity_elongation_factor": 0.0,
        "max_cost": 10.0,
        "decay_function": "linear",
    }
    values.update(overrides)
    return ProxemicCostmapConfig(**values)


def test_zero_pedestrians_yields_zero_cost() -> None:
    """No actors means no proxemic penalty."""
    points = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    costs = proxemic_cost_at_points(points, np.empty((0, 2)), np.empty((0, 2)), _enabled_config())
    assert costs.tolist() == [0.0, 0.0]


def test_cost_increases_as_distance_decreases() -> None:
    """Personal/social-zone penalties are monotonic with distance."""
    points = np.asarray([[1.1, 0.0], [0.6, 0.0], [0.1, 0.0]], dtype=float)
    costs = proxemic_cost_at_points(
        points,
        np.asarray([[0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0]], dtype=float),
        _enabled_config(),
    )
    assert costs[0] < costs[1] < costs[2]


def test_gaussian_decay_reaches_zero_at_radius_boundary() -> None:
    """Gaussian decay stays continuous at the configured zone boundary."""
    points = np.asarray([[0.0, 0.0], [0.6, 0.0], [1.2, 0.0], [1.3, 0.0]], dtype=float)
    costs = proxemic_cost_at_points(
        points,
        np.asarray([[0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0]], dtype=float),
        _enabled_config(
            decay_function="gaussian",
            personal_radius=0.0,
            social_radius=1.2,
            personal_weight=0.0,
            social_weight=1.0,
        ),
    )

    assert costs[0] == pytest.approx(1.0)
    assert 0.0 < costs[1] < costs[0]
    assert costs[2] == pytest.approx(0.0)
    assert costs[3] == pytest.approx(0.0)


def test_velocity_elongation_increases_forward_cost_field() -> None:
    """Moving pedestrians stretch the soft zone in their velocity direction."""
    pedestrian = np.asarray([[0.0, 0.0]], dtype=float)
    velocity = np.asarray([[1.0, 0.0]], dtype=float)
    cfg = _enabled_config(velocity_elongation_factor=1.0)

    forward = proxemic_cost_at_points(np.asarray([[1.5, 0.0]]), pedestrian, velocity, cfg)
    lateral = proxemic_cost_at_points(np.asarray([[0.0, 1.5]]), pedestrian, velocity, cfg)

    assert forward[0] > lateral[0]
    assert forward[0] > 0.0
    assert lateral[0] == 0.0


def test_finite_cap_is_respected() -> None:
    """Overlapping personal zones cannot exceed the configured maximum cost."""
    cost = proxemic_cost_at_points(
        np.asarray([[0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        np.zeros((2, 2), dtype=float),
        _enabled_config(personal_weight=10.0, social_weight=10.0, max_cost=3.0),
    )
    assert cost.tolist() == [3.0]


def test_disabled_layer_is_no_op() -> None:
    """Disabled config returns zero cost even when points are inside zones."""
    cost = proxemic_cost_at_points(
        np.asarray([[0.0, 0.0]], dtype=float),
        np.asarray([[0.0, 0.0]], dtype=float),
        np.asarray([[1.0, 0.0]], dtype=float),
        ProxemicCostmapConfig(enabled=False),
    )
    assert cost.tolist() == [0.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"personal_radius": -0.1},
        {"social_radius": -0.1},
        {"social_radius": 0.2, "personal_radius": 0.4},
        {"personal_weight": float("nan")},
        {"velocity_elongation_factor": -1.0},
        {"decay_function": "exponential"},
    ],
)
def test_malformed_config_fails_closed(kwargs: dict[str, float | str]) -> None:
    """Invalid layer parameters fail at config construction."""
    with pytest.raises(ValueError):
        _enabled_config(**kwargs)


def test_config_hash_changes_with_parameters() -> None:
    """Metadata hash is stable but sensitive to layer parameters."""
    assert config_hash(_enabled_config()) == config_hash(_enabled_config())
    assert config_hash(_enabled_config()) != config_hash(_enabled_config(social_radius=1.4))
