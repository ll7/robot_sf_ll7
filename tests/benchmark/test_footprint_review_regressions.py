"""Analytic regressions for PR #10231's independent footprint review."""

import inspect
import math

import numpy as np
import pytest

from robot_sf.benchmark.metrics import EpisodeData
from robot_sf.benchmark.near_miss_ttc import compute_ttc_near_miss_diagnostic
from robot_sf.benchmark.path_utils import compute_completion_reference_length
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle


def _barrier_map():
    """Return a wall whose left edge is x=4, in a 12 by 10 metre domain."""
    return MapDefinition(
        12,
        10,
        [Obstacle([(4, 0), (6, 0), (6, 4), (4, 4)])],
        [((1, 4), (2, 4), (2, 6))],
        [],
        [((8, 3), (10, 3), (10, 7))],
        [(0, 12, 0, 0), (0, 12, 10, 10), (0, 0, 0, 10), (12, 12, 0, 10)],
        [],
        [],
        [],
        [],
    )


def _reference(start, goal, **kwargs):
    """Use the public API; main's legacy binding has no radius argument."""
    if "robot_radius" in inspect.signature(compute_completion_reference_length).parameters:
        kwargs["robot_radius"] = 1.0
    return compute_completion_reference_length(
        _barrier_map(), np.array(start), np.array(goal), **kwargs
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_physically_clear_waypoint_inside_approximation_band(reverse):
    # All points on this segment clear the physical wall by at least 1.003m.
    start, goal = (1.5, 2.0), (2.997, 2.0)
    if reverse:
        start, goal = goal, start
    assert _reference(start, goal) == pytest.approx(1.497)


def test_already_complete_physical_zone_inside_approximation_band():
    assert (
        _reference(
            (2.997, 2.5),
            (2.997, 2.5),
            completion_policy="goal_zone_entry_v1",
            goal_zone=np.array([(2.996, 2), (2.999, 2), (2.999, 3), (2.996, 3)]),
        )
        == 0.0
    )


@pytest.mark.parametrize("goal", [(3.0, 2.0), (3.001, 2.0)])
def test_contact_or_overlapping_waypoint_remains_unavailable(goal):
    assert math.isnan(_reference((1.5, 2.0), goal))


def _empty_episode(metadata):
    """Supply valid timing and empty population arrays to the public diagnostic."""
    return EpisodeData(
        robot_pos=np.array([[0.0, 0.0], [0.1, 0.0]]),
        robot_vel=np.ones((2, 2)),
        robot_acc=np.zeros((2, 2)),
        peds_pos=np.empty((2, 0, 2)),
        ped_forces=np.empty((2, 0, 2)),
        goal=np.array([10.0, 0.0]),
        dt=0.1,
        episode_metadata=metadata,
    )


def test_empty_population_diagnostic_refuses_unknown_definition():
    data = _empty_episode({"footprint_metric_schema_version": "forged"})
    with pytest.raises(ValueError, match="unsupported footprint metric definition"):
        compute_ttc_near_miss_diagnostic(data)


def test_empty_population_diagnostic_preserves_footprint_definition():
    data = _empty_episode({"footprint_metric_schema_version": "robot-sf-footprint.v1"})
    result = compute_ttc_near_miss_diagnostic(data)
    assert result["near_miss_ttc__status"] == "no-pedestrians"
    assert result["near_miss_ttc__count"] == 0.0
    assert math.isnan(result["near_miss_ttc__min_ttc_s"])
    assert result.get("near_miss_ttc__definition") == "robot-sf-footprint.v1"


@pytest.mark.parametrize("field", ["robot_radius", "ped_radius"])
@pytest.mark.parametrize("radius", [-1.0, float("nan"), None])
def test_empty_population_diagnostic_refuses_invalid_footprint_radius(field, radius):
    data = _empty_episode({"footprint_metric_schema_version": "robot-sf-footprint.v1"})
    setattr(data, field, radius)
    with pytest.raises(ValueError, match="footprint radius must be finite and nonnegative"):
        compute_ttc_near_miss_diagnostic(data)


def test_legacy_empty_population_diagnostic_retains_unmarked_zero():
    result = compute_ttc_near_miss_diagnostic(_empty_episode({}))
    assert result["near_miss_ttc__status"] == "no-pedestrians"
    assert result["near_miss_ttc__count"] == 0.0
    assert "near_miss_ttc__definition" not in result
