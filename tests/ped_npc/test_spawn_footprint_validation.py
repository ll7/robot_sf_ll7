"""Tests for spawn-footprint validation against the robot start pose (#9403)."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.ped_npc.ped_population import (
    SpawnFootprintReport,
    sample_route,
    validate_spawn_footprints,
)


def test_seed_111_reset_geometry_reports_collision() -> None:
    """The recorded #9051 seed-111 reset overlap must trip the validator."""
    report = validate_spawn_footprints(
        (3.775659648764763, 9.590650608805753),
        1.0,
        [(3.4662989677686893, 8.645132035086736), (5.0, 5.0)],
        0.4,
    )

    assert isinstance(report, SpawnFootprintReport)
    assert report.collision is True
    assert report.overlapping_rows == (0,)
    assert report.min_clearance_m == pytest.approx(-0.4052, abs=1e-4)


def test_clean_separation_reports_no_collision() -> None:
    """A clear reset must pass with a positive minimum clearance."""
    report = validate_spawn_footprints((0.0, 0.0), 1.0, [(5.0, 5.0)], 0.4)

    assert report.collision is False
    assert report.overlapping_rows == ()
    assert report.min_clearance_m == pytest.approx(5.6711, abs=1e-4)


def test_empty_or_nonfinite_rows_stay_explicit() -> None:
    """No finite rows yields no collision with an explicit null minimum."""
    report = validate_spawn_footprints((0.0, 0.0), 1.0, [], 0.4)

    assert report.collision is False
    assert report.min_clearance_m is None

    report = validate_spawn_footprints((0.0, 0.0), 1.0, [(float("nan"), 0.0), ("far", None)], 0.4)

    assert report.collision is False
    assert report.min_clearance_m is None


def test_malformed_inputs_fail_closed() -> None:
    """Non-finite or malformed footprint inputs raise instead of guessing."""
    with pytest.raises(ValueError, match="finite"):
        validate_spawn_footprints((float("nan"), 0.0), 1.0, [(5.0, 5.0)], 0.4)
    with pytest.raises(ValueError, match="non-negative"):
        validate_spawn_footprints((0.0, 0.0), -1.0, [(5.0, 5.0)], 0.4)
    with pytest.raises(ValueError, match="malformed"):
        validate_spawn_footprints("nowhere", 1.0, [(5.0, 5.0)], 0.4)  # type: ignore[arg-type]


def test_route_sampling_places_peds_without_robot_check() -> None:
    """Characterize the #9403 gap: route sampling ignores the robot footprint.

    A straight route through the robot start places a pedestrian inside its
    footprint; the sampler enforces obstacle avoidance only. This test pins
    the current behavior the validator exists to catch.
    """
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.0, 0.0), (10.0, 0.0)],
        spawn_zone=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        goal_zone=((9.0, 0.0), (10.0, 0.0), (10.0, 1.0)),
    )
    robot_xy = (5.0, 0.0)

    points, _sec_id = sample_route(
        route,
        1,
        sidewalk_width=0.1,
        offset=5.0,
        rng=np.random.default_rng(111),
    )

    report = validate_spawn_footprints(robot_xy, 1.0, points, 0.4)
    assert report.collision is True
    assert report.min_clearance_m is not None and report.min_clearance_m < 0.0
