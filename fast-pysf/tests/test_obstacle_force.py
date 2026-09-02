"""Geometry-derived tests for obstacle force calculations."""

import math
from itertools import pairwise

import numpy as np
import pytest
from pysocialforce.config import (
    DEFAULT_OBSTACLE_FORCE_LAW,
    LEGACY_SHIFTED_GRADIENT_V1,
    SURFACE_DISTANCE_UNIT_NORMAL_V2,
    ObstacleForceConfig,
    obstacle_force_law_metadata,
    resolve_obstacle_force_law,
)
from pysocialforce.forces import (
    ObstacleForce,
    obstacle_force,
    obstacle_force_for_law,
    obstacle_force_surface_distance_unit_normal,
)


class TestObstacleForce:
    """Test suite for obstacle force calculations."""

    def test_single_point_obstacle(self):
        """A degenerate obstacle produces a finite symmetric repulsion."""
        obstacle = (1, 1, 1, 1)  # Single point obstacle
        ortho_vec = (0, 1)  # Orthogonal vector
        ped_pos = (2, 2)  # Pedestrian position
        ped_radius = 0.5  # Pedestrian radius

        # The point-to-pedestrian distance is sqrt(2) - radius.  The
        # potential-field gradient contributes one more inverse-distance
        # factor, so each equal coordinate is 1 / distance**4.
        distance = math.sqrt(2) - ped_radius
        expected_component = 1 / distance**4
        actual_force = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

        assert all(math.isfinite(component) for component in actual_force)
        assert actual_force == pytest.approx(
            (expected_component, expected_component), rel=1e-12, abs=1e-12
        )

    def test_orthogonal_hit_within_segment(self):
        """An intersection at the pedestrian position has zero direction."""
        obstacle = (0, 0, 2, 2)  # Obstacle line segment
        ortho_vec = (1, 0)  # Orthogonal vector
        ped_pos = (1, 1)  # Pedestrian position
        ped_radius = 0.1  # Pedestrian radius

        actual_force = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

        assert actual_force == pytest.approx((0.0, 0.0), abs=1e-12)

    def test_orthogonal_miss_outside_segment(self):
        """An outside projection uses the nearest endpoint direction."""
        obstacle = (0, 0, 1, 0)  # Obstacle line segment
        ortho_vec = (0, 1)  # Orthogonal vector
        ped_pos = (2, 2)  # Pedestrian position
        ped_radius = 0.1  # Pedestrian radius

        # The projection misses the segment, so (1, 0) is the nearest
        # endpoint.  The force is the endpoint distance gradient divided by
        # distance**3, yielding (1 / distance**4, 2 / distance**4).
        distance = math.sqrt(5) - ped_radius
        expected_force = (1 / distance**4, 2 / distance**4)

        actual_force = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

        assert all(math.isfinite(component) for component in actual_force)
        assert actual_force == pytest.approx(expected_force, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("obstacle", "ortho_vec", "ped_pos", "ped_radius", "surface_point"),
    [
        ((1.0, 1.0, 1.0, 1.0), (0.0, 1.0), (2.0, 2.0), 0.2, (1.0, 1.0)),
        ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0), (2.0, 2.0), 0.2, (1.0, 0.0)),
        ((0.0, 0.0, 2.0, 0.0), (0.0, 1.0), (1.0, 1.0), 0.2, (1.0, 0.0)),
    ],
)
def test_surface_distance_unit_normal_matches_point_endpoint_and_segment_analytics(
    obstacle, ortho_vec, ped_pos, ped_radius, surface_point
):
    """The corrected law uses the raw unit normal for point, endpoint, and segment cases."""
    raw_dx = ped_pos[0] - surface_point[0]
    raw_dy = ped_pos[1] - surface_point[1]
    raw_distance = math.hypot(raw_dx, raw_dy)
    surface_distance = raw_distance - ped_radius
    expected = (
        raw_dx / raw_distance / surface_distance**3,
        raw_dy / raw_distance / surface_distance**3,
    )

    actual = obstacle_force_surface_distance_unit_normal(obstacle, ortho_vec, ped_pos, ped_radius)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("obstacle", "ortho_vec", "ped_pos", "ped_radius"),
    [
        ((1.0, 1.0, 1.0, 1.0), (0.0, 1.0), (2.0, 2.0), 0.2),
        ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0), (2.0, 2.0), 0.2),
        ((0.0, 0.0, 2.0, 0.0), (0.0, 1.0), (1.0, 1.0), 0.2),
    ],
)
def test_unversioned_dispatch_reproduces_legacy_obstacle_force_exactly(
    obstacle, ortho_vec, ped_pos, ped_radius
):
    """Unversioned and default dispatch preserve the pre-versioning kernel exactly."""
    legacy = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

    assert obstacle_force_for_law(obstacle, ortho_vec, ped_pos, ped_radius) == legacy
    assert (
        obstacle_force_for_law(
            obstacle,
            ortho_vec,
            ped_pos,
            ped_radius,
            {"schema": "frozen_unversioned_fixture"},
        )
        == legacy
    )


def test_obstacle_force_law_resolution_and_metadata_are_explicit():
    """Law resolution defaults old metadata to legacy and records site conventions."""
    assert resolve_obstacle_force_law() == DEFAULT_OBSTACLE_FORCE_LAW
    assert resolve_obstacle_force_law("") == LEGACY_SHIFTED_GRADIENT_V1
    assert resolve_obstacle_force_law({}) == LEGACY_SHIFTED_GRADIENT_V1
    assert (
        resolve_obstacle_force_law({"law_version": SURFACE_DISTANCE_UNIT_NORMAL_V2})
        == SURFACE_DISTANCE_UNIT_NORMAL_V2
    )
    assert ObstacleForceConfig().law_version == LEGACY_SHIFTED_GRADIENT_V1

    metadata = obstacle_force_law_metadata(
        SURFACE_DISTANCE_UNIT_NORMAL_V2,
        site="fast_pysf",
        geometry_convention="map_line_endpoints_orthogonal_vector",
        radius_convention="threshold_plus_agent_radius_sigma",
    )
    assert metadata == {
        "schema_version": "obstacle_force_law_metadata.v1",
        "law_version": SURFACE_DISTANCE_UNIT_NORMAL_V2,
        "site": "fast_pysf",
        "geometry_convention": "map_line_endpoints_orthogonal_vector",
        "radius_convention": "threshold_plus_agent_radius_sigma",
        "compatibility_mode": "corrected_opt_in",
    }

    with pytest.raises(ValueError, match="unsupported obstacle-force law"):
        resolve_obstacle_force_law("unknown_obstacle_force_law")


def test_obstacle_force_component_dispatches_corrected_law_without_changing_default():
    """The registered force component selects v2 only for an explicit opt-in."""

    class _Peds:
        agent_radius = 0.35

        @staticmethod
        def pos():
            return np.array([[2.0, 2.0]], dtype=float)

    class _Simulation:
        peds = _Peds()

        @staticmethod
        def get_raw_obstacles():
            return np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 1.0]], dtype=float)

    legacy_config = ObstacleForceConfig(threshold=-0.57)
    legacy_component = ObstacleForce(legacy_config, _Simulation())
    legacy_expected = obstacle_force((1.0, 1.0, 1.0, 1.0), (0.0, 1.0), (2.0, 2.0), -0.57)
    np.testing.assert_array_equal(legacy_component()[0], np.asarray(legacy_expected) * 10.0)
    assert legacy_component.law_metadata()["law_version"] == LEGACY_SHIFTED_GRADIENT_V1

    corrected_config = ObstacleForceConfig(
        threshold=-0.57,
        law_version=SURFACE_DISTANCE_UNIT_NORMAL_V2,
    )
    corrected_component = ObstacleForce(corrected_config, _Simulation())
    corrected_expected = obstacle_force_surface_distance_unit_normal(
        (1.0, 1.0, 1.0, 1.0), (0.0, 1.0), (2.0, 2.0), -0.57
    )
    np.testing.assert_array_equal(corrected_component()[0], np.asarray(corrected_expected) * 10.0)
    assert corrected_component.law_metadata()["law_version"] == SURFACE_DISTANCE_UNIT_NORMAL_V2


def test_corrected_point_force_is_finite_and_monotonic_near_contact():
    """Positive near-contact distances remain finite and grow monotonically toward contact."""
    obstacle = (0.0, 0.0, 0.0, 0.0)
    distances = (0.001, 0.01, 0.05, 0.1, 0.2)
    magnitudes = []
    for distance in distances:
        force = obstacle_force_for_law(
            obstacle,
            (0.0, 1.0),
            (distance, 0.0),
            -0.57,
            SURFACE_DISTANCE_UNIT_NORMAL_V2,
        )
        assert all(math.isfinite(component) for component in force)
        magnitudes.append(math.hypot(*force))

    assert all(left > right for left, right in pairwise(magnitudes))
    assert obstacle_force_for_law(
        obstacle,
        (0.0, 1.0),
        (0.0, 0.0),
        -0.57,
        SURFACE_DISTANCE_UNIT_NORMAL_V2,
    ) == (0.0, 0.0)
