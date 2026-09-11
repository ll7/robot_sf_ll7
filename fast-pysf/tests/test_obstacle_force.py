"""Geometry-derived tests for obstacle force calculations."""

import math
from dataclasses import asdict, replace
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
    resolve_obstacle_force_law_with_mode,
)
from pysocialforce.forces import (
    ObstacleForce,
    all_obstacle_forces_for_law,
    obstacle_force,
    obstacle_force_for_law,
    obstacle_force_surface_distance_unit_normal,
    surface_distance_unit_normal_force,
    surface_distance_unit_normal_force_vectors,
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
    ("obstacle", "ortho_vec", "ped_pos", "ped_radius", "surface_point"),
    [
        ((1.0, 1.0, 1.0, 1.0), (0.0, 1.0), (2.0, 2.0), 0.2, (1.0, 1.0)),
        ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0), (2.0, 2.0), 0.2, (1.0, 0.0)),
        ((0.0, 0.0, 2.0, 0.0), (0.0, 1.0), (1.0, 1.0), 0.2, (1.0, 0.0)),
    ],
)
def test_unversioned_dispatch_reproduces_legacy_obstacle_force_exactly(
    obstacle, ortho_vec, ped_pos, ped_radius, surface_point
):
    """Unversioned and default dispatch preserve the pre-versioning kernel exactly."""
    dx = ped_pos[0] - surface_point[0]
    dy = ped_pos[1] - surface_point[1]
    shifted_distance = max(math.hypot(dx, dy) - ped_radius, 1e-5)
    expected = (dx / shifted_distance**4, dy / shifted_distance**4)
    legacy = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

    assert legacy == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert (
        obstacle_force_for_law(
            obstacle,
            ortho_vec,
            ped_pos,
            ped_radius,
            LEGACY_SHIFTED_GRADIENT_V1,
        )
        == legacy
    )
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


@pytest.mark.parametrize("law_version", [None, SURFACE_DISTANCE_UNIT_NORMAL_V2])
def test_line_segment_batch_dispatch_matches_scalar_geometry(law_version):
    """The fast-pysf batch path matches its point/endpoint/segment scalar owner."""
    obstacles = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    ped_positions = np.array([[2.0, 2.0], [1.0, 1.0]], dtype=float)
    ped_radius = 0.2
    actual = np.zeros((len(ped_positions), 2), dtype=float)

    all_obstacle_forces_for_law(
        actual,
        ped_positions,
        obstacles,
        ped_radius,
        law_version,
    )

    expected = np.zeros_like(actual)
    for ped_index, ped_pos in enumerate(ped_positions):
        for obstacle in obstacles:
            expected[ped_index] += obstacle_force_for_law(
                obstacle[:4], obstacle[4:], ped_pos, ped_radius, law_version
            )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_obstacle_force_law_resolution_and_metadata_are_explicit():
    """Law resolution defaults old metadata to legacy and records site conventions."""
    assert resolve_obstacle_force_law() == DEFAULT_OBSTACLE_FORCE_LAW
    assert resolve_obstacle_force_law_with_mode() == (
        DEFAULT_OBSTACLE_FORCE_LAW,
        "defaulted_missing",
    )
    assert resolve_obstacle_force_law("") == LEGACY_SHIFTED_GRADIENT_V1
    assert resolve_obstacle_force_law_with_mode("") == (
        LEGACY_SHIFTED_GRADIENT_V1,
        "historical_unversioned",
    )
    assert resolve_obstacle_force_law({}) == LEGACY_SHIFTED_GRADIENT_V1
    assert (
        resolve_obstacle_force_law({"law_version": SURFACE_DISTANCE_UNIT_NORMAL_V2})
        == SURFACE_DISTANCE_UNIT_NORMAL_V2
    )
    assert resolve_obstacle_force_law_with_mode(
        {"law_version": SURFACE_DISTANCE_UNIT_NORMAL_V2}
    ) == (SURFACE_DISTANCE_UNIT_NORMAL_V2, "explicit")
    assert (
        resolve_obstacle_force_law(
            {
                "law_version": SURFACE_DISTANCE_UNIT_NORMAL_V2,
                "obstacle_force_law": SURFACE_DISTANCE_UNIT_NORMAL_V2,
            }
        )
        == SURFACE_DISTANCE_UNIT_NORMAL_V2
    )
    assert ObstacleForceConfig().law_version == LEGACY_SHIFTED_GRADIENT_V1
    assert ObstacleForceConfig().obstacle_force_law_resolution_mode == "defaulted_missing"

    metadata = obstacle_force_law_metadata(
        SURFACE_DISTANCE_UNIT_NORMAL_V2,
        site="fast_pysf",
        geometry_convention="map_line_endpoints_orthogonal_vector",
        radius_convention="threshold_plus_agent_radius_sigma",
    )
    assert metadata == {
        "schema_version": "obstacle_force_law_metadata.v2",
        "law_version": SURFACE_DISTANCE_UNIT_NORMAL_V2,
        "site": "fast_pysf",
        "geometry_convention": "map_line_endpoints_orthogonal_vector",
        "radius_convention": "threshold_plus_agent_radius_sigma",
        "compatibility_mode": "corrected_opt_in",
        "enabled": True,
        "applied": True,
        "resolution_mode": "explicit",
    }

    with pytest.raises(ValueError, match="unsupported obstacle-force law"):
        resolve_obstacle_force_law("unknown_obstacle_force_law")
    with pytest.raises(ValueError, match="conflicting obstacle-force law selectors"):
        resolve_obstacle_force_law(
            {
                "law_version": None,
                "obstacle_force_law": SURFACE_DISTANCE_UNIT_NORMAL_V2,
            }
        )


def test_obstacle_force_metadata_hashes_numerical_parameters() -> None:
    """Site metadata carries a stable hash for the parameters that affect the law."""
    metadata = obstacle_force_law_metadata(
        site="fast_pysf",
        geometry_convention="map_line_endpoints_orthogonal_vector",
        radius_convention="threshold_plus_agent_radius_sigma",
        parameters={"factor": 10.0, "distance_floor": 1e-5},
        source_commit="abc123",
        config_hash="cfg123",
    )

    assert metadata["parameters"] == {"factor": 10.0, "distance_floor": 1e-5}
    assert len(metadata["parameters_sha256"]) == 64
    assert metadata["source_commit"] == "abc123"
    assert metadata["config_hash"] == "cfg123"


@pytest.mark.parametrize(
    ("law_version", "resolution_mode"),
    [
        (None, "defaulted_missing"),
        ("", "historical_unversioned"),
        (SURFACE_DISTANCE_UNIT_NORMAL_V2, "explicit"),
    ],
)
def test_obstacle_force_config_copy_preserves_selector_provenance(
    law_version, resolution_mode
) -> None:
    """Dataclass copies retain selector provenance without changing serialized values."""
    config = ObstacleForceConfig(law_version=law_version)
    copied = replace(config)

    assert copied.obstacle_force_law_resolution_mode == resolution_mode
    assert copied.law_version == config.law_version
    assert asdict(copied)["law_version"] == config.law_version


def test_obstacle_force_config_copy_with_law_override_is_explicit() -> None:
    """Replacing the law itself recomputes provenance for the new selector."""
    copied = replace(ObstacleForceConfig(), law_version=SURFACE_DISTANCE_UNIT_NORMAL_V2)

    assert copied.obstacle_force_law_resolution_mode == "explicit"


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
    legacy_metadata = legacy_component.law_metadata()
    assert legacy_metadata["law_version"] == LEGACY_SHIFTED_GRADIENT_V1
    assert legacy_metadata["resolution_mode"] == "defaulted_missing"

    corrected_config = ObstacleForceConfig(
        threshold=-0.57,
        law_version=SURFACE_DISTANCE_UNIT_NORMAL_V2,
    )
    corrected_component = ObstacleForce(corrected_config, _Simulation())
    corrected_expected = obstacle_force_surface_distance_unit_normal(
        (1.0, 1.0, 1.0, 1.0), (0.0, 1.0), (2.0, 2.0), -0.57
    )
    np.testing.assert_array_equal(corrected_component()[0], np.asarray(corrected_expected) * 10.0)
    corrected_metadata = corrected_component.law_metadata()
    assert corrected_metadata["law_version"] == SURFACE_DISTANCE_UNIT_NORMAL_V2
    assert corrected_metadata["resolution_mode"] == "explicit"


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


@pytest.mark.parametrize(
    ("obstacle", "ortho_vec", "surface_point"),
    [
        ((0.0, 0.0, 0.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
        ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
        ((0.0, 0.0, 2.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
    ],
)
def test_corrected_line_segment_branches_are_finite_and_monotonic_at_clamp_boundary(
    obstacle, ortho_vec, surface_point
):
    """Point, endpoint, and segment branches retain finite clamped contact behavior."""
    ped_radius = 0.2
    floor = 1e-5
    raw_distances = (
        ped_radius + 3 * floor,
        ped_radius + 2 * floor,
        ped_radius + floor,
        ped_radius + 0.5 * floor,
        ped_radius + 0.1 * floor,
    )
    magnitudes = []
    for raw_distance in raw_distances:
        if obstacle == (0.0, 0.0, 0.0, 0.0):
            ped_pos = (raw_distance, 0.0)
        elif obstacle == (0.0, 0.0, 1.0, 0.0):
            ped_pos = (surface_point[0] + raw_distance, surface_point[1])
        else:
            ped_pos = (surface_point[0], surface_point[1] + raw_distance)
        force = obstacle_force_for_law(
            obstacle,
            ortho_vec,
            ped_pos,
            ped_radius,
            SURFACE_DISTANCE_UNIT_NORMAL_V2,
        )
        assert all(math.isfinite(component) for component in force)
        magnitudes.append(math.hypot(*force))

    assert all(left <= right for left, right in pairwise(magnitudes))
    assert magnitudes[-2] == pytest.approx(magnitudes[-1], rel=1e-12, abs=1e-3)

    if obstacle == (0.0, 0.0, 0.0, 0.0):
        contact_pos = (0.0, 0.0)
    else:
        contact_pos = surface_point
    assert obstacle_force_for_law(
        obstacle,
        ortho_vec,
        contact_pos,
        ped_radius,
        SURFACE_DISTANCE_UNIT_NORMAL_V2,
    ) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("raw_distance", "dx_to_surface", "dy_to_surface", "ped_radius"),
    [
        (math.nan, 1.0, 0.0, 0.2),
        (math.inf, 1.0, 0.0, 0.2),
        (1.0, math.nan, 0.0, 0.2),
        (1.0, math.inf, 0.0, 0.2),
        (1.0, 1.0, 0.0, math.nan),
        (1.0, 1.0, 0.0, math.inf),
    ],
)
def test_corrected_scalar_force_rejects_nonfinite_inputs(
    raw_distance, dx_to_surface, dy_to_surface, ped_radius
):
    """The opt-in scalar law fails closed for non-finite geometric inputs."""
    assert surface_distance_unit_normal_force(
        raw_distance, dx_to_surface, dy_to_surface, ped_radius
    ) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("obstacle", "ortho_vec", "ped_pos", "ped_radius"),
    [
        ((math.nan, 0.0, 0.0, 0.0), (0.0, 1.0), (1.0, 0.0), 0.2),
        ((0.0, 0.0, 1.0, 0.0), (math.nan, 1.0), (1.0, 0.0), 0.2),
        ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0), (math.nan, 0.0), 0.2),
        ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0), (1.0, 0.0), math.nan),
    ],
)
def test_corrected_geometry_rejects_nonfinite_inputs(obstacle, ortho_vec, ped_pos, ped_radius):
    """The opt-in geometry dispatcher never emits a non-finite force."""
    assert obstacle_force_surface_distance_unit_normal(
        obstacle, ortho_vec, ped_pos, ped_radius
    ) == (0.0, 0.0)


def test_corrected_vector_force_rejects_nonfinite_offsets():
    """Planner-style point forces reject infinite effective radius offsets."""
    positions = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    actual = surface_distance_unit_normal_force_vectors(positions, np.array([np.inf, 0.2]))

    np.testing.assert_array_equal(actual[0], np.zeros(2))
    assert np.all(np.isfinite(actual))
