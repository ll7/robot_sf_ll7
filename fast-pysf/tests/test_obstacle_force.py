"""Geometry-derived tests for obstacle force calculations."""

import math

import pytest
from pysocialforce.forces import obstacle_force


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
