"""Module TestObstacleForce auto-generated docstring."""

import unittest

from pysocialforce.forces import obstacle_force


class TestObstacleForce(unittest.TestCase):
    """TestObstacleForce class."""

    def test_single_point_obstacle(self):
        """Test obstacle_force with an obstacle that is a single point."""
        obstacle = (1, 1, 1, 1)  # Single point obstacle
        ortho_vec = (0, 1)  # Orthogonal vector
        ped_pos = (2, 2)  # Pedestrian position
        ped_radius = 0.5  # Pedestrian radius

        # Expected result with dummy implementations
        expected_force = (1.431559360205666, 1.431559360205666)
        actual_force = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

        self.assertEqual(
            expected_force,
            actual_force,
            msg="Obstacle force with single point obstacle is incorrect.",
        )

    def test_orthogonal_hit_within_segment(self):
        """Test when orthogonal projection hits within the obstacle segment."""
        obstacle = (0, 0, 2, 2)  # Obstacle line segment
        ortho_vec = (1, 0)  # Orthogonal vector
        ped_pos = (1, 1)  # Pedestrian position
        ped_radius = 0.1  # Pedestrian radius

        expected_force = (0, 0)  # Expected result with dummy implementations
        actual_force = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

        self.assertEqual(expected_force, actual_force)

    def test_orthogonal_miss_outside_segment(self):
        """Test when orthogonal projection misses the obstacle segment."""
        obstacle = (0, 0, 1, 0)  # Obstacle line segment
        ortho_vec = (0, 1)  # Orthogonal vector
        ped_pos = (2, 2)  # Pedestrian position
        ped_radius = 0.1  # Pedestrian radius

        # Expected result verified via test execution
        # Represents force magnitude when pedestrian is outside obstacle influence
        expected_force = (0.048033001116246005, 0.09606600223249201)

        actual_force = obstacle_force(obstacle, ortho_vec, ped_pos, ped_radius)

        self.assertEqual(expected_force, actual_force)


# Add more test cases as needed...


if __name__ == "__main__":
    unittest.main()
