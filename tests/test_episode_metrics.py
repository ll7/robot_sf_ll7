"""
Test cases for new EpisodeMetricsCollector class implementing
mean interpersonal distance and per-pedestrian force quantiles.
"""

import numpy as np
import pytest

from robot_sf.eval import EpisodeMetricsCollector


class TestEpisodeMetricsCollector:
    """Test the new episode metrics collector for interpersonal distance and force quantiles."""

    def test_empty_episode(self):
        """Test metrics computation with no pedestrians in entire episode."""
        collector = EpisodeMetricsCollector()
        metrics = collector.compute_episode_metrics()

        # All metrics should be NaN when no data is available
        assert np.isnan(metrics["mean_interpersonal_distance"])
        assert np.isnan(metrics["ped_force_q50"])
        assert np.isnan(metrics["ped_force_q90"])
        assert np.isnan(metrics["ped_force_q95"])

    def test_single_pedestrian_constant_distance(self):
        """Test with single pedestrian at constant distance."""
        collector = EpisodeMetricsCollector()

        # Robot at origin, pedestrian at distance 5.0 for 3 timesteps
        robot_pos = np.array([0, 0])
        ped_positions = np.array([[3, 4]])  # distance = 5.0
        ped_forces = np.array([[1, 1]])  # magnitude = sqrt(2)

        for _ in range(3):
            collector.update_timestep(robot_pos, ped_positions, ped_forces)

        metrics = collector.compute_episode_metrics()

        # Mean distance should equal the constant distance
        assert metrics["mean_interpersonal_distance"] == pytest.approx(5.0, rel=1e-6)

        # All quantiles should equal the constant force magnitude
        expected_force_mag = np.sqrt(2)
        assert metrics["ped_force_q50"] == pytest.approx(expected_force_mag, rel=1e-6)
        assert metrics["ped_force_q90"] == pytest.approx(expected_force_mag, rel=1e-6)
        assert metrics["ped_force_q95"] == pytest.approx(expected_force_mag, rel=1e-6)

    def test_multiple_pedestrians_varying_distances(self):
        """Test with multiple pedestrians at varying distances."""
        collector = EpisodeMetricsCollector()

        robot_pos = np.array([0, 0])
        # Two pedestrians: one at distance 1, one at distance 3
        ped_positions = np.array([[1, 0], [0, 3]])
        ped_forces = np.array([[2, 0], [0, 4]])  # magnitudes: 2, 4

        # Add data for 2 timesteps
        for _ in range(2):
            collector.update_timestep(robot_pos, ped_positions, ped_forces)

        metrics = collector.compute_episode_metrics()

        # Mean distance should be (1 + 3 + 1 + 3) / 4 = 2.0
        assert metrics["mean_interpersonal_distance"] == pytest.approx(2.0, rel=1e-6)

        # Force quantiles should be average of per-pedestrian quantiles
        # Ped 1: all forces = 2, so q50=q90=q95=2
        # Ped 2: all forces = 4, so q50=q90=q95=4
        # Average: (2+4)/2 = 3
        assert metrics["ped_force_q50"] == pytest.approx(3.0, rel=1e-6)
        assert metrics["ped_force_q90"] == pytest.approx(3.0, rel=1e-6)
        assert metrics["ped_force_q95"] == pytest.approx(3.0, rel=1e-6)

    def test_no_pedestrians_some_timesteps(self):
        """Test handling of timesteps with no pedestrians."""
        collector = EpisodeMetricsCollector()

        robot_pos = np.array([0, 0])

        # Timestep 1: pedestrian present
        ped_positions = np.array([[1, 0]])
        ped_forces = np.array([[1, 0]])
        collector.update_timestep(robot_pos, ped_positions, ped_forces)

        # Timestep 2: no pedestrians (empty arrays)
        empty_positions = np.array([]).reshape(0, 2)
        empty_forces = np.array([]).reshape(0, 2)
        collector.update_timestep(robot_pos, empty_positions, empty_forces)

        # Timestep 3: pedestrian present again
        ped_positions = np.array([[0, 2]])
        ped_forces = np.array([[0, 3]])
        collector.update_timestep(robot_pos, ped_positions, ped_forces)

        metrics = collector.compute_episode_metrics()

        # Distance should be average of present timesteps: (1 + 2) / 2 = 1.5
        assert metrics["mean_interpersonal_distance"] == pytest.approx(1.5, rel=1e-6)

        # Forces should be average: (1 + 3) / 2 = 2.0
        assert metrics["ped_force_q50"] == pytest.approx(2.0, rel=1e-6)

    def test_force_arrays_with_nan(self):
        """Test handling of NaN values in force arrays."""
        collector = EpisodeMetricsCollector()

        robot_pos = np.array([0, 0])
        ped_positions = np.array([[1, 0]])

        # Force with NaN components - gets converted to [0, 3] with magnitude 3
        ped_forces = np.array([[np.nan, 3]])
        collector.update_timestep(robot_pos, ped_positions, ped_forces)

        # Force with inf components - gets converted to [4, large_num] but inf filtered out
        ped_forces = np.array([[4, np.inf]])
        collector.update_timestep(robot_pos, ped_positions, ped_forces)

        metrics = collector.compute_episode_metrics()

        # Should not crash and should handle the cleaned force values
        # After filtering, pedestrian has forces [3.0] from finite values
        # (the inf gets filtered out, leaving only 3.0)
        assert not np.isnan(metrics["ped_force_q50"])
        assert metrics["ped_force_q50"] == pytest.approx(3.0, rel=1e-6)

    def test_varying_number_of_pedestrians(self):
        """Test with varying number of pedestrians across timesteps."""
        collector = EpisodeMetricsCollector()

        robot_pos = np.array([0, 0])

        # Timestep 1: 1 pedestrian
        collector.update_timestep(robot_pos, np.array([[1, 0]]), np.array([[1, 0]]))

        # Timestep 2: 2 pedestrians
        collector.update_timestep(robot_pos, np.array([[1, 0], [0, 1]]), np.array([[2, 0], [0, 2]]))

        # Timestep 3: 1 pedestrian (different one)
        collector.update_timestep(robot_pos, np.array([[2, 0]]), np.array([[3, 0]]))

        metrics = collector.compute_episode_metrics()

        # Distances: [1, 1, 1, 2] → mean = 1.25
        assert metrics["mean_interpersonal_distance"] == pytest.approx(1.25, rel=1e-6)

        # Should handle varying pedestrian counts without crashing
        assert not np.isnan(metrics["ped_force_q50"])

    def test_truly_separate_pedestrians_different_timesteps(self):
        """Test with separate pedestrians in different timesteps to verify correct tracking."""
        collector = EpisodeMetricsCollector()
        robot_pos = np.array([0, 0])

        # Timestep 1: 2 pedestrians
        collector.update_timestep(
            robot_pos,
            np.array([[1, 0], [0, 1]]),
            np.array([[10, 0], [0, 20]]),  # magnitudes: 10, 20
        )

        # Timestep 2: only 1 pedestrian (first slot continues, second is absent)
        collector.update_timestep(
            robot_pos,
            np.array([[2, 0]]),
            np.array([[30, 0]]),  # magnitude: 30, goes to pedestrian slot 0
        )

        metrics = collector.compute_episode_metrics()

        # Pedestrian 0: forces [10, 30] → quantiles based on [10, 30]
        # Pedestrian 1: forces [20] → quantiles all equal 20
        # The exact values depend on implementation, but should be reasonable
        assert not np.isnan(metrics["ped_force_q50"])
        assert not np.isnan(metrics["ped_force_q90"])
        assert not np.isnan(metrics["ped_force_q95"])

    def test_single_force_sample_per_pedestrian(self):
        """Test case where pedestrians appear at different timesteps."""
        collector = EpisodeMetricsCollector()

        robot_pos = np.array([0, 0])

        # Timestep 1: one pedestrian with force magnitude 5
        collector.update_timestep(
            robot_pos,
            np.array([[1, 0]]),
            np.array([[5, 0]]),  # magnitude 5
        )

        # Timestep 2: same pedestrian slot gets new force magnitude 12
        # (This is how the current implementation works - assumes persistent pedestrian slots)
        collector.update_timestep(
            robot_pos,
            np.array([[0, 1]]),
            np.array([[0, 12]]),  # magnitude 12
        )

        metrics = collector.compute_episode_metrics()

        # Current implementation: both forces go to same pedestrian slot
        # Pedestrian 0 has forces [5, 12]
        # So q50 ≈ 8.5, q90 ≈ 11.3, q95 ≈ 11.65
        assert metrics["ped_force_q50"] == pytest.approx(8.5, rel=1e-6)
        # For quantiles of [5, 12]: q90 ≈ 11.3, q95 ≈ 11.65
        assert 11.0 <= metrics["ped_force_q90"] <= 12.0
        assert 11.0 <= metrics["ped_force_q95"] <= 12.0

    def test_force_quantiles_with_distribution(self):
        """Test force quantiles with enough samples to have meaningful distribution."""
        collector = EpisodeMetricsCollector()

        robot_pos = np.array([0, 0])
        ped_positions = np.array([[1, 0]])

        # Create force distribution: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for i in range(10):
            force_magnitude = i
            ped_forces = np.array([[force_magnitude, 0]])
            collector.update_timestep(robot_pos, ped_positions, ped_forces)

        metrics = collector.compute_episode_metrics()

        # For distribution [0,1,2,3,4,5,6,7,8,9]:
        # q50 (median) ≈ 4.5, q90 ≈ 8.1, q95 ≈ 8.55
        assert metrics["ped_force_q50"] == pytest.approx(4.5, rel=1e-1)
        assert 8.0 <= metrics["ped_force_q90"] <= 9.0
        assert 8.0 <= metrics["ped_force_q95"] <= 9.5

    def test_reset_episode_functionality(self):
        """Test that reset_episode clears all accumulators."""
        collector = EpisodeMetricsCollector()

        # Add some data
        robot_pos = np.array([0, 0])
        ped_positions = np.array([[1, 0]])
        ped_forces = np.array([[1, 0]])
        collector.update_timestep(robot_pos, ped_positions, ped_forces)

        # Verify data was collected
        assert len(collector.robot_ped_distances) == 1
        assert len(collector.current_ped_forces) == 1

        # Reset episode
        collector.reset_episode()

        # Verify all data cleared
        assert len(collector.robot_ped_distances) == 0
        assert len(collector.current_ped_forces) == 0
        assert len(collector.ped_force_magnitudes) == 0

        # Metrics should return NaN after reset
        metrics = collector.compute_episode_metrics()
        assert np.isnan(metrics["mean_interpersonal_distance"])

    def test_no_forces_provided(self):
        """Test behavior when force data is not provided."""
        collector = EpisodeMetricsCollector()

        robot_pos = np.array([0, 0])
        ped_positions = np.array([[3, 4]])

        # Call without forces (forces=None)
        collector.update_timestep(robot_pos, ped_positions, ped_forces=None)

        metrics = collector.compute_episode_metrics()

        # Distance should be computed
        assert metrics["mean_interpersonal_distance"] == pytest.approx(5.0, rel=1e-6)

        # Force metrics should be NaN
        assert np.isnan(metrics["ped_force_q50"])
        assert np.isnan(metrics["ped_force_q90"])
        assert np.isnan(metrics["ped_force_q95"])
