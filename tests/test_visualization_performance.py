"""
Performance tests for visualization generation functions.

Tests that visualization generation completes within acceptable time limits.
"""

import time

import pytest

from robot_sf.benchmark.visualization import generate_benchmark_plots, generate_benchmark_videos


@pytest.mark.slow
class TestVisualizationPerformance:
    """Test performance constraints for visualization generation."""

    @pytest.fixture
    def sample_episodes_file(self, tmp_path):
        """Create a sample episodes file for performance testing."""
        import json

        # Create a moderate-sized dataset (10 episodes) for performance testing
        episodes = []
        for i in range(10):
            episode = {
                "episode_id": f"ep_{i:03d}",
                "seed": 42 + i,
                "scenario_id": f"scenario_{(i % 3) + 1}",
                "scenario_params": {"algo": ["socialforce", "random", "greedy"][i % 3]},
                "algo": ["socialforce", "random", "greedy"][i % 3],
                "metrics": {
                    "collisions": i % 5,
                    "success": (i % 4) != 0,  # 75% success rate
                    "snqi": 0.5 + (i % 10) * 0.05,  # SNQI between 0.5-0.95
                    "path_length": 10.0 + i * 0.5,
                    "completion_time": 5.0 + i * 0.2,
                },
                "timing": {
                    "total_time": 5.0 + i * 0.2,
                    "planning_time": 0.1 + i * 0.01,
                    "execution_time": 4.9 + i * 0.19,
                },
                "status": "completed",
                "trajectory_data": [
                    [j * 0.1, j * 0.1]
                    for j in range(50)  # 50 timesteps
                ],
            }
            episodes.append(episode)

        episodes_file = tmp_path / "episodes.jsonl"
        with open(episodes_file, "w", encoding="utf-8") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        return str(episodes_file)

    def test_generate_benchmark_plots_performance(self, sample_episodes_file, tmp_path):
        """Test that plot generation completes within 30 seconds."""
        output_dir = tmp_path / "output"

        start_time = time.time()
        artifacts = generate_benchmark_plots(sample_episodes_file, str(output_dir))
        end_time = time.time()

        duration = end_time - start_time

        # Should complete within 30 seconds
        assert duration < 30.0, f"Plot generation took {duration:.2f}s, expected < 30.0s"

        # Should generate at least some artifacts
        assert len(artifacts) > 0

        # All artifacts should be generated successfully
        for artifact in artifacts:
            assert artifact.status == "generated"
            assert artifact.file_size > 0

    def test_generate_benchmark_videos_performance(self, sample_episodes_file, tmp_path):
        """Test that video generation completes within 60 seconds."""
        output_dir = tmp_path / "output"

        start_time = time.time()
        generate_benchmark_videos(
            sample_episodes_file,
            str(output_dir),
            fps=15,  # Lower FPS for faster testing
            max_duration=5.0,  # Shorter videos for testing
        )
        end_time = time.time()

        duration = end_time - start_time

        # Should complete within 60 seconds
        assert duration < 60.0, f"Video generation took {duration:.2f}s, expected < 60.0s"

        # Videos might be empty if trajectory data is insufficient
        # This is acceptable for performance testing

    def test_generate_benchmark_plots_with_filters_performance(
        self,
        sample_episodes_file,
        tmp_path,
    ):
        """Test that filtered plot generation is also performant."""
        output_dir = tmp_path / "output"

        start_time = time.time()
        generate_benchmark_plots(
            sample_episodes_file,
            str(output_dir),
            scenario_filter="scenario_1",
        )
        end_time = time.time()

        duration = end_time - start_time

        # Should complete quickly even with filters
        assert duration < 10.0, f"Filtered plot generation took {duration:.2f}s, expected < 10.0s"

    def test_visualization_memory_usage(self, sample_episodes_file, tmp_path):
        """Test that visualization generation doesn't have excessive memory usage."""
        import os

        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available, skipping memory usage test")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        output_dir = tmp_path / "output"
        artifacts = generate_benchmark_plots(sample_episodes_file, str(output_dir))

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 100MB)
        assert memory_increase < 100.0, (
            f"Memory usage increased by {memory_increase:.1f}MB, expected < 100MB"
        )

        # Should have generated artifacts
        assert len(artifacts) > 0
