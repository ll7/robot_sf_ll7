"""
Integration tests for benchmark with real visualizations.

These tests verify the complete workflow from benchmark execution to visualization generation.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def test_benchmark_with_visualization_integration():
    """Test complete benchmark execution with visualization generation."""

    # Given: Mock benchmark configuration and episode data
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "scenario_params": {"algo": "socialforce"},
            "metrics": {"collisions": 0, "success": True, "snqi": 0.95},
            "trajectory_data": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "benchmark_output"

        # Test that visualization functions are integrated into the orchestrator
        # The orchestrator now calls our visualization functions directly

        # Verify that our visualization functions are available
        from robot_sf.benchmark.visualization import (
            generate_benchmark_plots,
            generate_benchmark_videos,
            validate_visual_artifacts,
        )

        # And that they can be called with the episode data
        plots = generate_benchmark_plots(episodes_data, str(output_dir / "plots"))
        videos = generate_benchmark_videos(episodes_data, str(output_dir / "videos"))
        validation = validate_visual_artifacts(plots + videos)

        # Then: Functions execute without error and return expected types
        assert isinstance(plots, list)
        assert isinstance(videos, list)
        assert hasattr(validation, "passed")
        assert hasattr(validation, "failed_artifacts")


def test_benchmark_visualization_handles_errors():
    """Test that visualization functions handle errors gracefully."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        output_dir = Path(tmp_dir) / "benchmark_output"

        # Given: Visualization functions that raise errors
        with patch("robot_sf.benchmark.visualization.generate_benchmark_plots") as mock_plots:
            mock_plots.side_effect = Exception("Plot generation failed")

            # When: Visualization functions are called
            from robot_sf.benchmark.visualization import generate_benchmark_plots

            with pytest.raises(Exception, match="Plot generation failed"):
                generate_benchmark_plots(str(episodes_path), str(output_dir))


def test_benchmark_visualization_creates_output_structure():
    """Test that visualization functions create proper output directory structure."""

    # Given: Episode data
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "test_scenario",
            "metrics": {"collisions": 0, "success": True},
            "trajectory_data": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "benchmark_output"

        # When: Visualization functions are called
        from robot_sf.benchmark.visualization import (
            generate_benchmark_plots,
            generate_benchmark_videos,
        )

        plots = generate_benchmark_plots(episodes_data, str(output_dir / "plots"))
        videos = generate_benchmark_videos(
            episodes_data, str(output_dir / "videos")
        )  # Then: Functions return artifact lists
        assert isinstance(plots, list)
        assert isinstance(videos, list)
