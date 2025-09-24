"""
Contract tests for generate_benchmark_videos function.

These tests define the expected behavior and will fail until the implementation is complete.
"""

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.visualization import generate_benchmark_videos


def test_generate_benchmark_videos_contract():
    """Test that generate_benchmark_videos meets its contract."""

    # Given: Episode data with trajectory information
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "scenario_params": {"algo": "socialforce"},
            "trajectory_data": [
                [0.0, 0.0, 0.0, 0.0],  # [robot_x, robot_y, ped_x, ped_y]
                [1.0, 1.0, 1.1, 1.1],
                [2.0, 2.0, 2.1, 2.1],
            ],
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "output"

        # When: Function is called
        artifacts = generate_benchmark_videos(episodes_path, output_dir)

        # Then: Contract is satisfied
        assert isinstance(artifacts, list)
        assert len(artifacts) > 0
        for artifact in artifacts:
            assert artifact.status == "generated"
            assert artifact.file_size > 0
            assert (output_dir / "videos" / artifact.filename).exists()
            assert artifact.filename.endswith(".mp4")


def test_generate_benchmark_videos_with_custom_settings():
    """Test generate_benchmark_videos with custom fps and duration."""

    # Given: Episode data with trajectories
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "trajectory_data": [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "output"

        # When: Function called with custom settings
        artifacts = generate_benchmark_videos(episodes_path, output_dir, fps=24, max_duration=5.0)

        # Then: Videos generated with custom settings
        assert len(artifacts) > 0
        for artifact in artifacts:
            assert artifact.filename.endswith(".mp4")


def test_generate_benchmark_videos_missing_trajectory():
    """Test generate_benchmark_videos handles episodes without trajectory data."""

    # Given: Episode data without trajectory information
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "metrics": {"collisions": 0, "success": True},
            # No trajectory_data field
        }
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "output"

        # When/Then: Should raise VisualizationError for no trajectory data
        with pytest.raises(Exception):  # Could be VisualizationError or other
            generate_benchmark_videos(episodes_path, output_dir)
