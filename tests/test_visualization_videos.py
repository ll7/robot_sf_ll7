"""
Contract tests for generate_benchmark_videos function.

These tests define the expected behavior and will fail until the implementation is complete.
"""

import json
import tempfile
from pathlib import Path

from robot_sf.benchmark.visualization import generate_benchmark_videos


def test_generate_benchmark_videos_contract():
    """Test that generate_benchmark_videos meets its contract."""

    # Given: Episode data containing replay information
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "scenario_params": {"algo": "socialforce"},
            "replay_steps": [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.6, 0.3, 0.1],
                [1.0, 1.3, 0.9, 0.2],
            ],
            "replay_peds": [
                [(0.2, 0.3)],
                [(0.4, 0.6)],
                [(0.7, 1.0)],
            ],
            "replay_actions": [[0.1, 0.0], [0.05, 0.02], [0.0, 0.0]],
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

        # Then: A video artifact is produced for the replay episode
        assert isinstance(artifacts, list)
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.status == "generated"
        assert artifact.format == "mp4"
        assert artifact.file_size > 0
        video_path = output_dir / "videos" / artifact.filename
        assert video_path.exists()
        assert video_path.stat().st_size == artifact.file_size


def test_generate_benchmark_videos_with_custom_settings():
    """Test generate_benchmark_videos with custom fps and duration."""

    # Given: Episode data with replay steps
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "replay_steps": [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.3, 0.2, 0.1],
                [1.0, 0.8, 0.6, 0.2],
            ],
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
        assert isinstance(artifacts, list)
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert artifact.status == "generated"
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

        # When/Then: Should return empty list (no exception raised)
        artifacts = generate_benchmark_videos(episodes_path, output_dir)
        assert isinstance(artifacts, list)
        assert len(artifacts) == 0
