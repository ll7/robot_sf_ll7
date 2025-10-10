"""
Contract tests for generate_benchmark_plots function.

These tests define the expected behavior and will fail until the implementation is complete.
"""

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.visualization import generate_benchmark_plots


def test_generate_benchmark_plots_contract():
    """Test that generate_benchmark_plots meets its contract."""

    # Given: Valid episode data
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "classic_001",
            "scenario_params": {"algo": "socialforce"},
            "algo": "socialforce",
            "metrics": {"collisions": 0, "success": True, "snqi": 0.95},
        },
        {
            "episode_id": "ep_002",
            "scenario_id": "classic_001",
            "scenario_params": {"algo": "socialforce"},
            "algo": "socialforce",
            "metrics": {"collisions": 1, "success": False, "snqi": 0.75},
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "output"

        # When: Function is called
        artifacts = generate_benchmark_plots(episodes_path, output_dir)

        # Then: Contract is satisfied
        assert isinstance(artifacts, list)
        assert len(artifacts) > 0
        for artifact in artifacts:
            assert artifact.status == "generated"
            assert artifact.file_size > 0
            assert (output_dir / "plots" / artifact.filename).exists()
            assert artifact.filename.endswith(".pdf")


def test_generate_benchmark_plots_with_filters():
    """Test generate_benchmark_plots with scenario and baseline filters."""

    # Given: Episode data with different scenarios
    episodes_data = [
        {
            "episode_id": "ep_001",
            "scenario_id": "scenario_a",
            "scenario_params": {"algo": "socialforce"},
            "algo": "socialforce",
            "metrics": {"collisions": 0, "success": True},
        },
        {
            "episode_id": "ep_002",
            "scenario_id": "scenario_b",
            "scenario_params": {"algo": "random"},
            "algo": "random",
            "metrics": {"collisions": 2, "success": False},
        },
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        episodes_path = Path(tmp_dir) / "episodes.jsonl"
        with open(episodes_path, "w") as f:
            for ep in episodes_data:
                f.write(json.dumps(ep) + "\n")

        output_dir = Path(tmp_dir) / "output"

        # When: Function called with filters
        artifacts = generate_benchmark_plots(
            episodes_path,
            output_dir,
            scenario_filter="scenario_a",
        )

        # Then: Only filtered scenarios included
        assert len(artifacts) > 0
        for artifact in artifacts:
            # Should only include scenario_a data
            assert "scenario_a" in artifact.source_data


def test_generate_benchmark_plots_missing_file():
    """Test generate_benchmark_plots handles missing episode file."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_path = Path(tmp_dir) / "missing.jsonl"
        output_dir = Path(tmp_dir) / "output"

        # When/Then: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            generate_benchmark_plots(missing_path, output_dir)
