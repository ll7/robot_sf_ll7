#!/usr/bin/env python3
"""Tests for oscillatory_control_profile.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path so we can import the script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.analysis.oscillatory_control_profile import (
    build_summary,
    load_episodes,
    percentile_table,
    scenario_family_from_id,
    write_hybrid_scenario_csv,
    write_per_planner_csv,
)


def test_scenario_family_from_id_classic():
    """Test extracting scenario family from classic scenario IDs."""
    assert scenario_family_from_id("classic_bottleneck_high") == "bottleneck"
    assert scenario_family_from_id("classic_bottleneck_low") == "bottleneck"
    assert scenario_family_from_id("classic_cross_trap_medium") == "cross_trap"


def test_scenario_family_from_id_francis():
    """Test extracting scenario family from francis2023 scenario IDs."""
    assert scenario_family_from_id("francis2023_accompanying_peer") == "accompanying_peer"
    assert scenario_family_from_id("francis2023_blind_corner") == "blind_corner"


def test_scenario_family_from_id_other():
    """Test extracting scenario family from unknown scenario IDs."""
    assert scenario_family_from_id("unknown_scenario") == "unknown_scenario"


def test_load_episodes_empty_dir():
    """Test loading episodes from an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        episodes = load_episodes(tmpdir)
        assert episodes == []


def test_load_episodes_with_data():
    """Test loading episodes with mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock planner directory
        planner_dir = os.path.join(tmpdir, "test_planner__differential_drive")
        os.makedirs(planner_dir)
        episodes_file = os.path.join(planner_dir, "episodes.jsonl")

        # Write a mock episode
        episode = {
            "scenario_id": "classic_bottleneck_high",
            "seed": 20,
            "safety_predicates": {
                "oscillatory_control_predicate": {
                    "fields": {
                        "progress_ratio": 0.95,
                        "command_source_changes": 3,
                        "heading_rate_sign_changes": 5,
                        "linear_velocity_sign_changes": 2,
                        "mean_abs_jerk": 0.5,
                        "path_length_m": 20.0,
                        "net_progress_m": 19.0,
                    }
                }
            },
        }
        with open(episodes_file, "w") as f:
            f.write(json.dumps(episode) + "\n")

        episodes = load_episodes(tmpdir)
        assert len(episodes) == 1
        assert episodes[0]["planner"] == "test_planner"
        assert episodes[0]["scenario_family"] == "bottleneck"
        assert episodes[0]["progress_ratio"] == 0.95


def test_percentile_table():
    """Test percentile table computation."""
    episodes = [
        {"planner": "A", "progress_ratio": 0.9},
        {"planner": "A", "progress_ratio": 0.8},
        {"planner": "A", "progress_ratio": 0.7},
        {"planner": "B", "progress_ratio": 0.5},
        {"planner": "B", "progress_ratio": 0.6},
    ]
    result = percentile_table(episodes, "planner", "progress_ratio")
    assert "A" in result
    assert "B" in result
    assert result["A"]["N"] == 3
    assert result["B"]["N"] == 2
    assert abs(result["A"]["p50"] - 0.8) < 0.01


def test_write_per_planner_csv():
    """Test writing per-planner percentile CSV."""
    episodes = [
        {
            "planner": "test_planner",
            "scenario_id": "classic_bottleneck_high",
            "scenario_family": "bottleneck",
            "seed": 20,
            "progress_ratio": 0.95,
            "command_source_changes": 3,
            "heading_rate_sign_changes": 5,
            "linear_velocity_sign_changes": 2,
            "mean_abs_jerk": 0.5,
            "path_length_m": 20.0,
            "net_progress_m": 19.0,
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        output_path = Path(f.name)

    try:
        write_per_planner_csv(episodes, output_path)
        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()
            assert "test_planner" in content
            assert "progress_ratio_mean" in content
    finally:
        output_path.unlink()


def test_write_hybrid_scenario_csv():
    """Test writing hybrid scenario family CSV."""
    episodes = [
        {
            "planner": "scenario_adaptive_hybrid_orca_v1",
            "scenario_id": "classic_bottleneck_high",
            "scenario_family": "bottleneck",
            "seed": 20,
            "progress_ratio": 0.95,
            "command_source_changes": 3,
        }
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        output_path = Path(f.name)

    try:
        write_hybrid_scenario_csv(episodes, output_path)
        assert output_path.exists()
        with open(output_path) as f:
            content = f.read()
            assert "bottleneck" in content
            assert "scenario_adaptive_hybrid_orca_v1" in content
    finally:
        output_path.unlink()


def test_build_summary():
    """Test building summary text for issue comment."""
    episodes = [
        {
            "planner": "test_planner",
            "scenario_id": "classic_bottleneck_high",
            "scenario_family": "bottleneck",
            "seed": 20,
            "progress_ratio": 0.95,
            "command_source_changes": 3,
            "heading_rate_sign_changes": 5,
            "linear_velocity_sign_changes": 2,
            "mean_abs_jerk": 0.5,
            "path_length_m": 20.0,
            "net_progress_m": 19.0,
        }
    ]
    summary = build_summary(episodes)
    assert "Per-Planner Progress Ratio" in summary
    assert "test_planner" in summary
    assert "Worst Wanderers" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
