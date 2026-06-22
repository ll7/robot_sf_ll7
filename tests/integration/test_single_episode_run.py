"""Integration test: run single scenario → produce 1 episode line (T015).

This test validates the end-to-end flow from scenario specification to
episode JSONL output, ensuring the basic benchmark runner pipeline works.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.runner import run_episode


def test_single_episode_produces_jsonl_line():
    """Test that running a single episode produces valid JSONL output."""
    # Minimal scenario using keys the generator actually accepts
    # (see robot_sf/benchmark/scenario_generator.py normalize_generation_parameters):
    # density ∈ {low, med, high}, flow ∈ {uni, bi, cross, merge}.
    scenario_params = {
        "id": "test_basic",
        "density": "low",
        "flow": "uni",
        "obstacle": "open",
    }

    seed = 42

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "episodes.jsonl"

        # Run single episode using the runner function
        episode_data = run_episode(
            scenario_params=scenario_params,
            seed=seed,
            horizon=50,
            algo="random",  # Use a simple algorithm
        )

        # Write episode data to JSONL
        with output_path.open("w") as f:
            f.write(json.dumps(episode_data) + "\n")

        # Verify output exists and is valid JSON
        assert output_path.exists()

        with output_path.open("r") as f:
            lines = f.readlines()

        assert len(lines) == 1

        # Parse and validate episode data structure
        episode = json.loads(lines[0])

        # Check required fields
        assert "episode_id" in episode
        assert "scenario_id" in episode
        assert "seed" in episode
        assert "metrics" in episode
        assert "status" in episode

        # Verify values match input
        assert episode["scenario_id"] == scenario_params["id"]
        assert episode["seed"] == seed


def test_single_episode_with_metrics():
    """Test that episode includes computed metrics."""
    scenario_params = {
        "id": "test_metrics",
        "density": "med",
        "flow": "uni",
        "obstacle": "open",
    }

    episode_data = run_episode(
        scenario_params=scenario_params,
        seed=123,
        horizon=50,
        algo="random",
    )

    # Check metrics structure
    metrics = episode_data["metrics"]

    # Verify key metrics are present (should be numeric)
    expected_metrics = [
        "collisions",
        "near_misses",
        "success",
        "time_to_goal_norm",
        "path_efficiency",
    ]

    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], int | float)


if __name__ == "__main__":
    pytest.main([__file__])
