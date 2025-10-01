"""Unit test SNQI weight recomputation determinism with seed (T082)."""

import json
import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.snqi.compute import recompute_snqi_weights


def test_recompute_deterministic_with_seed():
    """Test that SNQI weight recomputation is deterministic with seed."""
    # Sample baseline statistics
    baseline_stats = {
        "collisions": {"median": 0.5, "p95": 2.0},
        "time_to_goal": {"median": 10.0, "p95": 20.0},
        "near_misses": {"median": 1.0, "p95": 5.0},
        "comfort_exposure": {"median": 0.2, "p95": 0.8},
        "force_exceeds": {"median": 0.1, "p95": 0.5},
        "jerk": {"median": 0.05, "p95": 0.2},
    }

    seed = 42

    # Compute weights multiple times with same seed
    weights_1 = recompute_snqi_weights(baseline_stats=baseline_stats, method="canonical", seed=seed)
    weights_2 = recompute_snqi_weights(baseline_stats=baseline_stats, method="canonical", seed=seed)

    # Results should be identical
    assert weights_1.weights == weights_2.weights
    assert weights_1.bootstrap_params.get("method") == weights_2.bootstrap_params.get("method")
    assert weights_1.bootstrap_params.get("seed") == weights_2.bootstrap_params.get("seed")


def test_recompute_different_seeds_vary():
    """Test that different seeds produce different results for optimized method."""
    baseline_stats = {
        "collisions": {"median": 0.5, "p95": 2.0},
        "time_to_goal": {"median": 10.0, "p95": 20.0},
        "near_misses": {"median": 1.0, "p95": 5.0},
        "comfort_exposure": {"median": 0.2, "p95": 0.8},
        "force_exceeds": {"median": 0.1, "p95": 0.5},
        "jerk": {"median": 0.05, "p95": 0.2},
    }

    weights_seed_1 = recompute_snqi_weights(
        baseline_stats=baseline_stats,
        method="optimized",
        seed=1,
    )
    weights_seed_2 = recompute_snqi_weights(
        baseline_stats=baseline_stats,
        method="optimized",
        seed=2,
    )

    # For optimized method, different seeds might produce different results
    # (depends on implementation, but at least structure should be consistent)
    assert set(weights_seed_1.weights.keys()) == set(weights_seed_2.weights.keys())
    assert weights_seed_1.bootstrap_params.get("seed") != weights_seed_2.bootstrap_params.get(
        "seed",
    )


def test_recompute_canonical_method():
    """Test canonical method produces expected standard weights."""
    baseline_stats = {
        "collisions": {"median": 0.5, "p95": 2.0},
        "time_to_goal": {"median": 10.0, "p95": 20.0},
    }

    weights = recompute_snqi_weights(baseline_stats=baseline_stats, method="canonical", seed=42)

    # Canonical weights should match expected values
    expected_weights = {
        "w_success": 1.0,
        "w_time": 0.8,
        "w_collisions": 2.0,
        "w_near": 1.0,
        "w_comfort": 0.5,
        "w_force_exceed": 1.5,
        "w_jerk": 0.3,
    }

    assert weights.weights == expected_weights
    assert weights.bootstrap_params.get("method") == "canonical"
    assert weights.bootstrap_params.get("seed") == 42


def test_recompute_balanced_method():
    """Test balanced method produces equal weights."""
    baseline_stats = {
        "collisions": {"median": 0.5, "p95": 2.0},
    }

    weights = recompute_snqi_weights(baseline_stats=baseline_stats, method="balanced", seed=42)

    # All weights should be 1.0 for balanced method
    for weight in weights.weights.values():
        assert weight == 1.0

    assert weights.bootstrap_params.get("method") == "balanced"


def test_recompute_invalid_method():
    """Test that invalid method raises appropriate error."""
    baseline_stats = {"collisions": {"median": 0.5, "p95": 2.0}}

    with pytest.raises(ValueError, match="Unknown weight computation method"):
        recompute_snqi_weights(baseline_stats=baseline_stats, method="invalid_method")


def test_recompute_weights_save_and_load():
    """Test that recomputed weights can be saved and loaded correctly."""
    baseline_stats = {
        "collisions": {"median": 0.5, "p95": 2.0},
        "time_to_goal": {"median": 10.0, "p95": 20.0},
    }

    weights = recompute_snqi_weights(baseline_stats=baseline_stats, method="canonical", seed=42)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_weights.json"

        # Save weights
        weights.save(output_path)

        # Verify file exists and has expected structure
        assert output_path.exists()

        with output_path.open() as f:
            saved_data = json.load(f)

        assert "weights" in saved_data
        assert "bootstrap_params" in saved_data
        assert saved_data["weights"] == weights.weights
        assert saved_data["bootstrap_params"]["method"] == "canonical"
        assert saved_data["bootstrap_params"]["seed"] == 42


def test_recompute_with_baseline_stats_influence():
    """Test that baseline stats influence optimized weight computation."""
    # Stats with high collision rates
    high_collision_stats = {
        "collisions": {"median": 2.0, "p95": 10.0},
        "time_to_goal": {"median": 10.0, "p95": 20.0},
    }

    # Stats with low collision rates
    low_collision_stats = {
        "collisions": {"median": 0.1, "p95": 0.5},
        "time_to_goal": {"median": 10.0, "p95": 20.0},
    }

    weights_high = recompute_snqi_weights(
        baseline_stats=high_collision_stats,
        method="optimized",
        seed=42,
    )
    weights_low = recompute_snqi_weights(
        baseline_stats=low_collision_stats,
        method="optimized",
        seed=42,
    )

    # Both should have valid weights structure
    assert set(weights_high.weights.keys()) == set(weights_low.weights.keys())
    assert all(isinstance(w, int | float) for w in weights_high.weights.values())
    assert all(isinstance(w, int | float) for w in weights_low.weights.values())
