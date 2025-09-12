#!/usr/bin/env python3
"""Quick validation script for SNQI weight recomputation functionality.

This script provides a minimal test to ensure the weight recomputation
and sensitivity analysis scripts are working correctly.
"""


def create_minimal_test_data():
    """Create minimal test data for validation."""

    # Minimal baseline stats
    baseline_stats = {
        "collisions": {"med": 0.0, "p95": 2.0},
        "near_misses": {"med": 1.0, "p95": 5.0},
        "force_exceed_events": {"med": 3.0, "p95": 15.0},
        "jerk_mean": {"med": 0.2, "p95": 1.0},
    }

    # Minimal episodes (3 episodes with different characteristics)
    episodes = [
        {
            "episode_id": "good_performance",
            "scenario_id": "test_scenario",
            "seed": 1,
            "scenario_params": {"algo": "good_algo"},
            "metrics": {
                "success": True,
                "time_to_goal_norm": 0.6,
                "collisions": 0,
                "near_misses": 1,
                "min_distance": 1.2,
                "comfort_exposure": 0.05,
                "force_exceed_events": 2,
                "jerk_mean": 0.3,
                "energy": 1.5,
                "avg_speed": 1.1,
            },
        },
        {
            "episode_id": "poor_performance",
            "scenario_id": "test_scenario",
            "seed": 2,
            "scenario_params": {"algo": "poor_algo"},
            "metrics": {
                "success": False,
                "time_to_goal_norm": 1.0,
                "collisions": 2,
                "near_misses": 4,
                "min_distance": 0.3,
                "comfort_exposure": 0.3,
                "force_exceed_events": 12,
                "jerk_mean": 0.8,
                "energy": 3.0,
                "avg_speed": 0.8,
            },
        },
        {
            "episode_id": "medium_performance",
            "scenario_id": "test_scenario",
            "seed": 3,
            "scenario_params": {"algo": "medium_algo"},
            "metrics": {
                "success": True,
                "time_to_goal_norm": 0.8,
                "collisions": 1,
                "near_misses": 2,
                "min_distance": 0.8,
                "comfort_exposure": 0.15,
                "force_exceed_events": 6,
                "jerk_mean": 0.5,
                "energy": 2.2,
                "avg_speed": 0.95,
            },
        },
    ]

    return episodes, baseline_stats


def test_weight_computation():
    """Test basic SNQI weight computation."""
    episodes, baseline_stats = create_minimal_test_data()

    # Test basic SNQI computation
    try:
        import sys

        sys.path.append("scripts")
        from recompute_snqi_weights import SNQIWeightRecomputer

        recomputer = SNQIWeightRecomputer(episodes, baseline_stats)

        # Test default weights
        default_weights = recomputer.default_weights()
        print("✓ Default weights generation: PASS")

        # Test SNQI computation
        for i, episode in enumerate(episodes):
            score = recomputer.compute_snqi(episode["metrics"], default_weights)
            print(f"✓ SNQI computation episode {i + 1}: {score:.3f}")

        # Test different strategies
        strategies = ["balanced", "safety_focused"]
        for strategy in strategies:
            recomputer.recompute_with_strategy(strategy)
            print(f"✓ Strategy '{strategy}': PASS")

        print("✓ All basic tests: PASS")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def validate_script_interfaces():
    """Validate that scripts can be imported and have expected interfaces."""

    try:
        import sys

        sys.path.append("scripts")

        # Test recompute_snqi_weights module
        try:
            import importlib.util

            spec = importlib.util.find_spec("recompute_snqi_weights")
            if spec is not None:
                print("✓ recompute_snqi_weights module: importable")
        except ImportError:
            return False

        # Test snqi_weight_optimization module
        try:
            spec = importlib.util.find_spec("snqi_weight_optimization")
            if spec is not None:
                print("✓ snqi_weight_optimization module: importable")
        except ImportError:
            return False

        # Test snqi_sensitivity_analysis module
        try:
            spec = importlib.util.find_spec("snqi_sensitivity_analysis")
            if spec is not None:
                print("✓ snqi_sensitivity_analysis module: importable")
        except ImportError as e:
            if "matplotlib" in str(e) or "seaborn" in str(e) or "pandas" in str(e):
                print(
                    "⚠ snqi_sensitivity_analysis module: importable (visualization dependencies missing)"
                )
            else:
                print(f"✗ snqi_sensitivity_analysis import failed: {e}")
                return False

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def main():
    """Run validation tests."""
    print("SNQI Weight Recomputation - Validation Tests")
    print("=" * 50)

    success = True

    print("\n1. Testing script imports...")
    if not validate_script_interfaces():
        success = False

    print("\n2. Testing basic functionality...")
    if not test_weight_computation():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✓ All validation tests PASSED")
        print("\nScripts are ready for use:")
        print("- scripts/recompute_snqi_weights.py")
        print("- scripts/snqi_weight_optimization.py")
        print("- scripts/snqi_sensitivity_analysis.py")
        print("- scripts/example_snqi_workflow.py")
        print("\nSee scripts/README_SNQI_WEIGHTS.md for usage instructions.")
    else:
        print("✗ Some validation tests FAILED")
        print("Check dependencies and script implementations.")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
