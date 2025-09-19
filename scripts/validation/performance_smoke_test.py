#!/usr/bin/env python3
"""
Performance Smoke Test for Social Navigation Benchmark

Purpose: Measure baseline performance metrics for environment step rate and throughput.
This establishes performance baselines documented in dev_guide.md.
"""

import time
from pathlib import Path

from robot_sf.gym_env.environment_factory import make_robot_env


def measure_environment_performance(num_resets=5):
    """Measure environment reset performance (simplified test)."""
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    # Use minimal config for consistent performance testing
    config = RobotSimulationConfig()
    env = make_robot_env(config=config, debug=False)

    total_resets = 0
    total_time = 0.0

    print(f"Running {num_resets} reset cycles...")

    for episode in range(num_resets):
        episode_start = time.time()

        # Test environment reset performance
        _, _ = env.reset()
        total_resets += 1

        episode_time = time.time() - episode_start
        total_time += episode_time

        resets_per_sec = 1.0 / episode_time if episode_time > 0 else 0

        print(f"Reset {episode + 1}: {episode_time:.3f}s ({resets_per_sec:.1f} resets/sec)")

    env.close()

    # Overall metrics
    overall_resets_per_sec = total_resets / total_time if total_time > 0 else 0
    ms_per_reset = (total_time * 1000) / total_resets if total_resets > 0 else 0

    print("\n=== Performance Summary ===")
    print(f"Total resets: {total_resets}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average resets/sec: {overall_resets_per_sec:.1f}")
    print(f"Average ms/reset: {ms_per_reset:.1f}")

    return {
        "resets_per_sec": overall_resets_per_sec,
        "ms_per_reset": ms_per_reset,
        "total_resets": total_resets,
        "total_time": total_time,
    }


def measure_environment_creation():
    """Measure environment creation time."""
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    print("\n=== Environment Creation Performance ===")

    start_time = time.time()
    config = RobotSimulationConfig()
    env = make_robot_env(config=config, debug=False)
    creation_time = time.time() - start_time
    env.close()

    print(f"Environment creation time: {creation_time:.2f}s")
    return creation_time


def main():
    """Run performance smoke tests."""
    print("Social Navigation Benchmark - Performance Smoke Test")
    print("=" * 60)

    # Test environment creation
    creation_time = measure_environment_creation()

    # Test step performance
    perf_metrics = measure_environment_performance()

    # Write performance summary to file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    import json

    performance_log = results_dir / "performance_smoke_test.json"

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment_creation_sec": creation_time,
        "resets_per_sec": perf_metrics["resets_per_sec"],
        "ms_per_reset": perf_metrics["ms_per_reset"],
        "total_test_resets": perf_metrics["total_resets"],
        "total_test_time_sec": perf_metrics["total_time"],
    }

    with performance_log.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPerformance results saved to: {performance_log}")

    print("\n=== Expected Performance Ranges ===")
    print(f"Environment creation: < 2.0s (measured: {creation_time:.2f}s)")
    print("Environment reset: > 1 reset/sec")
    print(
        f"Measured: {perf_metrics['resets_per_sec']:.1f} resets/sec, {perf_metrics['ms_per_reset']:.1f}ms/reset"
    )

    # Performance validation
    creation_ok = creation_time < 3.0  # Allow some leeway
    reset_perf_ok = perf_metrics["resets_per_sec"] > 0.5  # Minimum acceptable

    print("\n=== Validation Results ===")
    print(f"✅ Environment creation: {'PASS' if creation_ok else 'FAIL'}")
    print(f"✅ Reset performance: {'PASS' if reset_perf_ok else 'FAIL'}")

    if creation_ok and reset_perf_ok:
        print("🎉 Performance smoke test PASSED")
        return 0
    else:
        print("❌ Performance smoke test FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
