#!/usr/bin/env python3
"""Performance Smoke Test for Social Navigation Benchmark.

Purpose:
    Measure baseline performance characteristics (environment creation + reset throughput)
    and guard against severe performance regressions without creating unnecessary CI flakes.

Design:
    We use two threshold tiers (soft + hard). Soft thresholds express the *expected* range;
    breaching them triggers a WARNING. Hard thresholds represent a severe regression and
    trigger a non‑zero exit status (FAIL).

Threshold strategy (defaults):
    - Environment creation soft < 3.0s, hard < 8.0s
    - Reset throughput soft > 0.50 resets/sec, hard > 0.20 resets/sec

Environment variable overrides (all optional):
    ROBOT_SF_PERF_CREATION_SOFT   (float)
    ROBOT_SF_PERF_CREATION_HARD   (float)
    ROBOT_SF_PERF_RESET_SOFT      (float)
    ROBOT_SF_PERF_RESET_HARD      (float)
    ROBOT_SF_PERF_ENFORCE         ("1" to fail on soft threshold breach; default relaxed)

CI behaviour:
    On GitHub Actions (GITHUB_ACTIONS == "true"), soft threshold breaches become warnings
    unless ROBOT_SF_PERF_ENFORCE=1 is set. This reduces spurious failures caused by transient
    shared runner slowdowns while still catching catastrophic regressions (hard breaches).

Outputs:
    - JSON summary written to results/performance_smoke_test.json
    - Structured PASS/WARN/FAIL lines for quick parsing

Exit codes:
    0 → All metrics within hard thresholds (soft breaches may have produced WARN)
    1 → At least one hard threshold breached OR (enforce mode) a soft breach occurred

See also: docs/dev_guide.md performance section for rationale and tuning guidance.
"""

import os
import sys
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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        print(f"WARNING: Invalid float for {name}; using default {default}", file=sys.stderr)
        return default


def main() -> int:
    """Run performance smoke tests with soft/hard threshold evaluation."""
    print("Social Navigation Benchmark - Performance Smoke Test")
    print("=" * 60)

    # Threshold configuration (allow env overrides)
    creation_soft = _env_float("ROBOT_SF_PERF_CREATION_SOFT", 3.0)
    creation_hard = _env_float("ROBOT_SF_PERF_CREATION_HARD", 8.0)
    reset_soft = _env_float("ROBOT_SF_PERF_RESET_SOFT", 0.50)
    reset_hard = _env_float("ROBOT_SF_PERF_RESET_HARD", 0.20)
    enforce = os.environ.get("ROBOT_SF_PERF_ENFORCE", "0") == "1"
    on_ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"

    if enforce:
        print("[Mode] ENFORCING soft performance thresholds (ROBOT_SF_PERF_ENFORCE=1)")
    else:
        mode = "CI-relaxed" if on_ci else "local-relaxed"
        print(f"[Mode] Soft threshold breaches become WARN ({mode}); hard thresholds still FAIL")

    # Measure
    creation_time = measure_environment_creation()
    perf_metrics = measure_environment_performance()

    resets_per_sec = perf_metrics["resets_per_sec"]

    # Persist results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    import json

    performance_log = results_dir / "performance_smoke_test.json"
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment_creation_sec": creation_time,
        "resets_per_sec": resets_per_sec,
        "ms_per_reset": perf_metrics["ms_per_reset"],
        "total_test_resets": perf_metrics["total_resets"],
        "total_test_time_sec": perf_metrics["total_time"],
        "thresholds": {
            "creation_soft": creation_soft,
            "creation_hard": creation_hard,
            "reset_soft": reset_soft,
            "reset_hard": reset_hard,
            "enforce": enforce,
            "ci": on_ci,
        },
    }
    with performance_log.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nPerformance results saved to: {performance_log}")

    # Evaluate
    creation_soft_ok = creation_time <= creation_soft
    creation_hard_ok = creation_time <= creation_hard
    reset_soft_ok = resets_per_sec >= reset_soft
    reset_hard_ok = resets_per_sec >= reset_hard

    print("\n=== Thresholds ===")
    print(f"Creation soft<= {creation_soft:.2f}s | hard<= {creation_hard:.2f}s")
    print(f"Reset soft>= {reset_soft:.2f}/s | hard>= {reset_hard:.2f}/s")

    def status(soft_ok: bool, hard_ok: bool) -> str:
        if hard_ok and soft_ok:
            return "PASS"
        if hard_ok and not soft_ok:
            return "WARN" if not enforce and on_ci else "FAIL" if enforce else "WARN"
        return "FAIL"

    creation_status = status(creation_soft_ok, creation_hard_ok)
    reset_status = status(reset_soft_ok, reset_hard_ok)

    print("\n=== Validation Results ===")
    print(f"Environment creation: {creation_status} ({creation_time:.2f}s)")
    print(
        "Reset throughput: "
        f"{reset_status} ({resets_per_sec:.2f} resets/sec, {perf_metrics['ms_per_reset']:.2f} ms/reset)",
    )

    hard_fail = not creation_hard_ok or not reset_hard_ok
    soft_fail_enforced = enforce and (not creation_soft_ok or not reset_soft_ok)

    if hard_fail:
        print("❌ Performance smoke test FAILED (hard threshold breach)")
        return 1
    if soft_fail_enforced:
        print("❌ Performance smoke test FAILED (soft threshold breach under enforce mode)")
        return 1

    # If we reach here, at worst we emitted WARN lines
    if creation_status == "PASS" and reset_status == "PASS":
        print("🎉 Performance smoke test PASSED (all soft thresholds met)")
    else:
        print("⚠️ Performance smoke test PASSED with WARNINGS (see above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
