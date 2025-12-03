#!/usr/bin/env python3
"""Performance Smoke Test for Social Navigation Benchmark with telemetry hooks.

Purpose:
    Measure baseline performance characteristics (environment creation + reset throughput)
    and guard against regressions while optionally emitting telemetry/recommendation data.

Threshold strategy (defaults):
    - Environment creation soft < 3.0s, hard < 8.0s
    - Reset throughput soft > 0.50 resets/sec, hard > 0.20 resets/sec

Environment variables (all optional):
    ROBOT_SF_PERF_CREATION_SOFT / HARD
    ROBOT_SF_PERF_RESET_SOFT / HARD
    ROBOT_SF_PERF_ENFORCE ("1" to fail on soft breach)

Outputs:
    - JSON summary under the benchmarks artifact category (override with --json-output)
    - Optional telemetry JSONL snapshot (--telemetry-output)
    - PASS/WARN/FAIL console output with recommendation hints
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.telemetry.models import (
    PerformanceRecommendation,
    RecommendationSeverity,
    serialize_payload,
)
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)


@dataclass(slots=True)
class SmokeTestResult:
    """Structured result for performance smoke tests."""

    timestamp: datetime
    creation_seconds: float
    resets_per_sec: float
    ms_per_reset: float
    total_resets: int
    total_time_sec: float
    thresholds: dict[str, float | bool]
    statuses: dict[str, str]
    recommendations: tuple[PerformanceRecommendation, ...] = ()
    scenario: str | None = None
    exit_code: int = 0

    def to_dict(self) -> dict[str, Any]:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        payload = {
            "timestamp": self.timestamp.isoformat(timespec="seconds"),
            "environment_creation_sec": self.creation_seconds,
            "resets_per_sec": self.resets_per_sec,
            "ms_per_reset": self.ms_per_reset,
            "total_test_resets": self.total_resets,
            "total_test_time_sec": self.total_time_sec,
            "thresholds": self.thresholds,
            "statuses": self.statuses,
            "scenario": self.scenario,
            "exit_code": self.exit_code,
            "recommendations": [serialize_payload(rec) for rec in self.recommendations],
        }
        return payload


def measure_environment_performance(
    num_resets: int = 5,
    config: RobotSimulationConfig | None = None,
) -> dict[str, float]:
    """Measure environment reset performance (simplified test).

    Args:
        num_resets: Number of environment resets to benchmark
        config: Optional pre-loaded simulation config (scenario overrides applied)
    """

    config = config or RobotSimulationConfig()

    env = make_robot_env(config=copy.deepcopy(config), debug=False)

    total_resets = 0
    total_time = 0.0

    print(f"Running {num_resets} reset cycles...")

    for episode in range(num_resets):
        episode_start = time.time()
        env.reset()
        total_resets += 1
        episode_time = time.time() - episode_start
        total_time += episode_time
        resets_per_sec = 1.0 / episode_time if episode_time > 0 else 0.0
        print(f"Reset {episode + 1}: {episode_time:.3f}s ({resets_per_sec:.1f} resets/sec)")

    env.close()

    overall_resets_per_sec = total_resets / total_time if total_time > 0 else 0.0
    ms_per_reset = (total_time * 1000) / total_resets if total_resets > 0 else 0.0

    print("\n=== Performance Summary ===")
    print(f"Total resets: {total_resets}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average resets/sec: {overall_resets_per_sec:.1f}")
    print(f"Average ms/reset: {ms_per_reset:.1f}")

    return {
        "resets_per_sec": overall_resets_per_sec,
        "ms_per_reset": ms_per_reset,
        "total_resets": float(total_resets),
        "total_time": total_time,
    }


def measure_environment_creation(config: RobotSimulationConfig | None = None) -> float:
    """Measure environment creation time."""

    print("\n=== Environment Creation Performance ===")
    start_time = time.time()
    env = make_robot_env(config=copy.deepcopy(config or RobotSimulationConfig()), debug=False)
    creation_time = time.time() - start_time
    env.close()
    print(f"Environment creation time: {creation_time:.2f}s")
    return creation_time


def run_performance_smoke_test(
    *,
    num_resets: int = 5,
    scenario: str | None = None,
    include_recommendations: bool = True,
    creation_soft: float | None = None,
    creation_hard: float | None = None,
    reset_soft: float | None = None,
    reset_hard: float | None = None,
    enforce: bool | None = None,
    on_ci: bool | None = None,
) -> SmokeTestResult:
    """TODO docstring. Document this function.

    Args:
        num_resets: TODO docstring.
        scenario: TODO docstring.
        include_recommendations: TODO docstring.
        creation_soft: TODO docstring.
        creation_hard: TODO docstring.
        reset_soft: TODO docstring.
        reset_hard: TODO docstring.
        enforce: TODO docstring.
        on_ci: TODO docstring.

    Returns:
        TODO docstring.
    """
    creation_soft = (
        creation_soft
        if creation_soft is not None
        else _env_float("ROBOT_SF_PERF_CREATION_SOFT", 3.0)
    )
    creation_hard = (
        creation_hard
        if creation_hard is not None
        else _env_float("ROBOT_SF_PERF_CREATION_HARD", 8.0)
    )
    reset_soft = (
        reset_soft if reset_soft is not None else _env_float("ROBOT_SF_PERF_RESET_SOFT", 0.50)
    )
    reset_hard = (
        reset_hard if reset_hard is not None else _env_float("ROBOT_SF_PERF_RESET_HARD", 0.20)
    )
    enforce = (
        enforce if enforce is not None else os.environ.get("ROBOT_SF_PERF_ENFORCE", "0") == "1"
    )
    on_ci = on_ci if on_ci is not None else os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    config, scenario_label = _load_scenario_config(scenario)
    creation_time = measure_environment_creation(config)
    perf_metrics = measure_environment_performance(num_resets, config=config)
    resets_per_sec = perf_metrics["resets_per_sec"]

    creation_soft_ok = creation_time <= creation_soft
    creation_hard_ok = creation_time <= creation_hard
    reset_soft_ok = resets_per_sec >= reset_soft
    reset_hard_ok = resets_per_sec >= reset_hard

    statuses = {
        "creation": _status_label(creation_soft_ok, creation_hard_ok, enforce),
        "reset": _status_label(reset_soft_ok, reset_hard_ok, enforce),
    }
    if "FAIL" in statuses.values():
        overall = "FAIL"
    elif "WARN" in statuses.values():
        overall = "WARN"
    else:
        overall = "PASS"
    statuses["overall"] = overall

    hard_fail = not creation_hard_ok or not reset_hard_ok
    soft_breach = not creation_soft_ok or not reset_soft_ok
    soft_fail_enforced = soft_breach and enforce
    exit_code = 1 if hard_fail or soft_fail_enforced else 0

    thresholds = {
        "creation_soft": creation_soft,
        "creation_hard": creation_hard,
        "reset_soft": reset_soft,
        "reset_hard": reset_hard,
        "enforce": enforce,
        "ci": on_ci,
    }

    recommendations: tuple[PerformanceRecommendation, ...] = ()
    if include_recommendations:
        recommendations = _build_recommendations(
            statuses,
            creation_time,
            resets_per_sec,
            thresholds,
        )

    return SmokeTestResult(
        timestamp=datetime.now(UTC),
        creation_seconds=creation_time,
        resets_per_sec=resets_per_sec,
        ms_per_reset=perf_metrics["ms_per_reset"],
        total_resets=int(perf_metrics["total_resets"]),
        total_time_sec=perf_metrics["total_time"],
        thresholds=thresholds,
        statuses=statuses,
        recommendations=recommendations,
        scenario=scenario_label or scenario,
        exit_code=exit_code,
    )


def parse_args() -> argparse.Namespace:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--num-resets",
        type=int,
        default=5,
        help="Number of environment resets to benchmark (default: 5)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Override the JSON summary path (defaults to output/benchmarks/performance_smoke_test.json)",
    )
    parser.add_argument(
        "--telemetry-output",
        type=Path,
        help="Optional telemetry JSONL file to append a summary snapshot",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        help="Optional scenario config (YAML) applied before measuring performance",
    )
    parser.add_argument(
        "--include-recommendations",
        action="store_true",
        help="Emit recommendation guidance when thresholds breach",
    )
    return parser.parse_args()


def main() -> int:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    print("Social Navigation Benchmark - Performance Smoke Test")
    print("=" * 60)

    args = parse_args()

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

    result = run_performance_smoke_test(
        num_resets=args.num_resets,
        scenario=args.scenario,
        include_recommendations=args.include_recommendations,
        creation_soft=creation_soft,
        creation_hard=creation_hard,
        reset_soft=reset_soft,
        reset_hard=reset_hard,
        enforce=enforce,
        on_ci=on_ci,
    )

    print("\n=== Thresholds ===")
    print(f"Creation soft<= {creation_soft:.2f}s | hard<= {creation_hard:.2f}s")
    print(f"Reset soft>= {reset_soft:.2f}/s | hard>= {reset_hard:.2f}/s")

    print("\n=== Validation Results ===")
    print(f"Environment creation: {result.statuses['creation']} ({result.creation_seconds:.2f}s)")
    print(
        "Reset throughput: "
        f"{result.statuses['reset']} ({result.resets_per_sec:.2f} resets/sec, {result.ms_per_reset:.2f} ms/reset)",
    )

    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            actions = " | ".join(rec.suggested_actions)
            print(f"  - [{rec.severity.value.upper()}] {rec.message} :: {actions}")

    output_path = args.json_output if args.json_output is not None else _default_summary_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"\nPerformance results saved to: {output_path}")

    if args.telemetry_output is not None:
        _write_telemetry_snapshot(args.telemetry_output, result)
        print(f"Telemetry snapshot appended to: {args.telemetry_output}")

    if result.exit_code == 0:
        if result.statuses["overall"] == "PASS":
            print("🎉 Performance smoke test PASSED (all soft thresholds met)")
        elif result.statuses["overall"] == "WARN":
            print("⚠️ Performance smoke test PASSED with WARNINGS (see above)")
    elif not result.recommendations:
        print("❌ Performance smoke test FAILED")
    else:
        print("❌ Performance smoke test FAILED (see recommendations above)")
    return result.exit_code


def _default_summary_path() -> Path:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    ensure_canonical_tree(categories=("benchmarks",))
    return get_artifact_category_path("benchmarks") / "performance_smoke_test.json"


def _status_label(soft_ok: bool, hard_ok: bool, enforce: bool) -> str:
    """TODO docstring. Document this function.

    Args:
        soft_ok: TODO docstring.
        hard_ok: TODO docstring.
        enforce: TODO docstring.

    Returns:
        TODO docstring.
    """
    if hard_ok and soft_ok:
        return "PASS"
    if hard_ok and not soft_ok:
        if enforce:
            return "FAIL"
        return "WARN"
    return "FAIL"


def _build_recommendations(
    statuses: dict[str, str],
    creation_time: float,
    resets_per_sec: float,
    thresholds: dict[str, float | bool],
) -> tuple[PerformanceRecommendation, ...]:
    """TODO docstring. Document this function.

    Args:
        statuses: TODO docstring.
        creation_time: TODO docstring.
        resets_per_sec: TODO docstring.
        thresholds: TODO docstring.

    Returns:
        TODO docstring.
    """
    recommendations: list[PerformanceRecommendation] = []
    timestamp_ms = int(time.time() * 1000)
    if statuses["creation"] != "PASS":
        severity = (
            RecommendationSeverity.CRITICAL
            if statuses["creation"] == "FAIL"
            else RecommendationSeverity.WARNING
        )
        recommendations.append(
            PerformanceRecommendation(
                trigger="env_creation_slow",
                severity=severity,
                message="Environment creation time exceeded baseline thresholds.",
                suggested_actions=(
                    "Reuse initialized environments between tests",
                    "Ensure fast-pysf backend is selected and warmed up",
                ),
                evidence={
                    "creation_seconds": round(creation_time, 3),
                    "soft_limit": thresholds["creation_soft"],
                    "hard_limit": thresholds["creation_hard"],
                },
                timestamp_ms=timestamp_ms,
            ),
        )
    if statuses["reset"] != "PASS":
        severity = (
            RecommendationSeverity.CRITICAL
            if statuses["reset"] == "FAIL"
            else RecommendationSeverity.WARNING
        )
        recommendations.append(
            PerformanceRecommendation(
                trigger="reset_throughput_drop",
                severity=severity,
                message="Environment reset throughput dropped below baseline.",
                suggested_actions=(
                    "Lower reset count or run on dedicated hardware",
                    "Check for background CPU-intensive processes",
                ),
                evidence={
                    "resets_per_sec": round(resets_per_sec, 3),
                    "soft_threshold": thresholds["reset_soft"],
                    "hard_threshold": thresholds["reset_hard"],
                },
                timestamp_ms=timestamp_ms,
            ),
        )
    return tuple(recommendations)


def _write_telemetry_snapshot(path: Path, result: SmokeTestResult) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        result: TODO docstring.
    """
    payload = {
        "timestamp_ms": int(result.timestamp.timestamp() * 1000),
        "step_id": "performance_smoke_test",
        "steps_per_sec": result.resets_per_sec,
        "notes": "perf-smoke-summary",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _env_float(name: str, default: float) -> float:
    """TODO docstring. Document this function.

    Args:
        name: TODO docstring.
        default: TODO docstring.

    Returns:
        TODO docstring.
    """
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        print(f"WARNING: Invalid float for {name}; using default {default}", file=sys.stderr)
        return default


def _load_scenario_config(
    scenario: str | None,
) -> tuple[RobotSimulationConfig, str | None]:
    """Load a scenario config if available and return config plus scenario id."""

    if not scenario:
        return RobotSimulationConfig(), None
    scenario_path = Path(scenario)
    if not scenario_path.exists():
        print(f"Scenario file not found: {scenario_path} (using default config)")
        return RobotSimulationConfig(), scenario
    try:
        scenarios = load_scenarios(scenario_path)
        selected = select_scenario(scenarios, None)
    except ValueError as exc:
        print(f"Invalid scenario config '{scenario_path}': {exc} (using default config)")
        return RobotSimulationConfig(), scenario_path.stem
    config = build_robot_config_from_scenario(selected, scenario_path=scenario_path)
    scenario_id = str(
        selected.get("name") or selected.get("scenario_id") or scenario_path.stem,
    )
    return config, scenario_id


if __name__ == "__main__":
    sys.exit(main())
