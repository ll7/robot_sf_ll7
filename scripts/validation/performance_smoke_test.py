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
from collections.abc import Mapping
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
class StepLoopMetrics:
    """Advisory simulator step-loop attribution for performance smoke output.

    Cold-path metrics are always reported in ``first_step_sec``, ``step_loop_sec``,
    and ``steps_per_sec`` and should correspond to the first real ``env.step``
    after a reset.
    """

    step_samples: int
    first_step_sec: float
    step_loop_sec: float
    steady_step_loop_sec: float
    steps_per_sec: float
    steady_steps_per_sec: float
    warmup_excluded: bool = False
    warmup_first_step_sec: float | None = None
    warmup_step_loop_sec: float | None = None
    warmup_steps_per_sec: float | None = None
    measurement_mode: str = "cold_only"

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        """Serialize step-loop metrics to a JSON-friendly mapping."""
        return {
            "step_samples": self.step_samples,
            "first_step_sec": self.first_step_sec,
            "step_loop_sec": self.step_loop_sec,
            "steady_step_loop_sec": self.steady_step_loop_sec,
            "steps_per_sec": self.steps_per_sec,
            "steady_steps_per_sec": self.steady_steps_per_sec,
            "warmup_excluded": self.warmup_excluded,
            "warmup_first_step_sec": self.warmup_first_step_sec,
            "warmup_step_loop_sec": self.warmup_step_loop_sec,
            "warmup_steps_per_sec": self.warmup_steps_per_sec,
            "measurement_mode": self.measurement_mode,
        }


@dataclass(slots=True)
class ScenarioProfileMetadata:
    """Minimal scenario metadata for reproducible step-profile snapshots."""

    scenario_id: str | None
    scenario_name: str | None
    scenario_path: str | None
    density: str | None = None
    density_advisory: str | None = None


@dataclass(slots=True)
class StepProfileMetrics:
    """Deterministic-ish diagnostic profile emitted for high-density snapshots."""

    step_samples: int
    first_step_sec: float
    step_loop_sec: float
    steady_step_loop_sec: float
    steps_per_sec: float
    steady_steps_per_sec: float
    warmup_excluded: bool = False
    warmup_first_step_sec: float | None = None
    warmup_step_loop_sec: float | None = None
    warmup_steps_per_sec: float | None = None
    measurement_mode: str = "cold_only"
    scenario_id: str | None = None
    scenario_name: str | None = None
    scenario_path: str | None = None
    density: str | None = None
    density_advisory: str | None = None
    pedestrian_count: int | None = None
    advisory: bool = True
    gating: str = "non-gating"

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        """Serialize step-profile diagnostics to a JSON-friendly mapping."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "scenario_path": self.scenario_path,
            "density": self.density,
            "density_advisory": self.density_advisory,
            "step_samples": self.step_samples,
            "first_step_sec": self.first_step_sec,
            "step_loop_sec": self.step_loop_sec,
            "steady_step_loop_sec": self.steady_step_loop_sec,
            "steps_per_sec": self.steps_per_sec,
            "steady_steps_per_sec": self.steady_steps_per_sec,
            "warmup_excluded": self.warmup_excluded,
            "warmup_first_step_sec": self.warmup_first_step_sec,
            "warmup_step_loop_sec": self.warmup_step_loop_sec,
            "warmup_steps_per_sec": self.warmup_steps_per_sec,
            "measurement_mode": self.measurement_mode,
            "pedestrian_count": self.pedestrian_count,
            "advisory": self.advisory,
            "gating": self.gating,
        }


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
    step_loop: StepLoopMetrics
    step_profile: StepProfileMetrics | None = None
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
            "step_loop": self.step_loop.to_dict(),
            "thresholds": self.thresholds,
            "statuses": self.statuses,
            "scenario": self.scenario,
            "step_profile": self.step_profile.to_dict() if self.step_profile is not None else None,
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


def _coerce_count(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value.is_integer() and value >= 0.0:
            return int(value)
        return None
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[0]) if shape[0] >= 0 else None
    try:
        size = len(value)
    except Exception:  # pragma: no cover - defensive for odd value objects
        size = None
    if size is not None and isinstance(size, int):
        return size if size >= 0 else None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _extract_pedestrian_count(obj: Any) -> int | None:
    if obj is None:
        return None
    for path in (
        ("simulator", "ped_pos"),
        ("sim", "ped_pos"),
        ("simulator", "pysf_state", "num_peds"),
        ("sim", "pysf_state", "num_peds"),
    ):
        current: Any = obj
        for part in path:
            if not hasattr(current, part):
                current = None
                break
            current = getattr(current, part)
        if current is None:
            continue
        count = _coerce_count(current)
        if count is not None:
            return count
    return None


def _measure_profile_pedestrian_count(config: RobotSimulationConfig | None = None) -> int | None:
    """Sample a single pedestrian count from the profiling env."""

    env_config = RobotSimulationConfig() if config is None else copy.deepcopy(config)
    env = make_robot_env(config=env_config, debug=False)
    action = (0.0, 0.0)
    try:
        env.reset()
        env.step(action)
        return _extract_pedestrian_count(env)
    except Exception as exc:  # pragma: no cover - defensive when env APIs drift
        print(f"Unable to sample profiling pedestrian count: {exc}", file=sys.stderr)
        return None
    finally:
        env.close()


def measure_step_loop_performance(  # noqa: C901
    step_samples: int = 10,
    config: RobotSimulationConfig | None = None,
    warmup_steps: int = 0,
) -> StepLoopMetrics:
    """Measure first-step and steady step-loop attribution.

    The result is advisory telemetry only. Existing smoke thresholds still gate
    environment creation and reset throughput, while these fields help diagnose
    whether future slowdowns are startup- or steady-step dominated.
    """

    if step_samples <= 0:
        raise ValueError("step_samples must be greater than zero")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    print("\n=== Step Loop Attribution ===")
    env = make_robot_env(config=copy.deepcopy(config or RobotSimulationConfig()), debug=False)
    action = (0.0, 0.0)

    try:
        env.reset()

        warmup_excluded = False
        warmup_first_step_sec = None
        warmup_step_loop_sec = None
        warmup_steps_per_sec = None
        measurement_mode = "cold_only"

        terminated = False
        truncated = False
        if warmup_steps > 0:
            measurement_mode = "cold_with_warmup"
            warmup_excluded = True
            warm_start = time.time()
            for sample in range(warmup_steps):
                if sample > 0:
                    # Keep the warmup run independent of the measured cold path.
                    if terminated or truncated:
                        env.reset()
                step_start = time.time()
                _, _, terminated, truncated, _ = env.step(action)
                if sample == 0:
                    warmup_first_step_sec = time.time() - step_start
            warmup_step_loop_sec = time.time() - warm_start
            warmup_steps_per_sec = (
                warmup_steps / warmup_step_loop_sec if warmup_step_loop_sec > 0 else 0.0
            )
            env.reset()

        loop_start = time.time()
        first_step_start = time.time()
        _, _, terminated, truncated, _ = env.step(action)
        first_step_sec = time.time() - first_step_start

        steady_start = time.time()
        for index in range(1, step_samples):
            if terminated or truncated:
                env.reset()
            _, _, terminated, truncated, _ = env.step(action)
        steady_step_loop_sec = time.time() - steady_start
        step_loop_sec = time.time() - loop_start
    finally:
        env.close()

    steps_per_sec = step_samples / step_loop_sec if step_loop_sec > 0 else 0.0
    steady_samples = max(0, step_samples - 1)
    steady_steps_per_sec = (
        steady_samples / steady_step_loop_sec
        if steady_samples > 0 and steady_step_loop_sec > 0
        else 0.0
    )

    print(f"Step samples: {step_samples}")
    print(f"First step: {first_step_sec:.3f}s")
    print(f"Total step loop: {step_loop_sec:.3f}s ({steps_per_sec:.2f} steps/sec)")
    if steady_samples:
        print(
            f"Steady step loop: {steady_step_loop_sec:.3f}s ({steady_steps_per_sec:.2f} steps/sec)",
        )
    else:
        print("Steady step loop: not available (single step sample)")

    return StepLoopMetrics(
        step_samples=step_samples,
        first_step_sec=first_step_sec,
        step_loop_sec=step_loop_sec,
        steady_step_loop_sec=steady_step_loop_sec,
        steps_per_sec=steps_per_sec,
        steady_steps_per_sec=steady_steps_per_sec,
        warmup_excluded=warmup_excluded,
        warmup_first_step_sec=warmup_first_step_sec,
        warmup_step_loop_sec=warmup_step_loop_sec,
        warmup_steps_per_sec=warmup_steps_per_sec,
        measurement_mode=measurement_mode,
    )


def measure_step_profile(
    *,
    step_samples: int = 10,
    config: RobotSimulationConfig | None = None,
    step_loop: StepLoopMetrics | None = None,
    scenario_metadata: ScenarioProfileMetadata | None = None,
) -> StepProfileMetrics:
    """Build diagnostic step-profile metadata for the smoke test contract."""

    loop_metrics = step_loop or measure_step_loop_performance(
        step_samples=step_samples, config=config
    )
    pedestrian_count = _measure_profile_pedestrian_count(config=config)
    return StepProfileMetrics(
        step_samples=loop_metrics.step_samples,
        first_step_sec=loop_metrics.first_step_sec,
        step_loop_sec=loop_metrics.step_loop_sec,
        steady_step_loop_sec=loop_metrics.steady_step_loop_sec,
        steps_per_sec=loop_metrics.steps_per_sec,
        steady_steps_per_sec=loop_metrics.steady_steps_per_sec,
        warmup_excluded=loop_metrics.warmup_excluded,
        warmup_first_step_sec=loop_metrics.warmup_first_step_sec,
        warmup_step_loop_sec=loop_metrics.warmup_step_loop_sec,
        warmup_steps_per_sec=loop_metrics.warmup_steps_per_sec,
        measurement_mode=loop_metrics.measurement_mode,
        scenario_id=scenario_metadata.scenario_id if scenario_metadata is not None else None,
        scenario_name=scenario_metadata.scenario_name if scenario_metadata is not None else None,
        scenario_path=scenario_metadata.scenario_path if scenario_metadata is not None else None,
        density=scenario_metadata.density if scenario_metadata is not None else None,
        density_advisory=scenario_metadata.density_advisory
        if scenario_metadata is not None
        else None,
        pedestrian_count=pedestrian_count,
    )


def run_performance_smoke_test(  # noqa: PLR0913
    *,
    num_resets: int = 5,
    step_samples: int = 10,
    warmup_steps: int = 0,
    scenario: str | None = None,
    scenario_name: str | None = None,
    include_recommendations: bool = True,
    creation_soft: float | None = None,
    creation_hard: float | None = None,
    reset_soft: float | None = None,
    reset_hard: float | None = None,
    enforce: bool | None = None,
    on_ci: bool | None = None,
) -> SmokeTestResult:
    """Run creation/reset performance checks and return a structured result.

    Args:
        num_resets: Number of resets to run for throughput measurement.
        step_samples: Number of measured simulator steps for advisory attribution.
        warmup_steps: Optional untimed steps before the measured loop. When non-zero,
            warm-start fields are populated and cold fields are flagged with
            ``warmup_excluded`` so callers do not treat them as cold-start timings.
        scenario: Optional scenario config path used to load a custom config.
        scenario_name: Optional scenario name/id to select from a multi-scenario config.
        include_recommendations: Include guidance strings in the result payload.
        creation_soft: Soft threshold (seconds) for environment creation time.
        creation_hard: Hard threshold (seconds) for environment creation time.
        reset_soft: Soft threshold (resets/sec) for environment reset throughput.
        reset_hard: Hard threshold (resets/sec) for environment reset throughput.
        enforce: When True, treat soft breaches as failures.
        on_ci: Override CI detection (defaults to GITHUB_ACTIONS env var).

    Returns:
        SmokeTestResult containing metrics, status labels, and recommendations.
    """
    if step_samples <= 0:
        raise ValueError("step_samples must be greater than zero")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

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
    config, scenario_label, scenario_metadata = _load_scenario_config(
        scenario, scenario_name=scenario_name
    )
    creation_time = measure_environment_creation(config)
    perf_metrics = measure_environment_performance(num_resets, config=config)
    step_loop = measure_step_loop_performance(
        step_samples,
        config=config,
        warmup_steps=warmup_steps,
    )
    step_profile = measure_step_profile(
        step_samples=step_samples,
        config=config,
        step_loop=step_loop,
        scenario_metadata=scenario_metadata,
    )
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
        step_loop=step_loop,
        step_profile=step_profile,
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
        "--step-samples",
        type=int,
        default=10,
        help="Number of simulator steps for advisory startup/steady attribution (default: 10)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=(
            "Optional untimed simulator steps before measured attribution. "
            "Non-zero values populate warm-start fields and set warmup_excluded=true."
        ),
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
        "--scenario-name",
        type=str,
        help="Optional scenario name/id to select from a multi-scenario config",
    )
    parser.add_argument(
        "--include-recommendations",
        action="store_true",
        help="Emit recommendation guidance when thresholds breach",
    )
    return parser.parse_args()


def main() -> int:  # noqa: C901
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
        step_samples=args.step_samples,
        warmup_steps=args.warmup_steps,
        scenario=args.scenario,
        scenario_name=args.scenario_name,
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
    print(
        "Step attribution: "
        f"first={result.step_loop.first_step_sec:.3f}s, "
        f"loop={result.step_loop.step_loop_sec:.3f}s "
        f"({result.step_loop.steps_per_sec:.2f} steps/sec), "
        f"steady={result.step_loop.steady_step_loop_sec:.3f}s "
        f"({result.step_loop.steady_steps_per_sec:.2f} steady steps/sec), "
        f"measurement_mode={result.step_loop.measurement_mode}, "
        f"warmup_excluded={result.step_loop.warmup_excluded}",
    )
    if result.step_loop.warmup_excluded:
        print(
            f"Warmup attribution: first={result.step_loop.warmup_first_step_sec:.3f}s "
            f"loop={result.step_loop.warmup_step_loop_sec:.3f}s "
            f"({result.step_loop.warmup_steps_per_sec:.2f} steps/sec)",
        )
    if result.step_profile is not None:
        print(
            f"Step profile: {result.step_profile.advisory=} gating={result.step_profile.gating} "
            f"scenario={result.step_profile.scenario_id} ped_count={result.step_profile.pedestrian_count}",
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
        "measurement_mode": result.step_loop.measurement_mode,
        "warmup_excluded": result.step_loop.warmup_excluded,
        "warmup_first_step_sec": result.step_loop.warmup_first_step_sec,
        "warmup_step_loop_sec": result.step_loop.warmup_step_loop_sec,
        "warmup_steps_per_sec": result.step_loop.warmup_steps_per_sec,
        "first_step_sec": result.step_loop.first_step_sec,
        "step_loop_sec": result.step_loop.step_loop_sec,
        "steady_step_loop_sec": result.step_loop.steady_step_loop_sec,
        "sim_steps_per_sec": result.step_loop.steps_per_sec,
        "steady_sim_steps_per_sec": result.step_loop.steady_steps_per_sec,
        "step_profile_advisory": result.step_profile.advisory
        if result.step_profile is not None
        else None,
        "step_profile_gating": result.step_profile.gating
        if result.step_profile is not None
        else None,
        "step_profile_scenario_id": result.step_profile.scenario_id
        if result.step_profile is not None
        else None,
        "step_profile_pedestrian_count": result.step_profile.pedestrian_count
        if result.step_profile is not None
        else None,
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
    *,
    scenario_name: str | None = None,
) -> tuple[RobotSimulationConfig, str | None, ScenarioProfileMetadata | None]:
    """Load a scenario config if available and return config plus scenario id/path."""

    if not scenario:
        return RobotSimulationConfig(), None, None
    scenario_path = Path(scenario)
    if not scenario_path.exists():
        print(f"Scenario file not found: {scenario_path} (using default config)")
        return RobotSimulationConfig(), scenario, None
    try:
        scenarios = load_scenarios(scenario_path)
        selected = select_scenario(scenarios, scenario_name)
    except ValueError as exc:
        print(f"Invalid scenario config '{scenario_path}': {exc} (using default config)")
        return (
            RobotSimulationConfig(),
            scenario_name or scenario_path.stem,
            ScenarioProfileMetadata(
                scenario_id=scenario_name or scenario_path.stem,
                scenario_name=scenario_name or scenario_path.stem,
                scenario_path=str(scenario_path),
            ),
        )
    config = build_robot_config_from_scenario(selected, scenario_path=scenario_path)
    scenario_id = str(
        selected.get("name") or selected.get("scenario_id") or scenario_path.stem,
    )
    scenario_metadata = ScenarioProfileMetadata(
        scenario_id=scenario_id,
        scenario_name=str(
            selected.get("name") or selected.get("scenario_id") or scenario_path.stem
        ),
        scenario_path=str(scenario_path),
        density=_scenario_metadata_text(selected, "density"),
        density_advisory=_scenario_metadata_text(selected, "density_advisory"),
    )
    return config, scenario_id, scenario_metadata


def _scenario_metadata_text(scenario: Mapping[str, Any], key: str) -> str | None:
    """Return a string metadata value from a scenario entry when present."""

    metadata = scenario.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    if value is None:
        return None
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
