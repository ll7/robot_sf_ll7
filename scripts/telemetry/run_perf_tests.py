#!/usr/bin/env python3
"""Telemetry-aware wrapper for the performance smoke test.

This helper invokes ``scripts/validation/performance_smoke_test.py`` via its
public API, persists the structured results next to the run-tracker manifest,
and records a ``PerformanceTestResult`` entry so ``run_tracker_cli`` can display
historical throughput comparisons.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.telemetry import (
    ManifestWriter,
    PerformanceTestResult,
    PerformanceTestStatus,
    PipelineRunRecord,
    PipelineRunStatus,
    RunTrackerConfig,
    generate_run_id,
)
from scripts.validation import performance_smoke_test

PERF_STEP_ID = "performance_smoke_test"


def run_perf_tests(
    *,
    scenario: str | None,
    output_hint: str | None,
    num_resets: int,
    artifact_root: Path | None = None,
) -> tuple[int, Path | None]:
    """Execute the smoke test and record tracker artifacts.

    Returns the smoke-test exit code and the run directory containing artifacts.
    """

    if num_resets <= 0:
        raise ValueError("num_resets must be greater than zero")

    base_config = RunTrackerConfig(artifact_root=artifact_root)
    config, run_id = _resolve_output_destination(base_config, output_hint)
    try:
        writer = ManifestWriter(config, run_id)
    except FileExistsError as exc:  # pragma: no cover - surfaced during manual use
        msg = f"Run tracker directory already exists for run_id={run_id}. Choose a new --output."
        raise RuntimeError(msg) from exc

    scenario_path = Path(scenario) if scenario else None
    scenario_config_path = scenario_path if scenario_path and scenario_path.exists() else None
    started_at = datetime.now(UTC)
    result = performance_smoke_test.run_performance_smoke_test(
        num_resets=num_resets,
        scenario=scenario,
        include_recommendations=True,
    )
    scenario_label = result.scenario or scenario

    recommendations = list(result.recommendations)
    for index, recommendation in enumerate(recommendations):
        recommendation.evidence.setdefault("recommendation_index", index)

    summary_path = _write_summary(writer.run_directory, run_id, scenario_label, result)
    perf_status = _map_test_status(result.statuses.get("overall", "FAIL"))
    recommendation_count = len(recommendations)

    summary = _build_manifest_summary(
        result,
        summary_path,
        scenario_label,
        num_resets,
        recommendation_count,
    )

    if recommendations:
        writer.append_recommendations(recommendations)

    perf_test = PerformanceTestResult(
        test_id=scenario_label or PERF_STEP_ID,
        matrix=scenario_label or "default",
        throughput_baseline=float(result.thresholds.get("reset_soft", 0.0)),
        throughput_measured=result.resets_per_sec,
        duration_seconds=result.total_time_sec,
        status=perf_status,
        recommendations_ref=tuple(range(recommendation_count)) if recommendation_count else (),
    )
    writer.append_performance_test(perf_test)

    record = PipelineRunRecord(
        run_id=run_id,
        created_at=started_at,
        completed_at=result.timestamp,
        status=PipelineRunStatus.COMPLETED if result.exit_code == 0 else PipelineRunStatus.FAILED,
        enabled_steps=(PERF_STEP_ID,),
        artifact_dir=writer.run_directory,
        scenario_config_path=scenario_config_path,
        summary=summary,
    )
    writer.append_run_record(record)

    return result.exit_code, writer.run_directory


def _write_summary(
    run_directory: Path,
    run_id: str,
    scenario: str | None,
    result: performance_smoke_test.SmokeTestResult,
) -> Path:
    """Write summary.

    Args:
        run_directory: Auto-generated placeholder description.
        run_id: Auto-generated placeholder description.
        scenario: Auto-generated placeholder description.
        result: Auto-generated placeholder description.

    Returns:
        Path: Auto-generated placeholder description.
    """
    payload = result.to_dict()
    payload["run_id"] = run_id
    if scenario is not None:
        payload["scenario"] = scenario
    summary_path = run_directory / "perf_test_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def _build_manifest_summary(
    result: performance_smoke_test.SmokeTestResult,
    summary_path: Path,
    scenario: str | None,
    num_resets: int,
    recommendation_count: int,
) -> dict[str, Any]:
    """Build manifest summary.

    Args:
        result: Auto-generated placeholder description.
        summary_path: Auto-generated placeholder description.
        scenario: Auto-generated placeholder description.
        num_resets: Auto-generated placeholder description.
        recommendation_count: Auto-generated placeholder description.

    Returns:
        dict[str, Any]: Auto-generated placeholder description.
    """
    summary: dict[str, Any] = {
        "scenario": scenario or "default",
        "num_resets": num_resets,
        "creation_seconds": round(result.creation_seconds, 3),
        "resets_per_sec": round(result.resets_per_sec, 3),
        "ms_per_reset": round(result.ms_per_reset, 3),
        "total_time_sec": round(result.total_time_sec, 3),
        "status": result.statuses.get("overall", "unknown"),
        "thresholds": result.thresholds,
        "json_summary": str(summary_path),
    }
    if recommendation_count:
        summary["recommendation_count"] = recommendation_count
    return summary


def _resolve_output_destination(
    config: RunTrackerConfig,
    hint: str | None,
) -> tuple[RunTrackerConfig, str]:
    """Resolve output destination.

    Args:
        config: Auto-generated placeholder description.
        hint: Auto-generated placeholder description.

    Returns:
        tuple[RunTrackerConfig, str]: Auto-generated placeholder description.
    """
    run_id = generate_run_id("perf")
    if not hint:
        return config, run_id
    candidate = Path(hint).expanduser()
    parts = candidate.parts
    if "run-tracker" in parts:
        index = parts.index("run-tracker")
        tracker_root = Path(*parts[: index + 1])
        run_segment = Path(*parts[index + 1 :])
        if run_segment.parts:
            tracker_parent = (
                tracker_root.parent if tracker_root.parent != Path(".") else config.artifact_root
            )
            return RunTrackerConfig(artifact_root=tracker_parent), run_segment.as_posix()
        return config, run_id
    if candidate.is_absolute() or candidate.parent != Path("."):
        artifact_root = candidate.parent if candidate.parent != Path(".") else config.artifact_root
        return RunTrackerConfig(artifact_root=artifact_root), candidate.name or run_id
    return config, candidate.as_posix() or run_id


def _map_test_status(label: str) -> PerformanceTestStatus:
    """Map test status.

    Args:
        label: Auto-generated placeholder description.

    Returns:
        PerformanceTestStatus: Auto-generated placeholder description.
    """
    normalized = label.strip().upper() if label else "FAIL"
    if normalized == "PASS":
        return PerformanceTestStatus.PASSED
    if normalized == "WARN":
        return PerformanceTestStatus.SOFT_BREACH
    return PerformanceTestStatus.FAILED


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse args.

    Args:
        argv: Auto-generated placeholder description.

    Returns:
        argparse.Namespace: Auto-generated placeholder description.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        type=str,
        help="Optional scenario config (YAML) applied to the perf smoke test",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional run identifier or tracker path for perf-test artifacts",
    )
    parser.add_argument(
        "--num-resets",
        type=int,
        default=5,
        help="Number of environment resets to benchmark (default: 5)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Override the artifact root (defaults to output/)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main.

    Args:
        argv: Auto-generated placeholder description.

    Returns:
        int: Auto-generated placeholder description.
    """
    args = parse_args(argv)
    try:
        exit_code, run_dir = run_perf_tests(
            scenario=args.scenario,
            output_hint=args.output,
            num_resets=args.num_resets,
            artifact_root=args.artifact_root,
        )
    except RuntimeError as exc:
        print(f"Performance test failed: {exc}")
        return 2
    except ValueError as exc:
        print(f"Invalid arguments: {exc}")
        return 2
    if run_dir is not None:
        print(f"Perf test artifacts written to {run_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
