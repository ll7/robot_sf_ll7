"""Planner inclusion gate for benchmark promotion review."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import compute_aggregates, read_jsonl
from robot_sf.benchmark.runner import run_batch

DEFAULT_INCLUSION_MATRIX = Path("configs/scenarios/planner_sanity_matrix_v1.yaml")
DEFAULT_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
REPORT_SCHEMA_VERSION = "planner-inclusion-check.v1"


@dataclass(frozen=True)
class InclusionCriteria:
    """Mechanical thresholds for one planner inclusion-check run."""

    min_episodes: int = 1
    min_success_rate: float = 0.5
    max_collision_rate: float = 0.0
    max_runtime_sec: float = 60.0


def _safe_name(value: str) -> str:
    """Convert a planner identifier into a filesystem-safe report stem.

    Returns:
        Sanitized filename stem.
    """
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def _finite_float(value: Any) -> float | None:
    """Return a finite float or None when the value is missing/non-finite."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unwrap_json_value(value: Any) -> Any:
    """Normalize common non-standard scalar/container wrappers before JSON conversion.

    Returns:
        Unwrapped value suitable for further JSON-safe normalization.
    """

    current = value
    for _ in range(4):
        if isinstance(current, Path):
            return current.as_posix()
        if current is None or isinstance(
            current, bool | int | float | str | Mapping | list | tuple
        ):
            return current

        item = getattr(current, "item", None)
        if callable(item):
            try:
                current = item()
                continue
            except (ValueError, TypeError, AttributeError):
                pass

        tolist = getattr(current, "tolist", None)
        if callable(tolist):
            try:
                current = tolist()
                continue
            except (ValueError, TypeError, AttributeError):
                pass

        break
    return current


def _jsonable_leaf(value: Any) -> tuple[bool, Any]:
    """Return whether a value can be emitted as a JSON scalar immediately.

    Returns:
        Pair of ``handled`` flag and normalized scalar candidate.
    """

    item = _unwrap_json_value(value)
    if item is None or isinstance(item, bool | int | str):
        return True, item
    if isinstance(item, float):
        return True, item if math.isfinite(item) else None
    return False, item


def _jsonable_child(value: Any, stack: list[tuple[Any, Any]]) -> Any:
    """Normalize one nested JSON child and queue container work when needed.

    Returns:
        JSON-safe scalar, placeholder container, or stringified fallback.
    """

    handled, item = _jsonable_leaf(value)
    if handled:
        return item
    if isinstance(item, Mapping):
        child: dict[str, Any] = {}
        stack.append((item, child))
        return child
    if isinstance(item, list | tuple):
        child = []
        stack.append((item, child))
        return child
    return str(item)


def _extend_jsonable_mapping(
    source: Mapping[Any, Any],
    target: dict[str, Any],
    stack: list[tuple[Any, Any]],
) -> None:
    """Populate one JSON object container without recursion."""

    for key, raw_value in source.items():
        target[str(key)] = _jsonable_child(raw_value, stack)


def _extend_jsonable_sequence(
    source: list[Any] | tuple[Any, ...],
    target: list[Any],
    stack: list[tuple[Any, Any]],
) -> None:
    """Populate one JSON array container without recursion."""

    for raw_value in source:
        target.append(_jsonable_child(raw_value, stack))


def to_jsonable_payload(value: Any) -> Any:
    """Convert nested payloads into JSON-safe builtins without recursion.

    Returns:
        JSON-safe payload containing only built-in scalar/container types.
    """

    handled, root_value = _jsonable_leaf(value)
    if handled:
        return root_value
    if isinstance(root_value, Mapping):
        root: dict[str, Any] | list[Any] = {}
        stack: list[tuple[Any, Any]] = [(root_value, root)]
    elif isinstance(root_value, list | tuple):
        root = []
        stack = [(root_value, root)]
    else:
        return str(root_value)

    while stack:
        source, target = stack.pop()
        if isinstance(source, Mapping):
            _extend_jsonable_mapping(source, target, stack)
            continue
        _extend_jsonable_sequence(source, target, stack)

    return root


def _aggregate_mean(aggregates: dict[str, Any], group: str, metric: str) -> float | None:
    """Read one aggregate metric mean from a grouped aggregate payload.

    Returns:
        Finite aggregate mean, or None when absent/non-finite.
    """
    group_payload = aggregates.get(group)
    if not isinstance(group_payload, dict):
        return None
    metric_payload = group_payload.get(metric)
    if not isinstance(metric_payload, dict):
        return None
    return _finite_float(metric_payload.get("mean"))


def _non_finite_aggregate_paths(payload: Any, *, prefix: str = "") -> list[str]:
    """Collect paths whose aggregate numeric values are NaN or infinite.

    Returns:
        Dotted/list-index paths for non-finite aggregate values.
    """
    paths: list[str] = []
    stack: list[tuple[Any, str]] = [(payload, prefix)]
    while stack:
        current, current_prefix = stack.pop()
        current = _unwrap_json_value(current)
        if isinstance(current, Mapping):
            for key, value in current.items():
                child_prefix = f"{current_prefix}.{key}" if current_prefix else str(key)
                stack.append((value, child_prefix))
            continue
        if isinstance(current, list | tuple):
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], f"{current_prefix}[{index}]"))
            continue
        if isinstance(current, bool):
            continue
        try:
            number = float(current)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            paths.append(current_prefix)
    return paths


def build_inclusion_report(  # noqa: PLR0913
    *,
    algo: str,
    algo_config: str | None,
    benchmark_profile: str,
    matrix: Path,
    schema: Path,
    output_dir: Path,
    episodes_path: Path,
    summary: dict[str, Any],
    aggregates: dict[str, Any],
    runtime_sec: float,
    criteria: InclusionCriteria,
) -> dict[str, Any]:
    """Build a reviewable planner inclusion report from run and aggregate artifacts.

    Returns:
        Versioned planner inclusion report payload.
    """
    group = str(algo)
    total_jobs = int(summary.get("total_jobs", 0) or 0)
    written = int(summary.get("written", 0) or 0)
    failures = summary.get("failures", [])
    failure_count = (
        len(failures)
        if isinstance(failures, list)
        else int(summary.get("failed_jobs", summary.get("failed", 0)) or 0)
    )
    success_rate = _aggregate_mean(aggregates, group, "success")
    collision_rate = _aggregate_mean(aggregates, group, "collisions")
    non_finite_paths = _non_finite_aggregate_paths(aggregates)

    checks = {
        "schema_valid": {
            "passed": total_jobs > 0 and written == total_jobs and failure_count == 0,
            "observed": {"total_jobs": total_jobs, "written": written, "failures": failure_count},
            "reason": (
                "all scheduled episodes produced schema-valid records"
                if total_jobs > 0 and written == total_jobs and failure_count == 0
                else "runner did not produce schema-valid records for every scheduled episode"
            ),
        },
        "no_nan_aggregates": {
            "passed": not non_finite_paths,
            "observed": {"non_finite_paths": non_finite_paths},
            "reason": (
                "aggregate metrics are finite"
                if not non_finite_paths
                else "aggregate metrics contain NaN or infinite values"
            ),
        },
        "bounded_runtime": {
            "passed": runtime_sec <= criteria.max_runtime_sec,
            "observed": runtime_sec,
            "threshold": {"max_runtime_sec": criteria.max_runtime_sec},
            "reason": (
                "runtime is within the configured bound"
                if runtime_sec <= criteria.max_runtime_sec
                else "runtime exceeds the configured bound"
            ),
        },
        "minimum_episode_count": {
            "passed": written >= criteria.min_episodes,
            "observed": written,
            "threshold": {"min_episodes": criteria.min_episodes},
            "reason": (
                "enough episodes were written"
                if written >= criteria.min_episodes
                else "too few episodes were written"
            ),
        },
        "minimum_success_rate": {
            "passed": success_rate is not None and success_rate >= criteria.min_success_rate,
            "observed": success_rate,
            "threshold": {"min_success_rate": criteria.min_success_rate},
            "reason": (
                "success rate meets the configured threshold"
                if success_rate is not None and success_rate >= criteria.min_success_rate
                else "success rate is missing or below threshold"
            ),
        },
        "maximum_collision_rate": {
            "passed": collision_rate is not None and collision_rate <= criteria.max_collision_rate,
            "observed": collision_rate,
            "threshold": {"max_collision_rate": criteria.max_collision_rate},
            "reason": (
                "collision rate meets the configured threshold"
                if collision_rate is not None and collision_rate <= criteria.max_collision_rate
                else "collision rate is missing or above threshold"
            ),
        },
    }
    failure_reasons = [
        f"{name}: {payload['reason']}" for name, payload in checks.items() if not payload["passed"]
    ]
    decision = "pass" if not failure_reasons else "revise"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "decision": decision,
        "failure_reasons": failure_reasons,
        "planner": {
            "algo": algo,
            "algo_config": algo_config,
            "benchmark_profile": benchmark_profile,
        },
        "reference": {
            "matrix": str(matrix),
            "schema": str(schema),
        },
        "criteria": asdict(criteria),
        "checks": checks,
        "metrics": {
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "runtime_sec": runtime_sec,
            "episodes": written,
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "episodes_jsonl": str(episodes_path),
        },
        "run_summary": {
            "total_jobs": total_jobs,
            "written": written,
            "failures": failure_count,
            "benchmark_availability": summary.get("benchmark_availability"),
        },
        "interpretation": (
            "This is an inclusion-quality review gate. Passing does not automatically change "
            "planner status or establish leaderboard rank."
        ),
    }


def run_planner_inclusion_check(  # noqa: PLR0913
    *,
    algo: str,
    matrix: Path = DEFAULT_INCLUSION_MATRIX,
    schema: Path = DEFAULT_SCHEMA_PATH,
    output_dir: Path,
    algo_config: Path | None = None,
    benchmark_profile: str = "experimental",
    base_seed: int = 0,
    repeats: int | None = 1,
    horizon: int = 250,
    dt: float = 0.1,
    workers: int = 1,
    record_forces: bool = True,
    socnav_missing_prereq_policy: str = "fail-fast",
    resume: bool = False,
    criteria: InclusionCriteria = InclusionCriteria(),
) -> dict[str, Any]:
    """Run one planner on the reference slice and write a gate report.

    Returns:
        Versioned planner inclusion report payload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_name(algo) or "planner"
    episodes_path = output_dir / f"{stem}_episodes.jsonl"
    report_path = output_dir / f"{stem}_inclusion_report.json"
    if episodes_path.exists() and not resume:
        episodes_path.unlink()

    started = time.perf_counter()
    summary = run_batch(
        scenarios_or_path=matrix,
        out_path=episodes_path,
        schema_path=schema,
        base_seed=base_seed,
        repeats_override=repeats,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        video_enabled=False,
        video_renderer="none",
        append=resume,
        fail_fast=False,
        progress_cb=None,
        algo=algo,
        algo_config_path=algo_config,
        benchmark_profile=benchmark_profile,
        socnav_missing_prereq_policy=socnav_missing_prereq_policy,
        workers=workers,
        resume=resume,
    )
    runtime_sec = time.perf_counter() - started
    records = read_jsonl(episodes_path)
    aggregates = compute_aggregates(records) if records else {}
    report = build_inclusion_report(
        algo=algo,
        algo_config=str(algo_config) if algo_config is not None else None,
        benchmark_profile=benchmark_profile,
        matrix=matrix,
        schema=schema,
        output_dir=output_dir,
        episodes_path=episodes_path,
        summary=summary,
        aggregates=aggregates,
        runtime_sec=runtime_sec,
        criteria=criteria,
    )
    report["artifacts"]["report_json"] = str(report_path)
    report_path.write_text(
        json.dumps(to_jsonable_payload(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
