#!/usr/bin/env python3
"""Run the pull-request promoted-planner benchmark smoke."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.aggregate import compute_aggregates, read_jsonl
from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "configs/scenarios/single/pr_promoted_planner_smoke.yaml"
DEFAULT_BASELINE = ROOT / "configs/benchmarks/pr_promoted_planner_smoke_baseline.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output/benchmarks/pr_promoted_planner_smoke"
DEFAULT_ALGORITHMS = ("goal", "social_force", "orca")
ALLOWED_READINESS = {"native", "adapter"}
REPORT_METRICS = (
    "success.mean",
    "collisions.mean",
    "near_misses.mean",
    "time_to_goal_norm.mean",
    "path_efficiency.mean",
)


@dataclass(frozen=True)
class PlannerResult:
    """Result for one planner smoke run."""

    algorithm: str
    command: list[str]
    returncode: int
    duration_seconds: float
    episodes_path: Path
    log_path: Path
    availability: dict[str, Any]
    record_count: int
    metrics: dict[str, float]
    deltas: dict[str, float | None]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the planner met all smoke gates."""
        return not self.failures


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _metric_value(summary: dict[str, Any], algorithm: str, metric: str) -> float | None:
    """Read a flattened aggregate metric such as ``success.mean``."""
    metric_name, stat_name = metric.split(".", maxsplit=1)
    value = summary.get(algorithm, {}).get(metric_name, {}).get(stat_name)
    if isinstance(value, int | float):
        return float(value)
    return None


def _parse_summary_event(stdout: str) -> dict[str, Any]:
    """Parse the final structured benchmark summary from command stdout."""
    for raw_line in reversed(stdout.splitlines()):
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "benchmark.run.summary":
            return payload
    return {}


def _runner_command(
    *,
    algorithm: str,
    matrix: Path,
    episodes_path: Path,
    horizon: int,
    dt: float,
) -> list[str]:
    """Build the benchmark runner command for one planner."""
    runner = shutil.which("robot_sf_bench")
    if runner:
        base = [runner]
    else:
        base = [sys.executable, "-c", "from robot_sf.benchmark.cli import main; main()"]
    return [
        *base,
        "run",
        "--matrix",
        str(matrix),
        "--out",
        str(episodes_path),
        "--algo",
        algorithm,
        "--repeats",
        "1",
        "--horizon",
        str(horizon),
        "--dt",
        str(dt),
        "--workers",
        "1",
        "--no-video",
        "--benchmark-profile",
        "baseline-safe",
        "--no-resume",
        "--fail-fast",
        "--structured-output",
        "json",
        "--external-log-noise",
        "suppress",
    ]


def _evaluate_failures(
    *,
    algorithm: str,
    returncode: int,
    duration_seconds: float,
    runtime_budget_seconds: float,
    availability: dict[str, Any],
    record_count: int,
    metrics: dict[str, float],
    baseline: dict[str, Any],
) -> tuple[str, ...]:
    """Return fail-closed smoke gate violations."""
    failures: list[str] = []
    failures.extend(
        _basic_gate_failures(
            returncode=returncode,
            duration_seconds=duration_seconds,
            runtime_budget_seconds=runtime_budget_seconds,
            record_count=record_count,
        )
    )
    failures.extend(_availability_failures(availability))
    failures.extend(
        _baseline_metric_failures(algorithm=algorithm, metrics=metrics, baseline=baseline)
    )
    return tuple(failures)


def _basic_gate_failures(
    *,
    returncode: int,
    duration_seconds: float,
    runtime_budget_seconds: float,
    record_count: int,
) -> list[str]:
    """Return command/runtime/record-count gate failures."""
    failures = []
    if returncode != 0:
        failures.append(f"runner exited with {returncode}")
    if duration_seconds > runtime_budget_seconds:
        failures.append(
            f"runtime {duration_seconds:.1f}s exceeded budget {runtime_budget_seconds:.1f}s"
        )
    if record_count != 1:
        failures.append(f"expected exactly 1 episode record, found {record_count}")
    return failures


def _availability_failures(availability: dict[str, Any]) -> list[str]:
    """Return benchmark availability gate failures."""
    failures = []
    availability_status = availability.get("availability_status")
    readiness_status = availability.get("readiness_status")
    benchmark_success = availability.get("benchmark_success")
    if availability_status != "available":
        failures.append(f"availability_status={availability_status!r}")
    if readiness_status not in ALLOWED_READINESS:
        failures.append(f"readiness_status={readiness_status!r}")
    if benchmark_success is not True:
        failures.append(f"benchmark_success={benchmark_success!r}")
    return failures


def _baseline_metric_failures(
    *,
    algorithm: str,
    metrics: dict[str, float],
    baseline: dict[str, Any],
) -> list[str]:
    """Return baseline threshold failures for one planner."""
    failures = []
    planner_baseline = baseline.get("planners", {}).get(algorithm, {})
    min_success = planner_baseline.get("minimum_success_mean")
    if isinstance(min_success, int | float):
        observed = metrics.get("success.mean")
        if observed is None or observed < float(min_success):
            failures.append(f"success.mean={observed!r} below minimum {float(min_success):.3f}")
    max_collisions = planner_baseline.get("maximum_collisions_mean")
    if isinstance(max_collisions, int | float):
        observed = metrics.get("collisions.mean")
        if observed is None or observed > float(max_collisions):
            failures.append(
                f"collisions.mean={observed!r} above maximum {float(max_collisions):.3f}"
            )
    max_near_misses = planner_baseline.get("maximum_near_misses_mean")
    if isinstance(max_near_misses, int | float):
        observed = metrics.get("near_misses.mean")
        if observed is None or observed > float(max_near_misses):
            failures.append(
                f"near_misses.mean={observed!r} above maximum {float(max_near_misses):.3f}"
            )
    return failures


def _run_planner(
    *,
    algorithm: str,
    matrix: Path,
    output_root: Path,
    horizon: int,
    dt: float,
    runtime_budget_seconds: float,
    baseline: dict[str, Any],
) -> PlannerResult:
    """Run and summarize one planner smoke benchmark."""
    episodes_path = output_root / "episodes" / f"{algorithm}.jsonl"
    log_path = output_root / "logs" / f"{algorithm}.log"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if episodes_path.is_file():
        episodes_path.unlink()

    command = _runner_command(
        algorithm=algorithm,
        matrix=matrix,
        episodes_path=episodes_path,
        horizon=horizon,
        dt=dt,
    )
    env = dict(os.environ)
    env.setdefault("LOGURU_LEVEL", "WARNING")
    env.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("MPLBACKEND", "Agg")

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=runtime_budget_seconds + 30.0,
    )
    duration_seconds = time.perf_counter() - start
    log_path.write_text(proc.stdout, encoding="utf-8")

    event = _parse_summary_event(proc.stdout)
    availability = event.get("benchmark_availability") if isinstance(event, dict) else {}
    if not isinstance(availability, dict):
        availability = {}

    records = read_jsonl(episodes_path)
    summary = compute_aggregates(records, expected_algorithms={algorithm}) if records else {}
    metrics = {
        metric: value
        for metric in REPORT_METRICS
        if (value := _metric_value(summary, algorithm, metric)) is not None
    }
    reference = baseline.get("planners", {}).get(algorithm, {}).get("reference_metrics", {})
    deltas = {
        metric: (
            metrics[metric] - float(reference[metric])
            if metric in metrics and isinstance(reference.get(metric), int | float)
            else None
        )
        for metric in REPORT_METRICS
    }
    failures = _evaluate_failures(
        algorithm=algorithm,
        returncode=proc.returncode,
        duration_seconds=duration_seconds,
        runtime_budget_seconds=runtime_budget_seconds,
        availability=availability,
        record_count=len(records),
        metrics=metrics,
        baseline=baseline,
    )
    return PlannerResult(
        algorithm=algorithm,
        command=command,
        returncode=proc.returncode,
        duration_seconds=duration_seconds,
        episodes_path=episodes_path,
        log_path=log_path,
        availability=availability,
        record_count=len(records),
        metrics=metrics,
        deltas=deltas,
        failures=failures,
    )


def _write_combined_episodes(results: list[PlannerResult], output_root: Path) -> Path:
    """Write a combined JSONL file from per-planner episodes."""
    combined_path = output_root / "episodes.jsonl"
    with combined_path.open("w", encoding="utf-8") as out_handle:
        for result in results:
            if not result.episodes_path.is_file():
                continue
            for line in result.episodes_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out_handle.write(line)
                    out_handle.write("\n")
    return combined_path


def _planner_payload(result: PlannerResult) -> dict[str, Any]:
    """Convert one planner result into a JSON payload."""
    return {
        "algorithm": result.algorithm,
        "passed": result.passed,
        "returncode": result.returncode,
        "duration_seconds": result.duration_seconds,
        "record_count": result.record_count,
        "availability": result.availability,
        "metrics": result.metrics,
        "deltas": result.deltas,
        "failures": list(result.failures),
        "episodes_path": _repo_relative(result.episodes_path),
        "log_path": _repo_relative(result.log_path),
        "command": result.command,
    }


def build_report_payload(
    *,
    results: list[PlannerResult],
    matrix: Path,
    baseline_path: Path,
    output_root: Path,
    combined_episodes_path: Path,
    runtime_budget_seconds: float,
    total_duration_seconds: float,
) -> dict[str, Any]:
    """Build the machine-readable smoke report payload."""
    passed = all(result.passed for result in results)
    return {
        "schema_version": "pr-promoted-planner-smoke-result.v1",
        "passed": passed,
        "algorithms": [result.algorithm for result in results],
        "scenario_matrix": _repo_relative(matrix),
        "baseline_path": _repo_relative(baseline_path),
        "output_root": _repo_relative(output_root),
        "episodes_path": _repo_relative(combined_episodes_path),
        "runtime_budget_seconds_per_planner": runtime_budget_seconds,
        "total_duration_seconds": total_duration_seconds,
        "planners": [_planner_payload(result) for result in results],
    }


def render_markdown_report(payload: dict[str, Any]) -> str:
    """Render a concise workflow summary table."""
    lines = [
        "# PR promoted planner smoke",
        "",
        f"- Result: {'PASS' if payload['passed'] else 'FAIL'}",
        f"- Scenario matrix: `{payload['scenario_matrix']}`",
        f"- Baseline: `{payload['baseline_path']}`",
        f"- Episodes: `{payload['episodes_path']}`",
        f"- Total runtime: {payload['total_duration_seconds']:.1f}s",
        "",
        "| Planner | Gate | Readiness | Success | Collisions | Near misses | Time norm delta | Failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["planners"]:
        metrics = row["metrics"]
        deltas = row["deltas"]
        availability = row["availability"]
        failures = "; ".join(row["failures"]) if row["failures"] else ""
        lines.append(
            "| {algorithm} | {gate} | {readiness} | {success:.3f} | {collisions:.3f} | "
            "{near_misses:.3f} | {time_delta} | {failures} |".format(
                algorithm=row["algorithm"],
                gate="PASS" if row["passed"] else "FAIL",
                readiness=availability.get("readiness_status", "unknown"),
                success=metrics.get("success.mean", float("nan")),
                collisions=metrics.get("collisions.mean", float("nan")),
                near_misses=metrics.get("near_misses.mean", float("nan")),
                time_delta=(
                    f"{deltas['time_to_goal_norm.mean']:+.3f}"
                    if deltas.get("time_to_goal_norm.mean") is not None
                    else "n/a"
                ),
                failures=failures.replace("|", "\\|"),
            )
        )
    lines.append("")
    lines.append(
        "Failure conditions: runner exit, missing episode, unavailable planner, fallback/degraded "
        "readiness, success below the tracked minimum, collisions or near misses above the tracked "
        "maximum, or per-planner runtime over budget."
    )
    lines.append("")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHMS),
        help="Planner algorithms to smoke-test.",
    )
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--runtime-budget-seconds", type=float, default=90.0)
    parser.add_argument(
        "--github-step-summary",
        type=Path,
        default=Path(os.environ["GITHUB_STEP_SUMMARY"])
        if os.environ.get("GITHUB_STEP_SUMMARY")
        else None,
        help="Optional GitHub Actions step summary file to append Markdown to.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PR promoted planner smoke."""
    args = _build_parser().parse_args(argv)
    matrix = args.matrix.resolve()
    baseline_path = args.baseline.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    baseline = _load_json(baseline_path)
    start = time.perf_counter()
    results = [
        _run_planner(
            algorithm=algorithm,
            matrix=matrix,
            output_root=output_root,
            horizon=args.horizon,
            dt=args.dt,
            runtime_budget_seconds=args.runtime_budget_seconds,
            baseline=baseline,
        )
        for algorithm in args.algorithms
    ]
    total_duration_seconds = time.perf_counter() - start
    combined_episodes_path = _write_combined_episodes(results, output_root)
    payload = build_report_payload(
        results=results,
        matrix=matrix,
        baseline_path=baseline_path,
        output_root=output_root,
        combined_episodes_path=combined_episodes_path,
        runtime_budget_seconds=args.runtime_budget_seconds,
        total_duration_seconds=total_duration_seconds,
    )
    summary_json = output_root / "summary.json"
    summary_md = output_root / "summary.md"
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown = render_markdown_report(payload)
    summary_md.write_text(markdown, encoding="utf-8")
    if args.github_step_summary is not None:
        args.github_step_summary.parent.mkdir(parents=True, exist_ok=True)
        with args.github_step_summary.open("a", encoding="utf-8") as handle:
            handle.write(markdown)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
