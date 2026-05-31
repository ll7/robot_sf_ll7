#!/usr/bin/env python3
"""Run a local one-command Robot SF smoke benchmark demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from loguru import logger

# Keep import-time Robot SF registry logs out of the default one-command demo.
logger.remove()
logger.add(sys.stderr, level="ERROR")

from robot_sf.benchmark.aggregate import compute_aggregates, read_jsonl  # noqa: E402
from robot_sf.benchmark.runner import run_batch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "configs/scenarios/single/planner_sanity_simple.yaml"
DEFAULT_SCHEMA = ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "output/demo/smoke_benchmark"
DEFAULT_PLANNERS = ("simple_policy", "social_force")
DEFAULT_HORIZON = 300
DEFAULT_DT = 0.1
CLAIM_BOUNDARY = "Local demo output is not durable benchmark evidence unless promoted separately."


@dataclass(frozen=True)
class PlannerDemoResult:
    """Summary for one planner row in the smoke demo."""

    planner: str
    status: str
    episodes_path: Path
    record_count: int
    duration_seconds: float
    run_summary: dict[str, Any]
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable planner summary."""
        payload: dict[str, Any] = {
            "planner": self.planner,
            "status": self.status,
            "episodes_path": str(self.episodes_path),
            "record_count": self.record_count,
            "duration_seconds": round(self.duration_seconds, 3),
            "run_summary": self.run_summary,
        }
        if self.error:
            payload["error"] = self.error
        return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_demo_logging(*, verbose: bool) -> None:
    """Keep the one-command demo readable unless detailed logs are requested."""
    logger.remove()
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="ERROR")


def _metric_mean(aggregate_summary: dict[str, Any], planner: str, metric: str) -> str:
    """Return a compact aggregate mean string for a planner metric."""
    value = aggregate_summary.get(planner, {}).get(metric, {}).get("mean")
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return "n/a"


def _planner_status(
    *,
    error: str | None,
    record_count: int,
    run_summary: dict[str, Any],
) -> str:
    """Classify a planner demo row."""
    if error:
        return "failed"
    failures = run_summary.get("failures") or []
    if failures:
        return "failed"
    if record_count <= 0:
        return "failed"
    return "passed"


def _run_planner(
    *,
    planner: str,
    matrix: Path,
    output_root: Path,
    horizon: int,
    dt: float,
    workers: int,
) -> PlannerDemoResult:
    """Run one planner through the benchmark runner."""
    episodes_path = output_root / "episodes" / f"{planner}.jsonl"
    if episodes_path.exists():
        episodes_path.unlink()
    started = time.perf_counter()
    error: str | None = None
    run_summary: dict[str, Any] = {}
    try:
        run_summary = run_batch(
            matrix,
            out_path=episodes_path,
            schema_path=DEFAULT_SCHEMA,
            horizon=horizon,
            dt=dt,
            record_forces=False,
            algo=planner,
            workers=workers,
            resume=False,
            benchmark_profile="baseline-safe",
        )
    except Exception as exc:  # pragma: no cover - exercised through tests with monkeypatch
        error = str(exc)
    duration_seconds = time.perf_counter() - started
    records = read_jsonl(episodes_path, strict=False) if episodes_path.exists() else []
    status = _planner_status(
        error=error,
        record_count=len(records),
        run_summary=run_summary,
    )
    return PlannerDemoResult(
        planner=planner,
        status=status,
        episodes_path=episodes_path,
        record_count=len(records),
        duration_seconds=duration_seconds,
        run_summary=run_summary,
        error=error,
    )


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    """Write a concise Markdown report for humans."""
    rows = [
        "| Planner | Status | Records | Success mean | Collisions mean | Notes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    aggregate_summary = summary.get("aggregate_summary", {})
    for planner in summary["planners"]:
        note = planner.get("error") or ""
        rows.append(
            "| {planner} | {status} | {records} | {success} | {collisions} | {note} |".format(
                planner=planner["planner"],
                status=planner["status"],
                records=planner["record_count"],
                success=_metric_mean(aggregate_summary, planner["planner"], "success"),
                collisions=_metric_mean(aggregate_summary, planner["planner"], "collisions"),
                note=note.replace("|", "\\|"),
            )
        )

    body = "\n".join(
        [
            "# Robot SF Smoke Benchmark Demo",
            "",
            f"- Status: {'passed' if summary['passed'] else 'failed'}",
            f"- Scenario matrix: `{summary['matrix']}`",
            f"- Output root: `{summary['output_root']}`",
            f"- Claim boundary: {summary['claim_boundary']}",
            "",
            "## Planner Results",
            "",
            *rows,
            "",
            "## Artifacts",
            "",
            f"- Summary JSON: `{summary['artifacts']['summary_json']}`",
            f"- Report Markdown: `{summary['artifacts']['report_md']}`",
            f"- Episode JSONL directory: `{summary['artifacts']['episodes_dir']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def run_demo(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    matrix: Path = DEFAULT_MATRIX,
    planners: tuple[str, ...] = DEFAULT_PLANNERS,
    horizon: int = DEFAULT_HORIZON,
    dt: float = DEFAULT_DT,
    workers: int = 1,
) -> dict[str, Any]:
    """Run the local smoke benchmark demo and write summary artifacts."""
    output_root = Path(output_root)
    matrix = Path(matrix)
    output_root.mkdir(parents=True, exist_ok=True)

    planner_results = [
        _run_planner(
            planner=planner,
            matrix=matrix,
            output_root=output_root,
            horizon=horizon,
            dt=dt,
            workers=workers,
        )
        for planner in planners
    ]
    episode_paths = [
        result.episodes_path for result in planner_results if result.episodes_path.exists()
    ]
    records = read_jsonl(episode_paths, strict=False) if episode_paths else []
    aggregate_summary = compute_aggregates(records) if records else {}

    summary_path = output_root / "summary.json"
    report_path = output_root / "report.md"
    payload: dict[str, Any] = {
        "schema_version": "robot_sf_smoke_demo.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "matrix": str(matrix),
        "output_root": str(output_root),
        "planners": [result.to_json_dict() for result in planner_results],
        "passed": all(result.status == "passed" for result in planner_results),
        "claim_boundary": CLAIM_BOUNDARY,
        "aggregate_summary": aggregate_summary,
        "parameters": {
            "horizon": int(horizon),
            "dt": float(dt),
            "workers": int(workers),
        },
        "artifacts": {
            "summary_json": str(summary_path),
            "report_md": str(report_path),
            "episodes_dir": str(output_root / "episodes"),
        },
    }
    _write_json(summary_path, payload)
    _write_report(report_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the smoke demo command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--planners", nargs="+", default=list(DEFAULT_PLANNERS))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed simulator and map-runner logs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the smoke demo CLI."""
    args = build_parser().parse_args(argv)
    _configure_demo_logging(verbose=args.verbose)
    summary = run_demo(
        output_root=args.output_root,
        matrix=args.matrix,
        planners=tuple(args.planners),
        horizon=args.horizon,
        dt=args.dt,
        workers=args.workers,
    )
    status = "passed" if summary["passed"] else "failed"
    print(f"Robot SF smoke benchmark demo {status}.")
    print(f"Output: {summary['output_root']}")
    print(
        "Planners: "
        + ", ".join(
            f"{planner['planner']}={planner['status']} ({planner['record_count']} records)"
            for planner in summary["planners"]
        )
    )
    print(CLAIM_BOUNDARY)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
