#!/usr/bin/env python3
"""Emit ``mechanism_trace.v1`` rows from ORCA residual planner-decision traces."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from robot_sf.benchmark.mechanism_trace import (
    SCHEMA_VERSION,
    emit_orca_residual_rows,
    validate_mechanism_trace_payload,
)

OutputFormat = Literal["json", "jsonl"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE = REPO_ROOT / "tests/benchmark/fixtures/orca_residuals_planner_decision_trace.v1.json"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner-decision-trace",
        type=Path,
        default=DEFAULT_TRACE,
        help=(
            "Path to a planner-decision trace JSON list (defaults to "
            "tests/benchmark/fixtures/orca_residuals_planner_decision_trace.v1.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for mechanism-trace JSON or JSONL rows.",
    )
    parser.add_argument(
        "--trace-uri",
        default=None,
        help="URI to embed in each emitted row's trace_uri field.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Write mechanism trace payload as JSON ('mechanism_trace.v1') or JSONL rows.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path with diagnostic-only provenance and row counts.",
    )
    return parser


def load_planner_decision_trace(path: Path) -> list[dict[str, Any]]:
    """Load planner-decision trace entries from a fixture file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected planner-decision trace list in {path}, got {type(payload).__name__}"
        )
    return [entry for entry in payload if isinstance(entry, Mapping)]


def build_orca_residual_mechanism_trace_payload(
    planner_decision_trace: list[dict[str, Any]],
    *,
    trace_uri: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a validated mechanism_trace.v1 payload from ORCA-residual steps."""
    rows = emit_orca_residual_rows(
        planner_decision_trace,
        trace_uri=trace_uri,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at or datetime.now(UTC).isoformat(),
        "rows": rows,
    }
    validate_mechanism_trace_payload(payload)
    return payload


def write_mechanism_trace_payload(
    payload: dict[str, Any],
    output_path: Path,
    output_format: OutputFormat,
) -> None:
    """Write a mechanism trace payload as schema JSON or row JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("rows", [])
    if output_format == "json":
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_generation_command(args: argparse.Namespace) -> str:
    """Return a shell-safe command string for the requested emission."""

    command = [
        "uv",
        "run",
        "python",
        "scripts/tools/emit_orca_residual_mechanism_trace.py",
        "--planner-decision-trace",
        str(args.planner_decision_trace),
        "--output",
        str(args.output),
        "--format",
        str(args.format),
        "--trace-uri",
        str(args.trace_uri or args.planner_decision_trace),
    ]
    if args.report is not None:
        command.extend(["--report", str(args.report)])
    return shlex.join(command)


def git_commit(repo_root: Path = REPO_ROOT) -> str:
    """Return the current Git commit, or ``unknown`` outside Git."""

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_emission_report(
    payload: dict[str, Any],
    *,
    command: str,
    planner_decision_trace: Path,
    output: Path,
) -> dict[str, Any]:
    """Build a compact diagnostic-only emission report."""

    counts: dict[str, int] = {}
    for row in payload.get("rows", []):
        if isinstance(row, Mapping):
            classification = str(row.get("classification", "unknown"))
            counts[classification] = counts.get(classification, 0) + 1

    return {
        "schema_version": "orca_residual_mechanism_trace_emission.v1",
        "issue": 2981,
        "claim_boundary": "diagnostic_only",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": git_commit(),
        "command": command,
        "planner_decision_trace": planner_decision_trace.as_posix(),
        "rows_path": output.as_posix(),
        "rows_count": len(payload.get("rows", [])),
        "classification_counts": dict(sorted(counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    """Run emission from CLI."""
    args = build_parser().parse_args(argv)
    planner_trace = load_planner_decision_trace(args.planner_decision_trace)
    trace_uri = args.trace_uri or str(args.planner_decision_trace)
    payload = build_orca_residual_mechanism_trace_payload(
        planner_trace,
        trace_uri=trace_uri,
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_mechanism_trace_payload(
        payload,
        output_path=args.output,
        output_format=args.format,
    )
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        report = build_emission_report(
            payload,
            command=build_generation_command(args),
            planner_decision_trace=args.planner_decision_trace,
            output=args.output,
        )
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
