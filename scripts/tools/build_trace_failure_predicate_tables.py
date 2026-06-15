"""Build denominator-aware trace failure predicate tables from simulation traces.

These artifacts are analysis diagnostics and must not be treated as benchmark outcomes
unless the underlying workflow is tied to a predeclared benchmark matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import robot_sf.analysis_workbench.trace_failure_predicates as tfp
from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    load_simulation_trace_export,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def build_trace_failure_predicate_tables(
    *,
    traces: list[Path],
    scenario_family: str | None = None,
    matrix: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Build aggregate predicate tables from one or more trace paths."""
    loaded_traces: list[SimulationTraceExport] = []
    failed_trace_ids: list[str] = []
    for trace_path in traces:
        try:
            loaded_traces.append(load_simulation_trace_export(trace_path))
        except (ValueError, json.JSONDecodeError):
            failed_trace_ids.append(trace_path.name)
    payload = tfp.aggregate_trace_failure_predicate_tables(
        loaded_traces,
        scenario_family=scenario_family,
        matrix=matrix,
        failed_trace_ids=failed_trace_ids,
    )
    return payload, failed_trace_ids


def write_trace_failure_predicate_tables(
    *,
    traces: list[Path],
    scenario_family: str | None = None,
    matrix: Mapping[str, Any] | None = None,
    output_json: Path,
    output_markdown: Path,
    output_denominator_health_json: Path | None = None,
) -> tuple[Path, Path] | tuple[Path, Path, Path]:
    """Load traces, build aggregate tables, and persist JSON/Markdown outputs."""
    payload, _failed_trace_ids = build_trace_failure_predicate_tables(
        traces=traces,
        scenario_family=scenario_family,
        matrix=matrix,
    )
    denominator_health_report = tfp.build_trace_predicate_denominator_health_report(payload)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_markdown.write_text(
        tfp.render_trace_failure_predicate_markdown(payload, denominator_health_report),
        encoding="utf-8",
    )
    if output_denominator_health_json is None:
        return output_json, output_markdown

    output_denominator_health_json.parent.mkdir(parents=True, exist_ok=True)
    output_denominator_health_json.write_text(
        json.dumps(denominator_health_report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_json, output_markdown, output_denominator_health_json


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for trace-failure predicate table generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        action="append",
        help="Path to one or more simulation_trace_export.v1 JSON traces.",
    )
    parser.add_argument("--scenario-family", help="Optional aggregate scenario-family label.")
    parser.add_argument(
        "--matrix",
        type=Path,
        help=(
            "Optional predeclared trace-predicate benchmark matrix YAML. Without it, outputs are "
            "diagnostic-only and claim-ineligible."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("trace_failure_predicate_tables.json"),
        help="JSON output path. Defaults to ./trace_failure_predicate_tables.json.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("trace_failure_predicate_tables.md"),
        help="Markdown output path. Defaults to ./trace_failure_predicate_tables.md.",
    )
    parser.add_argument(
        "--denominator-health-json-output",
        type=Path,
        help="Optional JSON output path for the denominator health report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    matrix = None
    if args.matrix is not None:
        try:
            matrix = tfp.load_trace_predicate_matrix(args.matrix)
            matrix["_source_path"] = str(args.matrix)
        except (OSError, ValueError) as exc:
            print(f"{exc}", file=sys.stderr)
            return 1
    try:
        output_paths = write_trace_failure_predicate_tables(
            traces=args.trace,
            scenario_family=args.scenario_family,
            matrix=matrix,
            output_json=args.json_output,
            output_markdown=args.markdown_output,
            output_denominator_health_json=args.denominator_health_json_output,
        )
    except (OSError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    json_output, markdown_output = output_paths[:2]
    print(f"wrote table JSON to {json_output}")
    print(f"wrote table markdown to {markdown_output}")
    if len(output_paths) == 3:
        print(f"wrote denominator health JSON to {output_paths[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
