"""Build denominator-aware trace failure predicate tables from simulation traces.

These artifacts are analysis diagnostics and must not be treated as benchmark outcomes
unless the underlying workflow is tied to a predeclared benchmark matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.simulation_trace_export import load_simulation_trace_export
from robot_sf.analysis_workbench.trace_failure_predicates import (
    aggregate_trace_failure_predicate_tables,
    render_trace_failure_predicate_markdown,
)


def build_trace_failure_predicate_tables(
    *,
    traces: list[Path],
    scenario_family: str | None = None,
) -> dict[str, Any]:
    """Build aggregate predicate tables from one or more trace paths."""
    loaded = [load_simulation_trace_export(trace_path) for trace_path in traces]
    return aggregate_trace_failure_predicate_tables(
        loaded,
        scenario_family=scenario_family,
    )


def write_trace_failure_predicate_tables(
    *,
    traces: list[Path],
    scenario_family: str | None = None,
    output_json: Path,
    output_markdown: Path,
) -> tuple[Path, Path]:
    """Load traces, build aggregate tables, and persist JSON/Markdown outputs."""
    payload = build_trace_failure_predicate_tables(
        traces=traces,
        scenario_family=scenario_family,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    output_markdown.write_text(
        render_trace_failure_predicate_markdown(payload),
        encoding="utf-8",
    )
    return output_json, output_markdown


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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        json_output, markdown_output = write_trace_failure_predicate_tables(
            traces=args.trace,
            scenario_family=args.scenario_family,
            output_json=args.json_output,
            output_markdown=args.markdown_output,
        )
    except (OSError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    print(f"wrote table JSON to {json_output}")
    print(f"wrote table markdown to {markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
