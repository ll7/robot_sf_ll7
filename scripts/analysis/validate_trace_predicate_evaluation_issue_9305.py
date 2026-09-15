"""Validate and report the issue #9305 trace-predicate evaluation contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.analysis_workbench.trace_predicate_validation import (
    TracePredicateValidationError,
    build_trace_predicate_validation_report,
    load_trace_predicate_validation_set,
    write_trace_predicate_validation_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate a versioned trace-predicate evaluation set and write its "
            "diagnostic-only report. No simulator or campaign is run."
        )
    )
    parser.add_argument("--evaluation-set", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve repository-relative trace URIs.",
    )
    parser.add_argument(
        "--expected-source-commit",
        help="Optional exact lowercase commit SHA required by the evaluation set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate one evaluation set and emit a report or an explicit error."""
    args = build_parser().parse_args(argv)
    try:
        evaluation_set = load_trace_predicate_validation_set(
            args.evaluation_set,
            repo_root=args.repo_root,
            expected_source_commit=args.expected_source_commit,
        )
        report = build_trace_predicate_validation_report(
            evaluation_set,
            repo_root=args.repo_root,
            expected_source_commit=args.expected_source_commit,
        )
        write_trace_predicate_validation_report(
            report,
            args.output_json,
            out_markdown=args.output_markdown,
        )
    except TracePredicateValidationError as exc:
        print(f"trace predicate validation unavailable: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
