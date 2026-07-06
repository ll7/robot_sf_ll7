#!/usr/bin/env python3
"""Validate hybrid-learning evidence matrix rows and emit a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.hybrid_evidence_matrix import (
    DEFAULT_SYNTHESIS_PREREQUISITE_COUNT,
    HybridEvidenceMatrixValidationError,
    build_hybrid_prerequisite_matrix_file,
    build_hybrid_synthesis_report_file,
    validate_hybrid_evidence_file,
)
from robot_sf.common.artifact_paths import get_repository_root


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate hybrid-learning evidence matrix rows against the #1499 contract."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="YAML or JSON evidence-matrix file."
    )
    parser.add_argument(
        "--repo-root",
        default=get_repository_root(),
        type=Path,
        help="Repository root used to resolve repository-relative provenance paths.",
    )
    parser.add_argument(
        "--check-git-history",
        action="store_true",
        help=(
            "Also verify that each git SHA token in commit_artifact resolves to a commit in the "
            "local repository history. Default validation remains format-only."
        ),
    )
    parser.add_argument(
        "--prerequisite-matrix",
        action="store_true",
        help=(
            "Emit the #1489 prerequisite/status matrix instead of the row-validation report: "
            "classify each component lane as missing, blocked, ready, or complete and decide "
            "whether the synthesis gate is open."
        ),
    )
    parser.add_argument(
        "--synthesis-report",
        action="store_true",
        help=(
            "Emit the #1489 per-mechanism synthesis recommendation report "
            "(continue/revise/stop/gather_more_evidence) built from the prerequisite matrix. "
            "Stays fail-closed: a verdict is authoritative only when synthesis_verdict_promoted "
            "is true (gate open). Takes precedence over --prerequisite-matrix."
        ),
    )
    parser.add_argument(
        "--expected-component",
        action="append",
        default=None,
        dest="expected_components",
        metavar="NAME",
        help=(
            "Component name expected to have a row (repeatable). Expected components with no "
            "matching row are reported as 'missing'. Only used with --prerequisite-matrix."
        ),
    )
    parser.add_argument(
        "--prerequisite-count",
        type=int,
        default=DEFAULT_SYNTHESIS_PREREQUISITE_COUNT,
        help=(
            "Minimum number of 'complete' (synthesis-eligible) lanes that open the synthesis "
            "gate. Only used with --prerequisite-matrix."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a matrix file and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        if args.synthesis_report or args.prerequisite_matrix:
            builder = (
                build_hybrid_synthesis_report_file
                if args.synthesis_report
                else build_hybrid_prerequisite_matrix_file
            )
            report = builder(
                args.input,
                expected_components=args.expected_components,
                repo_root=args.repo_root,
                check_git_history=args.check_git_history,
                prerequisite_count=args.prerequisite_count,
            )
            print(json.dumps(report, indent=2, sort_keys=True))
            # Fail-closed exit code: a blocked-but-valid gate is a successful
            # emission (exit 0); only invalid rows are a hard error (exit 2).
            # Both report shapes echo ``rows_valid`` (the synthesis report
            # forwards the matrix signal), so invalid rows still exit 2.
            return 0 if report["rows_valid"] else 2
        report = validate_hybrid_evidence_file(
            args.input,
            repo_root=args.repo_root,
            check_git_history=args.check_git_history,
        )
    except HybridEvidenceMatrixValidationError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
