#!/usr/bin/env python3
"""Validate predictive same-seed comparison outcome rows and emit a JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.predictive_same_seed_row_summary import (
    PredictiveSameSeedRowSummaryValidationError,
    validate_predictive_same_seed_row_summary_file,
)
from robot_sf.common.artifact_paths import get_repository_root


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Validate predictive same-seed row summaries against the #1550 contract."
    )
    parser.add_argument("--input", required=True, type=Path, help="YAML or JSON row-summary file.")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate a row-summary file and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        report = validate_predictive_same_seed_row_summary_file(
            args.input,
            repo_root=args.repo_root,
            check_git_history=args.check_git_history,
        )
    except PredictiveSameSeedRowSummaryValidationError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "valid" else 2


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
