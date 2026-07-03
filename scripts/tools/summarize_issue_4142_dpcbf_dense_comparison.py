#!/usr/bin/env python3
"""Summarize the issue #4142 dense DPCBF comparison from per-arm artifacts (read-only).

Consumes the resolved three-arm run plan
(:mod:`robot_sf.benchmark.issue_4142_dpcbf_dense_runner`) and reads each arm's per-episode
JSONL output (``cbf_off``, ``cbf_collision_cone_on``, ``cbf_dynamic_parabolic_v1_on``) into
a fail-closed comparison summary (schema
``robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1``). It runs no episodes and
authorizes no campaign.

Because execution stays authorization-gated, no arm output exists in a fresh checkout, so
the summary is expected to report ``results_incomplete`` with a per-arm artifact manifest
that records each missing output path. Fallback/degraded/failed/ineligible rows are counted
as caveats, never success evidence.

Examples:
    # Human-readable Markdown summary against the current checkout.
    uv run python scripts/tools/summarize_issue_4142_dpcbf_dense_comparison.py

    # JSON summary; non-zero exit unless every required arm artifact is present (CI gate).
    uv run python scripts/tools/summarize_issue_4142_dpcbf_dense_comparison.py \
        --format json --fail-on-incomplete
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.issue_4142_dpcbf_dense_runner import DEFAULT_OUTPUT_DIR, PACKET_PATH
from robot_sf.benchmark.issue_4142_dpcbf_dense_summary import (
    DenseComparisonSummaryError,
    render_markdown,
    summarize_dense_comparison,
    to_dict,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path(PACKET_PATH),
        help="Path to the comparison packet YAML (default: %(default)s).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Directory that repo-relative packet/config/artifact paths resolve against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory the per-arm output JSONL files are read from (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: %(default)s).",
    )
    parser.add_argument(
        "--fail-on-incomplete",
        action="store_true",
        help="Exit non-zero unless the summary reaches 'complete' (all arm artifacts present).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the comparison summary and emit a report.

    Returns:
        Process exit code (0 on success; 1 when incomplete and ``--fail-on-incomplete``;
        2 on run-plan build error).
    """
    args = _parse_args(argv)
    try:
        summary = summarize_dense_comparison(
            repo_root=args.repo_root,
            packet_path=args.packet,
            output_dir=args.output_dir,
        )
    except DenseComparisonSummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(to_dict(summary), indent=2, sort_keys=True))
    else:
        print(render_markdown(summary))

    if args.fail_on_incomplete and summary.status != "complete":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
