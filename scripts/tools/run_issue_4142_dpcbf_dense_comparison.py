#!/usr/bin/env python3
"""Consume the issue #4142 dense DPCBF comparison packet into a run plan (dry-run).

Resolves the predeclared comparison packet
``configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml`` into a concrete, ordered
three-arm run plan (``cbf_off``, ``cbf_collision_cone_on``, ``cbf_dynamic_parabolic_v1_on``)
using the canonical readiness validator as the single source of truth. It runs no episodes
and authorizes no campaign.

The default (and only supported) mode is a dry-run: the plan is printed, nothing is written
to disk, and execution stays authorization-gated. Passing ``--execute`` fails closed --
running the dense comparison requires explicit human/Slurm authorization that this slice
does not grant.

Examples:
    # Human-readable Markdown run plan against the current checkout.
    uv run python scripts/tools/run_issue_4142_dpcbf_dense_comparison.py

    # JSON plan, non-zero exit unless the plan is fully resolved (CI/preflight gate).
    uv run python scripts/tools/run_issue_4142_dpcbf_dense_comparison.py \
        --format json --fail-on-blocked

    # Attempting execution fails closed (execution is authorization-gated).
    uv run python scripts/tools/run_issue_4142_dpcbf_dense_comparison.py --execute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    PACKET_PATH,
    DpcbfDenseReadinessError,
)
from robot_sf.benchmark.issue_4142_dpcbf_dense_runner import (
    DEFAULT_OUTPUT_DIR,
    DenseComparisonExecutionGatedError,
    DenseComparisonRunnerError,
    build_run_plan,
    execute_run_plan,
    render_markdown,
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
        help="Directory that repo-relative packet/config paths resolve against.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help="Directory the per-arm output JSONL paths are planned under (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: %(default)s).",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero unless the plan reaches 'plan_ready_campaign_gated'.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Attempt to execute the comparison. Fails closed: execution is "
            "authorization-gated and out of scope for this runner slice."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the run plan and emit a report (or fail closed on --execute).

    Returns:
        Process exit code (0 on success; 1 when blocked and ``--fail-on-blocked``;
        2 on packet load error; 3 when ``--execute`` hits the authorization gate).
    """
    args = _parse_args(argv)
    try:
        plan = build_run_plan(
            repo_root=args.repo_root,
            packet_path=args.packet,
            output_dir=args.output_dir,
        )
    except (DpcbfDenseReadinessError, DenseComparisonRunnerError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(to_dict(plan), indent=2, sort_keys=True))
    else:
        print(render_markdown(plan))

    if args.execute:
        try:
            execute_run_plan(plan)
        except DenseComparisonExecutionGatedError as exc:
            print(f"execution gated: {exc}", file=sys.stderr)
            return 3

    if args.fail_on_blocked and not plan.is_executable_in_principle:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
