#!/usr/bin/env python3
"""Resolve the #4142 DPCBF packet or run its bounded authorization-gated local executor.

Dry-run is the default. ``--execute`` requires the exact public authorization ID and never
submits Slurm/GPU work; failed or degraded arms remain caveats.
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
    DenseComparisonProvenanceMismatchError,
    DenseComparisonRunnerError,
    build_run_plan,
    execute_run_plan,
    manifest_to_dict,
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
            "Run bounded local episodes for all three arms. Requires the exact public "
            "authorization ID via --authorization; otherwise fails closed and writes nothing."
        ),
    )
    parser.add_argument(
        "--authorization",
        type=str,
        default=None,
        help=(
            "Exact public authorization ID required to actually run episodes. A boolean, "
            "environment variable, implicit TTY, or bare --execute flag is insufficient."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the run plan and emit a report (or fail closed on --execute).

    Returns:
        Process exit code (0 on success; 1 when blocked and ``--fail-on-blocked``;
        2 on packet load error; 3 when ``--execute`` hits the authorization gate;
        4 on a provenance mismatch; 5 when an authorized run did not complete all arms).
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
            manifest = execute_run_plan(
                plan,
                authorization=args.authorization,
                repo_root=args.repo_root,
            )
        except DenseComparisonExecutionGatedError as exc:
            print(f"execution gated: {exc}", file=sys.stderr)
            return 3
        except DenseComparisonProvenanceMismatchError as exc:
            print(f"provenance mismatch: {exc}", file=sys.stderr)
            return 4
        print(json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True))
        if manifest.status != "complete":
            print(
                f"execution incomplete: manifest status {manifest.status!r}; "
                "failed/degraded arms are caveats, never success evidence.",
                file=sys.stderr,
            )
            return 5

    if args.fail_on_blocked and not plan.is_executable_in_principle:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
