#!/usr/bin/env python3
"""Preflight the issue #4142 dense Dynamic Parabolic CBF comparison packet (read-only).

Validates that the predeclared comparison packet
``configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml`` keeps its three CBF arms
(``cbf_off``, ``cbf_collision_cone_on``, ``cbf_dynamic_parabolic_v1_on``) predeclared and
fail-closed, that each arm's adapter config exists and matches its runtime variant, and
that fallback/degraded rows stay excluded from success evidence. It reuses the canonical
CBF runtime validator and makes no benchmark, safety-performance, or collision-reduction
claim; it runs no episodes and authorizes no campaign.

Examples:
    # Human-readable Markdown report against the current checkout.
    uv run python scripts/tools/check_issue_4142_dpcbf_dense_readiness.py

    # JSON output, non-zero exit when the packet is not campaign-ready (CI/preflight gate).
    uv run python scripts/tools/check_issue_4142_dpcbf_dense_readiness.py \
        --format json --fail-on-blocked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.issue_4142_dpcbf_dense_readiness import (
    PACKET_PATH,
    DpcbfDenseReadinessError,
    evaluate_readiness,
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
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: %(default)s).",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero unless the packet reaches 'inputs_ready_campaign_gated'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate readiness and emit a report.

    Returns:
        Process exit code (0 on success; 1 when blocked and ``--fail-on-blocked``).
    """
    args = _parse_args(argv)
    try:
        readiness = evaluate_readiness(repo_root=args.repo_root, packet_path=args.packet)
    except DpcbfDenseReadinessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(to_dict(readiness), indent=2, sort_keys=True))
    else:
        print(render_markdown(readiness))

    if args.fail_on_blocked and not readiness.inputs_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
