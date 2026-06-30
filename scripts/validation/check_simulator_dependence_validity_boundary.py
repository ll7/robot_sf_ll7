#!/usr/bin/env python3
"""Check issue #3207 simulator-dependence validity-boundary evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.simulator_dependence_validity_boundary import (
    DECISION_SUPPORTED,
    build_simulator_dependence_decision,
    load_json_mapping,
    write_simulator_dependence_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = (
    "docs/context/evidence/issue_3207_fidelity_sensitivity_actual_slice_2026-06-23/summary.json"
)


def display_output_path(path: Path) -> str:
    """Return a readable output path for repo-local or external packet files."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=None,
        help="Promoted fidelity-sensitivity summary JSON.",
    )
    parser.add_argument(
        "--manifest-check",
        default=None,
        help="Optional fidelity_sweep_manifest_check.json input.",
    )
    parser.add_argument(
        "--expected-axis",
        action="append",
        default=[],
        help="Axis name required in rank_stability.axes. May be repeated.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path for the decision packet JSON.",
    )
    parser.add_argument(
        "--require-claim-ready",
        action="store_true",
        help="Exit non-zero when the packet is not claim-ready.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    summary_path = args.summary or DEFAULT_SUMMARY
    summary = load_json_mapping(REPO_ROOT / summary_path)
    manifest_check = (
        load_json_mapping(REPO_ROOT / args.manifest_check) if args.manifest_check else None
    )
    packet = build_simulator_dependence_decision(
        summary,
        manifest_check=manifest_check,
        expected_axes=args.expected_axis,
    )
    if args.out:
        out_path = write_simulator_dependence_decision(packet, REPO_ROOT / args.out)
        print(
            f"wrote simulator-dependence validity-boundary packet: {display_output_path(out_path)}"
        )
    else:
        print(packet["decision"])

    if args.require_claim_ready and packet["decision"] != DECISION_SUPPORTED:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
