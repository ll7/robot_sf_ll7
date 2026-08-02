"""Build / validate the Gate 0 post-hoc feasibility-audit decision JSON (issue #6640).

Emits the machine-readable decision that classifies each collision-envelope radius-sensitivity
outcome as ``re-derivable`` or ``replay-required`` for the frozen ``0.0.3.post1`` release rows.
The decision is deterministic and reproducible from the metric contract; running this script
always writes the same bytes for the same contract.

Usage:

    uv run python scripts/benchmark/build_radius_sensitivity_gate0_decision.py \\
        --output docs/context/radius_sensitivity_gate0_audit_issue_6640.json

    # Re-load and re-validate an existing decision file:
    uv run python scripts/benchmark/build_radius_sensitivity_gate0_decision.py \\
        --validate docs/context/radius_sensitivity_gate0_audit_issue_6640.json

This is a diagnostic decision record only. It does not run benchmark episodes, change frozen
0.0.3.post1 metric semantics, run production compute, or establish a radius-sensitivity result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf.benchmark.radius_sensitivity_gate0_audit import (
    GATE0_DECISION_SCHEMA,
    load_gate0_decision,
    validate_gate0_decision,
    write_gate0_decision,
)

DEFAULT_OUTPUT = Path("docs/context/radius_sensitivity_gate0_audit_issue_6640.json")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and build or validate the Gate 0 decision."""
    parser = argparse.ArgumentParser(
        description=(
            "Build or validate the Gate 0 post-hoc feasibility-audit decision JSON "
            "(radius_sensitivity_gate0_decision.v1)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write the decision JSON (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        help="Re-load and re-validate an existing decision JSON file instead of writing.",
    )
    args = parser.parse_args(argv)

    if args.validate is not None:
        decision = load_gate0_decision(args.validate)
        print(f"validated {GATE0_DECISION_SCHEMA}: {args.validate}", file=sys.stderr)
        summary = decision["summary"]
        print(
            "summary: "
            f"{summary['total_outcomes']} outcomes, "
            f"{summary['re_derivable_count']} re-derivable, "
            f"{summary['replay_required_count']} replay-required",
            file=sys.stderr,
        )
        return 0

    path = write_gate0_decision(args.output)
    # Round-trip: reload to prove the written file validates.
    validate_gate0_decision(load_gate0_decision(path))
    print(f"wrote and round-trip-validated {GATE0_DECISION_SCHEMA}: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
