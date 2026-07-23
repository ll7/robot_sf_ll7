#!/usr/bin/env python3
"""Run the blind-corner zero-success diagnostic for issue #5596.

Produces a machine-readable JSON report explaining why
``francis2023_blind_corner`` stays zero-success for the scripted traversal,
with envelope-sensitivity oracle, route-follow intervention, and clearance
comparison evidence.

Usage:
    python scripts/tools/run_blind_corner_diagnostic_issue_5596.py
        --output docs/context/evidence/issue_5596_blind_corner_diagnostic/blind_corner_diagnostic.json
"""
# evidence-writer-exempt: references evidence paths but does not write to evidence tree; guarded by AST analysis

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.scenario_certification.feasibility_oracle import (
    ISSUE_5596_BLIND_CORNER_SCENARIO_ID,
    build_issue_5596_blind_corner_diagnostic,
)

DEFAULT_MANIFEST = Path("configs/scenarios/francis2023.yaml")


def main() -> None:
    """Parse arguments and run the blind-corner diagnostic."""
    parser = argparse.ArgumentParser(
        description="Blind-corner zero-success diagnostic (issue #5596)"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Scenario manifest YAML containing the blind-corner cell",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/context/evidence/issue_5596_blind_corner_diagnostic/blind_corner_diagnostic.json"
        ),
        help="Output path for the diagnostic JSON report",
    )
    parser.add_argument(
        "--envelope-radii",
        type=float,
        nargs="+",
        default=[1.0, 0.5],
        help="Envelope radii to probe (nominal first, then reduced)",
    )
    args = parser.parse_args()

    print(f"Running issue #5596 diagnostic for {ISSUE_5596_BLIND_CORNER_SCENARIO_ID}...")
    print(f"  Manifest: {args.manifest}")
    print(f"  Envelope radii: {args.envelope_radii}")
    print(f"  Output:   {args.output}")

    try:
        report = build_issue_5596_blind_corner_diagnostic(
            args.manifest,
            envelope_radii_m=tuple(args.envelope_radii),
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report written to {args.output}")

    # Print mechanism verdict for visibility.
    mechanism = report.get("mechanism", {})
    supported = mechanism.get("supported_explanation", "unknown")
    print(f"Mechanism: {supported}")
    print(f"  Claim boundary: {mechanism.get('claim_boundary', 'N/A')}")

    oracle = report.get("oracle_verdict", {})
    nominal = oracle.get("nominal_verdict", {})
    print(f"  Oracle nominal feasible: {nominal.get('feasible')}")
    print(f"  Oracle category: {oracle.get('category', nominal.get('status', 'N/A'))}")

    rf = report.get("route_follow_intervention_verdict", {})
    results = rf.get("results", [])
    if results:
        print(f"  Route-follow nominal: {results[0].get('status', 'N/A')}")


if __name__ == "__main__":
    main()
