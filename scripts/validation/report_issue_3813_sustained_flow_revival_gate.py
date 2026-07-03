#!/usr/bin/env python3
"""Report issue #3813 sustained-flow revival gate from issue #3810 h600 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.sustained_flow_revival_gate import (
    DEFAULT_H600_INTERACTION_EXPOSURE_EVIDENCE,
    build_sustained_flow_revival_gate_report,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interaction-exposure",
        type=Path,
        default=DEFAULT_H600_INTERACTION_EXPOSURE_EVIDENCE,
        help="Issue #3810 h600 interaction-exposure diagnostics JSON.",
    )
    parser.add_argument(
        "--claim-impact",
        type=Path,
        default=None,
        help=(
            "Optional JSON report stating affected rows and whether wait-it-out exclusions "
            "or caveats change claim decisions."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit full JSON report.")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Run the sustained-flow revival gate CLI."""

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    interaction_exposure = _load_json(args.interaction_exposure)
    claim_impact = _load_json(args.claim_impact) if args.claim_impact is not None else None
    report = build_sustained_flow_revival_gate_report(
        interaction_exposure,
        interaction_exposure_evidence_path=args.interaction_exposure,
        claim_impact=claim_impact,
        claim_impact_evidence_path=args.claim_impact,
    )
    payload = report.to_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"decision={payload['decision']} "
            f"evidence_status={payload['evidence_status']} "
            f"blocking_reasons={len(payload['blocking_reasons'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
