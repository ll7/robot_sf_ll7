#!/usr/bin/env python3
"""Radius-binding canary runner for the #6600 collision-envelope campaign (Gate 1).

Issue #6641 Gate 1: before any production SLURM sweep (Gate 2), prove on at least one
geometry-sensitive scenario that a declared robot collision-envelope radius propagates
consistently to all five binding surfaces -- simulator collision geometry, obstacle and
pedestrian contact logic, feasibility/oracle calculations, metric metadata and output
rows, and planner inputs that consume the radius -- and emit a machine-readable go/no-go
verdict per surface.

This is a small bounded CPU-only canary. It does NOT submit a production sweep and does
NOT change the frozen ``0.0.3.post1`` metric semantics. Any inconsistent or silently
ignored radius binding is a fail-closed no-go (exit code 1) that stops the campaign.

Default target: the geometry-sensitive ``francis2023_narrow_doorway`` scenario at the
#6600 fixed radius treatment (0.5 m, 0.8 m, and the 1.0 m release baseline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.radius_binding_canary import (
    CAMPAIGN_ENVELOPE_RADII_M,
    DEFAULT_TOLERANCE_M,
    DIAGNOSTIC_CLAIM_BOUNDARY,
    canary_verdict_to_dict,
    run_radius_binding_canary,
    validate_tolerance_m,
)
from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCENARIO = _REPO_ROOT / "configs/scenarios/single/francis2023_narrow_doorway.yaml"
_CAMPAIGN_ID = "issue_6600_gate_1"
_REPORT_SCHEMA = "radius_binding_canary_report.v1"


def _parse_tolerance(raw_value: str) -> float:
    """Parse a finite, non-negative radius comparison tolerance for the CLI."""
    try:
        return validate_tolerance_m(float(raw_value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    """Parse the bounded canary command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=_DEFAULT_SCENARIO,
        help="Geometry-sensitive scenario YAML (default: francis2023_narrow_doorway).",
    )
    parser.add_argument(
        "--scenario-name",
        default=None,
        help="Scenario entry name to select from the YAML (default: first entry).",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=list(CAMPAIGN_ENVELOPE_RADII_M),
        help="Envelope radii (metres) to probe (default: 0.5 0.8 1.0).",
    )
    parser.add_argument(
        "--tolerance",
        type=_parse_tolerance,
        default=DEFAULT_TOLERANCE_M,
        help=(
            "Radius comparison tolerance in metres (default: exact binding; capped at "
            "the canary safety bound)."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write the machine-readable canary report.",
    )
    return parser.parse_args()


def _select_scenario(scenario_path: Path, scenario_name: str | None) -> dict:
    """Load and select one scenario entry from a YAML manifest."""
    scenarios = [dict(item) for item in load_scenarios(scenario_path)]
    if not scenarios:
        raise ValueError(f"no scenarios found in {scenario_path}")
    if scenario_name is None:
        return scenarios[0]
    for scenario in scenarios:
        if scenario.get("name") == scenario_name:
            return scenario
    raise ValueError(f"scenario {scenario_name!r} not found in {scenario_path}")


def build_report(
    scenario: dict,
    *,
    scenario_path: Path,
    radii: list[float],
    tolerance: float,
) -> dict:
    """Run the canary at each radius and assemble the machine-readable report."""
    tolerance = validate_tolerance_m(tolerance)
    if not radii:
        raise ValueError("radii must contain at least one target radius")
    verdicts = [
        canary_verdict_to_dict(
            run_radius_binding_canary(
                scenario,
                radius,
                scenario_path=scenario_path,
                tolerance_m=tolerance,
            )
        )
        for radius in radii
    ]
    go = all(verdict["go"] for verdict in verdicts)
    return {
        "schema": _REPORT_SCHEMA,
        "canary_schema": "radius_binding_canary.v1",
        "campaign": _CAMPAIGN_ID,
        "issue": 6641,
        "parent_issue": 6600,
        "scenario_id": verdicts[0]["scenario_id"] if verdicts else scenario_path.stem,
        "scenario_path": str(scenario_path),
        "radii_m": [float(radius) for radius in radii],
        "tolerance_m": float(tolerance),
        "go": go,
        "verdicts": verdicts,
        "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
    }


def main() -> int:
    """Run the Gate 1 radius-binding canary and return a fail-closed process status."""
    args = parse_args()
    scenario_path = Path(args.scenario).expanduser()
    if not scenario_path.is_absolute():
        scenario_path = (_REPO_ROOT / scenario_path).resolve()
    if not scenario_path.exists():
        print(f"error: scenario file not found: {scenario_path}", file=sys.stderr)
        return 2
    if any(radius <= 0 for radius in args.radii):
        print("error: all --radii must be positive", file=sys.stderr)
        return 2
    try:
        scenario = _select_scenario(scenario_path, args.scenario_name)
        report = build_report(
            scenario,
            scenario_path=scenario_path,
            radii=list(args.radii),
            tolerance=args.tolerance,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json is not None:
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
