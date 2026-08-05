#!/usr/bin/env python3
"""Run the issue #6645 narrow-doorway radius-binding audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.narrow_doorway_radius_audit import (
    DEFAULT_SCENARIO_PATH,
    build_audit_report,
    render_report,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse audit command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=_REPO_ROOT / DEFAULT_SCENARIO_PATH,
        help="Scenario manifest to audit.",
    )
    parser.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 1.0],
        help="Positive envelope radii for the runtime canary.",
    )
    parser.add_argument("--out-json", type=Path, help="Optional report output path.")
    return parser.parse_args()


def main() -> int:
    """Run the audit and return zero only when every check passes."""
    args = parse_args()
    scenario_path = args.scenario.expanduser()
    if not scenario_path.is_absolute():
        scenario_path = (_REPO_ROOT / scenario_path).resolve()
    report = build_audit_report(scenario_path, radii_m=tuple(args.radii))
    rendered = render_report(report)
    if args.out_json is not None:
        output_path = args.out_json.expanduser()
        if not output_path.is_absolute():
            output_path = (_REPO_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
