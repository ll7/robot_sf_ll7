#!/usr/bin/env python3
"""Emit issue #5574's deterministic zero-success candidate-cell oracle report.

The report runs the existing planner-free oracle on the Francis 2023 narrow-doorway and
blind-corner cells by default. It is diagnostic-only evidence: no learned planner or benchmark
campaign is executed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def _build_parser() -> argparse.ArgumentParser:
    """Build the issue-specific oracle report CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario_config",
        type=Path,
        nargs="?",
        default=Path("configs/scenarios/francis2023.yaml"),
        help="Scenario manifest containing the candidate cells.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        help="Scenario cell to inspect; repeat for multiple cells (defaults to issue candidates).",
    )
    parser.add_argument(
        "--nominal-envelope-radius",
        type=float,
        help="Nominal collision-envelope radius in metres (default: repository nominal).",
    )
    parser.add_argument(
        "--reduced-envelope-radius",
        type=float,
        action="append",
        dest="reduced_envelope_radii",
        help="Reduced collision-envelope radius in metres; repeat for more probes.",
    )
    parser.add_argument(
        "--rollout-seed",
        type=int,
        help="Override the first seed in each scenario manifest.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
        help="Log level for simulator diagnostics (default: WARNING).",
    )
    parser.add_argument("--output", type=Path, help="Write the report to this JSON path.")
    return parser


def _resolve_envelope_radii(
    nominal_radius: float | None,
    reduced_radii: list[float] | None,
    default_radii: tuple[float, ...],
) -> tuple[float, ...]:
    """Resolve CLI envelope probes while preventing an implicit mismatched pair."""
    nominal = default_radii[0] if nominal_radius is None else nominal_radius
    if reduced_radii is None:
        if nominal != default_radii[0]:
            raise ValueError(
                "--reduced-envelope-radius is required when changing the nominal radius"
            )
        return default_radii
    return (nominal, *reduced_radii)


def main() -> int:
    """Run the issue #5574 oracle report command."""
    parser = _build_parser()
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    from robot_sf.scenario_certification.feasibility_oracle import (
        DEFAULT_ENVELOPE_RADII_M,
        DEFAULT_ISSUE_5574_SCENARIO_IDS,
        build_issue_5574_feasibility_report,
    )

    try:
        envelope_radii = _resolve_envelope_radii(
            args.nominal_envelope_radius,
            args.reduced_envelope_radii,
            DEFAULT_ENVELOPE_RADII_M,
        )
        report = build_issue_5574_feasibility_report(
            args.scenario_config,
            scenario_ids=tuple(args.scenario_ids or DEFAULT_ISSUE_5574_SCENARIO_IDS),
            envelope_radii_m=envelope_radii,
            rollout_seed=args.rollout_seed,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
