#!/usr/bin/env python3
"""Preflight the pedestrian-model assumptions/artifacts for HSFM + TTC experiments (#3481).

Read-only inventory CLI. It documents the current pedestrian force-model assumptions, probes
that the entry-point surfaces an HSFM/TTC experiment would touch are importable (fail-closed),
and lists the still-missing prerequisites (HSFM heading state, FoV attenuation, TTC term,
fixtures, versioned parameters, calibration data) as explicit blockers.

Modes:

- default: render a compact Markdown report and exit non-zero only if a *required* entry-point
  surface is missing.
- ``--json``: emit the machine-readable report instead of Markdown.
- ``--list``: print the static inventory (assumptions, surfaces, prerequisites) without probing.

This tool implements no force law, changes no scenario behavior, runs no benchmark, and makes
no realism claim. See ``robot_sf/research/ped_model_assumption_inventory.py``.
"""

from __future__ import annotations

import argparse
import json

from robot_sf.research.ped_model_assumption_inventory import (
    CURRENT_ASSUMPTIONS,
    ENTRY_POINT_SURFACES,
    EXPERIMENT_PREREQUISITES,
    build_inventory_report,
    render_markdown,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable inventory report as JSON.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the static inventory (no checkout probing) as JSON and exit 0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the inventory preflight CLI and return a shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)

    if args.list:
        payload = {
            "issue": 3481,
            "assumptions": [a.to_dict() for a in CURRENT_ASSUMPTIONS],
            "entry_point_surfaces": [s.to_dict() for s in ENTRY_POINT_SURFACES],
            "experiment_prerequisites": [p.to_dict() for p in EXPERIMENT_PREREQUISITES],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    report = build_inventory_report()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return report.exit_code()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
