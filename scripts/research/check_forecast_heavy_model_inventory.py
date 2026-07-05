#!/usr/bin/env python3
"""Preflight heavy forecast-model families / offline-experiment surface (#2845).

This read-only CLI documents candidate heavy predictor families, probes the
offline evaluation surfaces needed by a future experiment, and reports the
minimum-experiment blockers. It trains nothing, runs no inference, adds no
dependency, runs no benchmark, and makes no model-quality claim.
"""

from __future__ import annotations

import argparse
import json

from robot_sf.research.forecast_heavy_model_inventory import (
    ENTRY_POINT_SURFACES,
    EXPERIMENT_PREREQUISITES,
    MODEL_FAMILIES,
    build_inventory_report,
    build_revival_decision_packet,
    render_markdown,
    render_revival_decision_packet_markdown,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable inventory report JSON.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print static inventory (no checkout probing) JSON and exit 0.",
    )
    parser.add_argument(
        "--decision-packet",
        action="store_true",
        help="Emit the fail-closed issue #2845 revival decision packet.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run inventory preflight CLI and return shell-friendly exit code."""
    args = build_arg_parser().parse_args(argv)
    if args.list:
        payload = {
            "issue": 2845,
            "model_families": [m.to_dict() for m in MODEL_FAMILIES],
            "entry_point_surfaces": [s.to_dict() for s in ENTRY_POINT_SURFACES],
            "experiment_prerequisites": [p.to_dict() for p in EXPERIMENT_PREREQUISITES],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    report = build_inventory_report()
    if args.decision_packet:
        packet = build_revival_decision_packet(report)
        if args.json:
            print(json.dumps(packet.to_dict(), indent=2, sort_keys=True))
        else:
            print(render_revival_decision_packet_markdown(packet))
        return report.exit_code()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return report.exit_code()


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
