#!/usr/bin/env python3
"""Preflight the heavy forecast-model families / offline-experiment surface (#2845).

Read-only inventory CLI. It documents the candidate heavy predictor families
(AgentFormer-like / transformer / CVAE / diffusion) with their planning-stage compute cost,
inference latency, uncertainty quality, and repository integration burden; probes that the
offline-evaluation entry-point surfaces such an experiment would touch are importable
(fail-closed); and lists the still-missing minimum-offline-experiment prerequisites (a staged
held-out dataset, a heavy-model -> ForecastBatch adapter, a CPU runtime budget, the study
report, plus external dependency/checkpoint decisions) as explicit blockers.

Modes:

- default: render a compact Markdown report and exit non-zero only if a *required*
  offline-evaluation surface is missing.
- ``--json``: emit the machine-readable report instead of Markdown.
- ``--list``: print the static inventory (families, surfaces, prerequisites) without probing.

This tool trains nothing, runs no inference, adds no dependency, runs no benchmark, and makes no
model-quality claim. The per-family tiers are literature-derived planning estimates, not
repository measurements. See ``robot_sf/research/forecast_heavy_model_inventory.py``.
"""

from __future__ import annotations

import argparse
import json

from robot_sf.research.forecast_heavy_model_inventory import (
    ENTRY_POINT_SURFACES,
    EXPERIMENT_PREREQUISITES,
    MODEL_FAMILIES,
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
            "issue": 2845,
            "model_families": [m.to_dict() for m in MODEL_FAMILIES],
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
