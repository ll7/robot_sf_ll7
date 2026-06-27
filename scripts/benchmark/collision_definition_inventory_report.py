#!/usr/bin/env python3
"""Preflight report for the two collision/near-miss definitions (issue #3724).

The benchmark metric (``robot_sf/benchmark/metrics.py``) classifies collision and
near-miss events with a radius-aware *clearance* rule, while the SNQI proxy and
policy-search validation paths (``robot_sf/gym_env/snqi_proxy.py``,
``scripts/validation/policy_search_common.py``) use the *raw center distance*
against the named constants. The two regimes label the same geometry differently
across a wide band (see issue #3724).

This report is **diagnostic only**: it inventories where the two regimes diverge
over a deterministic synthetic center-distance sweep, prints a human-readable
summary, and optionally writes a JSON payload. It does **not** change any
threshold, metric, proxy, or validation behavior, and it does **not** choose a
canonical definition (that remains ``decision-required`` on issue #3724).

With ``--fail-on-divergence`` the command exits non-zero when the regimes
disagree (fail-closed), so it can be wired into a preflight gate that must stay
aware of the inconsistency until it is resolved.

Usage::

    uv run python scripts/benchmark/collision_definition_inventory_report.py
    uv run python scripts/benchmark/collision_definition_inventory_report.py \
        --json-out output/collision_definition_inventory.json --fail-on-divergence
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import TYPE_CHECKING

from robot_sf.benchmark.collision_definition_inventory import (
    DEFAULT_PED_RADIUS,
    DEFAULT_ROBOT_RADIUS,
    collision_definition_inventory,
    format_divergence_report,
    synthetic_center_distance_sweep,
)
from robot_sf.benchmark.constants import COLLISION_DIST, NEAR_MISS_DIST

if TYPE_CHECKING:
    from collections.abc import Sequence

CLAIM_BOUNDARY = (
    "diagnostic_only: inventories where the clearance-based benchmark metric and the "
    "center-distance proxy/validation paths label collision/near-miss differently. It does "
    "not change thresholds, metrics, the proxy, or validation, and does not choose a canonical "
    "definition (decision-required, issue #3724)."
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, default=0.0, help="Sweep start (m).")
    parser.add_argument("--stop", type=float, default=3.0, help="Sweep stop (m, inclusive).")
    parser.add_argument("--step", type=float, default=0.05, help="Sweep step (m).")
    parser.add_argument(
        "--robot-radius", type=float, default=DEFAULT_ROBOT_RADIUS, help="Robot radius (m)."
    )
    parser.add_argument(
        "--ped-radius", type=float, default=DEFAULT_PED_RADIUS, help="Pedestrian radius (m)."
    )
    parser.add_argument(
        "--collision-dist",
        type=float,
        default=COLLISION_DIST,
        help="Center-distance regime collision threshold (m).",
    )
    parser.add_argument(
        "--near-miss-dist",
        type=float,
        default=NEAR_MISS_DIST,
        help="Shared near-miss upper bound (m).",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Optional path to write the JSON inventory payload.",
    )
    parser.add_argument(
        "--fail-on-divergence",
        action="store_true",
        help="Exit non-zero when the two regimes disagree (fail-closed).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preflight inventory report.

    Returns:
        Process exit code (0 on success; 1 when divergence is found and
        ``--fail-on-divergence`` is set).
    """
    args = _build_parser().parse_args(argv)

    sweep = synthetic_center_distance_sweep(start=args.start, stop=args.stop, step=args.step)
    inventory = collision_definition_inventory(
        sweep,
        robot_radius=args.robot_radius,
        ped_radius=args.ped_radius,
        collision_dist=args.collision_dist,
        near_miss_dist=args.near_miss_dist,
    )

    print(CLAIM_BOUNDARY)
    print(format_divergence_report(inventory))

    if args.json_out is not None:
        payload = {"claim_boundary": CLAIM_BOUNDARY, "inventory": inventory.to_dict()}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    if args.fail_on_divergence and not inventory.regimes_agree:
        print(
            "FAIL: collision/near-miss definitions diverge "
            f"({inventory.divergent}/{inventory.sample_count} samples).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
