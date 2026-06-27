#!/usr/bin/env python3
"""Preflight report for SNQI per-term normalization status (issue #3699).

``compute_snqi`` (``robot_sf/benchmark/snqi/compute.py``) assembles the SNQI
terms on inconsistent scales: the ``time`` and ``comfort`` penalty terms enter
raw and unbounded, while the count-type penalty terms (collisions, near-misses,
force-exceed events, jerk) are baseline-normalized to ``[0, 1]``. Mixing raw and
baseline-normalized terms makes the weight coefficients non-comparable as
relative priorities and lets an un-normalized term dominate the composite
regardless of its weight (see issue #3699).

This report is **diagnostic only**: it inventories each term's scaling regime,
flags the mixed-scale condition and any baseline-normalized term lacking
median/p95 coverage, prints a human-readable summary, and optionally writes a
JSON payload. It does **not** change the SNQI formula, the weights,
``normalize_metric``, or any emitted score, and it does **not** choose between
the normalize vs. clip-and-document remedies (that remains ``decision-required``
on issue #3699).

With ``--fail-on-mixed-scale`` the command exits non-zero while penalty terms
span both scales (fail-closed), so it can gate a preflight that must stay aware
of the inconsistency until it is resolved. With ``--baseline-stats`` plus
``--fail-on-missing-baseline`` it also fails when a normalized term would be
silently zeroed for lack of baseline coverage.

Usage::

    uv run python scripts/benchmark/snqi_normalization_inventory_report.py
    uv run python scripts/benchmark/snqi_normalization_inventory_report.py \
        --baseline-stats output/benchmarks/baseline_stats.json \
        --json-out output/snqi_normalization_inventory.json --fail-on-mixed-scale
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import TYPE_CHECKING

from robot_sf.benchmark.snqi.normalization_inventory import (
    build_snqi_normalization_inventory,
    format_normalization_report,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

CLAIM_BOUNDARY = (
    "diagnostic_only: inventories the per-term scaling of compute_snqi and flags where raw, "
    "unbounded terms (time, comfort) are mixed with baseline-normalized terms, making the "
    "weights non-comparable. It does not change the SNQI formula, weights, normalize_metric, or "
    "any emitted score, and does not choose a remedy (decision-required, issue #3699)."
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stats",
        type=pathlib.Path,
        default=None,
        help="Optional baseline-stats JSON (metric -> {med, p95}) to check coverage.",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Optional path to write the JSON inventory payload.",
    )
    parser.add_argument(
        "--fail-on-mixed-scale",
        action="store_true",
        help="Exit non-zero while penalty terms span raw and normalized scales (fail-closed).",
    )
    parser.add_argument(
        "--fail-on-missing-baseline",
        action="store_true",
        help=(
            "Exit non-zero when a baseline-normalized term lacks median/p95 coverage "
            "(requires --baseline-stats)."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preflight normalization inventory report.

    Returns:
        Process exit code (0 on success; 1 when a fail-closed flag trips).
    """
    args = _build_parser().parse_args(argv)

    baseline_stats: dict[str, dict[str, float]] | None = None
    if args.baseline_stats is not None:
        if not args.baseline_stats.exists():
            print(f"baseline-stats file not found: {args.baseline_stats}", file=sys.stderr)
            return 2
        baseline_stats = json.loads(args.baseline_stats.read_text(encoding="utf-8"))

    inventory = build_snqi_normalization_inventory(baseline_stats)

    print(CLAIM_BOUNDARY)
    print(format_normalization_report(inventory))

    if args.json_out is not None:
        payload = {"claim_boundary": CLAIM_BOUNDARY, "inventory": inventory.to_dict()}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    exit_code = 0
    if args.fail_on_mixed_scale and inventory.mixed_scale:
        raw = ", ".join(t.term for t in inventory.raw_penalty_terms)
        print(
            f"FAIL: SNQI penalty terms mix raw ({raw}) and baseline-normalized scales; "
            "weights are not comparable as relative priorities.",
            file=sys.stderr,
        )
        exit_code = 1
    if args.fail_on_missing_baseline and inventory.missing_baseline_coverage:
        missing = ", ".join(t.metric_key for t in inventory.missing_baseline_coverage)
        print(
            f"FAIL: baseline-normalized SNQI terms lack median/p95 coverage ({missing}); "
            "normalize_metric would silently zero them.",
            file=sys.stderr,
        )
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
