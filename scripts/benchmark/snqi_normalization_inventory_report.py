#!/usr/bin/env python3
"""Preflight report SNQI per-term normalization status (issue #3699).

`compute_snqi` (``robot_sf/benchmark/snqi/compute.py``) assembles SNQI terms on
inconsistent scales: ``time`` and ``comfort`` penalty terms are raw/unbounded,
while count-type penalty terms (collisions, near-misses, force-exceed events, jerk)
are baseline-normalized to ``[0, 1]``. Mixing raw and normalized terms makes
weight coefficients non-comparable as relative priorities and can let an
un-normalized penalty dominate composite regardless of weight (issue #3699).

This report is diagnostic-only: inventories each term's scaling regime, flags
mixed-scale conditions and missing baseline coverage, prints a human-readable
report, and optionally writes JSON payload. It does **not** change SNQI formula,
weights, or emitted scores, and does not resolve the issue's normalize-vs-bounded
decision (that remains decision-required for follow-up #3978).

`--fail-on-mixed-scale` exits non-zero when penalty terms span raw/normalized
scales; `--baseline-stats` plus `--fail-on-missing-baseline` exits non-zero when
normalized terms lack median/p95 coverage.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Sequence, Tuple

from robot_sf.benchmark.snqi.normalization_inventory import (
    build_snqi_contribution_diagnostics,
    build_snqi_normalization_inventory,
    format_normalization_report,
)

CLAIM_BOUNDARY = (
    "secondary_diagnostic_only: SNQI normalization inventory diagnostics for issue #3699. "
    "Does not change compute_snqi, weights, or emitted scores."
)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stats",
        type=pathlib.Path,
        default=None,
        help="Optional baseline-stats JSON (metric -> {med, p95}) check coverage.",
    )
    parser.add_argument(
        "--json-out",
        type=pathlib.Path,
        default=None,
        help="Optional path for JSON inventory payload.",
    )
    parser.add_argument(
        "--metrics",
        type=pathlib.Path,
        default=None,
        help="Optional episode metrics JSON for contribution diagnostics.",
    )
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        default=None,
        help="Optional SNQI weights JSON for contribution diagnostics.",
    )
    parser.add_argument(
        "--fail-on-mixed-scale",
        action="store_true",
        help="Exit non-zero while penalty terms span raw/normalized scales.",
    )
    parser.add_argument(
        "--fail-on-missing-baseline",
        action="store_true",
        help=(
            "Exit non-zero when baseline-normalized term lacks median/p95 coverage "
            "(requires --baseline-stats)."
        ),
    )
    return parser


def _load_json_object(path: pathlib.Path) -> dict[str, Any] | None:
    """Load a JSON object and return it, or ``None`` on input failures."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"invalid JSON input: {exc}", file=sys.stderr)
        return None
    if not isinstance(payload, dict):
        print(f"{path}: expected JSON object", file=sys.stderr)
        return None
    return payload


def _load_contribution_inputs(
    metrics_path: pathlib.Path | None, weights_path: pathlib.Path | None
) -> Tuple[int, dict[str, Any] | None, dict[str, Any] | None]:
    """Load optional metrics/weights JSON for contribution diagnostics."""
    if metrics_path is None and weights_path is None:
        return 0, None, None
    if metrics_path is None or weights_path is None:
        print("--metrics and --weights must be provided together", file=sys.stderr)
        return 2, None, None
    if not metrics_path.exists():
        print(f"metrics file not found: {metrics_path}", file=sys.stderr)
        return 2, None, None
    if not weights_path.exists():
        print(f"weights file not found: {weights_path}", file=sys.stderr)
        return 2, None, None

    metrics = _load_json_object(metrics_path)
    if metrics is None:
        return 2, None, None
    weights = _load_json_object(weights_path)
    if weights is None:
        return 2, None, None
    return 0, metrics, weights


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Run preflight normalization inventory report."""
    args = _build_parser().parse_args(argv)

    if args.fail_on_missing_baseline and args.baseline_stats is None:
        print(
            "--fail-on-missing-baseline requires --baseline-stats",
            file=sys.stderr,
        )
        return 2

    baseline_stats: dict[str, dict[str, float]] | None = None
    if args.baseline_stats is not None:
        if not args.baseline_stats.exists():
            print(f"baseline-stats file not found: {args.baseline_stats}", file=sys.stderr)
            return 2
        raw = _load_json_object(args.baseline_stats)
        if raw is None:
            return 2
        baseline_stats = {}
        for key, stats in raw.items():
            if not isinstance(key, str):
                continue
            if not isinstance(stats, dict):
                continue
            med = stats.get("med")
            p95 = stats.get("p95")
            if isinstance(med, (int, float)) and isinstance(p95, (int, float)):
                baseline_stats[key] = {"med": float(med), "p95": float(p95)}
        if not baseline_stats:
            print("baseline-stats did not contain usable median/p95 entries", file=sys.stderr)
            return 2

    input_exit_code, metrics, weights = _load_contribution_inputs(args.metrics, args.weights)
    if input_exit_code:
        return input_exit_code

    inventory = build_snqi_normalization_inventory(baseline_stats)

    contribution_payload = None
    if metrics is not None and weights is not None:
        # Assumption: contribution analysis is diagnostics-only and mirrors current
        # mixed-scale compute_snqi behavior intentionally.
        contribution_payload = build_snqi_contribution_diagnostics(
            metrics,
            weights,
            baseline_stats or {},
        )

    print(CLAIM_BOUNDARY)
    print(format_normalization_report(inventory))

    if contribution_payload is not None:
        print("Contribution diagnostics:")
        for term in contribution_payload["terms"]:
            print(
                f" - {term['term']}: scaled={term['scaled_value']}; "
                f"weight={term['weight']}; signed={term['signed_contribution']}; "
                f"absolute_share={term['absolute_share']}"
            )

    if args.json_out is not None:
        payload = {
            "claim_boundary": CLAIM_BOUNDARY,
            "inventory": inventory.to_dict(),
        }
        if contribution_payload is not None:
            payload["contributions"] = contribution_payload
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    exit_code = 0
    if args.fail_on_mixed_scale and inventory.mixed_scale:
        raw = ", ".join(term.term for term in inventory.raw_penalty_terms)
        print(
            f"FAIL: SNQI penalty terms mix raw ({raw}) baseline-normalized scales; "
            "weights are not comparable relative priorities.",
            file=sys.stderr,
        )
        exit_code = 1

    if args.fail_on_missing_baseline and inventory.missing_baseline_coverage:
        missing = ", ".join(metric for metric in inventory.missing_baseline_coverage)
        print(
            "FAIL: baseline-normalized SNQI terms lack median/p95 coverage "
            f"({missing}); normalize_metric would silently zero them.",
            file=sys.stderr,
        )
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
