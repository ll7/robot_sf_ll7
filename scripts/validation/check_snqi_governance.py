#!/usr/bin/env python3
"""Fail-closed SNQI governance preflight for issues #3723 and #3699.

This script is deliberately diagnostic only. It makes the current Social
Navigation Quality Index (SNQI) weight provenance and term-normalization
situation explicit, but it does not choose canonical weights, change scoring,
or promote SNQI to a primary safety ranking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from robot_sf.benchmark.snqi.exit_codes import (
    EXIT_INPUT_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
)
from robot_sf.benchmark.snqi.normalization_inventory import (
    build_snqi_contribution_diagnostics,
    build_snqi_normalization_inventory,
)
from robot_sf.benchmark.snqi.weights_inventory import build_inventory_report

CLAIM_BOUNDARY = (
    "secondary_diagnostic_only: this preflight reports unresolved SNQI governance "
    "blockers from issues #3723 and #3699. It does not choose canonical weights, "
    "change normalization, change compute_snqi output, or make SNQI a primary "
    "safety ranking."
)

CONTRIBUTION_DIAGNOSTIC_METRICS = {
    "success": 1.0,
    "time_to_goal_norm": 3.0,
    "collisions": 9.0,
    "near_misses": 9.0,
    "comfort_exposure": 2.0,
    "force_exceed_events": 9.0,
    "jerk_mean": 9.0,
}
CONTRIBUTION_DIAGNOSTIC_WEIGHTS = {
    "w_success": 1.0,
    "w_time": 1.0,
    "w_collisions": 1.0,
    "w_near": 1.0,
    "w_comfort": 1.0,
    "w_force_exceed": 1.0,
    "w_jerk": 1.0,
}
CONTRIBUTION_DIAGNOSTIC_BASELINE_STATS = {
    "collisions": {"med": 0.0, "p95": 1.0},
    "near_misses": {"med": 0.0, "p95": 1.0},
    "force_exceed_events": {"med": 0.0, "p95": 1.0},
    "jerk_mean": {"med": 0.0, "p95": 1.0},
}


def _load_baseline_stats(path: Path | None) -> dict[str, dict[str, float]] | None:
    """Load optional baseline stats JSON for normalization coverage checks."""
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read baseline stats {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"could not parse baseline stats {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"baseline stats must be a JSON object: {path}")
    return raw


def build_governance_report(
    *,
    repo_root: Path,
    baseline_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build a structured SNQI governance diagnostic report."""
    weights = build_inventory_report(repo_root)
    normalization = build_snqi_normalization_inventory(baseline_stats)
    contribution_diagnostics = build_snqi_contribution_diagnostics(
        CONTRIBUTION_DIAGNOSTIC_METRICS,
        CONTRIBUTION_DIAGNOSTIC_WEIGHTS,
        CONTRIBUTION_DIAGNOSTIC_BASELINE_STATS,
    )

    blockers: list[dict[str, Any]] = []
    if weights.has_blocking_conflict:
        blockers.append(
            {
                "issue": 3723,
                "kind": "weight_provenance_conflict",
                "detail": (
                    "More than one SNQI source currently declares or implies "
                    "canonical status while disagreeing on weight direction."
                ),
            }
        )
    if normalization.mixed_scale:
        mixed_inputs = [
            {
                "term": term.term,
                "metric_key": term.metric_key,
                "weight_name": term.weight_name,
                "normalization_status": term.normalization_status,
                "measurement_basis": term.measurement_basis,
            }
            for term in normalization.penalty_terms
        ]
        blockers.append(
            {
                "issue": 3699,
                "kind": "mixed_normalization_basis",
                "detail": (
                    "SNQI penalty terms mix raw, unbounded terms with "
                    "baseline-normalized terms, so weights are not comparable "
                    "relative priorities."
                ),
                "mixed_inputs": mixed_inputs,
            }
        )
    if baseline_stats is not None and normalization.missing_baseline_coverage:
        blockers.append(
            {
                "issue": 3699,
                "kind": "missing_baseline_coverage",
                "detail": (
                    "At least one baseline-normalized SNQI term lacks median/p95 "
                    "coverage and would be silently zeroed by normalize_metric."
                ),
                "metrics": [t.metric_key for t in normalization.missing_baseline_coverage],
            }
        )

    return {
        "schema_version": "snqi_governance_preflight.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "status": "failed" if blockers else "passed",
        "blockers": blockers,
        "weights": weights.to_dict(),
        "normalization": normalization.to_dict(),
        "normalization_contributions": contribution_diagnostics,
    }


def _print_text_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable report."""
    print(CLAIM_BOUNDARY)
    print(f"SNQI governance status: {report['status']}")
    if report["blockers"]:
        print("Blocking diagnostics:")
        for blocker in report["blockers"]:
            print(f"  - issue #{blocker['issue']} {blocker['kind']}: {blocker['detail']}")
    else:
        print("No blocking SNQI governance diagnostics detected.")

    weight_records = report["weights"]["records"]
    print("Weight sources:")
    for record in weight_records:
        relpath = record["relpath"] or "<code default>"
        status = "available" if record["available"] else f"unavailable: {record['load_error']}"
        fingerprint = record["content_sha256"] or "n/a"
        print(
            f"  - {record['name']} ({record['kind']}, {relpath}): "
            f"canonical={record['declares_canonical']}; "
            f"dominant={record['dominant_term']}; scale={record['scale_class']}; "
            f"sha256={fingerprint}; {status}"
        )

    weight_conflicts = report["weights"]["conflicts"]
    if weight_conflicts:
        print("Weight provenance conflicts:")
        for conflict in weight_conflicts:
            sources = ", ".join(conflict["sources"])
            print(f"  - {conflict['severity']} {conflict['kind']} ({sources})")

    normalization = report["normalization"]
    print(
        "Normalization basis: "
        f"mixed_scale={normalization['mixed_scale']}; "
        f"raw_penalty_terms={', '.join(normalization['raw_penalty_terms']) or 'none'}; "
        f"unbounded_terms={', '.join(normalization['unbounded_terms']) or 'none'}"
    )
    print("Term normalization status:")
    for term in normalization["terms"]:
        role = "penalty" if term["is_penalty"] else "reward"
        print(
            f"  - {term['term']} ({term['metric_key']}, {term['weight_name']}): "
            f"{term['normalization_status']}; role={role}; "
            f"basis={term['measurement_basis']}; note={term['note']}"
        )
    contributions = report["normalization_contributions"]
    print(
        "Contribution diagnostic: "
        f"raw_penalty_share={contributions['raw_penalty_absolute_share']:.3f}; "
        "baseline_normalized_penalty_share="
        f"{contributions['baseline_normalized_penalty_absolute_share']:.3f}; "
        f"raw_penalty_terms_dominate={contributions['raw_penalty_terms_dominate']}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to inspect (default: current working directory).",
    )
    parser.add_argument(
        "--baseline-stats",
        type=Path,
        help="Optional baseline stats JSON for missing median/p95 coverage checks.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    parser.add_argument("--json-out", type=Path, help="Write JSON report to this path.")
    parser.add_argument(
        "--allow-current-blockers",
        action="store_true",
        help="Return success while still reporting current unresolved governance blockers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the combined SNQI governance preflight."""
    args = _build_parser().parse_args(argv)
    try:
        baseline_stats = _load_baseline_stats(args.baseline_stats)
        report = build_governance_report(repo_root=args.repo_root, baseline_stats=baseline_stats)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text_report(report)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if report["blockers"] and not args.allow_current_blockers:
        return EXIT_VALIDATION_ERROR
    return EXIT_SUCCESS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
