#!/usr/bin/env python3
"""Fail-closed SNQI term-normalization inventory for issue #3699.

This checker is diagnostic only. It reports the current Social Navigation
Quality Index (SNQI) term scaling basis and fails when penalty terms mix raw
and baseline-normalized inputs. It does not change SNQI scoring, weights, or
normalization behavior.
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

CLAIM_BOUNDARY = (
    "secondary_diagnostic_only: reports issue #3699 SNQI term-normalization "
    "status. It does not normalize raw terms, alter weights, change "
    "compute_snqi output, or promote SNQI to a primary safety ranking."
)


def _load_baseline_stats(path: Path | None) -> dict[str, dict[str, float]] | None:
    """Load optional baseline stats for normalized-term coverage diagnostics."""
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


def _load_json_object(path: Path | None, *, label: str) -> dict[str, Any] | None:
    """Load an optional JSON object for synthetic contribution diagnostics."""
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(f"{label} file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load {label} JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be JSON object: {path}")
    return raw


def build_normalization_preflight_report(
    baseline_stats: dict[str, dict[str, float]] | None = None,
    *,
    metrics: dict[str, Any] | None = None,
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the issue #3699 normalization inventory preflight payload."""
    if (metrics is None) != (weights is None):
        raise ValueError("metrics and weights must be provided together")

    inventory = build_snqi_normalization_inventory(baseline_stats)
    blockers: list[dict[str, Any]] = []

    if inventory.mixed_scale:
        blockers.append(
            {
                "issue": 3699,
                "kind": "mixed_normalization_basis",
                "detail": (
                    "SNQI penalty terms mix raw inputs with baseline-normalized "
                    "inputs, so weights are not comparable relative priorities."
                ),
                "raw_penalty_terms": [term.term for term in inventory.raw_penalty_terms],
                "normalized_penalty_terms": [
                    term.term for term in inventory.normalized_penalty_terms
                ],
            }
        )

    if baseline_stats is not None and inventory.missing_baseline_coverage:
        blockers.append(
            {
                "issue": 3699,
                "kind": "missing_baseline_coverage",
                "detail": (
                    "At least one baseline-normalized SNQI term lacks median/p95 "
                    "coverage and would be silently zeroed by normalize_metric."
                ),
                "metrics": [term.metric_key for term in inventory.missing_baseline_coverage],
            }
        )

    report: dict[str, Any] = {
        "schema_version": "snqi_normalization_inventory_preflight.v1",
        "claim_boundary": CLAIM_BOUNDARY,
        "status": "failed" if blockers else "passed",
        "blockers": blockers,
        "normalization": inventory.to_dict(),
    }

    decision_packet = {
        "issue": 3699,
        "decision_required": inventory.mixed_scale,
        "assumption": (
            "No score semantics changed; this report only exposes mixed-basis "
            "normalization diagnostics and baseline-coverage gaps for issue #3699."
        ),
        "mixed_scale": inventory.mixed_scale,
    }

    if metrics is not None and weights is not None:
        contributions = build_snqi_contribution_diagnostics(
            metrics,
            weights,
            baseline_stats or {},
        )
        report["contributions"] = contributions
        contract = contributions["normalization_contract"]
        decision_packet.update(
            {
                "normalization_contract_status": contract["status"],
                "weights_comparable": contract["weights_comparable"],
                "raw_penalty_absolute_share": contract["raw_penalty_absolute_share"],
                "baseline_normalized_penalty_absolute_share": contract["baseline_normalized_penalty_absolute_share"],
                "raw_penalty_terms": contract["raw_unbounded_penalty_terms"],
                "baseline_normalized_penalty_terms": contract["baseline_normalized_penalty_terms"],
                "weight_bound_exceedance_terms": contract["weight_bound_exceedance_terms"],
            }
        )

    report["normalization_checker"] = decision_packet
    return report


def _print_text_report(report: dict[str, Any]) -> None:
    """Print a compact human-readable preflight report."""
    print(CLAIM_BOUNDARY)
    print(f"SNQI normalization inventory status: {report['status']}")
    if report["blockers"]:
        print("Blocking diagnostics:")
        for blocker in report["blockers"]:
            print(f" - issue #{blocker['issue']} {blocker['kind']}: {blocker['detail']}")
    else:
        print("No blocking SNQI normalization diagnostics detected.")

    normalization = report["normalization"]
    print(
        "Normalization basis: "
        f"mixed_scale={normalization['mixed_scale']}; "
        f"raw_penalty_terms={', '.join(normalization['raw_penalty_terms']) or 'none'}; "
        "baseline_normalized_penalty_terms="
        f"{', '.join(normalization['normalized_penalty_terms']) or 'none'}"
    )
    version_contract = normalization["score_version_contract"]
    print(
        "Score version contract: "
        f"{version_contract['score_version']}; "
        f"status={version_contract['status']}; "
        f"diagnostic_only={version_contract['diagnostic_only']}"
    )
    print("Term normalization status:")
    for term in normalization["terms"]:
        role = "penalty" if term["is_penalty"] else "reward"
        print(
            f" - {term['term']} ({term['metric_key']}, {term['weight_name']}): "
            f"{term['normalization_status']}; role={role}; "
            f"basis={term['measurement_basis']}; note={term['note']}"
        )
    if "contributions" in report:
        contract = report["contributions"]["normalization_contract"]
        print(
            "Contribution contract: "
            f"status={contract['status']}; "
            f"weights_comparable={contract['weights_comparable']}; "
            f"raw_penalty_share={contract['raw_penalty_absolute_share']:.6g}; "
            "baseline_normalized_penalty_share="
            f"{contract['baseline_normalized_penalty_absolute_share']:.6g}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-stats",
        type=Path,
        help="Optional baseline-stats JSON for normalized-term coverage checks.",
    )
    parser.add_argument("--metrics", type=Path, help="Optional fixture metrics JSON object.")
    parser.add_argument("--weights", type=Path, help="Optional fixture weights JSON object.")
    parser.add_argument("--json", action="store_true", help="Emit JSON to stdout.")
    parser.add_argument("--json-out", type=Path, help="Write JSON report to this path.")
    parser.add_argument(
        "--allow-mixed-basis",
        action="store_true",
        help="Exit 0 after reporting current mixed-basis diagnostics.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the issue #3699 normalization inventory preflight."""
    args = _build_parser().parse_args(argv)
    try:
        baseline_stats = _load_baseline_stats(args.baseline_stats)
        metrics = _load_json_object(args.metrics, label="metrics")
        weights = _load_json_object(args.weights, label="weights")
        report = build_normalization_preflight_report(
            baseline_stats,
            metrics=metrics,
            weights=weights,
        )
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

    if report["blockers"] and not args.allow_mixed_basis:
        return EXIT_VALIDATION_ERROR
    return EXIT_SUCCESS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
