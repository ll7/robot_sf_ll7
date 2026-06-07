#!/usr/bin/env python3
"""Create a controlled counterfactual scenario-pair manifest for mechanism tests."""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.scenario_certification.perturbation_family_registry import (
    perturbation_family,
    validate_perturbation_family_parameters,
)
from robot_sf.scenario_certification.perturbation_preflight import (
    PERTURBATION_MANIFEST_SCHEMA_VERSION,
    preflight_perturbation_manifest,
    preflight_to_dict,
)

COUNTERFACTUAL_PAIR_SCHEMA_VERSION = "counterfactual_scenario_pair.v1"
MECHANISM_TAXONOMY_SCHEMA_VERSION = "counterfactual_mechanism_taxonomy.v1"
CLAIM_BOUNDARY = "candidate mechanism-test inputs only; not benchmark evidence"
_SUCCESS_EVIDENCE_CANDIDATE = "eligible_success_evidence_candidate"
_SUPPORTED_FEATURES = frozenset({"robot_route_offset"})
MECHANISM_TAXONOMY_LABELS = (
    "clearance_pressure",
    "bottleneck_negotiation",
    "occlusion_exposure",
    "signal_compliance",
    "cyclist_interaction",
    "actuation_clipping",
)
_MECHANISM_TAXONOMY_BY_FEATURE = {
    "robot_route_offset": {
        "label": "clearance_pressure",
        "label_source": MECHANISM_TAXONOMY_SCHEMA_VERSION,
        "mechanism_hypothesis": (
            "A bounded robot-route offset changes clearance pressure while holding the scenario, "
            "planner, and seed fixed."
        ),
        "expected_metric_direction": {
            "clearance_min_distance_m": "direction_depends_on_offset_sign",
            "collision_or_near_miss_risk": "may_increase_when_offset_reduces_clearance",
            "success": "no_directional_claim_from_pair_manifest",
        },
        "validity_constraints": [
            "baseline and intervention must both pass perturbation preflight",
            "seed and source scenario must remain unchanged",
            "single pair is a mechanism hypothesis input, not causal evidence",
        ],
    }
}

if TYPE_CHECKING:
    from collections.abc import Sequence


class CounterfactualPairError(ValueError):
    """Raised when a counterfactual pair cannot be created safely."""


def _build_parser() -> argparse.ArgumentParser:
    """Build the counterfactual scenario-pair parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Source scenario id to pair.")
    parser.add_argument(
        "--feature",
        required=True,
        help="Perturbation family to vary. This narrow slice supports robot_route_offset.",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        required=True,
        help="Feature magnitude. For robot_route_offset this is dx_m in meters.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Replay seed held fixed.")
    parser.add_argument(
        "--scenario-config",
        type=Path,
        required=True,
        help="Scenario matrix containing the source scenario.",
    )
    parser.add_argument("--output", type=Path, help="Write the YAML pair manifest to this path.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the YAML pair manifest instead of writing --output.",
    )
    return parser


def create_pair_manifest(
    *,
    source: str,
    feature: str,
    magnitude: float,
    seed: int,
    scenario_config: Path,
) -> dict[str, Any]:
    """Create a preflight-validated counterfactual pair payload.

    Returns:
        dict[str, Any]: YAML-safe pair manifest payload.
    """
    if seed < 0:
        raise CounterfactualPairError("seed must be non-negative")
    family = _supported_family(feature)
    parameters = _intervention_parameters(feature=family.name, magnitude=magnitude)
    parameter_reasons, family_entry = validate_perturbation_family_parameters(
        family.name,
        parameters,
    )
    if parameter_reasons:
        raise CounterfactualPairError("; ".join(parameter_reasons))

    baseline_variant_id = f"{source}_baseline_seed_{seed}"
    intervention_variant_id = f"{source}_{family.name}_{_magnitude_token(magnitude)}_seed_{seed}"
    perturbation_manifest = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"{source}_{family.name}_counterfactual_seed_{seed}",
        "scenario_config": scenario_config.as_posix(),
        "seed_controls": {
            "baseline_seeds": [seed],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": abs(magnitude),
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": baseline_variant_id,
                "scenario_id": source,
                "family": "noop",
                "seeds": [seed],
            },
            {
                "variant_id": intervention_variant_id,
                "scenario_id": source,
                "family": family.name,
                "seeds": [seed],
                "parameters": dict(parameters),
            },
        ],
    }
    preflight_payload = _preflight_embedded_manifest(perturbation_manifest)
    _require_pair_preflight_success(preflight_payload)

    return {
        "schema_version": COUNTERFACTUAL_PAIR_SCHEMA_VERSION,
        "baseline": {
            "scenario_id": source,
            "variant_id": baseline_variant_id,
            "family": "noop",
            "seed": seed,
        },
        "intervention": {
            "scenario_id": source,
            "variant_id": intervention_variant_id,
            "family": family.name,
            "seed": seed,
            "parameters": dict(parameters),
        },
        "changed_feature": family.name,
        "changed_factor": family.name,
        "unchanged_controls": {
            "scenario_config": scenario_config.as_posix(),
            "source_scenario_id": source,
            "seed": seed,
            "replay_seed_policy": "explicit",
            "baseline_family": "noop",
            "intervention_family": family.name,
        },
        "validity_status": "valid",
        "expected_mechanism": {
            "description": family_entry.description,
            "target_surface": family_entry.target_surface,
            "semantic_boundary": family_entry.semantic_boundary,
        },
        "mechanism_taxonomy": _mechanism_taxonomy_for_feature(family.name),
        "pair_report": _pair_report(
            source=source,
            feature=family.name,
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "perturbation_manifest": perturbation_manifest,
        "preflight": preflight_payload,
    }


def _supported_family(feature: str):
    """Return the registered family entry for a supported counterfactual feature."""
    try:
        family = perturbation_family(feature)
    except ValueError as exc:
        raise CounterfactualPairError(f"unsupported counterfactual feature: {feature}") from exc
    if family.name not in _SUPPORTED_FEATURES:
        raise CounterfactualPairError(
            f"unsupported counterfactual feature: {feature}; "
            f"supported: {sorted(_SUPPORTED_FEATURES)}"
        )
    return family


def _intervention_parameters(*, feature: str, magnitude: float) -> dict[str, float]:
    """Map the CLI feature magnitude onto registered perturbation-family parameters."""
    if feature != "robot_route_offset":
        raise CounterfactualPairError(f"unsupported counterfactual feature: {feature}")
    if not math.isfinite(magnitude):
        raise CounterfactualPairError("magnitude must be finite")
    if magnitude == 0.0:
        raise CounterfactualPairError("magnitude must be non-zero for a counterfactual pair")
    max_magnitude = abs(magnitude)
    return {
        "dx_m": magnitude,
        "dy_m": 0.0,
        "max_magnitude_m": max_magnitude,
    }


def _mechanism_taxonomy_for_feature(feature: str) -> dict[str, Any]:
    """Return the explicit mechanism taxonomy row for a supported feature.

    Returns:
        Mechanism taxonomy row.
    """
    row = _MECHANISM_TAXONOMY_BY_FEATURE.get(feature)
    if row is None:
        raise CounterfactualPairError(f"missing mechanism taxonomy for feature: {feature}")
    label = row.get("label")
    if label not in MECHANISM_TAXONOMY_LABELS:
        raise CounterfactualPairError(f"unsupported mechanism taxonomy label: {label!r}")
    return dict(row)


def _pair_report(*, source: str, feature: str) -> dict[str, Any]:
    """Return reportability metadata for why-first consumers.

    Returns:
        Counterfactual pair report row.
    """
    return {
        "base_scenario_id": source,
        "counterfactual_scenario_id": source,
        "changed_factor": feature,
        "artifact_manifest_ref": "perturbation_manifest",
        "expected_vs_observed_metric_change": {
            "status": "not_available",
            "reason": "no smoke-run metrics were supplied to this pair manifest",
        },
    }


def _preflight_embedded_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Preflight an embedded perturbation manifest through the existing file API."""
    with tempfile.TemporaryDirectory(prefix="robot_sf_counterfactual_pair_") as tmp_dir:
        manifest_path = Path(tmp_dir) / "perturbation_manifest.yaml"
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        report = preflight_perturbation_manifest(manifest_path)
    payload = preflight_to_dict(report)
    payload["manifest_path"] = "embedded:perturbation_manifest"
    payload["scenario_config"] = str(manifest["scenario_config"])
    return _compact_preflight_payload(payload)


def _compact_preflight_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a reviewable preflight summary without route-planner internals."""
    compact_results = []
    for row in payload["results"]:
        certificate = row.get("certificate")
        compact_certificate = None
        if isinstance(certificate, dict):
            checks = (
                certificate.get("checks") if isinstance(certificate.get("checks"), dict) else {}
            )
            compact_certificate = {
                "classification": certificate.get("classification"),
                "benchmark_eligibility": certificate.get("benchmark_eligibility"),
                "reasons": list(certificate.get("reasons") or []),
                "route_count": checks.get("route_count"),
            }
        compact_results.append(
            {
                "variant_id": row["variant_id"],
                "scenario_id": row["scenario_id"],
                "family": row["family"],
                "seeds": list(row["seeds"]),
                "validity_status": row["validity_status"],
                "benchmark_evidence_status": row["benchmark_evidence_status"],
                "reasons": list(row["reasons"]),
                "perturbation_summary": dict(row["perturbation_summary"]),
                "certificate": compact_certificate,
            }
        )
    return {
        "schema_version": payload["schema_version"],
        "manifest_id": payload["manifest_id"],
        "manifest_path": payload["manifest_path"],
        "scenario_config": payload["scenario_config"],
        "results": compact_results,
    }


def _require_pair_preflight_success(preflight_payload: dict[str, Any]) -> None:
    """Fail closed unless both baseline and intervention are preflight-eligible."""
    results = preflight_payload.get("results")
    if not isinstance(results, list) or len(results) != 2:
        raise CounterfactualPairError("preflight did not return exactly baseline and intervention")
    failures = [
        row
        for row in results
        if row.get("validity_status") != "valid"
        or row.get("benchmark_evidence_status") != _SUCCESS_EVIDENCE_CANDIDATE
    ]
    if failures:
        reasons = []
        for row in failures:
            variant_id = row.get("variant_id", "<unknown>")
            row_reasons = row.get("reasons") or ["preflight did not prove eligibility"]
            reasons.append(f"{variant_id}: {'; '.join(str(reason) for reason in row_reasons)}")
        raise CounterfactualPairError(
            "preflight rejected counterfactual pair: " + " | ".join(reasons)
        )


def _magnitude_token(magnitude: float) -> str:
    """Return a stable identifier token for a feature magnitude."""
    raw = f"{magnitude:.3f}".replace("-", "m").replace(".", "p")
    return raw


def dump_pair_manifest(payload: dict[str, Any]) -> str:
    """Serialize a pair manifest deterministically."""
    return yaml.safe_dump(payload, sort_keys=False)


def main(argv: Sequence[str] | None = None) -> int:
    """Run counterfactual scenario-pair creation."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    try:
        payload = create_pair_manifest(
            source=args.source,
            feature=args.feature,
            magnitude=args.magnitude,
            seed=args.seed,
            scenario_config=args.scenario_config,
        )
    except CounterfactualPairError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    output = dump_pair_manifest(payload)
    if args.dry_run:
        print(output, end="")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
