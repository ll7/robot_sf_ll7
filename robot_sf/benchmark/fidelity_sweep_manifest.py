"""Dry-run fidelity sweep manifest builder for issue #3207.

This module only enumerates the planned fidelity sensitivity matrix. It does
not bind config patches to runtime objects, run benchmark episodes, or promote
benchmark evidence.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.fidelity_sensitivity import validate_fidelity_sensitivity_config

FIDELITY_SWEEP_MANIFEST_SCHEMA = "fidelity-sweep-manifest.v1"
FIDELITY_SWEEP_MANIFEST_CHECK_SCHEMA = "fidelity-sweep-manifest-check.v1"
UNRESOLVED_RUNTIME_BINDING = "unresolved_runtime_binding"
DRY_RUN_CLAIM_BOUNDARY = (
    "dry-run manifest only: enumerates the fixed issue #3207 fidelity sensitivity scope; "
    "not benchmark evidence, not simulator-realism evidence, not sim-to-real evidence, "
    "and not paper-facing evidence."
)
SURFACE_RELATIONSHIP = (
    "axes is canonical for this dry-run manifest; fidelity_axes is retained as a "
    "secondary analysis-contract surface and is not used to enumerate variants."
)


@dataclass(frozen=True)
class ManifestOptions:
    """Stable metadata for a dry-run manifest build."""

    config_path: str
    git_head: str = "unknown"
    dry_run: bool = True


def build_fidelity_sweep_manifest(
    config: Mapping[str, Any],
    *,
    options: ManifestOptions,
) -> dict[str, Any]:
    """Build a deterministic dry-run manifest from the canonical `axes` surface.

    Returns:
        JSON-serializable manifest payload.
    """
    if not options.dry_run:
        raise ValueError("fidelity sweep manifest builder only supports dry-run manifests")

    validated = validate_fidelity_sensitivity_config(config)
    fixed_scope = validated["fixed_scope"]
    axes = [_axis_manifest(axis) for axis in validated["axes"]]
    return {
        "schema_version": FIDELITY_SWEEP_MANIFEST_SCHEMA,
        "issue": int(validated.get("issue", 3207)),
        "study_id": str(validated["study_id"]),
        "status": "manifest_dry_run_only",
        "dry_run": True,
        "evidence_status": "not_benchmark_evidence",
        "claim_boundary": DRY_RUN_CLAIM_BOUNDARY,
        "config_path": options.config_path,
        "git_head": options.git_head,
        "fixed_scope": copy.deepcopy(fixed_scope),
        "seeds": copy.deepcopy(fixed_scope["seeds"]),
        "planner_groups": copy.deepcopy(fixed_scope["planner_groups"]),
        "baseline": copy.deepcopy(validated.get("baseline")),
        "ranking": copy.deepcopy(validated["ranking"]),
        "metrics": copy.deepcopy(validated["metrics"]),
        "result_contract": copy.deepcopy(validated.get("result_contract")),
        "axis_count": len(axes),
        "axes": axes,
        "config_surface_relationship": _surface_relationship(validated),
    }


def write_fidelity_sweep_manifest(manifest: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a deterministic JSON dry-run manifest.

    Returns:
        Path to the written JSON manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "fidelity_sweep_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def check_fidelity_sweep_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize manifest factor coverage without upgrading evidence status.

    Returns:
        JSON-serializable checker payload with coverage counts and boundary violations.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("fidelity sweep manifest must be a mapping")
    axes = manifest.get("axes")
    if not isinstance(axes, list):
        raise ValueError("fidelity sweep manifest must contain axes list")

    payload_kind_counts: dict[str, int] = {}
    unresolved_runtime_bindings = 0
    total_variants = 0
    axis_summaries: list[dict[str, Any]] = []
    violations: list[str] = []

    for axis in axes:
        axis_summary = _check_manifest_axis(
            axis,
            payload_kind_counts=payload_kind_counts,
            violations=violations,
        )
        total_variants += int(axis_summary["variant_count"])
        unresolved_runtime_bindings += int(axis_summary["unresolved_runtime_binding_count"])
        axis_summaries.append(axis_summary["axis_summary"])

    _append_manifest_boundary_violations(manifest, violations)

    return {
        "schema_version": FIDELITY_SWEEP_MANIFEST_CHECK_SCHEMA,
        "status": "manifest_check_only",
        "evidence_status": "not_benchmark_evidence",
        "claim_boundary": (
            "checker summary only: reviews dry-run fidelity factor coverage and "
            "no-evidence boundaries; does not run sensitivity studies or establish "
            "simulator-dependence conclusions."
        ),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_status": manifest.get("status"),
        "axis_count": len(axes),
        "variant_count": total_variants,
        "payload_kind_counts": dict(sorted(payload_kind_counts.items())),
        "unresolved_runtime_binding_count": unresolved_runtime_bindings,
        "axis_summaries": axis_summaries,
        "violations": violations,
        "passes": not violations,
    }


def write_fidelity_sweep_manifest_check(
    check_summary: Mapping[str, Any], output_dir: str | Path
) -> Path:
    """Write deterministic JSON fidelity sweep manifest checker summary.

    Returns:
        Path written JSON checker summary.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    check_path = out / "fidelity_sweep_manifest_check.json"
    check_path.write_text(
        json.dumps(check_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return check_path


def _check_manifest_axis(
    axis: Any,
    *,
    payload_kind_counts: dict[str, int],
    violations: list[str],
) -> dict[str, Any]:
    """Check one axis and update shared coverage counters.

    Returns:
        Axis summary plus per-axis counts used by the top-level checker.
    """
    if not isinstance(axis, dict):
        raise ValueError("fidelity sweep manifest axis must be mapping")
    variants = axis.get("variants")
    if not isinstance(variants, list) or not variants:
        violations.append(f"axis {axis.get('key')!r} has no variants")
        variants = []

    axis_payload_kinds: dict[str, int] = {}
    baseline_variants = 0
    unresolved_runtime_bindings = 0
    actual_baseline_key = None
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("fidelity sweep manifest variant must be mapping")
        if variant.get("baseline", False):
            baseline_variants += 1
            actual_baseline_key = variant.get("key")
        payload_kind_value = variant.get("payload_kind")
        payload_kind = str(payload_kind_value) if payload_kind_value is not None else "missing"
        payload_kind_counts[payload_kind] = payload_kind_counts.get(payload_kind, 0) + 1
        axis_payload_kinds[payload_kind] = axis_payload_kinds.get(payload_kind, 0) + 1
        if variant.get("runtime_binding_status") == UNRESOLVED_RUNTIME_BINDING:
            unresolved_runtime_bindings += 1
        else:
            violations.append(
                f"axis {axis.get('key')!r} variant {variant.get('key')!r} "
                "does not preserve unresolved runtime binding status"
            )

    if baseline_variants != 1:
        violations.append(f"axis {axis.get('key')!r} must have exactly one baseline variant")
    elif axis.get("baseline_variant") != actual_baseline_key:
        violations.append(
            f"axis {axis.get('key')!r} baseline_variant {axis.get('baseline_variant')!r} "
            f"does not match baseline variant key {actual_baseline_key!r}"
        )
    return {
        "variant_count": len(variants),
        "unresolved_runtime_binding_count": unresolved_runtime_bindings,
        "axis_summary": {
            "key": str(axis.get("key", "")),
            "variant_count": len(variants),
            "baseline_variant": axis.get("baseline_variant"),
            "payload_kind_counts": dict(sorted(axis_payload_kinds.items())),
        },
    }


def _append_manifest_boundary_violations(
    manifest: Mapping[str, Any], violations: list[str]
) -> None:
    """Append dry-run and no-evidence boundary violations in-place."""
    if manifest.get("dry_run") is not True:
        violations.append("manifest must remain dry_run=true")
    if manifest.get("status") != "manifest_dry_run_only":
        violations.append("manifest status must remain manifest_dry_run_only")
    if manifest.get("evidence_status") != "not_benchmark_evidence":
        violations.append("manifest evidence_status must remain not_benchmark_evidence")
    claim_boundary = str(manifest.get("claim_boundary", ""))
    for required_phrase in (
        "not benchmark evidence",
        "not simulator-realism evidence",
        "not sim-to-real evidence",
        "not paper-facing evidence",
    ):
        if required_phrase not in claim_boundary:
            violations.append(f"claim_boundary must include {required_phrase!r}")


def _axis_manifest(axis: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one canonical axis while preserving source variant order.

    Returns:
        JSON-serializable axis manifest.
    """
    variants = [_variant_manifest(variant) for variant in axis["variants"]]
    baseline_variants = [variant["key"] for variant in variants if variant["baseline"]]
    if len(baseline_variants) != 1:
        raise ValueError(f"axis {axis.get('key')!r} must mark exactly one baseline variant")
    return {
        "key": str(axis["key"]),
        "rationale": str(axis.get("rationale", "")),
        "variant_count": len(variants),
        "baseline_variant": baseline_variants[0],
        "variants": variants,
    }


def _variant_manifest(variant: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve patch/noise payloads without claiming runtime executability.

    Returns:
        JSON-serializable variant manifest.
    """
    patch = copy.deepcopy(variant.get("patch")) if "patch" in variant else None
    observation_noise = (
        copy.deepcopy(variant.get("observation_noise")) if "observation_noise" in variant else None
    )
    return {
        "key": str(variant["key"]),
        "baseline": bool(variant.get("baseline", False)),
        "patch": patch,
        "observation_noise": observation_noise,
        "payload_kind": _payload_kind(patch=patch, observation_noise=observation_noise),
        "runtime_binding_status": UNRESOLVED_RUNTIME_BINDING,
        "runtime_binding_note": (
            "Payload copied from config for manifest review only; no runtime patch binding or "
            "benchmark execution was performed."
        ),
    }


def _payload_kind(*, patch: Any, observation_noise: Any) -> str:
    if patch is not None and observation_noise is not None:
        return "patch_and_observation_noise"
    if patch is not None:
        return "patch"
    if observation_noise is not None:
        return "observation_noise"
    return "none"


def _surface_relationship(config: Mapping[str, Any]) -> dict[str, Any]:
    fidelity_axes = config.get("fidelity_axes")
    secondary_axis_keys = (
        sorted(str(key) for key in fidelity_axes) if isinstance(fidelity_axes, dict) else []
    )
    return {
        "canonical_surface": "axes",
        "secondary_surface": "fidelity_axes" if isinstance(fidelity_axes, dict) else None,
        "manifest_source": "axes",
        "canonical_axis_keys": [str(axis["key"]) for axis in config["axes"]],
        "secondary_axis_keys": secondary_axis_keys,
        "relationship": SURFACE_RELATIONSHIP,
    }


__all__ = [
    "DRY_RUN_CLAIM_BOUNDARY",
    "FIDELITY_SWEEP_MANIFEST_CHECK_SCHEMA",
    "FIDELITY_SWEEP_MANIFEST_SCHEMA",
    "UNRESOLVED_RUNTIME_BINDING",
    "ManifestOptions",
    "build_fidelity_sweep_manifest",
    "check_fidelity_sweep_manifest",
    "write_fidelity_sweep_manifest",
    "write_fidelity_sweep_manifest_check",
]
