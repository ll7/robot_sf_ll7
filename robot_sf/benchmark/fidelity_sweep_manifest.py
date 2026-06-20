"""Dry-run fidelity sweep manifest builder for issue #3207.

This module only enumerates the planned fidelity sensitivity matrix. It does
not bind config patches to runtime objects, run benchmark episodes, or promote
benchmark evidence.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.fidelity_sensitivity import validate_fidelity_sensitivity_config

if TYPE_CHECKING:
    from collections.abc import Mapping

FIDELITY_SWEEP_MANIFEST_SCHEMA = "fidelity-sweep-manifest.v1"
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
    "FIDELITY_SWEEP_MANIFEST_SCHEMA",
    "UNRESOLVED_RUNTIME_BINDING",
    "ManifestOptions",
    "build_fidelity_sweep_manifest",
    "write_fidelity_sweep_manifest",
]
