"""Dry-run factorial-ablation manifest + checker for the safety wrapper (issue #3501).

The merged planner-agnostic safety wrapper (``robot_sf.robot.safety_wrapper``, landed in
#3591) is the mitigation lever. The strongest available thesis result is causal: *the
framework identifies a mitigation lever and quantifies its effect*. Quantifying that effect
requires a ``planner x {wrapper off, wrapper on}`` factorial run, with paired seeds applied
identically to both arms, so the only difference between the two arms is the wrapper.

This module is the **opt-in design/manifest slice** of that work: it enumerates the factorial
cells and **checks** that the configured ablation is well formed (exactly two wrapper arms
{off, on}, one baseline, predeclared wrapper thresholds that match the merged
``SafetyWrapperConfig`` contract, every planner present in both arms, paired seeds shared
identically across arms). It binds no runtime objects, runs no benchmark episodes, tunes no
thresholds, and makes no mitigation-effectiveness claim. The ablation campaign and live wiring
are deliberate downstream follow-ups.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.robot.safety_wrapper import SAFETY_WRAPPER_SCHEMA, SafetyWrapperConfig

SAFETY_WRAPPER_ABLATION_SCHEMA = "safety-wrapper-ablation-manifest.v1"
CONFIG_SCHEMA_VERSION = "safety-wrapper-ablation.v1"

WRAPPER_OFF_ARM = "wrapper_off"
WRAPPER_ON_ARM = "wrapper_on"

DRY_RUN_CLAIM_BOUNDARY = (
    "dry-run factorial-ablation manifest only: enumerates the fixed issue #3501 "
    "planner x {wrapper off, wrapper on} cells and paired seeds; not benchmark evidence, "
    "not a mitigation-effectiveness result, and not paper-facing evidence."
)


@dataclass(frozen=True)
class ManifestOptions:
    """Stable metadata for a dry-run manifest build."""

    config_path: str
    git_head: str = "unknown"
    dry_run: bool = True


def load_safety_wrapper_ablation_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a safety-wrapper ablation YAML config.

    Returns:
        Validated config mapping.
    """
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"safety-wrapper ablation config must be a mapping: {config_path}")
    return validate_safety_wrapper_ablation_config(payload)


def validate_safety_wrapper_ablation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the issue #3501 factorial-ablation design contract.

    Checks the wrapper on/off factorization and provenance the manifest depends on:
    exactly two arms keyed ``wrapper_off``/``wrapper_on``, exactly one baseline (the off
    arm), the off arm disabled and the on arm enabled, and the on-arm thresholds usable as
    a real ``SafetyWrapperConfig`` (predeclared, no per-planner tuning).

    Returns:
        Shallow-normalized dictionary with the original config values.
    """
    normalized = dict(config)
    if normalized.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    _validate_fixed_scope(normalized.get("fixed_scope"))
    _validate_wrapper_arms(normalized.get("wrapper_arms"))
    return normalized


def _validate_fixed_scope(fixed_scope: Any) -> None:
    """Require a scenario set, a non-empty paired-seed list, and >=1 planner group."""
    if not isinstance(fixed_scope, Mapping):
        raise ValueError("fixed_scope must be a mapping")
    if not fixed_scope.get("scenario_set"):
        raise ValueError("fixed_scope.scenario_set is required")
    seeds = fixed_scope.get("seeds")
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)) or len(seeds) == 0:
        raise ValueError("fixed_scope.seeds must be a non-empty list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("fixed_scope.seeds must be unique (paired seeds applied to both arms)")
    planner_groups = fixed_scope.get("planner_groups")
    if (
        not isinstance(planner_groups, Sequence)
        or isinstance(planner_groups, (str, bytes))
        or len(planner_groups) == 0
    ):
        raise ValueError("fixed_scope.planner_groups must be a non-empty list")
    if len(set(planner_groups)) != len(planner_groups):
        raise ValueError("fixed_scope.planner_groups must be unique")


def _validate_wrapper_arms(arms: Any) -> None:
    """Require exactly the {off, on} factorial arms with one baseline and valid thresholds."""
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
        raise ValueError("wrapper_arms must be a list")
    if not all(isinstance(arm, Mapping) for arm in arms):
        raise ValueError("each entry in wrapper_arms must be a mapping")
    keys = [str(arm.get("key")) for arm in arms if isinstance(arm, Mapping)]
    if keys != [WRAPPER_OFF_ARM, WRAPPER_ON_ARM] and sorted(keys) != sorted(
        [WRAPPER_OFF_ARM, WRAPPER_ON_ARM]
    ):
        raise ValueError(
            f"wrapper_arms must be exactly [{WRAPPER_OFF_ARM!r}, {WRAPPER_ON_ARM!r}], got {keys!r}"
        )
    by_key = {str(arm["key"]): arm for arm in arms}
    baselines = [key for key, arm in by_key.items() if bool(arm.get("baseline", False))]
    if baselines != [WRAPPER_OFF_ARM]:
        raise ValueError(
            f"exactly one baseline arm is required and it must be {WRAPPER_OFF_ARM!r}, "
            f"got {baselines!r}"
        )
    if bool(by_key[WRAPPER_OFF_ARM].get("enabled", False)):
        raise ValueError(f"{WRAPPER_OFF_ARM!r} arm must have enabled: false")
    if not bool(by_key[WRAPPER_ON_ARM].get("enabled", False)):
        raise ValueError(f"{WRAPPER_ON_ARM!r} arm must have enabled: true")
    # The on-arm thresholds must construct a real SafetyWrapperConfig: this ties the manifest
    # to the merged wrapper contract and rejects unusable (non-positive) thresholds.
    _wrapper_config_from_arm(by_key[WRAPPER_ON_ARM])


def _wrapper_config_from_arm(arm: Mapping[str, Any]) -> SafetyWrapperConfig:
    """Build the merged ``SafetyWrapperConfig`` for an arm (validates thresholds).

    Returns:
        The constructed, threshold-validated wrapper config for the arm.
    """
    raw = arm.get("config") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("wrapper_on arm config must be a mapping")
    allowed = {
        "pedestrian_caution_radius_m",
        "capped_speed_m_s",
        "ttc_veto_threshold_s",
        "clearance_veto_m",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"wrapper_on arm config has unknown threshold keys: {sorted(unknown)}")
    return SafetyWrapperConfig(enabled=True, **{key: float(raw[key]) for key in raw})


def build_safety_wrapper_ablation_manifest(
    config: Mapping[str, Any],
    *,
    options: ManifestOptions,
) -> dict[str, Any]:
    """Build a deterministic dry-run factorial-ablation manifest.

    Enumerates one cell per ``(planner, wrapper_arm)`` pair, each carrying the shared paired
    seeds, and records provenance fields that tie the design back to the merged wrapper.

    Returns:
        JSON-serializable manifest payload.
    """
    if not options.dry_run:
        raise ValueError("safety-wrapper ablation manifest builder only supports dry-run manifests")

    validated = validate_safety_wrapper_ablation_config(config)
    fixed_scope = validated["fixed_scope"]
    seeds = list(fixed_scope["seeds"])
    planner_groups = list(fixed_scope["planner_groups"])
    arms = [_arm_manifest(arm) for arm in validated["wrapper_arms"]]
    cells = _enumerate_cells(planner_groups, arms, seeds)
    return {
        "schema_version": SAFETY_WRAPPER_ABLATION_SCHEMA,
        "safety_wrapper_schema": SAFETY_WRAPPER_SCHEMA,
        "issue": int(validated.get("issue", 3501)),
        "study_id": str(validated["study_id"]),
        "status": "manifest_dry_run_only",
        "dry_run": True,
        "evidence_status": "not_benchmark_evidence",
        "claim_boundary": DRY_RUN_CLAIM_BOUNDARY,
        "config_path": options.config_path,
        "git_head": options.git_head,
        "fixed_scope": copy.deepcopy(fixed_scope),
        "seeds": seeds,
        "planner_groups": planner_groups,
        "wrapper_arms": arms,
        "primary_outcomes": copy.deepcopy(validated.get("primary_outcomes", [])),
        "event_ledger_target": validated.get("event_ledger_target"),
        "result_contract": copy.deepcopy(validated.get("result_contract")),
        "cell_count": len(cells),
        "cells": cells,
        "factorial_check": check_factorial_ablation(planner_groups, arms, seeds),
    }


def _arm_manifest(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one wrapper arm and echo its predeclared thresholds.

    Returns:
        JSON-serializable arm manifest.
    """
    enabled = bool(arm.get("enabled", False))
    wrapper_config: dict[str, Any] | None = None
    if enabled:
        cfg = _wrapper_config_from_arm(arm)
        wrapper_config = {
            "pedestrian_caution_radius_m": cfg.pedestrian_caution_radius_m,
            "capped_speed_m_s": cfg.capped_speed_m_s,
            "ttc_veto_threshold_s": cfg.ttc_veto_threshold_s,
            "clearance_veto_m": cfg.clearance_veto_m,
        }
    return {
        "key": str(arm["key"]),
        "baseline": bool(arm.get("baseline", False)),
        "enabled": enabled,
        "wrapper_config": wrapper_config,
        "thresholds_source": "predeclared_fixed_no_per_planner_tuning",
        "runtime_binding_status": "unresolved_runtime_binding",
        "runtime_binding_note": (
            "Arm config copied for manifest review only; no runtime wrapper binding or "
            "benchmark execution was performed."
        ),
    }


def _enumerate_cells(
    planner_groups: Sequence[str],
    arms: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    """Enumerate one factorial cell per ``(planner, arm)`` with the shared paired seeds.

    Returns:
        Deterministic list of cell descriptors ordered planner-major, arm-minor.
    """
    cells: list[dict[str, Any]] = []
    for planner in planner_groups:
        for arm in arms:
            cells.append(
                {
                    "planner": str(planner),
                    "wrapper_arm": str(arm["key"]),
                    "wrapper_enabled": bool(arm["enabled"]),
                    "baseline": bool(arm["baseline"]),
                    "seeds": list(seeds),
                }
            )
    return cells


def check_factorial_ablation(
    planner_groups: Sequence[str],
    arms: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Check wrapper on/off factorization and paired-seed completeness.

    Verifies every planner appears in both the off and on arm, that the two arms are the
    expected {off, on} pair, and that the paired seeds are shared identically across arms.

    Returns:
        Structured check report with ``complete`` plus per-condition booleans and counts.
    """
    arm_keys = sorted(str(arm["key"]) for arm in arms)
    arms_are_off_on = arm_keys == sorted([WRAPPER_OFF_ARM, WRAPPER_ON_ARM])
    enabled_by_key = {str(arm["key"]): bool(arm["enabled"]) for arm in arms}
    off_on_enabled = (
        enabled_by_key.get(WRAPPER_OFF_ARM) is False and enabled_by_key.get(WRAPPER_ON_ARM) is True
    )
    seeds_paired = len(seeds) > 0 and len(set(seeds)) == len(seeds)
    expected_cells = len(planner_groups) * 2
    complete = (
        arms_are_off_on
        and off_on_enabled
        and seeds_paired
        and len(planner_groups) > 0
        and len(set(planner_groups)) == len(planner_groups)
    )
    return {
        "complete": bool(complete),
        "arms_are_off_on": bool(arms_are_off_on),
        "off_on_enabled": bool(off_on_enabled),
        "seeds_paired_across_arms": bool(seeds_paired),
        "planner_count": len(planner_groups),
        "expected_cell_count": expected_cells,
        "seeds_per_cell": len(seeds),
    }


def write_safety_wrapper_ablation_manifest(
    manifest: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write a deterministic JSON dry-run manifest.

    Returns:
        Path to the written JSON manifest.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "safety_wrapper_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DRY_RUN_CLAIM_BOUNDARY",
    "SAFETY_WRAPPER_ABLATION_SCHEMA",
    "WRAPPER_OFF_ARM",
    "WRAPPER_ON_ARM",
    "ManifestOptions",
    "build_safety_wrapper_ablation_manifest",
    "check_factorial_ablation",
    "load_safety_wrapper_ablation_config",
    "validate_safety_wrapper_ablation_config",
    "write_safety_wrapper_ablation_manifest",
]
