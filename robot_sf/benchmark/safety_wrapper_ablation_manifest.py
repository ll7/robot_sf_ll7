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
EXPECTED_WRAPPER_ARMS = (WRAPPER_OFF_ARM, WRAPPER_ON_ARM)
PAIRING_KEY_FIELDS = ("planner", "scenario_id", "seed")
REQUIRED_ROW_FIELDS = (
    "study_id",
    "planner",
    "wrapper_arm",
    "scenario_id",
    "seed",
    "software_commit",
    "event_ledger",
    "metric_values",
    "wrapper_intervention_rate",
)

REQUIRED_EVENT_LEDGER_SCHEMA = "EpisodeEventLedger.v1"
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
    _validate_result_contract(normalized.get("result_contract"))
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


def _validate_result_contract(result_contract: Any) -> None:
    """Require row-level provenance and pairing fields for later ablation checks."""
    if not isinstance(result_contract, Mapping):
        raise ValueError("result_contract must be mapping")
    required_outputs = result_contract.get("required_outputs")
    if not isinstance(required_outputs, Sequence) or isinstance(required_outputs, (str, bytes)):
        raise ValueError("result_contract.required_outputs must be list")
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in required_outputs]
    if missing:
        raise ValueError(f"result_contract.required_outputs missing fields: {missing}")
    pairing_fields = result_contract.get("pairing_key_fields")
    if tuple(pairing_fields or ()) != PAIRING_KEY_FIELDS:
        raise ValueError(f"result_contract.pairing_key_fields must be {list(PAIRING_KEY_FIELDS)!r}")
    expected_arms = result_contract.get("expected_wrapper_arms")
    if tuple(expected_arms or ()) != EXPECTED_WRAPPER_ARMS:
        raise ValueError(
            f"result_contract.expected_wrapper_arms must be {list(EXPECTED_WRAPPER_ARMS)!r}"
        )


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
        "row_contract": {
            "required_fields": list(REQUIRED_ROW_FIELDS),
            "pairing_key_fields": list(PAIRING_KEY_FIELDS),
            "expected_wrapper_arms": list(EXPECTED_WRAPPER_ARMS),
            "pairing_rule": (
                "Every (planner, scenario_id, seed) group must contain exactly one "
                "wrapper_off row and one wrapper_on row before comparison."
            ),
        },
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


def check_factorial_ablation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check emitted ablation rows are pairable before any with/without comparison.

    The checker is deliberately pure and fail-closed: it does not infer missing
    provenance, silently drop duplicate arms, or compare unpaired rows.

    Returns:
        Structured completeness report with missing fields, duplicate pair rows,
        and incomplete off/on pairs.
    """
    missing_required_fields: list[dict[str, Any]] = []
    invalid_provenance_fields: list[dict[str, Any]] = []
    duplicate_pair_rows: list[dict[str, Any]] = []
    unexpected_wrapper_arms: list[str] = []
    groups: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = {}

    for index, row in enumerate(rows):
        missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
        if missing:
            missing_required_fields.append({"row_index": index, "fields": missing})
        invalid = _row_provenance_errors(row)
        if invalid:
            invalid_provenance_fields.append({"row_index": index, "fields": invalid})
        arm = str(row.get("wrapper_arm", ""))
        if arm not in EXPECTED_WRAPPER_ARMS:
            unexpected_wrapper_arms.append(arm)
        key = tuple(row.get(field) for field in PAIRING_KEY_FIELDS)
        by_arm = groups.setdefault(key, {})
        by_arm.setdefault(arm, []).append(row)
        if len(by_arm[arm]) > 1:
            duplicate_pair_rows.append(
                {
                    "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
                    "wrapper_arm": arm,
                    "count": len(by_arm[arm]),
                }
            )

    incomplete_pairs = [
        {
            "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
            "wrapper_arms": sorted(by_arm),
        }
        for key, by_arm in groups.items()
        if set(by_arm) != set(EXPECTED_WRAPPER_ARMS)
        or any(len(rows_for_arm) != 1 for rows_for_arm in by_arm.values())
    ]
    pair_provenance_mismatches = _pair_provenance_mismatches(groups)
    complete = (
        len(rows) > 0
        and not missing_required_fields
        and not invalid_provenance_fields
        and not unexpected_wrapper_arms
        and not duplicate_pair_rows
        and not incomplete_pairs
        and not pair_provenance_mismatches
    )
    return {
        "complete": bool(complete),
        "row_count": len(rows),
        "pair_count": len(groups),
        "required_fields": list(REQUIRED_ROW_FIELDS),
        "pairing_key_fields": list(PAIRING_KEY_FIELDS),
        "expected_wrapper_arms": list(EXPECTED_WRAPPER_ARMS),
        "missing_required_fields": missing_required_fields,
        "invalid_provenance_fields": invalid_provenance_fields,
        "unexpected_wrapper_arms": sorted(set(unexpected_wrapper_arms)),
        "duplicate_pair_rows": duplicate_pair_rows,
        "incomplete_pairs": incomplete_pairs,
        "pair_provenance_mismatches": pair_provenance_mismatches,
    }


def _pair_provenance_mismatches(
    groups: Mapping[tuple[Any, ...], Mapping[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    """Return paired-row provenance fields that disagree within an off/on contrast."""
    mismatches: list[dict[str, Any]] = []
    for key, by_arm in groups.items():
        if set(by_arm) != set(EXPECTED_WRAPPER_ARMS) or any(
            len(rows_for_arm) != 1 for rows_for_arm in by_arm.values()
        ):
            continue
        off_row = by_arm[WRAPPER_OFF_ARM][0]
        on_row = by_arm[WRAPPER_ON_ARM][0]
        fields = [
            field
            for field in ("study_id", "software_commit")
            if off_row.get(field) != on_row.get(field)
        ]
        if fields:
            mismatches.append(
                {
                    "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
                    "fields": fields,
                }
            )
    return mismatches


def _row_provenance_errors(row: Mapping[str, Any]) -> list[str]:
    """Return row fields whose values are unusable for a paired ablation packet."""
    invalid: list[str] = []
    for field in ("study_id", "planner", "scenario_id", "software_commit"):
        value = row.get(field)
        if not isinstance(value, str) or not value.strip():
            invalid.append(field)

    seed = row.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        invalid.append("seed")

    event_ledger = row.get("event_ledger")
    if (
        not isinstance(event_ledger, Mapping)
        or event_ledger.get("schema_version") != REQUIRED_EVENT_LEDGER_SCHEMA
    ):
        invalid.append("event_ledger")

    metric_values = row.get("metric_values")
    if not isinstance(metric_values, Mapping) or not metric_values:
        invalid.append("metric_values")

    intervention_rate = row.get("wrapper_intervention_rate")
    if (
        not isinstance(intervention_rate, int | float)
        or isinstance(intervention_rate, bool)
        or intervention_rate < 0.0
        or intervention_rate > 1.0
    ):
        invalid.append("wrapper_intervention_rate")

    return invalid


def load_safety_wrapper_ablation_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load emitted ablation rows for the opt-in result checker.

    JSONL is the expected format for benchmark rows. A JSON list is accepted for
    compact fixtures and hand-authored decision packets.

    Returns:
        List of row mappings suitable for ``check_factorial_ablation_rows``.
    """
    row_path = Path(path)
    if row_path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            row_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if not isinstance(row, dict):
                raise ValueError(f"{row_path}:{line_number} must contain a JSON object row")
            rows.append(row)
        return rows

    payload = json.loads(row_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{row_path} must contain a JSON list or JSONL object rows")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"{row_path} must contain only JSON object rows")
    return list(payload)


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
    "check_factorial_ablation_rows",
    "load_safety_wrapper_ablation_config",
    "load_safety_wrapper_ablation_rows",
    "validate_safety_wrapper_ablation_config",
    "write_safety_wrapper_ablation_manifest",
]
