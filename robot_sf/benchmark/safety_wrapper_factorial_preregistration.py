"""Pre-registration harness for issue #3501 wrapper on/off factorial smoke rows.

The harness is deliberately CPU-only and no-submit: it validates a tracked
pre-registration config, checks the wrapper arms through the runtime validator,
and emits deterministic planned rows. It does not execute benchmark episodes.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.safety_wrapper_runtime import runtime_config_from_mapping

SCHEMA_VERSION = "robot_sf.issue_3501_safety_wrapper_factorial_preregistration.v1"
EXPECTED_WRAPPER_ARMS = ("wrapper_off", "wrapper_on")
PAIRING_KEY_FIELDS = ("study_id", "planner", "scenario_family", "scenario_id", "seed")


def load_factorial_preregistration_config(path: str | Path) -> dict[str, Any]:
    """Load and validate an issue #3501 factorial pre-registration config.

    Returns:
        Validated configuration payload.
    """

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"factorial pre-registration config must be mapping: {config_path}")
    return validate_factorial_preregistration_config(payload, config_path=config_path)


def validate_factorial_preregistration_config(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the no-submit wrapper on/off CPU-smoke factorial contract.

    Returns:
        Shallow-normalized configuration payload.
    """

    normalized = dict(config)
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if int(normalized.get("issue", 0)) != 3501:
        raise ValueError("issue must be 3501")
    if normalized.get("benchmark_evidence") is not False:
        raise ValueError("benchmark_evidence must be false for pre-registration")
    _reject_transient_routing_state(normalized)
    _validate_source_paths(normalized, config_path=config_path)
    _validate_scenario_family(normalized.get("scenario_family"), config_path=config_path)
    _validate_policy(normalized.get("policy"), config_path=config_path)
    _validate_wrapper_arms(normalized.get("wrapper_arms"))
    if tuple(normalized.get("pairing_key_fields") or ()) != PAIRING_KEY_FIELDS:
        raise ValueError(f"pairing_key_fields must be {list(PAIRING_KEY_FIELDS)!r}")
    return normalized


def build_preregistration_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic planned rows for the paired CPU-smoke factorial.

    Returns:
        Planned-row packet with pair-check summary.
    """

    validated = validate_factorial_preregistration_config(config)
    scenario_family = validated["scenario_family"]
    policy = copy.deepcopy(validated["policy"])
    policy_fingerprint = _stable_hash(policy)
    arms = _normalized_arms(validated["wrapper_arms"])
    rows: list[dict[str, Any]] = []

    for scenario_id in scenario_family["scenario_ids"]:
        for seed in scenario_family["seeds"]:
            for arm in arms:
                rows.append(
                    {
                        "study_id": str(validated["study_id"]),
                        "planner": str(policy["planner_key"]),
                        "policy": copy.deepcopy(policy),
                        "policy_fingerprint": policy_fingerprint,
                        "scenario_family": str(scenario_family["key"]),
                        "scenario_matrix": str(scenario_family["scenario_matrix"]),
                        "scenario_id": str(scenario_id),
                        "seed": int(seed),
                        "wrapper_arm": str(arm["key"]),
                        "baseline": bool(arm["baseline"]),
                        "safety_wrapper": copy.deepcopy(arm["safety_wrapper"]),
                        "safety_wrapper_runtime_config": copy.deepcopy(
                            arm["safety_wrapper_runtime_config"]
                        ),
                    }
                )

    pair_check = check_planned_rows(rows)
    return {
        "schema_version": "robot_sf.issue_3501_safety_wrapper_factorial_plan.v1",
        "source_schema_version": SCHEMA_VERSION,
        "issue": 3501,
        "study_id": str(validated["study_id"]),
        "status": "planned_rows_cpu_smoke_only",
        "benchmark_evidence": False,
        "claim_boundary": str(validated["claim_boundary"]),
        "pairing_key_fields": list(PAIRING_KEY_FIELDS),
        "row_count": len(rows),
        "pair_check": pair_check,
        "rows": rows,
    }


def check_planned_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check that each planned pair differs only by wrapper arm/config.

    Returns:
        Pair-completeness report.
    """

    grouped: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = {}
    invalid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = [field for field in (*PAIRING_KEY_FIELDS, "wrapper_arm") if field not in row]
        if missing:
            invalid_rows.append({"row_index": index, "fields": missing})
            continue
        wrapper_arm = str(row["wrapper_arm"])
        key = tuple(row[field] for field in PAIRING_KEY_FIELDS)
        grouped.setdefault(key, {}).setdefault(wrapper_arm, []).append(row)

    incomplete_pairs: list[dict[str, Any]] = []
    policy_mismatches: list[dict[str, Any]] = []
    for key, by_arm in grouped.items():
        if set(by_arm) != set(EXPECTED_WRAPPER_ARMS) or any(
            len(arm_rows) != 1 for arm_rows in by_arm.values()
        ):
            incomplete_pairs.append(
                {
                    "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
                    "wrapper_arms": sorted(by_arm),
                }
            )
            continue
        off_row = by_arm["wrapper_off"][0]
        on_row = by_arm["wrapper_on"][0]
        if off_row.get("policy_fingerprint") != on_row.get("policy_fingerprint"):
            policy_mismatches.append(
                {
                    "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
                    "field": "policy_fingerprint",
                }
            )

    complete = bool(rows) and not invalid_rows and not incomplete_pairs and not policy_mismatches
    return {
        "complete": complete,
        "row_count": len(rows),
        "pair_count": len(grouped),
        "expected_wrapper_arms": list(EXPECTED_WRAPPER_ARMS),
        "invalid_rows": invalid_rows,
        "incomplete_pairs": incomplete_pairs,
        "policy_mismatches": policy_mismatches,
    }


def write_preregistration_plan(plan: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write a deterministic planned-row JSON artifact.

    Returns:
        Path to the written JSON plan.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "issue_3501_safety_wrapper_factorial_preregistration_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate_source_paths(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None,
) -> None:
    source_contracts = config.get("source_contracts")
    if not isinstance(source_contracts, Mapping):
        raise ValueError("source_contracts must be mapping")
    for key in ("ablation_design", "runtime_validator", "planner_source"):
        value = source_contracts.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"source_contracts.{key} must be non-empty string")

    if config_path is None:
        return
    for key in ("ablation_design", "planner_source"):
        path_text = str(source_contracts[key]).split("::", maxsplit=1)[0]
        if not _resolve_existing_path(path_text, config_path=Path(config_path)).exists():
            raise ValueError(f"source_contracts.{key} path does not exist: {path_text}")


def _validate_scenario_family(value: Any, *, config_path: str | Path | None) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("scenario_family must be mapping")
    for key in ("key", "scenario_matrix"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"scenario_family.{key} must be non-empty string")
    scenario_ids = value.get("scenario_ids")
    if (
        not isinstance(scenario_ids, Sequence)
        or isinstance(scenario_ids, (str, bytes))
        or not scenario_ids
        or not all(isinstance(item, str) and item.strip() for item in scenario_ids)
    ):
        raise ValueError("scenario_family.scenario_ids must be non-empty string list")
    seeds = value.get("seeds")
    if (
        not isinstance(seeds, Sequence)
        or isinstance(seeds, (str, bytes))
        or not seeds
        or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("scenario_family.seeds must be non-empty integer list")
    if len(set(seeds)) != len(seeds):
        raise ValueError("scenario_family.seeds must be unique and paired across arms")
    if config_path is not None:
        scenario_path = _resolve_existing_path(
            str(value["scenario_matrix"]),
            config_path=Path(config_path),
        )
        if not scenario_path.exists():
            raise ValueError(
                f"scenario_family.scenario_matrix path does not exist: {scenario_path}"
            )


def _validate_policy(value: Any, *, config_path: str | Path | None) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("policy must be mapping")
    for key in ("planner_key", "planner_source", "policy_config"):
        if key not in value:
            raise ValueError(f"policy.{key} is required")
    if not isinstance(value["planner_key"], str) or not value["planner_key"].strip():
        raise ValueError("policy.planner_key must be non-empty string")
    if not isinstance(value["planner_source"], str) or not value["planner_source"].strip():
        raise ValueError("policy.planner_source must be non-empty string")
    if not isinstance(value["policy_config"], Mapping):
        raise ValueError("policy.policy_config must be mapping")
    if config_path is not None:
        planner_keys = _load_declared_planner_keys(
            str(value["planner_source"]),
            config_path=Path(config_path),
        )
        if str(value["planner_key"]) not in planner_keys:
            raise ValueError(
                f"policy.planner_key {value['planner_key']!r} not declared by planner_source"
            )


def _validate_wrapper_arms(value: Any) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("wrapper_arms must be list")
    if not all(isinstance(arm, Mapping) for arm in value):
        raise ValueError("each wrapper_arms entry must be mapping")
    by_key = {str(arm.get("key")): arm for arm in value}
    if tuple(by_key) != EXPECTED_WRAPPER_ARMS:
        raise ValueError(f"wrapper_arms must be ordered {list(EXPECTED_WRAPPER_ARMS)!r}")
    baselines = [key for key, arm in by_key.items() if bool(arm.get("baseline", False))]
    if baselines != ["wrapper_off"]:
        raise ValueError("wrapper_off must be the only baseline arm")
    for key, arm in by_key.items():
        wrapper = arm.get("safety_wrapper")
        if not isinstance(wrapper, Mapping):
            raise ValueError(f"{key}.safety_wrapper must be mapping")
        runtime = runtime_config_from_mapping(wrapper)
        if asdict(runtime)["arm_key"] != key:
            raise ValueError(f"{key}.safety_wrapper.arm_key must match wrapper arm key")


def _normalized_arms(arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for arm in arms:
        runtime = runtime_config_from_mapping(arm["safety_wrapper"])
        normalized.append(
            {
                "key": str(arm["key"]),
                "baseline": bool(arm.get("baseline", False)),
                "safety_wrapper": copy.deepcopy(dict(arm["safety_wrapper"])),
                "safety_wrapper_runtime_config": asdict(runtime),
            }
        )
    return normalized


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resolve_existing_path(path_text: str, *, config_path: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    for parent in config_path.resolve().parents:
        parent_candidate = parent / candidate
        if parent_candidate.exists():
            return parent_candidate
    return cwd_candidate


def _load_declared_planner_keys(path_text: str, *, config_path: Path) -> set[str]:
    source_path = _resolve_existing_path(
        path_text.split("::", maxsplit=1)[0], config_path=config_path
    )
    if not source_path.exists():
        raise ValueError(f"policy.planner_source path does not exist: {source_path}")
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"policy.planner_source must be mapping: {source_path}")
    planners = payload.get("planners")
    if not isinstance(planners, Sequence) or isinstance(planners, (str, bytes)):
        raise ValueError(f"policy.planner_source missing planners list: {source_path}")
    return {
        str(planner["key"])
        for planner in planners
        if isinstance(planner, Mapping) and isinstance(planner.get("key"), str)
    }


def _reject_transient_routing_state(config: Mapping[str, Any]) -> None:
    forbidden_keys = {"target_host", "packet_lineage", "queue_route", "submit_host"}
    found = sorted(forbidden_keys & set(config))
    if found:
        raise ValueError(f"transient queue-routing state is forbidden in tracked config: {found}")


__all__ = [
    "EXPECTED_WRAPPER_ARMS",
    "PAIRING_KEY_FIELDS",
    "SCHEMA_VERSION",
    "build_preregistration_plan",
    "check_planned_rows",
    "load_factorial_preregistration_config",
    "validate_factorial_preregistration_config",
    "write_preregistration_plan",
]
