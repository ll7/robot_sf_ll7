"""Pre-run paired topology-gate preregistration checks for issue #3465."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "robot_sf.issue_3465_topology_gate_paired_preregistration.v1"
PLAN_SCHEMA_VERSION = "robot_sf.issue_3465_topology_gate_paired_readiness.v1"
EXPECTED_ARMS = ("topology_gate_disabled", "topology_gate_enabled")
GATE_FLAG = "near_parity_diversity_gate_enabled"
PAIRING_KEY_FIELDS = ("study_id", "planner", "scenario_id", "seed", "horizon")
TRANSIENT_QUEUE_KEYS = frozenset(
    {
        "target_host",
        "queue_target",
        "packet_lineage",
        "packet_lineage_pointer",
        "target_hosts",
        "submit_host",
    }
)


def load_topology_gate_paired_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the issue #3465 paired topology-gate preregistration YAML.

    Returns:
        Validated config mapping.
    """

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(
            f"topology gate paired preregistration config must be mapping: {config_path}"
        )
    return validate_topology_gate_paired_config(payload, config_path=config_path)


def validate_topology_gate_paired_config(
    config: Mapping[str, Any], *, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate the no-submit paired enabled/disabled config contract.

    Returns:
        Shallow-normalized config mapping.
    """

    normalized = copy.deepcopy(dict(config))
    _reject_transient_queue_state(normalized)
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    if int(normalized.get("issue", -1)) != 3465:
        raise ValueError("issue must be 3465")
    if normalized.get("benchmark_evidence") is not False:
        raise ValueError("benchmark_evidence must be false for this pre-run preregistration")
    if not isinstance(normalized.get("study_id"), str) or not normalized["study_id"].strip():
        raise ValueError("study_id must be non-empty string")
    _validate_authorization(normalized.get("campaign_authorization"))
    _validate_source_contracts(normalized.get("source_contracts"), config_path=config_path)
    _validate_benchmark_contract(normalized.get("benchmark_contract"))
    _validate_readiness(normalized.get("readiness"))
    _validate_output_paths(normalized.get("output_paths"))

    arms = normalized.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError(
            "arms must contain exactly topology_gate_disabled and topology_gate_enabled"
        )
    arm_report = check_topology_gate_arms(arms)
    if not arm_report["complete"]:
        raise ValueError(f"topology gate arms invalid: {arm_report}")

    pinned_hash = str(normalized["readiness"].get("topology_gate_config_hash", "")).strip()
    actual_hash = topology_gate_config_hash(arms)
    if pinned_hash != actual_hash:
        raise ValueError(
            "readiness.topology_gate_config_hash does not match arms "
            f"(expected {actual_hash}, got {pinned_hash or '<empty>'})"
        )
    normalized["readiness"]["computed_topology_gate_config_hash"] = actual_hash
    return normalized


def topology_gate_config_hash(arms: Sequence[Mapping[str, Any]]) -> str:
    """Return stable hash for the paired arm policy configs.

    Returns:
        SHA-256 digest for canonical arm policy config payload.
    """

    canonical = json.dumps(_canonical_arm_payload(arms), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_topology_gate_readiness_packet(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic pre-run readiness packet without executing benchmark episodes.

    Returns:
        Readiness packet with fail-closed blockers and planned rows.
    """

    validated = validate_topology_gate_paired_config(config)
    arm_report = check_topology_gate_arms(validated["arms"])
    rows = _planned_rows(validated)
    row_report = check_planned_rows(rows)
    readiness = validated["readiness"]
    blockers: list[str] = []
    if not bool(readiness.get("corrective_complete")) and not bool(
        readiness.get("corrective_waiver_recorded")
    ):
        blockers.append("blocked_corrective_issue")
    if not bool(readiness.get("no_known_fallback_degraded_promotion")) or readiness.get(
        "known_ineligible_rows"
    ):
        blockers.append("blocked_ineligible_rows")
    if not arm_report["complete"] or not row_report["complete"]:
        blockers.append("blocked_pairing_contract")
    status = "ready_for_paired_run" if not blockers else blockers[0]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "source_schema_version": SCHEMA_VERSION,
        "issue": 3465,
        "study_id": validated["study_id"],
        "status": status,
        "blockers": blockers,
        "benchmark_evidence": False,
        "claim_boundary": validated["claim_boundary"],
        "corrective_issue": readiness["corrective_issue"],
        "topology_gate_config_hash": readiness["computed_topology_gate_config_hash"],
        "pairing_key_fields": list(PAIRING_KEY_FIELDS),
        "row_count": len(rows),
        "arm_check": arm_report,
        "row_check": row_report,
        "rows": rows,
        "output_paths": copy.deepcopy(validated["output_paths"]),
    }


def check_topology_gate_arms(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify enabled/disabled arms differ only by the near-parity gate flag.

    Returns:
        Arm completeness and mismatch report.
    """

    invalid_rows: list[dict[str, Any]] = []
    by_key: dict[str, Mapping[str, Any]] = {}
    for index, arm in enumerate(arms):
        missing = [field for field in ("key", "gate_enabled", "policy_config") if field not in arm]
        if missing:
            invalid_rows.append({"row_index": index, "fields": missing})
            continue
        by_key[str(arm["key"])] = arm
    incomplete_pairs: list[dict[str, Any]] = []
    if set(by_key) != set(EXPECTED_ARMS):
        incomplete_pairs.append({"topology_gate_arms": sorted(by_key)})

    field_mismatches: list[dict[str, Any]] = []
    flag_mismatches: list[dict[str, Any]] = []
    if not incomplete_pairs:
        disabled = by_key["topology_gate_disabled"]
        enabled = by_key["topology_gate_enabled"]
        if disabled.get("gate_enabled") is not False or enabled.get("gate_enabled") is not True:
            flag_mismatches.append({"field": "gate_enabled"})
        disabled_policy = disabled.get("policy_config")
        enabled_policy = enabled.get("policy_config")
        if not isinstance(disabled_policy, Mapping) or not isinstance(enabled_policy, Mapping):
            invalid_rows.append({"row_index": -1, "fields": ["policy_config"]})
        else:
            diff_paths = _diff_paths(disabled_policy, enabled_policy)
            allowed = {(GATE_FLAG,)}
            unexpected = sorted(".".join(path) for path in diff_paths if path not in allowed)
            if unexpected:
                field_mismatches.append(
                    {"allowed_difference": GATE_FLAG, "unexpected_fields": unexpected}
                )
            if (
                disabled_policy.get(GATE_FLAG) is not False
                or enabled_policy.get(GATE_FLAG) is not True
            ):
                flag_mismatches.append({"field": f"policy_config.{GATE_FLAG}"})

    complete = (
        not invalid_rows and not incomplete_pairs and not field_mismatches and not flag_mismatches
    )
    return {
        "complete": complete,
        "expected_arms": list(EXPECTED_ARMS),
        "allowed_difference": GATE_FLAG,
        "invalid_rows": invalid_rows,
        "incomplete_pairs": incomplete_pairs,
        "field_mismatches": field_mismatches,
        "flag_mismatches": flag_mismatches,
    }


def check_planned_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check each planner/scenario/seed/horizon pair has one enabled and one disabled row.

    Returns:
        Pair completeness report for planned rows.
    """

    grouped: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = {}
    invalid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        missing = [
            field for field in (*PAIRING_KEY_FIELDS, "topology_gate_arm") if field not in row
        ]
        if missing:
            invalid_rows.append({"row_index": index, "fields": missing})
            continue
        key = tuple(row[field] for field in PAIRING_KEY_FIELDS)
        grouped.setdefault(key, {}).setdefault(str(row["topology_gate_arm"]), []).append(row)

    incomplete_pairs: list[dict[str, Any]] = []
    for key, by_arm in grouped.items():
        if set(by_arm) != set(EXPECTED_ARMS) or any(
            len(arm_rows) != 1 for arm_rows in by_arm.values()
        ):
            incomplete_pairs.append(
                {
                    "pairing_key": dict(zip(PAIRING_KEY_FIELDS, key, strict=True)),
                    "topology_gate_arms": sorted(by_arm),
                }
            )
    return {
        "complete": bool(rows) and not invalid_rows and not incomplete_pairs,
        "row_count": len(rows),
        "pair_count": len(grouped),
        "expected_arms": list(EXPECTED_ARMS),
        "invalid_rows": invalid_rows,
        "incomplete_pairs": incomplete_pairs,
    }


def write_readiness_packet(packet: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Write deterministic issue #3465 readiness JSON packet.

    Returns:
        Path to the written JSON packet.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "issue_3465_topology_gate_paired_readiness.json"
    path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _planned_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    contract = config["benchmark_contract"]
    rows: list[dict[str, Any]] = []
    for planner in contract["planner_set"]:
        for scenario_id in contract["scenario_set"]:
            for seed in contract["seed_list"]:
                for arm in config["arms"]:
                    rows.append(
                        {
                            "study_id": str(config["study_id"]),
                            "planner": str(planner),
                            "scenario_id": str(scenario_id),
                            "seed": int(seed),
                            "horizon": int(contract["horizon"]),
                            "topology_gate_arm": str(arm["key"]),
                            "near_parity_diversity_gate_enabled": bool(arm["gate_enabled"]),
                            "native_adapter_fallback_degraded_required": "native_or_adapter_only",
                        }
                    )
    return rows


def _canonical_arm_payload(arms: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "key": str(arm.get("key")),
            "gate_enabled": bool(arm.get("gate_enabled")),
            "policy_config": copy.deepcopy(arm.get("policy_config")),
        }
        for arm in sorted(arms, key=lambda item: str(item.get("key")))
    ]


def _diff_paths(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[tuple[str, ...]]:
    keys = set(left) | set(right)
    paths: set[tuple[str, ...]] = set()
    for key in keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            for child in _diff_paths(left_value, right_value):
                paths.add((str(key), *child))
        elif left_value != right_value:
            paths.add((str(key),))
    return paths


def _reject_transient_queue_state(value: Any, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text in TRANSIENT_QUEUE_KEYS:
                dotted = ".".join((*path, key_text))
                raise ValueError(
                    f"transient queue-routing state is not allowed in tracked config: {dotted}"
                )
            _reject_transient_queue_state(child, path=(*path, key_text))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_transient_queue_state(child, path=(*path, str(index)))


def _validate_authorization(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("campaign_authorization must be mapping")
    for key in (
        "compute_submit_authorized",
        "no_slurm_submission_in_this_pr",
        "no_gpu_submission_in_this_pr",
    ):
        if key not in value:
            raise ValueError(f"campaign_authorization.{key} must be present")
    if value["compute_submit_authorized"] is not False:
        raise ValueError("campaign_authorization.compute_submit_authorized must be false")
    if value["no_slurm_submission_in_this_pr"] is not True:
        raise ValueError("campaign_authorization.no_slurm_submission_in_this_pr must be true")
    if value["no_gpu_submission_in_this_pr"] is not True:
        raise ValueError("campaign_authorization.no_gpu_submission_in_this_pr must be true")


def _validate_source_contracts(value: Any, *, config_path: str | Path | None) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("source_contracts must be mapping")
    for key in ("enabled_candidate", "disabled_candidate", "promotion_gate"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"source_contracts.{key} must be non-empty string")
        if config_path is not None:
            path_text = str(value[key]).split("::", maxsplit=1)[0]
            repo_root = _repo_root_from_config_path(Path(config_path))
            if not (repo_root / path_text).exists():
                raise ValueError(f"source_contracts.{key} path does not exist: {path_text}")


def _validate_benchmark_contract(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("benchmark_contract must be mapping")
    for key in ("planner_set", "scenario_set", "seed_list"):
        if not isinstance(value.get(key), list) or not value[key]:
            raise ValueError(f"benchmark_contract.{key} must be non-empty list")
    if not isinstance(value.get("horizon"), int) or int(value["horizon"]) <= 0:
        raise ValueError("benchmark_contract.horizon must be positive integer")


def _validate_readiness(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("readiness must be mapping")
    if int(value.get("corrective_issue", -1)) != 3463:
        raise ValueError("readiness.corrective_issue must be 3463")
    for key in (
        "corrective_complete",
        "corrective_waiver_recorded",
        "no_known_fallback_degraded_promotion",
    ):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"readiness.{key} must be boolean")
    if not isinstance(value.get("known_ineligible_rows"), list):
        raise ValueError("readiness.known_ineligible_rows must be list")


def _validate_output_paths(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("output_paths must be mapping")
    for key in (
        "planned_readiness_packet",
        "future_campaign_output_root",
        "future_evidence_packet",
    ):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"output_paths.{key} must be non-empty string")


def _repo_root_from_config_path(config_path: Path) -> Path:
    path = config_path.resolve()
    if (
        len(path.parents) >= 3
        and path.parent.name == "benchmarks"
        and path.parent.parent.name == "configs"
    ):
        return path.parents[2]
    return Path.cwd()
