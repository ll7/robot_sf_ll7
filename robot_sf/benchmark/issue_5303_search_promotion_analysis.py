"""Fail-closed analysis for the issue #5303 diagnostic search-stage handoff.

This module reads only the per-attempt JSONL emitted by the frozen diagnostic
command.  It never runs planners and it never promotes a result: three search
seeds per arm cannot satisfy the frozen two-sided null-test gate, so a structurally
complete diagnostic packet is still classified ``inconclusive``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    DEFAULT_MANIFEST_PATH,
    preflight_issue_5303_contract,
)

SCHEMA_VERSION = "issue_5303_search_promotion_analysis.v1"
OUTCOME_ROW_SCHEMA_VERSION = "issue_5303_search_promotion_outcome_row.v1"
DEFAULT_CONTRACT_PATH = Path("configs/adversarial/issue_5303_search_promotion_contract.yaml")


@dataclass(frozen=True)
class Issue5303AnalysisResult:
    """Structured result for the frozen diagnostic-only accounting analysis."""

    ready: bool
    decision: str
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    accounting: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible, stable analysis payload."""
        return {
            "schema_version": SCHEMA_VERSION,
            "ready": self.ready,
            "decision": self.decision,
            "promotion_eligible": False,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "accounting": self.accounting,
            "null_tests": {
                "status": "not_run_diagnostic_only",
                "reason": (
                    "The frozen three-seed design cannot satisfy the two-sided p<=0.05 gate; "
                    "this command is an accounting diagnostic, not a promotion analysis."
                ),
            },
        }


def _canonical_sha256(payload: dict[str, Any]) -> str:
    """Return the digest used by the frozen immutable-record field."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolve(repo_root: Path, raw_path: Path) -> Path:
    """Resolve one possibly repository-relative path.

    Returns:
        Absolute or repository-root-relative resolved path.
    """
    return raw_path if raw_path.is_absolute() else repo_root / raw_path


def _load_jsonl(path: Path, blockers: list[str]) -> list[dict[str, Any]]:
    """Load JSONL rows while retaining a precise parse blocker for every bad line.

    Returns:
        Parsed mapping rows; malformed lines are recorded in ``blockers``.
    """
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        blockers.append(f"outcome rows could not be read: {exc}")
        return rows
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            blockers.append(f"outcome rows contains a blank line at {line_number}")
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            blockers.append(f"outcome rows line {line_number} is not JSON: {exc.msg}")
            continue
        if not isinstance(payload, dict):
            blockers.append(f"outcome rows line {line_number} must be an object")
            continue
        rows.append(payload)
    return rows


def _required_contract_fields(contract: dict[str, Any], blockers: list[str]) -> set[str]:
    """Extract the frozen row schema fields or fail closed when it is malformed.

    Returns:
        Required field names, or an empty set after a malformed-schema blocker.
    """
    schema = contract.get("outcome_row_schema")
    if not isinstance(schema, dict):
        blockers.append("contract outcome_row_schema must be a mapping")
        return set()
    if schema.get("schema_version") != OUTCOME_ROW_SCHEMA_VERSION:
        blockers.append(
            f"contract outcome_row_schema.schema_version must be {OUTCOME_ROW_SCHEMA_VERSION!r}"
        )
    fields = schema.get("required_fields")
    if not isinstance(fields, list) or not all(isinstance(item, str) and item for item in fields):
        blockers.append(
            "contract outcome_row_schema.required_fields must be a non-empty string list"
        )
        return set()
    if len(fields) != len(set(fields)):
        blockers.append("contract outcome_row_schema.required_fields must not contain duplicates")
    return set(fields)


def _frozen_row_bindings(contract: dict[str, Any], blockers: list[str]) -> dict[str, str]:
    """Extract the frozen provenance values that every diagnostic row must carry.

    The immutable row digest only proves that a row was not changed after it was
    written.  These bindings connect that self-hashed row to the preregistered
    scenario, search space, planners, objective, and execution mode.

    Returns:
        Mapping from outcome-row field names to their required frozen values.
    """
    provenance = contract.get("input_provenance")
    entries = provenance.get("required_inputs") if isinstance(provenance, dict) else None
    indexed_inputs = (
        {
            entry.get("id"): entry
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }
        if isinstance(entries, list)
        else {}
    )
    bindings: dict[str, str] = {}
    input_sources = {
        "scenario_config_path": ("diagnostic_scenario_template", "path"),
        "scenario_config_sha256": ("diagnostic_scenario_template", "sha256"),
        "search_space_path": ("search_space", "path"),
        "search_space_sha256": ("search_space", "sha256"),
        "target_planner_config_path": ("target_planner_config", "path"),
        "target_planner_config_sha256": ("target_planner_config", "sha256"),
        "neutral_reference_planner_config_path": ("neutral_reference_planner_config", "path"),
        "neutral_reference_planner_config_sha256": (
            "neutral_reference_planner_config",
            "sha256",
        ),
    }
    for row_field, (input_id, input_field) in input_sources.items():
        entry = indexed_inputs.get(input_id)
        value = entry.get(input_field) if isinstance(entry, dict) else None
        if not isinstance(value, str) or not value:
            blockers.append(
                f"contract input_provenance {input_id!r} must provide non-empty {input_field!r}"
            )
            continue
        bindings[row_field] = value

    family_split = contract.get("family_split")
    scenario_family = (
        family_split.get("fresh_outcome_family") if isinstance(family_split, dict) else None
    )
    if not isinstance(scenario_family, str) or not scenario_family:
        blockers.append("contract family_split.fresh_outcome_family must be a non-empty string")
    else:
        bindings["scenario_family"] = scenario_family

    step3_execution = contract.get("step3_execution")
    diagnostic_objective = (
        step3_execution.get("diagnostic_objective") if isinstance(step3_execution, dict) else None
    )
    if not isinstance(diagnostic_objective, str) or not diagnostic_objective:
        blockers.append("contract step3_execution.diagnostic_objective must be a non-empty string")
    else:
        bindings["objective"] = diagnostic_objective
    required_execution_mode = (
        step3_execution.get("required_execution_mode")
        if isinstance(step3_execution, dict)
        else None
    )
    if not isinstance(required_execution_mode, str) or not required_execution_mode:
        blockers.append(
            "contract step3_execution.required_execution_mode must be a non-empty string"
        )
    else:
        bindings["execution_mode"] = required_execution_mode
    return bindings


def analyze_issue_5303_search_promotion(  # noqa: C901, PLR0912, PLR0915
    outcomes_path: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    repo_root: Path | None = None,
) -> Issue5303AnalysisResult:
    """Validate full intention-to-search accounting and return a fixed inconclusive decision.

    All scheduled attempts remain in each arm's primary denominator, including duplicate,
    invalid, missing, or failed candidates.  Normalized hashes are deduplicated globally
    *within an arm* only for the unique-candidate endpoint; they never delete attempts.

    Returns:
        Fail-closed accounting result with a fixed ``inconclusive`` decision.
    """
    root = (repo_root or Path.cwd()).resolve()
    contract_path = _resolve(root, contract_path)
    manifest_path = _resolve(root, manifest_path)
    outcomes_path = _resolve(root, outcomes_path)
    blockers: list[str] = []
    warnings: list[str] = []

    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return Issue5303AnalysisResult(
            ready=False,
            decision="inconclusive",
            blockers=(f"contract could not be loaded: {exc}",),
            warnings=(),
        )
    if not isinstance(contract, dict):
        return Issue5303AnalysisResult(
            ready=False,
            decision="inconclusive",
            blockers=("contract must be a mapping",),
            warnings=(),
        )

    frozen_preflight = preflight_issue_5303_contract(
        contract_path=contract_path,
        manifest_path=manifest_path,
        repo_root=root,
    )
    if not frozen_preflight.ready:
        blockers.extend(
            f"frozen contract preflight failed: {blocker}" for blocker in frozen_preflight.blockers
        )

    required_fields = _required_contract_fields(contract, blockers)
    frozen_row_bindings = _frozen_row_bindings(contract, blockers)
    methods_block = contract.get("methods") if isinstance(contract.get("methods"), dict) else {}
    methods = tuple(
        str(entry.get("name"))
        for entry in methods_block.get("entries", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    )
    budget = contract.get("budget") if isinstance(contract.get("budget"), dict) else {}
    candidate_budget = budget.get("candidate_budget_per_search_seed_per_method")
    seeds_raw = budget.get("search_seeds")
    seeds = (
        tuple(seed for seed in seeds_raw if isinstance(seed, int))
        if isinstance(seeds_raw, list)
        else ()
    )
    if methods != ("optuna", "random"):
        blockers.append("contract methods must be exactly optuna and random")
    if not isinstance(candidate_budget, int) or candidate_budget < 1:
        blockers.append("contract candidate budget must be a positive integer")
        candidate_budget = 0
    if len(seeds) != 3 or len(set(seeds)) != 3:
        blockers.append("contract search seeds must be exactly three unique integers")

    rows = _load_jsonl(outcomes_path, blockers)
    observed_attempts: dict[str, set[tuple[int, int]]] = {method: set() for method in methods}
    duplicates_by_arm: dict[str, int] = dict.fromkeys(methods, 0)
    normalized_hashes: dict[str, set[str]] = {method: set() for method in methods}
    attrition_by_arm: dict[str, int] = dict.fromkeys(methods, 0)
    invalid_by_arm: dict[str, int] = dict.fromkeys(methods, 0)
    required_field_failures = 0
    immutable_hash_failures = 0
    frozen_binding_failures: dict[str, int] = dict.fromkeys(frozen_row_bindings, 0)

    for row_number, row in enumerate(rows, start=1):
        missing = sorted(required_fields - set(row))
        if missing:
            required_field_failures += 1
            blockers.append(f"row {row_number} is missing required fields: {missing}")
            continue
        if row.get("schema_version") != OUTCOME_ROW_SCHEMA_VERSION:
            blockers.append(f"row {row_number} has an unsupported schema_version")
        if row.get("execution_stage") != "search":
            blockers.append(f"row {row_number} must be a search-stage attempt")
        arm = row.get("arm")
        search_seed = row.get("search_seed")
        candidate_index = row.get("candidate_index")
        if (
            not isinstance(arm, str)
            or arm not in observed_attempts
            or isinstance(search_seed, bool)
            or not isinstance(search_seed, int)
            or isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
        ):
            blockers.append(
                f"row {row_number} has an invalid arm/search_seed/candidate_index identity"
            )
            continue
        if search_seed not in seeds or not 0 <= candidate_index < candidate_budget:
            blockers.append(f"row {row_number} is outside the frozen arm/seed/budget matrix")
            continue
        if row.get("method") != arm:
            blockers.append(f"row {row_number} method must match its arm")
        attempt_key = (search_seed, candidate_index)
        if attempt_key in observed_attempts[arm]:
            blockers.append(f"row {row_number} duplicates scheduled attempt {arm}:{attempt_key}")
            continue
        observed_attempts[arm].add(attempt_key)

        immutable_record_sha256 = row.get("immutable_record_sha256")
        immutable_source = dict(row)
        immutable_source.pop("immutable_record_sha256", None)
        if immutable_record_sha256 != _canonical_sha256(immutable_source):
            immutable_hash_failures += 1
            blockers.append(f"row {row_number} immutable_record_sha256 does not match row content")

        for binding_field, frozen_value in frozen_row_bindings.items():
            if row.get(binding_field) != frozen_value:
                frozen_binding_failures[binding_field] += 1

        normalized_hash = row.get("normalized_candidate_config_sha256")
        if not isinstance(normalized_hash, str) or not normalized_hash:
            blockers.append(f"row {row_number} lacks a normalized candidate hash")
        elif normalized_hash in normalized_hashes[arm]:
            duplicates_by_arm[arm] += 1
        else:
            normalized_hashes[arm].add(normalized_hash)

        if row.get("admission_decision") != "not_admitted_diagnostic_only":
            blockers.append(f"row {row_number} must remain not admitted in diagnostic-only mode")
        if row.get("exclusion_reason") != "diagnostic_only_no_replay_reference_or_second_context":
            blockers.append(f"row {row_number} must record the diagnostic-only exclusion reason")
        availability_status = row.get("availability_status")
        if availability_status in {
            "fallback",
            "degraded",
            "unavailable",
            "not_available",
        }:
            attrition_by_arm[arm] += 1
            blockers.append(
                f"row {row_number} has {availability_status!r} execution availability; it remains "
                "in the primary denominator but cannot support a diagnostic readiness result"
            )
        certification = row.get("certification")
        certification_status = (
            certification.get("status") if isinstance(certification, dict) else None
        )
        primary_failure = row.get("primary_failure_mechanism")
        if certification_status != "passed" or primary_failure in {
            "invalid_candidate",
            "evaluation_error",
        }:
            invalid_by_arm[arm] += 1
            blockers.append(
                f"row {row_number} is invalid or unevaluable; it remains in the primary "
                "denominator and cannot be complete-case evidence"
            )

    for binding_field, failure_count in frozen_binding_failures.items():
        if failure_count:
            blockers.append(
                f"{failure_count} row(s) do not match frozen {binding_field!r} binding from "
                "the contract"
            )

    expected_attempts = len(seeds) * candidate_budget
    missing_attempts = {
        arm: max(0, expected_attempts - len(observed_attempts.get(arm, set()))) for arm in methods
    }
    for arm, missing_count in missing_attempts.items():
        if missing_count:
            blockers.append(
                f"{arm} has {missing_count} missing scheduled attempts; primary denominator remains "
                f"{expected_attempts} under intention-to-search"
            )

    accounting = {
        "primary_estimand": "tpe_minus_random_unique_fully_admitted_weak_points",
        "primary_denominator_policy": "intention_to_search_all_scheduled_attempts",
        "expected_attempts_per_arm": expected_attempts,
        "observed_attempts_per_arm": {
            arm: len(observed_attempts.get(arm, set())) for arm in methods
        },
        "missing_attempts_per_arm": missing_attempts,
        "global_within_arm_normalized_hash_duplicates": duplicates_by_arm,
        "unique_normalized_hashes_per_arm": {
            arm: len(normalized_hashes.get(arm, set())) for arm in methods
        },
        "recorded_fallback_degraded_or_unavailable_rows": attrition_by_arm,
        "recorded_invalid_or_unevaluable_rows": invalid_by_arm,
        "complete_case_sensitivity": "not_run_diagnostic_only",
        "fully_admitted_yield": "not_estimated_diagnostic_only",
        "required_field_failure_count": required_field_failures,
        "immutable_hash_failure_count": immutable_hash_failures,
        "frozen_contract_sha256": frozen_preflight.metadata.get("contract_file_sha256"),
        "frozen_contract_preflight_ready": frozen_preflight.ready,
        "frozen_row_binding_failure_counts": frozen_binding_failures,
    }
    if any(duplicates_by_arm.values()):
        warnings.append(
            "normalized duplicate attempts are retained in the intention-to-search denominator and "
            "collapsed globally within each arm only for a future unique-candidate endpoint"
        )
    return Issue5303AnalysisResult(
        ready=not blockers,
        decision="inconclusive",
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        accounting=accounting,
    )
