"""Fail-closed analysis for the issue #5303 diagnostic search-stage handoff.

This module reads only the per-attempt JSONL emitted by the frozen diagnostic
command.  It never runs planners and it never promotes a result: three search
seeds per arm cannot satisfy the frozen two-sided null-test gate, so a structurally
complete diagnostic packet is still classified ``inconclusive``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from robot_sf.adversarial.config import CandidateSpec, Pose2D, SearchSpaceConfig
from robot_sf.adversarial.objectives import constraints_first_lexicographic_score
from robot_sf.benchmark.issue_5303_search_promotion_preregistration import (
    DEFAULT_MANIFEST_PATH,
    preflight_issue_5303_contract,
)

SCHEMA_VERSION = "issue_5303_search_promotion_analysis.v1"
OUTCOME_ROW_SCHEMA_VERSION = "issue_5303_search_promotion_outcome_row.v1"
DEFAULT_CONTRACT_PATH = Path("configs/adversarial/issue_5303_search_promotion_contract.yaml")
EXPECTED_EXECUTION_CONTEXT_LABEL = "diagnostic_adapter_context_a"
DIAGNOSTIC_NOT_RUN = "not_run_diagnostic_only"
_FROZEN_CANDIDATE_FIELDS = frozenset(
    {
        "start",
        "goal",
        "spawn_time_s",
        "pedestrian_speed_mps",
        "pedestrian_delay_s",
        "scenario_seed",
    }
)
_FROZEN_POSE_FIELDS = frozenset({"x", "y", "theta"})
_FROZEN_CONSTRAINTS_FIRST_FIELDS = frozenset(
    {
        "status",
        "collision_or_severe_intrusion",
        "liveness_or_goal_completion",
        "comfort_and_efficiency",
    }
)
_FROZEN_COMFORT_FIELDS = frozenset({"snqi", "near_misses", "path_efficiency"})
_FROZEN_CERTIFICATION_FIELDS = frozenset({"schema_version", "status", "reason", "details"})
_FROZEN_CERTIFICATION_SCHEMA_VERSION = "scenario_cert.v1"
_FROZEN_CERTIFICATION_STATUSES = frozenset({"passed", "failed", "not_available"})
_FROZEN_PRIMARY_FAILURE_MECHANISMS = frozenset(
    {
        "collision",
        "severe_intrusion",
        "timeout",
        "incomplete",
        "success",
        "invalid_candidate",
        "evaluation_error",
    }
)


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
    bindings["execution_context_label"] = EXPECTED_EXECUTION_CONTEXT_LABEL
    return bindings


def _finite_number(value: Any, *, field_name: str, errors: list[str]) -> float | None:
    """Parse one finite JSON number for a frozen candidate field.

    Returns:
        Parsed finite value, or ``None`` after recording an error.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{field_name} must be a finite number")
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        errors.append(f"{field_name} must be a finite number")
        return None
    return parsed


def _comfort_metric_errors(field_name: str, metric: Any) -> list[str]:
    """Return validation errors for one optional comfort/efficiency metric."""
    errors: list[str] = []
    parsed = _finite_number(
        metric,
        field_name=f"comfort_and_efficiency.{field_name}",
        errors=errors,
    )
    if parsed is None:
        return errors
    if field_name == "near_misses" and parsed < 0.0:
        errors.append("comfort_and_efficiency.near_misses must be non-negative")
    elif field_name == "path_efficiency" and not 0.0 <= parsed <= 1.0:
        errors.append("comfort_and_efficiency.path_efficiency must be between 0 and 1")
    return errors


def _candidate_search_space_errors(  # noqa: C901
    candidate: dict[str, Any], search_space: SearchSpaceConfig
) -> list[str]:
    """Validate a serialized candidate against the frozen search-space contract.

    Returns:
        Validation errors; an empty list means the candidate is in the frozen space.
    """
    errors: list[str] = []
    missing = sorted(_FROZEN_CANDIDATE_FIELDS - set(candidate))
    unexpected = sorted(set(candidate) - _FROZEN_CANDIDATE_FIELDS)
    if missing:
        errors.append(f"missing candidate fields: {missing}")
    if unexpected:
        errors.append(f"unexpected candidate fields: {unexpected}")
    if errors:
        return errors

    pose_values: dict[str, dict[str, float]] = {}
    for pose_name in ("start", "goal"):
        raw_pose = candidate.get(pose_name)
        if not isinstance(raw_pose, dict):
            errors.append(f"{pose_name} must be an object")
            continue
        pose_missing = sorted(_FROZEN_POSE_FIELDS - set(raw_pose))
        pose_unexpected = sorted(set(raw_pose) - _FROZEN_POSE_FIELDS)
        if pose_missing:
            errors.append(f"{pose_name} is missing fields: {pose_missing}")
        if pose_unexpected:
            errors.append(f"{pose_name} has unexpected fields: {pose_unexpected}")
        if pose_missing or pose_unexpected:
            continue
        values: dict[str, float] = {}
        for field_name in sorted(_FROZEN_POSE_FIELDS):
            parsed = _finite_number(
                raw_pose.get(field_name),
                field_name=f"{pose_name}.{field_name}",
                errors=errors,
            )
            if parsed is not None:
                values[field_name] = parsed
        if len(values) == len(_FROZEN_POSE_FIELDS):
            pose_values[pose_name] = values

    scalar_values: dict[str, float] = {}
    for field_name in (
        "spawn_time_s",
        "pedestrian_speed_mps",
        "pedestrian_delay_s",
    ):
        parsed = _finite_number(candidate.get(field_name), field_name=field_name, errors=errors)
        if parsed is not None:
            scalar_values[field_name] = parsed
    scenario_seed = candidate.get("scenario_seed")
    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int):
        errors.append("scenario_seed must be an integer")

    if errors:
        return errors
    candidate_spec = CandidateSpec(
        start=Pose2D(**pose_values["start"]),
        goal=Pose2D(**pose_values["goal"]),
        spawn_time_s=scalar_values["spawn_time_s"],
        pedestrian_speed_mps=scalar_values["pedestrian_speed_mps"],
        pedestrian_delay_s=scalar_values["pedestrian_delay_s"],
        scenario_seed=int(scenario_seed),
    )
    return search_space.validate_candidate(candidate_spec)


def _normalized_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical numeric representation of a valid frozen candidate.

    JSON permits equivalent integers and floats (for example, ``2`` and ``2.0``),
    but their serialized digests differ. Candidate identity must follow the
    validated representation so numeric spelling cannot evade deduplication.

    Returns:
        Candidate payload with continuous fields serialized as floats and the
        scenario seed serialized as an integer.
    """
    return {
        "start": {
            field_name: float(candidate["start"][field_name])
            for field_name in sorted(_FROZEN_POSE_FIELDS)
        },
        "goal": {
            field_name: float(candidate["goal"][field_name])
            for field_name in sorted(_FROZEN_POSE_FIELDS)
        },
        "spawn_time_s": float(candidate["spawn_time_s"]),
        "pedestrian_speed_mps": float(candidate["pedestrian_speed_mps"]),
        "pedestrian_delay_s": float(candidate["pedestrian_delay_s"]),
        "scenario_seed": int(candidate["scenario_seed"]),
    }


def _constraints_first_outcome_errors(value: Any) -> list[str]:  # noqa: C901
    """Validate the complete observed constraints-first outcome projection.

    Returns:
        Validation errors; an empty list means the outcome is complete.
    """
    if not isinstance(value, dict):
        return ["must be an object"]
    errors: list[str] = []
    missing = sorted(_FROZEN_CONSTRAINTS_FIRST_FIELDS - set(value))
    unexpected = sorted(set(value) - _FROZEN_CONSTRAINTS_FIRST_FIELDS)
    if missing:
        errors.append(f"missing fields: {missing}")
    if unexpected:
        errors.append(f"unexpected fields: {unexpected}")
    if value.get("status") != "observed":
        errors.append("status must be 'observed'")
    for field_name in ("collision_or_severe_intrusion", "liveness_or_goal_completion"):
        if not isinstance(value.get(field_name), bool):
            errors.append(f"{field_name} must be boolean")
    comfort = value.get("comfort_and_efficiency")
    if not isinstance(comfort, dict):
        errors.append("comfort_and_efficiency must be an object")
    else:
        comfort_missing = sorted(_FROZEN_COMFORT_FIELDS - set(comfort))
        comfort_unexpected = sorted(set(comfort) - _FROZEN_COMFORT_FIELDS)
        if comfort_missing:
            errors.append(f"comfort_and_efficiency is missing fields: {comfort_missing}")
        if comfort_unexpected:
            errors.append(f"comfort_and_efficiency has unexpected fields: {comfort_unexpected}")
        for field_name in _FROZEN_COMFORT_FIELDS:
            metric = comfort.get(field_name)
            if metric is not None:
                errors.extend(_comfort_metric_errors(field_name, metric))
    return errors


def _certification_errors(value: Any) -> list[str]:
    """Validate the complete ``scenario_cert.v1`` status carried by a row.

    Returns:
        Validation errors; an empty list means the certification payload is complete.
    """
    if not isinstance(value, dict):
        return ["must be an object"]
    errors: list[str] = []
    missing = sorted(_FROZEN_CERTIFICATION_FIELDS - set(value))
    unexpected = sorted(set(value) - _FROZEN_CERTIFICATION_FIELDS)
    if missing:
        errors.append(f"missing fields: {missing}")
    if unexpected:
        errors.append(f"unexpected fields: {unexpected}")
    if value.get("schema_version") != _FROZEN_CERTIFICATION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {_FROZEN_CERTIFICATION_SCHEMA_VERSION!r}")
    if value.get("status") not in _FROZEN_CERTIFICATION_STATUSES:
        errors.append(f"status must be one of {sorted(_FROZEN_CERTIFICATION_STATUSES)!r}")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("reason must be a non-empty string")
    if not isinstance(value.get("details"), dict):
        errors.append("details must be an object")
    return errors


def _constraints_first_objective_value_errors(
    objective_value: Any, outcome: dict[str, Any]
) -> list[str]:
    """Validate that one objective value matches the frozen shared formula.

    Returns:
        Value errors, or an empty list when the value is finite and exact.
    """
    if (
        isinstance(objective_value, bool)
        or not isinstance(objective_value, (int, float))
        or not math.isfinite(float(objective_value))
    ):
        return []
    score = float(objective_value)
    expected = constraints_first_lexicographic_score(outcome)
    if expected is None:
        return ["objective_value cannot be derived from the frozen observed outcome"]
    if not math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-12):
        return [
            f"objective_value {score} does not match the frozen constraints-first value {expected}"
        ]
    return []


def _primary_failure_mechanism_errors(primary_failure: Any, outcome: dict[str, Any]) -> list[str]:
    """Validate attribution against the observed constraints-first outcome tier.

    Returns:
        Validation errors; an empty list means the mechanism matches the outcome tier.
    """
    if not isinstance(primary_failure, str):
        return []
    if primary_failure in {"invalid_candidate", "evaluation_error"}:
        return []
    if outcome["collision_or_severe_intrusion"]:
        expected = {"collision", "severe_intrusion"}
        tier = "collision/severe-intrusion"
    elif outcome["liveness_or_goal_completion"]:
        expected = {"timeout", "incomplete"}
        tier = "liveness"
    else:
        expected = {"success"}
        tier = "successful completion"
    if primary_failure not in expected:
        return [
            f"primary_failure_mechanism {primary_failure!r} contradicts the observed {tier} "
            f"tier; expected one of {sorted(expected)!r}"
        ]
    return []


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
    frozen_search_space: SearchSpaceConfig | None = None
    frozen_search_space_path = frozen_row_bindings.get("search_space_path")
    if frozen_search_space_path:
        try:
            frozen_search_space = SearchSpaceConfig.from_file(
                _resolve(root, Path(frozen_search_space_path))
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            blockers.append(f"frozen search space could not be loaded: {exc}")
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
    normalized_candidate_hash_failures = 0
    candidate_schema_failures = 0
    frozen_binding_failures: dict[str, int] = dict.fromkeys(frozen_row_bindings, 0)
    execution_commits: set[str] = set()

    for row_number, row in enumerate(rows, start=1):
        row_fields = set(row)
        missing = sorted(required_fields - row_fields)
        unexpected = sorted(row_fields - required_fields)
        if missing or unexpected:
            required_field_failures += 1
            if missing:
                blockers.append(f"row {row_number} is missing required fields: {missing}")
            if unexpected:
                blockers.append(f"row {row_number} has unexpected fields: {unexpected}")
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

        expected_row_id = f"{arm}:{search_seed}:{candidate_index:04d}:search"
        if row.get("row_id") != expected_row_id:
            blockers.append(f"row {row_number} row_id does not match its scheduled attempt")

        for binding_field, frozen_value in frozen_row_bindings.items():
            if row.get(binding_field) != frozen_value:
                frozen_binding_failures[binding_field] += 1

        candidate = row.get("candidate")
        normalized_hash = row.get("normalized_candidate_config_sha256")
        verified_candidate_hash: str | None = None
        if not isinstance(candidate, dict):
            candidate_schema_failures += 1
            normalized_candidate_hash_failures += 1
            blockers.append(f"row {row_number} candidate must be an object")
        else:
            candidate_errors: list[str] = []
            if frozen_search_space is not None:
                candidate_errors = _candidate_search_space_errors(candidate, frozen_search_space)
                if candidate_errors:
                    candidate_schema_failures += 1
                    blockers.append(
                        f"row {row_number} candidate does not match the frozen search space: "
                        + "; ".join(candidate_errors)
                    )
            if not candidate_errors and frozen_search_space is not None:
                verified_candidate_hash = _canonical_sha256(
                    _normalized_candidate_payload(candidate)
                )
            else:
                verified_candidate_hash = _canonical_sha256(candidate)
            if normalized_hash != verified_candidate_hash:
                normalized_candidate_hash_failures += 1
                blockers.append(
                    f"row {row_number} normalized candidate hash does not match candidate content"
                )
        if verified_candidate_hash in normalized_hashes[arm]:
            duplicates_by_arm[arm] += 1
        elif verified_candidate_hash is not None:
            normalized_hashes[arm].add(verified_candidate_hash)

        candidate_seed = candidate.get("scenario_seed") if isinstance(candidate, dict) else None
        if (
            isinstance(candidate_seed, bool)
            or not isinstance(candidate_seed, int)
            or row.get("execution_seed") != candidate_seed
        ):
            blockers.append(f"row {row_number} execution_seed must match candidate.scenario_seed")
        seed_lineage = row.get("seed_lineage")
        expected_seed_lineage = {
            "search_seed": search_seed,
            "candidate_scenario_seed": candidate_seed,
            "deterministic_replay_seed": None,
            "confirmation_seeds": [],
            "second_context_seed": None,
        }
        if (
            not isinstance(seed_lineage, dict)
            or set(seed_lineage) != set(expected_seed_lineage)
            or any(seed_lineage.get(key) != value for key, value in expected_seed_lineage.items())
        ):
            blockers.append(f"row {row_number} seed_lineage does not match the diagnostic attempt")

        execution_commit = row.get("execution_commit")
        if (
            not isinstance(execution_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}", execution_commit) is None
        ):
            blockers.append(f"row {row_number} execution_commit must be a 40-character Git SHA")
        else:
            execution_commits.add(execution_commit)

        if row.get("admission_decision") != "not_admitted_diagnostic_only":
            blockers.append(f"row {row_number} must remain not admitted in diagnostic-only mode")
        if row.get("exclusion_reason") != "diagnostic_only_no_replay_reference_or_second_context":
            blockers.append(f"row {row_number} must record the diagnostic-only exclusion reason")
        for field_name in (
            "deterministic_replay",
            "confirmation_target",
            "confirmation_reference",
            "second_execution_context",
        ):
            if row.get(field_name) != DIAGNOSTIC_NOT_RUN:
                blockers.append(f"row {row_number} {field_name} must remain {DIAGNOSTIC_NOT_RUN!r}")
        if row.get("stable_attribution_evidence") != "not_collected_diagnostic_only":
            blockers.append(
                f"row {row_number} stable_attribution_evidence must remain diagnostic-only"
            )
        if row.get("recertification_lineage") != "issue_6139_frozen_input":
            blockers.append(f"row {row_number} has invalid recertification lineage")

        readiness_status = row.get("readiness_status")
        required_readiness = frozen_row_bindings.get("execution_mode", "").strip().lower()
        if (
            not isinstance(readiness_status, str)
            or readiness_status.strip().lower() != required_readiness
        ):
            blockers.append(
                f"row {row_number} readiness_status must match frozen execution mode "
                f"{required_readiness!r}"
            )
        availability_status = row.get("availability_status")
        if availability_status != "available":
            attrition_by_arm[arm] += 1
            blockers.append(
                f"row {row_number} has {availability_status!r} execution availability; it remains "
                "in the primary denominator but cannot support a diagnostic readiness result"
            )
        constraints_first_outcome = row.get("constraints_first_outcome")
        outcome_errors = _constraints_first_outcome_errors(constraints_first_outcome)
        if outcome_errors:
            blockers.append(
                f"row {row_number} has an incomplete constraints-first outcome: "
                + "; ".join(outcome_errors)
            )
        objective_value = row.get("objective_value")
        if (
            isinstance(objective_value, bool)
            or not isinstance(objective_value, (int, float))
            or not math.isfinite(float(objective_value))
        ):
            blockers.append(f"row {row_number} objective_value must be finite")
        elif not outcome_errors:
            objective_value_errors = _constraints_first_objective_value_errors(
                objective_value, constraints_first_outcome
            )
            blockers.extend(f"row {row_number} {error}" for error in objective_value_errors)
        certification = row.get("certification")
        certification_errors = _certification_errors(certification)
        if certification_errors:
            blockers.append(
                f"row {row_number} has an incomplete certification payload: "
                + "; ".join(certification_errors)
            )
        certification_status = (
            certification.get("status") if isinstance(certification, dict) else None
        )
        primary_failure = row.get("primary_failure_mechanism")
        primary_failure_is_known = (
            isinstance(primary_failure, str)
            and primary_failure in _FROZEN_PRIMARY_FAILURE_MECHANISMS
        )
        if not primary_failure_is_known:
            blockers.append(
                f"row {row_number} primary_failure_mechanism must be one of "
                f"{sorted(_FROZEN_PRIMARY_FAILURE_MECHANISMS)!r}"
            )
        primary_failure_errors = (
            _primary_failure_mechanism_errors(primary_failure, constraints_first_outcome)
            if primary_failure_is_known and not outcome_errors
            else []
        )
        blockers.extend(f"row {row_number} {error}" for error in primary_failure_errors)
        if (
            certification_errors
            or certification_status != "passed"
            or not primary_failure_is_known
            or primary_failure_errors
            or primary_failure in {"invalid_candidate", "evaluation_error"}
        ):
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
    if len(execution_commits) > 1:
        blockers.append("diagnostic rows must share one execution_commit")

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
        "normalized_candidate_hash_failure_count": normalized_candidate_hash_failures,
        "candidate_schema_failure_count": candidate_schema_failures,
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
