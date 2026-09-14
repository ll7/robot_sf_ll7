"""Outcome-free packet, identity, canary, and lineage validators for #8891.

Only packet/source bytes and planned fixture data are read.  No sampler,
planner, simulator, campaign runner, or result file is imported or invoked.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PACKET_SCHEMA_VERSION = "matched-budget-temporal-robustness-packet.v1"
IDENTITY_SCHEMA_VERSION = "matched-budget-temporal-robustness-identities.v1"
LEDGER_SCHEMA_VERSION = "temporal-robustness-call-ledger.v1"
CANARY_SCHEMA_VERSION = "matched-budget-temporal-robustness-canary.v1"
TEMPORAL_SIDECAR_SCHEMA_VERSION = "temporal-mechanism-sidecar.v1"
EXPECTED_OBJECTIVES = ("worst_case_snqi", "temporal_robustness")
EXPECTED_FAMILIES = ("random", "optuna", "cmaes")
EXPECTED_BUDGETS = (16, 32, 64)
EXPECTED_SEARCH_SEEDS = (1101, 2202, 3303)
EXPECTED_PROPERTIES = ("clearance", "ttc", "goal", "progress", "collision")
EXPECTED_MODES = ("native", "fallback", "degraded", "unavailable", "synthetic_fixture")
CANARY_SEED_BASE = 8_800_000
ROLE_SEED_BASE = 8_900_000
CONFIRMATION_COUNT = 5
RUN_COUNT = (
    len(EXPECTED_OBJECTIVES)
    * len(EXPECTED_FAMILIES)
    * len(EXPECTED_BUDGETS)
    * len(EXPECTED_SEARCH_SEEDS)
)
SLOT_COUNT = (
    len(EXPECTED_OBJECTIVES)
    * len(EXPECTED_FAMILIES)
    * len(EXPECTED_SEARCH_SEEDS)
    * sum(EXPECTED_BUDGETS)
)
SIMULATOR_INVOCATIONS_PER_SLOT = 1 + 1 + CONFIRMATION_COUNT
SIMULATOR_CALL_BUDGET = SLOT_COUNT * SIMULATOR_INVOCATIONS_PER_SLOT
SIMULATOR_CALL_BUDGET_BY_SEARCH_BUDGET = {
    budget: budget * SIMULATOR_INVOCATIONS_PER_SLOT for budget in EXPECTED_BUDGETS
}

EXPECTED_PACKET_ID = "issue_8891_temporal_robustness_matched_budget_v1"
EXPECTED_PACKET_STATUS = "diagnostic_only_preflight"
EXPECTED_EVIDENCE_TIER = "preflight_valid"
CONFIRMATION_THRESHOLD = 3

_REQUIRED_INPUTS = set(
    "parent_manifest scenario_template search_space objective_registry robustness samplers runner benchmark_runner certification replay confirmation".split()
)
_FORBIDDEN_FIELDS = set(
    "outcome objective_value observed_value result_rows simulator_output slurm_job_id job_id target_host submitted_at".split()
)
_GATE_STATES = ("certification_state", "replay_state", "independent_seed_state")
_GATE_ORDER = ("certification", "deterministic_replay", "independent_confirmation")
_GATE_STATE_VALUES = {"not_run", "passed", "failed", "excluded"}
_ADMISSION_STATUSES = {
    "not_admitted",
    "confirmed_failure",
    "null",
    "inconclusive",
    "invalid",
    "unavailable",
    "blocked",
    "excluded",
}
_SIDECAR_ALLOWED_KEYS = {
    "schema_version",
    "status",
    "candidate_id",
    "property_ids",
    "properties",
    "monitor",
    "admission_status",
    "failure_basis",
}
_SIDECAR_PROPERTY_ALLOWED_KEYS = {
    "property_id",
    "signed_margin",
    "activation_time_s",
    "certification_state",
    "replay_state",
    "independent_seed_state",
    "execution_mode",
    "provenance",
}
_SIDECAR_MONITOR_ALLOWED_KEYS = {"dt_s", "sample_count", "artifact_only"}
_SIDECAR_PROVENANCE_ALLOWED_KEYS = {"source", "packet_digest"}
_LINEAGE_STATE_BY_PHASE = {
    "certification": "certification_state",
    "replay": "replay_state",
    "confirmation": "independent_seed_state",
}
_SECONDARY_OUTCOMES = "first_confirmed_failure_attempt valid_rate invalid_rate simulator_invocations certification_rate replay_rate confirmation_rate monitor_artifact_exclusions missingness result_class".split()
_RESULT_CLASSES = "confirmed_failure null inconclusive invalid unavailable blocked".split()
_PACKET_KEYS = "schema_version issue parent_issue claim_eligible"
_PACKET_ALLOWED_KEYS = {
    "schema_version",
    "packet_id",
    "issue",
    "parent_issue",
    "status",
    "evidence_tier",
    "claim_eligible",
    "claim_boundary",
    "execution_boundary",
    "source",
    "objectives",
    "excluded_search_families",
    "search_families",
    "search_family_builder",
    "initialization_policy",
    "scenario",
    "budget",
    "seed_policy",
    "call_accounting",
    "gates",
    "monitor_contract",
    "analysis_contract",
    "private_ops",
    "validation",
}
_SOURCE_ALLOWED_KEYS = {"base_ref", "base_commit", "hash_algorithm", "inputs"}
_INPUT_ALLOWED_KEYS = {"path", "sha256", "working_tree_sha256"}
_OBJECTIVE_ALLOWED_KEYS = {"id", "role", "owner", "sidecar_schema"}
_SEARCH_FAMILY_ALLOWED_KEYS = {"id", "role", "owner"}
_INITIALIZATION_ALLOWED_KEYS = {
    "mode",
    "sampler_instance",
    "proposal_order",
    "warm_start",
    "replacement_rows",
    "optimizer_defaults",
    "post_outcome_changes",
}
_SCENARIO_ALLOWED_KEYS = {
    "template",
    "search_space",
    "policy",
    "template_max_episode_steps",
    "horizon_steps",
    "dt_s",
    "parameter_order",
    "parameters",
}
_SCENARIO_PARAMETER_ALLOWED_KEYS = {"name", "bounds"}
_SCENARIO_BOUNDS_ALLOWED_KEYS = {"min", "max"}
_EXECUTION_KEYS = "run_campaign run_simulator submit_slurm registered_search admit_evidence".split()
_BUDGET_KEYS = (
    "budgets run_count search_attempt_slots simulator_call_budget"
    " simulator_call_budget_by_search_budget simulator_invocations_per_search_slot"
    " budget_unit simulator_call_budget_policy simulator_call_budget_is_separate"
    " no_post_outcome_budget_change"
)
_BUDGET_ALLOWED_KEYS = set(_BUDGET_KEYS.split()) | {"matched_across"}
_ACCOUNTING_KEYS = (
    "schema_version authoritative_counter search_counter one_row_per_attempt_or_call hidden_retries"
)
_ACCOUNTING_ALLOWED_KEYS = set(_ACCOUNTING_KEYS.split()) | {"classes", "rule"}
_LEDGER_KEYS = ("phase", "seed_role", "consumes_search_slot", "simulator_invocations")
_SEED_ALLOWED_KEYS = {
    "search_seeds",
    "scenario_seed_domain",
    "candidate_seed_mode",
    "role_seed_base",
    "canary_seed_base",
    "replay",
    "confirmation",
    "disjointness",
}
_SEED_DOMAIN_ALLOWED_KEYS = {"min", "max"}
_REPLAY_ALLOWED_KEYS = {"count_per_candidate", "derivation"}
_CONFIRMATION_ALLOWED_KEYS = {
    "count_per_candidate",
    "threshold",
    "derivation",
    "source_rule",
}
_MONITOR_KEYS = "property_ids sidecar_schema monitor_only_excluded"
_MONITOR_ALLOWED_KEYS = {
    "property_ids",
    "sidecar_schema",
    "required_mechanism_fields",
    "monitor_artifact_fields",
    "discretization",
    "monitor_only_excluded",
    "source_semantics_unchanged",
}
_ANALYSIS_KEYS = "primary_estimand primary_unit numerator denominator"
_ANALYSIS_EXPECTED = (
    "confirmed_failure_discovery_rate_by_simulator_call_budget",
    "simulator_call_budget",
    "candidate_slot_passing_certification_replay_and_independent_confirmation",
    "simulator_call_budget_per_cell",
)
_ANALYSIS_ALLOWED_KEYS = {
    "primary_estimand",
    "primary_unit",
    "numerator",
    "denominator",
    "secondary_outcomes",
    "uncertainty",
    "multiplicity",
    "deduplication_key",
    "duplicate_policy",
    "missingness_policy",
    "result_classes",
    "resampling_unit",
    "raw_objective_values_are_failures",
    "post_outcome_design_changes",
}
_UNCERTAINTY_ALLOWED_KEYS = {"interval", "confidence", "resampling_unit"}
_GATES_ALLOWED_KEYS = {
    "order",
    "certification",
    "deterministic_replay",
    "independent_confirmation",
    "combined_failure_rule",
    "confirmation_threshold",
}
_GATE_ALLOWED_KEYS = {"owner", "criterion", "required"}
_PRIVATE_OPS_ALLOWED_KEYS = {
    "execution_authorized",
    "scheduler_submission_allowed",
    "stage_command",
    "resource_estimate",
    "storage_estimate",
    "preservation",
}
_RESOURCE_ESTIMATE_ALLOWED_KEYS = {
    "search_attempt_slots",
    "certification_records",
    "deterministic_replay_calls",
    "confirmation_calls",
    "max_simulator_invocations",
    "max_trace_bytes_per_simulator_call",
    "raw_trace_ceiling_bytes",
    "estimate_basis",
}
_STORAGE_ESTIMATE_ALLOWED_KEYS = {
    "raw_trace_ceiling_bytes",
    "derived_summary_ceiling_bytes",
    "total_planning_ceiling_bytes",
}
_PRESERVATION_ALLOWED_KEYS = {
    "raw_out_of_git",
    "raw_and_derived_separate",
    "checksum_algorithm",
    "retain_invalid_unavailable_failed_rows",
    "promote_only_reviewed_compact_summary",
    "local_output_is_disposable",
}
_VALIDATION_ALLOWED_KEYS = {"commands", "no_campaign"}
_CALL_RULES = {
    "search_evaluation": ("search", "search", True, 1),
    "search_invalid_proposal": ("search", "search", True, 0),
    "search_evaluation_failure": ("search", "search", True, 1),
    "certification": ("certification", "none", False, 0),
    "deterministic_replay": ("replay", "replay", False, 1),
    "independent_confirmation": ("confirmation", "confirmation", False, 1),
}


class PacketError(ValueError):
    """Raised when a packet or row is unsafe or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _expect(actual: Any, expected: Any, label: str) -> None:
    _require(actual == expected, f"{label} mismatch")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _list(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list) and value, f"{name} must be a non-empty list")
    return value


def _norm(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _check(
    mapping: Mapping[str, Any], keys: Sequence[str] | str, expected: Sequence[Any], label: str
) -> None:
    keys = keys.split() if isinstance(keys, str) else keys
    _expect(
        tuple(_norm(mapping.get(key)) for key in keys),
        tuple(_norm(value) for value in expected),
        label,
    )


def _check_values(mapping: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        _expect(_norm(mapping.get(key)), _norm(value), f"{label}.{key}")


def _reject_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    """Reject fields outside the versioned contract allowlist."""
    unknown = set(mapping) - allowed
    _require(
        not unknown,
        f"{label} contains unsupported fields: {sorted(str(key) for key in unknown)}",
    )


def _ids(value: Any, name: str) -> tuple[str, ...]:
    return tuple(str(_mapping(item, name).get("id")) for item in _list(value, name))


def _finite(value: Any, label: str, *, minimum: float | None = None) -> None:
    valid = isinstance(value, (int, float)) and not isinstance(value, bool)
    _require(valid and math.isfinite(float(value)), f"invalid {label}")
    if minimum is not None:
        _require(float(value) >= minimum, f"invalid {label}")


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with stable canonical JSON."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _repo_file(root: Path, value: Any, name: str) -> Path:
    _require(isinstance(value, str) and value.strip(), f"{name} must be a path")
    path = (root / value).resolve()
    _require(path.is_relative_to(root.resolve()), f"{name} must be repo-relative")
    _require(path.is_file(), f"{name} not found: {value}")
    return path


def _forbid_outcomes(value: Any) -> None:
    if isinstance(value, Mapping):
        _require(not _FORBIDDEN_FIELDS.intersection(value), "outcome fields are not allowed")
        children = value.values()
    elif isinstance(value, list):
        children = value
    else:
        return
    for child in children:
        _forbid_outcomes(child)


def _git_output(root: Path, *args: str) -> bytes:
    """Read immutable Git data and turn lookup failures into packet errors."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise PacketError(f"immutable source lookup failed: {' '.join(args)}") from exc
    return result.stdout


def _resolve_source_commit(packet: Mapping[str, Any], root: Path) -> str:
    """Resolve and ancestry-check the packet's immutable source commit."""
    source = _mapping(packet.get("source"), "source")
    commit = str(source.get("base_commit", ""))
    _require(
        len(commit) == 40
        and commit == commit.lower()
        and not set(commit) - set("0123456789abcdef"),
        "source.base_commit must be a lowercase full SHA",
    )
    try:
        resolved = (
            _git_output(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
        )
    except PacketError as exc:
        raise PacketError("source.base_commit cannot be resolved") from exc
    _expect(resolved, commit, "source.base_commit resolution")
    base_ref = str(source.get("base_ref", ""))
    _require(base_ref == "origin/main", "source.base_ref must be origin/main")
    try:
        ref_commit = (
            _git_output(root, "rev-parse", "--verify", f"{base_ref}^{{commit}}").decode().strip()
        )
        ancestry = subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, ref_commit],
            check=False,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise PacketError("source.base_ref cannot be resolved") from exc
    _require(ancestry.returncode == 0, "source.base_commit is not an ancestor of source.base_ref")
    return commit


def _input_paths(packet: Mapping[str, Any], root: Path, *, source_commit: str) -> dict[str, Path]:
    inputs = _mapping(_mapping(packet.get("source"), "source").get("inputs"), "source.inputs")
    paths = {}
    for input_id, raw in inputs.items():
        item = _mapping(raw, f"source.inputs.{input_id}")
        _reject_unknown_keys(item, _INPUT_ALLOWED_KEYS, f"source.inputs.{input_id}")
        path = _repo_file(root, item.get("path"), f"source.inputs.{input_id}.path")
        digest = str(item.get("sha256", "")).lower()
        _require(
            len(digest) == 64 and not set(digest) - set("0123456789abcdef"),
            f"source hash {input_id} must be SHA-256",
        )
        relative_path = path.relative_to(root.resolve()).as_posix()
        try:
            immutable_bytes = _git_output(root, "show", f"{source_commit}:{relative_path}")
        except PacketError as exc:
            raise PacketError(f"immutable source path cannot be resolved: {relative_path}") from exc
        _expect(
            hashlib.sha256(immutable_bytes).hexdigest(),
            digest,
            f"immutable source hash {input_id}",
        )
        working_tree_digest = item.get("working_tree_sha256", digest)
        _require(
            isinstance(working_tree_digest, str)
            and len(working_tree_digest) == 64
            and working_tree_digest == working_tree_digest.lower()
            and not set(working_tree_digest) - set("0123456789abcdef"),
            f"working-tree source hash {input_id} must be SHA-256",
        )
        _expect(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            working_tree_digest,
            f"working-tree source hash {input_id}",
        )
        paths[str(input_id)] = path
    return paths


def _parameter_signature(value: Any) -> list[tuple[str, Any, Any]]:
    return [
        (str(item["name"]), item["bounds"]["min"], item["bounds"]["max"])
        for item in _list(value, "scenario.parameters")
    ]


def _validate_source_semantics(packet: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    source = _mapping(packet["source"], "source")
    commit = str(source.get("base_commit", ""))
    _require(
        len(commit) == 40 and not set(commit) - set("0123456789abcdef"),
        "source.base_commit must be a full SHA",
    )
    _expect(source.get("base_ref"), "origin/main", "source.base_ref")
    scenario = _mapping(packet["scenario"], "scenario")
    space = _mapping(
        yaml.safe_load(paths["search_space"].read_text(encoding="utf-8")), "search space"
    )
    variables = _mapping(space.get("variables"), "search_space.variables")
    expected = [(name, item.get("min"), item.get("max")) for name, item in variables.items()]
    _expect(_parameter_signature(scenario.get("parameters")), expected, "scenario parameters")
    expected_order = tuple(variables)
    _expect(tuple(scenario.get("parameter_order", ())), expected_order, "parameter order")
    _check(scenario, ("policy", "horizon_steps", "dt_s"), ("goal", 100, 0.1), "scenario")
    template = _mapping(
        yaml.safe_load(paths["scenario_template"].read_text(encoding="utf-8")), "scenario template"
    )
    first = _mapping(_list(template.get("scenarios"), "scenario_template.scenarios")[0], "scenario")
    simulation = _mapping(first.get("simulation_config"), "scenario_template.simulation_config")
    _expect(
        simulation.get("max_episode_steps"),
        scenario.get("template_max_episode_steps"),
        "template horizon",
    )


def _validate_packet_nested_keys(packet: Mapping[str, Any]) -> None:
    """Reject unversioned fields in packet sections with nested mappings."""
    objectives = _list(packet.get("objectives"), "objectives")
    for index, objective in enumerate(objectives):
        _reject_unknown_keys(
            _mapping(objective, f"objectives[{index}]"),
            _OBJECTIVE_ALLOWED_KEYS,
            f"objectives[{index}]",
        )
    search_families = _list(packet.get("search_families"), "search_families")
    for index, family in enumerate(search_families):
        _reject_unknown_keys(
            _mapping(family, f"search_families[{index}]"),
            _SEARCH_FAMILY_ALLOWED_KEYS,
            f"search_families[{index}]",
        )

    scenario = _mapping(packet.get("scenario"), "scenario")
    _reject_unknown_keys(scenario, _SCENARIO_ALLOWED_KEYS, "scenario")
    for index, parameter in enumerate(_list(scenario.get("parameters"), "scenario.parameters")):
        parameter_mapping = _mapping(parameter, f"scenario.parameters[{index}]")
        _reject_unknown_keys(
            parameter_mapping,
            _SCENARIO_PARAMETER_ALLOWED_KEYS,
            f"scenario.parameters[{index}]",
        )
        _reject_unknown_keys(
            _mapping(parameter_mapping.get("bounds"), f"scenario.parameters[{index}].bounds"),
            _SCENARIO_BOUNDS_ALLOWED_KEYS,
            f"scenario.parameters[{index}].bounds",
        )
    _reject_unknown_keys(
        _mapping(packet.get("initialization_policy"), "initialization_policy"),
        _INITIALIZATION_ALLOWED_KEYS,
        "initialization policy",
    )

    budget = _mapping(packet.get("budget"), "budget")
    _reject_unknown_keys(budget, _BUDGET_ALLOWED_KEYS, "budget")
    seed = _mapping(packet.get("seed_policy"), "seed_policy")
    _reject_unknown_keys(seed, _SEED_ALLOWED_KEYS, "seed policy")
    _reject_unknown_keys(
        _mapping(seed.get("scenario_seed_domain"), "scenario seed domain"),
        _SEED_DOMAIN_ALLOWED_KEYS,
        "scenario seed domain",
    )
    _reject_unknown_keys(_mapping(seed.get("replay"), "replay"), _REPLAY_ALLOWED_KEYS, "replay")
    _reject_unknown_keys(
        _mapping(seed.get("confirmation"), "confirmation"),
        _CONFIRMATION_ALLOWED_KEYS,
        "confirmation",
    )

    accounting = _mapping(packet.get("call_accounting"), "call_accounting")
    _reject_unknown_keys(accounting, _ACCOUNTING_ALLOWED_KEYS, "call accounting")
    classes = _mapping(accounting.get("classes"), "call_accounting.classes")
    for name in _CALL_RULES:
        class_mapping = _mapping(classes.get(name), f"call class {name}")
        _reject_unknown_keys(class_mapping, set(_LEDGER_KEYS), f"call class {name}")

    gates = _mapping(packet.get("gates"), "gates")
    _reject_unknown_keys(gates, _GATES_ALLOWED_KEYS, "gates")
    for name in gates.get("order", ()):
        _reject_unknown_keys(
            _mapping(gates.get(name), f"gate.{name}"),
            _GATE_ALLOWED_KEYS,
            f"gate.{name}",
        )
    monitor = _mapping(packet.get("monitor_contract"), "monitor_contract")
    _reject_unknown_keys(monitor, _MONITOR_ALLOWED_KEYS, "monitor contract")
    analysis = _mapping(packet.get("analysis_contract"), "analysis_contract")
    _reject_unknown_keys(analysis, _ANALYSIS_ALLOWED_KEYS, "analysis contract")
    _reject_unknown_keys(
        _mapping(analysis.get("uncertainty"), "uncertainty"),
        _UNCERTAINTY_ALLOWED_KEYS,
        "uncertainty",
    )

    ops = _mapping(packet.get("private_ops"), "private_ops")
    _reject_unknown_keys(ops, _PRIVATE_OPS_ALLOWED_KEYS, "private_ops")
    _reject_unknown_keys(
        _mapping(ops.get("resource_estimate"), "resource estimate"),
        _RESOURCE_ESTIMATE_ALLOWED_KEYS,
        "resource estimate",
    )
    _reject_unknown_keys(
        _mapping(ops.get("storage_estimate"), "storage estimate"),
        _STORAGE_ESTIMATE_ALLOWED_KEYS,
        "storage estimate",
    )
    _reject_unknown_keys(
        _mapping(ops.get("preservation"), "preservation"),
        _PRESERVATION_ALLOWED_KEYS,
        "preservation",
    )
    _reject_unknown_keys(
        _mapping(packet.get("validation"), "validation"),
        _VALIDATION_ALLOWED_KEYS,
        "validation",
    )


def validate_packet(packet: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Validate the source-bound, non-executing packet."""
    _require(isinstance(packet, Mapping), "packet must be a mapping")
    _reject_unknown_keys(packet, _PACKET_ALLOWED_KEYS, "packet")
    _forbid_outcomes(packet)
    _check(packet, _PACKET_KEYS, (PACKET_SCHEMA_VERSION, 8891, 5326, False), "packet")
    _check_values(
        packet,
        {
            "packet_id": EXPECTED_PACKET_ID,
            "status": EXPECTED_PACKET_STATUS,
            "evidence_tier": EXPECTED_EVIDENCE_TIER,
        },
        "packet metadata",
    )
    boundary = str(packet.get("claim_boundary", "")).lower()
    _require(
        all(
            p in boundary for p in ("does not run", "no campaign", "no benchmark", "no publication")
        ),
        "claim boundary is incomplete",
    )
    execution = _mapping(packet.get("execution_boundary"), "execution_boundary")
    _reject_unknown_keys(execution, set(_EXECUTION_KEYS), "execution boundary")
    _check_values(execution, dict.fromkeys(_EXECUTION_KEYS, False), "execution boundary")
    source = _mapping(packet.get("source"), "source")
    _reject_unknown_keys(source, _SOURCE_ALLOWED_KEYS, "source")
    source_commit = _resolve_source_commit(packet, repo_root)
    paths = _input_paths(packet, repo_root, source_commit=source_commit)
    _require(_REQUIRED_INPUTS.issubset(paths), "source inputs are incomplete")
    _validate_source_semantics(packet, paths)
    _validate_packet_nested_keys(packet)
    _expect(
        (
            _ids(packet.get("objectives"), "objective"),
            _ids(packet.get("search_families"), "search family"),
        ),
        (EXPECTED_OBJECTIVES, EXPECTED_FAMILIES),
        "objective/search rosters",
    )
    _expect(
        packet.get("search_family_builder"),
        "robot_sf.adversarial.samplers.build_sampler",
        "search family builder",
    )
    init = _mapping(packet.get("initialization_policy"), "initialization_policy")
    _check_values(
        init,
        {
            "mode": "cold_start",
            "sampler_instance": "one_per_run_cell",
            "proposal_order": "ascending_attempt_index",
            "warm_start": "forbidden",
            "replacement_rows": "forbidden",
            "optimizer_defaults": "existing_builder_defaults_source_bound",
            "post_outcome_changes": "forbidden",
        },
        "initialization policy",
    )
    _expect(packet.get("excluded_search_families"), ["coordinate"], "coordinate exclusion")
    budget = _mapping(packet.get("budget"), "budget")
    _check(
        budget,
        _BUDGET_KEYS,
        (
            EXPECTED_BUDGETS,
            RUN_COUNT,
            SLOT_COUNT,
            SIMULATOR_CALL_BUDGET,
            SIMULATOR_CALL_BUDGET_BY_SEARCH_BUDGET,
            SIMULATOR_INVOCATIONS_PER_SLOT,
            "simulator_invocation",
            "fixed_per_cell_stop_no_padding_or_retry",
            True,
            True,
        ),
        "budget grid",
    )
    seed = _mapping(packet.get("seed_policy"), "seed_policy")
    _check_values(
        seed,
        {
            "search_seeds": EXPECTED_SEARCH_SEEDS,
            "candidate_seed_mode": "index_derived",
            "scenario_seed_domain": {"min": 100, "max": 999},
            "canary_seed_base": CANARY_SEED_BASE,
            "role_seed_base": ROLE_SEED_BASE,
        },
        "search seeds",
    )
    _expect(_mapping(seed.get("replay"), "replay").get("count_per_candidate"), 1, "replay count")
    confirmation = _mapping(seed.get("confirmation"), "confirmation")
    _check(
        confirmation,
        "count_per_candidate threshold",
        (CONFIRMATION_COUNT, "3_of_5_inherited"),
        "confirmation",
    )
    _require(
        CANARY_SEED_BASE > 999
        and ROLE_SEED_BASE > CANARY_SEED_BASE + 6 * (CONFIRMATION_COUNT + 2) - 1,
        "seed role domains overlap",
    )
    accounting = _mapping(packet.get("call_accounting"), "call_accounting")
    _check(
        accounting,
        _ACCOUNTING_KEYS,
        (LEDGER_SCHEMA_VERSION, "simulator_invocations", "search_attempt_slots", True, "reject"),
        "call accounting",
    )
    classes = _mapping(accounting.get("classes"), "call_accounting.classes")
    _require(set(classes) == set(_CALL_RULES), "call classes are incomplete")
    for name in _CALL_RULES:
        class_mapping = _mapping(classes[name], f"call class {name}")
        _check(class_mapping, _LEDGER_KEYS, _CALL_RULES[name], f"call class {name}")
    gates = _mapping(packet.get("gates"), "gates")
    _check_values(
        gates,
        {
            "order": _GATE_ORDER,
            "combined_failure_rule": "all_three_gates_pass",
            "confirmation_threshold": "3_of_5_inherited",
        },
        "gate contract",
    )
    _require(
        all(
            _mapping(gates.get(name), f"gate.{name}").get("required") is True
            for name in gates["order"]
        ),
        "gate requirements",
    )
    monitor = _mapping(packet.get("monitor_contract"), "monitor_contract")
    _check(
        monitor,
        _MONITOR_KEYS,
        (EXPECTED_PROPERTIES, TEMPORAL_SIDECAR_SCHEMA_VERSION, True),
        "property IDs",
    )
    _expect(
        monitor.get("discretization"),
        {
            "dt_s": 0.1,
            "dt_must_equal_evaluation_dt": True,
            "activation_time_from_sample_index": True,
        },
        "discretization",
    )
    analysis = _mapping(packet.get("analysis_contract"), "analysis_contract")
    _check(analysis, _ANALYSIS_KEYS, _ANALYSIS_EXPECTED, "analysis")
    _require(
        analysis.get("resampling_unit") == "search_seed"
        and analysis.get("raw_objective_values_are_failures") is False
        and analysis.get("post_outcome_design_changes") == "forbidden"
        and analysis.get("multiplicity") == "report_every_predeclared_cell_no_posthoc_pooling"
        and analysis.get("deduplication_key") == "normalized_control_hash"
        and analysis.get("duplicate_policy")
        == "retain_lineage_and_exclude_duplicate_from_failure_denominator"
        and analysis.get("missingness_policy")
        == "preserve_missing_invalid_unavailable_failed_and_blocked_as_non_results"
        and set(_SECONDARY_OUTCOMES).issubset(analysis.get("secondary_outcomes", ()))
        and set(_RESULT_CLASSES).issubset(analysis.get("result_classes", ())),
        "analysis outcome contract is incomplete",
    )
    _check(
        _mapping(analysis.get("uncertainty"), "uncertainty"),
        "interval confidence resampling_unit",
        ("newcombe_unpooled_wilson", 0.95, "search_seed"),
        "uncertainty",
    )
    ops = _mapping(packet.get("private_ops"), "private_ops")
    _check_values(
        ops,
        {"execution_authorized": False, "scheduler_submission_allowed": False},
        "private execution boundary",
    )
    estimate = _mapping(ops.get("resource_estimate"), "resource estimate")
    _expect(
        estimate.get("max_simulator_invocations"),
        SIMULATOR_CALL_BUDGET,
        "simulator-call ceiling",
    )
    return {
        "status": "ok",
        "packet_schema": PACKET_SCHEMA_VERSION,
        "run_count": RUN_COUNT,
        "search_attempt_slots": SLOT_COUNT,
        "simulator_call_budget": SIMULATOR_CALL_BUDGET,
        "claim_eligible": False,
        "campaign_execution_allowed": False,
    }


def _seed_for(ordinal: int, offset: int) -> int:
    return ROLE_SEED_BASE + ordinal * (CONFIRMATION_COUNT + 1) + offset


def _identity_slot(run_id: str, attempt_index: int, ordinal: int) -> dict[str, Any]:
    return {
        "candidate_id": "candidate_"
        + canonical_sha256({"run_id": run_id, "attempt_index": attempt_index})[:16],
        "attempt_index": attempt_index,
        "candidate_ordinal": ordinal,
        "replay_seed": _seed_for(ordinal, 0),
        "confirmation_seeds": [
            _seed_for(ordinal, offset) for offset in range(1, CONFIRMATION_COUNT + 1)
        ],
    }


def build_expected_identities(
    packet: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Build deterministic identities without sampling candidates."""
    validate_packet(packet, repo_root=repo_root or Path.cwd())
    scenario = packet["scenario"]
    runs: list[dict[str, Any]] = []
    ordinal = 0
    for objective in EXPECTED_OBJECTIVES:
        for family in EXPECTED_FAMILIES:
            for budget in EXPECTED_BUDGETS:
                for search_seed in EXPECTED_SEARCH_SEEDS:
                    core = {
                        "objective_id": objective,
                        "search_family": family,
                        "budget": budget,
                        "search_seed": search_seed,
                        "scenario_template": scenario["template"],
                        "search_space": scenario["search_space"],
                        "horizon_steps": scenario["horizon_steps"],
                        "dt_s": scenario["dt_s"],
                    }
                    run_id = "run_" + canonical_sha256(core)[:16]
                    slots = [
                        _identity_slot(run_id, index, ordinal + index) for index in range(budget)
                    ]
                    runs.append({**core, "run_id": run_id, "candidate_slots": slots})
                    ordinal += budget
    payload: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "packet_digest": canonical_sha256(packet),
        "run_count": len(runs),
        "candidate_slot_count": ordinal,
        "runs": runs,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def validate_temporal_sidecar(
    sidecar: Mapping[str, Any], packet: Mapping[str, Any], *, candidate_id: str | None = None
) -> None:
    """Validate mechanism metadata, monitor provenance, and admission mode."""
    _require(isinstance(sidecar, Mapping), "temporal sidecar must be a mapping")
    _reject_unknown_keys(sidecar, _SIDECAR_ALLOWED_KEYS, "temporal sidecar")
    _expect(
        sidecar.get("schema_version"),
        TEMPORAL_SIDECAR_SCHEMA_VERSION,
        "temporal sidecar schema",
    )
    status = sidecar.get("status")
    _require(
        status in {"planned", "observed"} and str(sidecar.get("candidate_id", "")).strip(),
        "sidecar identity/status is missing",
    )
    _require(
        sidecar.get("admission_status") in _ADMISSION_STATUSES,
        "sidecar admission status is invalid",
    )
    observed = status == "observed"
    if candidate_id is not None:
        _expect(sidecar.get("candidate_id"), candidate_id, "sidecar candidate identity")
    properties = _list(sidecar.get("properties"), "temporal sidecar.properties")
    _expect(len(properties), len(EXPECTED_PROPERTIES), "sidecar property count")
    _expect(tuple(sidecar.get("property_ids", ())), EXPECTED_PROPERTIES, "sidecar property IDs")
    digest = canonical_sha256(packet)
    for property_id, raw in zip(EXPECTED_PROPERTIES, properties, strict=True):
        item = _mapping(raw, f"sidecar property {property_id}")
        _reject_unknown_keys(
            item, _SIDECAR_PROPERTY_ALLOWED_KEYS, f"sidecar property {property_id}"
        )
        _require(
            item.get("property_id") == property_id
            and "signed_margin" in item
            and "activation_time_s" in item,
            f"sidecar fields missing for {property_id}",
        )
        if observed:
            _finite(item["signed_margin"], f"signed margin for {property_id}")
        if item["activation_time_s"] is not None:
            _finite(item["activation_time_s"], f"activation time for {property_id}", minimum=0)
        _require(
            all(item.get(state) in _GATE_STATE_VALUES for state in _GATE_STATES),
            f"sidecar gate state missing for {property_id}",
        )
        mode = item.get("execution_mode")
        _require(mode in EXPECTED_MODES, f"invalid execution mode for {property_id}")
        if observed and mode != "native":
            _require(
                sidecar.get("admission_status") == "excluded",
                "fallback/degraded sidecar must be excluded",
            )
        provenance = _mapping(item.get("provenance"), f"sidecar provenance {property_id}")
        _reject_unknown_keys(
            provenance,
            _SIDECAR_PROVENANCE_ALLOWED_KEYS,
            f"sidecar provenance {property_id}",
        )
        _require(
            str(provenance.get("source", "")).strip() and provenance.get("packet_digest") == digest,
            f"sidecar packet provenance is missing or mismatched for {property_id}",
        )
    monitor = _mapping(sidecar.get("monitor"), "temporal sidecar.monitor")
    _reject_unknown_keys(monitor, _SIDECAR_MONITOR_ALLOWED_KEYS, "temporal sidecar.monitor")
    _expect(monitor.get("dt_s"), packet["scenario"]["dt_s"], "monitor discretization dt")
    _expect(monitor.get("artifact_only"), False, "monitor-only artifact")
    sample_count = monitor.get("sample_count")
    _require(
        type(sample_count) is int and sample_count >= 0 and (not observed or sample_count > 0),
        "monitor sample_count must be non-negative",
    )
    if not observed:
        _require(
            sidecar.get("admission_status") == "not_admitted"
            and all(
                item["signed_margin"] is None
                and item["activation_time_s"] is None
                and item["execution_mode"] == "synthetic_fixture"
                and all(item[state] == "not_run" for state in _GATE_STATES)
                for item in properties
            ),
            "planned sidecar cannot contain observations",
        )
    if sidecar.get("admission_status") == "confirmed_failure":
        _require(
            observed and sidecar.get("failure_basis") == "independent_confirmation",
            "objective or monitor value cannot establish failure",
        )
        _require(
            all(
                item["execution_mode"] == "native"
                and {item[state] for state in _GATE_STATES} == {"passed"}
                for item in properties
            ),
            "confirmed sidecar gates are incomplete",
        )
        _require(
            any(float(item["signed_margin"]) < 0.0 for item in properties),
            "confirmed sidecar requires a negative signed margin",
        )
        _require(
            all(
                item["activation_time_s"] is not None
                for item in properties
                if float(item["signed_margin"]) < 0.0
            ),
            "confirmed sidecar requires activation time for each violated property",
        )


def validate_call_ledger(
    packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    run_budget_limits: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    """Validate explicit per-cell simulator-call and search-slot accounting.

    ``run_budget_limits`` is supplied by the deterministic identity builder (or
    the disjoint canary builder) so a ledger cannot spend the packet-wide
    ceiling in one cell while claiming a smaller per-cell budget.
    """
    _require(
        isinstance(run_budget_limits, Mapping) and run_budget_limits,
        "per-run budget limits are required",
    )
    for run_id, limits in run_budget_limits.items():
        _require(
            isinstance(run_id, str)
            and isinstance(limits, tuple)
            and len(limits) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in limits)
            and all(value >= 0 for value in limits),
            f"invalid budget limits for {run_id!r}",
        )
    slots: set[tuple[str, int]] = set()
    gates: set[tuple[str, str, Any]] = set()
    calls: set[str] = set()
    counts = dict.fromkeys(_CALL_RULES, 0)
    search_slots_by_run: dict[str, int] = {}
    simulator_invocations_by_run: dict[str, int] = {}
    simulator_invocations = 0
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"ledger[{index}]")
        _forbid_outcomes(row)
        class_name = str(row.get("call_class", ""))
        _require(class_name in _CALL_RULES, f"unknown call class: {class_name}")
        phase, role, consumes_slot, expected_calls = _CALL_RULES[class_name]
        _check(
            row,
            ("schema_version", "phase", "seed_role", "simulator_invocations"),
            (LEDGER_SCHEMA_VERSION, phase, role, expected_calls),
            f"ledger {class_name}",
        )
        _require(
            str(row.get("candidate_id", "")) and str(row.get("run_id", "")),
            "ledger identities are required",
        )
        run_id = str(row["run_id"])
        _require(run_id in run_budget_limits, f"unknown run budget identity: {run_id}")
        seed = row.get("seed")
        _require(
            (role == "none" and seed is None)
            or (role != "none" and isinstance(seed, int) and not isinstance(seed, bool)),
            f"seed is invalid for {class_name}",
        )
        _require(row.get("retry_of") in (None, ""), "hidden retries are forbidden")
        _require(
            row.get("post_outcome_change") is False, "post-outcome packet changes are forbidden"
        )
        if row.get("admission_status") is not None:
            _require(
                row.get("admission_status") in _ADMISSION_STATUSES,
                "ledger admission status is invalid",
            )
        mode = row.get("execution_mode")
        _require(mode in EXPECTED_MODES, f"invalid execution mode: {mode}")
        _require(
            mode == "native" or row.get("admission_status") == "excluded",
            "fallback/degraded rows must be excluded",
        )
        if class_name == "search_invalid_proposal":
            _require(
                row.get("admission_status") in {"invalid", "excluded"},
                "invalid search proposals must be excluded",
            )
        attempt = row.get("attempt_index")
        _require(
            isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0,
            "attempt index is required",
        )
        key = (
            (str(row["run_id"]), attempt)
            if consumes_slot
            else (str(row["candidate_id"]), phase, row.get("seed"))
        )
        target = slots if consumes_slot else gates
        duplicate_message = (
            "duplicate search candidate slot"
            if consumes_slot
            else "duplicate certification/replay/confirmation call"
        )
        _require(key not in target, duplicate_message)
        target.add(key)
        search_slots_by_run[run_id] = search_slots_by_run.get(run_id, 0) + int(consumes_slot)
        if expected_calls:
            call_id = str(row.get("simulator_call_id", ""))
            _require(call_id and call_id not in calls, "simulator call IDs must be unique")
            calls.add(call_id)
            simulator_invocations += expected_calls
            simulator_invocations_by_run[run_id] = (
                simulator_invocations_by_run.get(run_id, 0) + expected_calls
            )
        else:
            _expect(row.get("simulator_call_id"), None, f"simulator ID for {class_name}")
        counts[class_name] += 1
    budget = _mapping(packet.get("budget"), "budget")
    _require(
        simulator_invocations <= budget["simulator_call_budget"],
        "simulator call budget exceeded",
    )
    _require(
        len(slots) <= budget["search_attempt_slots"],
        "search-slot budget exceeded",
    )
    for run_id in search_slots_by_run.keys() | simulator_invocations_by_run.keys():
        search_limit, simulator_limit = run_budget_limits[run_id]
        _require(
            search_slots_by_run[run_id] <= search_limit,
            f"per-run search-slot budget exceeded for {run_id}",
        )
        _require(
            simulator_invocations_by_run.get(run_id, 0) <= simulator_limit,
            f"per-run simulator call budget exceeded for {run_id}",
        )
    return {
        "status": "ok",
        "row_count": len(rows),
        "simulator_invocations": simulator_invocations,
        "class_counts": counts,
        "search_slots": len(slots),
        "search_slots_by_run": search_slots_by_run,
        "simulator_invocations_by_run": simulator_invocations_by_run,
    }


def _validate_result_lineage(
    packet: Mapping[str, Any], groups: Mapping[str, list[Mapping[str, Any]]]
) -> None:
    """Require complete gate records for every non-invalid search candidate."""
    _expect(
        packet["gates"]["confirmation_threshold"],
        "3_of_5_inherited",
        "confirmation threshold",
    )
    for candidate_id, candidate_rows in groups.items():
        search_rows = [row for row in candidate_rows if row.get("phase") == "search"]
        _expect(len(search_rows), 1, f"search lineage for {candidate_id}")
        if search_rows[0].get("call_class") == "search_invalid_proposal":
            _require(
                search_rows[0].get("admission_status") in {"invalid", "excluded"},
                f"invalid search proposals must be excluded for {candidate_id}",
            )
            continue
        for phase, expected_count in (
            ("certification", 1),
            ("replay", 1),
            ("confirmation", CONFIRMATION_COUNT),
        ):
            phase_rows = [row for row in candidate_rows if row.get("phase") == phase]
            _expect(
                len(phase_rows),
                expected_count,
                f"{phase} lineage for {candidate_id}",
            )
            state_key = _LINEAGE_STATE_BY_PHASE[phase]
            _require(
                all(row.get(state_key) in _GATE_STATE_VALUES for row in phase_rows),
                f"{phase} lineage state is missing for {candidate_id}",
            )
        confirmation_rows = [row for row in candidate_rows if row.get("phase") == "confirmation"]
        confirmed = any(
            row.get("admission_status") == "confirmed_failure"
            or (
                isinstance(row.get("temporal_sidecar"), Mapping)
                and row["temporal_sidecar"].get("admission_status") == "confirmed_failure"
            )
            for row in candidate_rows
        )
        if confirmed:
            for phase, state_key in (
                ("certification", "certification_state"),
                ("replay", "replay_state"),
            ):
                _require(
                    all(
                        row.get(state_key) == "passed"
                        for row in candidate_rows
                        if row.get("phase") == phase
                    ),
                    f"{phase} gate must pass before confirmed failure admission for {candidate_id}",
                )
        _require(
            sum(row["independent_seed_state"] == "passed" for row in confirmation_rows)
            >= CONFIRMATION_THRESHOLD
            or not confirmed,
            f"confirmation lineage is below the 3-of-5 threshold for {candidate_id}",
        )


def validate_result_rows(
    packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate result lineage before any result is admitted."""
    expected_identity = build_expected_identities(packet, repo_root=repo_root or Path.cwd())
    if identities is not None:
        _require(isinstance(identities, Mapping), "identities must be a mapping")
        _expect(identities, expected_identity, "deterministic identities")
    identity = expected_identity
    index = {
        slot["candidate_id"]: (slot, run)
        for run in identity["runs"]
        for slot in run["candidate_slots"]
    }
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for raw in rows:
        row = _mapping(raw, "result row")
        _forbid_outcomes(row)
        candidate_id = str(row.get("candidate_id", ""))
        slot, run = index.get(candidate_id, (None, None))
        _require(slot is not None, "unknown or duplicate candidate identity")
        _require(
            row.get("post_outcome_change") is False, "post-outcome packet changes are forbidden"
        )
        _check(
            row,
            "packet_digest identity_sha256 run_id objective_id attempt_index",
            (
                identity["packet_digest"],
                identity["identity_sha256"],
                run["run_id"],
                run["objective_id"],
                slot["attempt_index"],
            ),
            "result identity",
        )
        phase = str(row.get("phase", ""))
        expected = {
            "search": run["search_seed"],
            "replay": slot["replay_seed"],
            "certification": None,
        }
        if phase == "confirmation":
            _require(
                row.get("seed") in slot["confirmation_seeds"], "confirmation seed identity mismatch"
            )
        else:
            _require(phase in expected, f"unknown result phase: {phase}")
            _expect(row.get("seed"), expected[phase], f"{phase} seed identity")
        if (
            run["objective_id"] == "temporal_robustness"
            and row.get("call_class") != "search_invalid_proposal"
        ):
            sidecar = _mapping(row.get("temporal_sidecar"), "temporal_sidecar")
            validate_temporal_sidecar(sidecar, packet, candidate_id=candidate_id)
            if row.get("admission_status") is not None:
                _expect(
                    row.get("admission_status"),
                    sidecar.get("admission_status"),
                    "result/sidecar admission status",
                )
        groups.setdefault(candidate_id, []).append(row)
    _validate_result_lineage(packet, groups)
    run_budget_limits = {
        run["run_id"]: (
            len(run["candidate_slots"]),
            len(run["candidate_slots"]) * SIMULATOR_INVOCATIONS_PER_SLOT,
        )
        for run in identity["runs"]
    }
    ledger = validate_call_ledger(packet, rows, run_budget_limits=run_budget_limits)
    ledger["validated_candidate_rows"] = len({row.get("candidate_id") for row in rows})
    return ledger


def _ledger_row(
    candidate_id: str,
    *,
    phase: str,
    call_class: str,
    seed: int | None,
    attempt_index: int = 0,
) -> dict[str, Any]:
    rule = _CALL_RULES[call_class]
    simulator_invocations, seed_role = rule[3], rule[1]
    return dict(  # noqa: C408 - keyword form keeps the synthetic fixture compact
        schema_version=LEDGER_SCHEMA_VERSION,
        candidate_id=candidate_id,
        run_id=candidate_id,
        attempt_index=attempt_index,
        phase=phase,
        call_class=call_class,
        simulator_invocations=simulator_invocations,
        simulator_call_id=(
            f"canary_{candidate_id}_{phase}_{attempt_index}" if simulator_invocations else None
        ),
        seed=seed,
        seed_role=seed_role,
        execution_mode="synthetic_fixture",
        admission_status="excluded",
        retry_of=None,
        post_outcome_change=False,
    )


def _planned_temporal_sidecar(packet: Mapping[str, Any], candidate_id: str) -> dict[str, Any]:
    digest = canonical_sha256(packet)
    sidecar = {
        "schema_version": TEMPORAL_SIDECAR_SCHEMA_VERSION,
        "status": "planned",
        "candidate_id": candidate_id,
        "property_ids": list(EXPECTED_PROPERTIES),
        "properties": [
            {
                "property_id": property_id,
                "signed_margin": None,
                "activation_time_s": None,
                **dict.fromkeys(_GATE_STATES, "not_run"),
                "execution_mode": "synthetic_fixture",
                "provenance": {"source": "planned_synthetic_fixture", "packet_digest": digest},
            }
            for property_id in EXPECTED_PROPERTIES
        ],
        "monitor": {"dt_s": packet["scenario"]["dt_s"], "sample_count": 0, "artifact_only": False},
        "admission_status": "not_admitted",
    }
    validate_temporal_sidecar(sidecar, packet, candidate_id=candidate_id)
    return sidecar


def build_canary_packet(
    packet: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Build six disjoint planned fixture candidates and ledger rows."""
    root = repo_root or Path.cwd()
    validate_packet(packet, repo_root=root)
    digest = canonical_sha256(packet)
    candidates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    cells = [
        (objective, family) for objective in EXPECTED_OBJECTIVES for family in EXPECTED_FAMILIES
    ]
    for canary_index, (objective, family) in enumerate(cells):
        search_seed = CANARY_SEED_BASE + canary_index
        replay_seed = CANARY_SEED_BASE + 6 + canary_index
        confirmation_seeds = [
            CANARY_SEED_BASE + 12 + canary_index * CONFIRMATION_COUNT + offset
            for offset in range(CONFIRMATION_COUNT)
        ]
        core = {
            "kind": "canary",
            "objective_id": objective,
            "search_family": family,
            "search_seed": search_seed,
        }
        candidate_id = "canary_" + canonical_sha256(core)[:16]
        candidate = {
            **core,
            "candidate_id": candidate_id,
            "replay_seed": replay_seed,
            "confirmation_seeds": confirmation_seeds,
            "status": "planned",
            "evidence_status": "diagnostic_only",
        }
        if objective == "temporal_robustness":
            candidate["temporal_sidecar"] = _planned_temporal_sidecar(packet, candidate_id)
        candidates.append(candidate)
        for phase, class_name, seed in (
            ("search", "search_evaluation", search_seed),
            ("certification", "certification", None),
            ("replay", "deterministic_replay", replay_seed),
        ):
            rows.append(_ledger_row(candidate_id, phase=phase, call_class=class_name, seed=seed))
        rows.extend(
            _ledger_row(
                candidate_id,
                phase="confirmation",
                call_class="independent_confirmation",
                seed=seed,
                attempt_index=offset,
            )
            for offset, seed in enumerate(confirmation_seeds)
        )
    run_budget_limits = {
        candidate["candidate_id"]: (1, SIMULATOR_INVOCATIONS_PER_SLOT) for candidate in candidates
    }
    ledger = validate_call_ledger(packet, rows, run_budget_limits=run_budget_limits)
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "packet_digest": digest,
        "claim_eligible": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ledger": ledger,
        "rows": rows,
    }
