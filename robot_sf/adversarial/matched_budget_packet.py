"""Outcome-free packet, identity, canary, and lineage validators for #8891.

Only packet/source bytes and planned fixture data are read.  No sampler,
planner, simulator, campaign runner, or result file is imported or invoked.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PACKET_SCHEMA_VERSION = "matched-budget-temporal-robustness-packet.v1"
IDENTITY_SCHEMA_VERSION = "matched-budget-temporal-robustness-identities.v1"
LEDGER_SCHEMA_VERSION = "temporal-robustness-call-ledger.v1"
CANARY_SCHEMA_VERSION = "matched-budget-temporal-robustness-canary.v1"
EXPECTED_OBJECTIVES = ("worst_case_snqi", "temporal_robustness")
EXPECTED_FAMILIES = ("random", "optuna", "cmaes")
EXPECTED_BUDGETS = (16, 32, 64)
EXPECTED_SEARCH_SEEDS = (1101, 2202, 3303)
EXPECTED_PROPERTIES = ("clearance", "ttc", "goal", "progress", "collision")
EXPECTED_MODES = ("native", "fallback", "degraded", "unavailable", "synthetic_fixture")
CANARY_SEED_BASE = 8_800_000
ROLE_SEED_BASE = 8_900_000
CONFIRMATION_COUNT = 5
RUN_COUNT = 54
SLOT_COUNT = 2016

_REQUIRED_INPUTS = set(
    "parent_manifest scenario_template search_space objective_registry robustness runner certification replay confirmation".split()
)
_FORBIDDEN_FIELDS = set(
    "outcome objective_value observed_value result_rows simulator_output slurm_job_id job_id target_host submitted_at".split()
)
_GATE_STATES = ("certification_state", "replay_state", "independent_seed_state")
_GATE_ORDER = ("certification", "deterministic_replay", "independent_confirmation")
_GATE_STATE_VALUES = {"not_run", "passed", "failed", "excluded"}
_SECONDARY_OUTCOMES = "first_confirmed_failure_attempt valid_rate invalid_rate simulator_invocations certification_rate replay_rate confirmation_rate monitor_artifact_exclusions missingness result_class".split()
_RESULT_CLASSES = "confirmed_failure null inconclusive invalid unavailable blocked".split()
_PACKET_KEYS = "schema_version issue parent_issue claim_eligible"
_EXECUTION_KEYS = "run_campaign run_simulator submit_slurm registered_search admit_evidence".split()
_BUDGET_KEYS = "budgets run_count search_attempt_slots simulator_call_budget_is_separate no_post_outcome_budget_change"
_ACCOUNTING_KEYS = (
    "schema_version authoritative_counter search_counter one_row_per_attempt_or_call hidden_retries"
)
_LEDGER_KEYS = ("phase", "seed_role", "consumes_search_slot", "simulator_invocations")
_MONITOR_KEYS = "property_ids sidecar_schema monitor_only_excluded"
_ANALYSIS_KEYS = "primary_estimand primary_unit numerator denominator"
_ANALYSIS_EXPECTED = (
    "confirmed_failure_discovery_rate_by_search_budget",
    "candidate_slot",
    "candidate_slot_passing_certification_replay_and_independent_confirmation",
    "scheduled_search_attempt_slots",
)
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


def _input_paths(packet: Mapping[str, Any], root: Path) -> dict[str, Path]:
    inputs = _mapping(_mapping(packet.get("source"), "source").get("inputs"), "source.inputs")
    paths = {}
    for input_id, raw in inputs.items():
        item = _mapping(raw, f"source.inputs.{input_id}")
        path = _repo_file(root, item.get("path"), f"source.inputs.{input_id}.path")
        digest = str(item.get("sha256", "")).lower()
        _expect(hashlib.sha256(path.read_bytes()).hexdigest(), digest, f"source hash {input_id}")
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


def validate_packet(packet: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Validate the source-bound, non-executing packet."""
    _require(isinstance(packet, Mapping), "packet must be a mapping")
    _forbid_outcomes(packet)
    _check(packet, _PACKET_KEYS, (PACKET_SCHEMA_VERSION, 8891, 5326, False), "packet")
    boundary = str(packet.get("claim_boundary", "")).lower()
    _require(
        all(
            p in boundary for p in ("does not run", "no campaign", "no benchmark", "no publication")
        ),
        "claim boundary is incomplete",
    )
    execution = _mapping(packet.get("execution_boundary"), "execution_boundary")
    _check_values(execution, dict.fromkeys(_EXECUTION_KEYS, False), "execution boundary")
    paths = _input_paths(packet, repo_root)
    _require(_REQUIRED_INPUTS.issubset(paths), "source inputs are incomplete")
    _validate_source_semantics(packet, paths)
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
        budget, _BUDGET_KEYS, (EXPECTED_BUDGETS, RUN_COUNT, SLOT_COUNT, True, True), "budget grid"
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
        _check(classes[name], _LEDGER_KEYS, _CALL_RULES[name], f"call class {name}")
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
        monitor, _MONITOR_KEYS, (EXPECTED_PROPERTIES, "robustness-report.v1", True), "property IDs"
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
    _expect(estimate.get("max_simulator_invocations"), SLOT_COUNT * 7, "simulator-call ceiling")
    return {
        "status": "ok",
        "packet_schema": PACKET_SCHEMA_VERSION,
        "run_count": RUN_COUNT,
        "search_attempt_slots": SLOT_COUNT,
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
    _expect(sidecar.get("schema_version"), "robustness-report.v1", "temporal sidecar schema")
    status = sidecar.get("status")
    _require(
        status in {"planned", "observed"} and str(sidecar.get("candidate_id", "")).strip(),
        "sidecar identity/status is missing",
    )
    observed = status == "observed"
    if candidate_id is not None:
        _expect(sidecar.get("candidate_id"), candidate_id, "sidecar candidate identity")
    properties = _list(sidecar.get("properties"), "temporal sidecar.properties")
    _expect(tuple(sidecar.get("property_ids", ())), EXPECTED_PROPERTIES, "sidecar property IDs")
    digest = canonical_sha256(packet)
    for property_id, raw in zip(EXPECTED_PROPERTIES, properties, strict=True):
        item = _mapping(raw, f"sidecar property {property_id}")
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
        _require(
            str(provenance.get("source", "")).strip() and provenance.get("packet_digest") == digest,
            f"sidecar packet provenance is missing or mismatched for {property_id}",
        )
    monitor = _mapping(sidecar.get("monitor"), "temporal sidecar.monitor")
    _expect(monitor.get("dt_s"), packet["scenario"]["dt_s"], "monitor discretization dt")
    _expect(monitor.get("artifact_only"), False, "monitor-only artifact")
    sample_count = monitor.get("sample_count")
    _require(
        type(sample_count) is int and sample_count >= 0 and (not observed or sample_count > 0),
        "monitor sample_count must be non-negative",
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


def validate_call_ledger(
    packet: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate explicit simulator-call and search-slot accounting."""
    slots: set[tuple[str, int]] = set()
    gates: set[tuple[str, str, Any]] = set()
    calls: set[str] = set()
    counts = dict.fromkeys(_CALL_RULES, 0)
    simulator_invocations = 0
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"ledger[{index}]")
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
        mode = row.get("execution_mode")
        _require(mode in EXPECTED_MODES, f"invalid execution mode: {mode}")
        _require(
            mode == "native" or row.get("admission_status") == "excluded",
            "fallback/degraded rows must be excluded",
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
        if expected_calls:
            call_id = str(row.get("simulator_call_id", ""))
            _require(call_id and call_id not in calls, "simulator call IDs must be unique")
            calls.add(call_id)
            simulator_invocations += expected_calls
        else:
            _expect(row.get("simulator_call_id"), None, f"simulator ID for {class_name}")
        counts[class_name] += 1
    return {
        "status": "ok",
        "row_count": len(rows),
        "simulator_invocations": simulator_invocations,
        "class_counts": counts,
        "search_slots": len(slots),
    }


def validate_result_rows(
    packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    identities: Mapping[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Validate result lineage before any result is admitted."""
    identity = identities or build_expected_identities(packet, repo_root=repo_root)
    index = {
        slot["candidate_id"]: (slot, run)
        for run in identity["runs"]
        for slot in run["candidate_slots"]
    }
    for raw in rows:
        row = _mapping(raw, "result row")
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
            validate_temporal_sidecar(
                _mapping(row.get("temporal_sidecar"), "temporal_sidecar"),
                packet,
                candidate_id=candidate_id,
            )
    ledger = validate_call_ledger(packet, rows)
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
        "schema_version": "robustness-report.v1",
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
    ledger = validate_call_ledger(packet, rows)
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "packet_digest": digest,
        "claim_eligible": False,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "ledger": ledger,
        "rows": rows,
    }
