#!/usr/bin/env python3
"""Fail-closed checker for the issue #6561 pedestrian desired-speed activation preflight.

The checker is deliberately check-only. It validates the disjoint-seed
activation-preflight specification against the frozen #6561 protocol, compiles
the exact preflight identity set without importing a campaign launcher, and can
classify a disjoint-seed diagnostics payload. Its diagnostics envelope is a
checker-owned structural integrity contract, not a scientific/native diagnostics
schema. No canonical native owner is currently available, so complete results
remain blocked and not admitted. The checker never executes a registered seed,
submits compute, or produces benchmark evidence.

Diagnostics payload shape (produced by the private preflight executor)::

    {
        "schema_version": "robot_sf.issue_6561_pedestrian_speed_activation_diagnostics.v2",
        "status": "complete",
        "seeds": [311, 312, 313, 314],
        "provenance": {
            "protocol_semantic_hash": "...",
            "preflight_config_semantic_hash": "...",
            "preflight_manifest_hash": "...",
        },
        "rows": [
            {
                "identity_key": "...",
                "scenario_id": "...",
                "scenario_source_sha256": "...",
                "regime_id": "slow_distributed",
                "planner_id": "...",
                "planner_config_sha256": "...",
                "seed": 311,
                "runtime_controls": {"desired_speed_mean": 0.65, "desired_speed_std": 0.2},
                "horizon_steps": 600,
                "dt_seconds": 0.1,
                "robot_speed_cap_m_s": 2.0,
                "execution_mode": "native",
                "registered": false,
                "preflight": true,
                "row_status": "native",
                "diagnostics": {
                    "configured_desired_speed_mean_m_s": 0.65,
                    "configured_desired_speed_std_m_s": 0.2,
                    "realized_desired_speed_mean_m_s": 0.66,
                    "realized_desired_speed_std_m_s": 0.21,
                    "initial_spawn_speed_mean_m_s": 0.5,
                    "initial_spawn_speed_peak_m_s": 0.52,
                    "time_to_desired_speed_target_seconds": 0.8,
                    "acceleration_transient_steps": 5,
                    "desired_speed_activation_fraction": 0.9,
                },
            }
        ],
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from scripts.validation.check_issue_6561_pedestrian_speed_protocol import load_protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_6561_pedestrian_speed_activation_preflight.yaml"
)
SCHEMA_VERSION = "robot_sf.issue_6561_pedestrian_speed_activation_preflight.v1"
DIAGNOSTICS_SCHEMA_VERSION = "robot_sf.issue_6561_pedestrian_speed_activation_diagnostics.v2"
RESULT_SCHEMA_VERSION = "robot_sf.issue_6561_pedestrian_speed_activation_result.v2"
EXPECTED_PROTOCOL_SEMANTIC_HASH = "6dca14e2021394fcfc9116418d23ebd351cba5ffd135abfe84d6c140ee1ced3d"
EXPECTED_PREFLIGHT_CONFIG_SEMANTIC_HASH = (
    "c3a537c4cfde5e0bba3666de44ab5f6bdd6bb1db2a18ebc731e871f818034c1b"
)
EXPECTED_PREFLIGHT_MANIFEST_HASH = (
    "4a5ccdfd2502da4b9033bdd0a5a3141c86b0c12ca94e3ee613a7b526d9815fa9"
)
NOT_EVIDENCE_BANNER = "NOT BENCHMARK EVIDENCE -- DISJOINT-SEED ACTIVATION CHECK ONLY"
EXPECTED_SEEDS = tuple(range(111, 141))
EXPECTED_PREFLIGHT_SEEDS = (311, 312, 313, 314)
EXPECTED_PREFLIGHT_CONFIG_PATH = (
    "configs/benchmarks/issue_6561_pedestrian_speed_activation_preflight.yaml"
)
EXPECTED_PROTOCOL_CONFIG_PATH = "configs/benchmarks/issue_6561_pedestrian_speed_protocol.yaml"
EXPECTED_TREATED_REGIMES = (
    ("slow_distributed", 0.65, 0.2),
    ("typical_distributed", 1.3, 0.2),
)
EXPECTED_INITIAL_SPAWN_SPEED_TOLERANCE_M_S = 0.05
ACTIVATION_CLASSIFICATIONS = (
    "intervention_activated",
    "intervention_inactive",
    "invalid_transient",
    "invalid_initial_speed",
    "not_evaluable",
)
EXPECTED_DIAGNOSTIC_FIELDS = (
    "configured_desired_speed_mean_m_s",
    "configured_desired_speed_std_m_s",
    "realized_desired_speed_mean_m_s",
    "realized_desired_speed_std_m_s",
    "initial_spawn_speed_mean_m_s",
    "initial_spawn_speed_peak_m_s",
    "time_to_desired_speed_target_seconds",
    "acceleration_transient_steps",
    "desired_speed_activation_fraction",
)
EXPECTED_FORBIDDEN_ROW_STATUSES = (
    "missing",
    "duplicate",
    "failed",
    "fallback",
    "degraded",
    "provenance_invalid",
    "intervention_not_activated",
    "not_available",
    "unavailable",
    "blocked",
)
ALLOWED_ROW_STATUSES = frozenset(("native", *EXPECTED_FORBIDDEN_ROW_STATUSES))
DIAGNOSTICS_TERMINAL_STATUSES = frozenset(("complete", "not_available", "failed"))
EXPECTED_PROVENANCE_FIELDS = frozenset(
    {
        "protocol_semantic_hash",
        "preflight_config_semantic_hash",
        "preflight_manifest_hash",
    }
)
EXPECTED_PREFLIGHT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "issue",
        "child_issue",
        "status",
        "protocol_config",
        "protocol_semantic_hash",
        "preflight_config_semantic_hash",
        "preflight_manifest_hash",
        "claim_boundary",
        "registered_seed_block",
        "preflight_seed_block",
        "treated_regimes",
        "reference_regime",
        "expected_preflight_scenario_count",
        "expected_preflight_regime_count",
        "expected_preflight_planner_count",
        "expected_preflight_seed_count",
        "expected_preflight_cell_count",
        "activation_rule",
        "canonical_native_diagnostics_contract",
        "validation_contract",
    }
)
EXPECTED_ROW_FIELDS = frozenset(
    {
        "identity_key",
        "scenario_id",
        "scenario_source_sha256",
        "regime_id",
        "planner_id",
        "planner_config_sha256",
        "seed",
        "runtime_controls",
        "horizon_steps",
        "dt_seconds",
        "robot_speed_cap_m_s",
        "execution_mode",
        "registered",
        "preflight",
        "row_status",
        "diagnostics",
    }
)
CANONICAL_OWNER_UNAVAILABLE_REASON = "canonical_native_activation_diagnostics_owner_unavailable"
_SEVERITY = {
    "intervention_activated": 0,
    "intervention_inactive": 1,
    "invalid_transient": 2,
    "invalid_initial_speed": 3,
    "not_evaluable": 4,
}


class PreflightError(ValueError):
    """Raised when the activation preflight specification or payload is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be a mapping")
    return dict(value)


def _strict_int(value: Any, field: str) -> int:
    _require(type(value) is int, f"{field} must be an integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be a finite number",
    )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise PreflightError(f"{field} must be a finite number") from exc
    _require(math.isfinite(result), f"{field} must be finite")
    return result


def _required_string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value != "", f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, field: str) -> str:
    result = _required_string(value, field)
    _require(re.fullmatch(r"[0-9a-f]{64}", result) is not None, f"{field} must be a SHA-256 hash")
    return result


def _canonical_hash(value: Any) -> str:
    try:
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"cannot canonicalize non-finite or unsupported value: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preflight_config_semantic_hash(payload: Mapping[str, Any]) -> str:
    """Hash the config without its self-referential semantic-hash field."""
    canonical_payload = dict(payload)
    canonical_payload.pop("preflight_config_semantic_hash", None)
    return _canonical_hash(canonical_payload)


def load_preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    protocol_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the preflight specification and its declared protocol."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    _require(config_file.is_file(), f"preflight config not found: {config_file}")
    try:
        payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PreflightError(f"cannot read preflight config {config_file}: {exc}") from exc
    payload = _mapping(payload, "preflight config")
    declared_protocol = protocol_path or payload.get("protocol_config")
    _require(
        isinstance(declared_protocol, (str, Path)),
        "protocol_config must be a path string",
    )
    protocol_file = Path(declared_protocol)
    if not protocol_file.is_absolute():
        protocol_file = REPO_ROOT / protocol_file
    _require(protocol_file.is_file(), f"protocol config not found: {declared_protocol}")
    return payload, load_protocol(protocol_file)


def _protocol_regimes(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    regime_contract = _mapping(protocol.get("pedestrian_speed_contract"), "speed contract")
    regimes = regime_contract.get("regimes")
    _require(isinstance(regimes, list) and regimes, "protocol regimes must be a non-empty list")
    result: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(regimes):
        regime = _mapping(value, f"protocol regimes[{index}]")
        regime_id = _required_string(
            regime.get("regime_id"), f"protocol regimes[{index}].regime_id"
        )
        _require(regime_id not in result, f"duplicate protocol regime: {regime_id}")
        result[regime_id] = regime
    return result


def _protocol_planners(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    roster = _mapping(protocol.get("planner_contract"), "planner_contract").get("roster")
    _require(isinstance(roster, list) and roster, "protocol planner roster must be non-empty")
    result: list[dict[str, Any]] = []
    planner_ids: set[str] = set()
    for index, value in enumerate(roster):
        planner = _mapping(value, f"protocol planner roster[{index}]")
        planner_id = _required_string(
            planner.get("planner_id"), f"protocol planner roster[{index}].planner_id"
        )
        _require(planner_id not in planner_ids, f"duplicate protocol planner: {planner_id}")
        planner_ids.add(planner_id)
        result.append(planner)
    return result


def _protocol_scenarios(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = _mapping(protocol.get("scenario_contract"), "scenario_contract").get(
        "selected_scenarios"
    )
    _require(isinstance(selected, list) and selected, "protocol scenarios must be non-empty")
    result: list[dict[str, Any]] = []
    scenario_ids: set[str] = set()
    for index, value in enumerate(selected):
        scenario = _mapping(value, f"protocol scenarios[{index}]")
        scenario_id = _required_string(
            scenario.get("scenario_id"), f"protocol scenarios[{index}].scenario_id"
        )
        _require(scenario_id not in scenario_ids, f"duplicate protocol scenario: {scenario_id}")
        scenario_ids.add(scenario_id)
        result.append(scenario)
    return result


def validate_preflight(  # noqa: PLR0915 - linear validation of frozen contract fields
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Validate every frozen preflight field without executing any row."""
    header = _mapping(payload, "preflight config")
    _require(
        set(header) == EXPECTED_PREFLIGHT_TOP_LEVEL_KEYS,
        "preflight top-level fields drifted from the frozen contract",
    )
    _require(header.get("schema_version") == SCHEMA_VERSION, "preflight schema drifted")
    _require(header.get("issue") == 6561, "preflight issue must be 6561")
    _require(header.get("child_issue") == 8888, "preflight child issue must be 8888")
    _require(header.get("status") == "preflight_only", "preflight status must be preflight_only")
    _require(
        header.get("protocol_config") == EXPECTED_PROTOCOL_CONFIG_PATH,
        "preflight protocol_config path drifted",
    )
    _require(
        header.get("protocol_semantic_hash") == EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "preflight declared protocol semantic hash drifted",
    )
    _require(
        _canonical_hash(protocol) == EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "complete frozen protocol semantic contract drifted",
    )
    _require(
        header.get("preflight_config_semantic_hash") == EXPECTED_PREFLIGHT_CONFIG_SEMANTIC_HASH,
        "preflight config semantic hash drifted",
    )
    _require(
        _sha256(header.get("preflight_manifest_hash"), "preflight_manifest_hash")
        == EXPECTED_PREFLIGHT_MANIFEST_HASH,
        "preflight manifest hash declaration drifted",
    )
    claim_boundary = _required_string(header.get("claim_boundary"), "claim_boundary").lower()
    _require(
        "not benchmark evidence" in claim_boundary, "claim boundary must forbid benchmark evidence"
    )

    registered = _mapping(header.get("registered_seed_block"), "registered_seed_block")
    _require(
        set(registered) == {"first", "last"},
        "registered_seed_block fields drifted",
    )
    _require(
        (
            _strict_int(registered.get("first"), "registered_seed_block.first"),
            _strict_int(registered.get("last"), "registered_seed_block.last"),
        )
        == (111, 140),
        "registered seed block must remain 111-140",
    )
    seed_contract = _mapping(protocol.get("seed_contract"), "seed_contract")
    protocol_seeds = seed_contract.get("seeds")
    _require(
        isinstance(protocol_seeds, list)
        and tuple(
            _strict_int(seed, f"protocol seed_contract.seeds[{index}]")
            for index, seed in enumerate(protocol_seeds)
        )
        == EXPECTED_SEEDS,
        "protocol registered seeds drifted from 111-140",
    )

    preflight_block = _mapping(header.get("preflight_seed_block"), "preflight_seed_block")
    _require(
        set(preflight_block) == {"source", "seeds", "disjoint_from_registered", "rationale"},
        "preflight_seed_block fields drifted",
    )
    _require(
        preflight_block.get("source") == "disjoint_activation_preflight",
        "preflight seed source drifted",
    )
    raw_seeds = preflight_block.get("seeds")
    typed_seeds = (
        [
            _strict_int(seed, f"preflight_seed_block.seeds[{index}]")
            for index, seed in enumerate(raw_seeds)
        ]
        if isinstance(raw_seeds, list)
        else []
    )
    overlap = sorted(set(typed_seeds) & set(EXPECTED_SEEDS))
    _require(not overlap, f"preflight seeds overlap the registered block: {overlap}")
    _require(
        isinstance(raw_seeds, list) and typed_seeds == list(EXPECTED_PREFLIGHT_SEEDS),
        "preflight seeds must remain exactly 311-314",
    )
    seeds = list(EXPECTED_PREFLIGHT_SEEDS)
    _require(
        preflight_block.get("disjoint_from_registered") is True,
        "preflight seeds must declare disjoint_from_registered",
    )

    protocols_regimes = _protocol_regimes(protocol)
    treated = header.get("treated_regimes")
    _require(
        isinstance(treated, list)
        and tuple(entry.get("regime_id") for entry in treated if isinstance(entry, Mapping))
        == tuple(regime_id for regime_id, _, _ in EXPECTED_TREATED_REGIMES),
        "treated regimes must remain the two frozen intervention regimes in order",
    )
    treated_ids: list[str] = []
    for index, (expected_id, expected_mean, expected_std) in enumerate(EXPECTED_TREATED_REGIMES):
        entry = _mapping(treated[index], f"treated_regimes[{index}]")
        _require(
            set(entry)
            == {
                "regime_id",
                "configured_desired_speed_mean_m_s",
                "configured_desired_speed_std_m_s",
            },
            f"treated_regimes[{index}] fields drifted",
        )
        regime_id = _required_string(entry.get("regime_id"), f"treated_regimes[{index}].regime_id")
        treated_ids.append(regime_id)
        _require(regime_id == expected_id, f"treated regime drifted at index {index}")
        _require(regime_id in protocols_regimes, f"unknown treated regime: {regime_id}")
        configured_mean = _finite_float(
            entry.get("configured_desired_speed_mean_m_s"),
            f"treated_regimes[{index}].configured_desired_speed_mean_m_s",
        )
        configured_std = _finite_float(
            entry.get("configured_desired_speed_std_m_s"),
            f"treated_regimes[{index}].configured_desired_speed_std_m_s",
        )
        _require(
            (configured_mean, configured_std) == (expected_mean, expected_std),
            f"configured activation inputs drifted for {regime_id}",
        )
        runtime = _mapping(protocols_regimes[regime_id].get("runtime_controls"), "runtime controls")
        _require(
            runtime
            == {
                "ped_speed_tier": None,
                "desired_speed_mean": expected_mean,
                "desired_speed_std": expected_std,
                "desired_speed_seed": "episode_seed",
            },
            f"configured mean drifted for {regime_id}",
        )
    _require(
        tuple(treated_ids) == tuple(regime_id for regime_id, _, _ in EXPECTED_TREATED_REGIMES),
        "treated regimes must be unique",
    )
    reference = _mapping(header.get("reference_regime"), "reference_regime")
    _require(
        reference
        == {
            "regime_id": "legacy_default",
            "treatment": "not_a_treated_intervention",
        },
        "reference regime must remain descriptive only",
    )

    scenario_count = len(_protocol_scenarios(protocol))
    planner_count = len(_protocol_planners(protocol))
    _require(
        _strict_int(
            header.get("expected_preflight_scenario_count"),
            "expected_preflight_scenario_count",
        )
        == scenario_count
        == protocol["scenario_contract"].get("scenario_count")
        == 6,
        "preflight scenario count drifted",
    )
    _require(
        _strict_int(
            header.get("expected_preflight_planner_count"),
            "expected_preflight_planner_count",
        )
        == planner_count
        == protocol["planner_contract"].get("planner_count")
        == 4,
        "preflight planner count drifted",
    )
    _require(
        _strict_int(
            header.get("expected_preflight_regime_count"), "expected_preflight_regime_count"
        )
        == 2,
        "preflight regime count must be 2",
    )
    _require(
        _strict_int(header.get("expected_preflight_seed_count"), "expected_preflight_seed_count")
        == len(seeds)
        == 4,
        "preflight seed count drifted",
    )
    expected_cells = scenario_count * 2 * planner_count * len(seeds)
    _require(
        _strict_int(header.get("expected_preflight_cell_count"), "expected_preflight_cell_count")
        == expected_cells
        == 192,
        f"preflight cell count must be {expected_cells}",
    )

    rule = _mapping(header.get("activation_rule"), "activation_rule")
    _require(
        set(rule)
        == {
            "target_tolerance_m_s",
            "minimum_activation_fraction",
            "maximum_spawn_transient_seconds",
            "initial_spawn_speed_expected_m_s",
            "initial_spawn_speed_tolerance_m_s",
            "diagnostic_fields",
            "classifications",
            "forbidden_row_statuses",
        },
        "activation_rule fields drifted",
    )
    protocol_rule = _mapping(protocol.get("activation_contract"), "activation_contract")
    _require(
        _finite_float(rule.get("target_tolerance_m_s"), "target_tolerance_m_s")
        == _finite_float(protocol_rule["target_tolerance_m_s"], "protocol target_tolerance_m_s")
        == 0.2,
        "activation tolerance drifted from the protocol",
    )
    _require(
        _finite_float(rule.get("minimum_activation_fraction"), "minimum_activation_fraction")
        == _finite_float(
            protocol_rule["minimum_activation_fraction"],
            "protocol minimum_activation_fraction",
        )
        == 0.8,
        "activation fraction threshold drifted from the protocol",
    )
    _require(
        _finite_float(
            rule.get("maximum_spawn_transient_seconds"),
            "maximum_spawn_transient_seconds",
        )
        == _finite_float(
            protocol_rule["maximum_spawn_transient_seconds"],
            "protocol maximum_spawn_transient_seconds",
        )
        == 2.0,
        "spawn transient threshold drifted from the protocol",
    )
    spawn = _mapping(protocol["pedestrian_speed_contract"].get("spawn"), "spawn contract")
    _require(
        _finite_float(
            rule.get("initial_spawn_speed_expected_m_s"),
            "initial_spawn_speed_expected_m_s",
        )
        == _finite_float(spawn["initial_speed_m_s"], "protocol initial_speed_m_s")
        == 0.5,
        "initial spawn speed drifted from the protocol",
    )
    _require(
        _finite_float(
            rule.get("initial_spawn_speed_tolerance_m_s"),
            "initial_spawn_speed_tolerance_m_s",
        )
        == EXPECTED_INITIAL_SPAWN_SPEED_TOLERANCE_M_S,
        "initial spawn speed tolerance drifted",
    )
    _require(
        tuple(rule.get("diagnostic_fields", ()))
        == EXPECTED_DIAGNOSTIC_FIELDS
        == tuple(protocol_rule["required_diagnostics"]),
        "activation diagnostic fields drifted from the protocol",
    )
    _require(
        tuple(rule.get("classifications", ())) == ACTIVATION_CLASSIFICATIONS,
        "activation classifications drifted",
    )
    forbidden = protocol.get("validation_contract", {}).get("forbidden_row_statuses", ())
    _require(
        tuple(rule.get("forbidden_row_statuses", ())) == EXPECTED_FORBIDDEN_ROW_STATUSES,
        "activation forbidden row statuses drifted",
    )
    _require(
        set(forbidden).issubset(set(EXPECTED_FORBIDDEN_ROW_STATUSES)),
        "forbidden row statuses must cover the protocol list",
    )

    canonical = _mapping(
        header.get("canonical_native_diagnostics_contract"),
        "canonical_native_diagnostics_contract",
    )
    _require(
        canonical
        == {
            "owner": "unavailable",
            "schema_version": "unavailable",
            "status": "blocked",
            "admission": "not_admitted",
            "availability_status": "not_available",
            "reason_code": CANONICAL_OWNER_UNAVAILABLE_REASON,
        },
        "canonical native diagnostics ownership must remain explicitly blocked",
    )

    validation = _mapping(header.get("validation_contract"), "validation_contract")
    _require(
        set(validation)
        == {
            "check_only_command",
            "classification_command",
            "production_execution_in_this_pr",
            "registered_rows_allowed",
        },
        "validation_contract fields drifted",
    )
    _require(
        validation.get("production_execution_in_this_pr") is False,
        "preflight must not enable production execution",
    )
    _require(
        validation.get("registered_rows_allowed") is False,
        "preflight must not allow registered rows",
    )
    _require(
        _preflight_config_semantic_hash(header) == EXPECTED_PREFLIGHT_CONFIG_SEMANTIC_HASH,
        "preflight config contents drifted from its semantic hash",
    )


def _identity_key(scenario_id: str, regime_id: str, planner_id: str, seed: int) -> str:
    """Return the stable identity key used by the frozen preflight manifest."""
    return f"{scenario_id}__{regime_id}__{planner_id}__{seed}"


def _build_preflight_identities(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build manifest identities after the config and protocol have been validated."""
    scenarios = _protocol_scenarios(protocol)
    planners = _protocol_planners(protocol)
    regimes = _protocol_regimes(protocol)
    treated = [entry["regime_id"] for entry in payload["treated_regimes"]]
    seeds = list(payload["preflight_seed_block"]["seeds"])
    baseline = _mapping(protocol.get("baseline_protocol"), "baseline_protocol")
    horizon_steps = _strict_int(baseline.get("horizon_steps"), "baseline_protocol.horizon_steps")
    dt_seconds = _finite_float(baseline.get("dt_seconds"), "baseline_protocol.dt_seconds")
    robot_speed_cap = _finite_float(
        baseline.get("robot_speed_cap_m_s"),
        "baseline_protocol.robot_speed_cap_m_s",
    )
    execution_mode = _required_string(
        baseline.get("execution_mode"), "baseline_protocol.execution_mode"
    )
    _require(execution_mode == "native", "baseline execution_mode must be native")

    identities: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(scenarios):
        scenario_id = _required_string(
            scenario.get("scenario_id"), f"protocol scenarios[{scenario_index}].scenario_id"
        )
        scenario_hash = _sha256(
            scenario.get("source_sha256"),
            f"protocol scenarios[{scenario_index}].source_sha256",
        )
        for regime_id in treated:
            regime = regimes[regime_id]
            runtime_controls = _mapping(
                regime.get("runtime_controls"),
                f"protocol regime {regime_id}.runtime_controls",
            )
            for planner_index, planner in enumerate(planners):
                planner_id = _required_string(
                    planner.get("planner_id"),
                    f"protocol planner roster[{planner_index}].planner_id",
                )
                planner_hash = planner.get("config_sha256")
                if planner_hash is not None:
                    planner_hash = _sha256(
                        planner_hash,
                        f"protocol planner roster[{planner_index}].config_sha256",
                    )
                for seed in seeds:
                    identities.append(
                        {
                            "identity_key": _identity_key(scenario_id, regime_id, planner_id, seed),
                            "scenario_id": scenario_id,
                            "scenario_source_sha256": scenario_hash,
                            "regime_id": regime_id,
                            "runtime_controls": dict(runtime_controls),
                            "planner_id": planner_id,
                            "planner_config_sha256": planner_hash,
                            "seed": seed,
                            "horizon_steps": horizon_steps,
                            "dt_seconds": dt_seconds,
                            "robot_speed_cap_m_s": robot_speed_cap,
                            "execution_mode": execution_mode,
                            "registered": False,
                            "preflight": True,
                        }
                    )
    return identities


def _manifest_hash(payload: Mapping[str, Any], identities: list[dict[str, Any]]) -> str:
    return _canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "issue": payload["issue"],
            "child_issue": payload["child_issue"],
            "seed_block": payload["preflight_seed_block"],
            "identities": identities,
        }
    )


def compile_preflight_manifest(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the exact disjoint-seed preflight identities without executing any row."""
    validate_preflight(payload, protocol)
    identities = _build_preflight_identities(payload, protocol)
    keys = [row["identity_key"] for row in identities]
    expected_cells = _strict_int(
        payload["expected_preflight_cell_count"], "expected_preflight_cell_count"
    )
    _require(len(identities) == expected_cells, f"preflight identity count is not {expected_cells}")
    _require(len(set(keys)) == len(keys), "duplicate preflight identities")
    _require(
        not ({row["seed"] for row in identities} & set(EXPECTED_SEEDS)),
        "preflight identities must not use registered seeds",
    )
    manifest_hash = _manifest_hash(payload, identities)
    _require(
        manifest_hash == payload["preflight_manifest_hash"] == EXPECTED_PREFLIGHT_MANIFEST_HASH,
        "compiled preflight manifest hash drifted",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "banner": NOT_EVIDENCE_BANNER,
        "issue": payload["issue"],
        "child_issue": payload["child_issue"],
        "expected_cell_count": expected_cells,
        "identity_count": len(identities),
        "unique_identity_count": len(set(keys)),
        "protocol_semantic_hash": payload["protocol_semantic_hash"],
        "preflight_config_semantic_hash": payload["preflight_config_semantic_hash"],
        "preflight_manifest_hash": manifest_hash,
        "identities": identities,
    }


def _row_classification(
    row: Mapping[str, Any],
    *,
    rule: Mapping[str, Any],
    forbidden_statuses: set[str],
    dt_seconds: float,
) -> dict[str, Any]:
    row_status = row["row_status"]
    base = {
        "identity_key": row["identity_key"],
        "regime_id": row["regime_id"],
        "row_status": row_status,
    }
    if row_status in forbidden_statuses:
        return {**base, "classification": "not_evaluable", "reason": f"row_status:{row_status}"}
    diagnostics = _mapping(row["diagnostics"], "native diagnostics")
    tolerance = _finite_float(rule["target_tolerance_m_s"], "target_tolerance_m_s")
    spawn_expected = _finite_float(
        rule["initial_spawn_speed_expected_m_s"],
        "initial_spawn_speed_expected_m_s",
    )
    spawn_tolerance = _finite_float(
        rule["initial_spawn_speed_tolerance_m_s"],
        "initial_spawn_speed_tolerance_m_s",
    )
    spawn_mean = _finite_float(
        diagnostics["initial_spawn_speed_mean_m_s"],
        "initial_spawn_speed_mean_m_s",
    )
    spawn_peak = _finite_float(
        diagnostics["initial_spawn_speed_peak_m_s"],
        "initial_spawn_speed_peak_m_s",
    )
    if (
        abs(spawn_mean - spawn_expected) > spawn_tolerance
        or abs(spawn_peak - spawn_expected) > spawn_tolerance
    ):
        return {
            **base,
            "classification": "invalid_initial_speed",
            "reason": f"spawn_mean={spawn_mean}, spawn_peak={spawn_peak}",
        }
    transient_steps = _strict_int(
        diagnostics["acceleration_transient_steps"],
        "acceleration_transient_steps",
    )
    try:
        transient = transient_steps * dt_seconds
    except OverflowError as exc:
        raise PreflightError(str(exc)) from exc
    if transient > _finite_float(
        rule["maximum_spawn_transient_seconds"],
        "maximum_spawn_transient_seconds",
    ):
        return {
            **base,
            "classification": "invalid_transient",
            "reason": f"transient_seconds={transient}",
        }
    fraction = _finite_float(
        diagnostics["desired_speed_activation_fraction"],
        "desired_speed_activation_fraction",
    )
    time_to_target = _finite_float(
        diagnostics["time_to_desired_speed_target_seconds"],
        "time_to_desired_speed_target_seconds",
    )
    configured_mean = _finite_float(
        diagnostics["configured_desired_speed_mean_m_s"],
        "configured_desired_speed_mean_m_s",
    )
    realized_mean = _finite_float(
        diagnostics["realized_desired_speed_mean_m_s"],
        "realized_desired_speed_mean_m_s",
    )
    if abs(realized_mean - configured_mean) > tolerance:
        return {
            **base,
            "classification": "intervention_inactive",
            "reason": f"mean_offset={abs(realized_mean - configured_mean)}",
        }
    if fraction < _finite_float(
        rule["minimum_activation_fraction"],
        "minimum_activation_fraction",
    ):
        return {
            **base,
            "classification": "intervention_inactive",
            "reason": f"activation_fraction={fraction}",
        }
    if time_to_target > _finite_float(
        rule["maximum_spawn_transient_seconds"],
        "maximum_spawn_transient_seconds",
    ):
        return {
            **base,
            "classification": "intervention_inactive",
            "reason": f"time_to_target={time_to_target}",
        }
    return {**base, "classification": "intervention_activated", "reason": "activation_rule_passed"}


def _validate_diagnostic_values(
    row: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> None:
    """Validate finite diagnostics and bind configured inputs to one manifest identity."""
    diagnostics = _mapping(row.get("diagnostics"), "diagnostics")
    _require(
        set(diagnostics) == set(EXPECTED_DIAGNOSTIC_FIELDS),
        f"diagnostics fields drifted for {expected_identity['identity_key']}",
    )
    expected_controls = _mapping(expected_identity["runtime_controls"], "expected runtime controls")
    configured_mean = _finite_float(
        diagnostics["configured_desired_speed_mean_m_s"],
        "configured_desired_speed_mean_m_s",
    )
    configured_std = _finite_float(
        diagnostics["configured_desired_speed_std_m_s"],
        "configured_desired_speed_std_m_s",
    )
    _require(
        configured_mean
        == _finite_float(
            expected_controls.get("desired_speed_mean"), "expected desired_speed_mean"
        ),
        f"configured desired-speed mean drifted for {expected_identity['identity_key']}",
    )
    _require(
        configured_std
        == _finite_float(expected_controls.get("desired_speed_std"), "expected desired_speed_std"),
        f"configured desired-speed std drifted for {expected_identity['identity_key']}",
    )
    realized_mean = _finite_float(
        diagnostics["realized_desired_speed_mean_m_s"],
        "realized_desired_speed_mean_m_s",
    )
    realized_std = _finite_float(
        diagnostics["realized_desired_speed_std_m_s"],
        "realized_desired_speed_std_m_s",
    )
    spawn_mean = _finite_float(
        diagnostics["initial_spawn_speed_mean_m_s"],
        "initial_spawn_speed_mean_m_s",
    )
    spawn_peak = _finite_float(
        diagnostics["initial_spawn_speed_peak_m_s"],
        "initial_spawn_speed_peak_m_s",
    )
    time_to_target = _finite_float(
        diagnostics["time_to_desired_speed_target_seconds"],
        "time_to_desired_speed_target_seconds",
    )
    transient_steps = _strict_int(
        diagnostics["acceleration_transient_steps"],
        "acceleration_transient_steps",
    )
    activation_fraction = _finite_float(
        diagnostics["desired_speed_activation_fraction"],
        "desired_speed_activation_fraction",
    )
    _require(configured_std >= 0.0, "configured desired-speed std is out of range")
    _require(realized_mean >= 0.0, "realized desired-speed mean is out of range")
    _require(realized_std >= 0.0, "realized desired-speed std is out of range")
    _require(spawn_mean >= 0.0, "initial spawn speed mean is out of range")
    _require(
        spawn_peak >= 0.0 and spawn_peak >= spawn_mean,
        "initial spawn speed peak is out of range",
    )
    _require(time_to_target >= 0.0, "time to desired-speed target is out of range")
    _require(transient_steps >= 0, "acceleration transient steps are out of range")
    _require(0.0 <= activation_fraction <= 1.0, "activation fraction is out of range")


def _validate_row(
    row: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
) -> str:
    """Validate one exact manifest row before applying activation classifications."""
    _require(
        set(row) == EXPECTED_ROW_FIELDS,
        f"diagnostics row fields drifted for {expected_identity['identity_key']}",
    )
    _require(
        _required_string(row.get("identity_key"), "diagnostics row.identity_key")
        == expected_identity["identity_key"],
        f"identity_key does not match the frozen manifest: {row.get('identity_key')!r}",
    )
    for field in ("scenario_id", "regime_id", "planner_id", "execution_mode"):
        _require(
            _required_string(row.get(field), f"diagnostics row.{field}")
            == expected_identity[field],
            f"diagnostics row.{field} does not match the frozen manifest",
        )
    _require(
        _sha256(row.get("scenario_source_sha256"), "diagnostics row.scenario_source_sha256")
        == expected_identity["scenario_source_sha256"],
        f"scenario source hash drifted for {expected_identity['identity_key']}",
    )
    row_planner_hash = row.get("planner_config_sha256")
    if expected_identity["planner_config_sha256"] is not None:
        row_planner_hash = _sha256(row_planner_hash, "diagnostics row.planner_config_sha256")
    _require(
        row_planner_hash == expected_identity["planner_config_sha256"],
        f"planner config hash drifted for {expected_identity['identity_key']}",
    )
    _require(
        _strict_int(row.get("seed"), "diagnostics row.seed") == expected_identity["seed"],
        f"seed drifted for {expected_identity['identity_key']}",
    )
    _require(
        _mapping(row.get("runtime_controls"), "diagnostics row.runtime_controls")
        == expected_identity["runtime_controls"],
        f"runtime controls drifted for {expected_identity['identity_key']}",
    )
    _require(
        _strict_int(row.get("horizon_steps"), "diagnostics row.horizon_steps")
        == expected_identity["horizon_steps"],
        f"horizon drifted for {expected_identity['identity_key']}",
    )
    _require(
        _finite_float(row.get("dt_seconds"), "diagnostics row.dt_seconds")
        == expected_identity["dt_seconds"],
        f"dt drifted for {expected_identity['identity_key']}",
    )
    _require(
        _finite_float(row.get("robot_speed_cap_m_s"), "diagnostics row.robot_speed_cap_m_s")
        == expected_identity["robot_speed_cap_m_s"],
        f"robot speed cap drifted for {expected_identity['identity_key']}",
    )
    _require(row.get("registered") is False, "activation diagnostics row cannot be registered")
    _require(row.get("preflight") is True, "activation diagnostics row must be preflight")
    row_status = _required_string(row.get("row_status"), "diagnostics row.row_status")
    _require(row_status in ALLOWED_ROW_STATUSES, f"unsupported row_status: {row_status!r}")
    if row_status == "native":
        _validate_diagnostic_values(row, expected_identity)
    elif row.get("diagnostics") is not None:
        # A forbidden row may carry diagnostics for debugging, but those values
        # still belong to the same checker-owned structural contract. Do not let
        # malformed or non-finite values hide behind a non-admitted status.
        _validate_diagnostic_values(row, expected_identity)
    return row_status


def _validate_diagnostics_envelope(
    diagnostics: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    """Validate the checker-owned envelope without asserting an unowned native schema."""
    envelope = _mapping(diagnostics, "diagnostics payload")
    status = _required_string(envelope.get("status"), "diagnostics.status")
    _require(
        status in DIAGNOSTICS_TERMINAL_STATUSES,
        f"unsupported diagnostics status: {status!r}",
    )
    required_fields = {"schema_version", "status", "seeds", "provenance"}
    expected_fields = required_fields | ({"rows"} if status == "complete" else {"reason"})
    _require(
        set(envelope) == expected_fields,
        "diagnostics payload fields do not match its declared status",
    )
    _require(
        envelope.get("schema_version") == DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostics schema_version is unsupported",
    )
    raw_seeds = envelope.get("seeds")
    _require(
        isinstance(raw_seeds, list)
        and [
            _strict_int(seed, f"diagnostics.seeds[{index}]") for index, seed in enumerate(raw_seeds)
        ]
        == list(EXPECTED_PREFLIGHT_SEEDS),
        "diagnostics seeds must match the frozen preflight seed block exactly",
    )
    provenance = _mapping(envelope.get("provenance"), "diagnostics.provenance")
    _require(
        set(provenance) == EXPECTED_PROVENANCE_FIELDS,
        "diagnostics provenance fields do not match the checker contract",
    )
    _require(
        _sha256(provenance.get("protocol_semantic_hash"), "provenance.protocol_semantic_hash")
        == payload["protocol_semantic_hash"],
        "diagnostics protocol provenance drifted",
    )
    _require(
        _sha256(
            provenance.get("preflight_config_semantic_hash"),
            "provenance.preflight_config_semantic_hash",
        )
        == payload["preflight_config_semantic_hash"],
        "diagnostics preflight config provenance drifted",
    )
    _require(
        _sha256(provenance.get("preflight_manifest_hash"), "provenance.preflight_manifest_hash")
        == manifest["preflight_manifest_hash"],
        "diagnostics preflight manifest provenance drifted",
    )
    if status != "complete":
        _required_string(envelope.get("reason"), f"diagnostics.{status}.reason")
    return status


def _activation_verdict(per_regime: Mapping[str, Mapping[str, Any]]) -> str:
    classifications = {entry["classification"] for entry in per_regime.values()}
    if classifications == {"intervention_activated"}:
        return "activation_pass"
    if classifications & {"invalid_transient", "invalid_initial_speed", "not_evaluable"}:
        return "invalid_incomplete"
    return "intervention_not_activated"


def _result(
    *,
    status: str,
    input_status: str,
    reason_code: str,
    reason: str,
    activation_ok: bool = False,
    per_regime: Mapping[str, Any] | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a result that cannot be mistaken for admitted benchmark evidence."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "banner": NOT_EVIDENCE_BANNER,
        "ok": False,
        "activation_ok": activation_ok,
        "activation_verdict": (
            _activation_verdict(per_regime) if per_regime else "invalid_incomplete"
        ),
        "status": status,
        "input_status": input_status,
        "admission_status": "not_admitted",
        "availability_status": status if status in {"failed", "not_available"} else "not_available",
        "benchmark_success": False,
        "canonical_native_diagnostics_owner": "unavailable",
        "canonical_native_diagnostics_status": "blocked",
        "reason_code": reason_code,
        "reason": reason,
        "registered_seed_overlap": False,
        "per_regime": dict(per_regime or {}),
        "rows": list(rows or []),
    }


def classify_activation(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a disjoint-seed diagnostics payload under the frozen activation rule."""
    validate_preflight(payload, protocol)
    manifest = compile_preflight_manifest(payload, protocol)
    status = _validate_diagnostics_envelope(
        diagnostics,
        payload=payload,
        manifest=manifest,
    )
    if status != "complete":
        envelope = _mapping(diagnostics, "diagnostics payload")
        return _result(
            status=status,
            input_status=status,
            reason_code=f"diagnostics_{status}",
            reason=envelope["reason"],
        )
    envelope = _mapping(diagnostics, "diagnostics payload")
    rows = envelope.get("rows")
    _require(isinstance(rows, list) and rows, "diagnostics rows must be a non-empty list")
    expected_by_key = {row["identity_key"]: row for row in manifest["identities"]}
    rule = _mapping(payload.get("activation_rule"), "activation_rule")
    forbidden = set(rule.get("forbidden_row_statuses", ()))
    dt_seconds = _finite_float(
        _mapping(protocol.get("baseline_protocol"), "baseline_protocol")["dt_seconds"],
        "baseline_protocol.dt_seconds",
    )
    classified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, "diagnostics row")
        row_key = _required_string(
            row.get("identity_key"),
            f"diagnostics row[{index}].identity_key",
        )
        if row_key in seen:
            raise PreflightError(f"duplicate diagnostics identity: {row_key}")
        if row_key not in expected_by_key:
            raise PreflightError(f"unknown diagnostics identity: {row_key}")
        seen.add(row_key)
        _validate_row(row, expected_by_key[row_key])
        classified.append(
            _row_classification(
                row,
                rule=rule,
                forbidden_statuses=forbidden,
                dt_seconds=dt_seconds,
            )
        )
    missing = sorted(set(expected_by_key) - seen)
    _require(
        not missing,
        "diagnostics identities are incomplete; "
        f"missing_count={len(missing)}, first_missing={missing[:3]}",
    )
    _require(
        len(seen) == len(expected_by_key),
        "diagnostics identity count does not match the frozen preflight manifest",
    )
    treated = tuple(entry["regime_id"] for entry in payload["treated_regimes"])
    per_regime: dict[str, dict[str, Any]] = {}
    for regime_id in treated:
        regime_rows = [entry for entry in classified if entry["regime_id"] == regime_id]
        _require(
            len(regime_rows) == 96,
            f"diagnostics rows are incomplete for treated regime {regime_id}",
        )
        worst = max(
            regime_rows,
            key=lambda entry: _SEVERITY[entry["classification"]],
        )
        per_regime[regime_id] = {
            "classification": worst["classification"],
            "row_count": len(regime_rows),
            "worst_identity": worst["identity_key"],
            "reason": worst["reason"],
        }
    activated = all(
        entry["classification"] == "intervention_activated" for entry in per_regime.values()
    )
    return _result(
        status="blocked",
        input_status="complete",
        reason_code=CANONICAL_OWNER_UNAVAILABLE_REASON,
        reason=(
            "No canonical native activation-diagnostics producer or schema owner is available; "
            "the structural classifier result is not admitted."
        ),
        activation_ok=activated,
        per_regime=per_regime,
        rows=classified,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check-only", action="store_true", help="validate and compile only")
    parser.add_argument("--diagnostics", help="JSON diagnostics payload to classify")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args()


def _error_result(exc: Exception) -> dict[str, Any]:
    """Render malformed input as a structured non-success result for the CLI."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "banner": NOT_EVIDENCE_BANNER,
        "ok": False,
        "activation_ok": False,
        "activation_verdict": "invalid_incomplete",
        "status": "failed",
        "input_status": "failed",
        "admission_status": "not_admitted",
        "availability_status": "failed",
        "benchmark_success": False,
        "canonical_native_diagnostics_owner": "unavailable",
        "canonical_native_diagnostics_status": "blocked",
        "reason_code": "invalid_diagnostics_contract",
        "reason": str(exc),
        "error": str(exc),
        "registered_seed_overlap": None,
        "per_regime": {},
        "rows": [],
    }


def _emit(payload: Mapping[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"status={payload.get('status')} reason={payload.get('reason')}")


def main() -> int:
    """Run the check-only validation, manifest compilation, or payload classification."""
    args = _parse_args()
    try:
        payload, protocol = load_preflight(args.config)
        if args.diagnostics:
            diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
            result = classify_activation(payload, protocol, diagnostics)
            _emit(result, args.format)
            return 0 if result["ok"] else 1
        manifest = compile_preflight_manifest(payload, protocol)
        summary = {key: value for key, value in manifest.items() if key != "identities"}
        _emit(summary, args.format)
        return 0
    except (OSError, OverflowError, ValueError) as exc:
        _emit(_error_result(exc), args.format)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
