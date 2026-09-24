#!/usr/bin/env python3
"""Fail-closed checker for the issue #6561 pedestrian desired-speed activation preflight.

The checker is deliberately check-only. It validates the disjoint-seed
activation-preflight specification against the frozen #6561 protocol, compiles
the exact preflight identity set without importing a campaign launcher, and can
classify a disjoint-seed diagnostics payload. It never executes a registered
seed, submits compute, or produces benchmark evidence.

Diagnostics payload shape (produced by the private preflight executor)::

    {
        "schema_version": "robot_sf.issue_6561_pedestrian_speed_activation_diagnostics.v1",
        "seeds": [311, 312, 313, 314],
        "rows": [
            {
                "scenario_id": "...",
                "regime_id": "slow_distributed",
                "planner_id": "...",
                "seed": 311,
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
EXPECTED_PROTOCOL_SEMANTIC_HASH = "6dca14e2021394fcfc9116418d23ebd351cba5ffd135abfe84d6c140ee1ced3d"
NOT_EVIDENCE_BANNER = "NOT BENCHMARK EVIDENCE -- DISJOINT-SEED ACTIVATION CHECK ONLY"
EXPECTED_SEEDS = tuple(range(111, 141))
ACTIVATION_CLASSIFICATIONS = (
    "intervention_activated",
    "intervention_inactive",
    "invalid_transient",
    "invalid_initial_speed",
    "not_evaluable",
)
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


def _canonical_hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_preflight(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    protocol_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the preflight specification and its declared protocol."""
    config_file = Path(config_path)
    _require(config_file.is_file(), f"preflight config not found: {config_file}")
    payload = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    payload = _mapping(payload, "preflight config")
    declared_protocol = protocol_path or payload.get("protocol_config")
    _require(isinstance(declared_protocol, str), "protocol_config must be a path string")
    protocol_file = REPO_ROOT / declared_protocol
    _require(protocol_file.is_file(), f"protocol config not found: {declared_protocol}")
    return payload, load_protocol(protocol_file)


def _protocol_regimes(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    regime_contract = _mapping(protocol.get("pedestrian_speed_contract"), "speed contract")
    regimes = regime_contract.get("regimes")
    _require(isinstance(regimes, list) and regimes, "protocol regimes must be a non-empty list")
    return {str(regime["regime_id"]): dict(regime) for regime in regimes}


def _protocol_planners(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    roster = _mapping(protocol.get("planner_contract"), "planner_contract").get("roster")
    _require(isinstance(roster, list) and roster, "protocol planner roster must be non-empty")
    return [dict(planner) for planner in roster]


def _protocol_scenarios(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = _mapping(protocol.get("scenario_contract"), "scenario_contract").get(
        "selected_scenarios"
    )
    _require(isinstance(selected, list) and selected, "protocol scenarios must be non-empty")
    return [dict(scenario) for scenario in selected]


def validate_preflight(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    """Validate every frozen preflight field without executing any row."""
    header = _mapping(payload, "preflight config")
    _require(header.get("schema_version") == SCHEMA_VERSION, "preflight schema drifted")
    _require(header.get("issue") == 6561, "preflight issue must be 6561")
    _require(header.get("child_issue") == 8888, "preflight child issue must be 8888")
    _require(header.get("status") == "preflight_only", "preflight status must be preflight_only")
    _require(
        header.get("protocol_semantic_hash") == EXPECTED_PROTOCOL_SEMANTIC_HASH,
        "preflight declared protocol semantic hash drifted",
    )

    registered = _mapping(header.get("registered_seed_block"), "registered_seed_block")
    _require(
        (registered.get("first"), registered.get("last")) == (111, 140),
        "registered seed block must remain 111-140",
    )
    seed_contract = _mapping(protocol.get("seed_contract"), "seed_contract")
    _require(
        tuple(int(seed) for seed in seed_contract.get("seeds", ())) == EXPECTED_SEEDS,
        "protocol registered seeds drifted from 111-140",
    )

    preflight_block = _mapping(header.get("preflight_seed_block"), "preflight_seed_block")
    raw_seeds = preflight_block.get("seeds")
    _require(isinstance(raw_seeds, list) and raw_seeds, "preflight seeds must be non-empty")
    seeds = [int(seed) for seed in raw_seeds]
    _require(len(seeds) == len(set(seeds)), "preflight seeds must be unique")
    _require(sorted(seeds) == seeds, "preflight seeds must be sorted")
    overlap = sorted(set(seeds) & set(EXPECTED_SEEDS))
    _require(not overlap, f"preflight seeds overlap the registered block: {overlap}")
    _require(
        preflight_block.get("disjoint_from_registered") is True,
        "preflight seeds must declare disjoint_from_registered",
    )

    protocols_regimes = _protocol_regimes(protocol)
    treated = header.get("treated_regimes")
    _require(
        isinstance(treated, list) and len(treated) == 2, "exactly two treated regimes required"
    )
    treated_ids: list[str] = []
    for entry in treated:
        entry = _mapping(entry, "treated regime")
        regime_id = str(entry.get("regime_id"))
        treated_ids.append(regime_id)
        _require(regime_id in protocols_regimes, f"unknown treated regime: {regime_id}")
        _require(
            regime_id != "legacy_default",
            "legacy_default is the reference regime, not a treated intervention",
        )
        runtime = _mapping(protocols_regimes[regime_id].get("runtime_controls"), "runtime controls")
        _require(
            float(entry.get("configured_desired_speed_mean_m_s"))
            == float(runtime.get("desired_speed_mean")),
            f"configured mean drifted for {regime_id}",
        )
        _require(
            float(entry.get("configured_desired_speed_std_m_s"))
            == float(runtime.get("desired_speed_std")),
            f"configured std drifted for {regime_id}",
        )
    _require(len(set(treated_ids)) == 2, "treated regimes must be unique")
    reference = _mapping(header.get("reference_regime"), "reference_regime")
    _require(
        reference.get("regime_id") == "legacy_default", "reference regime must be legacy_default"
    )
    _require(
        reference.get("treatment") == "not_a_treated_intervention",
        "reference regime must not be treated as an intervention",
    )

    scenario_count = len(_protocol_scenarios(protocol))
    planner_count = len(_protocol_planners(protocol))
    _require(
        header.get("expected_preflight_scenario_count")
        == scenario_count
        == protocol["scenario_contract"].get("scenario_count"),
        "preflight scenario count drifted",
    )
    _require(
        header.get("expected_preflight_planner_count")
        == planner_count
        == protocol["planner_contract"].get("planner_count"),
        "preflight planner count drifted",
    )
    _require(header.get("expected_preflight_regime_count") == 2, "preflight regime count must be 2")
    _require(
        header.get("expected_preflight_seed_count") == len(seeds),
        "preflight seed count drifted",
    )
    expected_cells = scenario_count * 2 * planner_count * len(seeds)
    _require(
        header.get("expected_preflight_cell_count") == expected_cells,
        f"preflight cell count must be {expected_cells}",
    )

    rule = _mapping(header.get("activation_rule"), "activation_rule")
    protocol_rule = _mapping(protocol.get("activation_contract"), "activation_contract")
    _require(
        float(rule.get("target_tolerance_m_s")) == float(protocol_rule["target_tolerance_m_s"]),
        "activation tolerance drifted from the protocol",
    )
    _require(
        float(rule.get("minimum_activation_fraction"))
        == float(protocol_rule["minimum_activation_fraction"]),
        "activation fraction threshold drifted from the protocol",
    )
    _require(
        float(rule.get("maximum_spawn_transient_seconds"))
        == float(protocol_rule["maximum_spawn_transient_seconds"]),
        "spawn transient threshold drifted from the protocol",
    )
    spawn = _mapping(protocol["pedestrian_speed_contract"].get("spawn"), "spawn contract")
    _require(
        float(rule.get("initial_spawn_speed_expected_m_s")) == float(spawn["initial_speed_m_s"]),
        "initial spawn speed drifted from the protocol",
    )
    _require(
        tuple(rule.get("diagnostic_fields", ())) == tuple(protocol_rule["required_diagnostics"]),
        "activation diagnostic fields drifted from the protocol",
    )
    _require(
        tuple(rule.get("classifications", ())) == ACTIVATION_CLASSIFICATIONS,
        "activation classifications drifted",
    )
    forbidden = protocol.get("validation_contract", {}).get("forbidden_row_statuses", ())
    _require(
        set(forbidden).issubset(set(rule.get("forbidden_row_statuses", ()))),
        "forbidden row statuses must cover the protocol list",
    )

    validation = _mapping(header.get("validation_contract"), "validation_contract")
    _require(
        validation.get("production_execution_in_this_pr") is False,
        "preflight must not enable production execution",
    )
    _require(
        validation.get("registered_rows_allowed") is False,
        "preflight must not allow registered rows",
    )


def compile_preflight_manifest(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile the exact disjoint-seed preflight identities without executing rows."""
    validate_preflight(payload, protocol)
    scenarios = _protocol_scenarios(protocol)
    planners = _protocol_planners(protocol)
    regimes = _protocol_regimes(protocol)
    treated = [str(entry["regime_id"]) for entry in payload["treated_regimes"]]
    seeds = [int(seed) for seed in payload["preflight_seed_block"]["seeds"]]
    baseline = _mapping(protocol.get("baseline_protocol"), "baseline_protocol")

    identities: list[dict[str, Any]] = []
    for scenario in scenarios:
        for regime_id in treated:
            for planner in planners:
                for seed in seeds:
                    identities.append(
                        {
                            "identity_key": (
                                f"{scenario['scenario_id']}__{regime_id}__"
                                f"{planner['planner_id']}__{seed}"
                            ),
                            "scenario_id": scenario["scenario_id"],
                            "scenario_source_sha256": scenario["source_sha256"],
                            "regime_id": regime_id,
                            "runtime_controls": dict(regimes[regime_id]["runtime_controls"]),
                            "planner_id": planner["planner_id"],
                            "planner_config_sha256": planner["config_sha256"],
                            "seed": seed,
                            "horizon_steps": int(baseline["horizon_steps"]),
                            "dt_seconds": float(baseline["dt_seconds"]),
                            "robot_speed_cap_m_s": float(baseline["robot_speed_cap_m_s"]),
                            "execution_mode": baseline["execution_mode"],
                            "registered": False,
                            "preflight": True,
                        }
                    )
    keys = [row["identity_key"] for row in identities]
    expected_cells = payload["expected_preflight_cell_count"]
    _require(len(identities) == expected_cells, f"preflight identity count is not {expected_cells}")
    _require(len(set(keys)) == len(keys), "duplicate preflight identities")
    _require(
        not ({row["seed"] for row in identities} & set(EXPECTED_SEEDS)),
        "preflight identities must not use registered seeds",
    )
    manifest_hash = _canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "issue": payload["issue"],
            "child_issue": payload["child_issue"],
            "seed_block": payload["preflight_seed_block"],
            "identities": identities,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "banner": NOT_EVIDENCE_BANNER,
        "issue": payload["issue"],
        "child_issue": payload["child_issue"],
        "expected_cell_count": expected_cells,
        "identity_count": len(identities),
        "unique_identity_count": len(set(keys)),
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
    row_status = str(row.get("row_status", ""))
    base = {
        "identity_key": (
            f"{row.get('scenario_id')}__{row.get('regime_id')}__"
            f"{row.get('planner_id')}__{row.get('seed')}"
        ),
        "regime_id": str(row.get("regime_id")),
        "row_status": row_status,
    }
    if row_status in forbidden_statuses:
        return {**base, "classification": "not_evaluable", "reason": f"row_status:{row_status}"}
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {**base, "classification": "not_evaluable", "reason": "diagnostics_missing"}
    missing = [field for field in rule["diagnostic_fields"] if field not in diagnostics]
    if missing:
        return {**base, "classification": "not_evaluable", "reason": f"fields_missing:{missing}"}
    tolerance = float(rule["target_tolerance_m_s"])
    spawn_expected = float(rule["initial_spawn_speed_expected_m_s"])
    spawn_tolerance = float(rule["initial_spawn_speed_tolerance_m_s"])
    spawn_mean = float(diagnostics["initial_spawn_speed_mean_m_s"])
    spawn_peak = float(diagnostics["initial_spawn_speed_peak_m_s"])
    if (
        abs(spawn_mean - spawn_expected) > spawn_tolerance
        or abs(spawn_peak - spawn_expected) > spawn_tolerance
    ):
        return {
            **base,
            "classification": "invalid_initial_speed",
            "reason": f"spawn_mean={spawn_mean}, spawn_peak={spawn_peak}",
        }
    transient = float(diagnostics["acceleration_transient_steps"]) * dt_seconds
    if transient > float(rule["maximum_spawn_transient_seconds"]):
        return {
            **base,
            "classification": "invalid_transient",
            "reason": f"transient_seconds={transient}",
        }
    fraction = float(diagnostics["desired_speed_activation_fraction"])
    time_to_target = float(diagnostics["time_to_desired_speed_target_seconds"])
    configured_mean = float(diagnostics["configured_desired_speed_mean_m_s"])
    realized_mean = float(diagnostics["realized_desired_speed_mean_m_s"])
    if abs(realized_mean - configured_mean) > tolerance:
        return {
            **base,
            "classification": "intervention_inactive",
            "reason": f"mean_offset={abs(realized_mean - configured_mean)}",
        }
    if fraction < float(rule["minimum_activation_fraction"]):
        return {
            **base,
            "classification": "intervention_inactive",
            "reason": f"activation_fraction={fraction}",
        }
    if time_to_target > float(rule["maximum_spawn_transient_seconds"]):
        return {
            **base,
            "classification": "intervention_inactive",
            "reason": f"time_to_target={time_to_target}",
        }
    return {**base, "classification": "intervention_activated", "reason": "activation_rule_passed"}


def classify_activation(
    payload: Mapping[str, Any],
    protocol: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a disjoint-seed diagnostics payload under the frozen activation rule."""
    validate_preflight(payload, protocol)
    rows = diagnostics.get("rows")
    _require(isinstance(rows, list) and rows, "diagnostics rows must be a non-empty list")
    preflight_seeds = {int(seed) for seed in payload["preflight_seed_block"]["seeds"]}
    treated = {str(entry["regime_id"]) for entry in payload["treated_regimes"]}
    rule = _mapping(payload.get("activation_rule"), "activation_rule")
    forbidden = set(rule.get("forbidden_row_statuses", ()))
    dt_seconds = float(
        _mapping(protocol.get("baseline_protocol"), "baseline_protocol")["dt_seconds"]
    )
    classified: list[dict[str, Any]] = []
    for raw_row in rows:
        row = _mapping(raw_row, "diagnostics row")
        seed = int(row.get("seed", -1))
        if seed not in preflight_seeds:
            raise PreflightError(f"diagnostics row uses non-preflight seed: {seed}")
        if str(row.get("regime_id")) not in treated:
            raise PreflightError(f"diagnostics row uses untreated regime: {row.get('regime_id')}")
        classified.append(
            _row_classification(
                row,
                rule=rule,
                forbidden_statuses=forbidden,
                dt_seconds=dt_seconds,
            )
        )
    per_regime: dict[str, dict[str, Any]] = {}
    for regime_id in sorted(treated):
        regime_rows = [entry for entry in classified if entry["regime_id"] == regime_id]
        _require(regime_rows, f"no diagnostics rows for treated regime {regime_id}")
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
    return {
        "schema_version": SCHEMA_VERSION,
        "banner": NOT_EVIDENCE_BANNER,
        "ok": activated,
        "registered_seed_overlap": False,
        "per_regime": per_regime,
        "rows": classified,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--check-only", action="store_true", help="validate and compile only")
    parser.add_argument("--diagnostics", help="JSON diagnostics payload to classify")
    parser.add_argument("--format", choices=("json", "text"), default="json")
    return parser.parse_args()


def main() -> int:
    """Run the check-only validation, manifest compilation, or payload classification."""
    args = _parse_args()
    payload, protocol = load_preflight(args.config)
    if args.diagnostics:
        diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
        result = classify_activation(payload, protocol, diagnostics)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    manifest = compile_preflight_manifest(payload, protocol)
    summary = {key: value for key, value in manifest.items() if key != "identities"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
