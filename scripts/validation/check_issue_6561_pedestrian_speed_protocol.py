#!/usr/bin/env python3
"""Fail-closed checker for the issue #6561 pedestrian-speed protocol.

This checker is deliberately check-only. It validates the frozen protocol and
materializes the exact registered identity set without importing a campaign
launcher, submitting compute, or running an episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_6561_pedestrian_speed_protocol.yaml"
SCHEMA_VERSION = "robot_sf.issue_6561_pedestrian_speed_protocol.v1"
EXPECTED_SCENARIOS = (
    "classic_head_on_corridor_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_merging_medium",
    "classic_overtaking_medium",
    "classic_station_platform_medium",
)
EXPECTED_REGIMES = ("legacy_default", "slow_distributed", "typical_distributed")
EXPECTED_PLANNERS = (
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "ppo",
    "orca",
    "prediction_planner",
)
EXPECTED_SEEDS = tuple(range(111, 141))
EXPECTED_LEDGER_FIELDS = (
    "decision_at",
    "protocol_work_started_at",
    "protocol_frozen_at",
    "preflight_passed_at",
    "slurm_submitted_at",
    "scheduler_started_at",
    "scheduler_completed_at",
    "artifacts_retrieved_at",
    "integrity_validated_at",
    "analysis_completed_at",
    "evidence_frozen_at",
    "dissertation_admission_decided_at",
)
FORBIDDEN_TRANSIENT_KEYS = {
    "host",
    "job_id",
    "queue",
    "scratch_path",
    "target_host",
    "worktree",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"cannot read referenced file {path}: {exc}") from exc


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_no_transient_state(value: Any, path: str = "protocol") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                str(key).lower() not in FORBIDDEN_TRANSIENT_KEYS,
                f"{path}.{key} contains transient routing state",
            )
            _assert_no_transient_state(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_transient_state(child, f"{path}[{index}]")


def _resolve_repo_path(raw_path: str, field: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        raise ValueError(f"{field} must be repository-relative")
    resolved = REPO_ROOT / path
    _require(resolved.exists(), f"{field} does not exist: {raw_path}")
    return resolved


def _source_scenario_ids(path: Path) -> set[str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot parse scenario source {path}: {exc}") from exc
    scenarios = _mapping(payload, str(path)).get("scenarios")
    _require(isinstance(scenarios, list), f"{path} must contain a scenarios list")
    return {
        str(row["name"])
        for row in scenarios
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }


def load_protocol(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the tracked protocol configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read protocol {path}: {exc}") from exc
    _require(isinstance(payload, dict), "protocol must be a mapping")
    validate_protocol(payload)
    return payload


def _validate_header(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema_version drifted")
    _require(payload.get("issue") == 6561, "issue must be 6561")
    _require(payload.get("status") == "protocol_only", "status must remain protocol_only")
    boundary = _mapping(payload.get("execution_boundary"), "execution_boundary")
    for field in (
        "protocol_only_pr",
        "registered_campaign_in_this_pr",
        "compute_submit_authorized",
        "paper_or_dissertation_claim_edits",
        "fallback_or_degraded_success_allowed",
    ):
        _require(
            boundary.get(field) is (True if field == "protocol_only_pr" else False),
            f"execution_boundary.{field} drifted",
        )
    ordering = _mapping(payload.get("ordering_gate"), "ordering_gate")
    _require(ordering.get("robot_speed_campaign_issue") == 6102, "#6102 ordering gate drifted")
    _require(
        ordering.get("production_submission_allowed_in_protocol_pr") is False,
        "production submission must remain disabled in the protocol PR",
    )


def _validate_baseline(payload: dict[str, Any]) -> None:
    baseline = _mapping(payload.get("baseline_protocol"), "baseline_protocol")
    matrix_path = _resolve_repo_path(str(baseline.get("scenario_matrix")), "scenario_matrix")
    _require(
        _sha256(matrix_path) == baseline.get("scenario_matrix_sha256"),
        "scenario matrix hash drifted",
    )
    seed_path = _resolve_repo_path(str(baseline.get("seed_set_path")), "seed_set_path")
    _require(_sha256(seed_path) == baseline.get("seed_set_sha256"), "seed set hash drifted")
    _require(baseline.get("horizon_steps") == 600, "horizon_steps must be 600")
    _require(float(baseline.get("dt_seconds")) == 0.1, "dt_seconds must be 0.1")
    _require(float(baseline.get("robot_speed_cap_m_s")) == 2.0, "robot speed cap must be 2.0")
    _require(baseline.get("execution_mode") == "native", "execution_mode must be native")


def _validate_scenarios(payload: dict[str, Any]) -> None:
    contract = _mapping(payload.get("scenario_contract"), "scenario_contract")
    rows = contract.get("selected_scenarios")
    _require(isinstance(rows, list), "selected_scenarios must be a list")
    _require(
        tuple(row.get("scenario_id") for row in rows) == EXPECTED_SCENARIOS,
        "scenario order or identifiers drifted",
    )
    _require(contract.get("scenario_count") == len(EXPECTED_SCENARIOS), "scenario count drifted")
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, f"selected_scenarios[{index}]")
        source = _resolve_repo_path(str(row.get("source_path")), f"scenario[{index}].source_path")
        _require(
            _sha256(source) == row.get("source_sha256"), f"scenario[{index}] source hash drifted"
        )
        _require(
            row["scenario_id"] in _source_scenario_ids(source),
            f"scenario[{index}] identifier is absent from its source",
        )
    pairing = _mapping(contract.get("pairing"), "scenario pairing")
    for field in (
        "same_scenario_seed_set_across_regimes",
        "same_scenario_seed_set_across_planners",
        "scenario_order_frozen",
        "no_seed_substitution",
    ):
        _require(pairing.get(field) is True, f"scenario pairing.{field} must be true")


def _validate_planners(payload: dict[str, Any]) -> None:
    contract = _mapping(payload.get("planner_contract"), "planner_contract")
    rows = contract.get("roster")
    _require(isinstance(rows, list), "planner roster must be a list")
    _require(
        tuple(row.get("planner_id") for row in rows) == EXPECTED_PLANNERS,
        "planner roster order or identifiers drifted",
    )
    _require(contract.get("planner_count") == len(EXPECTED_PLANNERS), "planner count drifted")
    for index, row_value in enumerate(rows):
        row = _mapping(row_value, f"planner roster[{index}]")
        config_path = row.get("config_path")
        config_hash = row.get("config_sha256")
        if config_path is None:
            _require(row["planner_id"] == "orca", "only ORCA may use the native registry binding")
            _require(config_hash is None, "ORCA config hash must remain null")
            _require(row.get("binding") == "native_orca_registry", "ORCA binding drifted")
            continue
        config = _resolve_repo_path(str(config_path), f"planner[{index}].config_path")
        _require(_sha256(config) == config_hash, f"planner[{index}] config hash drifted")
    ppo = next(row for row in rows if row["planner_id"] == "ppo")
    _require(ppo.get("estimand") == "zero_shot_ood_robustness", "PPO estimand drifted")
    _require(ppo.get("retraining") == "forbidden", "PPO retraining policy drifted")


def _validate_seeds(payload: dict[str, Any]) -> None:
    contract = _mapping(payload.get("seed_contract"), "seed_contract")
    _require(tuple(contract.get("seeds", ())) == EXPECTED_SEEDS, "seed set drifted from 111-140")
    _require(contract.get("seed_count") == len(EXPECTED_SEEDS), "seed count drifted")


def _validate_speed_contract(payload: dict[str, Any]) -> None:
    contract = _mapping(payload.get("pedestrian_speed_contract"), "pedestrian_speed_contract")
    spawn = _mapping(contract.get("spawn"), "spawn")
    _require(
        float(spawn.get("initial_speed_m_s")) == 0.5, "initial spawn speed must remain 0.5 m/s"
    )
    _require(spawn.get("initial_speed_binding") == "released_default", "spawn binding drifted")
    _require(spawn.get("change_forbidden") is True, "spawn change must remain forbidden")
    regimes = contract.get("regimes")
    _require(isinstance(regimes, list), "speed regimes must be a list")
    _require(
        tuple(row.get("regime_id") for row in regimes) == EXPECTED_REGIMES,
        "speed regime order or identifiers drifted",
    )
    _require(contract.get("regime_count") == len(EXPECTED_REGIMES), "regime count drifted")
    expected_controls = {
        "legacy_default": {
            "ped_speed_tier": None,
            "desired_speed_mean": None,
            "desired_speed_std": None,
            "desired_speed_seed": None,
        },
        "slow_distributed": {
            "ped_speed_tier": None,
            "desired_speed_mean": 0.65,
            "desired_speed_std": 0.2,
            "desired_speed_seed": "episode_seed",
        },
        "typical_distributed": {
            "ped_speed_tier": None,
            "desired_speed_mean": 1.3,
            "desired_speed_std": 0.2,
            "desired_speed_seed": "episode_seed",
        },
    }
    for row_value in regimes:
        row = _mapping(row_value, "speed regime")
        controls = _mapping(row.get("runtime_controls"), f"{row.get('regime_id')}.runtime_controls")
        _require(
            controls == expected_controls[row["regime_id"]],
            f"{row['regime_id']} runtime controls drifted",
        )


def _validate_metrics_and_inference(payload: dict[str, Any]) -> None:
    metrics = _mapping(payload.get("metric_contract"), "metric_contract")
    _require(
        metrics.get("primary_metrics") == ["success_rate", "collision_rate", "near_miss_rate"],
        "primary metric contract drifted",
    )
    _require(
        metrics.get("exposure_metrics")
        == [
            "time_to_goal_norm",
            "total_exposure_seconds",
            "travel_distance_m",
            "mean_clearance_m",
            "min_clearance_m",
        ],
        "exposure metric contract drifted",
    )
    _require(
        metrics.get("typed_collision_metrics")
        == [
            "ped_collision_rate",
            "obstacle_collision_rate",
            "agent_collision_rate",
            "unclassified_collision_rate",
        ],
        "typed collision metric contract drifted",
    )
    inference = _mapping(payload.get("inference_contract"), "inference_contract")
    _require(inference.get("comparison") == "regime_minus_legacy_default", "comparison drifted")
    _require(
        inference.get("resampling_unit") == "paired_scenario_seed_block", "resampling unit drifted"
    )
    _require(inference.get("bootstrap_replicates") == 2000, "bootstrap replicate count drifted")
    _require(
        inference.get("multiplicity", {}).get("method") == "holm_bonferroni",
        "multiplicity method drifted",
    )
    _require(
        inference.get("multiplicity", {}).get("tests_per_planner") == 6,
        "multiplicity family size drifted",
    )
    _require(
        inference.get("multiplicity", {}).get("familywise_alpha") == 0.05,
        "familywise alpha drifted",
    )
    _require(
        inference.get("multiplicity", {}).get("directional_family_alpha") == 0.025,
        "directional alpha drifted",
    )
    _require(
        inference.get("harm_margins")
        == {
            "success_rate": -0.05,
            "collision_rate": 0.02,
            "near_miss_rate": 0.05,
        },
        "harm margins drifted",
    )


def _validate_activation_and_ledger(payload: dict[str, Any]) -> None:
    activation = _mapping(payload.get("activation_contract"), "activation_contract")
    required = activation.get("required_diagnostics")
    _require(isinstance(required, list) and len(required) == 9, "activation diagnostics drifted")
    _require(activation.get("target_tolerance_m_s") == 0.2, "activation tolerance drifted")
    _require(
        activation.get("minimum_activation_fraction") == 0.8, "minimum activation fraction drifted"
    )
    _require(
        activation.get("maximum_spawn_transient_seconds") == 2.0, "spawn transient bound drifted"
    )
    ledger = _mapping(payload.get("turnaround_ledger"), "turnaround_ledger")
    _require(
        tuple(ledger.get("required_timestamps", ())) == EXPECTED_LEDGER_FIELDS,
        "turnaround ledger fields drifted",
    )
    _require(ledger.get("retain_failed_attempts") is True, "failed attempts must be retained")
    _require(
        ledger.get("private_scheduler_details_allowed") is False,
        "private scheduler details must remain excluded",
    )


def validate_protocol(payload: dict[str, Any]) -> None:
    """Validate every frozen protocol field without running production code."""
    _validate_header(payload)
    _validate_baseline(payload)
    _validate_scenarios(payload)
    _validate_planners(payload)
    _validate_seeds(payload)
    _validate_speed_contract(payload)
    _validate_metrics_and_inference(payload)
    _validate_activation_and_ledger(payload)
    manifest = _mapping(payload.get("manifest_contract"), "manifest_contract")
    _require(manifest.get("expected_cell_count") == 2160, "manifest cell count must be 2160")
    _require(
        manifest.get("manifest_hash_algorithm") == "sha256_json_canonical_sort_keys",
        "manifest hash algorithm drifted",
    )
    _require(
        manifest.get("registered_rows_allowed_in_protocol_pr") is False,
        "registered rows must remain disabled",
    )
    validation = _mapping(payload.get("validation_contract"), "validation_contract")
    _require(
        validation.get("production_launcher_in_this_pr") is False,
        "production launcher must remain out of the protocol PR",
    )
    _assert_no_transient_state(payload)


def compile_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Compile the exact 2,160 registered identities from a validated protocol."""
    scenarios = _mapping(payload["scenario_contract"], "scenario_contract")["selected_scenarios"]
    regimes = _mapping(payload["pedestrian_speed_contract"], "pedestrian_speed_contract")["regimes"]
    planners = _mapping(payload["planner_contract"], "planner_contract")["roster"]
    seeds = _mapping(payload["seed_contract"], "seed_contract")["seeds"]
    baseline = _mapping(payload["baseline_protocol"], "baseline_protocol")
    identities: list[dict[str, Any]] = []
    for scenario in scenarios:
        for regime in regimes:
            for planner in planners:
                for seed in seeds:
                    row = {
                        "identity_key": (
                            f"{scenario['scenario_id']}__{regime['regime_id']}__"
                            f"{planner['planner_id']}__{seed}"
                        ),
                        "scenario_id": scenario["scenario_id"],
                        "scenario_source_sha256": scenario["source_sha256"],
                        "regime_id": regime["regime_id"],
                        "runtime_controls": dict(regime["runtime_controls"]),
                        "planner_id": planner["planner_id"],
                        "planner_config_sha256": planner["config_sha256"],
                        "seed": int(seed),
                        "horizon_steps": int(baseline["horizon_steps"]),
                        "dt_seconds": float(baseline["dt_seconds"]),
                        "robot_speed_cap_m_s": float(baseline["robot_speed_cap_m_s"]),
                        "execution_mode": baseline["execution_mode"],
                        "registered": True,
                    }
                    identities.append(row)
    keys = [row["identity_key"] for row in identities]
    _require(len(identities) == 2160, "identity count is not 2160")
    _require(len(set(keys)) == len(keys), "duplicate registered identities")
    expected_hash = _canonical_hash(
        {
            "schema_version": payload["schema_version"],
            "study_id": payload["study_id"],
            "identities": identities,
        }
    )
    manifest_contract = _mapping(payload["manifest_contract"], "manifest_contract")
    _require(expected_hash == manifest_contract["manifest_hash"], "manifest hash drifted")
    return {
        "schema_version": payload["schema_version"],
        "study_id": payload["study_id"],
        "expected_cell_count": 2160,
        "identity_count": len(identities),
        "unique_identity_count": len(set(keys)),
        "manifest_hash": expected_hash,
        "identities": identities,
    }


def main() -> int:
    """Validate the protocol and optionally print the complete identity manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--manifest", action="store_true", help="print all frozen identities")
    args = parser.parse_args()
    payload = load_protocol(args.config)
    manifest = compile_manifest(payload)
    if args.manifest:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {key: manifest[key] for key in manifest if key != "identities"}, sort_keys=True
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
