#!/usr/bin/env python3
"""Fail-closed checker for the issue #5578 robot speed-tier preregistration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs/benchmarks/issue_5578_robot_speed_tier_preregistration.yaml"
SCHEMA_VERSION = "robot_sf.issue_5578_robot_speed_tier_preregistration.v1"
EXPECTED_SEEDS = list(range(111, 141))
EXPECTED_SCENARIOS = {
    "classic_head_on_corridor_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_merging_medium",
    "classic_overtaking_medium",
    "classic_station_platform_medium",
}
EXPECTED_PLANNERS = {
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "ppo",
    "orca",
    "prediction_planner",
}
FORBIDDEN_ROUTING_KEYS = {
    "host",
    "target_host",
    "slurm",
    "job_id",
    "packet_lineage",
    "queue_route",
    "worktree",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be a mapping")
    return value


def _finite_number(
    value: Any, field: str, *, positive: bool = False, allow_negative: bool = False
) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} must be numeric"
    )
    number = float(value)
    _require(math.isfinite(number), f"{field} must be finite")
    if positive:
        _require(number > 0.0, f"{field} must be positive")
    elif not allow_negative:
        _require(number >= 0.0, f"{field} must be non-negative")
    return number


def _assert_no_transient_routing_state(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                str(key).lower() not in FORBIDDEN_ROUTING_KEYS,
                f"{path}.{key} contains transient queue/host routing state",
            )
            _assert_no_transient_routing_state(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_transient_routing_state(child, f"{path}[{index}]")


def _scenario_ids(path: Path) -> set[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenarios = _mapping(payload, str(path)).get("scenarios")
    _require(isinstance(scenarios, list), f"{path} must declare a scenarios list")
    return {str(row.get("name")) for row in scenarios if isinstance(row, dict)}


def load_preregistration(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the tracked issue #5578 preregistration."""
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read preregistration {path}: {exc}") from exc
    _require(isinstance(payload, dict), "preregistration must be a mapping")
    validate_preregistration(payload)
    return payload


def _validate_header(payload: dict[str, Any]) -> None:
    """Validate identity and fail-closed execution boundaries."""
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema_version drifted")
    _require(payload.get("issue") == 5578, "issue must be 5578")
    _require(payload.get("governance_issue") == 5557, "governance_issue must reference #5557")
    _require(payload.get("status") == "preregistration", "status must remain preregistration")
    boundary = _mapping(payload.get("execution_boundary"), "execution_boundary")
    for key in (
        "full_benchmark_campaign_in_this_pr",
        "compute_submit_authorized",
        "paper_claim_edits",
        "metric_semantics_changes",
        "fallback_or_degraded_success_allowed",
    ):
        _require(boundary.get(key) is False, f"execution_boundary.{key} must be false")


def _validate_baseline(payload: dict[str, Any]) -> None:
    """Validate the release-aligned horizon, timestep, and seed-set references."""
    baseline = _mapping(payload.get("baseline_protocol"), "baseline_protocol")
    for field in ("release_reference", "scenario_matrix"):
        _require(
            isinstance(baseline.get(field), str) and baseline[field],
            f"baseline_protocol.{field} required",
        )
        _require(
            (REPO_ROOT / baseline[field]).is_file(), f"missing baseline path: {baseline[field]}"
        )
    _require(
        isinstance(baseline.get("seed_set_name"), str), "baseline_protocol.seed_set_name required"
    )
    _require(
        baseline.get("seed_set_name") == "paper_eval_s30", "release seed set must be paper_eval_s30"
    )
    _require(baseline.get("horizon_steps") == 600, "horizon_steps must be 600")
    _require(baseline.get("dt_seconds") == 0.1, "dt_seconds must be 0.1")


def _validate_scenarios(payload: dict[str, Any]) -> None:
    """Validate the fixed six-row middle-band scenario subset."""
    scenario_contract = _mapping(payload.get("scenario_contract"), "scenario_contract")
    selected = scenario_contract.get("selected_scenarios")
    _require(isinstance(selected, list), "scenario_contract.selected_scenarios must be a list")
    _require(len(selected) == 6, "exactly six scenarios are required")
    selected_ids: set[str] = set()
    for index, row_value in enumerate(selected):
        row = _mapping(row_value, f"selected_scenarios[{index}]")
        scenario_id = row.get("scenario_id")
        source_path = row.get("source_path")
        _require(
            isinstance(scenario_id, str) and scenario_id,
            f"selected_scenarios[{index}].scenario_id required",
        )
        _require(scenario_id not in selected_ids, f"duplicate scenario: {scenario_id}")
        selected_ids.add(scenario_id)
        _require(
            isinstance(source_path, str) and source_path, f"{scenario_id}.source_path required"
        )
        source = REPO_ROOT / source_path
        _require(source.is_file(), f"missing scenario source: {source_path}")
        _require(scenario_id in _scenario_ids(source), f"{scenario_id} missing from {source_path}")
    _require(
        selected_ids == EXPECTED_SCENARIOS,
        "scenario subset drifted from the six declared middle-band rows",
    )
    pairing = _mapping(scenario_contract.get("fixed_pairing"), "scenario_contract.fixed_pairing")
    for field in pairing:
        _require(pairing[field] is True, f"scenario_contract.fixed_pairing.{field} must be true")


def _validate_seeds(payload: dict[str, Any]) -> None:
    """Validate the explicit release S30 seed schedule and its source."""
    seed_policy = _mapping(payload.get("seed_policy"), "seed_policy")
    _require(seed_policy.get("set_name") == "paper_eval_s30", "seed policy must use paper_eval_s30")
    _require(
        seed_policy.get("seeds") == EXPECTED_SEEDS,
        "seed schedule must match paper_eval_s30 exactly",
    )
    seed_path = seed_policy.get("source_path")
    _require(
        isinstance(seed_path, str) and (REPO_ROOT / seed_path).is_file(),
        "seed source path is missing",
    )


def _validate_speed_axis(payload: dict[str, Any]) -> None:
    """Validate speed caps and the existing runtime binding contract."""
    speed_axis = _mapping(payload.get("robot_speed_axis"), "robot_speed_axis")
    _require(speed_axis.get("baseline_cap_m_s") == 2.0, "baseline speed cap must be 2.0 m/s")
    tiers = speed_axis.get("tiers")
    _require(isinstance(tiers, list) and len(tiers) == 3, "exactly three speed tiers are required")
    tier_caps = [
        _finite_number(_mapping(tier, "speed tier").get("cap_m_s"), "tier.cap_m_s", positive=True)
        for tier in tiers
    ]
    _require(tier_caps == [2.0, 3.0, 4.2], "speed tiers must be exactly 2.0, 3.0, and 4.2 m/s")
    override = _mapping(speed_axis.get("override_contract"), "override_contract")
    _require(
        override.get("runtime_binding") == "robot_config.drive_speed_cap", "runtime binding drifted"
    )
    runtime_reference = override.get("runtime_reference_config")
    _require(
        isinstance(runtime_reference, str) and (REPO_ROOT / runtime_reference).is_file(),
        "runtime reference config is missing",
    )
    _require(override.get("robot_only_axis") is True, "speed axis must be robot-only")
    _require(
        override.get("pedestrian_speed_axis_fixed") is True, "pedestrian speed axis must be fixed"
    )
    _require(
        override.get("supported_drive_models")
        == {"bicycle_drive": "max_velocity", "differential_drive": "max_linear_speed"},
        "drive-model speed fields drifted",
    )


def _validate_roster(payload: dict[str, Any]) -> None:
    """Validate the four-arm reduced roster and canonical config paths."""
    roster = _mapping(payload.get("planner_roster"), "planner_roster")
    arms = roster.get("arms")
    _require(isinstance(arms, list) and len(arms) == 4, "exactly four planner arms are required")
    planner_ids = {str(_mapping(arm, "planner arm").get("planner_id")) for arm in arms}
    _require(planner_ids == EXPECTED_PLANNERS, "planner roster drifted")
    for arm_value in arms:
        arm = _mapping(arm_value, "planner arm")
        config_path_value = arm.get("config_path")
        if config_path_value is not None:
            _require(
                isinstance(config_path_value, str) and (REPO_ROOT / config_path_value).is_file(),
                f"missing planner config: {config_path_value}",
            )


def _validate_inference(payload: dict[str, Any]) -> None:
    """Validate the #5557 resampling, estimand, multiplicity, and decision rule."""
    inference = _mapping(payload.get("inference_contract"), "inference_contract")
    _require(
        inference.get("resampling_unit") == "scenario_clustered_hierarchical",
        "resampling unit is not frozen",
    )
    _require(
        inference.get("inference_population") == "fixed_declared_suite",
        "inference population is not frozen",
    )
    _require(inference.get("estimand") == "paired_delta", "estimand is not frozen")
    _require(
        inference.get("primary_metrics") == ["success_rate", "collision_rate", "near_miss_rate"],
        "primary metrics drifted",
    )
    multiplicity = _mapping(inference.get("multiplicity"), "inference_contract.multiplicity")
    _require(multiplicity.get("method") == "holm_bonferroni", "multiplicity method is not frozen")
    _require(
        multiplicity.get("tests_per_planner") == 6,
        "multiplicity family must contain six tests per planner",
    )
    decision = _mapping(inference.get("decision_rule"), "inference_contract.decision_rule")
    _require(decision.get("confidence_level") == 0.95, "confidence level must be 0.95")
    _finite_number(
        decision.get("success_rate_harm_threshold"),
        "decision_rule.success_rate_harm_threshold",
        allow_negative=True,
    )
    for field in ("collision_rate_harm_threshold", "near_miss_rate_harm_threshold"):
        _finite_number(decision.get(field), f"decision_rule.{field}")
    _require(
        set(_mapping(decision.get("labels"), "decision_rule.labels"))
        == {"materially_harmful", "no_material_shift", "inconclusive"},
        "decision labels incomplete",
    )


def _validate_outputs(payload: dict[str, Any]) -> None:
    """Validate typed outcomes and provenance requirements."""
    result_contract = _mapping(payload.get("result_contract"), "result_contract")
    _require(
        result_contract.get("expected_speed_scenario_cells") == 18,
        "speed x scenario cell count must be 18",
    )
    _require(result_contract.get("expected_cell_count") == 2160, "full cell count must be 2160")
    _require(
        set(result_contract.get("required_per_tier_summary", []))
        == {
            "success_rate",
            "collision_rate",
            "near_miss_rate",
            "typed_collision_breakdown",
            "paired_delta_vs_cap_2_0",
        },
        "per-tier output contract incomplete",
    )
    provenance = _mapping(payload.get("provenance_contract"), "provenance_contract")
    _require(
        isinstance(provenance.get("required"), list) and len(provenance["required"]) >= 5,
        "provenance contract incomplete",
    )
    _require(
        provenance.get("raw_episode_policy") == "keep_raw_jsonl_videos_logs_out_of_git",
        "raw artifact policy drifted",
    )


def validate_preregistration(payload: dict[str, Any]) -> None:
    """Validate the complete issue #5578 contract and raise on the first blocker."""
    _assert_no_transient_routing_state(payload)
    _validate_header(payload)
    _validate_baseline(payload)
    _validate_scenarios(payload)
    _validate_seeds(payload)
    _validate_speed_axis(payload)
    _validate_roster(payload)
    _validate_inference(payload)
    _validate_outputs(payload)


def build_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a compact machine-readable checker report."""
    return {
        "status": "preregistration_valid",
        "schema_version": payload["schema_version"],
        "issue": payload["issue"],
        "study_id": payload["study_id"],
        "speed_tier_count": len(payload["robot_speed_axis"]["tiers"]),
        "scenario_count": len(payload["scenario_contract"]["selected_scenarios"]),
        "planner_count": len(payload["planner_roster"]["arms"]),
        "seed_count": len(payload["seed_policy"]["seeds"]),
        "expected_cell_count": payload["result_contract"]["expected_cell_count"],
        "claim_boundary": payload["claim_boundary"],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the checker CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="Emit a JSON report on success.")
    args = parser.parse_args(argv)
    try:
        payload = load_preregistration(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"FAIL: issue #5578 preregistration invalid: {exc}")
        return 2
    report = build_report(payload)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "PASS: issue #5578 robot speed-tier preregistration valid "
            f"({report['scenario_count']} scenarios x {report['speed_tier_count']} tiers x "
            f"{report['planner_count']} planners x {report['seed_count']} seeds)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
