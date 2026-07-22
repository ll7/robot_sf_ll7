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
EXPECTED_ACTIVATION_DIAGNOSTICS = {
    "commanded_speed_mean_m_s",
    "realized_speed_mean_m_s",
    "realized_speed_peak_m_s",
    "fraction_above_2_0_mps",
    "cap_saturation_fraction",
    "resolved_actuation_envelope",
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


def _nonempty_string(value: Any, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string")
    return value.strip()


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
    return {
        str(row["name"])
        for row in scenarios
        if isinstance(row, dict) and row.get("name") is not None
    }


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
    _require(payload.get("amendment_issue") == 6100, "amendment_issue must reference #6100")
    _require(payload.get("status") == "preregistration", "status must remain preregistration")
    _require(
        payload.get("primary_claim_scope") == "per_planner_robustness",
        "primary_claim_scope must be per_planner_robustness",
    )
    _require(
        payload.get("ranking_claim_scope") == "descriptive_only",
        "ranking_claim_scope must be descriptive_only",
    )
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
    """Validate the fixed six-row middle-band scenario subset and justifications."""
    scenario_contract = _mapping(payload.get("scenario_contract"), "scenario_contract")
    _nonempty_string(
        scenario_contract.get("scenario_count_justification"), "scenario_count_justification"
    )
    _nonempty_string(scenario_contract.get("power_sensitivity_note"), "power_sensitivity_note")
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
    """Validate speed caps, tier actuation parameters, and activation contract."""
    speed_axis = _mapping(payload.get("robot_speed_axis"), "robot_speed_axis")
    _require(speed_axis.get("baseline_cap_m_s") == 2.0, "baseline speed cap must be 2.0 m/s")
    tiers = speed_axis.get("tiers")
    _require(isinstance(tiers, list) and len(tiers) == 3, "exactly three speed tiers are required")
    tier_caps = [
        _finite_number(_mapping(tier, "speed tier").get("cap_m_s"), "tier.cap_m_s", positive=True)
        for tier in tiers
    ]
    _require(tier_caps == [2.0, 3.0, 4.2], "speed tiers must be exactly 2.0, 3.0, and 4.2 m/s")

    for index, tier_val in enumerate(tiers):
        tier = _mapping(tier_val, f"robot_speed_axis.tiers[{index}]")
        cap = _finite_number(tier.get("cap_m_s"), f"tier[{index}].cap_m_s", positive=True)
        _nonempty_string(tier.get("drive_model"), f"tier[{index}].drive_model")
        _finite_number(tier.get("max_accel_m_s2"), f"tier[{index}].max_accel_m_s2", positive=True)
        max_decel = _finite_number(
            tier.get("max_decel_m_s2"), f"tier[{index}].max_decel_m_s2", positive=True
        )
        stopping_dist = _finite_number(
            tier.get("stopping_distance_envelope_m"),
            f"tier[{index}].stopping_distance_envelope_m",
            positive=True,
        )
        expected_dist = cap**2 / (2.0 * max_decel)
        _require(
            math.isclose(stopping_dist, expected_dist, abs_tol=1e-6),
            f"tier[{index}] stopping_distance_envelope_m inconsistent: "
            f"expected {expected_dist:.4f}, got {stopping_dist:.4f}",
        )
        scaling = _mapping(
            tier.get("command_action_scaling"), f"tier[{index}].command_action_scaling"
        )
        _nonempty_string(scaling.get("scaling_method"), f"tier[{index}].scaling_method")

    tier_4_2 = _mapping(tiers[2], "tier_4_2")
    _require(
        tier_4_2.get("drive_model") == "bicycle_drive",
        "4.2 m/s tier drive_model must be bicycle_drive",
    )
    _require(tier_4_2.get("max_accel_m_s2") == 2.1, "4.2 m/s tier max_accel_m_s2 must be 2.1")
    _require(tier_4_2.get("max_decel_m_s2") == 4.2, "4.2 m/s tier max_decel_m_s2 must be 4.2")
    _require(
        tier_4_2.get("stopping_distance_envelope_m") == 2.1,
        "4.2 m/s tier stopping_distance_envelope_m must be 2.1",
    )

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

    activation = _mapping(speed_axis.get("activation_contract"), "activation_contract")
    req_diag = activation.get("required_diagnostics")
    _require(isinstance(req_diag, list), "activation_contract.required_diagnostics must be a list")
    _require(
        set(req_diag) == EXPECTED_ACTIVATION_DIAGNOSTICS,
        "activation_contract.required_diagnostics drifted",
    )
    rule = _mapping(
        activation.get("minimum_activation_rule"), "activation_contract.minimum_activation_rule"
    )
    _finite_number(
        rule.get("min_fraction_above_2_0_mps"), "minimum_activation_rule.min_fraction_above_2_0_mps"
    )
    _finite_number(rule.get("min_peak_speed_m_s"), "minimum_activation_rule.min_peak_speed_m_s")
    _nonempty_string(rule.get("rule_definition"), "minimum_activation_rule.rule_definition")


def _validate_roster(payload: dict[str, Any]) -> None:
    """Validate the four-arm reduced roster, PPO OOD estimand, and top hybrid selection."""
    roster = _mapping(payload.get("planner_roster"), "planner_roster")
    arms = roster.get("arms")
    _require(isinstance(arms, list) and len(arms) == 4, "exactly four planner arms are required")
    planner_ids: set[str] = set()
    for arm_value in arms:
        arm = _mapping(arm_value, "planner arm")
        planner_id = arm.get("planner_id")
        _require(
            isinstance(planner_id, str) and planner_id, "planner_id must be a non-empty string"
        )
        planner_ids.add(planner_id)
        config_path_value = arm.get("config_path")
        if config_path_value is not None:
            _require(
                isinstance(config_path_value, str) and config_path_value,
                "planner config_path must be a non-empty string when provided",
            )
            _require(
                (REPO_ROOT / config_path_value).is_file(),
                f"missing planner config: {config_path_value}",
            )
        if planner_id == "ppo":
            _require(
                arm.get("estimand_type") == "zero_shot_ood_robustness",
                "PPO estimand_type must be zero_shot_ood_robustness",
            )
            _require(
                arm.get("retraining_status") == "none_zero_shot_eval_only",
                "PPO retraining_status must be none_zero_shot_eval_only",
            )
        if planner_id == "scenario_adaptive_hybrid_orca_v2_collision_guard":
            _require(
                arm.get("role") in {"top_hybrid_promoted", "top_hybrid_candidate"},
                "top hybrid role must be top_hybrid_promoted or top_hybrid_candidate",
            )
    _require(planner_ids == EXPECTED_PLANNERS, "planner roster drifted")


def _validate_inference(payload: dict[str, Any]) -> None:
    """Validate resampling, estimand, multiplicity, and safety interpretation notes."""
    inference = _mapping(payload.get("inference_contract"), "inference_contract")
    _require(
        inference.get("resampling_unit") == "paired_seed_block",
        "resampling unit must be paired_seed_block",
    )
    _require(
        inference.get("inference_population") == "fixed_declared_suite",
        "inference population must be fixed_declared_suite",
    )
    _require(inference.get("estimand") == "paired_delta", "estimand must be paired_delta")
    _require(
        inference.get("primary_claim_scope") == "per_planner_robustness",
        "primary_claim_scope must be per_planner_robustness",
    )
    _require(
        inference.get("ranking_claim_scope") == "descriptive_only",
        "ranking_claim_scope must be descriptive_only",
    )
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
    _require(
        multiplicity.get("hypothesis_alignment") == "margin_aligned_one_sided",
        "hypothesis_alignment must be margin_aligned_one_sided",
    )
    decision = _mapping(inference.get("decision_rule"), "inference_contract.decision_rule")
    _require(decision.get("confidence_level") == 0.95, "confidence level must be 0.95")
    _require(
        decision.get("hypothesis_type") == "margin_aligned_one_sided",
        "hypothesis_type must be margin_aligned_one_sided",
    )
    _finite_number(
        decision.get("success_rate_harm_threshold"),
        "decision_rule.success_rate_harm_threshold",
        allow_negative=True,
    )
    for field in ("collision_rate_harm_threshold", "near_miss_rate_harm_threshold"):
        _finite_number(decision.get(field), f"decision_rule.{field}")
    _require(
        set(_mapping(decision.get("labels"), "decision_rule.labels"))
        == {
            "materially_harmful",
            "no_material_shift",
            "inconclusive",
            "intervention_not_activated",
        },
        "decision labels incomplete",
    )

    safety_notes = _mapping(
        inference.get("safety_interpretation_contract"), "safety_interpretation_contract"
    )
    _nonempty_string(safety_notes.get("collision_frequency_note"), "collision_frequency_note")
    _nonempty_string(safety_notes.get("speed_range_validity_note"), "speed_range_validity_note")


def _validate_outputs(payload: dict[str, Any]) -> None:
    """Validate required output contract keys, activation diagnostics, and provenance."""
    result_contract = _mapping(payload.get("result_contract"), "result_contract")
    _require(
        result_contract.get("expected_speed_scenario_cells") == 18,
        "speed x scenario cell count must be 18",
    )
    _require(result_contract.get("expected_cell_count") == 2160, "full cell count must be 2160")
    required_keys = result_contract.get("required_cell_keys")
    _require(isinstance(required_keys, list), "required_cell_keys must be a list")
    _require(
        EXPECTED_ACTIVATION_DIAGNOSTICS.issubset(set(required_keys)),
        "required_cell_keys must include all activation diagnostics",
    )
    summary = result_contract.get("required_per_tier_summary")
    _require(isinstance(summary, list), "required_per_tier_summary must be a list")
    _require(
        {
            "success_rate",
            "collision_rate",
            "near_miss_rate",
            "typed_collision_breakdown",
            "paired_delta_vs_cap_2_0",
            "activation_diagnostics_summary",
            "exposure_summary",
        }.issubset(set(summary)),
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
        "governance_issue": payload["governance_issue"],
        "amendment_issue": payload["amendment_issue"],
        "study_id": payload["study_id"],
        "primary_claim_scope": payload["primary_claim_scope"],
        "ranking_claim_scope": payload["ranking_claim_scope"],
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
