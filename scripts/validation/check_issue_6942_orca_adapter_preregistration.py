#!/usr/bin/env python3
"""Validate the issue #6942 ORCA adapter-hedge preregistration.

The checker validates the tracked protocol and referenced repository paths only.
It does not import Robot SF runtime modules, execute native ORCA, submit compute,
or inspect benchmark output. A valid packet intentionally reports ``blocked``
because approval and campaign execution are separate gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from scripts.validation.check_preregistration_inference_contract import (
    InferenceContractError,
    check_inference_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / (
    "configs/benchmarks/issue_6942_orca_adapter_hedge_preregistration.yaml"
)
SCHEMA_VERSION = "robot_sf.issue_6942_orca_adapter_hedge_preregistration.v1"
EXPECTED_SCENARIOS = (
    "classic_head_on_corridor_medium",
    "classic_doorway_medium",
    "classic_group_crossing_medium",
    "classic_merging_medium",
    "classic_overtaking_medium",
    "classic_station_platform_medium",
)
EXPECTED_SEEDS = tuple(range(111, 141))
EXPECTED_SCENARIO_MATRIX = "configs/scenarios/issue_6474_social_compliance_nominal.yaml"
REQUIRED_TRACE_FIELDS = (
    "schema_version",
    "step",
    "robot_heading_rad",
    "planned_velocity_world_mps",
    "planned_speed_mps",
    "executed_command_vw",
    "realized_velocity_world_mps",
    "executed_speed_mps",
    "angle_error_rad",
    "speed_delta_mps",
)
FORBIDDEN_TRANSIENT_KEYS = {
    "host",
    "target_host",
    "slurm",
    "job_id",
    "queue",
    "queue_route",
    "worktree",
    "output_dir",
    "run_id",
}


class PacketError(ValueError):
    """Raised when the tracked preregistration is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    _require(isinstance(value, dict), f"{key} must be a mapping")
    return value


def _repo_path(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be a path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts, f"{field} must be repo-relative")
    return value


def _walk_for_transient_keys(value: Any, path: str = "packet") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                str(key).lower() not in FORBIDDEN_TRANSIENT_KEYS,
                f"{path}.{key} is transient routing or local-output state",
            )
            _walk_for_transient_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_for_transient_keys(child, f"{path}[{index}]")


def _require_file(value: Any, field: str, *, root: Path) -> str:
    relative = _repo_path(value, field)
    _require((root / relative).is_file(), f"{field} is missing: {relative}")
    return relative


def _require_path(value: Any, field: str, *, root: Path) -> str:
    relative = _repo_path(value, field)
    _require((root / relative).exists(), f"{field} is missing: {relative}")
    return relative


def load_packet(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load a YAML packet and require a mapping root."""
    packet_path = Path(path)
    if not packet_path.is_absolute():
        packet_path = REPO_ROOT / packet_path
    try:
        payload = yaml.safe_load(packet_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PacketError(f"cannot read preregistration {packet_path}: {exc}") from exc
    _require(isinstance(payload, dict), "preregistration must be a YAML mapping")
    return payload


def validate_packet(  # noqa: PLR0915
    packet: dict[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate the protocol without importing project runtime modules or running a campaign."""
    root = repo_root or REPO_ROOT
    _walk_for_transient_keys(packet)
    _require(packet.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(packet.get("issue") == 6942, "issue must be 6942")
    _require(packet.get("predecessor_issue") == 6615, "predecessor_issue must be 6615")
    _require(packet.get("status") == "preregistration", "status must remain preregistration")
    _require(packet.get("evidence_tier") == "proposal", "evidence_tier must remain proposal")
    claim_boundary = str(packet.get("claim_boundary", "")).lower()
    for phrase in ("preregistration", "does not run", "native-orca equivalence", "paper"):
        _require(phrase in claim_boundary, f"claim_boundary must mention {phrase}")

    execution = _mapping(packet, "execution_boundary")
    for key in (
        "representative_campaign_run_in_this_change",
        "native_canary_run_in_this_change",
        "compute_submit_authorized",
        "paper_or_dissertation_claim_edits",
        "metric_semantics_changes",
        "fallback_or_degraded_success_allowed",
    ):
        _require(execution.get(key) is False, f"execution_boundary.{key} must be false")

    approval = _mapping(packet, "domain_approval")
    _require(approval.get("required") is True, "domain_approval.required must be true")
    _require(approval.get("status") == "pending", "domain_approval.status must remain pending")
    decisions = approval.get("required_decisions")
    _require(
        isinstance(decisions, list) and len(decisions) >= 5, "approval decisions are incomplete"
    )

    baseline = _mapping(packet, "baseline_protocol")
    scenario_matrix = _require_file(
        baseline.get("scenario_matrix"), "baseline_protocol.scenario_matrix", root=root
    )
    _require(scenario_matrix == EXPECTED_SCENARIO_MATRIX, "scenario matrix identity mismatch")
    try:
        matrix_payload = yaml.safe_load((root / scenario_matrix).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PacketError(f"cannot read scenario matrix: {exc}") from exc
    _require(isinstance(matrix_payload, dict), "scenario matrix must be a YAML mapping")
    _require(
        tuple(matrix_payload.get("select_scenarios", ())) == EXPECTED_SCENARIOS,
        "scenario matrix selection mismatch",
    )
    _require(
        baseline.get("scenario_population") == "fixed_declared_suite",
        "scenario population must be fixed",
    )
    _require(baseline.get("horizon_steps") == 600, "horizon_steps must be 600")
    _require(baseline.get("dt_seconds") == 0.1, "dt_seconds must be 0.1")
    _require(baseline.get("scenario_count") == 6, "scenario_count must be 6")
    scenarios = baseline.get("selected_scenarios")
    _require(isinstance(scenarios, list), "selected_scenarios must be a list")
    _require(
        tuple(row.get("scenario_id") for row in scenarios if isinstance(row, dict))
        == EXPECTED_SCENARIOS,
        "scenario order mismatch",
    )
    _require(len(scenarios) == len(EXPECTED_SCENARIOS), "selected_scenarios length mismatch")
    for index, row in enumerate(scenarios):
        _require(isinstance(row, dict), f"selected_scenarios[{index}] must be a mapping")
        _require_file(row.get("source_path"), f"selected_scenarios[{index}].source_path", root=root)
        _require(
            isinstance(row.get("scenario_family"), str),
            f"selected_scenarios[{index}].scenario_family required",
        )
    pairing = _mapping(baseline, "pairing")
    for key in (
        "same_scenario_seed_set_across_arms",
        "same_initial_snapshot_across_arms",
        "same_simulator_physics_steps_across_arms",
        "scenario_order_frozen",
        "no_seed_substitution",
    ):
        _require(pairing.get(key) is True, f"pairing.{key} must be true")

    seed_policy = _mapping(packet, "seed_policy")
    _require_file(seed_policy.get("source_path"), "seed_policy.source_path", root=root)
    _require(seed_policy.get("set_name") == "paper_eval_s30", "seed set must be paper_eval_s30")
    _require(tuple(seed_policy.get("seeds", ())) == EXPECTED_SEEDS, "seed set mismatch")
    _require(seed_policy.get("seed_count") == 30, "seed_count must be 30")
    _require(seed_policy.get("episode_cell_count") == 180, "episode_cell_count must be 180")

    comparator = _mapping(packet, "comparator_contract")
    _require(
        comparator.get("comparator_id") == "same_native_rvo2_velocity_with_or_without_projection",
        "comparator_id mismatch",
    )
    solver = _mapping(comparator, "native_solver")
    _require(solver.get("implementation") == "vendored_python_rvo2", "native solver mismatch")
    _require(
        solver.get("upstream_repo") == "https://github.com/mit-acl/Python-RVO2",
        "upstream repo mismatch",
    )
    _require(
        solver.get("upstream_commit") == "56b245132ea104ee8a621ddf65b8a3dd85028ed2",
        "upstream commit mismatch",
    )
    _require_path(solver.get("vendored_path"), "native_solver.vendored_path", root=root)
    _require(solver.get("required_execution_mode") == "native", "native execution mode required")
    _require(solver.get("fallback_allowed") is False, "native fallback must be disallowed")
    arms = comparator.get("arms")
    _require(isinstance(arms, list) and len(arms) == 2, "comparator must declare two arms")
    arm_ids = tuple(arm.get("arm_id") for arm in arms if isinstance(arm, dict))
    _require(
        arm_ids == ("native_world_velocity_counterfactual", "adapter_projected_unicycle"),
        "comparator arm order mismatch",
    )
    _require(
        comparator.get("same_native_velocity_proposals") is True, "native proposals must be paired"
    )
    _require(
        comparator.get("planner_behavior_is_not_reestimated") is True,
        "planner behavior must not be reestimated",
    )
    _require(
        comparator.get("projection_is_the_only_declared_arm_difference") is True,
        "projection must be the only arm difference",
    )
    _require(
        arms[0].get("planner_anchor")
        == "robot_sf.planner.socnav_orca.ORCAPlannerAdapter.plan_velocity_world",
        "counterfactual planner anchor mismatch",
    )
    _require(
        arms[1].get("projection_anchor")
        == "robot_sf.planner.socnav_orca.ORCAPlannerAdapter._velocity_world_to_command",
        "projection anchor mismatch",
    )
    _require(
        arms[1].get("trace_schema_version") == "orca_adapter_trace.v1", "trace schema mismatch"
    )

    trace = _mapping(packet, "trace_contract")
    _require(
        trace.get("schema_version") == "orca_adapter_trace.v1", "trace_contract schema mismatch"
    )
    _require(
        tuple(trace.get("required_fields", ())) == REQUIRED_TRACE_FIELDS,
        "trace field order mismatch",
    )
    _require(
        trace.get("pairing_key") == ["scenario_id", "seed", "episode_step"],
        "trace pairing key mismatch",
    )
    _require(
        "reject" in str(trace.get("nonfinite_policy", "")).lower(), "nonfinite policy must reject"
    )
    _require(
        "do_not_impute" in str(trace.get("missing_field_policy", "")).lower(),
        "missing field policy must not impute",
    )

    estimands = _mapping(packet, "estimand_contract")
    primary = estimands.get("primary_estimands")
    _require(
        isinstance(primary, list) and len(primary) == 2,
        "exactly two primary estimands are required",
    )
    _require(
        tuple(item.get("estimand_id") for item in primary if isinstance(item, dict))
        == ("angle_error_rad", "forward_speed_loss_mps"),
        "primary estimand order mismatch",
    )
    for index, item in enumerate(primary):
        _require(isinstance(item, dict), f"primary_estimands[{index}] must be a mapping")
        _require(
            float(item.get("materiality_threshold", 0.0)) > 0.0,
            f"primary_estimands[{index}] threshold must be positive",
        )
    _require(
        "diagnostic" in str(estimands.get("outcome_boundary", "")).lower(),
        "outcome boundary must remain diagnostic",
    )

    inference = _mapping(packet, "inference_contract")
    try:
        check_inference_contract(packet, repo_root=root)
    except InferenceContractError as exc:
        raise PacketError(f"inference contract invalid: {exc}") from exc
    _require(
        inference.get("inference_population_id") == "fixed_declared_suite",
        "inference population identity mismatch",
    )
    _require(
        inference.get("resampling_unit_id") == "paired_scenario_seed_block",
        "resampling unit identity mismatch",
    )
    _require(inference.get("bootstrap_samples") == 2000, "bootstrap sample count mismatch")
    _require(inference.get("confidence_level") == 0.95, "confidence level mismatch")
    _require(inference.get("bootstrap_seed") == 123, "bootstrap seed mismatch")
    _require(
        inference.get("missingness") == "complete_case_only_no_imputation",
        "missingness policy mismatch",
    )
    _require(
        "holm" in str(inference.get("multiplicity", "")).lower(),
        "multiplicity policy must use Holm",
    )

    provenance = _mapping(packet, "provenance_contract")
    required_provenance = provenance.get("required_fields")
    _require(
        isinstance(required_provenance, list) and len(required_provenance) >= 12,
        "provenance fields are incomplete",
    )
    for field in ("source_commit", "exact_command", "fallback_used"):
        _require(field in required_provenance, f"{field} provenance is required")

    preflight = _mapping(packet, "preflight_contract")
    _require(preflight.get("native_dependency") == "rvo2", "native dependency must be rvo2")
    required_paths = preflight.get("required_paths")
    _require(
        isinstance(required_paths, list) and len(required_paths) >= 6,
        "preflight paths are incomplete",
    )
    for index, path in enumerate(required_paths):
        _require_file(path, f"preflight_contract.required_paths[{index}]", root=root)
    canary = _mapping(preflight, "native_canary")
    _require(canary.get("required_before_campaign") is True, "native canary must be required")
    _require(
        "run_orca_adapter_validation_issue_6615.py" in str(canary.get("command_reference", "")),
        "canary command must reference #6615 smoke",
    )
    _require(
        preflight.get("representative_runner") == "separate_reviewed_implementation_required",
        "representative runner must remain separate",
    )

    stop_rules = packet.get("stop_rules")
    _require(isinstance(stop_rules, list) and len(stop_rules) >= 8, "stop rules are incomplete")
    _require(any("fallback" in str(rule) for rule in stop_rules), "fallback stop rule is required")
    _require(any("approval" in str(rule) for rule in stop_rules), "approval stop rule is required")

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 6942,
        "status": "blocked",
        "evidence_tier": "proposal",
        "project_imports_performed": False,
        "execution_authorized": False,
        "domain_approval_status": approval["status"],
        "scenario_count": len(scenarios),
        "seed_count": len(seed_policy["seeds"]),
        "episode_cell_count": seed_policy["episode_cell_count"],
        "native_dependency": preflight["native_dependency"],
        "fallback_allowed": solver["fallback_allowed"],
        "checks": {
            "referenced_paths_exist": True,
            "pairing_contract_frozen": True,
            "native_comparator_frozen": True,
            "trace_contract_frozen": True,
            "inference_contract_frozen": True,
            "approval_pending": True,
            "campaign_not_run": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Validate the packet and emit a machine-readable blocked status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a prose summary")
    args = parser.parse_args(argv)
    try:
        result = validate_packet(load_packet(args.config))
    except (OSError, PacketError, TypeError, ValueError) as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, sort_keys=True))
        else:
            print(f"FAIL: issue #6942 preregistration invalid: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("PASS: issue #6942 preregistration valid; execution remains blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
