"""Physical-unit audit of the readiness matrix's representative planner configs.

This is a config and drive-envelope contract, not a claim that a planner is safe or
benchmark-ready. Metres used for goals, surface clearance, and centre-distance
triggers have different semantics; only the latter's safety relation is tested.
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.planner.dwa import DWAPlannerConfig
from robot_sf.planner.hybrid_rule_local_planner import HybridRuleLocalPlannerConfig
from robot_sf.planner.risk_dwa import RiskDWAPlannerConfig
from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.robot.actuation_envelope import stopping_distance
from robot_sf.robot.differential_drive import DifferentialDriveSettings

ROOT = Path(__file__).resolve().parents[2]
READINESS_MATRIX = ROOT / "configs/benchmarks/planner_readiness_matrix_v1.yaml"
HYBRID_V3 = ROOT / "configs/algos/hybrid_rule_v3_static_margin0_waypoint2.yaml"

# The matrix's four algo= rows use SocNav defaults or a goal-only heuristic.
# DWA, Risk-DWA and hybrid defaults are included because YAML omits some fields.
# A new unit-bearing field must be classified here before this audit passes.
AUDITED_UNIT_FIELDS = frozenset(
    """
    actuation_max_angular_accel actuation_max_linear_accel actuation_max_linear_decel
    actuation_max_yaw_rate caution_clearance clearance_distance control_dt control_period
    corridor_subgoal_goal_stall_progress_3s corridor_subgoal_max_lateral_offset
    corridor_subgoal_min_nearest_ped_distance corridor_subgoal_min_route_remaining_distance
    corridor_subgoal_route_stall_progress_3s corridor_subgoal_route_regression_1s
    corridor_subgoal_speed
    corridor_subgoal_static_clearance_buffer deadlock_progress_threshold
    desired_dynamic_clearance desired_static_clearance dt emergency_clearance
    forecast_variant_dt_s forecast_variant_horizons_s forecast_variant_risk_distance_m
    freezing_speed_threshold global_route_probe_waypoint_distance goal_far_distance
    goal_posterior_crossing_lateral_margin goal_posterior_near_distance
    goal_posterior_turn_rate goal_posterior_yield_speed goal_tolerance
    guard_first_step_ped_clearance guard_hard_obstacle_clearance guard_hard_ped_clearance
    guard_min_ttc guard_near_field_distance guard_rollout_dt hard_collision_horizon hard_safety_margin
    height hrvo_neighbor_dist hrvo_time_horizon hrvo_uncertainty_offset
    linear_candidates angular_candidates lookahead_distances
    max_angular_accel max_angular_acceleration max_angular_speed max_linear_accel
    max_linear_acceleration max_linear_decel max_linear_speed moderate_distance_human
    moderate_speed native_occupancy_resolution near_distance near_field_distance
    near_field_speed_cap near_human_angular_limit_distance near_human_max_angular_speed
    occupancy_heading_sweep occupancy_lookahead orca_commit_distance
    orca_forward_probe_distance orca_neighbor_dist orca_obstacle_margin orca_obstacle_range
    orca_side_probe_offset orca_stall_progress_epsilon orca_stall_speed_threshold
    orca_time_horizon orca_time_horizon_obst pedestrian_radius pedestrian_radius_default
    prediction_dt predictive_candidate_heading_deltas predictive_candidate_speeds
    predictive_foresight_front_corridor_half_width predictive_foresight_front_corridor_length
    predictive_foresight_near_distance predictive_foresight_rollout_dt
    predictive_hard_clearance_distance predictive_near_distance
    predictive_near_field_distance predictive_near_field_heading_deltas
    predictive_near_field_speed_cap predictive_near_field_speed_samples
    predictive_pedestrian_radius predictive_phase_commit_clearance
    predictive_phase_recover_clearance predictive_phase_recover_progress
    predictive_phase_yield_clearance predictive_progress_escape_clearance_margin
    predictive_progress_escape_distance predictive_progress_risk_distance
    predictive_reverse_candidate_speeds predictive_robot_radius predictive_rollout_dt
    predictive_safe_distance predictive_ttc_distance predictive_uncertainty_base_std
    predictive_uncertainty_growth_per_step progress_escape_distance
    progress_escape_speed proxemic_costmap_personal_radius proxemic_costmap_social_radius
    recovery_reorient_angular_speed resolution robot_radius robot_radius_default
    rollout_dt rollout_horizon route_guide_commitment_progress_threshold
    route_rescue_progress_threshold route_trace_recovery_goal_stall_progress_3s
    route_trace_recovery_min_nearest_ped_distance
    route_trace_recovery_min_route_remaining_distance
    route_trace_recovery_route_stall_progress_3s
    route_trace_recovery_route_regression_1s sacadrl_pref_speed safe_distance
    safety_margin slow_distance_human social_force_desired_speed
    social_force_goal_approach_clearance social_force_goal_approach_max_speed
    social_force_goal_approach_radius social_force_goal_approach_stop_distance
    social_force_obstacle_range social_force_tau soft_prediction_horizon
    static_clearance_escape_max_speed static_clearance_escape_min_clearance
    static_clearance_escape_tolerance static_corridor_transit_initial_band
    static_corridor_transit_min_progress_3s static_corridor_transit_tolerance
    static_hard_safety_margin static_recenter_probe_speed
    static_safety_gate_clearance_tolerance static_safety_gate_min_clearance
    static_safety_gate_progress_threshold stop_distance_human v_max very_slow_speed
    omega_max step_timeout_sec
    waypoint_switch_distance width
    """.split()
)

# Name matches that have no direct length/speed/acceleration meaning. A scoring
# weight, cell count, horizon step count, occupancy probability, or synthetic
# uncertainty gain must not be compared numerically to metres or m/s.
NON_PHYSICAL_NAMES = frozenset(
    """
    noise_std predictive_sequence_beam_width predictive_uncertainty_density_scale
    predictive_uncertainty_speed_scale social_force_max_force
    social_force_obstacle_factor social_force_factor social_force_gamma
    social_force_lambda_importance social_force_n social_force_n_prime
    route_rescue_progress_weight_boost
    """.split()
)
NON_PHYSICAL_SUFFIXES = (
    "_weight",
    "_gain",
    "_ratio",
    "_scale",
    "_factor",
    "_cells",
    "_steps",
    "_samples",
    "_enabled",
    "_count",
    "_max_points",
    "_max_neighbors",
)
UNIT_TOKENS = (
    "speed",
    "accel",
    "decel",
    "distance",
    "clearance",
    "radius",
    "horizon",
    "tolerance",
    "margin",
    "dt",
    "period",
    "lookahead",
    "yaw_rate",
    "turn_rate",
    "offset",
    "length",
    "resolution",
    "width",
    "height",
    "progress",
    "band",
    "range",
    "tau",
    "heading_sweep",
    "heading_deltas",
)
DRIVE_LINEAR_SPEED_FIELDS = frozenset(
    """
    corridor_subgoal_speed freezing_speed_threshold goal_posterior_yield_speed
    linear_candidates max_linear_speed moderate_speed near_field_speed_cap v_max
    orca_stall_speed_threshold predictive_candidate_speeds
    predictive_near_field_speed_cap predictive_near_field_speed_samples
    predictive_reverse_candidate_speeds progress_escape_speed sacadrl_pref_speed
    social_force_desired_speed social_force_goal_approach_max_speed
    static_clearance_escape_max_speed static_recenter_probe_speed very_slow_speed
    """.split()
)
DRIVE_ANGULAR_SPEED_FIELDS = frozenset(
    """
    actuation_max_yaw_rate angular_candidates goal_posterior_turn_rate
    max_angular_speed near_human_max_angular_speed omega_max
    recovery_reorient_angular_speed
    """.split()
)
DRIVE_ACCEL_FIELDS = frozenset(
    """
    actuation_max_angular_accel actuation_max_linear_accel actuation_max_linear_decel
    max_angular_accel max_angular_acceleration max_linear_accel
    max_linear_acceleration max_linear_decel
    """.split()
)
RADIUS_FIELDS = frozenset(
    """
    pedestrian_radius pedestrian_radius_default predictive_pedestrian_radius
    predictive_robot_radius proxemic_costmap_personal_radius
    proxemic_costmap_social_radius robot_radius robot_radius_default
    social_force_goal_approach_radius
    """.split()
)
CENTRE_DISTANCE_FIELDS = frozenset(
    """
    corridor_subgoal_min_nearest_ped_distance goal_posterior_near_distance
    hrvo_neighbor_dist moderate_distance_human near_human_angular_limit_distance
    orca_neighbor_dist route_trace_recovery_min_nearest_ped_distance
    slow_distance_human stop_distance_human
    """.split()
)
# These names are physical but have different semantics or known drive mismatches.
# They receive finite/sign checks, with the exception reason documented in README.
DRIVE_RELATION_EXCEPTIONS = frozenset(
    """
    actuation_max_angular_accel actuation_max_linear_accel actuation_max_linear_decel
    actuation_max_yaw_rate angular_candidates max_angular_accel
    max_angular_acceleration max_angular_speed omega_max pedestrian_radius
    pedestrian_radius_default predictive_pedestrian_radius predictive_robot_radius
    proxemic_costmap_personal_radius proxemic_costmap_social_radius
    robot_radius robot_radius_default social_force_goal_approach_radius
    corridor_subgoal_min_nearest_ped_distance goal_posterior_near_distance
    hrvo_neighbor_dist near_human_angular_limit_distance orca_neighbor_dist
    route_trace_recovery_min_nearest_ped_distance
    """.split()
)
ZERO_OR_NEGATIVE_ALLOWED = frozenset(
    """
    predictive_reverse_candidate_speeds static_hard_safety_margin
    route_trace_recovery_route_stall_progress_3s
    corridor_subgoal_route_regression_1s route_trace_recovery_route_regression_1s
    route_trace_recovery_goal_stall_progress_3s
    corridor_subgoal_route_stall_progress_3s
    corridor_subgoal_goal_stall_progress_3s
    static_corridor_transit_min_progress_3s
    predictive_candidate_heading_deltas predictive_near_field_heading_deltas
    angular_candidates
    """.split()
)


def _number_leaves(value: Any, prefix: str = ""):
    """Yield numeric YAML leaves by full key, preserving nested duplicates."""
    if isinstance(value, dict):
        for name, child in value.items():
            key = f"{prefix}.{name}" if prefix else str(name)
            if isinstance(child, dict):
                yield from _number_leaves(child, key)
            elif not isinstance(child, bool) and isinstance(child, (int, float, list, tuple)):
                yield key, child


def _unit_bearing(name: str) -> bool:
    """Conservatively identify dimensioned names; unknown matches fail the inventory."""
    if name == "predictive_near_field_speed_samples":
        return True
    if name in NON_PHYSICAL_NAMES or name.endswith(NON_PHYSICAL_SUFFIXES):
        return False
    if name in {"obstacle_threshold", "orca_obstacle_threshold", "risk_threshold"}:
        return False
    if name == "social_force_obstacle_threshold":
        return False
    return (
        name.endswith("_sec")
        or name
        in {
            "linear_candidates",
            "angular_candidates",
            "v_max",
            "omega_max",
            "guard_min_ttc",
            "predictive_uncertainty_base_std",
            "predictive_uncertainty_growth_per_step",
            "corridor_subgoal_route_regression_1s",
            "route_trace_recovery_route_regression_1s",
            "predictive_near_field_speed_samples",
            "hrvo_neighbor_dist",
            "orca_neighbor_dist",
        }
        or any(token in name for token in UNIT_TOKENS)
    )


def _representative_values():
    matrix = yaml.safe_load(READINESS_MATRIX.read_text(encoding="utf-8"))
    rows = matrix["rows"]
    assert len(rows) >= 14, "readiness matrix unexpectedly lost planner rows"
    for row in rows:
        entrypoint = row["representative_entrypoint"]
        if not entrypoint.endswith(".yaml"):
            assert entrypoint in {
                "algo=goal",
                "algo=social_force",
                "algo=orca",
                "algo=sacadrl",
                "algo=socnav_bench",
            }
            continue
        path = ROOT / entrypoint
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        yield entrypoint, dict(_number_leaves(payload))
    for cls in (
        HybridRuleLocalPlannerConfig,
        SocNavPlannerConfig,
        DWAPlannerConfig,
        RiskDWAPlannerConfig,
    ):
        config = cls()
        values = {field.name: getattr(config, field.name) for field in fields(config)}
        yield f"{cls.__name__} defaults", values


def _numbers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def test_representative_planner_physical_field_inventory_and_ranges() -> None:
    """Audit every recognized physical-unit field in matrix YAML and four defaults."""
    seen: set[str] = set()
    for source, values in _representative_values():
        for key, value in values.items():
            name = key.rsplit(".", maxsplit=1)[-1]
            if not _unit_bearing(name):
                continue
            seen.add(name)
            assert name in AUDITED_UNIT_FIELDS, f"unclassified unit-bearing field: {source}.{key}"
            numbers = tuple(_numbers(value))
            assert numbers, f"missing numeric physical value: {source}.{key}"
            assert all(math.isfinite(number) for number in numbers), (source, key, numbers)
            if name == "step_timeout_sec":
                assert all(number > 0 for number in numbers), (source, key, numbers)
            if name not in ZERO_OR_NEGATIVE_ALLOWED:
                assert all(number >= 0 for number in numbers), (source, key, numbers)
    assert seen == AUDITED_UNIT_FIELDS, (
        f"stale inventory: missing={sorted(AUDITED_UNIT_FIELDS - seen)}, "
        f"unclassified={sorted(seen - AUDITED_UNIT_FIELDS)}"
    )


def test_representative_yaml_linear_speed_caps_fit_drive() -> None:
    """Explicit forward speed limits in matrix YAML cannot exceed the drive cap."""
    drive = DifferentialDriveSettings()
    for source, values in _representative_values():
        if source.endswith("defaults"):
            continue  # Unbound SocNav defaults are overridden or clipped at runtime.
        for key, value in values.items():
            name = key.rsplit(".", maxsplit=1)[-1]
            if name in {
                "max_linear_speed",
                "near_field_speed_cap",
                "progress_escape_speed",
                "v_max",
            }:
                assert float(value) <= drive.max_linear_speed, (source, key, value)


def test_drive_related_fields_have_a_check_or_explicit_exception() -> None:  # noqa: C901
    """Do not silently treat metre and actuator fields as generic positive numbers."""
    related = (
        DRIVE_LINEAR_SPEED_FIELDS
        | DRIVE_ANGULAR_SPEED_FIELDS
        | DRIVE_ACCEL_FIELDS
        | RADIUS_FIELDS
        | CENTRE_DISTANCE_FIELDS
    )
    assert related <= AUDITED_UNIT_FIELDS
    checked = (
        DRIVE_LINEAR_SPEED_FIELDS
        | {
            "goal_posterior_turn_rate",
            "near_human_max_angular_speed",
            "recovery_reorient_angular_speed",
        }
        | {"max_linear_acceleration", "max_linear_accel", "max_linear_decel"}
        | {"moderate_distance_human", "slow_distance_human", "stop_distance_human"}
    )
    assert related == checked | DRIVE_RELATION_EXCEPTIONS
    names_requiring_drive_classification = (
        {
            name
            for name in AUDITED_UNIT_FIELDS
            if any(token in name for token in ("speed", "accel", "decel", "radius"))
        }
        | CENTRE_DISTANCE_FIELDS
        | {
            "v_max",
            "omega_max",
            "linear_candidates",
            "angular_candidates",
            "goal_posterior_turn_rate",
            "actuation_max_yaw_rate",
        }
    )
    assert names_requiring_drive_classification == related

    drive = DifferentialDriveSettings()
    for source, values in _representative_values():
        for key, value in values.items():
            name = key.rsplit(".", maxsplit=1)[-1]
            if name not in checked:
                continue
            numbers = tuple(_numbers(value))
            if name in DRIVE_LINEAR_SPEED_FIELDS:
                # SocNav's unbound default is a preferred speed, not the drive's
                # effective maximum. The map-runner action adapter clips it.
                if source == "SocNavPlannerConfig defaults" and name == "max_linear_speed":
                    assert numbers == (3.0,)
                    continue
                assert all(abs(number) <= drive.max_linear_speed for number in numbers), (
                    source,
                    key,
                    numbers,
                )
            elif name in {
                "goal_posterior_turn_rate",
                "near_human_max_angular_speed",
                "recovery_reorient_angular_speed",
            }:
                assert all(number <= drive.max_angular_speed for number in numbers), (
                    source,
                    key,
                    numbers,
                )
            elif name == "max_linear_acceleration":
                assert all(number <= drive.max_linear_accel for number in numbers), (
                    source,
                    key,
                    numbers,
                )
            elif name in {"max_linear_accel", "max_linear_decel"}:
                # The selected hybrid v3 mismatch is asserted in the strict #9726
                # expected failure below. Other config sources must satisfy drive.
                if source == "HybridRuleLocalPlannerConfig defaults" or source == str(
                    HYBRID_V3.relative_to(ROOT)
                ):
                    continue
                limit = (
                    drive.max_linear_accel if name == "max_linear_accel" else drive.max_linear_decel
                )
                assert all(number <= limit for number in numbers), (source, key, numbers)
            elif name in {"moderate_distance_human", "slow_distance_human", "stop_distance_human"}:
                # These three are centre-distance speed gates. V3/default are
                # the known #9726 mismatch; v3 is tested by strict xfail below.
                if (
                    source
                    in {
                        "HybridRuleLocalPlannerConfig defaults",
                        str(HYBRID_V3.relative_to(ROOT)),
                    }
                    and name != "moderate_distance_human"
                ):
                    continue
                pedestrian_radius = float(values["pedestrian_radius_default"])
                contact = drive.radius + pedestrian_radius
                required = (
                    contact + stopping_distance(drive.max_linear_speed, drive.max_linear_decel)
                    if name == "slow_distance_human"
                    else contact
                )
                assert all(number >= required for number in numbers), (
                    source,
                    key,
                    numbers,
                    required,
                )
            else:
                raise AssertionError(f"drive-related field has no assertion: {source}.{key}")


def test_dwa_clearance_score_span_covers_drive_contact_scale() -> None:
    """DWA's scoring normalization spans contact scale; this is no safety gate."""
    drive = DifferentialDriveSettings()
    config = DWAPlannerConfig()
    assert config.max_linear_acceleration <= drive.max_linear_accel
    contact = drive.radius + config.pedestrian_radius
    assert config.clearance_distance >= contact


@pytest.mark.xfail(
    strict=True,
    reason="issue #9726: selected hybrid v3 centre thresholds and braking exceed drive envelope",
)
def test_selected_hybrid_v3_stop_and_braking_match_drive() -> None:
    """Check only the selected v3: a future v4 must get its own passing case."""
    config = yaml.safe_load(HYBRID_V3.read_text(encoding="utf-8"))
    assert config["planner_variant"] == "hybrid_rule_v3_teb_like_rollout"
    drive = DifferentialDriveSettings()
    contact = drive.radius + float(config["pedestrian_radius_default"])
    stopping = stopping_distance(drive.max_linear_speed, drive.max_linear_decel)
    violations = []
    if float(config["stop_distance_human"]) < contact:
        violations.append("stop centre distance falls inside contact distance")
    if float(config["slow_distance_human"]) < contact + stopping:
        violations.append("slow centre distance cannot cover drive stopping distance")
    if float(config["max_linear_accel"]) > drive.max_linear_accel:
        violations.append("rollout forward acceleration exceeds drive")
    if float(config["max_linear_decel"]) > drive.max_linear_decel:
        violations.append("rollout braking exceeds drive")
    assert not violations, "; ".join(violations)
