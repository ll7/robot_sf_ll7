"""Physical-unit audit of release-roster and readiness-matrix planner configs.

This is a config and drive-envelope contract, not a claim that a planner is safe or
benchmark-ready. The audited sources are the release campaign's planner list (its
``algo_config`` files resolved through ``base_config_path`` and every scenario
override), the readiness matrix's representative YAML files, and four planner code
defaults. Metres used for goals, surface clearance, and centre-distance triggers
have different semantics; only the safety relation of the centre-distance speed
gates is tested.

Known violations are pinned value-for-value in ``KNOWN_VIOLATIONS``. A passing
ledger test fails on any drift, including a known violation that gets worse, and
one strict expected failure per tracking issue flips when that issue is fixed.
"""

from __future__ import annotations

import math
from dataclasses import fields
from functools import cache
from typing import Any

import pytest
import yaml

from robot_sf.planner.dwa import DWAPlannerConfig
from robot_sf.planner.hybrid_rule_local_planner import HybridRuleLocalPlannerConfig
from robot_sf.planner.risk_dwa import RiskDWAPlannerConfig
from robot_sf.planner.socnav_base import SocNavPlannerConfig
from robot_sf.robot.actuation_envelope import stopping_distance
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sim.sim_config import SimulationSettings
from tests.metamorphic.planner_arms import (
    ROOT,
    release_campaign_planners,
    resolve_release_algo_config,
)
from tests.metamorphic.planner_arms import load_yaml as _load_yaml

READINESS_MATRIX = ROOT / "configs/benchmarks/planner_readiness_matrix_v1.yaml"
HYBRID_V3 = ROOT / "configs/algos/hybrid_rule_v3_static_margin0_waypoint2.yaml"
DRIVE = DifferentialDriveSettings()
SIM = SimulationSettings()
# Physical contact: drive body radius plus the simulator's pedestrian radius.
CONTACT_DISTANCE = DRIVE.radius + SIM.ped_radius
DRIVE_STOPPING_DISTANCE = stopping_distance(DRIVE.max_linear_speed, DRIVE.max_linear_decel)
# A time step longer than ten simulator steps, or an angular cap above ten times
# the drive's, is a unit error (ms for s, deg for rad), not a tuning choice.
MAX_TIME_STEP_S = 10.0 * SIM.time_per_step_in_secs
MAX_ANGULAR_SPEED = 10.0 * DRIVE.max_angular_speed
MAX_ANGULAR_ACCEL = 10.0 * DRIVE.max_angular_accel

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
    first_step_obstacle_clearance first_step_ped_clearance
    forecast_variant_dt_s forecast_variant_horizons_s forecast_variant_risk_distance_m
    freezing_speed_threshold global_route_probe_waypoint_distance goal_far_distance
    goal_posterior_crossing_lateral_margin goal_posterior_near_distance
    goal_posterior_turn_rate goal_posterior_yield_speed goal_tolerance
    guard_first_step_ped_clearance guard_hard_obstacle_clearance guard_hard_ped_clearance
    guard_min_ttc guard_near_field_distance guard_rollout_dt hard_collision_horizon
    hard_obstacle_clearance hard_ped_clearance hard_safety_margin
    height hrvo_neighbor_dist hrvo_time_horizon hrvo_uncertainty_offset
    init_angular_std init_linear_std
    linear_candidates angular_candidates lookahead_distances
    max_angular_accel max_angular_acceleration max_angular_speed max_linear_accel
    max_linear_acceleration max_linear_decel max_linear_speed min_angular_std
    min_linear_std moderate_distance_human
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
    social_force_obstacle_v2_length
    v4_braking_margin v4_moderate_clearance_human v4_reaction_time
    v4_slow_clearance_human v4_stop_clearance_human
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
    "_std",
)
TIME_STEP_FIELDS = frozenset(
    """
    control_dt control_period dt forecast_variant_dt_s guard_rollout_dt prediction_dt
    predictive_foresight_rollout_dt predictive_rollout_dt rollout_dt
    """.split()
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
DRIVE_LINEAR_ACCEL_FIELDS = frozenset(
    """
    actuation_max_linear_accel max_linear_accel max_linear_acceleration
    """.split()
)
DRIVE_LINEAR_DECEL_FIELDS = frozenset(("actuation_max_linear_decel", "max_linear_decel"))
DRIVE_ANGULAR_ACCEL_FIELDS = frozenset(
    ("actuation_max_angular_accel", "max_angular_accel", "max_angular_acceleration")
)
ROBOT_RADIUS_FIELDS = frozenset(("predictive_robot_radius", "robot_radius", "robot_radius_default"))
PEDESTRIAN_RADIUS_FIELDS = frozenset(
    ("pedestrian_radius", "pedestrian_radius_default", "predictive_pedestrian_radius")
)
# Body radii must lie in [physical radius, 1.5 x physical radius]: smaller plans
# through contact; a diameter (2 x) or larger is a unit error. Proxemic zones and
# the goal-approach radius are not bodies; they are bounded for sign and magnitude.
RADIUS_UNIT_FACTOR = 1.5
OTHER_RADIUS_FIELDS = frozenset(
    """
    proxemic_costmap_personal_radius proxemic_costmap_social_radius
    social_force_goal_approach_radius
    """.split()
)
# Hybrid v4 speed levels are pedestrian SURFACE clearances (centre distance minus
# both radii); the v3 centre-distance gates are unused by v4.
V4_CLEARANCE_LEVELS = (
    ("v4_stop_clearance_human", "very_slow_speed"),
    ("v4_slow_clearance_human", "moderate_speed"),
)
V4_CLEARANCE_FIELDS = frozenset(
    ("v4_stop_clearance_human", "v4_slow_clearance_human", "v4_moderate_clearance_human")
)
CENTRE_DISTANCE_GATE_FIELDS = frozenset(
    ("moderate_distance_human", "slow_distance_human", "stop_distance_human")
)
# Perception, neighbour-search, and route-recovery ranges, not speed gates.
CENTRE_DISTANCE_RANGE_FIELDS = frozenset(
    """
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
    angular_candidates v4_reaction_time
    """.split()
)
# SocNav's unbound default is a preferred speed, not the drive's effective maximum;
# the map-runner action adapter clips it to the drive. Every other source must fit.
LINEAR_SPEED_RUNTIME_CLIPPED_SOURCES = frozenset(("SocNavPlannerConfig defaults",))

HYBRID_RELEASE_CONFIGS = tuple(
    f"configs/policy_search/candidates/{name}_s30_h600_release.yaml"
    for name in (
        "hybrid_rule_v3_fast_progress_static_escape",
        "hybrid_rule_v3_fast_progress_static_escape_continuous",
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
        "scenario_adaptive_hybrid_orca_v2_collision_guard",
    )
)
_HYBRID_DEFAULTS = "HybridRuleLocalPlannerConfig defaults"
_HYBRID_V3_REPRESENTATIVE = "configs/algos/hybrid_rule_v3_static_margin0_waypoint2.yaml"
_HYBRID_PORTFOLIO = "configs/algos/hybrid_portfolio_camera_ready.yaml"
HYBRID_V4_BASE = "configs/algos/hybrid_rule_v4_clearance_braking.yaml"
# The #9747 v4 twins of the four release hybrid arms; they must fit the drive.
HYBRID_V4_RELEASE_TWINS = tuple(
    f"configs/policy_search/candidates/{name}_s30_h600_release.yaml"
    for name in (
        "hybrid_rule_v4_fast_progress_static_escape",
        "hybrid_rule_v4_fast_progress_static_escape_continuous",
        "scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4",
        "scenario_adaptive_hybrid_orca_v2_collision_guard_v4",
    )
)
HYBRID_V4_SOURCES = frozenset((HYBRID_V4_BASE, *HYBRID_V4_RELEASE_TWINS))
_PREDICTION_PLANNER = "configs/algos/prediction_planner_camera_ready.yaml"

# (rule, source, key) -> offending value. Values are pinned exactly: a fix
# removes an entry, and a regression that worsens a known violation fails too.
KNOWN_VIOLATIONS: dict[str, dict[tuple[str, str, str], Any]] = {
    # Hybrid rule planner: rollout and speed caps beyond the drive envelope, and
    # human speed gates measured as centre distance inside or near contact.
    "#9726": {
        **{
            ("linear_speed_exceeds_drive", source, "max_linear_speed"): 3.0
            for source in HYBRID_RELEASE_CONFIGS
        },
        **{
            ("linear_accel_exceeds_drive", source, "max_linear_accel"): 3.0
            for source in HYBRID_RELEASE_CONFIGS
        },
        **{
            ("linear_decel_exceeds_drive", source, "max_linear_decel"): 2.5
            for source in HYBRID_RELEASE_CONFIGS
        },
        **{
            ("stop_gate_inside_contact", source, "stop_distance_human"): 0.5
            for source in (*HYBRID_RELEASE_CONFIGS, _HYBRID_DEFAULTS, _HYBRID_V3_REPRESENTATIVE)
        },
        **{
            ("slow_gate_below_contact_plus_stopping", source, "slow_distance_human"): 1.0
            for source in (*HYBRID_RELEASE_CONFIGS, _HYBRID_DEFAULTS, _HYBRID_V3_REPRESENTATIVE)
        },
        **{
            ("linear_accel_exceeds_drive", source, key): 2.0
            for source in (_HYBRID_DEFAULTS, _HYBRID_V3_REPRESENTATIVE)
            for key in ("max_linear_accel",)
        },
        ("linear_accel_exceeds_drive", _HYBRID_DEFAULTS, "actuation_max_linear_accel"): 2.0,
        ("linear_decel_exceeds_drive", _HYBRID_DEFAULTS, "actuation_max_linear_decel"): 2.5,
        ("linear_decel_exceeds_drive", _HYBRID_DEFAULTS, "max_linear_decel"): 2.5,
        ("linear_decel_exceeds_drive", _HYBRID_V3_REPRESENTATIVE, "max_linear_decel"): 2.5,
    },
    # VV-7 geometry audit: planner robot-body radii smaller than the 1.0 m drive body.
    "#9750": {
        ("robot_radius_below_drive", "DWAPlannerConfig defaults", "robot_radius"): 0.25,
        ("robot_radius_below_drive", "configs/algos/dwa_classic.yaml", "robot_radius"): 0.25,
        ("robot_radius_below_drive", _HYBRID_DEFAULTS, "robot_radius_default"): 0.3,
        ("robot_radius_below_drive", _HYBRID_V3_REPRESENTATIVE, "robot_radius_default"): 0.3,
        **{
            ("robot_radius_below_drive", source, "robot_radius_default"): 0.3
            for source in HYBRID_RELEASE_CONFIGS
        },
        ("robot_radius_below_drive", "SocNavPlannerConfig defaults", "predictive_robot_radius"): (
            0.3
        ),
        ("robot_radius_below_drive", _PREDICTION_PLANNER, "predictive_robot_radius"): 0.25,
        # Planner pedestrian bodies smaller than the simulator's 0.4 m pedestrian.
        **{
            ("pedestrian_radius_below_simulator", source, "pedestrian_radius_default"): 0.3
            for source in (*HYBRID_RELEASE_CONFIGS, _HYBRID_DEFAULTS, _HYBRID_V3_REPRESENTATIVE)
        },
        **{
            ("pedestrian_radius_below_simulator", source, "pedestrian_radius"): 0.3
            for source in ("DWAPlannerConfig defaults", "configs/algos/dwa_classic.yaml")
        },
        (
            "pedestrian_radius_below_simulator",
            "SocNavPlannerConfig defaults",
            "predictive_pedestrian_radius",
        ): 0.3,
        (
            "pedestrian_radius_below_simulator",
            _PREDICTION_PLANNER,
            "predictive_pedestrian_radius",
        ): (0.25),
    },
}


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
            "v4_reaction_time",
        }
        or any(token in name for token in UNIT_TOKENS)
    )


def _release_values():
    """Yield each release algo_config resolved for the default and every override scenario."""
    entries = [(entry["algo"], entry.get("algo_config")) for entry in release_campaign_planners()]
    entries += [("hybrid_rule_local_planner", path) for path in HYBRID_V4_RELEASE_TWINS]
    for algo, path in entries:
        if not path:
            continue
        manifest = _load_yaml(path)
        scenarios = {"__default__"}
        for block in ("scenario_overrides", "scenario_algo_overrides"):
            if isinstance(manifest.get(block), dict):
                scenarios.update(manifest[block])
        for scenario in sorted(scenarios):
            resolved_algo, config = resolve_release_algo_config(algo, path, scenario)
            if path in HYBRID_V4_SOURCES and resolved_algo == "hybrid_rule_local_planner":
                assert config.get("planner_variant") == "hybrid_rule_v4_clearance_braking", path
            yield path, dict(_number_leaves(config))


@cache
def _all_values() -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return every audited (source, flattened numeric fields) pair."""
    result: list[tuple[str, dict[str, Any]]] = []
    matrix = _load_yaml(READINESS_MATRIX.relative_to(ROOT))
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
        result.append((entrypoint, dict(_number_leaves(_load_yaml(entrypoint)))))
    result.extend(_release_values())
    result.append((HYBRID_V4_BASE, dict(_number_leaves(_load_yaml(HYBRID_V4_BASE)))))
    for cls in (
        HybridRuleLocalPlannerConfig,
        SocNavPlannerConfig,
        DWAPlannerConfig,
        RiskDWAPlannerConfig,
    ):
        config = cls()
        values = {field.name: getattr(config, field.name) for field in fields(config)}
        result.append((f"{cls.__name__} defaults", values))
    return tuple(result)


def _numbers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def _field_violations(source: str, name: str, numbers: tuple[float, ...]):  # noqa: C901
    """Yield (rule, offending value) pairs for one physical field."""
    largest = max(abs(number) for number in numbers)
    if name in TIME_STEP_FIELDS and not all(0.0 < n <= MAX_TIME_STEP_S for n in numbers):
        yield "time_step_out_of_range", largest
    if (
        name in DRIVE_LINEAR_SPEED_FIELDS
        and source not in LINEAR_SPEED_RUNTIME_CLIPPED_SOURCES
        and largest > DRIVE.max_linear_speed
    ):
        yield "linear_speed_exceeds_drive", largest
    if name in DRIVE_ANGULAR_SPEED_FIELDS and largest > MAX_ANGULAR_SPEED:
        yield "angular_speed_unit_bound", largest
    if name in DRIVE_LINEAR_ACCEL_FIELDS and largest > DRIVE.max_linear_accel:
        yield "linear_accel_exceeds_drive", largest
    if name in DRIVE_LINEAR_DECEL_FIELDS and largest > DRIVE.max_linear_decel:
        yield "linear_decel_exceeds_drive", largest
    if name in DRIVE_ANGULAR_ACCEL_FIELDS and largest > MAX_ANGULAR_ACCEL:
        yield "angular_accel_unit_bound", largest
    if name in ROBOT_RADIUS_FIELDS and min(numbers) < DRIVE.radius:
        yield "robot_radius_below_drive", min(numbers)
    if name in ROBOT_RADIUS_FIELDS and largest > RADIUS_UNIT_FACTOR * DRIVE.radius:
        yield "robot_radius_unit_error", largest
    if name in PEDESTRIAN_RADIUS_FIELDS and min(numbers) < SIM.ped_radius:
        yield "pedestrian_radius_below_simulator", min(numbers)
    if name in PEDESTRIAN_RADIUS_FIELDS and largest > RADIUS_UNIT_FACTOR * SIM.ped_radius:
        yield "pedestrian_radius_unit_error", largest
    if name in OTHER_RADIUS_FIELDS and not all(0.0 < n <= 10.0 * DRIVE.radius for n in numbers):
        yield "radius_unit_bound", largest
    if source in HYBRID_V4_SOURCES:
        return  # v4 speed levels are surface clearances; see _v4_level_violations.
    if name == "stop_distance_human" and min(numbers) < CONTACT_DISTANCE:
        yield "stop_gate_inside_contact", min(numbers)
    if name == "slow_distance_human" and min(numbers) < (
        CONTACT_DISTANCE + DRIVE_STOPPING_DISTANCE
    ):
        yield "slow_gate_below_contact_plus_stopping", min(numbers)
    if name == "moderate_distance_human" and min(numbers) < CONTACT_DISTANCE:
        yield "moderate_gate_inside_contact", min(numbers)


@cache
def computed_violations() -> frozenset[tuple[tuple[str, str, str], float]]:
    """Return every drive-envelope and unit-bound violation across the audited sources."""
    found: set[tuple[tuple[str, str, str], float]] = set()
    for source, values in _all_values():
        for key, value in values.items():
            name = key.rsplit(".", maxsplit=1)[-1]
            if not _unit_bearing(name):
                continue
            numbers = tuple(_numbers(value))
            if not numbers:
                continue
            for rule, offending in _field_violations(source, name, numbers):
                found.add(((rule, source, key), float(offending)))
    return frozenset(found)


def _ledger() -> frozenset[tuple[tuple[str, str, str], float]]:
    return frozenset(
        (identity, float(value))
        for entries in KNOWN_VIOLATIONS.values()
        for identity, value in entries.items()
    )


def _violations_for(issue: str) -> list[str]:
    tracked = set(KNOWN_VIOLATIONS[issue])
    return sorted(
        f"{rule}: {source}.{key}={value}"
        for (rule, source, key), value in computed_violations()
        if (rule, source, key) in tracked
    )


def test_release_campaign_planner_configs_are_audited() -> None:
    """The audit reads the release roster, including its base configs and overrides."""
    entries = release_campaign_planners()
    assert len({entry["key"] for entry in entries}) == 14, "release roster lost or gained arms"
    audited_sources = {source for source, _values in _all_values()}
    for entry in entries:
        if entry.get("algo_config"):
            assert entry["algo_config"] in audited_sources, entry
    # base_config_path inheritance must reach the audited values: the hybrid base
    # file sets stop_distance_human, which no release candidate overrides.
    _algo, resolved = resolve_release_algo_config(
        "hybrid_rule_local_planner", HYBRID_RELEASE_CONFIGS[0]
    )
    assert resolved["planner_variant"] == "hybrid_rule_v3_teb_like_rollout"
    assert resolved["stop_distance_human"] == 0.5
    assert resolved["max_linear_speed"] == 3.0
    # A scenario algo override resolves to its own planner and base config.
    algo, orca = resolve_release_algo_config(
        "hybrid_rule_local_planner", HYBRID_RELEASE_CONFIGS[2], "francis2023_leave_group"
    )
    assert algo == "orca"
    assert orca["max_linear_speed"] == 1.15


def test_representative_planner_physical_field_inventory_and_ranges() -> None:
    """Audit every recognized physical-unit field in all sources for class, sign and finiteness."""
    seen: set[str] = set()
    for source, values in _all_values():
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


def test_drive_related_fields_have_a_rule_or_a_stated_reason() -> None:
    """Every speed, acceleration, radius, time-step and centre-distance name has a rule."""
    ruled = (
        TIME_STEP_FIELDS
        | DRIVE_LINEAR_SPEED_FIELDS
        | DRIVE_ANGULAR_SPEED_FIELDS
        | DRIVE_LINEAR_ACCEL_FIELDS
        | DRIVE_LINEAR_DECEL_FIELDS
        | DRIVE_ANGULAR_ACCEL_FIELDS
        | ROBOT_RADIUS_FIELDS
        | PEDESTRIAN_RADIUS_FIELDS
        | OTHER_RADIUS_FIELDS
        | CENTRE_DISTANCE_GATE_FIELDS
    )
    assert ruled <= AUDITED_UNIT_FIELDS
    needs_rule = {
        name
        for name in AUDITED_UNIT_FIELDS
        if any(token in name for token in ("speed", "accel", "decel", "radius"))
        or name.endswith("_dt")
        or name in {"dt", "control_period", "forecast_variant_dt_s"}
    } | {
        "v_max",
        "omega_max",
        "linear_candidates",
        "angular_candidates",
        "goal_posterior_turn_rate",
        "actuation_max_yaw_rate",
    }
    assert needs_rule <= ruled, sorted(needs_rule - ruled)
    # Centre-distance names are either speed gates (checked) or declared ranges.
    centre_names = {name for name in AUDITED_UNIT_FIELDS if name.endswith(("_distance_human",))}
    assert centre_names == CENTRE_DISTANCE_GATE_FIELDS
    assert not (CENTRE_DISTANCE_RANGE_FIELDS & ruled)


def test_known_drive_envelope_violations_match_the_ledger_exactly() -> None:
    """New, worsened, or fixed violations all fail until the ledger is updated."""
    computed = computed_violations()
    ledger = _ledger()
    unexpected = sorted(
        f"{rule}: {source}.{key}={value}" for (rule, source, key), value in computed - ledger
    )
    resolved = sorted(
        f"{rule}: {source}.{key}={value}" for (rule, source, key), value in ledger - computed
    )
    assert not unexpected and not resolved, (
        f"unlisted violations={unexpected}; listed but not observed={resolved}"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="issue #9726: release hybrid configs exceed the 2.0 m/s and 1.0 m/s^2 drive",
)
def test_release_hybrid_configs_fit_drive_envelope() -> None:
    """The four release hybrid arms must command within the differential drive."""
    violations = [
        item
        for item in _violations_for("#9726")
        if any(source in item for source in HYBRID_RELEASE_CONFIGS)
    ]
    assert not violations, violations


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="issue #9726: hybrid code-default stop/slow centre distances sit inside contact",
)
def test_hybrid_code_default_stop_and_slow_distance_clear_contact() -> None:
    """Code defaults apply wherever YAML omits a field, so they need their own check."""
    config = HybridRuleLocalPlannerConfig()
    violations = []
    if config.stop_distance_human < CONTACT_DISTANCE:
        violations.append(f"stop {config.stop_distance_human} < contact {CONTACT_DISTANCE}")
    if config.slow_distance_human < CONTACT_DISTANCE + DRIVE_STOPPING_DISTANCE:
        violations.append(
            f"slow {config.slow_distance_human} < contact + stopping "
            f"{CONTACT_DISTANCE + DRIVE_STOPPING_DISTANCE}"
        )
    assert not violations, "; ".join(violations)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "issue #9750: planner robot and pedestrian body radii are smaller than the "
        "1.0 m drive body and the 0.4 m simulator pedestrian"
    ),
)
def test_planner_robot_radius_matches_drive_body() -> None:
    """A planner that models smaller bodies than the simulator plans through contact."""
    violations = _violations_for("#9750")
    assert not violations, violations


def test_dwa_clearance_score_span_covers_drive_contact_scale() -> None:
    """DWA's scoring normalization spans contact scale; this is no safety gate."""
    config = DWAPlannerConfig()
    assert config.max_linear_acceleration <= DRIVE.max_linear_accel
    contact = DRIVE.radius + config.pedestrian_radius
    assert config.clearance_distance >= contact


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="issue #9726: selected hybrid v3 centre thresholds and braking exceed drive envelope",
)
def test_selected_hybrid_v3_stop_and_braking_match_drive() -> None:
    """Check only the selected v3: a future v4 must get its own passing case."""
    config = yaml.safe_load(HYBRID_V3.read_text(encoding="utf-8"))
    assert config["planner_variant"] == "hybrid_rule_v3_teb_like_rollout"
    violations = []
    if float(config["stop_distance_human"]) < CONTACT_DISTANCE:
        violations.append("stop centre distance falls inside contact distance")
    if float(config["slow_distance_human"]) < CONTACT_DISTANCE + DRIVE_STOPPING_DISTANCE:
        violations.append("slow centre distance cannot cover drive stopping distance")
    if float(config["max_linear_accel"]) > DRIVE.max_linear_accel:
        violations.append("rollout forward acceleration exceeds drive")
    if float(config["max_linear_decel"]) > DRIVE.max_linear_decel:
        violations.append("rollout braking exceeds drive")
    assert not violations, "; ".join(violations)


def test_hybrid_v4_release_twins_fit_the_drive_rules() -> None:
    """The #9747 v4 twins are audited and break no drive-limit or radius rule."""
    audited = {source for source, _values in _all_values()}
    assert HYBRID_V4_SOURCES <= audited
    violations = sorted(
        f"{rule}: {source}.{key}={value}"
        for (rule, source, key), value in computed_violations()
        if source in HYBRID_V4_SOURCES
    )
    assert not violations, violations


def test_observed_and_planner_read_radii_match_drive_and_simulator() -> None:
    """The observation and the planners read body radii, not diameters or halves.

    The structured observation must carry the drive's robot radius and the
    simulator's pedestrian radius, and the ORCA and hybrid v4 adapters must read
    those exact values from it.
    """
    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.planner.hybrid_rule_local_planner import (
        HybridRuleLocalPlannerAdapter,
        build_hybrid_rule_local_planner_config,
    )
    from robot_sf.planner.socnav_orca import ORCAPlannerAdapter
    from tests.metamorphic.planner_arms import interaction_scene, robot_env_config

    config = robot_env_config(interaction_scene(), max_steps=3)
    env = make_robot_env(config=config, seed=8244)
    try:
        observation, _info = env.reset(seed=8244)
    finally:
        env.close()
    robot_radius = float(observation["robot_radius"][0])
    pedestrian_radius = float(observation["pedestrians_radius"][0])
    assert robot_radius == pytest.approx(DRIVE.radius)
    assert pedestrian_radius == pytest.approx(SIM.ped_radius)

    orca = ORCAPlannerAdapter(SocNavPlannerConfig())
    robot_state, _goal_state, ped_state = orca._socnav_fields(observation)
    assert float(robot_state["radius"][0]) == pytest.approx(DRIVE.radius)
    assert orca._extract_pedestrians(ped_state)[3] == pytest.approx(SIM.ped_radius)

    v4 = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(_load_yaml(HYBRID_V4_BASE))
    )
    state = v4._extract_state(observation)
    assert state["robot_radius"] == pytest.approx(DRIVE.radius)
    assert state["ped_radius"] == pytest.approx(SIM.ped_radius)
