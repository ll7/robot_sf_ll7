"""Real evaluator witnesses for opt-in feasibility diagnostics (#10092)."""

from types import SimpleNamespace

import numpy as np

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)
from tests.planner.test_hybrid_rule_local_planner import _obs


def _planner(**extra):
    cfg = build_hybrid_rule_local_planner_config(
        dict(
            planner_variant="hybrid_rule_v4_clearance_braking",
            debug_candidate_evaluator=True,
            hard_safety_margin=0.2,
            continuous_static_clearance_enabled=True,
            route_guide_enabled=False,
            **extra,
        )
    )
    planner = HybridRuleLocalPlannerAdapter(cfg)
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                map_def=SimpleNamespace(width=30, height=30),
                get_obstacle_lines=lambda: np.array([[0, 10, 30, 10]]),
            )
        )
    )
    return planner


def test_debug_forced_stop_reports_the_actual_exclusion_radius():
    """A physical 0.10 m clearance is rejected by a 0.20 m comfort margin."""
    planner = _planner()
    obs = _obs(robot=(4, 10.35), goal=(20, 10.35))  # body radius 0.25 m
    planner.plan(obs)
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    assert debug["chosen_stop"] and debug["stop_kind"] == "FORCED"
    assert debug["feasible_moving_count"] == 0
    assert debug["best_feasible_moving"] is None
    assert debug["constraints"] == [
        {
            "constraint": "exact_body_wall_overlap",
            "threshold_name": "exclusion_radius_m",
            "threshold": 0.45,
            "rejected": debug["candidate_count"],
        }
    ]


def test_debug_preferred_stop_retains_a_feasible_moving_witness():
    """An intentionally stop-favouring objective must not be called infeasible."""
    planner = _planner(
        goal_progress_weight=0,
        speed_preference_weight=0,
        freezing_weight=0,
        velocity_smoothness_weight=20,
        control_effort_weight=20,
    )
    planner.plan(_obs(robot=(4, 15), goal=(20, 15)))
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    assert debug["stop_kind"] == "PREFERRED"
    assert debug["feasible_moving_count"] > 0
    moving = debug["best_feasible_moving"]
    assert moving["command"][0] > 0
    assert moving["score"] < planner.last_decision()["selected_score"]


def test_opt_in_physical_exclusion_passes_clear_doorway_and_rejects_contact():
    """A 1 m body fits a 2.20 m doorway, while wall overlap remains forbidden."""
    from robot_sf.planner.hybrid_rule_local_planner import HybridRuleCandidate

    accepted = []
    for enabled in (False, True):
        planner = _planner(physical_static_exclusion_enabled=enabled)
        planner.bind_env(
            SimpleNamespace(
                simulator=SimpleNamespace(
                    map_def=SimpleNamespace(width=30, height=30),
                    get_obstacle_lines=lambda: np.array([[0, 10, 30, 10], [0, 12.2, 30, 12.2]]),
                )
            )
        )
        obs = _obs(robot=(4, 11.1), goal=(20, 11.1))
        obs["robot"]["radius"] = np.array([1.0])
        state = planner._extract_state(obs)
        result = planner._evaluate_candidate(
            candidate=HybridRuleCandidate(0.3, 0, "forward"),
            observation=obs,
            state=state,
            speed_cap=2,
            nearest_ped=float("inf"),
        )
        accepted.append(result["accepted"])
        if enabled:
            assert result["accepted"], "comfort margin still excludes a physically clear doorway"
            # Move the body 0.15 m toward the wall: radius 1.0 now overlaps by 0.05 m.
            obs["robot"]["position"] = np.array([4, 10.95])
            result = planner._evaluate_candidate(
                candidate=HybridRuleCandidate(0.3, 0, "forward"),
                observation=obs,
                state=planner._extract_state(obs),
                speed_cap=2,
                nearest_ped=float("inf"),
            )
            assert not result["accepted"]
            assert result["continuous_static_collision"]
            assert result["hard_static_clearance"] == 1.0
    assert accepted == [False, True], "comfort margin still excludes a physically clear doorway"


def test_opt_in_clearance_preference_uses_surface_gap():
    """A 0.40 m free surface gap earns 0.40/0.70 clearance preference, not 1."""
    import pytest

    from robot_sf.planner.hybrid_rule_local_planner import HybridRuleCandidate

    planner = _planner(physical_static_exclusion_enabled=True, desired_static_clearance=0.7)
    planner.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                map_def=SimpleNamespace(width=6.2, height=6.2),
                get_obstacle_lines=lambda: np.array([[0, 0.6, 6.2, 0.6]]),
            )
        )
    )
    obs = _obs(robot=(2, 2), goal=(5, 2))
    obs["robot"]["radius"] = np.array([1.0])
    grid = np.zeros((3, 31, 31))
    grid[0, 3, :] = 1  # y = 3 * 0.2 = 0.6; centre-to-wall = 1.4 m
    obs["occupancy_grid"] = grid
    obs["occupancy_grid_meta"] = {
        "origin": [0, 0],
        "resolution": [0.2],
        "size": [6.2, 6.2],
        "use_ego_frame": [0],
        "center_on_robot": [0],
        "channel_indices": [0, 1, 2],
        "robot_pose": [2, 2, 0],
    }
    result = planner._evaluate_candidate(
        candidate=HybridRuleCandidate(0.3, 0, "forward"),
        observation=obs,
        state=planner._extract_state(obs),
        speed_cap=2,
        nearest_ped=float("inf"),
    )
    assert result["accepted"]
    assert result["terms"]["static_clearance"] == pytest.approx(4 / 7)


def test_debug_records_dynamic_and_angular_constraints_in_physical_units():
    """Independent pedestrian gates remain hard, even with physical static exclusion."""
    planner = _planner(physical_static_exclusion_enabled=True, control_period=1.2)
    planner.plan(
        _obs(robot=(4, 15), goal=(20, 15), ped_positions=[(4.65, 15)], ped_velocities=[(0, 0)])
    )
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    constraints = {r["constraint"]: r for r in debug["constraints"]}
    assert constraints["dynamic_collision"]["threshold"] == 0.7  # 0.25 + 0.25 + 0.20
    assert constraints["excessive_angular_near_human"]["threshold"] == 0.7
    assert debug["constraint_context"]["near_human_activation_distance_m"] == 0.8
    assert debug["stop_kind"] == "FORCED"


def test_debug_records_braking_rejection_with_short_contact_horizon():
    """A contact after the short hard horizon still rejects an unsafe braking tail."""
    planner = _planner(hard_collision_horizon=0.2)
    planner.plan(
        _obs(
            robot=(4, 15),
            goal=(20, 15),
            speed=1.9,
            ped_positions=[(8, 15)],
            ped_velocities=[(-3, 0)],
        )
    )
    debug = (planner.last_decision() or {}).get("candidate_evaluator_debug")
    assert debug is not None, "opt-in evaluator evidence missing"
    constraints = {r["constraint"]: r for r in debug["constraints"]}
    assert constraints["braking_infeasible"]["threshold"] == 0.7
    assert constraints["braking_infeasible"]["rejected"] > 0
