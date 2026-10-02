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
