"""Pure candidate fixtures for the diagnostic guard ablation; no simulator stepping."""

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.planner.hybrid_rule_local_planner import (
    HybridRuleCandidate,
    HybridRuleLocalPlannerAdapter,
    build_hybrid_rule_local_planner_config,
)


def evaluate(disable, monkeypatch, static_reason=None):
    """Use fixed geometry with a colliding pedestrian and known rollout distances."""
    planner = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config(
            {
                "planner_variant": "hybrid_rule_v4_clearance_braking",
                "ablation_disable_hard_guard": disable,
                "rollout_horizon": 0.4,
                "rollout_dt": 0.2,
                "v4_braking_check_enabled": False,
                "continuous_static_clearance_enabled": static_reason == "static_collision",
            }
        )
    )
    monkeypatch.setattr(
        planner, "_min_obstacle_clearance", lambda *_: 0.2 if static_reason else 9.0
    )
    if static_reason == "static_collision":
        planner._continuous_static_context = SimpleNamespace()
        monkeypatch.setattr(planner, "_continuous_static_collision", lambda *_: True)
    state = {
        "robot_pos": np.array([0.0, 0.0]),
        "heading": 0.0,
        "current_speed": 0.0,
        "goal": np.array([20.0, 0.0]),
        "ped_pos": np.array([[1.15, 0.0]]),
        "ped_vel": np.array([[0.0, 0.0]]),
        "robot_radius": 1.0,
        "ped_radius": 0.4,
        "dt": 0.1,
        "observation": {},
    }
    return planner._evaluate_candidate(
        candidate=HybridRuleCandidate(0.2, 0.0, "dynamic_window"),
        observation={},
        state=state,
        speed_cap=0.2,
        nearest_ped=1.15,
    )


@pytest.mark.parametrize("reason", ["dynamic_collision", "static_collision", "static_clearance"])
def test_guard_off_scores_colliding_candidate_without_losing_clearance(monkeypatch, reason):
    """Guard-on vetoes; guard-off evaluates all rollout steps and retains soft clearance."""
    static = None if reason == "dynamic_collision" else reason
    guarded = evaluate(False, monkeypatch, static)
    assert guarded["accepted"] is False
    assert guarded["reason"] == reason
    unguarded = evaluate(True, monkeypatch, static)
    assert unguarded["accepted"] is True
    # Two 0.2-second steps at 0.2 m/s: the final pedestrian distance is 1.15 - 0.08.
    assert unguarded["min_dynamic_clearance"] == pytest.approx(1.07)
    assert unguarded["terms"]["dynamic_clearance"] == 0.0
    assert unguarded["min_static_clearance"] == (0.2 if static else 9.0)


def test_guard_ablation_keeps_near_human_angular_speed_limit():
    """The experimental rejection bypass cannot remove an action speed limit."""
    planner = HybridRuleLocalPlannerAdapter(
        build_hybrid_rule_local_planner_config({"ablation_disable_hard_guard": True})
    )
    rejection = planner._reject_excessive_angular_near_human(
        HybridRuleCandidate(0.0, 1.0, "dynamic_window"), 0.1
    )
    assert rejection["reason"] == "excessive_angular_near_human"
