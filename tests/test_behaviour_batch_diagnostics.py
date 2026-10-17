"""Supplemental batch probes; these never produce canonical receipt rows.

Bicycle binding is explicit. The optional arbitrator is called explicitly through
its old and additive APIs; neither probe activates a release default.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import robot_sf.planner.multimodal_trajectory_arbitrator as arbitration
from robot_sf.planner.classic_planner_adapter import PlannerActionAdapter
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter, SocNavPlannerConfig
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from tests.test_multimodal_trajectory_arbitration import (
    _action,
    _forecast,
    _join_belief,
    _mode,
    _risk_config,
)


@pytest.mark.parametrize("search", ["action", "sequence"])
def test_diagnostic_bicycle_binding_and_radius_share_the_executed_plant(search):
    """Compare bound/unbound paths without mutating the plant or reusable config."""
    config = SocNavPlannerConfig(
        predictive_rollout_dt=0.2,
        predictive_robot_radius=0.1,
        predictive_pedestrian_radius=0.4,
        predictive_clearance_model="surface_v2",
    )
    planner = PredictionPlannerAdapter(config)
    robot = BicycleDriveRobot(BicycleDriveSettings(radius=0.64, max_accel=0.7))
    observation = {"robot": {"speed": [0.0]}, "sim": {"timestep": [0.1]}}

    def rollout():
        if search == "action":
            return planner._rollout_robot(v=1.0, w=0.0, dt=0.2, steps=3, observation=observation)
        return planner._rollout_robot_sequence(
            sequence=[(1.0, 0.0)], segment_steps=3, dt=0.2, observation=observation
        )[0]

    unbound = rollout()
    planner.bind_env(SimpleNamespace(simulator=SimpleNamespace(robots=[robot])))
    bound = rollout()
    assert robot.state.velocity == 0.0
    np.testing.assert_allclose(robot.pos, (0.0, 0.0))
    adapter = PlannerActionAdapter(robot, robot.action_space, 0.1)
    expected = []
    for _ in range(3):
        for _ in range(2):
            robot.apply_action(tuple(adapter.from_velocity_command((1.0, 0.0))), 0.1)
        expected.append(robot.pos)
    np.testing.assert_allclose(bound, expected, atol=1e-8)
    assert not np.allclose(bound, unbound)
    future = np.array([[[1.5, 0.0]]])
    assert planner._min_predicted_distance(future_peds=future, mask=np.ones(1)) == pytest.approx(
        1.5 - 0.64 - 0.4
    )
    planner.bind_env(SimpleNamespace())
    np.testing.assert_allclose(rollout(), unbound)
    assert planner._min_predicted_distance(future_peds=future, mask=np.ones(1)) == pytest.approx(
        1.0
    )
    assert config.predictive_robot_radius == 0.1


@pytest.mark.parametrize("alpha", [0.9, math.nextafter(1.0, 0.0)])
def test_diagnostic_optional_arbitrator_tail_survives_evaluate_once_and_reorder(alpha):
    """Real mode estimation agrees across explicit APIs, even for a tiny tail."""
    forecast, belief = _join_belief(_forecast(1, (_mode("one", (1.0, 0.0), covariance=0.05),)))
    candidates = [_action("a"), _action("b")]
    kwargs = {
        "belief": belief,
        "risk_config": replace(_risk_config(), seed=1001),
        "arbitration_config": arbitration.MultimodalArbitrationConfig(
            cvar_alpha=alpha, uncertainty_quantile=0.5
        ),
        "route_progress": {"a": 0.0, "b": 0.0},
    }
    combined = arbitration.arbitrate_multimodal_trajectories(candidates, forecast, **kwargs)
    assert combined.status == "selected"
    for evaluation in combined.evaluations:
        assert evaluation.worst_mode_risk > 0.0
        assert evaluation.cvar_mode_risk == pytest.approx(evaluation.worst_mode_risk)
    base = arbitration.evaluate_multimodal_trajectories_base(candidates, forecast, **kwargs)
    assert base.evaluations == combined.evaluations
    reordered = arbitration.reorder_multimodal_trajectories(
        base, ["a", "b"], {"a": 1.0, "b": 0.0}, switch_cost_scale=1.0
    )
    assert reordered.status == "selected"
    assert reordered.selected_candidate_id == "b"
    assert reordered.candidate_set_id == base.candidate_set_id
    original = {item.candidate_id: item for item in base.evaluations}
    for item in reordered.evaluations:
        assert item.cvar_mode_risk == original[item.candidate_id].cvar_mode_risk
        assert item.track_risks == original[item.candidate_id].track_risks
    assert all(item.switch_cost == 0.0 for item in base.evaluations)


@pytest.mark.parametrize("search", ["action", "sequence"])
@pytest.mark.parametrize("separation,admitted", [(1.5, False), (1.8, True)])
def test_diagnostic_margin_admission_uses_bound_radius_for_both_searches(
    search, separation, admitted
):
    """Radius binding also governs the new synchronized surface-margin guard."""
    scores = []
    for configured_radius in (0.1, 0.64):
        planner = PredictionPlannerAdapter(
            SocNavPlannerConfig(
                predictive_robot_radius=configured_radius,
                predictive_pedestrian_radius=0.4,
                predictive_safe_distance=0.6,
                predictive_clearance_model="surface_v2",
            )
        )
        planner.bind_env(
            SimpleNamespace(
                simulator=SimpleNamespace(
                    robots=[SimpleNamespace(config=BicycleDriveSettings(radius=0.64))]
                )
            )
        )
        kwargs = {
            "observation": {
                "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
                "goal": {"current": [3.0, 0.0]},
                "sim": {"timestep": [0.1]},
            },
            "future_peds": np.array([[[separation, 0.0]]]),
            "mask": np.ones(1),
            "steps": 1,
        }
        score = (
            planner._score_action(**kwargs, v=0.0, w=0.0)
            if search == "action"
            else planner._score_action_sequence(**kwargs, sequence=[(0.0, 0.0)])
        )
        assert bool(np.isfinite(score)) is admitted
        scores.append(score)
    assert scores[0] == pytest.approx(scores[1])


def test_diagnostic_bicycle_braking_cover_includes_control_tick_interior():
    """At 1 m/s with 1 m/s² braking, each midpoint exceeds its chord by 5 mm."""
    planner = PredictionPlannerAdapter(SocNavPlannerConfig(predictive_rollout_dt=0.2))
    robot = BicycleDriveRobot(BicycleDriveSettings(max_decel=1.0))
    planner.bind_env(SimpleNamespace(simulator=SimpleNamespace(robots=[robot])))
    observation = {"robot": {"speed": [1.0]}, "sim": {"timestep": [0.1]}}
    cover = planner._interval_motion_cover(observation, np.zeros((3, 2)))
    np.testing.assert_allclose(cover, [0.005, 0.005, 0.005], atol=1e-12)
    assert robot.state.velocity == 0.0
