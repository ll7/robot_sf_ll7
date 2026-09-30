"""Regression witnesses for #10064: a rear/lateral approach needs motion."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, build_guarded_ppo_config
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config
from tests.planner.test_predictive_mppi_planner import _StubPredictor
from tests.planner.test_risk_dwa import _observation

ROOT = Path(__file__).parents[2]


def _risk_config():
    """Read the actual release command/dynamics contract."""
    return build_risk_dwa_config(
        yaml.safe_load((ROOT / "configs/algos/risk_dwa_release_v0_0_8.yaml").read_text())
    )


@pytest.mark.parametrize("ped,velocity", [((-1.6, 0.21), (1.04, -0.05)), ((0.0, 1.0), (0.0, -0.8))])
def test_all_rejected_rear_or_lateral_commands_move_and_improve_clearance(ped, velocity):
    """Every reachable rollout fails, but a moving command beats braking."""
    planner = RiskDWAPlannerAdapter(_risk_config())
    obs = _observation(goal=(7.0, -0.8), pedestrians=[ped], pedestrian_velocities=[velocity])
    robot, heading, goal, peds, velocities = planner._extract_robot_goal_ped(obs)
    for v in (0.0, 0.1):
        for w in (-0.1, 0.0, 0.1):
            assert planner._rollout_score(
                robot_pos=robot,
                heading=heading,
                goal=goal,
                command=(v, w),
                ped_pos=peds,
                ped_vel=velocities,
                observation=obs,
                current_speed=0.0,
            ) == float("-inf")
    command = planner.plan(obs)
    assert command[0] > 0.0

    # Independent horizon integration and pedestrian forecast, not the rank helper.
    def minimum(cmd):
        pos = robot.copy()
        angle = heading
        clearances = []
        for k in range(1, 17):
            pos += cmd[0] * 0.1 * np.array([np.cos(angle), np.sin(angle)])
            angle += cmd[1] * 0.1
            clearances.append(np.linalg.norm(peds[0] + velocities[0] * k * 0.1 - pos) - 1.4)
        return min(clearances)

    assert minimum(command) > minimum((0.0, 0.0)) + 0.01
    assert planner.diagnostics()["no_admissible_command"] is True
    assert planner.diagnostics()["recovery_kind"] == "least_bad_clearance"


def test_infeasible_progress_escape_is_selectable():
    """An escape outside the lattice wins even though its normal score is -inf."""
    planner = RiskDWAPlannerAdapter(
        replace(
            _risk_config(),
            dynamic_window_version="fixed_v1",
            linear_candidates=(0.0, 0.2),
            angular_candidates=(0.0,),
            max_linear_speed=0.6,
        )
    )
    command = planner.plan(
        _observation(
            goal=(7.0, 0.0), pedestrians=[(-1.6, 0.21)], pedestrian_velocities=[(1.04, 0.0)]
        )
    )
    assert command == pytest.approx((0.55, 0.0))
    assert planner.diagnostics()["no_admissible_command"] is True
    assert planner.diagnostics()["recovery_kind"] == "progress_escape"


def test_mppi_infeasible_escape_beats_zero_with_real_clearance_costs():
    """MPPI must retain an improving escape when every cost is invalid."""
    cfg = build_predictive_mppi_config(
        {
            "random_seed": 1001,
            "sample_count": 8,
            "iterations": 1,
            "init_linear_std": 0.0,
            "init_angular_std": 0.0,
            "rollout_dt": 0.1,
            "clearance_model": "surface_v2",
            "predictive_clearance_model": "surface_v2",
            "predictive_robot_radius": 1.0,
            "predictive_pedestrian_radius": 0.4,
        }
    )
    planner = PredictiveMPPIAdapter(cfg, allow_fallback=True)
    future = np.array([[[-1.6 + 1.04 * k * 0.1, 0.21] for k in range(1, 9)]])
    planner._predictor = _StubPredictor(future, anchor=(0.0, 0.0))
    obs = _observation(
        goal=(7.0, 0.0), pedestrians=[(-1.6, 0.21)], pedestrian_velocities=[(1.04, 0.0)]
    )
    command = planner.plan(obs)
    assert command[0] > 0.0
    assert planner.diagnostics()["no_admissible_command"] is True
    assert planner.diagnostics()["recovery_kind"] == "progress_escape"


def test_guard_infeasible_fallback_ignores_vetoed_ppo_rank():
    """A vetoed faster PPO proposal cannot force the guard to rear-endable stasis."""
    cfg = build_guarded_ppo_config(
        yaml.safe_load((ROOT / "configs/algos/guarded_ppo_release_v0_0_8.yaml").read_text())
    )
    guard = GuardedPPOAdapter(config=cfg, fallback_adapter=RiskDWAPlannerAdapter(_risk_config()))
    obs = _observation(
        goal=(7.0, 0.0), pedestrians=[(-1.6, 0.21)], pedestrian_velocities=[(1.04, 0.0)]
    )
    command, label = guard.choose_command(obs, (0.2, 0.0))
    assert command[0] > 0.0
    assert label == "fallback_best_effort"
    assert guard.diagnostics()["no_admissible_command"] is True
    assert guard.diagnostics()["recovery_kind"] == "least_bad_clearance"
