"""Physical PPO action-contract invariants; no learning or checkpoint hydration."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf.baselines.ppo import PPOPlanner
from robot_sf.benchmark.map_runner import map_runner
from robot_sf.benchmark.map_runner_policies.map_runner_actions import policy_command_to_env_action
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.robot.differential_drive import DifferentialDriveState
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.training import train_ppo

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml"
OLD = ROOT / (
    "configs/training/ppo/ablations/"
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml"
)


@pytest.mark.parametrize(
    ("speed", "output", "expected"),
    [
        ((0.6, 0.2), (-0.05, -0.04), (0.55, 0.16)),
        ((0.6, 0.2), (-0.25, -0.5), (0.5, 0.1)),
        ((1.95, 0.95), (0.5, 0.5), (2.0, 1.0)),
        ((0.05, -0.95), (-0.5, -0.5), (0.0, -1.0)),
    ],
)
def test_policy_output_has_identical_training_and_benchmark_motion(speed, output, expected):
    """Exercise both real env.step paths, including braking, caps and turn-rate ramp."""
    config = RobotSimulationConfig(ppo_action_semantics="velocity_delta")
    config.sim_config.sim_time_in_secs = 60.0
    benchmark_config = deepcopy(config)
    benchmark_config.ppo_action_semantics = "acceleration"
    training = make_robot_env(config=config, seed=1001)
    benchmark = make_robot_env(config=benchmark_config, seed=1001)
    try:
        training.reset(seed=1001)
        benchmark.reset(seed=1001)
        for env in (training, benchmark):
            robot = env.simulator.robots[0]
            robot.state = DifferentialDriveState(pose=((5.0, 5.0), 0.0), velocity=speed)
            # Initialise wheel state consistently with physical velocity.
            robot.state.wheel_speeds = robot.movement._resulting_wheel_speeds(speed)
        action = np.asarray(output, dtype=np.float32)
        planner_cfg = map_runner._ppo_planner_config(yaml.safe_load(BASELINE.read_text()))
        planner = PPOPlanner(planner_cfg, defer_model_loading=True)
        planner._initialized = True
        # Fixed policy output isolates the adapter/plant contract from model inference.
        planner._model = SimpleNamespace(predict=lambda *_args, **_kwargs: (action, None))
        command = planner.step({"robot_speed": np.asarray(speed, dtype=np.float32)})
        native = policy_command_to_env_action(
            env=benchmark,
            config=benchmark_config,
            command=(command["v"], command["omega"]),
        )
        training.step(action)
        benchmark.step(native)
        train_robot = training.simulator.robots[0]
        bench_robot = benchmark.simulator.robots[0]
        # Independent oracle: signed target deltas, then ±1 acceleration × 0.1 s.
        np.testing.assert_allclose(train_robot.current_speed, expected, atol=5e-8, rtol=0)
        np.testing.assert_array_equal(train_robot.current_speed, bench_robot.current_speed)
        np.testing.assert_array_equal(train_robot.pos, bench_robot.pos)
        assert train_robot.pose[1] == bench_robot.pose[1]
    finally:
        training.close()
        benchmark.close()


def test_original_recipe_resolves_legacy_goal_and_acceleration():
    """Record actual old defaults across the current scenario surface, rather than infer from YAML."""
    old = train_ppo.load_expert_training_config(OLD)
    scenarios = load_scenarios(old.scenario_config)
    assert len(scenarios) == 71
    for scenario in scenarios:
        config = build_robot_config_from_scenario(scenario, scenario_path=old.scenario_config)
        train_ppo._apply_env_overrides(config, old.env_overrides)
        assert config.ppo_action_semantics == "acceleration"
        assert config.robot_config.max_linear_speed == 3.0
        assert config.robot_config.allow_backwards is True
        assert config.sim_config.goal_completion_policy is None
        assert all(
            map_def.goal_completion_policy == "waypoint_radius_v1"
            for map_def in config.map_pool.map_defs.values()
        )
    assert old.evaluation.evaluation_seeds == ()
    assert old.evaluation.hold_out_scenarios == ()
