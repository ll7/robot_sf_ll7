"""Real training/release transitions, with an independently calculated plant oracle."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.baselines.ppo import PPOPlanner
from robot_sf.benchmark.map_runner import map_runner
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner_policies.map_runner_actions import policy_command_to_env_action
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training import imitation_config
from robot_sf.training.scenario_loader import load_scenarios
from scripts.training.train_ppo import (
    _apply_env_overrides,
    _deterministic_eval_seed_for_episode,
    _make_training_env,
    _randomize_eval_seeds,
    load_expert_training_config,
)

LEAVES = [
    f"configs/training/ppo/{'ablations/' if arm == 'a' else ''}expert_ppo_release_contract_{arm}_seed{seed}.yaml"
    for arm in ("a", "b")
    for seed in (1001, 1002)
]


@pytest.mark.parametrize("leaf", LEAVES)
def test_training_release_applied_command_parity(leaf):
    """Actual env.step must match the release delta adapter, including braking and limits."""
    recipe = load_expert_training_config(Path(leaf))
    path = Path("configs/scenarios/archetypes/issue_596_frame_consistency.yaml")
    scenario = next(s for s in load_scenarios(path) if s["name"] == "empty_map_8_directions_east")
    overrides = dict(recipe.env_overrides, predictive_foresight_device="cpu")
    training = _make_training_env(
        1001,
        scenario=scenario,
        scenario_definitions=None,
        scenario_path=path,
        exclude_scenarios=(),
        suite_name="contract-test",
        algorithm_name=recipe.policy_id,
        env_overrides=overrides,
        env_factory_kwargs=recipe.env_factory_kwargs,
        scenario_sampling=recipe.scenario_sampling,
    )()
    config = build_env_config(scenario, scenario_path=path)
    # Release observation builder plus the shipped per-arm foresight declaration.
    arm = "ppo" if "/ablations/" in leaf else "guarded_ppo"
    seed = recipe.seeds[0]
    release_path = (
        f"configs/baselines/ppo_release_contract_seed{seed}_cpu.yaml"
        if arm == "ppo"
        else f"configs/algos/guarded_ppo_release_contract_seed{seed}_cpu.yaml"
    )
    algo = yaml.safe_load(Path(release_path).read_text())
    _apply_env_overrides(
        config, {k: v for k, v in algo.items() if k.startswith("predictive_foresight_")}
    )
    evaluation = make_robot_env(config=config, seed=1001)
    planner = PPOPlanner(map_runner._ppo_planner_config(algo), defer_model_loading=True)
    kinematics = map_runner.resolve_benchmark_kinematics_model(
        robot_kinematics="differential_drive",
        command_limits=algo,
    )
    try:
        train_obs, _ = training.reset(seed=1001)
        eval_obs, _ = evaluation.reset(seed=1001)
        np.testing.assert_array_equal(training.action_space.low, [-2, -1])
        np.testing.assert_array_equal(training.action_space.high, [2, 1])
        assert training.observation_space == evaluation.observation_space
        for key in train_obs:
            np.testing.assert_allclose(train_obs[key], eval_obs[key], atol=1e-6, err_msg=key)
        # Independent reference: target=current+delta, speed clip, acceleration clip, dt.
        for state, output in [
            ((0.6, 0.2), (-0.05, -0.04)),
            ((0.6, 0.2), (-0.25, -0.1)),
            ((1.95, 0.95), (1.0, 0.5)),
            ((0.05, -0.95), (-1.0, -0.5)),
        ]:
            training.simulator.robots[0].state.velocity = state
            evaluation.simulator.robots[0].state.velocity = state
            training.state.sensors.reset_cache()
            evaluation.state.sensors.reset_cache()
            train_speed = training.state.sensors.next_obs()["robot_speed"]
            eval_speed = evaluation.state.sensors.next_obs()["robot_speed"]
            np.testing.assert_array_equal(train_speed, np.asarray(state, dtype=np.float32))
            np.testing.assert_array_equal(train_speed, eval_speed)
            observed_state = evaluation.state.sensors.next_obs()
            observed_speed = planner._current_unicycle_speed(observed_state)
            command = planner._action_vec_to_dict_from_array(np.array(output), observed_speed)
            accel = policy_command_to_env_action(
                env=evaluation,
                config=config,
                command=map_runner._project_with_feasibility(
                    model=kinematics,
                    command=(command["v"], command["omega"]),
                    meta={},
                ),
            )
            train_after, *_ = training.step(np.array(output))
            eval_after, *_ = evaluation.step(accel)
            np.testing.assert_array_equal(train_after["robot_speed"], eval_after["robot_speed"])
            target = np.clip(
                np.array(state, dtype=np.float32).astype(float) + output, [0, -1], [2, 1]
            )
            expected = np.clip(
                np.array(state) + 0.1 * np.clip((target - state) / 0.1, -1, 1), [0, -1], [2, 1]
            )
            np.testing.assert_allclose(
                training.simulator.robots[0].current_speed,
                evaluation.simulator.robots[0].current_speed,
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                training.simulator.robots[0].current_speed, expected, atol=1e-12
            )
            np.testing.assert_array_equal(
                train_after["robot_speed"], np.asarray(expected, dtype=np.float32)
            )
    finally:
        training.close()
        evaluation.close()
        planner.close()


@pytest.mark.parametrize("leaf", LEAVES)
def test_release_contract_effective_development_eval_seeds(leaf):
    """Selection uses the same explicit dev seed rather than the training seed fallback."""
    recipe = load_expert_training_config(Path(leaf))
    assert recipe.evaluation.evaluation_seeds == (1003,)
    assert recipe.evaluation.randomize_seeds is False
    assert _randomize_eval_seeds(recipe) is False
    assert tuple(
        _deterministic_eval_seed_for_episode(recipe, episode_idx=i, scenario_cycle_length=1)
        for i in range(3)
    ) == (1003, 1003, 1003)


@pytest.mark.parametrize("key", ["evaluation_seeds", "future_eval_knob"])
def test_loader_rejects_schema_valid_unconsumed_evaluation_key(tmp_path, monkeypatch, key):
    """The shared schema permits dataclass fields the YAML loader must not silently drop."""
    leaf = Path(LEAVES[0]).resolve()
    # Simulate a future dataclass/schema field without teaching the loader to consume it.
    monkeypatch.setattr(
        imitation_config, "_EVALUATION_KEYS", imitation_config._EVALUATION_KEYS | {key}
    )
    config = tmp_path / "ignored-evaluation-key.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "base_config": str(leaf),
                "evaluation": {
                    key: [1003],
                    "evaluation_seed_manifest": str(
                        Path(
                            "configs/training/ppo/ppo_release_contract_dev_eval_seeds.yaml"
                        ).resolve()
                    ),
                },
            }
        )
    )
    with pytest.raises(ValueError, match=f"Unconsumed evaluation keys: {key}"):
        load_expert_training_config(config)


@pytest.mark.parametrize("key", ["full_policy_analysis_on_new_best", "full_policy_analysis_videos"])
def test_loader_accepts_enabled_legacy_evaluation_switch(tmp_path, key):
    """Tracked recipes retain the historical no-op handling of legacy analysis switches."""
    config = tmp_path / "unsupported-evaluation-feature.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "base_config": str(Path(LEAVES[0]).resolve()),
                "evaluation": {
                    key: True,
                    "evaluation_seed_manifest": str(
                        Path(
                            "configs/training/ppo/ppo_release_contract_dev_eval_seeds.yaml"
                        ).resolve()
                    ),
                },
            }
        )
    )
    recipe = load_expert_training_config(config)
    assert recipe.evaluation.evaluation_seeds == (1003,)
