"""Observe the configured PPO action contract at the native plant."""

from pathlib import Path

import numpy as np

from robot_sf.training.scenario_loader import load_scenarios
from scripts.training.train_ppo import _make_training_env, load_expert_training_config


def test_configured_delta_actions_apply_velocity_change():
    """Small signed commands change speed per step, including braking at limits."""
    recipe = load_expert_training_config(
        "configs/training/ppo/expert_ppo_release_contract_b_seed1002.yaml"
    )
    path = Path("configs/scenarios/archetypes/issue_596_frame_consistency.yaml")
    scenario = next(s for s in load_scenarios(path) if s["name"] == "empty_map_8_directions_east")
    env = _make_training_env(
        1001,
        scenario=scenario,
        scenario_definitions=None,
        scenario_path=path,
        exclude_scenarios=(),
        suite_name="contract",
        algorithm_name="ppo",
        env_overrides=recipe.env_overrides,
        env_factory_kwargs=recipe.env_factory_kwargs,
        scenario_sampling={},
    )()
    try:
        env.reset(seed=1001)
        for state, output, expected in [
            ((0.6, 0.2), (-0.05, -0.04), (0.55, 0.16)),
            ((0.6, 0.2), (-0.25, -0.1), (0.5, 0.1)),
            ((1.95, 0.95), (1.0, 0.5), (2.0, 1.0)),
            ((0.05, -0.95), (-1.0, -0.5), (0.0, -1.0)),
        ]:
            env.simulator.robots[0].state.velocity = state
            obs, *_ = env.step(np.array(output))
            # Literal oracle: dt=.1, acceleration limit=1, no reverse.
            np.testing.assert_allclose(obs["robot_speed"], expected, atol=1e-6, rtol=0)
        np.testing.assert_array_equal(env.action_space.low, [-2.0, -1.0])
        np.testing.assert_array_equal(env.action_space.high, [2.0, 1.0])
    finally:
        env.close()


def test_recovery_recipes_load_disjoint_selection_manifest():
    """Campaign preflight must accept the authored pilot and full recipes."""
    for name, expected_steps in [("pilot", 983040), ("15m", 15000000)]:
        recipe = load_expert_training_config(f"configs/training/ppo/dev_delta_recovery_{name}.yaml")
        assert recipe.total_timesteps == expected_steps
        pool = set(recipe.scenario_sampling["episode_seed_pool"])
        selection = set(recipe.evaluation.evaluation_seeds)
        assert len(pool) == 20
        assert len(selection) == 10
        assert pool.isdisjoint(selection)
