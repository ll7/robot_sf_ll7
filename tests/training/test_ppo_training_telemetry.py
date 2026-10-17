"""Training diagnostics must include completed returns and the PPO update size."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_sf.training.scenario_loader import load_scenarios
from scripts.training.train_ppo import (
    _DirectWandbTrainingMetricsCallback,
    _extract_direct_wandb_train_metrics,
    _init_training_model,
    load_expert_training_config,
)


def test_training_vector_records_completed_episode_return():
    """SB3 receives episode returns even though the scenario env is recreated."""
    recipe = load_expert_training_config(
        "configs/training/ppo/expert_ppo_release_contract_b_seed1002.yaml"
    )
    path = Path("configs/scenarios/archetypes/issue_596_frame_consistency.yaml")
    scenario = next(s for s in load_scenarios(path) if s["name"] == "empty_map_8_directions_east")
    overrides = dict(recipe.env_overrides)
    overrides["sim_config"] = dict(overrides["sim_config"], sim_time_in_secs=0.1)
    recipe = replace(
        recipe, num_envs=1, worker_mode="dummy", env_overrides=overrides, scenario_config=path
    )
    _, env, *_ = _init_training_model(
        config=recipe,
        scenario=scenario,
        scenario_definitions=(),
        run_id="diagnostic",
        exclude_scenarios=(),
        tensorboard_log=None,
        resume_from=None,
    )
    try:
        env.reset()
        _, reward, done, info = env.step(np.zeros((1, 2)))
        assert done[0]
        assert info[0]["episode"]["l"] == 1
        np.testing.assert_allclose(info[0]["episode"]["r"], reward[0])
    finally:
        env.close()


def test_direct_training_metrics_include_update_kl():
    """The online path retains PPO's measured KL rather than omitting it."""
    model = SimpleNamespace(logger=SimpleNamespace(name_to_value={"train/approx_kl": 0.012}))
    assert _extract_direct_wandb_train_metrics(model) == {"train/approx_kl": 0.012}


def test_training_callback_reports_reward_terms_and_terminal_outcomes():
    """Rollout diagnostics distinguish reaching the goal from colliding there."""
    logged = []
    callback = _DirectWandbTrainingMetricsCallback(
        wandb_run=SimpleNamespace(log=lambda payload, **kw: logged.append(payload)),
    )
    callback.model = SimpleNamespace(
        num_timesteps=2, ep_info_buffer=[], logger=SimpleNamespace(name_to_value={})
    )
    callback.locals = {
        "infos": [
            {
                "meta": {
                    "reward_terms": {"progress": 0.2, "collision": 0.0},
                    "is_route_complete": True,
                }
            },
            {
                "meta": {
                    "reward_terms": {"progress": 0.4, "collision": -10.0},
                    "is_route_complete": True,
                    "is_obstacle_collision": True,
                }
            },
        ],
        "dones": [True, True],
    }
    assert callback._on_step()
    callback.log_after_train()
    assert logged[0]["rollout/completed_episodes"] == 2
    assert logged[0]["rollout/success_rate"] == 0.5
    assert logged[0]["rollout/collision_rate"] == 0.5
    np.testing.assert_allclose(logged[0]["reward_terms/progress"], 0.3)
    assert logged[0]["reward_terms/collision"] == -5.0
