"""Spawn receipts and PPO initialisation must preserve physical layout seeds."""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import fields
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from robot_sf.training.imitation_config import (
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
)
from scripts.training import train_ppo


class ReceiptTestEnv(gym.Env):
    """Expose resets without simulator or learning steps."""

    observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-1, 1, (1,), np.float32)})
    action_space = gym.spaces.Box(-1, 1, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        return {"state": np.zeros(1, np.float32)}, {"actual_seed": seed}

    def step(self, action):
        raise AssertionError("test must never train or simulate")


def _fake_environment(**kwargs):
    return ReceiptTestEnv()


def _fake_config(*args, **kwargs):
    return object()


def _spawn_worker(factory, artifact_root):
    """Fresh imports cannot inherit the old parent instrumentation."""
    import os

    from robot_sf.training.scenario_sampling import ScenarioSwitchingEnv

    os.environ["ROBOT_SF_ARTIFACT_ROOT"] = artifact_root
    original = ScenarioSwitchingEnv
    train_ppo.make_robot_env = _fake_environment
    train_ppo.ScenarioSwitchingEnv = lambda **kw: original(env_factory=_fake_environment, **kw)
    train_ppo.build_robot_config_from_scenario = _fake_config
    train_ppo._apply_env_overrides = lambda *a: None
    env = factory()
    try:
        env.reset(seed=factory.seed)
        env.reset()
        env.reset()
    finally:
        env.close()


def test_training_seed_receipts_survive_spawn(tmp_path):
    """Actual child resets persist ordered receipts without parent hooks."""
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for worker in range(2):
        factory = train_ppo._make_training_env(
            100000 + worker,
            scenario=None,
            scenario_definitions=[{"name": "one"}, {"name": "two"}],
            scenario_path=tmp_path / "matrix.yaml",
            exclude_scenarios=(),
            suite_name="test",
            algorithm_name="test",
            env_overrides={},
            env_factory_kwargs={},
            scenario_sampling={"episode_seed_pool": (100000, 100001, 100002)},
        )
        # Identical pickle state on old/new code makes the old failure about
        # absent child receipts, rather than an unsupported new keyword.
        object.__setattr__(factory, "training_seed_telemetry", True)
        object.__setattr__(factory, "worker_index", worker)
        p = ctx.Process(target=_spawn_worker, args=(factory, str(tmp_path)))
        p.start()
        processes.append(p)
    for p in processes:
        p.join(30)
        if p.is_alive():
            p.terminate()
            p.join()
            pytest.fail("worker timeout")
        assert p.exitcode == 0
    paths = list((tmp_path / "training_seed_receipts").glob("*.jsonl"))
    assert len(paths) == 2, "spawned workers did not persist episode seed receipts"
    records = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    for worker in range(2):
        rows = [r for r in records if r["worker_index"] == worker]
        assert [r["episode_index"] for r in rows] == [0, 1, 2]
        assert rows[0]["seed"] == 100000 + worker
        assert all(r["seed"] in (100000, 100001, 100002) for r in rows)
        assert all(r["scenario_id"] in ("one", "two") for r in rows)


def test_ppo_initialisation_is_independent_of_layout_seed(monkeypatch, tmp_path):
    """Equal init seeds yield equal actors; pending resets keep layout seeds."""
    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "matrix.yaml",
        seeds=(100000,),
        total_timesteps=10,
        policy_id="test",
        convergence=ConvergenceCriteria(1.1, 0, 10),
        evaluation=EvaluationSchedule(0, 1, step_schedule=((None, 10),)),
        num_envs=1,
        worker_mode="dummy",
        ppo_hyperparams={"n_steps": 2, "batch_size": 2},
    )
    config = SimpleNamespace(
        **{
            **{f.name: getattr(config, f.name) for f in fields(config)},
            "ppo_initialization_seed": 1001,
        }
    )
    original_ppo = train_ppo.PPO
    monkeypatch.setattr(train_ppo, "PPO", lambda *a, **kw: original_ppo(*a, device="cpu", **kw))
    monkeypatch.setattr(train_ppo, "_make_training_env", lambda *a, **kw: ReceiptTestEnv)
    monkeypatch.setattr(
        train_ppo,
        "_resolve_policy_selection",
        lambda c: ("MultiInputPolicy", {"net_arch": [4]}, "test"),
    )
    models = []
    envs = []
    try:
        for seed in (100000, 100001):
            c = SimpleNamespace(**{**vars(config), "seeds": (seed,)})
            train_ppo._prepare_seed_state(c)
            model, env, *_ = train_ppo._init_training_model(
                config=c,
                scenario=None,
                scenario_definitions=[{"name": "dev"}],
                exclude_scenarios=(),
                tensorboard_log=None,
                resume_from=None,
                run_id="test-seed-pair",
            )
            models.append(model)
            envs.append(env)
            assert model.seed == 1001, "PPO initialisation still follows layout seed"
            assert env.venv._seeds == [seed], "initialisation leaked into physical reset seeds"
        assert all(
            torch.equal(v, models[1].policy.state_dict()[k])
            for k, v in models[0].policy.state_dict().items()
        )
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("value", [True, -1, 1.5, "1001", 2**31])
def test_initialisation_seed_refuses_non_rng_integer(value, tmp_path):
    """An init control cannot silently coerce a boolean, fraction or RNG overflow."""
    with pytest.raises(ValueError, match="ppo_initialization_seed"):
        ExpertTrainingConfig.from_raw(
            scenario_config=tmp_path / "matrix.yaml",
            seeds=(100000,),
            total_timesteps=10,
            policy_id="test",
            convergence=ConvergenceCriteria(1.1, 0, 10),
            evaluation=EvaluationSchedule(0, 1, step_schedule=((None, 10),)),
            ppo_initialization_seed=value,
        )
