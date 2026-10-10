"""A fixed training seed must keep episode simulator seeds in its declared pool."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium import Env, spaces

from robot_sf.training.scenario_sampling import ScenarioSwitchingEnv
from scripts.training.train_ppo import _make_training_env


def _dummy_switching_env(
    *,
    seed: int | None = 1001,
    episode_seed_pool: tuple[object, ...] | list[object] = (1001,),
) -> ScenarioSwitchingEnv:
    class Dummy(Env):
        observation_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(1, dtype=np.float32), {}

    return ScenarioSwitchingEnv(
        scenario_sampler=lambda rng: ({"name": "a"}, "a"),
        scenario_path=Path("unused"),
        env_factory=lambda **kwargs: Dummy(),
        suite_name="dev",
        algorithm_name="ppo",
        seed=seed,
        episode_seed_pool=episode_seed_pool,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("bad_pool", [(), (True,), (-1,), (1.5,)])
def test_episode_seed_pool_rejects_invalid_pool_members(bad_pool):
    with pytest.raises(ValueError, match="episode_seed_pool must contain non-negative integer"):
        _dummy_switching_env(episode_seed_pool=bad_pool)


def test_episode_seed_pool_rejects_constructor_seed_outside_pool():
    with pytest.raises(
        ValueError,
        match=r"seed 1004 must belong to episode_seed_pool \(1001\.\.1001\); check num_envs",
    ):
        _dummy_switching_env(seed=1004, episode_seed_pool=(1001,))


def test_training_factory_keeps_construct_and_reset_seeds_in_pool(monkeypatch):
    """PPO wiring must restrict both creation and resets, including explicit resets."""
    seen = []

    class Dummy(Env):
        observation_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
        action_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            seen.append(seed)
            return np.zeros(1, dtype=np.float32), {}

    def factory(**kwargs):
        seen.append(kwargs["seed"])
        return Dummy()

    monkeypatch.setattr(
        "scripts.training.train_ppo.ScenarioSwitchingEnv",
        lambda **kwargs: ScenarioSwitchingEnv(env_factory=factory, **kwargs),
    )
    monkeypatch.setattr(
        "scripts.training.train_ppo.build_robot_config_from_scenario", lambda *a, **k: object()
    )
    monkeypatch.setattr("scripts.training.train_ppo._apply_env_overrides", lambda *a, **k: None)
    make = _make_training_env(
        1001,
        scenario=None,
        scenario_definitions=[{"name": "a"}],
        scenario_path=Path("unused"),
        exclude_scenarios=[],
        suite_name="dev",
        algorithm_name="ppo",
        env_overrides={},
        env_factory_kwargs={},
        scenario_sampling={"episode_seed_pool": [1001, 1002, 1003]},
    )
    env = make()
    for _ in range(40):
        env.reset()
    assert set(seen) <= {1001, 1002, 1003}, seen
    before = len(seen)

    with pytest.raises(
        ValueError,
        match=r"seed 1004 must belong to episode_seed_pool \(1001\.\.1003\); check num_envs",
    ):
        env.reset(seed=1004)
    assert len(seen) == before
    env.close()
