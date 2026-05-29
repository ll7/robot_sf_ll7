"""Signature tests for explicit Option A environment helpers.

Ensures discoverability and prevents regression back to **kwargs catch-all.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.gym_env import crowd_sim_env
from robot_sf.gym_env import environment_factory as environment_factory_module
from robot_sf.gym_env.base_env import BaseEnv
from robot_sf.gym_env.crowd_sim_env import CrowdSimEnv, CrowdSimulationConfig
from robot_sf.gym_env.env_config import SimulationSettings
from robot_sf.gym_env.environment_factory import (
    EnvironmentFactory,
    make_crowd_sim_env,
    make_image_robot_env,
    make_multi_robot_env,
    make_pedestrian_env,
    make_robot_env,
)
from robot_sf.gym_env.multi_robot_env import MultiRobotEnv
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig, RobotSimulationConfig


def _param_names(func):
    """Return function parameter names in declaration order.

    Args:
        func: Callable object to inspect.
    """
    return [p.name for p in inspect.signature(func).parameters.values()]


def _assert_no_var_keyword(func):
    """Assert public factory signatures do not expose catch-all keyword params."""
    assert inspect.Parameter.VAR_KEYWORD not in [
        p.kind for p in inspect.signature(func).parameters.values()
    ]


def test_make_robot_env_signature_explicit():
    """Robot env factory keeps the reviewed explicit public parameters."""
    params = _param_names(make_robot_env)
    assert "config" in params
    assert "reward_func" in params
    assert "debug" in params
    assert "record_video" in params
    _assert_no_var_keyword(make_robot_env)


def test_make_image_robot_env_signature_explicit():
    """Image robot env factory keeps the reviewed explicit public parameters."""
    params = _param_names(make_image_robot_env)
    assert "config" in params
    assert "debug" in params
    _assert_no_var_keyword(make_image_robot_env)


def test_make_pedestrian_env_signature_explicit():
    """Pedestrian env factory exposes config and robot-model parameters."""
    params = _param_names(make_pedestrian_env)
    assert "config" in params and "robot_model" in params
    _assert_no_var_keyword(make_pedestrian_env)


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (make_robot_env, {"fps": 30}),
        (make_image_robot_env, {"video_output_path": "legacy.mp4"}),
        (make_pedestrian_env, {"fps": 30}),
    ],
)
def test_public_factories_reject_retired_legacy_kwargs(factory, kwargs):
    """Retired catch-all factory kwargs should fail at the Python signature boundary."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        factory(**kwargs)


def test_robot_factory_normalizes_deprecated_force_flag_once(monkeypatch):
    """Internal factory should sync the legacy kwarg through the shared config helper."""
    captured = {}

    class FakeRobotEnv:
        """Minimal robot env stub for factory normalization assertions."""

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(environment_factory_module, "RobotEnv", FakeRobotEnv)
    config = RobotSimulationConfig()

    env = EnvironmentFactory.create_robot_env(
        config=config,
        use_image_obs=False,
        peds_have_obstacle_forces=False,
        reward_func=None,
        debug=False,
        recording_enabled=False,
        record_video=False,
        video_path=None,
        video_fps=None,
    )

    assert isinstance(env, FakeRobotEnv)
    assert config.peds_have_static_obstacle_forces is False
    assert config.peds_have_obstacle_forces is False
    assert captured["peds_have_obstacle_forces"] is False

    explicit_false_config = RobotSimulationConfig(peds_have_static_obstacle_forces=False)
    EnvironmentFactory.create_robot_env(
        config=explicit_false_config,
        use_image_obs=False,
        peds_have_obstacle_forces=True,
        reward_func=None,
        debug=False,
        recording_enabled=False,
        record_video=False,
        video_path=None,
        video_fps=None,
    )

    assert explicit_false_config.peds_have_static_obstacle_forces is False
    assert explicit_false_config.peds_have_obstacle_forces is False


def test_pedestrian_factory_preserves_explicit_static_force_config(monkeypatch):
    """Default legacy kwarg should not override an explicit canonical false value."""
    captured = {}

    class FakePedestrianEnv:
        """Minimal pedestrian env stub for factory normalization assertions."""

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        environment_factory_module, "_load_pedestrian_env", lambda: FakePedestrianEnv
    )
    config = PedestrianSimulationConfig(peds_have_static_obstacle_forces=False)

    env = EnvironmentFactory.create_pedestrian_env(
        robot_model=None,
        config=config,
        peds_have_obstacle_forces=True,
    )

    assert isinstance(env, FakePedestrianEnv)
    assert config.peds_have_static_obstacle_forces is False
    assert config.peds_have_obstacle_forces is False
    assert captured["peds_have_obstacle_forces"] is True


def test_make_multi_robot_env_signature_explicit():
    """Multi-robot env factory exposes robot-count and config parameters."""
    params = _param_names(make_multi_robot_env)
    assert "num_robots" in params and "config" in params


def test_make_crowd_sim_env_signature_explicit():
    """Crowd env factory exposes robot-free stepping and recording controls."""
    params = _param_names(make_crowd_sim_env)
    assert "config" in params
    assert "render_mode" in params
    assert "recording_enabled" in params
    assert "recording_dir" in params
    assert "recording_path" in params
    assert "video_path" in params
    assert "video_fps" in params


def test_make_crowd_sim_env_preserves_preconfigured_values(monkeypatch, tmp_path):
    """Crowd env factory should preserve explicit config values by default."""

    class FakePedestrianStates:
        def __init__(self):
            self._states = np.array(
                [[1.0, 2.0, 0.1, 0.2, 5.0, 6.0]],
                dtype=float,
            )

        @property
        def num_peds(self) -> int:
            return 1

        def pysf_states(self) -> np.ndarray:
            return self._states

    class FakeSimulator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.pysf_state = FakePedestrianStates()
            self.last_ped_forces = np.zeros((0, 2), dtype=float)
            self.last_actions = []

        @property
        def ped_pos(self) -> np.ndarray:
            return self.pysf_state.pysf_states()[:, 0:2]

        def step_once(self, actions):
            self.last_actions = actions

    class FakeMapPool:
        def __init__(self):
            self.map = SimpleNamespace(obstacles=[], ped_count=1)
            self.map_defs = {"fake": self.map}

        def choose_random_map(self):
            return self.map

        def get_map(self, map_id: str):
            return self.map_defs[map_id]

    monkeypatch.setattr(crowd_sim_env, "Simulator", FakeSimulator)

    recording_dir = tmp_path / "configured-dir"
    recording_path = tmp_path / "configured.jsonl"
    video_path = tmp_path / "configured.mp4"
    config = CrowdSimulationConfig(
        sim_config=SimulationSettings(
            sim_time_in_secs=0.2,
            time_per_step_in_secs=0.1,
        ),
        map_pool=FakeMapPool(),
        map_id="fake",
        peds_have_obstacle_forces=False,
        render_mode="rgb_array",
        recording_enabled=True,
        recording_dir=str(recording_dir),
        recording_path=str(recording_path),
        video_path=str(video_path),
        video_fps=12.5,
    )

    env = make_crowd_sim_env(config=config)

    try:
        assert isinstance(env, CrowdSimEnv)
        assert env.config.peds_have_obstacle_forces is False
        assert env.config.render_mode == "rgb_array"
        assert env.config.recording_enabled is True
        assert env.config.recording_dir == str(recording_dir)
        assert env.config.recording_path == str(recording_path)
        assert env.config.video_path == str(video_path)
        assert env.config.video_fps == 12.5
    finally:
        env.close()


def test_make_crowd_sim_env_applies_explicit_overrides(monkeypatch):
    """Crowd env factory should apply explicit overrides when config is omitted."""

    class FakeCrowdSimulationConfig:
        def __init__(self):
            self.map_id = None
            self.peds_have_obstacle_forces = True
            self.render_mode = None
            self.recording_enabled = False
            self.recording_dir = "recordings"
            self.recording_path = None
            self.video_path = None
            self.video_fps = None

    class FakeCrowdEnv:
        def __init__(self, config):
            self.config = config
            self.applied_seed = None

    applied = {}

    monkeypatch.setattr(
        environment_factory_module,
        "_apply_global_seed",
        lambda seed: applied.setdefault("seed", seed),
    )
    monkeypatch.setattr(
        environment_factory_module,
        "_load_crowd_sim_env",
        lambda: (FakeCrowdEnv, FakeCrowdSimulationConfig),
    )

    env = make_crowd_sim_env(
        seed=123,
        map_id="fake",
        peds_have_obstacle_forces=False,
        render_mode="rgb_array",
        recording_enabled=True,
        recording_dir="override-dir",
        recording_path="override.jsonl",
        video_path="override.mp4",
        video_fps=9.0,
    )

    assert isinstance(env, FakeCrowdEnv)
    assert applied["seed"] == 123
    assert env.applied_seed == 123
    assert env.config.map_id == "fake"
    assert env.config.peds_have_obstacle_forces is False
    assert env.config.render_mode == "rgb_array"
    assert env.config.recording_enabled is True
    assert env.config.recording_dir == "override-dir"
    assert env.config.recording_path == "override.jsonl"
    assert env.config.video_path == "override.mp4"
    assert env.config.video_fps == 9.0


def test_env_constructors_do_not_use_config_instance_defaults() -> None:
    """Avoid shared mutable config objects in environment constructor signatures."""
    env_classes = [BaseEnv, RobotEnv, RobotEnvWithImage, MultiRobotEnv]

    for env_class in env_classes:
        param = inspect.signature(env_class).parameters["env_config"]
        assert param.default is None
