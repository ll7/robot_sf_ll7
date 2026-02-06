"""Tests for multi-robot rendering and recording wiring."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

import robot_sf.gym_env.multi_robot_env as multi_robot_env_mod
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS


@dataclass
class _DummyMapPool:
    map_defs: dict[str, object]


def _build_config() -> SimpleNamespace:
    map_def = SimpleNamespace(obstacles=[])
    return SimpleNamespace(
        map_pool=_DummyMapPool(map_defs={"uni_campus_big": map_def}),
        sim_config=SimpleNamespace(
            time_per_step_in_secs=0.1,
            sim_time_in_secs=5.0,
            ped_radius=0.3,
            goal_radius=0.5,
        ),
        robot_config=SimpleNamespace(radius=0.4),
        lidar_config=SimpleNamespace(),
        render_scaling=10,
    )


class _FakeRobot:
    def parse_action(self, action):
        return np.asarray(action, dtype=float)


class _FakeSimulator:
    def __init__(self, num_robots: int):
        self.robots = [_FakeRobot() for _ in range(num_robots)]
        self.robot_navs = [object() for _ in range(num_robots)]
        self._robot_poses = [((float(i), 0.0), 0.0) for i in range(num_robots)]
        self._goal_pos = [(10.0, 0.0) for _ in range(num_robots)]
        self.ped_pos = np.zeros((0, 2), dtype=float)
        self.reset_calls = 0
        self.step_calls: list[list[np.ndarray]] = []

    @property
    def robot_poses(self):
        return self._robot_poses

    @property
    def goal_pos(self):
        return self._goal_pos

    def reset_state(self) -> None:
        self.reset_calls += 1

    def step_once(self, actions) -> None:
        self.step_calls.append(list(actions))


class _FakeRobotState:
    def __init__(self, nav, occupancy, sensors, d_t: float, sim_time_limit: float):
        self.nav = nav
        self.occupancy = occupancy
        self.sensors = sensors
        self.timestep = 0
        self._terminal = False
        self._step_counter = 0
        self._obs = {
            OBS_DRIVE_STATE: np.zeros((1,), dtype=np.float32),
            OBS_RAYS: np.zeros((2,), dtype=np.float32),
        }

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    def step(self):
        self.timestep += 1
        self._step_counter += 1
        self._obs = {
            OBS_DRIVE_STATE: np.array([self._step_counter], dtype=np.float32),
            OBS_RAYS: np.array([0.1, 0.2], dtype=np.float32),
        }
        return self._obs

    def reset(self):
        self.timestep = 0
        self._terminal = False
        self._obs = {
            OBS_DRIVE_STATE: np.zeros((1,), dtype=np.float32),
            OBS_RAYS: np.zeros((2,), dtype=np.float32),
        }
        return self._obs

    def meta_dict(self) -> dict:
        return {
            "step": self._step_counter,
            "is_pedestrian_collision": False,
            "is_robot_collision": False,
            "is_obstacle_collision": False,
            "is_robot_at_goal": False,
            "is_route_complete": False,
            "is_timesteps_exceeded": False,
            "max_sim_steps": 100,
        }


class _FakeSimulationView:
    instances: list[_FakeSimulationView] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.render_calls = 0
        self.exit_calls = 0
        self.show_lidar = True
        _FakeSimulationView.instances.append(self)

    def render(self, _state) -> None:
        self.render_calls += 1

    def exit_simulation(self) -> None:
        self.exit_calls += 1


def _patch_multi_robot_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_init_spaces(_cfg, _map_def):
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        observation_space = spaces.Dict(
            {
                OBS_DRIVE_STATE: spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
                OBS_RAYS: spaces.Box(low=0.0, high=10.0, shape=(2,), dtype=np.float32),
            }
        )
        return action_space, observation_space, observation_space

    def _fake_init_simulators(_cfg, _map_def, num_robots, random_start_pos=False):
        _ = random_start_pos
        return [_FakeSimulator(int(num_robots))]

    def _fake_init_collision_and_sensors(sim, _cfg, _orig_obs_space):
        count = len(sim.robots)
        return [SimpleNamespace() for _ in range(count)], [SimpleNamespace() for _ in range(count)]

    monkeypatch.setattr(multi_robot_env_mod, "init_spaces", _fake_init_spaces)
    monkeypatch.setattr(multi_robot_env_mod, "init_simulators", _fake_init_simulators)
    monkeypatch.setattr(
        multi_robot_env_mod,
        "init_collision_and_sensors",
        _fake_init_collision_and_sensors,
    )
    monkeypatch.setattr(multi_robot_env_mod, "RobotState", _FakeRobotState)
    monkeypatch.setattr(
        multi_robot_env_mod,
        "lidar_ray_scan",
        lambda pose, occupancy, lidar_cfg: (
            np.array([1.0], dtype=float),
            np.array([0.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        multi_robot_env_mod,
        "render_lidar",
        lambda robot_pos, distances, directions: np.array([[[0.0, 0.0], [1.0, 0.0]]], dtype=float),
    )
    monkeypatch.setattr(
        multi_robot_env_mod,
        "prepare_pedestrian_actions",
        lambda sim: np.zeros((0, 2, 2), dtype=float),
    )
    monkeypatch.setattr(multi_robot_env_mod, "SimulationView", _FakeSimulationView)


def test_multi_robot_env_recording_creates_per_robot_views(monkeypatch, tmp_path) -> None:
    """Record-video mode creates one view per robot with deterministic filenames."""
    _FakeSimulationView.instances.clear()
    _patch_multi_robot_dependencies(monkeypatch)
    env = multi_robot_env_mod.MultiRobotEnv(
        env_config=_build_config(),
        debug=False,
        num_robots=2,
        recording_enabled=True,
        record_video=True,
        video_path=str(tmp_path / "episode.mp4"),
        video_fps=12.0,
    )
    try:
        assert len(env._sim_views) == 2
        assert env.sim_ui is env._sim_views[0]
        paths = [view.kwargs.get("video_path") for view in env._sim_views]
        assert paths[0].endswith("episode_robot0.mp4")
        assert paths[1].endswith("episode_robot1.mp4")
    finally:
        env.close()
    assert all(view.exit_calls == 1 for view in _FakeSimulationView.instances)


def test_multi_robot_env_step_reset_render_and_step_agents(monkeypatch) -> None:
    """Step/reset/render paths should work with the refactored render support."""
    _FakeSimulationView.instances.clear()
    _patch_multi_robot_dependencies(monkeypatch)
    env = multi_robot_env_mod.MultiRobotEnv(
        env_config=_build_config(),
        debug=True,
        num_robots=2,
    )
    try:
        env.states[0]._terminal = True
        action = np.array([[0.2, 0.0], [0.1, 0.0]], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert OBS_DRIVE_STATE in obs
        assert OBS_RAYS in obs
        assert reward < 0.0
        assert terminated is True
        assert truncated is False
        assert "agents" in info
        assert env.simulators[0].reset_calls >= 1

        reset_obs, reset_info = env.reset(seed=13)
        assert OBS_DRIVE_STATE in reset_obs
        assert OBS_RAYS in reset_obs
        assert reset_info == {}
        assert env.applied_seed == 13

        env.render()
        assert all(view.render_calls == 1 for view in env._sim_views)

        stepped_obs, stepped_rewards, stepped_terms, stepped_info = env._step_agents(action)
        assert len(stepped_obs) == 1
        assert len(stepped_rewards) == 1
        assert len(stepped_terms) == 1
        assert len(stepped_info) == 1

        del env.single_action_space
        del env.single_observation_space
        action_space, observation_space = env._create_spaces()
        assert action_space.shape == (2,)
        assert OBS_DRIVE_STATE in observation_space.spaces
    finally:
        env.close()


def test_multi_robot_env_render_requires_view(monkeypatch) -> None:
    """Rendering without debug/recording should raise a clear error."""
    _patch_multi_robot_dependencies(monkeypatch)
    env = multi_robot_env_mod.MultiRobotEnv(
        env_config=_build_config(),
        debug=False,
        num_robots=1,
        record_video=False,
    )
    try:
        with pytest.raises(RuntimeError, match="Render unavailable"):
            env.render()
    finally:
        env.close()
