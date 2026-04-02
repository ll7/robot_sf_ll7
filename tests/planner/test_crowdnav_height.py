"""Tests for the experimental CrowdNav_HEIGHT adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.planner.crowdnav_height import CrowdNavHeightAdapter, build_crowdnav_height_config


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_upstream_repo(repo_root: Path) -> None:
    _write(repo_root / "training" / "__init__.py", "")
    _write(repo_root / "training" / "networks" / "__init__.py", "")
    _write(
        repo_root / "training" / "networks" / "model.py",
        """
import torch
import torch.nn as nn

LAST_INPUTS = {}

class _Base:
    def __init__(self):
        self.nenv = 1

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super().__init__()
        del obs_shape, action_space, base
        self.base = _Base()
        self.weight = nn.Parameter(torch.zeros(1))
        self._action_index = int(base_kwargs.testing.action_index)

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        del deterministic
        global LAST_INPUTS
        LAST_INPUTS = {key: value.detach().cpu().clone() for key, value in inputs.items()}
        LAST_INPUTS["masks"] = masks.detach().cpu().clone()
        action = torch.tensor([[self._action_index]], dtype=torch.long, device=self.weight.device)
        return (
            torch.zeros((1, 1), dtype=torch.float32, device=self.weight.device),
            action,
            torch.zeros((1, 1), dtype=torch.float32, device=self.weight.device),
            rnn_hxs,
        )
""",
    )


def _write_fake_checkpoint_dir(model_dir: Path, *, action_index: int) -> None:
    _write(model_dir / "configs" / "__init__.py", "")
    _write(
        model_dir / "configs" / "config.py",
        f"""
class BaseConfig:
    pass

class Config:
    env = BaseConfig()
    env.time_step = 0.1
    env.scenario = "circle_crossing"
    action_space = BaseConfig()
    action_space.kinematics = "turtlebot"
    sim = BaseConfig()
    sim.human_num = 3
    sim.human_num_range = 0
    sim.static_obs_num = 2
    sim.static_obs_num_range = 0
    lidar = BaseConfig()
    lidar.angular_res = 90
    lidar.sensor_range = 10.0
    robot = BaseConfig()
    robot.policy = "selfAttn_merge_srnn_lidar"
    robot.v_min = 0.0
    robot.v_max = 0.5
    robot.w_min = -1.0
    robot.w_max = 1.0
    SRNN = BaseConfig()
    SRNN.human_node_rnn_size = 4
    training = BaseConfig()
    training.num_processes = 1
    ppo = BaseConfig()
    ppo.num_steps = 1
    ppo.num_mini_batch = 1
    testing = BaseConfig()
    testing.action_index = {action_index}
""",
    )
    checkpoint_path = model_dir / "checkpoints" / "fake.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": torch.zeros(1)}, checkpoint_path)


def test_build_config_keeps_explicit_default_paths() -> None:
    """Config builder should expose the upstream repo and extracted checkpoint paths directly."""
    cfg = build_crowdnav_height_config({})
    assert cfg.repo_root == Path("output/repos/CrowdNav_HEIGHT")
    assert cfg.model_dir == Path(
        "output/external_checkpoints/crowdnav_height_extracted/HEIGHT/HEIGHT"
    )
    assert cfg.checkpoint_name == "237800.pt"


def test_adapter_requires_bound_obstacles(tmp_path: Path) -> None:
    """The adapter should fail fast when obstacle geometry has not been bound yet."""
    repo_root = tmp_path / "repo"
    model_dir = tmp_path / "model"
    _write_fake_upstream_repo(repo_root)
    _write_fake_checkpoint_dir(model_dir, action_index=0)
    adapter = CrowdNavHeightAdapter(
        build_crowdnav_height_config(
            {
                "repo_root": str(repo_root),
                "model_dir": str(model_dir),
                "checkpoint_name": "fake.pt",
            }
        )
    )
    with pytest.raises(RuntimeError, match="obstacle segments"):
        adapter.plan(
            {
                "robot": {
                    "position": [0.0, 0.0],
                    "heading": [0.0],
                    "velocity_xy": [0.0, 0.0],
                    "radius": [0.3],
                },
                "goal": {"current": [1.0, 0.0]},
                "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.3]},
                "sim": {"timestep": [0.1]},
            }
        )


def test_adapter_rebuilds_height_observation_and_accumulates_stateful_command(
    tmp_path: Path,
) -> None:
    """The adapter should rebuild the upstream dict obs and accumulate delta-v/delta-theta actions."""
    repo_root = tmp_path / "repo"
    model_dir = tmp_path / "model"
    _write_fake_upstream_repo(repo_root)
    _write_fake_checkpoint_dir(model_dir, action_index=0)
    adapter = CrowdNavHeightAdapter(
        build_crowdnav_height_config(
            {
                "repo_root": str(repo_root),
                "model_dir": str(model_dir),
                "checkpoint_name": "fake.pt",
            }
        )
    )
    adapter.bind_obstacle_segments([[2.0, -1.0, 2.0, 1.0]])
    obs = {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [0.0],
            "velocity_xy": [0.2, 0.0],
            "radius": [0.3],
        },
        "goal": {"current": [4.0, 0.0]},
        "pedestrians": {
            "positions": [[1.0, 0.0], [3.0, 0.0]],
            "velocities": [[0.0, 0.5], [0.0, -0.5]],
            "count": [2],
            "radius": [0.3],
        },
        "sim": {"timestep": [0.1]},
    }

    command_1, turn_1, meta_1 = adapter.act(obs, time_step=0.1)
    command_2, turn_2, meta_2 = adapter.act(obs, time_step=0.1)

    assert adapter._last_model_inputs is not None
    last_inputs = adapter._last_model_inputs
    assert command_1 == pytest.approx(0.05)
    assert turn_1 == pytest.approx(0.1)
    assert command_2 == pytest.approx(0.1)
    assert turn_2 == pytest.approx(0.2)
    assert meta_1["action_index"] == 0
    assert meta_1["projection_policy"] == ("upstream_discrete_delta_vw_to_unicycle_vw_stateful")
    assert meta_2["projected_command_vw"] == [pytest.approx(0.1), pytest.approx(0.2)]
    assert last_inputs["robot_node"].shape == (1, 5)
    assert last_inputs["spatial_edges"].shape == (3, 4)
    assert float(last_inputs["detected_human_num"][0]) == pytest.approx(2.0)
    assert float(last_inputs["point_clouds"][0, 0]) == pytest.approx(2.0)
    assert float(last_inputs["spatial_edges"][0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(last_inputs["spatial_edges"][0, 1]) == pytest.approx(1.0, abs=1e-6)

    adapter.reset()
    command_3, turn_3 = adapter.plan(obs)
    assert command_3 == pytest.approx(0.05)
    assert turn_3 == pytest.approx(0.1)


def test_adapter_accepts_flat_benchmark_observation_schema(tmp_path: Path) -> None:
    """The adapter should accept the flat benchmark observation contract used by map_runner."""
    repo_root = tmp_path / "repo"
    model_dir = tmp_path / "model"
    _write_fake_upstream_repo(repo_root)
    _write_fake_checkpoint_dir(model_dir, action_index=4)
    adapter = CrowdNavHeightAdapter(
        build_crowdnav_height_config(
            {
                "repo_root": str(repo_root),
                "model_dir": str(model_dir),
                "checkpoint_name": "fake.pt",
            }
        )
    )
    adapter.bind_obstacle_segments([[2.0, -1.0, 2.0, 1.0]])
    obs = {
        "robot_position": [0.0, 0.0],
        "robot_heading": [0.0],
        "robot_speed": [0.2, 0.0],
        "robot_velocity_xy": [0.2, 0.0],
        "robot_angular_velocity": [0.0],
        "robot_radius": [0.3],
        "goal_current": [4.0, 0.0],
        "goal_next": [4.0, 0.0],
        "pedestrians_positions": [[1.0, 0.0], [3.0, 0.0]] + [[0.0, 0.0]] * 62,
        "pedestrians_velocities": [[0.0, 0.5], [0.0, -0.5]] + [[0.0, 0.0]] * 62,
        "pedestrians_radius": [0.3],
        "pedestrians_count": [2],
        "sim_timestep": [0.1],
        "occupancy_grid": np.zeros((3, 4, 4), dtype=float),
    }

    linear, angular, meta = adapter.act(obs, time_step=0.1)
    assert linear == pytest.approx(0.0)
    assert angular == pytest.approx(0.0)
    assert meta["human_count"] == 2
    assert adapter._last_model_inputs is not None
    assert adapter._last_model_inputs["robot_node"].shape == (1, 5)


@pytest.mark.parametrize("seed", [7, 11, 23])
@pytest.mark.slow
def test_adapter_runs_live_socnav_smoke_for_multiple_seeds(seed: int) -> None:
    """The wrapped checkpoint should execute several live Robot SF episodes without fallback."""
    env = make_robot_env(
        config=RobotSimulationConfig(observation_mode=ObservationMode.SOCNAV_STRUCT),
        seed=seed,
        debug=False,
    )
    try:
        obs, _ = env.reset(seed=seed)
        adapter = CrowdNavHeightAdapter(build_crowdnav_height_config({}))
        adapter.bind_env(env)
        adapter.reset(seed=seed)

        command_history: list[tuple[float, float]] = []
        for _ in range(8):
            linear, angular, meta = adapter.act(obs, time_step=float(obs["sim"]["timestep"][0]))
            command_history.append((linear, angular))
            assert np.isfinite(linear)
            assert np.isfinite(angular)
            assert -adapter.config.max_linear_speed <= linear <= adapter.config.max_linear_speed
            assert -adapter.config.max_angular_speed <= angular <= adapter.config.max_angular_speed
            assert meta["projection_policy"] == "upstream_discrete_delta_vw_to_unicycle_vw_stateful"
            obs, reward, terminated, truncated, info = env.step(
                np.array([linear, angular], dtype=float)
            )
            assert np.isfinite(reward)
            assert "robot" in obs and "goal" in obs and "pedestrians" in obs
            if terminated or truncated:
                assert isinstance(info, dict)
                break

        assert len(command_history) >= 3
        assert adapter._last_model_inputs is not None
        assert adapter._last_model_inputs["robot_node"].shape == (1, 5)
        assert adapter._last_model_inputs["spatial_edges"].shape[1] == 4
        assert adapter._last_model_inputs["point_clouds"].shape[1] > 0
    finally:
        env.close()
