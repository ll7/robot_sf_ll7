"""Tests for the experimental SoNIC/CrowdNav adapter."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_sf.planner.sonic_crowdnav import SonicCrowdNavAdapter, build_sonic_crowdnav_config


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_upstream_repo(
    repo_root: Path,
    *,
    action_xy: tuple[float, float] = (1.2, 0.0),
    kinematics: str = "holonomic",
) -> None:
    _write(repo_root / "trained_models" / "__init__.py", "")
    _write(repo_root / "trained_models" / "SoNIC_GST" / "__init__.py", "")
    _write(repo_root / "trained_models" / "SoNIC_GST" / "configs" / "__init__.py", "")
    _write(repo_root / "rl" / "__init__.py", "")
    _write(repo_root / "rl" / "networks" / "__init__.py", "")
    _write(
        repo_root / "trained_models" / "SoNIC_GST" / "arguments.py",
        """
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--env-name", default="CrowdSimPredRealGST-v0")
    return parser.parse_args([])
""",
    )
    _write(
        repo_root / "trained_models" / "SoNIC_GST" / "configs" / "config.py",
        f"""
class BaseConfig:
    pass


class Config:
    env = BaseConfig()
    env.use_wrapper = True
    env.time_step = 0.25

    sim = BaseConfig()
    sim.human_num = 2
    sim.human_num_range = 1
    sim.predict_steps = 3
    sim.predict_method = "inferred"

    policy = BaseConfig()
    policy.constant_std = True

    robot = BaseConfig()
    robot.policy = "selfAttn_merge_srnn"
    robot.v_pref = 1.0
    robot.sensor = "coordinates"

    humans = BaseConfig()
    humans.policy = "orca"

    action_space = BaseConfig()
    action_space.kinematics = "{kinematics}"
""",
    )
    _write(
        repo_root / "rl" / "networks" / "model.py",
        f"""
import torch
import torch.nn as nn


class _Base:
    human_node_rnn_size = 8
    human_human_edge_rnn_size = 16


class Policy(nn.Module):
    LAST_INPUTS = None

    def __init__(self, obs_shape, action_space, config, base=None, base_kwargs=None):
        super().__init__()
        del obs_shape, action_space, config, base, base_kwargs
        self.base = _Base()
        self.weight = nn.Parameter(torch.zeros(1))

    def load_state_dict(self, _state, strict=False):
        del strict
        return ["dist.logstd._bias"], []

    def to(self, _device):
        return self

    def eval(self):
        return self

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        del deterministic
        type(self).LAST_INPUTS = {{key: value.detach().cpu().clone() for key, value in inputs.items()}}
        action = torch.tensor([[{action_xy[0]}, {action_xy[1]}]], dtype=torch.float32)
        return (
            torch.zeros((1, 1), dtype=torch.float32),
            action,
            torch.zeros((1, 1), dtype=torch.float32),
            rnn_hxs,
        )
""",
    )
    checkpoint_path = repo_root / "trained_models" / "SoNIC_GST" / "checkpoints" / "05207.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": torch.tensor([1.0])}, checkpoint_path)


def test_adapter_translates_nested_robot_sf_observation_and_projects_action(tmp_path: Path) -> None:
    """The adapter should rebuild the upstream contract and project a holonomic action."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SonicCrowdNavAdapter(
        build_sonic_crowdnav_config(
            {
                "repo_root": str(repo_root),
                "checkpoint_name": "05207.pt",
                "max_linear_speed": 1.5,
                "max_angular_speed": 2.0,
            }
        )
    )

    linear, angular, meta = adapter.act(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.0],
                "velocity_xy": [0.1, 0.0],
                "radius": [0.3],
            },
            "goal": {"current": [5.0, 0.0]},
            "pedestrians": {
                "positions": [[2.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
                "velocities": [[0.0, 0.0], [0.0, 0.1], [0.0, -0.1]],
                "count": [2],
            },
        },
        time_step=0.25,
    )

    assert linear == pytest.approx(1.0)
    assert angular == pytest.approx(0.0)
    assert meta["upstream_action_xy"] == [pytest.approx(1.0), pytest.approx(0.0)]
    assert meta["projected_command_vw"] == [pytest.approx(1.0), pytest.approx(0.0)]
    assert meta["source_action_kinematics"] == "holonomic"
    assert meta["detected_human_num"] == 2
    assert meta["parity_gaps"]

    inputs = adapter._policy.__class__.LAST_INPUTS
    assert inputs is not None
    assert inputs["robot_node"].shape == (1, 1, 7)
    assert inputs["temporal_edges"].shape == (1, 1, 2)
    assert inputs["spatial_edges"].shape == (1, 3, 8)
    assert inputs["conformity_scores"].shape == (1, 3, 3)
    assert inputs["visible_masks"].shape == (1, 3)
    assert inputs["detected_human_num"].shape == (1, 1)
    assert float(inputs["robot_node"][0, 0, 0]) == pytest.approx(0.0)
    assert float(inputs["robot_node"][0, 0, 3]) == pytest.approx(5.0)
    assert float(inputs["spatial_edges"][0, 0, 0]) == pytest.approx(1.0)
    assert float(inputs["spatial_edges"][0, 0, 1]) == pytest.approx(0.0)


def test_adapter_plan_velocity_world_returns_upstream_action_xy(tmp_path: Path) -> None:
    """The adapter should expose the raw upstream ActionXY command for holonomic passthrough."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root, action_xy=(0.3, -0.4))
    adapter = SonicCrowdNavAdapter(
        build_sonic_crowdnav_config(
            {
                "repo_root": str(repo_root),
                "checkpoint_name": "05207.pt",
                "max_linear_speed": 1.0,
                "max_angular_speed": 1.0,
            }
        )
    )

    velocity_world = adapter.plan_velocity_world(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.25],
                "velocity_xy": [0.0, 0.0],
                "radius": [0.3],
            },
            "goal": {"current": [2.0, 0.0]},
            "pedestrians": {
                "positions": [[1.0, 0.0]],
                "velocities": [[0.0, 0.0]],
                "count": [1],
            },
            "sim": {"timestep": 0.25},
        }
    )

    assert velocity_world == pytest.approx((0.3, -0.4))


def test_adapter_accepts_flat_single_human_payloads(tmp_path: Path) -> None:
    """Flat XY payloads should still normalize to one-row arrays."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root, action_xy=(0.0, 0.5))
    adapter = SonicCrowdNavAdapter(
        build_sonic_crowdnav_config(
            {
                "repo_root": str(repo_root),
                "checkpoint_name": "05207.pt",
            }
        )
    )

    linear, angular, meta = adapter.act(
        {
            "robot_position": [0.0, 0.0],
            "robot_heading": [0.0],
            "robot_velocity_xy": [0.0, 0.0],
            "robot_radius": [0.3],
            "goal_current": [0.0, 5.0],
            "pedestrians_positions": [1.0, 0.0],
            "pedestrians_velocities": [0.0, 0.0],
            "pedestrians_count": [1],
        },
        time_step=0.25,
    )

    assert linear == pytest.approx(0.0)
    assert angular == pytest.approx(1.0)
    assert meta["human_count"] == 1
    inputs = adapter._policy.__class__.LAST_INPUTS
    assert inputs is not None
    assert inputs["spatial_edges"].shape == (1, 3, 8)
    assert inputs["visible_masks"].shape == (1, 3)
    assert float(inputs["visible_masks"][0, 0]) == pytest.approx(1.0)


def test_adapter_fails_fast_on_missing_assets_or_incompatible_source(tmp_path: Path) -> None:
    """The adapter should refuse to load missing assets or unsupported upstream kinematics."""
    with pytest.raises(FileNotFoundError, match="SoNIC-Social-Nav checkout not found"):
        SonicCrowdNavAdapter(build_sonic_crowdnav_config({"repo_root": str(tmp_path / "missing")}))

    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root, kinematics="unicycle")
    with pytest.raises(RuntimeError, match="holonomic upstream checkpoints"):
        SonicCrowdNavAdapter(
            build_sonic_crowdnav_config(
                {
                    "repo_root": str(repo_root),
                    "checkpoint_name": "05207.pt",
                }
            )
        )


def test_real_upstream_checkout_smoke_runs_if_assets_are_available() -> None:
    """The checked-in SoNIC checkout should produce one finite Robot SF command."""
    repo_root = Path("output/repos/SoNIC-Social-Nav")
    checkpoint_path = repo_root / "trained_models" / "SoNIC_GST" / "checkpoints" / "05207.pt"
    if not repo_root.exists() or not checkpoint_path.exists():
        pytest.skip("SoNIC checkout or checkpoint is unavailable in this environment")

    adapter = SonicCrowdNavAdapter(
        build_sonic_crowdnav_config(
            {
                "repo_root": str(repo_root),
                "checkpoint_name": "05207.pt",
                "max_linear_speed": 1.0,
                "max_angular_speed": 1.0,
            }
        )
    )
    linear, angular, meta = adapter.act(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.0],
                "velocity_xy": [0.0, 0.0],
                "radius": [0.3],
            },
            "goal": {"current": [5.0, 0.0]},
            "pedestrians": {
                "positions": [[2.0, 0.0], [3.0, 1.0]],
                "velocities": [[0.0, 0.0], [0.0, 0.0]],
                "count": [2],
            },
        },
        time_step=0.25,
    )

    assert math.isfinite(linear)
    assert math.isfinite(angular)
    assert -1.0 <= linear <= 1.0
    assert -1.0 <= angular <= 1.0
    assert meta["source_action_kinematics"] == "holonomic"
    assert meta["upstream_policy"] == "rl.networks.model.Policy[selfAttn_merge_srnn]"
    assert meta["parity_gaps"]
    assert np.isfinite(np.asarray(meta["upstream_action_xy"], dtype=float)).all()


@pytest.mark.parametrize(
    ("model_name", "checkpoint_name"),
    [
        ("Ours_GST", "05207.pt"),
        ("GST_predictor_rand", "05207.pt"),
    ],
)
def test_real_gensafenav_checkout_smoke_runs_if_assets_are_available(
    model_name: str,
    checkpoint_name: str,
) -> None:
    """The checked-in GenSafeNav learned checkpoints should produce finite Robot SF commands."""
    repo_root = Path("output/repos/GenSafeNav")
    checkpoint_path = repo_root / "trained_models" / model_name / "checkpoints" / checkpoint_name
    if not repo_root.exists() or not checkpoint_path.exists():
        pytest.skip("GenSafeNav checkout or checkpoint is unavailable in this environment")

    adapter = SonicCrowdNavAdapter(
        build_sonic_crowdnav_config(
            {
                "repo_root": str(repo_root),
                "model_name": model_name,
                "checkpoint_name": checkpoint_name,
                "max_linear_speed": 1.0,
                "max_angular_speed": 1.0,
            }
        )
    )
    linear, angular, meta = adapter.act(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.0],
                "velocity_xy": [0.0, 0.0],
                "radius": [0.3],
            },
            "goal": {"current": [5.0, 0.0]},
            "pedestrians": {
                "positions": [[2.0, 0.0], [3.0, 1.0]],
                "velocities": [[0.0, 0.0], [0.0, 0.0]],
                "count": [2],
            },
        },
        time_step=0.25,
    )

    assert math.isfinite(linear)
    assert math.isfinite(angular)
    assert -1.0 <= linear <= 1.0
    assert -1.0 <= angular <= 1.0
    assert meta["source_action_kinematics"] == "holonomic"
    assert meta["upstream_policy"] == "rl.networks.model.Policy[selfAttn_merge_srnn]"
    assert meta["parity_gaps"]
    assert np.isfinite(np.asarray(meta["upstream_action_xy"], dtype=float)).all()
