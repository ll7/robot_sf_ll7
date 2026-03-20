"""Tests for the Social-Navigation-PyEnvs ORCA wrapper probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.tools import probe_social_navigation_pyenvs_orca_wrapper as probe
from scripts.tools.probe_social_navigation_pyenvs_orca_wrapper import (
    SocialNavigationPyEnvsORCAWrapper,
    _render_markdown,
    run_probe,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_upstream_repo(repo_root: Path) -> None:
    _write(
        repo_root / "crowd_nav" / "utils" / "action.py",
        "from collections import namedtuple\nActionXY = namedtuple('ActionXY', ['vx', 'vy'])\n",
    )
    _write(
        repo_root / "crowd_nav" / "utils" / "state.py",
        """
class FullState:
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px=px; self.py=py; self.vx=vx; self.vy=vy
        self.radius=radius; self.gx=gx; self.gy=gy; self.v_pref=v_pref; self.theta=theta
        self.position=(px, py); self.goal_position=(gx, gy); self.velocity=(vx, vy)

class ObservableState:
    def __init__(self, px, py, vx, vy, radius):
        self.px=px; self.py=py; self.vx=vx; self.vy=vy; self.radius=radius

class JointState:
    def __init__(self, self_state, human_states):
        self.self_state=self_state; self.human_states=human_states
""",
    )
    _write(
        repo_root / "crowd_nav" / "policy_no_train" / "orca.py",
        """
from crowd_nav.utils.action import ActionXY

class ORCA:
    def predict(self, state):
        dx = state.self_state.gx - state.self_state.px
        dy = state.self_state.gy - state.self_state.py
        return ActionXY(dx, dy)
""",
    )


def test_wrapper_maps_observation_into_upstream_policy(tmp_path: Path) -> None:
    """The wrapper should call the upstream ORCA policy and project ActionXY to `(v, w)`."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    wrapper = SocialNavigationPyEnvsORCAWrapper(
        repo_root, preferred_speed=1.0, max_linear_speed=1.0, max_angular_speed=2.0
    )
    command_v, command_w, meta = wrapper.act(
        {
            "robot_position": [0.0, 0.0],
            "robot_heading": [0.0],
            "robot_speed": [0.0],
            "robot_radius": [0.3],
            "goal_current": [1.0, 0.0],
            "pedestrians_positions": [[1.0, 1.0]],
            "pedestrians_velocities": [[0.0, 0.0]],
            "pedestrians_count": [1],
            "pedestrians_radius": [0.3],
        },
        time_step=0.1,
    )

    assert command_v == pytest.approx(1.0)
    assert command_w == pytest.approx(0.0)
    assert meta["upstream_action_xy"] == [1.0, 0.0]
    assert meta["projection_policy"] == "heading_safe_velocity_to_unicycle_vw"


def test_wrapper_accepts_nested_socnav_observation(tmp_path: Path) -> None:
    """The wrapper should also support the live nested SocNav observation format."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    wrapper = SocialNavigationPyEnvsORCAWrapper(repo_root)

    command_v, command_w, meta = wrapper.act(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.0],
                "speed": [0.0],
                "radius": [0.3],
            },
            "goal": {"current": [0.0, 1.0]},
            "pedestrians": {
                "positions": [[1.0, 1.0]],
                "velocities": [[0.0, 0.0]],
                "count": [1],
                "radius": [0.3],
            },
        },
        time_step=0.1,
    )

    assert command_v == pytest.approx(0.0)
    assert command_w > 0.0
    assert meta["upstream_action_xy"] == [0.0, 1.0]


class _FakeRobot:
    pass


class _FakeSimulator:
    def __init__(self) -> None:
        self.robots = [_FakeRobot()]


class _FakeEnv:
    def __init__(self) -> None:
        self.simulator = _FakeSimulator()
        self.action_space = object()
        self.env_config = type(
            "Cfg", (), {"sim_config": type("Sim", (), {"time_per_step_in_secs": 0.1})()}
        )()
        self._obs = {
            "robot_position": np.array([0.0, 0.0]),
            "robot_heading": np.array([0.0]),
            "robot_speed": np.array([0.0]),
            "robot_radius": np.array([0.3]),
            "goal_current": np.array([1.0, 0.0]),
            "pedestrians_positions": np.array([[1.0, 0.5]]),
            "pedestrians_velocities": np.array([[0.0, 0.0]]),
            "pedestrians_count": np.array([1]),
            "pedestrians_radius": np.array([0.3]),
        }

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        return dict(self._obs), {"seed": seed}

    def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
        return dict(self._obs), 0.0, False, False, {"action": action}

    def close(self) -> None:
        return None


class _FakePlannerActionAdapter:
    def __init__(self, robot: object, action_space: object, time_step: float) -> None:
        self.time_step = time_step

    def from_velocity_command(self, command: tuple[float, float]) -> dict[str, float]:
        return {"v": float(command[0]), "omega": float(command[1])}


def test_run_probe_executes_robot_sf_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The wrapper probe should execute at least one Robot SF step and report viability."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    monkeypatch.setattr(probe, "make_robot_env", lambda config, debug=False: _FakeEnv())
    monkeypatch.setattr(probe, "PlannerActionAdapter", _FakePlannerActionAdapter)

    report = run_probe(repo_root, seed=3, max_steps=2)

    assert report.verdict == "wrapper prototype viable"
    assert report.steps_executed == 2
    assert report.upstream_policy == "crowd_nav.policy_no_train.orca.ORCA"
    assert report.projection_policy == "heading_safe_velocity_to_unicycle_vw"


def test_render_markdown_mentions_real_robot_sf_step(tmp_path: Path) -> None:
    """Markdown should state the wrapper proof boundary clearly."""
    report = probe.WrapperProbeReport(
        issue=642,
        repo_root=str(tmp_path / "repo"),
        verdict="wrapper prototype viable",
        projection_policy="heading_safe_velocity_to_unicycle_vw",
        wrapper_boundary="test boundary",
        upstream_policy="crowd_nav.policy_no_train.orca.ORCA",
        steps_executed=1,
        latest_robot_command=[0.5, 0.1],
        latest_upstream_action_xy=[0.5, 0.2],
        latest_heading_error_rad=0.2,
        observation_keys=["goal_current", "robot_position"],
    )

    markdown = _render_markdown(report)
    assert "wrapper prototype viable" in markdown
    assert "real Robot SF step loop" in markdown
