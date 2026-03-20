"""Tests for benchmark-facing Social-Navigation-PyEnvs force-model adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.planner.social_navigation_pyenvs_force_model import (
    SocialNavigationPyEnvsForceModelAdapter,
    build_social_navigation_pyenvs_force_model_config,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_upstream_repo(repo_root: Path) -> None:
    _write(repo_root / "crowd_nav" / "__init__.py", "")
    _write(repo_root / "crowd_nav" / "policy_no_train" / "__init__.py", "")
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

class ObservableState:
    def __init__(self, px, py, vx, vy, radius):
        self.px=px; self.py=py; self.vx=vx; self.vy=vy; self.radius=radius

class JointState:
    def __init__(self, self_state, human_states):
        self.self_state=self_state; self.human_states=human_states
""",
    )
    _write(
        repo_root / "crowd_nav" / "policy_no_train" / "socialforce.py",
        """
from crowd_nav.utils.action import ActionXY

class SocialForce:
    def predict(self, state):
        return ActionXY(state.self_state.vx + 0.2, state.self_state.vy)
""",
    )
    _write(
        repo_root / "crowd_nav" / "policy_no_train" / "sfm_helbing.py",
        """
from crowd_nav.utils.action import ActionXY

class SFMHelbing:
    def predict(self, state):
        dx = state.self_state.gx - state.self_state.px
        dy = state.self_state.gy - state.self_state.py
        return ActionXY(dx, dy)
""",
    )


def test_build_config_defaults_to_requested_policy() -> None:
    """Config builder should preserve the selected default upstream policy."""
    cfg = build_social_navigation_pyenvs_force_model_config({}, default_policy_name="socialforce")
    assert cfg.policy_name == "socialforce"
    assert cfg.repo_root == Path("output/repos/Social-Navigation-PyEnvs")


def test_build_config_rejects_unknown_policy() -> None:
    """Unknown upstream policy names should fail fast."""
    with pytest.raises(ValueError, match="Unsupported Social-Navigation-PyEnvs force-model policy"):
        build_social_navigation_pyenvs_force_model_config(
            {"policy_name": "not_a_policy"},
            default_policy_name="socialforce",
        )


def test_build_config_rejects_negative_speed_limits() -> None:
    """Negative speed limits should be rejected at config-build time."""
    with pytest.raises(ValueError, match="must be non-negative"):
        build_social_navigation_pyenvs_force_model_config(
            {"preferred_speed": -0.1},
            default_policy_name="socialforce",
        )


def test_socialforce_adapter_uses_explicit_velocity_source(tmp_path: Path) -> None:
    """SocialForce should inherit the explicit planar self-velocity contract."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsForceModelAdapter(
        build_social_navigation_pyenvs_force_model_config(
            {"repo_root": str(repo_root), "policy_name": "socialforce"},
            default_policy_name="socialforce",
        )
    )
    command_v, command_w, meta = adapter.act(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.0],
                "speed": [0.6],
                "velocity_xy": [0.3, 0.0],
                "radius": [0.3],
            },
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {"positions": [], "velocities": []},
            "sim": {"timestep": 0.1},
        },
        time_step=0.1,
    )
    assert command_v == pytest.approx(0.5)
    assert command_w == pytest.approx(0.0)
    assert meta["upstream_action_xy"] == [0.5, 0.0]
    assert meta["self_velocity_source"] == "robot.velocity_xy"
    assert meta["upstream_policy"] == "crowd_nav.policy_no_train.socialforce.SocialForce"


def test_sfm_helbing_adapter_maps_goal_direction(tmp_path: Path) -> None:
    """SFM-Helbing should map upstream ActionXY outputs into projected unicycle commands."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsForceModelAdapter(
        build_social_navigation_pyenvs_force_model_config(
            {"repo_root": str(repo_root), "policy_name": "sfm_helbing"},
            default_policy_name="sfm_helbing",
        )
    )
    command_v, command_w, meta = adapter.act(
        {
            "robot_position": [0.0, 0.0],
            "robot_heading": [0.0],
            "robot_speed": [0.0],
            "robot_radius": [0.3],
            "goal_current": [0.0, 1.0],
            "pedestrians_positions": [[1.0, 1.0]],
            "pedestrians_velocities": [[0.0, 0.0]],
            "pedestrians_count": [1],
            "pedestrians_radius": [0.3],
        },
        time_step=0.1,
    )
    assert command_v == pytest.approx(0.0)
    assert command_w > 0.0
    assert meta["upstream_action_xy"] == [0.0, 1.0]
    assert meta["upstream_policy"] == "crowd_nav.policy_no_train.sfm_helbing.SFMHelbing"
