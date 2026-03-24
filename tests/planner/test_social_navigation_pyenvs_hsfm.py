"""Tests for benchmark-facing Social-Navigation-PyEnvs HSFM adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.planner.social_navigation_pyenvs_hsfm import (
    SocialNavigationPyEnvsHSFMAdapter,
    build_social_navigation_pyenvs_hsfm_config,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_upstream_repo(repo_root: Path) -> None:
    _write(repo_root / "crowd_nav" / "__init__.py", "")
    _write(repo_root / "crowd_nav" / "policy_no_train" / "__init__.py", "")
    _write(
        repo_root / "crowd_nav" / "utils" / "action.py",
        """
from collections import namedtuple
ActionXYW = namedtuple('ActionXYW', ['bvx', 'bvy', 'w'])
NewHeadedState = namedtuple('NewHeadedState', ['px', 'py', 'theta', 'bvx', 'bvy', 'w'])
""",
    )
    _write(
        repo_root / "crowd_nav" / "utils" / "state.py",
        """
class ObservableState:
    def __init__(self, px, py, vx, vy, radius):
        self.px=px; self.py=py; self.vx=vx; self.vy=vy; self.radius=radius

class ObservableStateHeaded:
    def __init__(self, px, py, vx, vy, radius, theta, omega):
        self.px=px; self.py=py; self.vx=vx; self.vy=vy
        self.radius=radius; self.theta=theta; self.omega=omega

class FullStateHeaded:
    def __init__(self, px, py, vx, vy, radius, gx, gy, v_pref, theta, w):
        self.px=px; self.py=py; self.vx=vx; self.vy=vy; self.radius=radius
        self.gx=gx; self.gy=gy; self.v_pref=v_pref; self.theta=theta; self.w=w

class JointState:
    def __init__(self, self_state, human_states):
        self.self_state=self_state; self.human_states=human_states
""",
    )
    _write(
        repo_root / "crowd_nav" / "policy_no_train" / "hsfm_new_guo.py",
        """
from crowd_nav.utils.action import ActionXYW

class HSFMNewGuo:
    def predict(self, state):
        return ActionXYW(state.self_state.vx + 0.2, 0.3, state.self_state.w + 0.4)
""",
    )
    _write(
        repo_root / "crowd_nav" / "policy_no_train" / "hsfm_farina.py",
        """
from crowd_nav.utils.action import NewHeadedState

class HSFMFarina:
    def predict(self, state):
        return NewHeadedState(
            state.self_state.px,
            state.self_state.py,
            state.self_state.theta,
            state.self_state.vx,
            state.self_state.vy + 0.2,
            state.self_state.w - 0.1,
        )
""",
    )


def test_build_config_defaults_to_requested_policy() -> None:
    """Config builder should preserve the selected default upstream HSFM policy."""
    cfg = build_social_navigation_pyenvs_hsfm_config({}, default_policy_name="hsfm_new_guo")
    assert cfg.policy_name == "hsfm_new_guo"
    assert cfg.repo_root == Path("output/repos/Social-Navigation-PyEnvs")


def test_build_config_rejects_unknown_policy() -> None:
    """Unknown upstream policy names should fail fast."""
    with pytest.raises(ValueError, match="Unsupported Social-Navigation-PyEnvs HSFM policy"):
        build_social_navigation_pyenvs_hsfm_config({"policy_name": "not_a_policy"})


def test_build_config_rejects_negative_speed_limits() -> None:
    """Negative speed limits should be rejected at config-build time."""
    with pytest.raises(ValueError, match="must be non-negative"):
        build_social_navigation_pyenvs_hsfm_config({"max_angular_speed": -0.1})


@pytest.mark.parametrize(
    "field,value", [("preferred_speed", float("nan")), ("max_linear_speed", float("inf"))]
)
def test_build_config_rejects_non_finite_speed_limits(field: str, value: float) -> None:
    """Non-finite speed limits should fail fast at config-build time."""
    with pytest.raises(ValueError, match="must be finite"):
        build_social_navigation_pyenvs_hsfm_config({field: value})


def test_hsfm_adapter_uses_explicit_velocity_and_angular_rate(tmp_path: Path) -> None:
    """HSFM should consume explicit planar velocity and angular rate from the observation."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsHSFMAdapter(
        build_social_navigation_pyenvs_hsfm_config(
            {"repo_root": str(repo_root), "policy_name": "hsfm_new_guo"}
        )
    )
    command_v, command_w, meta = adapter.act(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.0],
                "speed": [0.3, 0.2],
                "velocity_xy": [0.3, 0.0],
                "angular_velocity": [0.2],
                "radius": [0.3],
            },
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {"positions": [], "velocities": []},
            "sim": {"timestep": 0.1},
        },
        time_step=0.1,
    )
    assert command_v == pytest.approx(0.5)
    assert command_w == pytest.approx(1.0)
    assert meta["upstream_action_body_xyw"] == pytest.approx([0.5, 0.3, 0.6])
    assert meta["upstream_action_kind"] == "ActionXYW"
    assert meta["self_velocity_source"] == "robot.velocity_xy+robot.angular_velocity"


def test_hsfm_adapter_handles_new_headed_state_outputs(tmp_path: Path) -> None:
    """Runge-Kutta style HSFM outputs should still project into benchmark-visible commands."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsHSFMAdapter(
        build_social_navigation_pyenvs_hsfm_config(
            {"repo_root": str(repo_root), "policy_name": "hsfm_farina"}
        )
    )
    command_v, command_w, meta = adapter.act(
        {
            "robot_position": [0.0, 0.0],
            "robot_heading": [0.0],
            "robot_velocity_xy": [0.4, 0.0],
            "robot_angular_velocity": [0.3],
            "robot_radius": [0.3],
            "goal_current": [1.0, 0.0],
            "pedestrians_positions": [[1.0, 1.0]],
            "pedestrians_velocities": [[0.0, 0.0]],
            "pedestrians_count": [1],
            "pedestrians_radius": [0.3],
        },
        time_step=0.1,
    )
    assert command_v == pytest.approx(0.4)
    assert command_w > 0.0
    assert meta["upstream_action_kind"] == "NewHeadedState"
    assert meta["upstream_policy"] == "crowd_nav.policy_no_train.hsfm_farina.HSFMFarina"


def test_hsfm_adapter_requires_explicit_angular_velocity(tmp_path: Path) -> None:
    """Missing angular rate should fail fast instead of guessing the headed contract."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsHSFMAdapter(
        build_social_navigation_pyenvs_hsfm_config(
            {
                "repo_root": str(repo_root),
                "policy_name": "hsfm_new_guo",
                "max_angular_speed": 10.0,
            }
        )
    )
    with pytest.raises(ValueError, match="robot.angular_velocity"):
        adapter.plan(
            {
                "robot": {
                    "position": [0.0, 0.0],
                    "heading": [0.0],
                    "speed": [0.3, 0.0],
                    "velocity_xy": [0.3, 0.0],
                    "radius": [0.3],
                },
                "goal": {"current": [1.0, 0.0]},
                "pedestrians": {"positions": [], "velocities": []},
            }
        )


def test_hsfm_adapter_requires_explicit_radius_fields(tmp_path: Path) -> None:
    """Missing radius fields should fail fast instead of defaulting to a guessed value."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsHSFMAdapter(
        build_social_navigation_pyenvs_hsfm_config(
            {
                "repo_root": str(repo_root),
                "policy_name": "hsfm_new_guo",
                "max_angular_speed": 10.0,
            }
        )
    )
    with pytest.raises(ValueError, match="robot.radius"):
        adapter.plan(
            {
                "robot": {
                    "position": [0.0, 0.0],
                    "heading": [0.0],
                    "speed": [0.3, 0.0],
                    "velocity_xy": [0.3, 0.0],
                    "angular_velocity": [0.2],
                },
                "goal": {"current": [1.0, 0.0]},
                "pedestrians": {"positions": [[1.0, 1.0]], "velocities": [[0.0, 0.0]]},
            }
        )


def test_hsfm_adapter_requires_explicit_pedestrian_radius_fields(tmp_path: Path) -> None:
    """Missing pedestrian radius fields should fail fast when pedestrians are present."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsHSFMAdapter(
        build_social_navigation_pyenvs_hsfm_config(
            {
                "repo_root": str(repo_root),
                "policy_name": "hsfm_new_guo",
                "max_angular_speed": 10.0,
            }
        )
    )
    with pytest.raises(ValueError, match="pedestrians.radius"):
        adapter.plan(
            {
                "robot": {
                    "position": [0.0, 0.0],
                    "heading": [0.0],
                    "speed": [0.3, 0.0],
                    "velocity_xy": [0.3, 0.0],
                    "angular_velocity": [0.2],
                    "radius": [0.3],
                },
                "goal": {"current": [1.0, 0.0]},
                "pedestrians": {"positions": [[1.0, 1.0]], "velocities": [[0.0, 0.0]]},
            }
        )


def test_hsfm_adapter_plan_reads_flattened_sim_timestep(tmp_path: Path) -> None:
    """Flattened benchmark observations should preserve the configured timestep."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsHSFMAdapter(
        build_social_navigation_pyenvs_hsfm_config(
            {
                "repo_root": str(repo_root),
                "policy_name": "hsfm_new_guo",
                "max_angular_speed": 10.0,
            }
        )
    )
    command_v, command_w = adapter.plan(
        {
            "robot_position": [0.0, 0.0],
            "robot_heading": [0.0],
            "robot_velocity_xy": [0.3, 0.0],
            "robot_angular_velocity": [0.2],
            "robot_radius": [0.3],
            "goal_current": [1.0, 0.0],
            "pedestrians_positions": [],
            "pedestrians_velocities": [],
            "sim_timestep": [0.2],
        }
    )
    assert command_v == pytest.approx(0.5)
    assert command_w == pytest.approx(3.302097501352921)


@pytest.mark.parametrize("time_step", [float("nan"), float("inf"), 0.0, -1.0, None])
def test_hsfm_adapter_act_sanitizes_invalid_timestep(
    tmp_path: Path, time_step: float | None
) -> None:
    """Non-finite or non-positive timesteps should fall back to the safe default."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsHSFMAdapter(
        build_social_navigation_pyenvs_hsfm_config(
            {
                "repo_root": str(repo_root),
                "policy_name": "hsfm_new_guo",
                "max_angular_speed": 10.0,
            }
        )
    )
    command_v, command_w, meta = adapter.act(
        {
            "robot": {
                "position": [0.0, 0.0],
                "heading": [0.0],
                "speed": [0.3, 0.2],
                "velocity_xy": [0.3, 0.0],
                "angular_velocity": [0.2],
                "radius": [0.3],
            },
            "goal": {"current": [1.0, 0.0]},
            "pedestrians": {"positions": [], "velocities": []},
            "sim": {"timestep": 0.1},
        },
        time_step=time_step,
    )
    assert command_v == pytest.approx(0.5)
    assert command_w == pytest.approx(6.004195002705842)
    assert meta["projected_command_vw"] == pytest.approx([0.5, 6.004195002705842])
