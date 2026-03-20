"""Tests for the benchmark-facing Social-Navigation-PyEnvs ORCA adapter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from robot_sf.planner.social_navigation_pyenvs_orca import (
    SocialNavigationPyEnvsORCAAdapter,
    _upstream_import_context,
    build_social_navigation_pyenvs_orca_config,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_fake_upstream_repo(repo_root: Path) -> None:
    _write(repo_root / "crowd_nav" / "__init__.py", "")
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


def test_build_config_uses_repo_relative_default() -> None:
    """Config builder should keep the upstream checkout path explicit."""
    cfg = build_social_navigation_pyenvs_orca_config({})
    assert cfg.repo_root == Path("output/repos/Social-Navigation-PyEnvs")


def test_build_config_rejects_negative_speed_limits() -> None:
    """Config builder should fail fast on negative speed parameters."""
    with pytest.raises(ValueError, match="must be non-negative"):
        build_social_navigation_pyenvs_orca_config({"preferred_speed": -0.1})


def test_adapter_maps_flat_observation_into_upstream_policy(tmp_path: Path) -> None:
    """Flat map-runner observations should produce projected unicycle commands."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsORCAAdapter(
        build_social_navigation_pyenvs_orca_config({"repo_root": str(repo_root)})
    )

    command_v, command_w, meta = adapter.act(
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


def test_adapter_accepts_nested_socnav_observation(tmp_path: Path) -> None:
    """Structured SocNav observations should map to the upstream JointState contract."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsORCAAdapter(
        build_social_navigation_pyenvs_orca_config({"repo_root": str(repo_root)})
    )

    command_v, command_w, meta = adapter.act(
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
            "sim": {"timestep": 0.1},
        },
        time_step=0.1,
    )

    assert command_v == pytest.approx(0.0)
    assert command_w > 0.0
    assert meta["upstream_action_xy"] == [0.0, 1.0]
    assert (
        adapter.plan(
            {
                "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
                "goal": {"current": [0.0, 1.0]},
                "pedestrians": {},
                "sim": {"timestep": 0.1},
            }
        )[1]
        > 0.0
    )


def test_adapter_fails_fast_on_missing_required_fields(tmp_path: Path) -> None:
    """Required observation fields should raise instead of being silently zero-filled."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsORCAAdapter(
        build_social_navigation_pyenvs_orca_config({"repo_root": str(repo_root)})
    )

    with pytest.raises(ValueError, match="goal.current"):
        adapter.plan(
            {
                "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
                "goal": {},
                "pedestrians": {},
                "sim": {"timestep": 0.1},
            }
        )


def test_upstream_import_context_restores_crowd_nav_modules(tmp_path: Path) -> None:
    """Import context should not leak crowd_nav modules across different repo roots."""
    repo_root_a = tmp_path / "repo_a"
    repo_root_b = tmp_path / "repo_b"
    _write_fake_upstream_repo(repo_root_a)
    _write_fake_upstream_repo(repo_root_b)
    _write(repo_root_a / "crowd_nav" / "__init__.py", "MARKER = 'A'\n")
    _write(repo_root_b / "crowd_nav" / "__init__.py", "MARKER = 'B'\n")

    with _upstream_import_context(repo_root_a):
        mod_a = importlib.import_module("crowd_nav")
        assert mod_a.MARKER == "A"

    with _upstream_import_context(repo_root_b):
        mod_b = importlib.import_module("crowd_nav")
        assert mod_b.MARKER == "B"

    sys.modules.pop("crowd_nav", None)
