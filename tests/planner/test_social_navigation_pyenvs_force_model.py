"""Tests for benchmark-facing Social-Navigation-PyEnvs force-model adapters."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

from robot_sf.planner.social_navigation_pyenvs_force_model import (
    SocialNavigationPyEnvsForceModelAdapter,
    _build_socialforce_compat_module,
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


def _install_fake_socialforce_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake external socialforce backend that needs the compatibility shim."""
    backend = types.ModuleType("socialforce")
    backend.__version__ = "0.2.3"

    class FakeSimulator:
        def __init__(self, *, delta_t: float = 0.4) -> None:
            self.delta_t = delta_t

        def forward(self, state):
            class FakeTensor:
                def __init__(self, value):
                    self._value = value

                def detach(self):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return self._value

            value = [[0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0]]
            return FakeTensor(value)

    backend.Simulator = FakeSimulator
    monkeypatch.setitem(sys.modules, "socialforce", backend)


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


def test_socialforce_adapter_uses_explicit_velocity_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SocialForce should inherit the explicit planar self-velocity contract."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    _install_fake_socialforce_backend(monkeypatch)
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
    assert meta["runtime_strategy"] == "crowdnav_socialforce_compat_shim"
    assert meta["runtime_dependency"] == "socialforce==0.2.3"


def test_socialforce_adapter_requires_external_runtime_dependency(tmp_path: Path) -> None:
    """A clear runtime error should surface when the external socialforce package is absent."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "socialforce":
            raise ModuleNotFoundError("No module named 'socialforce'", name="socialforce")
        return real_import_module(name, package)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "robot_sf.planner.social_navigation_pyenvs_force_model.importlib.import_module",
        fake_import_module,
    )

    try:
        with pytest.raises(ModuleNotFoundError, match="uv run --with socialforce==0.2.3"):
            SocialNavigationPyEnvsForceModelAdapter(
                build_social_navigation_pyenvs_force_model_config(
                    {"repo_root": str(repo_root), "policy_name": "socialforce"},
                    default_policy_name="socialforce",
                )
            )
    finally:
        monkeypatch.undo()


def test_socialforce_compat_simulator_detaches_state_before_caching() -> None:
    """The compat shim should not retain autograd history across simulator steps."""
    backend = types.ModuleType("socialforce")

    class FakeSimulator:
        def __init__(self, *, delta_t: float = 0.4) -> None:
            self.delta_t = delta_t

        def forward(self, state):
            tracked_state = state.clone().detach().requires_grad_(True)
            return tracked_state * 2.0

    backend.Simulator = FakeSimulator
    compat = _build_socialforce_compat_module(backend)
    simulator = compat.Simulator(torch.tensor([[1.0, 0.0]]))

    simulator.step()

    assert simulator._state.requires_grad is False
    assert simulator.state.tolist() == [[2.0, 0.0]]


def test_socialforce_adapter_propagates_transitive_import_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transitive dependency failures should not be mislabeled as a missing socialforce install."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "socialforce":
            raise ModuleNotFoundError("No module named 'missing_backend_dep'")
        return real_import_module(name, package)

    monkeypatch.setattr(
        "robot_sf.planner.social_navigation_pyenvs_force_model.importlib.import_module",
        fake_import_module,
    )

    with pytest.raises(ModuleNotFoundError, match="missing_backend_dep"):
        SocialNavigationPyEnvsForceModelAdapter(
            build_social_navigation_pyenvs_force_model_config(
                {"repo_root": str(repo_root), "policy_name": "socialforce"},
                default_policy_name="socialforce",
            )
        )


def test_socialforce_adapter_rejects_unvalidated_backend_version(tmp_path: Path) -> None:
    """The runtime contract should only accept the validated socialforce backend version."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)

    class FakeBackend:
        __version__ = "0.2.4"

        class Simulator:
            def __init__(self, *, delta_t: float = 0.4) -> None:
                self.delta_t = delta_t

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setitem(sys.modules, "socialforce", FakeBackend())
    try:
        with pytest.raises(RuntimeError, match="socialforce==0.2.3"):
            SocialNavigationPyEnvsForceModelAdapter(
                build_social_navigation_pyenvs_force_model_config(
                    {"repo_root": str(repo_root), "policy_name": "socialforce"},
                    default_policy_name="socialforce",
                )
            )
    finally:
        monkeypatch.undo()


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


def test_plan_defaults_invalid_dt_to_safe_fallback(tmp_path: Path) -> None:
    """Malformed or non-finite timestep payloads should fall back to the safe default."""
    repo_root = tmp_path / "repo"
    _write_fake_upstream_repo(repo_root)
    adapter = SocialNavigationPyEnvsForceModelAdapter(
        build_social_navigation_pyenvs_force_model_config(
            {"repo_root": str(repo_root), "policy_name": "sfm_helbing"},
            default_policy_name="sfm_helbing",
        )
    )
    command = adapter.plan(
        {
            "robot_position": [0.0, 0.0],
            "robot_heading": [0.0],
            "robot_speed": [0.0],
            "robot_radius": [0.3],
            "goal_current": [1.0, 0.0],
            "pedestrians_positions": [],
            "pedestrians_velocities": [],
            "dt": [],
            "sim": {"timestep": float("nan")},
        }
    )
    assert command == pytest.approx((1.0, 0.0))
