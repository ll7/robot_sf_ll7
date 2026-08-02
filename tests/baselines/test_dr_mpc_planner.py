"""Tests for robot_sf.baselines.dr_mpc — DR-MPC baseline wrapper."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from robot_sf.baselines.dr_mpc import (
    DRMPCPlanner,
    DRMPCPlannerConfig,
    build_dr_mpc_config,
)


class TestDRMPCPlannerConfig:
    """Tests for DRMPCPlannerConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config must have sane baseline values."""
        cfg = DRMPCPlannerConfig()
        assert cfg.checkpoint_path is None
        assert cfg.device == "cpu"
        assert cfg.mode == "unicycle"
        assert cfg.v_max == 2.0
        assert cfg.omega_max == 1.0
        assert cfg.safety_clamp is True
        assert cfg.fallback_on_error is False
        assert cfg.include_in_paper is False

    def test_custom_values(self) -> None:
        """Custom config values must be stored."""
        cfg = DRMPCPlannerConfig(
            checkpoint_path="/tmp/ckpt.pt",
            device="cuda",
            v_max=3.0,
            omega_max=2.0,
        )
        assert cfg.checkpoint_path == "/tmp/ckpt.pt"
        assert cfg.device == "cuda"
        assert cfg.v_max == 3.0
        assert cfg.omega_max == 2.0


class TestBuildDrMpcConfig:
    """Tests for build_dr_mpc_config factory."""

    def test_none_input_returns_defaults(self) -> None:
        """None input must return default config."""
        cfg = build_dr_mpc_config(None)
        assert isinstance(cfg, DRMPCPlannerConfig)
        assert cfg.v_max == 2.0

    def test_empty_dict_returns_defaults(self) -> None:
        """An empty dict must return default config."""
        cfg = build_dr_mpc_config({})
        assert cfg.device == "cpu"

    def test_known_fields_applied(self) -> None:
        """Known fields from the dict must be applied."""
        cfg = build_dr_mpc_config({"v_max": 5.0, "device": "cuda"})
        assert cfg.v_max == 5.0
        assert cfg.device == "cuda"

    def test_unknown_fields_ignored(self) -> None:
        """Unknown fields must be silently ignored."""
        cfg = build_dr_mpc_config({"v_max": 1.0, "nonexistent_field": "value"})
        assert cfg.v_max == 1.0
        assert not hasattr(cfg, "nonexistent_field")

    def test_repo_root_expanded(self) -> None:
        """repo_root with ~ must be expanded."""
        cfg = build_dr_mpc_config({"repo_root": "~/dr_mpc"})
        assert "~" not in cfg.repo_root
        assert Path(cfg.repo_root).is_absolute()


class TestDRMPCPlanner:
    """Tests for DRMPCPlanner wrapper behavior."""

    def test_init_with_dict_config(self) -> None:
        """Init with a dict config must parse into DRMPCPlannerConfig."""
        planner = DRMPCPlanner({"v_max": 3.0})
        assert isinstance(planner.config, DRMPCPlannerConfig)
        assert planner.config.v_max == 3.0

    def test_init_with_dataclass_config(self) -> None:
        """Init with a DRMPCPlannerConfig must use it directly."""
        cfg = DRMPCPlannerConfig(v_max=4.0)
        planner = DRMPCPlanner(cfg)
        assert planner.config.v_max == 4.0

    def test_init_with_invalid_config_type_raises(self) -> None:
        """Init with an invalid config type must raise TypeError."""
        with pytest.raises(TypeError, match="Invalid config type"):
            DRMPCPlanner("not_a_config")  # type: ignore[arg-type]

    def test_reset_clears_state(self) -> None:
        """reset must clear cached policy and module."""
        planner = DRMPCPlanner({})
        planner._policy = "fake_policy"
        planner._module = "fake_module"
        planner.reset()
        assert planner._policy is None
        assert planner._module is None

    def test_reset_with_seed(self) -> None:
        """reset with a seed must update the internal seed."""
        planner = DRMPCPlanner({}, seed=1)
        planner.reset(seed=42)
        assert planner._seed == 42

    def test_configure_updates_config(self) -> None:
        """configure must update the config and clear cached state."""
        planner = DRMPCPlanner({})
        planner._policy = "fake"
        planner.configure({"v_max": 10.0})
        assert planner.config.v_max == 10.0
        assert planner._policy is None

    def test_close_clears_resources(self) -> None:
        """close must release cached policy and module."""
        planner = DRMPCPlanner({})
        planner._policy = "fake"
        planner._module = "fake"
        planner.close()
        assert planner._policy is None
        assert planner._module is None

    def test_step_raises_when_dependency_missing(self) -> None:
        """step must raise RuntimeError when the DR-MPC dependency is missing."""
        planner = DRMPCPlanner({})
        obs = {
            "dt": 0.1,
            "robot": {
                "position": [0.0, 0.0],
                "velocity": [0.0, 0.0],
                "goal": [1.0, 0.0],
                "radius": 0.3,
            },
            "agents": [],
            "obstacles": [],
        }
        with pytest.raises(RuntimeError, match="DR-MPC dependency"):
            planner.step(obs)

    def test_get_metadata_structure(self) -> None:
        """get_metadata must return algorithm, config, config_hash, and status."""
        planner = DRMPCPlanner({})
        meta = planner.get_metadata()
        assert meta["algorithm"] == "dr_mpc"
        assert "config" in meta
        assert "config_hash" in meta
        assert isinstance(meta["config_hash"], str)
        assert len(meta["config_hash"]) == 16

    def test_get_metadata_missing_dependency_status(self) -> None:
        """get_metadata must report missing_dependency when DR-MPC is absent."""
        planner = DRMPCPlanner({})
        meta = planner.get_metadata()
        assert meta["status"] == "missing_dependency"

    def test_clamp_action_velocity(self) -> None:
        """Safety clamp must scale velocity commands exceeding v_max."""
        planner = DRMPCPlanner({"v_max": 1.0, "safety_clamp": True})
        action = {"vx": 3.0, "vy": 4.0}
        planner._clamp_action(action)
        import math

        speed = math.hypot(action["vx"], action["vy"])
        assert speed == pytest.approx(1.0, abs=1e-6)

    def test_clamp_action_unicycle(self) -> None:
        """Safety clamp must limit v and omega for unicycle commands."""
        planner = DRMPCPlanner({"v_max": 1.0, "omega_max": 0.5, "safety_clamp": True})
        action = {"v": 5.0, "omega": 2.0}
        planner._clamp_action(action)
        assert action["v"] == pytest.approx(1.0)
        assert action["omega"] == pytest.approx(0.5)

    def test_clamp_action_negative_omega(self) -> None:
        """Safety clamp must clamp negative omega to -omega_max."""
        planner = DRMPCPlanner({"omega_max": 0.5, "safety_clamp": True})
        action = {"omega": -2.0}
        planner._clamp_action(action)
        assert action["omega"] == pytest.approx(-0.5)

    def test_clamp_disabled(self) -> None:
        """With safety_clamp=False, actions must not be modified."""
        planner = DRMPCPlanner({"v_max": 1.0, "safety_clamp": False})
        action = {"v": 100.0}
        planner._clamp_action(action)
        assert action["v"] == 100.0

    def test_import_module_caches(self) -> None:
        """_import_dr_mpc_module must cache the imported module."""
        planner = DRMPCPlanner({})
        fake_module = types.ModuleType("dr_mpc")
        planner._module = fake_module
        result = planner._import_dr_mpc_module()
        assert result is fake_module

    def test_build_policy_no_constructor_raises(self) -> None:
        """_build_policy must raise when the module lacks a supported constructor."""
        planner = DRMPCPlanner({})
        fake_module = types.ModuleType("dr_mpc")
        planner._module = fake_module
        with pytest.raises(RuntimeError, match="supported policy constructor"):
            planner._build_policy()
