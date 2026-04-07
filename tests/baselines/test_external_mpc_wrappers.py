"""Tests for the external MPC baseline wrappers."""

import os
import sys
import types
from pathlib import Path

import pytest

from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.dr_mpc import DRMPCPlanner, build_dr_mpc_config
from robot_sf.baselines.sicnav import SICNavPlanner, build_sicnav_config


def _make_robot_observation() -> dict[str, object]:
    return {
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


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_baseline_registry_contains_mpc_algorithms() -> None:
    """The baseline registry should expose the new SICNav and DR-MPC adapters."""
    names = list_baselines()
    assert "sicnav" in names
    assert "dr_mpc" in names
    assert get_baseline("sicnav") is SICNavPlanner
    assert get_baseline("dr_mpc") is DRMPCPlanner


def test_sicnav_step_raises_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing SICNav dependency should raise a runtime error during step execution."""
    planner = SICNavPlanner({}, seed=1)
    monkeypatch.setattr(
        planner,
        "_import_sicnav_module",
        lambda: (_ for _ in ()).throw(ImportError("SICNav dependency is missing.")),
    )
    with pytest.raises(RuntimeError, match="SICNav dependency"):
        planner.step(_make_robot_observation())


def test_dr_mpc_step_raises_when_dependency_missing() -> None:
    """A missing DR-MPC dependency should raise a runtime error during step execution."""
    planner = DRMPCPlanner({}, seed=1)
    with pytest.raises(RuntimeError, match="DR-MPC dependency"):
        planner.step(_make_robot_observation())


def test_sicnav_planner_uses_external_policy_from_repo_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SICNav wrapper should load a policy from a checked-out repo root."""
    repo_root = tmp_path / "sicnav_repo"
    _write(
        repo_root / "sicnav_diffusion" / "__init__.py",
        """
class SICNavPolicy:
    def __init__(self, checkpoint_path=None, solver=None, device=None):
        self.checkpoint_path = checkpoint_path
        self.solver = solver
        self.device = device
        self.seed_value = None

    def seed(self, seed):
        self.seed_value = seed

    def select_action(self, obs):
        return {"v": obs.robot["goal"][0] * 0.5, "omega": 0.1}
""",
    )
    _write(repo_root / "sicnav" / "__init__.py", "")

    original_cwd = Path.cwd()
    non_repo_cwd = tmp_path / "elsewhere"
    non_repo_cwd.mkdir()
    monkeypatch.chdir(non_repo_cwd)
    planner = SICNavPlanner(
        build_sicnav_config({"repo_root": os.path.relpath(repo_root, start=original_cwd)}),
        seed=1,
    )
    try:
        action = planner.step(_make_robot_observation())
        assert action == {"v": 0.5, "omega": 0.1}
        assert "sicnav_diffusion" in sys.modules
        assert planner._policy.seed_value == 1
        assert planner.get_metadata()["status"] == "ok"
    finally:
        sys.modules.pop("sicnav_diffusion", None)
        sys.modules.pop("sicnav", None)


def test_sicnav_metadata_marks_incompatible_module_as_missing(tmp_path: Path) -> None:
    """SICNav metadata should fail closed when the import lacks a policy constructor."""
    repo_root = tmp_path / "sicnav_repo"
    _write(repo_root / "sicnav_diffusion" / "__init__.py", "VALUE = 1\n")

    planner = SICNavPlanner(build_sicnav_config({"repo_root": str(repo_root)}), seed=1)
    try:
        assert planner.get_metadata()["status"] == "missing_dependency"
    finally:
        sys.modules.pop("sicnav_diffusion", None)
        sys.modules.pop("sicnav", None)


def test_dr_mpc_planner_uses_external_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DR-MPC wrapper should load a policy from the external package when available."""
    fake_module = types.ModuleType("dr_mpc")

    class FakePolicy:
        def __init__(self, checkpoint_path=None, device=None):
            self.checkpoint_path = checkpoint_path
            self.device = device

        def select_action(self, obs):
            return {"v": 0.25, "omega": -0.05}

    fake_module.DRMPCPolicy = FakePolicy
    monkeypatch.setitem(sys.modules, "dr_mpc", fake_module)
    planner = DRMPCPlanner({"checkpoint_path": "dummy.pt"}, seed=1)
    action = planner.step(_make_robot_observation())
    assert action == {"v": 0.25, "omega": -0.05}
    assert planner.get_metadata()["status"] == "ok"


def test_dr_mpc_config_builder_preserves_provenance_defaults() -> None:
    """DR-MPC config parsing should retain an explicit upstream repo root."""
    cfg = build_dr_mpc_config({})
    assert cfg.repo_root == "third_party/external_mpc_repos/dr_mpc"
    assert cfg.allow_testing_algorithms is True
    assert cfg.include_in_paper is False
