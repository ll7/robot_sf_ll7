"""Tests for the external MPC baseline wrappers."""

import math
import os
import sys
import types
from importlib.util import find_spec
from pathlib import Path

import pytest

from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.dr_mpc import DRMPCPlanner, build_dr_mpc_config
from robot_sf.baselines.dr_mpc import Observation as DRMPCObservation
from robot_sf.baselines.interface import Observation as InterfaceObservation
from robot_sf.baselines.sicnav import Observation as SICNavObservation
from robot_sf.baselines.sicnav import SICNavPlanner, build_sicnav_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SICNAV_STAGE_PATH = REPO_ROOT / "third_party" / "external_repos" / "sicnav"


def _make_robot_observation() -> dict[str, object]:
    """Return a minimal robot observation accepted by external MPC wrappers."""
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
    """Write a fake external-package file while creating parent directories."""
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


def test_sicnav_config_builder_uses_external_repos_default() -> None:
    """SICNav config defaults should point at the license-staged external repo path."""
    cfg = build_sicnav_config({})
    assert cfg.repo_root == "third_party/external_repos/sicnav"


def test_sicnav_skip_without_external_repo() -> None:
    """Smoke-check the staged SICNav checkout when it is present locally."""
    if not SICNAV_STAGE_PATH.exists():
        pytest.skip(f"SICNav external repo is not staged at {SICNAV_STAGE_PATH}")

    cfg = build_sicnav_config({})
    planner = SICNavPlanner(cfg, seed=1)
    assert planner._resolve_repo_root(cfg.repo_root) == SICNAV_STAGE_PATH.resolve()

    with planner._upstream_import_context():
        package_names = {"sicnav_diffusion", "sicnav"}
        package_paths = [SICNAV_STAGE_PATH / name for name in package_names]
        assert any(path.exists() for path in package_paths)
        assert any(find_spec(name) is not None for name in package_names)


def _campc_dependency_stack_available() -> bool:
    """Return True only when the full open-source campc dependency stack is importable."""
    return all(find_spec(name) is not None for name in ("casadi", "rvo2", "gym"))


def test_sicnav_campc_step_runs_against_staged_upstream() -> None:
    """Exercise the real CasADi/IPOPT CollisionAvoidMPC policy through wrapper.step().

    This is the dependency-backed smoke for issue #4870: it proves the dormant wrapper
    drives the pinned upstream on the redistributable path (CasADi + bundled IPOPT +
    python-RVO2; Acados/HSL intentionally absent). It skips cleanly when the staged
    checkout or the open-source dependency stack is not present (CI default).
    """
    if not SICNAV_STAGE_PATH.exists():
        pytest.skip(f"SICNav external repo is not staged at {SICNAV_STAGE_PATH}")
    if not _campc_dependency_stack_available():
        pytest.skip("Open-source campc dependency stack (casadi/rvo2/gym) is not installed")

    planner = SICNavPlanner(build_sicnav_config({}), seed=1)
    obs = {
        "dt": 0.25,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [6.0, 0.0],
            "radius": 0.25,
            "v_pref": 1.0,
        },
        "agents": [
            {
                "position": [3.0, 0.7 + 0.25 * i],
                "velocity": [-0.4, 0.0],
                "radius": 0.30,
                "goal": [-3.0, 0.7 + 0.25 * i],
            }
            for i in range(2)
        ],
        "obstacles": [],
    }

    action = planner.step(obs)

    assert set(action) == {"v", "omega"}
    assert math.isfinite(action["v"])
    assert math.isfinite(action["omega"])
    assert 0.0 <= action["v"] <= planner.config.v_max
    assert abs(action["omega"]) <= planner.config.omega_max + 1e-9
    # The metadata should reflect the real dependency-backed capability, not a gap.
    assert planner.get_metadata()["status"] == "ok"


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


def test_external_mpc_wrappers_reexport_shared_observation_type() -> None:
    """External MPC wrappers should expose the canonical baseline Observation."""
    assert DRMPCObservation is InterfaceObservation
    assert SICNavObservation is InterfaceObservation


def test_sicnav_planner_accepts_shared_observation_defaults(tmp_path: Path) -> None:
    """SICNav dict observations should use the shared default-obstacle container."""
    repo_root = tmp_path / "sicnav_repo"
    _write(
        repo_root / "sicnav_diffusion" / "__init__.py",
        """
class SICNavPolicy:
    def __init__(self, checkpoint_path=None, solver=None, device=None):
        self.checkpoint_path = checkpoint_path
        self.solver = solver
        self.device = device

    def select_action(self, obs):
        assert obs.obstacles == []
        return {"v": 0.5, "omega": 0.1}
""",
    )
    _write(repo_root / "sicnav" / "__init__.py", "")

    planner = SICNavPlanner(build_sicnav_config({"repo_root": str(repo_root)}), seed=1)
    obs = _make_robot_observation()
    obs.pop("obstacles")
    try:
        action = planner.step(obs)
        assert action == {"v": 0.5, "omega": 0.1}
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
        """External DR-MPC policy stub returning a fixed action."""

        def __init__(self, checkpoint_path=None, device=None):
            self.checkpoint_path = checkpoint_path
            self.device = device

        def select_action(self, obs):
            """Return the deterministic action expected by the wrapper test."""
            return {"v": 0.25, "omega": -0.05}

    fake_module.DRMPCPolicy = FakePolicy
    monkeypatch.setitem(sys.modules, "dr_mpc", fake_module)
    planner = DRMPCPlanner({"checkpoint_path": "dummy.pt"}, seed=1)
    action = planner.step(_make_robot_observation())
    assert action == {"v": 0.25, "omega": -0.05}
    assert planner.get_metadata()["status"] == "ok"


def test_dr_mpc_planner_accepts_shared_observation_defaults_and_clamps_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DR-MPC dict observations should use the shared default-obstacle container."""
    fake_module = types.ModuleType("dr_mpc")

    class FakePolicy:
        """External DR-MPC policy stub returning an over-limit velocity action."""

        def select_action(self, obs):
            assert obs.obstacles == []
            return {"vx": 3.0, "vy": 4.0}

    def load_policy(checkpoint_path=None, device=None):
        assert checkpoint_path == "dummy.pt"
        assert device == "cpu"
        return FakePolicy()

    fake_module.load_policy = load_policy
    monkeypatch.setitem(sys.modules, "dr_mpc", fake_module)
    planner = DRMPCPlanner({"checkpoint_path": "dummy.pt", "v_max": 2.0}, seed=1)
    obs = _make_robot_observation()
    obs.pop("obstacles")

    action = planner.step(obs)

    assert action == {"vx": pytest.approx(1.2), "vy": pytest.approx(1.6)}


def test_dr_mpc_config_builder_preserves_provenance_defaults() -> None:
    """DR-MPC config parsing should retain an explicit upstream repo root."""
    cfg = build_dr_mpc_config({})
    assert cfg.repo_root == "third_party/external_mpc_repos/dr_mpc"
    assert cfg.allow_testing_algorithms is True
    assert cfg.include_in_paper is False
