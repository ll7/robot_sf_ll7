"""Tests for the BRNE baseline planner adapter (issue #5318).

Exercises the bounded BRNE integration tier: corridor-class scenarios only,
fail-closed budget enforcement, native unicycle output.  Tests that need the
staged BRNE clone skip cleanly when it is absent (CI default).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.brne import BRNEPlanner, BRNEPlannerConfig, build_brne_config

REPO_ROOT = Path(__file__).resolve().parents[2]
BRNE_STAGE_PATH = REPO_ROOT / "third_party" / "external_repos" / "brne"


def _brne_dependency_stack_available() -> bool:
    """Return True only when the BRNE core dependency stack is importable."""
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "numba"))


def _make_observation(
    num_agents: int = 1,
    robot_pos: list[float] | None = None,
    robot_goal: list[float] | None = None,
) -> dict[str, object]:
    """Return a minimal observation accepted by the BRNE adapter."""
    return {
        "dt": 0.1,
        "robot": {
            "position": robot_pos or [0.0, 0.0],
            "velocity": [0.4, 0.0],
            "goal": robot_goal or [6.0, 0.0],
            "radius": 0.3,
        },
        "agents": [
            {
                "position": [3.0 - i * 0.5, 0.4 * ((-1) ** i)],
                "velocity": [-0.4, 0.0],
                "radius": 0.3,
            }
            for i in range(max(0, num_agents - 1))
        ],
        "obstacles": [],
    }


def _fake_brne_module() -> object:
    """Return a placeholder upstream module for isolated adapter tests."""
    return object()


def _fake_covariance(_brne: object) -> np.ndarray:
    """Return the minimal covariance placeholder for isolated adapter tests."""
    return np.eye(1)


# --- Registry ---


def test_baseline_registry_contains_brne() -> None:
    """The baseline registry should expose the BRNE adapter."""
    names = list_baselines()
    assert "brne" in names
    assert get_baseline("brne") is BRNEPlanner


# --- Config ---


def test_build_brne_config_defaults() -> None:
    """BRNE config defaults should point at the license-staged external repo path."""
    cfg = build_brne_config({})
    assert cfg.stage_path == "third_party/external_repos/brne"
    assert cfg.num_samples == 196
    assert cfg.maximum_agents == 8
    assert cfg.action_space == "unicycle"
    assert cfg.allow_testing_algorithms is True
    assert cfg.include_in_paper is False


def test_build_brne_config_with_overrides() -> None:
    """BRNE config should accept explicit overrides."""
    cfg = build_brne_config({"num_samples": 49, "corridor_y_min": -1.0, "v_max": 1.5})
    assert cfg.num_samples == 49
    assert cfg.corridor_y_min == -1.0
    assert cfg.v_max == 1.5


def test_build_brne_config_ignores_unknown_keys() -> None:
    """BRNE config builder should ignore keys not in the dataclass."""
    cfg = build_brne_config({"num_samples": 64, "bogus_key": 42})
    assert cfg.num_samples == 64
    assert not hasattr(cfg, "bogus_key")


# --- Planner initialization ---


def test_brne_planner_init_with_dict() -> None:
    """BRNE planner should accept a dict config."""
    planner = BRNEPlanner({"num_samples": 49})
    assert planner.config.num_samples == 49


def test_brne_planner_init_with_dataclass() -> None:
    """BRNE planner should accept a dataclass config."""
    cfg = BRNEPlannerConfig(num_samples=64)
    planner = BRNEPlanner(cfg)
    assert planner.config.num_samples == 64


def test_brne_planner_init_rejects_invalid_config() -> None:
    """BRNE planner should reject non-dict/non-dataclass config."""
    with pytest.raises(TypeError, match="Invalid config type"):
        BRNEPlanner(42)  # type: ignore[arg-type]


# --- Metadata ---


def test_brne_metadata_when_staged_repo_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Metadata should report missing_dependency when the staged repo is absent."""
    planner = BRNEPlanner({"stage_path": str(tmp_path / "nonexistent")})
    meta = planner.get_metadata()
    assert meta["algorithm"] == "brne"
    assert meta["status"] == "missing_dependency"
    assert "GPL-3.0" in meta["license"]


def test_brne_metadata_when_staged_repo_present() -> None:
    """Metadata should report ok when the staged repo is present."""
    if not BRNE_STAGE_PATH.exists():
        pytest.skip("BRNE external repo is not staged")
    planner = BRNEPlanner({})
    meta = planner.get_metadata()
    assert meta["algorithm"] == "brne"
    assert meta["status"] == "ok"


# --- Step fails closed when dependency missing ---


def test_brne_step_raises_when_stage_path_missing() -> None:
    """A missing staged BRNE clone should raise FileNotFoundError."""
    planner = BRNEPlanner({"stage_path": "/nonexistent/path/brne"})
    with pytest.raises(FileNotFoundError, match="BRNE core algorithm not found"):
        planner.step(_make_observation())


def test_brne_config_does_not_expose_unimplemented_adaptive_sampling() -> None:
    """The bounded adapter must not advertise an unused sampling policy."""
    cfg = build_brne_config({"adaptive_num_samples": True})
    assert not hasattr(cfg, "adaptive_num_samples")


def test_brne_solve_fails_closed_on_nonfinite_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-finite upstream weights must not propagate into a control action."""
    planner = BRNEPlanner({})
    monkeypatch.setattr(planner, "_ensure_brne_loaded", _fake_brne_module)
    monkeypatch.setattr(planner, "_ensure_cov", _fake_covariance)
    monkeypatch.setattr(
        planner,
        "_build_trajectories",
        lambda *_args: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.ones((2, 1, 2)),
        ),
    )
    monkeypatch.setattr(
        planner,
        "_brne_solve",
        lambda *_args: np.array([[np.nan, 1.0]]),
    )
    planner._jit_warmup_done = True

    assert planner.step(_make_observation(num_agents=1)) == {"v": 0.0, "omega": 0.0}


def test_brne_solve_uses_normalized_weighted_sum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalized sample weights preserve, rather than divide, control magnitude."""
    planner = BRNEPlanner({})
    monkeypatch.setattr(planner, "_ensure_brne_loaded", _fake_brne_module)
    monkeypatch.setattr(planner, "_ensure_cov", _fake_covariance)
    monkeypatch.setattr(
        planner,
        "_build_trajectories",
        lambda *_args: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.array([[[0.4, 0.1]], [[0.8, 0.3]]]),
        ),
    )
    monkeypatch.setattr(
        planner,
        "_brne_solve",
        lambda *_args: np.array([[0.25, 0.75]]),
    )
    planner._jit_warmup_done = True

    action = planner.step(_make_observation(num_agents=1))
    assert action == pytest.approx({"v": 0.7, "omega": 0.25})


# --- Integration: real upstream solve ---


@pytest.fixture(scope="module")
def staged_brne_available() -> bool:
    """Provide a boolean indicating staged BRNE availability."""
    if not BRNE_STAGE_PATH.exists():
        return False
    core_rel = "brne_nav/brne_py/brne_py/brne.py"
    if not (BRNE_STAGE_PATH / core_rel).is_file():
        return False
    return _brne_dependency_stack_available()


def test_brne_step_returns_valid_unicycle_action(staged_brne_available: bool) -> None:
    """The BRNE adapter should return a valid unicycle action."""
    if not staged_brne_available:
        pytest.skip("BRNE staged clone or dependency stack not available")
    planner = BRNEPlanner({"num_samples": 49, "maximum_agents": 4})
    obs = _make_observation(num_agents=2)
    action = planner.step(obs)
    assert set(action) == {"v", "omega"}
    assert math.isfinite(action["v"])
    assert math.isfinite(action["omega"])
    assert 0.0 <= action["v"] <= planner.config.v_max
    assert abs(action["omega"]) <= planner.config.omega_max + 1e-9


def test_brne_step_with_no_agents(staged_brne_available: bool) -> None:
    """BRNE should handle the single-robot (no pedestrian) case."""
    if not staged_brne_available:
        pytest.skip("BRNE staged clone or dependency stack not available")
    planner = BRNEPlanner({"num_samples": 49})
    obs = _make_observation(num_agents=1)
    action = planner.step(obs)
    assert set(action) == {"v", "omega"}
    assert math.isfinite(action["v"])
    assert math.isfinite(action["omega"])


def test_brne_step_caps_agents_at_maximum(staged_brne_available: bool) -> None:
    """BRNE should cap the agent count at maximum_agents."""
    if not staged_brne_available:
        pytest.skip("BRNE staged clone or dependency stack not available")
    planner = BRNEPlanner({"num_samples": 49, "maximum_agents": 3})
    obs = _make_observation(num_agents=10)
    action = planner.step(obs)
    assert set(action) == {"v", "omega"}
    assert math.isfinite(action["v"])


def test_brne_step_zero_motion_on_corridor_out_of_bounds(
    staged_brne_available: bool,
) -> None:
    """BRNE should return zero motion when the robot is outside the corridor."""
    if not staged_brne_available:
        pytest.skip("BRNE staged clone or dependency stack not available")
    planner = BRNEPlanner(
        {
            "num_samples": 49,
            "corridor_y_min": -0.1,
            "corridor_y_max": 0.1,
        }
    )
    obs = _make_observation(num_agents=2, robot_pos=[0.0, 5.0], robot_goal=[6.0, 5.0])
    action = planner.step(obs)
    assert action["v"] == 0.0
    assert action["omega"] == 0.0


def test_brne_step_budget_enforcement_returns_zero_motion(
    staged_brne_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BRNE should return zero motion when the solve exceeds the step budget."""
    if not staged_brne_available:
        pytest.skip("BRNE staged clone or dependency stack not available")
    planner = BRNEPlanner({"num_samples": 49, "step_budget_s": 1e-9})
    obs = _make_observation(num_agents=2)
    action = planner.step(obs)
    assert action["v"] == 0.0
    assert action["omega"] == 0.0


def test_brne_fallback_on_error_returns_zero_motion(
    staged_brne_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BRNE should return zero motion on error when fallback_on_error is True."""
    if not staged_brne_available:
        pytest.skip("BRNE staged clone or dependency stack not available")
    planner = BRNEPlanner({"num_samples": 49, "fallback_on_error": True})
    monkeypatch.setattr(
        planner, "_ensure_brne_loaded", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    action = planner.step(_make_observation(num_agents=2))
    assert action["v"] == 0.0
    assert action["omega"] == 0.0


def test_brne_reset_clears_state(staged_brne_available: bool) -> None:
    """BRNE reset should clear cached state."""
    if not staged_brne_available:
        pytest.skip("BRNE staged clone or dependency stack not available")
    planner = BRNEPlanner({"num_samples": 49})
    planner.step(_make_observation(num_agents=2))
    assert planner._lmat is not None
    assert planner._jit_warmup_done is True
    planner.reset()
    assert planner._lmat is None
    assert planner._jit_warmup_done is False
