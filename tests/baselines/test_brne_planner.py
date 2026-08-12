"""Tests for the BRNE baseline planner adapter (issue #5318).

Exercises the bounded BRNE integration tier: corridor-class scenarios only,
fail-closed budget enforcement, native unicycle output.  Tests that need the
staged BRNE clone skip cleanly when it is absent (CI default).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.baselines import brne as brne_module
from robot_sf.baselines import get_baseline, list_baselines
from robot_sf.baselines.brne import BRNEPlanner, BRNEPlannerConfig, build_brne_config
from robot_sf.baselines.interface import Observation

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
    assert meta["source_commit"] == brne_module.BRNE_PINNED_SHA
    assert meta["source_pin"] == brne_module.BRNE_PINNED_SHA
    assert meta["source_integrity"] == "clean_pinned_worktree"


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
    metadata = planner.get_metadata()
    assert metadata["runtime_status"] == "failed"
    assert metadata["failure_reasons"] == ["nonfinite_weights"]


def test_brne_solve_uses_plan_step_first_mean_normalized_weighted_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mean-normalized BRNE weights are aggregated by sample-weighted mean."""
    planner = BRNEPlanner({})
    plan_step_first_ensemble = np.array(
        [
            [[0.1, -0.2], [0.4, 0.0], [0.8, 0.3]],
            [[1.0, 0.5], [0.2, -0.4], [0.0, 0.2]],
        ]
    )
    mean_normalized_weights = np.array([[0.5, 1.0, 1.5]])
    expected_first_step_command = np.array([0.55, 0.1166666667])
    raw_weighted_sum = expected_first_step_command * mean_normalized_weights.size

    assert mean_normalized_weights.mean() == pytest.approx(1.0)
    assert mean_normalized_weights.sum() == pytest.approx(mean_normalized_weights.size)
    assert expected_first_step_command[0] <= planner.config.v_max
    assert abs(expected_first_step_command[1]) <= planner.config.omega_max

    monkeypatch.setattr(planner, "_ensure_brne_loaded", _fake_brne_module)
    monkeypatch.setattr(planner, "_ensure_cov", _fake_covariance)
    monkeypatch.setattr(
        planner,
        "_build_trajectories",
        lambda *_args: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            plan_step_first_ensemble,
        ),
    )
    monkeypatch.setattr(
        planner,
        "_brne_solve",
        lambda *_args: mean_normalized_weights,
    )
    planner._jit_warmup_done = True

    action = planner.step(_make_observation(num_agents=1))
    assert action == pytest.approx(
        {"v": expected_first_step_command[0], "omega": expected_first_step_command[1]}
    )
    assert action != pytest.approx({"v": raw_weighted_sum[0], "omega": raw_weighted_sum[1]})

    mechanism_step = planner.get_metadata()["mechanism_trace"]["steps"][0]
    assert mechanism_step["ensemble"]["control_ensemble_shape"] == [2, 3, 2]
    assert mechanism_step["ensemble"]["weight_shape"] == [1, 3]
    assert mechanism_step["ensemble"]["aggregation_mode"] == "plan_step_first"
    assert mechanism_step["ensemble"]["aggregation_formula"] == "mean_plan_step_first_over_samples"
    candidate_distribution = mechanism_step["ensemble"]["candidate_distribution"]
    assert candidate_distribution["status"] == "available"
    assert candidate_distribution["sample_count"] == 3
    assert candidate_distribution["plan_step_count"] == 2
    assert candidate_distribution["first"]["weighted_mean"] == pytest.approx(
        {"v_m_s": expected_first_step_command[0], "omega_rad_s": expected_first_step_command[1]}
    )
    assert candidate_distribution["first_to_second"]["weighted_mean_delta_v_m_s"] == pytest.approx(
        -0.3166666667
    )
    assert mechanism_step["pre_clamp_action"] == pytest.approx(
        {"v_m_s": expected_first_step_command[0], "omega_rad_s": expected_first_step_command[1]}
    )
    assert mechanism_step["selected_action"] == pytest.approx(
        {"v_m_s": expected_first_step_command[0], "omega_rad_s": expected_first_step_command[1]}
    )
    assert mechanism_step["action_clipping"] == {
        "v_clipped": False,
        "omega_clipped": False,
        "any_clipped": False,
    }


def test_brne_solve_uses_samples_first_mean_normalized_weighted_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy samples-first adapters use the same upstream weighted mean."""
    planner = BRNEPlanner({})
    samples_first_ensemble = np.array(
        [
            [[0.1, -0.2], [1.0, 0.5]],
            [[0.4, 0.0], [0.2, -0.4]],
            [[0.8, 0.3], [0.0, 0.2]],
        ]
    )
    mean_normalized_weights = np.array([[0.5, 1.0, 1.5]])
    expected_first_step_command = np.array([0.55, 0.1166666667])
    raw_weighted_sum = expected_first_step_command * mean_normalized_weights.size

    monkeypatch.setattr(planner, "_ensure_brne_loaded", _fake_brne_module)
    monkeypatch.setattr(planner, "_ensure_cov", _fake_covariance)
    monkeypatch.setattr(
        planner,
        "_build_trajectories",
        lambda *_args: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            samples_first_ensemble,
        ),
    )
    monkeypatch.setattr(
        planner,
        "_brne_solve",
        lambda *_args: mean_normalized_weights,
    )
    planner._jit_warmup_done = True

    action = planner.step(_make_observation(num_agents=1))
    assert action == pytest.approx(
        {"v": expected_first_step_command[0], "omega": expected_first_step_command[1]}
    )
    assert action != pytest.approx({"v": raw_weighted_sum[0], "omega": raw_weighted_sum[1]})

    mechanism_step = planner.get_metadata()["mechanism_trace"]["steps"][0]
    assert mechanism_step["ensemble"]["control_ensemble_shape"] == [3, 2, 2]
    assert mechanism_step["ensemble"]["weight_shape"] == [1, 3]
    assert mechanism_step["ensemble"]["aggregation_mode"] == "samples_first"
    assert mechanism_step["ensemble"]["aggregation_formula"] == "mean_samples_first_over_samples"


def test_brne_mechanism_trace_separates_pre_clamp_and_selected_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Safety clipping is observable without confusing it with the weighted command."""
    planner = BRNEPlanner({"v_max": 0.5, "omega_max": 0.1})
    monkeypatch.setattr(planner, "_ensure_brne_loaded", _fake_brne_module)
    monkeypatch.setattr(planner, "_ensure_cov", _fake_covariance)
    monkeypatch.setattr(
        planner,
        "_build_trajectories",
        lambda *_args: (
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.array([[[1.0, -2.0]]]),
        ),
    )
    monkeypatch.setattr(planner, "_brne_solve", lambda *_args: np.array([[1.0]]))
    planner._jit_warmup_done = True

    assert planner.step(_make_observation(num_agents=1)) == {"v": 0.5, "omega": -0.1}
    step = planner.get_metadata()["mechanism_trace"]["steps"][0]
    assert step["pre_clamp_action"] == {"v_m_s": 1.0, "omega_rad_s": -2.0}
    assert step["selected_action"] == {"v_m_s": 0.5, "omega_rad_s": -0.1}
    assert step["action_clipping"] == {
        "v_clipped": True,
        "omega_clipped": True,
        "any_clipped": True,
    }


def test_brne_declared_heading_takes_precedence_over_velocity() -> None:
    """Stationary or stale-velocity observations retain their declared heading."""
    pose = BRNEPlanner._infer_robot_pose(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([4.0, 0.0]),
        r_heading=math.pi / 2.0,
    )
    assert pose[2] == pytest.approx(math.pi / 2.0)


def test_brne_normalizes_samples_first_control_ensemble() -> None:
    """Legacy samples-first control arrays are normalized before trajectory use."""
    samples_first = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    normalized = BRNEPlanner._normalize_control_ensemble(samples_first, plan_steps=4)
    assert normalized.shape == (4, 3, 2)
    np.testing.assert_array_equal(normalized[0], samples_first[:, 0, :])


def test_brne_rejects_malformed_control_ensemble() -> None:
    """Malformed or ambiguous control tensors fail closed at the adapter boundary."""
    with pytest.raises(ValueError, match="invalid_control_ensemble_shape"):
        BRNEPlanner._normalize_control_ensemble(np.zeros((4, 3)), plan_steps=4)


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
    mechanism_trace = planner.get_metadata()["mechanism_trace"]
    assert mechanism_trace["schema_version"] == "brne-mechanism-trace.v1"
    assert mechanism_trace["status"] == "available"
    assert len(mechanism_trace["steps"]) == 1


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


def test_brne_solver_error_without_fallback_is_recorded_and_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native diagnostic path must surface solver failures when fallback is disabled."""
    planner = BRNEPlanner({"fallback_on_error": False})
    monkeypatch.setattr(
        planner,
        "_ensure_brne_loaded",
        lambda: (_ for _ in ()).throw(RuntimeError("solver boom")),
    )

    with pytest.raises(RuntimeError, match="BRNE solve failed: solver boom"):
        planner.step(_make_observation())

    metadata = planner.get_metadata()
    assert metadata["runtime_status"] == "failed"
    assert metadata["failure_reasons"] == ["solver_exception"]


def test_brne_step_accepts_canonical_observation_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The planner accepts both mapping and canonical Observation inputs."""
    planner = BRNEPlanner({})
    seen: list[Observation] = []

    def _solve(observation: Observation) -> dict[str, float]:
        seen.append(observation)
        return {"v": 0.0, "omega": 0.0}

    monkeypatch.setattr(planner, "_solve", _solve)
    observation = Observation(
        dt=0.1,
        robot={"position": [0.0, 0.0], "goal": [1.0, 0.0]},
        agents=[],
    )
    assert planner.step(observation) == {"v": 0.0, "omega": 0.0}
    assert seen == [observation]


def test_brne_seed_binding_is_cached_and_resettable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The upstream sampler receives the episode seed once per planner reset."""
    module = SimpleNamespace(rng=object())
    load_calls: list[Path] = []

    def _load(stage: Path) -> object:
        load_calls.append(stage)
        return module

    monkeypatch.setattr(brne_module, "_load_brne_module", _load)
    planner = BRNEPlanner({"stage_path": str(tmp_path / "brne")}, seed=12)

    assert planner._ensure_brne_loaded() is module
    assert isinstance(module.rng, np.random.Generator)
    assert planner._ensure_brne_loaded() is module
    assert len(load_calls) == 1

    planner.reset(seed=13)
    planner._ensure_brne_loaded()
    assert isinstance(module.rng, np.random.Generator)
    assert len(load_calls) == 1


def test_brne_loader_without_upstream_rng_is_still_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adapters without an upstream RNG attribute remain loadable."""
    module = SimpleNamespace()
    monkeypatch.setattr(brne_module, "_load_brne_module", lambda _stage: module)
    planner = BRNEPlanner({"stage_path": str(tmp_path / "brne")}, seed=12)

    assert planner._ensure_brne_loaded() is module
    assert planner._upstream_rng_seeded is True


def test_brne_covariance_and_runtime_state_caches_are_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Covariance caching and runtime diagnostics reset at episode/config boundaries."""
    calls = 0

    def _get_lmat(*_args: object) -> tuple[np.ndarray, None]:
        nonlocal calls
        calls += 1
        return np.eye(2), None

    planner = BRNEPlanner({"stage_path": str(tmp_path / "brne")})
    module = SimpleNamespace(get_Lmat_nb=_get_lmat)
    assert planner._ensure_cov(module).shape == (2, 2)
    assert planner._ensure_cov(module).shape == (2, 2)
    assert calls == 1

    planner._record_failure("duplicate")
    planner._record_failure("duplicate")
    assert planner.get_metadata()["failure_reasons"] == ["duplicate"]
    planner.reset(seed=23)
    assert planner.get_metadata()["runtime_status"] == "not_started"
    assert planner.get_metadata()["seed"] == 23
    planner.configure({"num_samples": 17})
    assert planner.get_metadata()["runtime_status"] == "not_started"
    planner.close()
    assert planner._brne is None


def test_brne_build_trajectories_uses_effective_control_sample_count() -> None:
    """Trajectory tensors follow the upstream control ensemble's effective count."""
    planner = BRNEPlanner({"num_samples": 49, "plan_steps": 4})
    plan_steps = 4
    effective_samples = 3

    def _get_ulist(*_args: object) -> np.ndarray:
        return np.zeros((plan_steps, effective_samples, 2))

    def _traj_sim(st0: np.ndarray, ulist: np.ndarray, _dt: float) -> np.ndarray:
        assert st0.shape == (3, effective_samples)
        assert ulist.shape == (plan_steps, effective_samples, 2)
        return np.zeros((plan_steps, 3, effective_samples))

    def _sample_normal(num_samples: int, steps: int, _lmat: np.ndarray) -> np.ndarray:
        return np.zeros((num_samples, steps))

    module = SimpleNamespace(
        get_ulist_essemble=_get_ulist,
        traj_sim_essemble=_traj_sim,
        mvn_sample_normal=_sample_normal,
    )
    xtraj, ytraj, ulist = planner._build_trajectories(
        module,
        np.eye(1),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.4, 0.0]),
        np.array([2.0, 0.0]),
        [{"position": [1.0, 0.5], "velocity": [0.0, 0.0]}],
        [(1.1, 0)],
        2,
        49,
        plan_steps,
        0.1,
    )

    assert xtraj.shape == (2 * effective_samples, plan_steps)
    assert ytraj.shape == (2 * effective_samples, plan_steps)
    assert ulist.shape == (plan_steps, effective_samples, 2)


def test_brne_metadata_marks_invalid_and_valid_source_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Metadata exposes both invalid and accepted source-integrity states."""
    core = tmp_path / brne_module.BRNE_CORE_REL
    core.parent.mkdir(parents=True)
    core.write_text("# fixture\n", encoding="utf-8")

    def _invalid(_path: Path) -> Path:
        raise RuntimeError("fixture provenance mismatch")

    monkeypatch.setattr(brne_module, "_validate_stage_provenance", _invalid)
    monkeypatch.setattr(
        brne_module.subprocess,
        "run",
        lambda *_args, **_kwargs: brne_module.subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="deadbeef\n", stderr=""
        ),
    )
    planner = BRNEPlanner({"stage_path": str(tmp_path)})
    invalid = planner.get_metadata()
    assert invalid["status"] == "invalid_provenance"
    assert invalid["source_commit"] == "deadbeef"
    assert invalid["source_integrity"] == "invalid"

    monkeypatch.setattr(
        brne_module,
        "_validate_stage_provenance",
        lambda path: path / brne_module.BRNE_CORE_REL,
    )
    valid = planner.get_metadata()
    assert valid["status"] == "ok"
    assert valid["source_commit"] == brne_module.BRNE_PINNED_SHA
    assert valid["source_integrity"] == "clean_pinned_worktree"


def test_brne_module_loader_rejects_missing_import_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed staged source must fail before attempting an import."""
    core = tmp_path / "brne_nav" / "brne_py" / "brne_py"
    core.mkdir(parents=True)
    (core / "brne.py").write_text("# fixture\n", encoding="utf-8")
    monkeypatch.delitem(brne_module.sys.modules, brne_module._BRNE_MODULE_NAME, raising=False)
    monkeypatch.setattr(
        brne_module, "_validate_stage_provenance", lambda path: path / brne_module.BRNE_CORE_REL
    )
    monkeypatch.setattr(brne_module.importlib.util, "spec_from_file_location", lambda *_args: None)

    with pytest.raises(ImportError, match="Could not build import spec"):
        brne_module._load_brne_module(tmp_path)


def test_brne_stage_provenance_rejects_wrong_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A staged core at another git commit is unavailable to the diagnostic."""
    core = tmp_path / brne_module.BRNE_CORE_REL
    core.parent.mkdir(parents=True)
    core.write_text("# fixture\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()

    def _git(*_args: object, **_kwargs: object) -> object:
        return brne_module.subprocess.CompletedProcess(
            args=["git"], returncode=0, stdout="deadbeef\n", stderr=""
        )

    monkeypatch.setattr(brne_module.subprocess, "run", _git)
    with pytest.raises(RuntimeError, match="commit mismatch"):
        brne_module._validate_stage_provenance(tmp_path)


def test_brne_pose_falls_back_to_goal_or_zero_heading() -> None:
    """Legacy observations without motion use goal bearing, then zero heading."""
    zero_velocity = np.zeros(2)
    pose_to_goal = BRNEPlanner._infer_robot_pose(
        np.array([0.0, 0.0]), zero_velocity, np.array([0.0, 2.0])
    )
    pose_without_goal = BRNEPlanner._infer_robot_pose(
        np.array([0.0, 0.0]), zero_velocity, np.array([0.0, 0.0])
    )
    assert pose_to_goal[2] == pytest.approx(math.pi / 2.0)
    assert pose_without_goal[2] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("ensemble", "plan_steps", "error"),
    [
        (np.zeros((3, 5, 2)), 4, "invalid_control_ensemble_plan_axis"),
        (np.zeros((4, 0, 2)), 4, "invalid_control_ensemble_sample_axis"),
        (np.full((4, 2, 2), np.nan), 4, "nonfinite_control_ensemble"),
    ],
)
def test_brne_rejects_ambiguous_or_nonfinite_control_ensembles(
    ensemble: np.ndarray, plan_steps: int, error: str
) -> None:
    """Only finite tensors with an explicit plan-step axis enter the solver."""
    with pytest.raises(ValueError, match=error):
        BRNEPlanner._normalize_control_ensemble(ensemble, plan_steps=plan_steps)


@pytest.mark.parametrize(
    ("ensemble", "weights", "reason"),
    [
        (np.zeros((2, 1)), np.ones((1, 2)), "invalid_control_ensemble_shape"),
        (np.zeros((2, 2, 2)), np.ones((1, 2, 1)), "invalid_control_ensemble_shape"),
        (np.zeros((2, 2, 2)), np.ones((1, 3)), "sample_weight_shape_mismatch"),
        (np.full((2, 2, 2), np.nan), np.ones((1, 2)), "nonfinite_control_command"),
    ],
)
def test_brne_solver_records_control_shape_and_finiteness_failures(
    ensemble: np.ndarray,
    weights: np.ndarray,
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed upstream outputs become classified stop actions."""
    planner = BRNEPlanner({})

    def _ensure_brne_loaded() -> object:
        return object()

    def _ensure_cov(_brne: object) -> np.ndarray:
        return np.eye(1)

    monkeypatch.setattr(planner, "_ensure_brne_loaded", _ensure_brne_loaded)
    monkeypatch.setattr(planner, "_ensure_cov", _ensure_cov)
    monkeypatch.setattr(
        planner,
        "_build_trajectories",
        lambda *_args: (np.zeros((1, 2)), np.zeros((1, 2)), ensemble),
    )
    monkeypatch.setattr(planner, "_brne_solve", lambda *_args: weights)
    planner._jit_warmup_done = True

    observation = _make_observation()
    observation["robot"]["heading"] = (
        "invalid" if reason != "nonfinite_control_command" else math.nan
    )

    action = planner.step(observation)
    assert action == {"v": 0.0, "omega": 0.0}
    assert planner.get_metadata()["failure_reasons"] == [reason]


def test_brne_clamp_can_be_disabled() -> None:
    """The diagnostic wrapper preserves the configured unclamped action path."""
    planner = BRNEPlanner({"safety_clamp": False})
    action = {"v": 4.0, "omega": 3.0}
    planner._clamp_action(action)
    assert action == {"v": 4.0, "omega": 3.0}


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
