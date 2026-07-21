"""Net-new unit tests for ``robot_sf/baselines/sicnav.py``.

These tests intentionally complement ``tests/baselines/test_external_mpc_wrappers.py``
rather than duplicate it. The sibling module already covers: the baseline registry
membership, the missing-dependency ``RuntimeError`` on ``step()``, the default
``repo_root`` for ``build_sicnav_config``, the staged-upstream smoke path, the
``SICNavPolicy`` constructor path through a repo root, the shared-observation default
obstacles, and the incompatible-module metadata.

This module covers the remaining behavior branches of ``SICNavPlanner`` and its helpers
that are reachable without the external CasADi/IPOPT/python-RVO2 stack:

* ``SICNavPlannerConfig`` defaults and ``build_sicnav_config`` provenance handling.
* ``_parse_config`` branches (dataclass identity, invalid type).
* ``_resolve_repo_root`` absolute/expanduser resolution.
* ``_is_sicnav_module`` and ``_has_supported_policy_constructor`` static helpers.
* ``reset``/``configure``/``close`` lifecycle hooks and policy/module caching.
* ``load_policy`` factory path and the unsupported-constructor ``RuntimeError``.
* ``use_upstream_campc=False`` skipping the campc probe.
* ``step`` non-dict action contract and per-step policy caching.
* ``_clamp_action`` branches for ``safety_clamp`` off, vx/vy scaling, ``v``/``omega``
  clamping, and every non-finite guard.
* ``_apply_seed`` branches (``None``, ``set_seed`` hook, absent hook).
* ``_CampcEnvAdapter``/``_CampcPolicyRunner`` unit behavior (seed, full-state mapping,
  non-finite guards, omega clamping, time_step<=0, env bookkeeping).
"""

from __future__ import annotations

import configparser
import math
import sys
from pathlib import Path

import pytest

from robot_sf.baselines.sicnav import (
    SICNavPlanner,
    SICNavPlannerConfig,
    _CampcEnvAdapter,
    _CampcPolicyRunner,
    build_sicnav_config,
)


def _make_robot_observation() -> dict[str, object]:
    """Return a minimal robot observation accepted by the SICNav wrapper."""
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


def _write(path: Path, text: str) -> None:
    """Write a fake upstream-package file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def _clean_sicnav_modules():
    """Remove any leaked sicnav-related modules before and after the test."""
    for name in list(sys.modules):
        if (
            name == "sicnav"
            or name.startswith("sicnav.")
            or name == "sicnav_diffusion"
            or name.startswith("sicnav_diffusion.")
        ):
            sys.modules.pop(name, None)
    yield
    for name in list(sys.modules):
        if (
            name == "sicnav"
            or name.startswith("sicnav.")
            or name == "sicnav_diffusion"
            or name.startswith("sicnav_diffusion.")
        ):
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# SICNavPlannerConfig defaults and build_sicnav_config provenance handling
# ---------------------------------------------------------------------------


def test_sicnav_planner_config_defaults() -> None:
    """The dataclass should expose the documented planner defaults."""
    cfg = SICNavPlannerConfig()
    assert cfg.checkpoint_path is None
    assert cfg.repo_root == "third_party/external_repos/sicnav"
    assert cfg.solver == "ipopt"
    assert cfg.device == "cpu"
    assert cfg.mode == "unicycle"
    assert cfg.v_max == 2.0
    assert cfg.omega_max == 1.0
    assert cfg.safety_clamp is True
    assert cfg.action_space == "unicycle"
    assert cfg.fallback_on_error is False
    assert cfg.allow_testing_algorithms is True
    assert cfg.include_in_paper is False
    assert cfg.policy_config_path is None
    assert cfg.env_config_path is None
    assert cfg.use_upstream_campc is True


def test_build_sicnav_config_none_returns_defaults() -> None:
    """``build_sicnav_config(None)`` should yield the documented defaults."""
    cfg = build_sicnav_config(None)
    assert cfg == SICNavPlannerConfig()


def test_build_sicnav_config_drops_unknown_keys_and_keeps_known() -> None:
    """The config builder should keep known fields and silently drop unknown ones."""
    cfg = build_sicnav_config(
        {
            "solver": "acados",
            "device": "cuda",
            "v_max": 1.5,
            "completely_unknown_key": "dropped",
            "another_unknown": 123,
        }
    )
    assert cfg.solver == "acados"
    assert cfg.device == "cuda"
    assert cfg.v_max == 1.5
    # Defaults preserved for untouched fields.
    assert cfg.repo_root == "third_party/external_repos/sicnav"
    assert cfg.omega_max == 1.0


def test_build_sicnav_config_expands_user_in_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """``repo_root`` should be ``~``-expanded while keeping a relative string form."""
    monkeypatch.setenv("HOME", "/home/test_user")
    cfg = build_sicnav_config({"repo_root": "~/repos/sicnav"})
    assert cfg.repo_root == "/home/test_user/repos/sicnav"


# ---------------------------------------------------------------------------
# _parse_config branches
# ---------------------------------------------------------------------------


def test_parse_config_accepts_dataclass_identity() -> None:
    """Passing a ``SICNavPlannerConfig`` directly should return it without copying."""
    cfg = SICNavPlannerConfig(solver="acados", v_max=1.25)
    planner = SICNavPlanner(cfg)
    assert planner.config is cfg


def test_parse_config_rejects_invalid_type() -> None:
    """A non-dict, non-dataclass config should raise ``TypeError``."""
    with pytest.raises(TypeError, match="Invalid config type"):
        SICNavPlanner(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_repo_root branches
# ---------------------------------------------------------------------------


def test_resolve_repo_root_absolute_path_is_returned_resolved(tmp_path: Path) -> None:
    """An absolute repo root should be resolved as-is against the filesystem."""
    planner = SICNavPlanner({"repo_root": str(tmp_path)})
    resolved = planner._resolve_repo_root(str(tmp_path))
    assert resolved == tmp_path.resolve()
    assert resolved.is_absolute()


def test_resolve_repo_root_relative_path_is_repo_root_relative() -> None:
    """A relative repo root should resolve against the Robot SF checkout root."""
    planner = SICNavPlanner({"repo_root": "third_party/external_repos/sicnav"})
    resolved = planner._resolve_repo_root("third_party/external_repos/sicnav")
    expected = (
        Path(__file__).resolve().parents[2] / "third_party" / "external_repos" / "sicnav"
    ).resolve()
    assert resolved == expected


# ---------------------------------------------------------------------------
# _is_sicnav_module and _has_supported_policy_constructor static helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("sicnav", True),
        ("sicnav.policy.campc", True),
        ("sicnav_diffusion", True),
        ("sicnav_diffusion.policy.sicnav_acados", True),
        ("crowd_sim_plus", True),
        ("crowd_sim_plus.envs.utils.state_plus", True),
        ("numpy", False),
        ("sicnav_other", False),
        ("my_sicnav", False),
        ("", False),
    ],
)
def test_is_sicnav_module_classifies_names(name: str, expected: bool) -> None:
    """The module-name classifier should match the SICNav/crowd_sim_plus families only."""
    assert SICNavPlanner._is_sicnav_module(name) is expected


def test_has_supported_policy_constructor_detects_sicnav_policy() -> None:
    """The constructor probe should accept a module exposing ``SICNavPolicy``."""
    import types

    module = types.ModuleType("fake")
    module.SICNavPolicy = lambda **kw: kw  # type: ignore[attr-defined]
    assert SICNavPlanner._has_supported_policy_constructor(module) is True


def test_has_supported_policy_constructor_detects_load_policy() -> None:
    """The constructor probe should accept a module exposing ``load_policy``."""
    import types

    module = types.ModuleType("fake")
    module.load_policy = lambda *a, **kw: None  # type: ignore[attr-defined]
    assert SICNavPlanner._has_supported_policy_constructor(module) is True


def test_has_supported_policy_constructor_rejects_module_without_factory() -> None:
    """The constructor probe should reject a module without a supported factory."""
    import types

    module = types.ModuleType("fake")
    module.other = 1  # type: ignore[attr-defined]
    assert SICNavPlanner._has_supported_policy_constructor(module) is False


def test_has_supported_policy_constructor_rejects_non_callable_attribute() -> None:
    """The constructor probe should reject a module whose factory name is not callable."""
    import types

    module = types.ModuleType("fake")
    module.SICNavPolicy = "not callable"  # type: ignore[attr-defined]
    assert SICNavPlanner._has_supported_policy_constructor(module) is False


# ---------------------------------------------------------------------------
# reset / configure / close lifecycle hooks and policy/module caching
# ---------------------------------------------------------------------------


def test_reset_updates_seed_and_clears_cached_state() -> None:
    """``reset(seed=...)`` should store the seed and drop cached policy/module."""
    planner = SICNavPlanner({}, seed=1)
    planner._policy = object()
    planner._module = object()
    planner.reset(seed=7)
    assert planner._seed == 7
    assert planner._policy is None
    assert planner._module is None


def test_reset_without_seed_preserves_seed() -> None:
    """``reset()`` without a seed should keep the previously configured seed."""
    planner = SICNavPlanner({}, seed=3)
    planner.reset()
    assert planner._seed == 3
    assert planner._policy is None
    assert planner._module is None


def test_configure_replaces_config_and_clears_state() -> None:
    """``configure`` should normalize a new config and drop cached policy/module."""
    planner = SICNavPlanner({"solver": "ipopt"}, seed=1)
    planner._policy = object()
    planner._module = object()
    planner.configure({"solver": "acados", "v_max": 0.5})
    assert planner.config.solver == "acados"
    assert planner.config.v_max == 0.5
    assert planner._policy is None
    assert planner._module is None


def test_configure_accepts_dataclass_directly() -> None:
    """``configure`` should accept a ``SICNavPlannerConfig`` instance."""
    planner = SICNavPlanner({"solver": "ipopt"})
    cfg = SICNavPlannerConfig(solver="acados")
    planner.configure(cfg)
    assert planner.config is cfg


def test_close_releases_cached_resources() -> None:
    """``close`` should drop both the cached policy and the cached module."""
    planner = SICNavPlanner({}, seed=1)
    planner._policy = object()
    planner._module = object()
    planner.close()
    assert planner._policy is None
    assert planner._module is None


def test_import_sicnav_module_caches_result(_clean_sicnav_modules, tmp_path: Path) -> None:
    """``_import_sicnav_module`` should cache and reuse the imported module."""
    repo_root = tmp_path / "sicnav_repo"
    _write(
        repo_root / "sicnav_diffusion" / "__init__.py",
        "VALUE = 123\n",
    )
    _write(repo_root / "sicnav" / "__init__.py", "")
    planner = SICNavPlanner({"repo_root": str(repo_root), "use_upstream_campc": False})
    module = planner._import_sicnav_module()
    assert module.VALUE == 123
    # Second call must return the cached object without re-importing.
    assert planner._module is module
    again = planner._import_sicnav_module()
    assert again is module


# ---------------------------------------------------------------------------
# load_policy factory path and unsupported-constructor RuntimeError
# ---------------------------------------------------------------------------


def test_build_policy_uses_load_policy_factory(_clean_sicnav_modules, tmp_path: Path) -> None:
    """The wrapper should drive a module-level ``load_policy`` factory when present."""
    repo_root = tmp_path / "sicnav_repo"
    _write(
        repo_root / "sicnav_diffusion" / "__init__.py",
        """
def load_policy(checkpoint_path=None, device=None, solver=None):
    class Policy:
        def select_action(self, obs):
            return {"v": 0.7, "omega": -0.2}
    return Policy()
""",
    )
    _write(repo_root / "sicnav" / "__init__.py", "")
    planner = SICNavPlanner(
        {
            "repo_root": str(repo_root),
            "use_upstream_campc": False,
            "checkpoint_path": "ckpt.pt",
            "solver": "ipopt",
            "device": "cpu",
        },
        seed=1,
    )
    action = planner.step(_make_robot_observation())
    assert action == {"v": 0.7, "omega": -0.2}


def test_build_policy_raises_when_no_supported_constructor(
    _clean_sicnav_modules, tmp_path: Path
) -> None:
    """A module without ``SICNavPolicy``/``load_policy`` should raise ``RuntimeError``."""
    repo_root = tmp_path / "sicnav_repo"
    _write(repo_root / "sicnav_diffusion" / "__init__.py", "VALUE = 1\n")
    _write(repo_root / "sicnav" / "__init__.py", "")
    planner = SICNavPlanner({"repo_root": str(repo_root), "use_upstream_campc": False}, seed=1)
    with pytest.raises(RuntimeError, match="does not expose a supported policy constructor"):
        planner.step(_make_robot_observation())


def test_use_upstream_campc_false_skips_campc_probe(
    _clean_sicnav_modules, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``use_upstream_campc=False`` should never call ``_build_campc_runner``."""
    repo_root = tmp_path / "sicnav_repo"
    _write(
        repo_root / "sicnav_diffusion" / "__init__.py",
        """
class SICNavPolicy:
    def __init__(self, **kw):
        pass
    def select_action(self, obs):
        return {"v": 0.1, "omega": 0.0}
""",
    )
    _write(repo_root / "sicnav" / "__init__.py", "")
    planner = SICNavPlanner({"repo_root": str(repo_root), "use_upstream_campc": False}, seed=1)

    def _fail_probe() -> None:
        raise AssertionError("campc probe must not run when use_upstream_campc is False")

    monkeypatch.setattr(planner, "_build_campc_runner", _fail_probe)
    action = planner.step(_make_robot_observation())
    assert action == {"v": 0.1, "omega": 0.0}


# ---------------------------------------------------------------------------
# step contract: non-dict action and per-step policy caching
# ---------------------------------------------------------------------------


def test_step_raises_value_error_on_non_dict_action(_clean_sicnav_modules, tmp_path: Path) -> None:
    """``step`` should reject a non-dict policy payload with ``ValueError``."""
    repo_root = tmp_path / "sicnav_repo"
    _write(
        repo_root / "sicnav_diffusion" / "__init__.py",
        """
class SICNavPolicy:
    def __init__(self, **kw):
        pass
    def select_action(self, obs):
        return [0.0, 0.0]
""",
    )
    _write(repo_root / "sicnav" / "__init__.py", "")
    planner = SICNavPlanner({"repo_root": str(repo_root), "use_upstream_campc": False}, seed=1)
    with pytest.raises(ValueError, match="invalid action payload"):
        planner.step(_make_robot_observation())


def test_step_caches_policy_across_calls(_clean_sicnav_modules, tmp_path: Path) -> None:
    """``step`` should build the policy once and reuse it on subsequent calls."""
    repo_root = tmp_path / "sicnav_repo"
    _write(
        repo_root / "sicnav_diffusion" / "__init__.py",
        """
class SICNavPolicy:
    def __init__(self, **kw):
        self.calls = 0
    def select_action(self, obs):
        self.calls += 1
        return {"v": float(self.calls), "omega": 0.0}
""",
    )
    _write(repo_root / "sicnav" / "__init__.py", "")
    planner = SICNavPlanner({"repo_root": str(repo_root), "use_upstream_campc": False}, seed=1)
    first = planner.step(_make_robot_observation())
    cached_policy = planner._policy
    second = planner.step(_make_robot_observation())
    assert planner._policy is cached_policy
    # The cached policy instance increments; a fresh build would reset calls to 1.
    assert first == {"v": 1.0, "omega": 0.0}
    assert second == {"v": 2.0, "omega": 0.0}


# ---------------------------------------------------------------------------
# _clamp_action branches
# ---------------------------------------------------------------------------


def _make_planner_with_action(action: dict[str, float], **config_kw) -> SICNavPlanner:
    """Build a planner whose cached policy returns a fixed action payload."""
    planner = SICNavPlanner(config_kw or {}, seed=1)

    class _FixedPolicy:
        def select_action(self, obs):
            return dict(action)

    planner._policy = _FixedPolicy()
    return planner


def test_clamp_action_disabled_passes_through_unchanged() -> None:
    """With ``safety_clamp=False`` an over-limit action should be returned unchanged."""
    planner = _make_planner_with_action(
        {"vx": 5.0, "vy": 5.0, "v": 9.0, "omega": 4.0},
        safety_clamp=False,
        v_max=1.0,
        omega_max=0.5,
    )
    action = planner.step(_make_robot_observation())
    assert action == {"vx": 5.0, "vy": 5.0, "v": 9.0, "omega": 4.0}


def test_clamp_action_scales_over_limit_vx_vy() -> None:
    """A vx/vy action exceeding ``v_max`` should be scaled down proportionally."""
    planner = _make_planner_with_action({"vx": 3.0, "vy": 4.0}, v_max=2.0)
    action = planner.step(_make_robot_observation())
    assert action["vx"] == pytest.approx(1.2)
    assert action["vy"] == pytest.approx(1.6)
    assert math.hypot(action["vx"], action["vy"]) == pytest.approx(2.0)


def test_clamp_action_leaves_under_limit_vx_vy() -> None:
    """A vx/vy action within ``v_max`` should pass through unchanged."""
    planner = _make_planner_with_action({"vx": 0.3, "vy": 0.4}, v_max=2.0)
    action = planner.step(_make_robot_observation())
    assert action == {"vx": 0.3, "vy": 0.4}


def test_clamp_action_raises_on_non_finite_vx_vy() -> None:
    """A non-finite vx/vy action should raise ``RuntimeError``."""
    planner = _make_planner_with_action({"vx": float("nan"), "vy": 0.0}, v_max=2.0)
    with pytest.raises(RuntimeError, match="non-finite velocity action"):
        planner.step(_make_robot_observation())


def test_clamp_action_clamps_v_to_v_max() -> None:
    """A unicycle ``v`` above ``v_max`` should be clamped down to ``v_max``."""
    planner = _make_planner_with_action({"v": 5.0, "omega": 0.0}, v_max=2.0)
    action = planner.step(_make_robot_observation())
    assert action["v"] == pytest.approx(2.0)


def test_clamp_action_clamps_negative_v_to_zero() -> None:
    """A negative unicycle ``v`` should be clamped up to zero."""
    planner = _make_planner_with_action({"v": -1.5, "omega": 0.0}, v_max=2.0)
    action = planner.step(_make_robot_observation())
    assert action["v"] == pytest.approx(0.0)


def test_clamp_action_raises_on_non_finite_v() -> None:
    """A non-finite unicycle ``v`` should raise ``RuntimeError``."""
    planner = _make_planner_with_action({"v": float("inf"), "omega": 0.0}, v_max=2.0)
    with pytest.raises(RuntimeError, match="non-finite velocity action"):
        planner.step(_make_robot_observation())


def test_clamp_action_clamps_omega_to_max() -> None:
    """An ``omega`` above ``omega_max`` should be clamped down."""
    planner = _make_planner_with_action({"v": 0.5, "omega": 3.0}, omega_max=1.0)
    action = planner.step(_make_robot_observation())
    assert action["omega"] == pytest.approx(1.0)


def test_clamp_action_clamps_omega_to_neg_max() -> None:
    """An ``omega`` below ``-omega_max`` should be clamped up."""
    planner = _make_planner_with_action({"v": 0.5, "omega": -3.0}, omega_max=1.0)
    action = planner.step(_make_robot_observation())
    assert action["omega"] == pytest.approx(-1.0)


def test_clamp_action_raises_on_non_finite_omega() -> None:
    """A non-finite ``omega`` should raise ``RuntimeError``."""
    planner = _make_planner_with_action({"v": 0.5, "omega": float("nan")}, omega_max=1.0)
    with pytest.raises(RuntimeError, match="non-finite angular action"):
        planner.step(_make_robot_observation())


def test_clamp_action_combined_v_and_omega() -> None:
    """Clamping should apply to ``v`` and ``omega`` independently in one pass."""
    planner = _make_planner_with_action({"v": 10.0, "omega": -10.0}, v_max=1.0, omega_max=0.5)
    action = planner.step(_make_robot_observation())
    assert action["v"] == pytest.approx(1.0)
    assert action["omega"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# _apply_seed branches
# ---------------------------------------------------------------------------


def test_apply_seed_none_is_noop() -> None:
    """A planner with ``seed=None`` should not seed-apply to an arbitrary policy."""
    planner = SICNavPlanner({}, seed=None)

    class _Policy:
        def __init__(self) -> None:
            self.seeded_with = "unset"

        def seed(self, s):
            self.seeded_with = s

    policy = _Policy()
    planner._apply_seed(policy)
    assert policy.seeded_with == "unset"


def test_apply_seed_prefers_seed_method() -> None:
    """``_apply_seed`` should call a ``seed`` hook when present."""
    planner = SICNavPlanner({}, seed=11)

    class _Policy:
        def __init__(self) -> None:
            self.seed_called = False
            self.set_seed_called = False

        def seed(self, s):
            self.seed_called = True
            self.last_seed = s

        def set_seed(self, s):
            self.set_seed_called = True

    policy = _Policy()
    planner._apply_seed(policy)
    assert policy.seed_called is True
    assert policy.last_seed == 11
    # When a ``seed`` hook exists, ``set_seed`` must not also be called.
    assert policy.set_seed_called is False


def test_apply_seed_falls_back_to_set_seed() -> None:
    """``_apply_seed`` should use ``set_seed`` when ``seed`` is absent."""
    planner = SICNavPlanner({}, seed=22)

    class _Policy:
        def __init__(self) -> None:
            self.set_seed_value = None

        def set_seed(self, s):
            self.set_seed_value = s

    policy = _Policy()
    planner._apply_seed(policy)
    assert policy.set_seed_value == 22


def test_apply_seed_without_any_hook_is_safe() -> None:
    """A policy without seed hooks should not break ``_apply_seed``."""
    planner = SICNavPlanner({}, seed=33)

    class _Policy:
        pass

    policy = _Policy()
    # Should not raise.
    planner._apply_seed(policy)


# ---------------------------------------------------------------------------
# _CampcEnvAdapter
# ---------------------------------------------------------------------------


def _make_env_config(
    time_step: float = 0.25, time_limit: float = 25.0
) -> configparser.RawConfigParser:
    """Build a minimal upstream env config with the keys the adapter reads."""
    config = configparser.RawConfigParser()
    config.add_section("env")
    config.set("env", "time_step", str(time_step))
    config.set("env", "time_limit", str(time_limit))
    return config


def test_campc_env_adapter_reads_time_settings() -> None:
    """The adapter should surface ``time_step``/``time_limit`` from the env config."""
    env_config = _make_env_config(time_step=0.4, time_limit=40.0)
    adapter = _CampcEnvAdapter(env_config)
    assert adapter.time_step == pytest.approx(0.4)
    assert adapter.time_limit == pytest.approx(40.0)
    assert adapter.config is env_config
    assert adapter.global_time == 0.0
    assert adapter.sim_env == "circle_crossing"
    assert adapter.human_observability is False


def test_campc_env_adapter_set_human_observability_coerces_to_bool() -> None:
    """``set_human_observability`` should coerce truthy values to ``bool``."""
    adapter = _CampcEnvAdapter(_make_env_config())
    assert adapter.human_observability is False
    adapter.set_human_observability(1)
    assert adapter.human_observability is True
    adapter.set_human_observability(0)
    assert adapter.human_observability is False


# ---------------------------------------------------------------------------
# _CampcPolicyRunner
# ---------------------------------------------------------------------------


class _RecordingFullState:
    """Minimal stand-in for the upstream ``FullState`` capturing its kwargs."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RecordingJointState:
    """Minimal stand-in for the upstream ``FullyObservableJointState``."""

    def __init__(self, self_state=None, human_states=None, static_obs=None):
        self.self_state = self_state
        self.human_states = human_states
        self.static_obs = static_obs


class _StubAction:
    """Upstream ``ActionRot`` stand-in with ``v`` and ``r`` attributes."""

    def __init__(self, v: float, r: float) -> None:
        self.v = v
        self.r = r


def _make_runner(
    action: _StubAction,
    *,
    omega_max: float = 1.0,
    time_step: float = 0.25,
) -> tuple[_CampcPolicyRunner, list]:
    """Build a runner over a policy that returns ``action`` and records predicts."""
    predict_calls: list = []

    class _StubPolicy:
        def predict(self, joint_state):
            predict_calls.append(joint_state)
            return action

    env = _CampcEnvAdapter(_make_env_config(time_step=time_step))
    runner = _CampcPolicyRunner(
        policy=_StubPolicy(),
        env=env,
        full_state_cls=_RecordingFullState,
        joint_state_cls=_RecordingJointState,
        omega_max=omega_max,
    )
    return runner, predict_calls


def test_campc_runner_seed_none_is_noop() -> None:
    """``seed(None)`` on the runner should return without seeding."""
    runner, _ = _make_runner(_StubAction(0.5, 0.0))
    # Should not raise and should not alter RNG state observably.
    runner.seed(None)


def test_campc_runner_seed_seeds_global_rng() -> None:
    """``seed(value)`` on the runner should seed the global RNGs."""
    import numpy as np

    runner, _ = _make_runner(_StubAction(0.5, 0.0))
    runner.seed(99)
    sample_after = np.random.random()
    # Re-seeding deterministically reproduces the same draw.
    runner.seed(99)
    assert np.random.random() == pytest.approx(sample_after)


def test_campc_runner_to_full_state_uses_explicit_goal_and_radius() -> None:
    """``_to_full_state`` should map explicit goal/radius and derive theta from velocity."""
    runner, _ = _make_runner(_StubAction(0.0, 0.0))
    state = runner._to_full_state(
        {
            "position": [1.0, 2.0],
            "velocity": [3.0, 4.0],
            "radius": 0.5,
            "goal": [9.0, 8.0],
        },
        v_pref=1.2,
    )
    assert state.px == 1.0
    assert state.py == 2.0
    assert state.vx == 3.0
    assert state.vy == 4.0
    assert state.radius == 0.5
    assert state.gx == 9.0
    assert state.gy == 8.0
    assert state.v_pref == 1.2
    assert state.theta == pytest.approx(math.atan2(4.0, 3.0))


def test_campc_runner_to_full_state_defaults_missing_fields() -> None:
    """Missing position/velocity/radius should fall back to documented defaults."""
    runner, _ = _make_runner(_StubAction(0.0, 0.0))
    state = runner._to_full_state({}, v_pref=0.8)
    assert state.px == 0.0
    assert state.py == 0.0
    assert state.vx == 0.0
    assert state.vy == 0.0
    assert state.radius == pytest.approx(0.3)
    # No velocity -> theta is zero.
    assert state.theta == 0.0


def test_campc_runner_to_full_state_projects_goal_when_absent() -> None:
    """A missing goal should project via the constant-velocity heuristic."""
    runner, _ = _make_runner(_StubAction(0.0, 0.0))
    state = runner._to_full_state(
        {"position": [1.0, 1.0], "velocity": [0.5, -0.25]},
        v_pref=0.8,
    )
    # gx = px + vx*2.0, gy = py + vy*2.0
    assert state.gx == pytest.approx(1.0 + 0.5 * 2.0)
    assert state.gy == pytest.approx(1.0 + (-0.25) * 2.0)


def test_campc_runner_to_full_state_none_radius_defaults() -> None:
    """An explicit ``None`` radius should fall back to the 0.3 default."""
    runner, _ = _make_runner(_StubAction(0.0, 0.0))
    state = runner._to_full_state({"radius": None}, v_pref=0.8)
    assert state.radius == pytest.approx(0.3)


def test_campc_runner_select_action_converts_action_rot_to_unicycle() -> None:
    """``select_action`` should convert ``ActionRot.r`` (per-step) into rad/s omega."""
    # r is the integrated angular control; with time_step=0.25 it divides back to rad/s.
    runner, predict_calls = _make_runner(_StubAction(v=0.6, r=0.05), omega_max=2.0, time_step=0.25)
    obs = {
        "dt": 0.25,
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [1.0, 0.0],
            "radius": 0.3,
            "v_pref": 1.0,
        },
        "agents": [],
    }
    action = runner.select_action(obs)
    assert set(action) == {"v", "omega"}
    assert action["v"] == pytest.approx(0.6)
    assert action["omega"] == pytest.approx(0.05 / 0.25)
    # The joint state should carry the robot self-state and no humans.
    joint = predict_calls[-1]
    assert joint.static_obs == []
    assert joint.human_states == []
    assert joint.self_state.gx == 1.0


def test_campc_runner_select_action_clamps_omega() -> None:
    """``select_action`` should clamp the derived omega to ``omega_max``."""
    runner, _ = _make_runner(_StubAction(v=0.3, r=10.0), omega_max=0.5, time_step=0.25)
    obs = {
        "dt": 0.25,
        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        "agents": [],
    }
    action = runner.select_action(obs)
    assert action["omega"] == pytest.approx(0.5)


def test_campc_runner_select_action_clamps_negative_omega() -> None:
    """Negative derived omega should clamp to ``-omega_max``."""
    runner, _ = _make_runner(_StubAction(v=0.3, r=-10.0), omega_max=0.5, time_step=0.25)
    obs = {
        "dt": 0.25,
        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        "agents": [],
    }
    action = runner.select_action(obs)
    assert action["omega"] == pytest.approx(-0.5)


def test_campc_runner_select_action_zero_time_step_yields_zero_omega() -> None:
    """A non-positive ``time_step`` should produce omega=0 to avoid division by zero."""
    runner, _ = _make_runner(_StubAction(v=0.4, r=0.1), omega_max=2.0, time_step=0.0)
    obs = {
        "dt": 0.0,
        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        "agents": [],
    }
    action = runner.select_action(obs)
    assert action["v"] == pytest.approx(0.4)
    assert action["omega"] == 0.0


def test_campc_runner_select_action_raises_on_non_finite_v() -> None:
    """A non-finite ``v`` from the policy should raise ``RuntimeError``."""
    runner, _ = _make_runner(_StubAction(v=float("nan"), r=0.0), time_step=0.25)
    obs = {
        "dt": 0.25,
        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        "agents": [],
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        runner.select_action(obs)


def test_campc_runner_select_action_raises_on_non_finite_r() -> None:
    """A non-finite ``r`` from the policy should raise ``RuntimeError``."""
    runner, _ = _make_runner(_StubAction(v=0.1, r=float("inf")), time_step=0.25)
    obs = {
        "dt": 0.25,
        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        "agents": [],
    }
    with pytest.raises(RuntimeError, match="non-finite"):
        runner.select_action(obs)


def test_campc_runner_select_action_advances_env_global_time() -> None:
    """``select_action`` should advance the adapter ``global_time`` by one step."""
    runner, _ = _make_runner(_StubAction(v=0.1, r=0.0), time_step=0.25)
    obs = {
        "dt": 0.25,
        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        "agents": [],
    }
    assert runner._env.global_time == 0.0
    runner.select_action(obs)
    assert runner._env.global_time == pytest.approx(0.25)
    runner.select_action(obs)
    assert runner._env.global_time == pytest.approx(0.5)


def test_campc_runner_select_action_maps_pedestrian_agents() -> None:
    """``select_action`` should convert each agent into a human full state."""
    runner, predict_calls = _make_runner(_StubAction(v=0.1, r=0.0), time_step=0.25)
    obs = {
        "dt": 0.25,
        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        "agents": [
            {"position": [2.0, 0.0], "velocity": [0.0, 1.0], "radius": 0.35},
        ],
    }
    runner.select_action(obs)
    joint = predict_calls[-1]
    assert len(joint.human_states) == 1
    human = joint.human_states[0]
    assert human.px == 2.0
    assert human.vy == 1.0
    assert human.radius == 0.35
    # Pedestrians are mapped with the runner-fixed v_pref of 0.8.
    assert human.v_pref == 0.8


def test_campc_runner_select_action_accepts_observation_dataclass() -> None:
    """``select_action`` should accept the ``Observation`` dataclass, not only dicts."""
    from robot_sf.baselines.interface import Observation

    runner, predict_calls = _make_runner(_StubAction(v=0.2, r=0.0), time_step=0.25)
    obs = Observation(
        dt=0.25,
        robot={"position": [0.0, 0.0], "velocity": [0.0, 0.0], "goal": [1.0, 0.0]},
        agents=[],
    )
    action = runner.select_action(obs)
    assert action["v"] == pytest.approx(0.2)
    assert predict_calls  # the policy was actually invoked


# ---------------------------------------------------------------------------
# get_metadata config_hash determinism and payload
# ---------------------------------------------------------------------------


def test_get_metadata_config_hash_is_deterministic() -> None:
    """Identical configs should produce identical ``config_hash`` values."""
    planner_a = SICNavPlanner({"solver": "ipopt", "v_max": 1.5}, seed=1)
    planner_b = SICNavPlanner({"solver": "ipopt", "v_max": 1.5}, seed=2)
    meta_a = planner_a.get_metadata()
    meta_b = planner_b.get_metadata()
    assert meta_a["config_hash"] == meta_b["config_hash"]
    assert meta_a["config_hash"]  # non-empty


def test_get_metadata_config_hash_differs_for_different_config() -> None:
    """Different configs should produce different ``config_hash`` values."""
    planner_a = SICNavPlanner({"solver": "ipopt"}, seed=1)
    planner_b = SICNavPlanner({"solver": "acados"}, seed=1)
    assert planner_a.get_metadata()["config_hash"] != planner_b.get_metadata()["config_hash"]


def test_get_metadata_payload_shape() -> None:
    """``get_metadata`` should expose the algorithm name and full config dict."""
    planner = SICNavPlanner({"solver": "ipopt", "v_max": 1.25}, seed=1)
    meta = planner.get_metadata()
    assert meta["algorithm"] == "sicnav"
    assert isinstance(meta["config"], dict)
    assert meta["config"]["solver"] == "ipopt"
    assert meta["config"]["v_max"] == 1.25
    assert "config_hash" in meta
    assert "status" in meta
