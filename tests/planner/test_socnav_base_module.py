"""Focused coverage for the extracted SocNav base/config module."""

from types import SimpleNamespace

import numpy as np

from robot_sf.planner import socnav
from robot_sf.planner import socnav_base as base

_BASE_NAMES = (
    "SamplingPlannerAdapter",
    "SocNavBenchComplexPolicy",
    "SocNavPlannerConfig",
    "SocNavPlannerPolicy",
    "TrivialReferencePlannerAdapter",
)


def _observation(*, goal: tuple[float, float] = (4.0, 0.0), pedestrians=None) -> dict:
    """Build a compact nested SocNav observation for adapter unit tests."""
    positions = np.asarray(pedestrians if pedestrians is not None else [], dtype=float)
    if positions.size == 0:
        positions = np.zeros((0, 2), dtype=float)
    return {
        "robot": {
            "position": np.array([0.0, 0.0]),
            "heading": np.array([0.0]),
            "radius": np.array([0.3]),
        },
        "goal": {"current": np.asarray(goal, dtype=float)},
        "pedestrians": {
            "positions": positions,
            "velocities": np.zeros_like(positions),
            "count": np.array([positions.shape[0]]),
            "radius": np.array([0.3]),
        },
        "sim": {"timestep": np.array([0.1])},
    }


def test_facade_reexports_base_names_with_identity() -> None:
    """All five base/config names re-export from the facade as the same objects."""
    for name in _BASE_NAMES:
        assert getattr(socnav, name) is getattr(base, name)
        assert name in dir(socnav)
        assert name in socnav.__all__


def test_config_default_construction() -> None:
    """Default config construction preserves documented defaults and class identity."""
    config = base.SocNavPlannerConfig()
    assert config.max_linear_speed == 3.0
    assert config.max_angular_speed == 1.0
    assert config.goal_tolerance == 0.25
    assert config.sacadrl_model_id == "ga3c_cadrl_iros18"
    assert config.predictive_model_id == "predictive_proxy_selected_v1"
    assert config.forecast_variant == "none"
    # The facade alias is the same class, so instances are interchangeable.
    assert isinstance(socnav.SocNavPlannerConfig(), base.SocNavPlannerConfig)


def test_policy_wraps_adapter() -> None:
    """The policy wrapper defaults to a sampling adapter and forwards act()."""
    policy = base.SocNavPlannerPolicy()
    assert isinstance(policy.adapter, base.SamplingPlannerAdapter)
    action = policy.act(_observation())
    assert isinstance(action, tuple)
    assert len(action) == 2

    adapter = base.SamplingPlannerAdapter()
    wrapped = base.SocNavPlannerPolicy(adapter=adapter)
    assert wrapped.adapter is adapter


def test_trivial_and_sampling_adapter_instantiation() -> None:
    """Reference and sampling adapters instantiate and emit bounded commands."""
    trivial = base.TrivialReferencePlannerAdapter()
    assert isinstance(trivial.config, base.SocNavPlannerConfig)
    linear, angular = trivial.plan(_observation())
    assert -trivial.config.max_linear_speed <= linear <= trivial.config.max_linear_speed
    assert -trivial.config.max_angular_speed <= angular <= trivial.config.max_angular_speed
    assert trivial.plan(_observation(goal=(0.1, 0.0))) == (0.0, 0.0)
    assert trivial.diagnostics()["adapter"] == "TrivialReferencePlannerAdapter"

    sampling = base.SamplingPlannerAdapter()
    assert isinstance(sampling.config, base.SocNavPlannerConfig)
    linear_s, angular_s = sampling.plan(_observation())
    assert -sampling.config.max_linear_speed <= linear_s <= sampling.config.max_linear_speed
    assert -sampling.config.max_angular_speed <= angular_s <= sampling.config.max_angular_speed


def test_base_adapter_lifecycle_and_goal_distance_objective() -> None:
    """Moved lifecycle hooks and goal-distance objective preserve their contracts."""
    adapter = base.TrivialReferencePlannerAdapter()
    adapter.plan(_observation())
    assert adapter.diagnostics()["steps"] == 1
    adapter.reset(seed=7)
    assert adapter.diagnostics()["steps"] == 0

    config = base.SocNavPlannerConfig(max_linear_speed=1.5)
    adapter.configure(config)
    assert adapter.config is config

    class _Trajectory:
        def __init__(self, positions: np.ndarray, valid_horizons: np.ndarray | None = None) -> None:
            self._positions = positions
            self.valid_horizons_n1 = valid_horizons

        def position_nk2(self) -> np.ndarray:
            return self._positions

    objective = base.SamplingPlannerAdapter._GoalDistanceObjective()
    assert objective.evaluate_function(_Trajectory(np.empty((0, 0, 2)))).size == 0

    objective.set_goal(np.array([1.0, 0.0]))
    positions = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [2.0, 0.0]],
        ]
    )
    np.testing.assert_allclose(objective.evaluate_function(_Trajectory(positions)), [0.0, 1.0])
    np.testing.assert_allclose(
        objective.evaluate_function(_Trajectory(positions, np.array([[1], [2]]))),
        [1.0, 1.0],
    )


def test_sampling_adapter_repulsion_and_upstream_root_fallbacks(tmp_path, monkeypatch) -> None:
    """Moved fallback helpers retain pedestrian avoidance and fail-closed root handling."""
    adapter = base.SamplingPlannerAdapter()
    linear, angular = adapter.plan(_observation(pedestrians=[(0.0, 1.0)]))
    assert linear > 0.0
    assert angular < 0.0

    missing_root = tmp_path / "missing-socnavbench"
    assert adapter._load_upstream_planner(missing_root) is None

    untrusted_root = tmp_path / "untrusted-socnavbench"
    untrusted_root.mkdir()
    monkeypatch.delenv("ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT", raising=False)
    assert adapter._load_upstream_planner(untrusted_root) is None

    monkeypatch.setenv("ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT", "yes")
    assert base.SamplingPlannerAdapter._allow_untrusted_socnav_root()


def test_sampling_adapter_timestep_helpers_preserve_defaults_and_params() -> None:
    """The extracted upstream-timestep helpers retain default and configured values."""
    adapter = base.SamplingPlannerAdapter()
    assert adapter._resolve_robot_dt(object()) == 0.1
    assert adapter._resolve_camera_dt(object()) == 0.1

    params = SimpleNamespace(
        robot_dynamics_params=SimpleNamespace(dt=0.25),
        camera_params=SimpleNamespace(dt=0.05),
    )
    assert adapter._resolve_robot_dt(params) == 0.25
    assert adapter._resolve_camera_dt(params) == 0.05


def test_complex_policy_uses_deferred_facade_adapter() -> None:
    """SocNavBenchComplexPolicy resolves the facade adapter via the deferred import."""
    policy = base.SocNavBenchComplexPolicy(allow_fallback=True)
    assert isinstance(policy, base.SocNavPlannerPolicy)
    assert isinstance(policy.adapter, socnav.SocNavBenchSamplingAdapter)
