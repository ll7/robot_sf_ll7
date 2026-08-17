"""TODO docstring. Document this module."""

from unittest.mock import Mock

import numpy as np
import pytest
from pysocialforce import Simulator

from robot_sf.sim import fast_pysf_wrapper as wrapper_module
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


def make_simple_sim():
    # Create a minimal simulator with two pedestrians and a small obstacle
    # state shape for 2 peds: [x, y, vx, vy, goalx, goaly, tau]
    """TODO docstring. Document this function."""
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0],
        ],
    )
    obstacles = [
        (2.5, 2.5, -1.0, 1.0),  # vertical line near x=2.5 from y=-1 to y=1
    ]
    sim = Simulator(state=state, obstacles=obstacles)
    return sim


def make_no_obstacle_sim():
    """Create a simulator with pedestrians but no obstacles."""
    state = np.array(
        [
            [0.0, 0.0, 0.1, 0.0, 5.0, 0.0, 1.0],
            [1.0, 0.25, -0.1, 0.0, 5.0, 0.0, 1.0],
        ],
    )
    return Simulator(state=state, obstacles=[])


def make_no_pedestrian_sim():
    """Create a simulator with obstacles but no pedestrians."""
    return Simulator(
        state=np.empty((0, 7), dtype=float),
        obstacles=[(2.0, 2.0, -1.0, 1.0)],
    )


def test_get_forces_at_point():
    """TODO docstring. Document this function."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    f = wrapper.get_forces_at([0.5, 0.0])
    assert f.shape == (2,)
    # force should be finite numbers
    assert np.all(np.isfinite(f))


def test_get_forces_at_points_matches_pointwise_force_queries():
    """Batched force sampling should preserve point-by-point semantics."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    points = np.array([[0.5, 0.0], [1.5, 0.25], [2.0, -0.5]], dtype=float)
    batched = wrapper.get_forces_at_points(points, include_desired=True, desired_goal=[5.0, 0.0])
    pointwise = np.vstack(
        [
            wrapper.get_forces_at(point, include_desired=True, desired_goal=[5.0, 0.0])
            for point in points
        ],
    )

    assert batched.shape == (len(points), 2)
    assert batched == pytest.approx(pointwise)


def test_get_forces_at_points_uses_batched_path_for_social_obstacle_and_desired(monkeypatch):
    """Batched force sampling should not call pointwise queries for safe kwargs."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    points = np.array([[0.5, 0.0], [1.5, 0.25], [2.0, -0.5]], dtype=float)
    kwargs = {"include_desired": True, "desired_goal": [5.0, 0.0]}
    pointwise = np.vstack([wrapper.get_forces_at(point, **kwargs) for point in points])

    def fail_pointwise(*_args, **_kwargs):
        raise AssertionError("get_forces_at_points should use the batched force path")

    monkeypatch.setattr(wrapper, "get_forces_at", fail_pointwise)

    batched = wrapper.get_forces_at_points(points, **kwargs)

    assert batched.shape == (len(points), 2)
    assert batched == pytest.approx(pointwise)


def test_get_forces_at_points_empty_returns_empty_force_rows():
    """Empty point batches should keep the force vector axis."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    forces = wrapper.get_forces_at_points(np.empty((0, 2), dtype=float))

    assert forces.shape == (0, 2)
    assert forces.dtype == float


def test_get_forces_at_points_matches_pointwise_without_obstacles():
    """Batched force sampling should handle simulations with no obstacles."""
    sim = make_no_obstacle_sim()
    wrapper = FastPysfWrapper(sim)

    points = np.array([[-0.25, 0.0], [0.5, 0.25], [2.0, -0.5]], dtype=float)
    batched = wrapper.get_forces_at_points(points)
    pointwise = np.vstack([wrapper.get_forces_at(point) for point in points])

    assert batched.shape == (len(points), 2)
    assert batched == pytest.approx(pointwise)


def test_get_forces_at_points_matches_pointwise_without_pedestrians():
    """Batched force sampling should handle obstacle-only simulations."""
    sim = make_no_pedestrian_sim()
    wrapper = FastPysfWrapper(sim)

    points = np.array([[0.5, 0.0], [1.5, 0.25], [2.0, -0.5]], dtype=float)
    batched = wrapper.get_forces_at_points(points, include_desired=True, desired_goal=[5.0, 0.0])
    pointwise = np.vstack(
        [
            wrapper.get_forces_at(point, include_desired=True, desired_goal=[5.0, 0.0])
            for point in points
        ],
    )
    obstacle_only = wrapper.get_forces_at_points(points)

    assert batched.shape == (len(points), 2)
    assert batched == pytest.approx(pointwise)
    assert np.linalg.norm(obstacle_only[0]) > 0, (
        "Obstacle force must remain active without pedestrians"
    )


def test_social_force_fallback_is_logged_and_recorded(monkeypatch):
    """A scalar social-force failure remains compatible but is observable."""
    sim = make_no_obstacle_sim()
    wrapper = FastPysfWrapper(sim)
    warning = Mock()

    def fail(*_args, **_kwargs):
        raise ValueError("forced social-force failure")

    monkeypatch.setattr(wrapper_module.logger, "warning", warning)
    monkeypatch.setattr(wrapper_module.pf_forces, "social_force_ped_ped", fail)

    result = wrapper._compute_social_force_at_point(np.array([2.0, 0.0]))

    assert result.shape == (2,)
    assert np.all(np.isfinite(result))
    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is True
    assert diagnostics["fallback_reason"] == "social_force_inverse_square"
    assert diagnostics["fallback_reasons"] == {"social_force_inverse_square": 2}
    assert warning.call_count == 1


def test_max_speed_fallback_is_logged_and_recorded(monkeypatch):
    """A malformed max-speed payload exposes the historical 1 m/s default."""
    sim = make_no_obstacle_sim()
    wrapper = FastPysfWrapper(sim)
    warning = Mock()

    class BrokenSpeedArray:
        def __array__(self, dtype=None):
            raise ValueError("forced max-speed failure")

    monkeypatch.setattr(wrapper_module.logger, "warning", warning)
    monkeypatch.setattr(sim.peds, "max_speeds", BrokenSpeedArray())

    result = wrapper._compute_desired_force(np.array([0.0, 0.0]), [1.0, 0.0])

    assert result[0] == pytest.approx(2.0)
    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is True
    assert diagnostics["fallback_reason"] == "max_speed_default"
    assert diagnostics["fallback_reasons"] == {"max_speed_default": 1}
    assert warning.call_count == 1


def test_obstacle_force_fallback_is_logged_and_recorded(monkeypatch):
    """An obstacle-kernel failure remains a zero contribution but is visible."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)
    warning = Mock()

    def fail(*_args, **_kwargs):
        raise ValueError("forced obstacle-force failure")

    monkeypatch.setattr(wrapper_module.logger, "warning", warning)
    monkeypatch.setattr(wrapper_module.pf_forces, "obstacle_force", fail)

    result = wrapper._compute_obstacle_force_at_point(np.array([0.5, 0.0]))

    np.testing.assert_array_equal(result, np.zeros(2))
    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is True
    assert diagnostics["fallback_reason"] == "obstacle_force_dropped"
    assert diagnostics["fallback_reasons"] == {"obstacle_force_dropped": 1}
    assert warning.call_count == 1


def test_robot_force_fallback_is_logged_and_recorded(monkeypatch):
    """A robot-kernel failure remains zero-valued but is visible."""
    sim = make_no_obstacle_sim()
    wrapper = FastPysfWrapper(sim)
    warning = Mock()

    def fail(*_args, **_kwargs):
        raise ValueError("forced robot-force failure")

    monkeypatch.setattr(wrapper_module.logger, "warning", warning)
    monkeypatch.setattr(wrapper_module.pf_forces, "robot_force", fail, raising=False)

    result = wrapper._compute_robot_force_at_point(np.array([0.5, 0.0]), {})

    np.testing.assert_array_equal(result, np.zeros(2))
    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is True
    assert diagnostics["fallback_reason"] == "robot_force_zero"
    assert diagnostics["fallback_reasons"] == {"robot_force_zero": 1}
    assert warning.call_count == 1


def test_unavailable_robot_force_is_logged_and_recorded(monkeypatch):
    """Requesting an unsupported robot kernel must not silently drop the term."""
    sim = make_no_obstacle_sim()
    wrapper = FastPysfWrapper(sim)
    warning = Mock()

    monkeypatch.setattr(wrapper_module.logger, "warning", warning)
    for name in ("robot_force", "robot_interaction_force_on_point", "force_robot"):
        monkeypatch.delattr(wrapper_module.pf_forces, name, raising=False)

    result = wrapper._compute_robot_force_at_point(np.array([0.5, 0.0]), {})

    np.testing.assert_array_equal(result, np.zeros(2))
    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is True
    assert diagnostics["fallback_reason"] == "robot_force_unavailable"
    assert diagnostics["fallback_reasons"] == {"robot_force_unavailable": 1}
    assert warning.call_count == 1


def test_batched_kernel_fallbacks_are_logged_and_recorded(monkeypatch):
    """Batched kernel failures expose their pointwise compatibility paths."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)
    warning = Mock()

    def fail(*_args, **_kwargs):
        raise ValueError("forced batch-kernel failure")

    monkeypatch.setattr(wrapper_module.logger, "warning", warning)
    monkeypatch.setattr(wrapper_module.pf_forces, "social_force_single_ped", fail)
    monkeypatch.setattr(wrapper_module.pf_forces, "all_obstacle_forces", fail)

    result = wrapper.get_forces_at_points([[0.5, 0.0], [1.5, 0.0]])

    assert result.shape == (2, 2)
    assert np.all(np.isfinite(result))
    diagnostics = wrapper.diagnostics()
    assert diagnostics["fallback"] is True
    assert diagnostics["fallback_reasons"] == {
        "social_force_batch_pointwise": 2,
        "obstacle_force_batch_pointwise": 1,
    }
    assert diagnostics["fallback_count"] == 3
    assert warning.call_count == 2


def test_wrapper_init_raises_value_error_for_negative_agents(monkeypatch):
    """Wrapper initialization should reject a negative pedestrian count."""
    sim = make_simple_sim()
    monkeypatch.setattr(sim.peds, "size", lambda: -1)
    with pytest.raises(ValueError, match=r"n_agents must be non-negative \(got -1\)"):
        FastPysfWrapper(sim)


def test_get_force_field():
    """TODO docstring. Document this function."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    xs = np.linspace(-1, 3, 9)
    ys = np.linspace(-1, 1, 5)
    field = wrapper.get_force_field(xs, ys)
    assert field.shape == (len(ys), len(xs), 2)
    assert np.all(np.isfinite(field))
