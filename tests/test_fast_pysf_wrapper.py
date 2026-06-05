"""TODO docstring. Document this module."""

import numpy as np
import pytest
from pysocialforce import Simulator

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

    assert batched.shape == (len(points), 2)
    assert batched == pytest.approx(pointwise)


def test_get_force_field():
    """TODO docstring. Document this function."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    xs = np.linspace(-1, 3, 9)
    ys = np.linspace(-1, 1, 5)
    field = wrapper.get_force_field(xs, ys)
    assert field.shape == (len(ys), len(xs), 2)
    assert np.all(np.isfinite(field))
