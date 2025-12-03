"""TODO docstring. Document this module."""

import numpy as np
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


def test_get_forces_at_point():
    """TODO docstring. Document this function."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    f = wrapper.get_forces_at([0.5, 0.0])
    assert f.shape == (2,)
    # force should be finite numbers
    assert np.all(np.isfinite(f))


def test_get_force_field():
    """TODO docstring. Document this function."""
    sim = make_simple_sim()
    wrapper = FastPysfWrapper(sim)

    xs = np.linspace(-1, 3, 9)
    ys = np.linspace(-1, 1, 5)
    field = wrapper.get_force_field(xs, ys)
    assert field.shape == (len(ys), len(xs), 2)
    assert np.all(np.isfinite(field))
