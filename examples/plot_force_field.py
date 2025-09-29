"""Small example that constructs a pysocialforce Simulator, wraps it and
plots a sampled force field using matplotlib.

Run:

    python examples/plot_force_field.py

"""

import matplotlib.pyplot as plt
import numpy as np
from pysocialforce import Simulator

from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


def make_demo_sim():
    # two pedestrians walking to the right, small vertical obstacle
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 5.0, 1.0, 1.0],
        ],
    )
    obstacles = [
        (2.5, 2.5, -1.0, 2.0),
    ]
    sim = Simulator(state=state, obstacles=obstacles)
    return sim


def main():
    sim = make_demo_sim()
    wrapper = FastPysfWrapper(sim)

    xs = np.linspace(-1, 5, 50)
    ys = np.linspace(-2, 3, 50)

    field = wrapper.get_force_field(xs, ys)
    U = field[:, :, 0]
    V = field[:, :, 1]

    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 6))
    _q = ax.quiver(X, Y, U, V, scale=5, width=0.003)
    ax.set_title("Sampled force field from pysocialforce")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
