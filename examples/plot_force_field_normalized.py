"""Plot normalized force vectors with a magnitude colormap and save to docs/img.

This produces `docs/img/force_field_example_norm.png`.
"""

import matplotlib.pyplot as plt
import numpy as np
from pysocialforce import Simulator

from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper


def make_demo_sim():
    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 5.0, 1.0, 1.0],
        ]
    )
    obstacles = [
        (2.5, 2.5, -1.0, 2.0),
    ]
    sim = Simulator(state=state, obstacles=obstacles)
    return sim


def main():
    sim = make_demo_sim()
    wrapper = FastPysfWrapper(sim)

    xs = np.linspace(-1, 5, 80)
    ys = np.linspace(-2, 3, 80)

    field = wrapper.get_force_field(xs, ys)
    U = field[:, :, 0]
    V = field[:, :, 1]
    mags = np.sqrt(U**2 + V**2)

    # Avoid division by zero
    mags_safe = np.where(mags == 0, 1.0, mags)
    Un = U / mags_safe
    Vn = V / mags_safe

    # Subsample for clarity: plot every 3rd vector
    step = 3
    X, Y = np.meshgrid(xs, ys)
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Us = Un[::step, ::step]
    Vs = Vn[::step, ::step]
    Ms = mags[::step, ::step]

    fig, ax = plt.subplots(figsize=(8, 6))
    q = ax.quiver(Xs, Ys, Us, Vs, Ms, cmap="viridis", scale=30, width=0.005)
    cb = fig.colorbar(q, ax=ax)
    cb.set_label("force magnitude")
    ax.set_title("Normalized force field (color = magnitude)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out = "docs/img/force_field_example_norm.png"
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
