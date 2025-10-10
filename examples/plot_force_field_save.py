"""Generate and save static images of the force field for documentation.

Writes:
- PNG: docs/img/force_field_example.png
- PDF: docs/figures/force_field_example.pdf (LaTeX-friendly)
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
        ],
    )
    obstacles = [
        (2.5, 2.5, -1.0, 2.0),
    ]
    sim = Simulator(state=state, obstacles=obstacles)
    return sim


def main():
    # LaTeX-friendly export settings (see docs/dev_guide.md)
    plt.rcParams.update(
        {
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        },
    )
    sim = make_demo_sim()
    wrapper = FastPysfWrapper(sim)

    xs = np.linspace(-1, 5, 80)
    ys = np.linspace(-2, 3, 80)
    # Build cache for faster interpolation if needed
    wrapper.build_force_grid_cache(xs, ys)

    field = wrapper.get_force_field(xs, ys)
    U = field[:, :, 0]
    V = field[:, :, 1]

    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(X, Y, U, V, scale=5, width=0.002)
    ax.set_title("Sampled force field from pysocialforce")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    out_png = "docs/img/force_field_example.png"
    fig.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")

    # Also export vector PDF to docs/figures for LaTeX inclusion
    from pathlib import Path

    pdf_path = Path("docs/figures/force_field_example.pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(pdf_path))
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
