"""Force-field heatmap + vector overlay figure generator.

Provides a reusable function and CLI entry point for producing LaTeX-ready
figures that visualise the magnitude and direction of the SocialForce field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pysocialforce import Simulator

from robot_sf.benchmark.plotting_style import apply_latex_style
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper

__all__ = ["generate_force_field_figure", "main"]


def _latex_rcparams() -> None:
    """TODO docstring. Document this function."""
    apply_latex_style()


def make_demo_sim() -> Simulator:
    """Return a minimal simulator with two pedestrians and one obstacle.

    Returns:
        Configured Simulator instance with demo scenario.
    """

    state = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 5.0, 1.0, 1.0],
        ],
    )
    obstacles = [
        (2.5, 2.5, -1.0, 2.0),
    ]
    return Simulator(state=state, obstacles=obstacles)


def generate_force_field_figure(
    out_png: str | Path,
    out_pdf: str | None = None,
    *,
    x_min: float = -1.0,
    x_max: float = 5.0,
    y_min: float = -2.0,
    y_max: float = 3.0,
    grid: int = 120,
    quiver_step: int = 5,
) -> None:
    """Generate a force-field heatmap with optional vector overlay outputs."""

    _latex_rcparams()

    sim = make_demo_sim()
    wrapper = FastPysfWrapper(sim)

    xs = np.linspace(float(x_min), float(x_max), int(grid))
    ys = np.linspace(float(y_min), float(y_max), int(grid))
    wrapper.build_force_grid_cache(xs, ys)

    field = wrapper.get_force_field(xs, ys)
    U = field[:, :, 0]
    V = field[:, :, 1]
    magnitude = np.hypot(U, V)

    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=200)
    im = ax.imshow(
        magnitude,
        origin="lower",
        extent=(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())),
        cmap="magma",
        alpha=0.95,
        aspect="equal",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("|F| (a.u.)")

    ax.quiver(
        X[::quiver_step, ::quiver_step],
        Y[::quiver_step, ::quiver_step],
        U[::quiver_step, ::quiver_step],
        V[::quiver_step, ::quiver_step],
        color="white",
        scale=10,
        width=0.003,
        alpha=0.8,
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Force field magnitude with flow vectors")

    out_png_path = Path(out_png)
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png_path, dpi=300)

    if out_pdf:
        out_pdf_path = Path(out_pdf)
        out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf_path)

    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    parser = argparse.ArgumentParser(description="Generate force-field figure outputs")
    parser.add_argument("--png", default="docs/img/fig-force-field.png")
    parser.add_argument("--pdf", default="docs/figures/fig-force-field.pdf")
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--y-min", type=float, default=-2.0)
    parser.add_argument("--y-max", type=float, default=3.0)
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument("--quiver-step", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """TODO docstring. Document this function.

    Args:
        argv: TODO docstring.

    Returns:
        TODO docstring.
    """
    args = build_arg_parser().parse_args(argv)
    generate_force_field_figure(
        out_png=args.png,
        out_pdf=args.pdf,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        grid=args.grid,
        quiver_step=args.quiver_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
