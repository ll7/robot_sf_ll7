"""Force-field heatmap + vector overlay figure generator.

Provides a reusable function and CLI entry point for producing LaTeX-ready
figures that visualise the magnitude and direction of the SocialForce field.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pysocialforce import Simulator

from robot_sf.benchmark.figures.export import save_publication_figure
from robot_sf.benchmark.figures.provenance import build_caption_fragment, build_provenance
from robot_sf.benchmark.figures.style import publication_style
from robot_sf.benchmark.plotting_style import apply_latex_style
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper

__all__ = ["generate_force_field_figure", "main"]


def _latex_rcparams() -> None:
    """Apply LaTeX-style plotting defaults."""
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


def generate_force_field_figure(  # noqa: PLR0913
    out_png: str | Path,
    out_pdf: str | None = None,
    *,
    x_min: float = -1.0,
    x_max: float = 5.0,
    y_min: float = -2.0,
    y_max: float = 3.0,
    grid: int = 120,
    quiver_step: int = 5,
    publication: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
    caption_fragment: str | None = None,
    size: str = "single",
    source_artifacts: list[dict[str, Any]] | None = None,
    generator_command: str | None = None,
) -> list[Path] | None:
    """Generate a force-field heatmap with optional vector overlay outputs.

    When ``publication`` is False (default) the legacy raster path is used:
    a PNG at ``out_png`` and an optional PDF at ``out_pdf``, with ad-hoc styling.
    When ``publication`` is True, the figure is rendered inside
    :func:`publication_style` and saved via :func:`save_publication_figure` in
    the requested ``formats`` with a provenance sidecar and optional caption
    fragment. The underlying field data is identical in both paths.

    Returns:
        List of written paths when ``publication`` is True, otherwise None.
    """
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

    if publication:
        return _render_force_field_publication(
            out_png,
            xs=xs,
            ys=ys,
            X=X,
            Y=Y,
            U=U,
            V=V,
            magnitude=magnitude,
            quiver_step=quiver_step,
            formats=formats,
            caption_fragment=caption_fragment,
            size=size,
            source_artifacts=source_artifacts,
            generator_command=generator_command,
        )

    _latex_rcparams()

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
    return None


def _render_force_field_publication(  # noqa: PLR0913
    out_png: str | Path,
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    magnitude: np.ndarray,
    quiver_step: int,
    formats: Sequence[str],
    caption_fragment: str | None,
    size: str,
    source_artifacts: list[dict[str, Any]] | None,
    generator_command: str | None,
) -> list[Path]:
    """Render the force field in publication style and save with provenance.

    Uses the same field arrays as the legacy path so data semantics are
    unchanged; only presentation, formats, and provenance differ.

    Returns:
        List of generated file paths.
    """
    with publication_style(size=size):
        fig, ax = plt.subplots()
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

        output_base = Path(out_png).with_suffix("")
        output_base.parent.mkdir(parents=True, exist_ok=True)
        provenance = build_provenance(
            generator_command=generator_command or "generate_force_field_figure",
            figure_formats=list(formats),
            source_artifacts=source_artifacts or [],
            claim_boundary="Force-field demo visualization; not benchmark evidence.",
        )
        saved = save_publication_figure(
            fig,
            output_base,
            formats=tuple(formats),
            provenance=provenance,
            caption_fragment=caption_fragment,
        )
    plt.close(fig)
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for force-field figure generation.

    Returns:
        Configured ArgumentParser instance.
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
    parser.add_argument(
        "--publication-style",
        action="store_true",
        default=False,
        help="Render in publication style and write provenance sidecar + caption",
    )
    parser.add_argument(
        "--format",
        default="pdf,png",
        help="Comma-separated output formats for publication mode (pdf,png,svg)",
    )
    parser.add_argument(
        "--caption-fragment",
        action="store_true",
        default=False,
        help="Write a LaTeX-ready .caption.tex sidecar (built from --campaign)",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign name used in provenance and caption fragment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for force-field figure generation.

    Returns:
        Exit code (0 for success).
    """
    args = build_arg_parser().parse_args(argv)
    formats = tuple(f.strip() for f in str(args.format).split(",") if f.strip())
    caption = build_caption_fragment(campaign_name=args.campaign) if args.caption_fragment else None
    generate_force_field_figure(
        out_png=args.png,
        out_pdf=args.pdf,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        grid=args.grid,
        quiver_step=args.quiver_step,
        publication=args.publication_style,
        formats=formats,
        caption_fragment=caption,
        generator_command=" ".join(sys.argv) if argv is None else " ".join(["force_field", *argv]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
