"""Force-field heatmap + vector overlay figure generator.

Provides a reusable function and CLI entry point for producing LaTeX-ready
figures that visualise the magnitude and direction of the SocialForce field.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pysocialforce import Simulator

from robot_sf.benchmark.figures.export import save_publication_figure
from robot_sf.benchmark.figures.provenance import (
    ProvenanceConfig,
    build_provenance,
)
from robot_sf.benchmark.figures.style import publication_style
from robot_sf.benchmark.plotting_style import apply_latex_style
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper

__all__ = ["generate_force_field_figure", "main"]

_VALID_FORMATS = ("pdf", "png", "svg")


def _validate_formats(formats: list[str]) -> tuple[str, ...]:
    """Validate and return format tuple.

    Args:
        formats: List of format strings to validate.

    Returns:
        Tuple of validated format strings.

    Raises:
        ValueError: If any format is not in the allowed set.
    """
    for fmt in formats:
        if fmt not in _VALID_FORMATS:
            raise ValueError(
                f"Unsupported format: {fmt!r}. Must be one of {tuple(_VALID_FORMATS)}."
            )
    return tuple(formats)


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
    out_png: str | Path | None = None,
    out_pdf: str | Path | None = None,
    *,
    out_dir: str | Path | None = None,
    figure_name: str = "fig-force-field",
    x_min: float = -1.0,
    x_max: float = 5.0,
    y_min: float = -2.0,
    y_max: float = 3.0,
    grid: int = 120,
    quiver_step: int = 5,
    formats: tuple[str, ...] | None = None,
    publication: bool = False,
    generator_command: str | None = None,
) -> list[Path]:
    """Generate a force-field heatmap with optional vector overlay outputs.

    Args:
        out_png: Legacy output PNG path (for backward compatibility).
        out_pdf: Legacy output PDF path (for backward compatibility).
        out_dir: Output directory for multi-format export.
        figure_name: Base name for output files (without extension).
        x_min: Minimum x coordinate for grid.
        x_max: Maximum x coordinate for grid.
        y_min: Minimum y coordinate for grid.
        y_max: Maximum y coordinate for grid.
        grid: Number of grid points per dimension.
        quiver_step: Step size for quiver vector overlay.
        formats: Tuple of output formats ("pdf", "png", "svg").
            Defaults to ("png", "pdf") when using new --formats flag,
            or legacy behavior when out_png/out_pdf specified.
        publication: Whether to apply publication-grade styling.
        generator_command: Override generator command string for provenance.

    Returns:
        List of generated file paths.
    """
    style_context = publication_style() if publication else _noop_context()

    with style_context:
        if not publication:
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
            extent=(
                float(xs.min()),
                float(xs.max()),
                float(ys.min()),
                float(ys.max()),
            ),
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

        # Determine output strategy: legacy paths vs multi-format
        if out_png is not None or out_pdf is not None:
            # Legacy backward-compatible path
            saved: list[Path] = []
            if out_png is not None:
                out_png_path = Path(out_png)
                out_png_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_png_path, dpi=300)
                saved.append(out_png_path)
            if out_pdf is not None:
                out_pdf_path = Path(out_pdf)
                out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_pdf_path)
                saved.append(out_pdf_path)
            plt.close(fig)
            return saved

        # Multi-format export path
        if formats is None:
            formats = ("pdf", "png")

        if out_dir is None:
            out_dir = "."

        output_base = Path(out_dir) / figure_name

        provenance_config = ProvenanceConfig(
            generator_command=generator_command or "generate_force_field_figure",
            figure_formats=list(formats),
        )
        provenance = build_provenance(provenance_config)

        saved = save_publication_figure(
            fig,
            output_base,
            formats=formats,
            provenance=provenance,
        )
        plt.close(fig)
        return saved


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for force-field figure generation.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Generate force-field figure outputs")
    parser.add_argument("--png", default=None, help="Output PNG path (legacy)")
    parser.add_argument("--pdf", default=None, help="Output PDF path (legacy)")
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats: pdf, png, svg. Default: pdf,png",
    )
    parser.add_argument(
        "--out-dir",
        default="docs/figures",
        help="Output directory for multi-format figures",
    )
    parser.add_argument(
        "--figure-name",
        default="fig-force-field",
        help="Base name for output files (default: fig-force-field)",
    )
    parser.add_argument(
        "--publication-style",
        action="store_true",
        default=False,
        help="Apply publication-grade styling",
    )
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--y-min", type=float, default=-2.0)
    parser.add_argument("--y-max", type=float, default=3.0)
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument("--quiver-step", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for force-field figure generation.

    Returns:
        Exit code (0 for success).
    """
    args = build_arg_parser().parse_args(argv)

    # Parse formats
    fmt_str = str(args.formats).strip()
    raw_formats = [f.strip() for f in fmt_str.split(",") if f.strip()]
    try:
        formats = _validate_formats(raw_formats)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr, flush=True)  # noqa: T201
        return 1

    generate_force_field_figure(
        out_png=args.png if args.png else None,
        out_pdf=args.pdf if args.pdf else None,
        out_dir=args.out_dir,
        figure_name=args.figure_name,
        x_min=float(args.x_min),
        x_max=float(args.x_max),
        y_min=float(args.y_min),
        y_max=float(args.y_max),
        grid=int(args.grid),
        quiver_step=int(args.quiver_step),
        formats=formats,
        publication=bool(args.publication_style),
        generator_command=" ".join(["generate_force_field_figure"] + (argv or []))
        or "generate_force_field_figure",
    )
    return 0


def _noop_context():
    """No-op context manager for non-publication style path.

    Returns:
        A nullcontext (no-op context manager).
    """
    return nullcontext()


if __name__ == "__main__":
    raise SystemExit(main())
