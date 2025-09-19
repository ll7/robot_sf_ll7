"""Force-field visualization integration for benchmark figures.

This module provides integration between the force-field visualization
functionality and the benchmark figure orchestrator.
"""

from __future__ import annotations

from pathlib import Path

from results.figures.fig_force_field import (
    generate_force_field_figure as _generate_force_field_figure,
)


def generate_force_field_figure(
    out_png: str | Path,
    out_pdf: str | None = None,
    *,
    x_min: float = -1.0,
    x_max: float = 6.0,
    y_min: float = -1.0,
    y_max: float = 3.0,
    grid: int = 40,
    quiver_step: int = 8,
) -> None:
    """Generate force-field visualization figure with heatmap and vector overlay.

    This is a wrapper around the force-field figure generation that integrates
    with the benchmark figure orchestrator.

    Args:
        out_png: Output PNG file path.
        out_pdf: Optional output PDF file path.
        x_min: Minimum x coordinate for the grid.
        x_max: Maximum x coordinate for the grid.
        y_min: Minimum y coordinate for the grid.
        y_max: Maximum y coordinate for the grid.
        grid: Grid resolution (number of points per dimension).
        quiver_step: Step size for quiver plot arrows.
    """
    _generate_force_field_figure(
        out_png=out_png,
        out_pdf=out_pdf,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        grid=grid,
        quiver_step=quiver_step,
    )


__all__ = ["generate_force_field_figure"]
