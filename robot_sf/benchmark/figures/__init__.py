"""Benchmark figures generation modules.

This package contains modules for generating various figures used in
benchmark analysis and reporting.

For publication-grade figures, use the :mod:`robot_sf.benchmark.figures.style`,
:mod:`robot_sf.benchmark.figures.provenance`, and :mod:`robot_sf.benchmark.figures.export`
modules.
"""

from robot_sf.benchmark.figures.export import save_publication_figure
from robot_sf.benchmark.figures.force_field import generate_force_field_figure
from robot_sf.benchmark.figures.provenance import (
    build_caption_fragment,
    build_provenance,
    write_caption_fragment,
    write_provenance,
)
from robot_sf.benchmark.figures.style import (
    figure_size,
    metric_label,
    planner_color,
    planner_palette,
    publication_style,
)
from robot_sf.benchmark.figures.thumbnails import ThumbMeta, save_montage, save_scenario_thumbnails

__all__ = [
    "ThumbMeta",
    "build_caption_fragment",
    "build_provenance",
    "figure_size",
    "generate_force_field_figure",
    "metric_label",
    "planner_color",
    "planner_palette",
    "publication_style",
    "save_montage",
    "save_publication_figure",
    "save_scenario_thumbnails",
    "write_caption_fragment",
    "write_provenance",
]
