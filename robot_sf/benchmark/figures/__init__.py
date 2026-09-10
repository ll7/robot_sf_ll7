"""Benchmark figures generation modules.

This package contains modules for generating various figures used in
benchmark analysis and reporting.

For publication-grade figures, use the :mod:`robot_sf.benchmark.figures.style`,
:mod:`robot_sf.benchmark.figures.provenance`, and :mod:`robot_sf.benchmark.figures.export`
modules. Stable metric and planner display metadata is provided by
:mod:`robot_sf.benchmark.figures.semantics`.
"""

from robot_sf.benchmark.figures.export import save_publication_figure
from robot_sf.benchmark.figures.force_field import generate_force_field_figure
from robot_sf.benchmark.figures.provenance import (
    build_caption_fragment,
    build_provenance,
    write_caption_fragment,
    write_provenance,
)
from robot_sf.benchmark.figures.semantics import (
    SemanticsRegistry,
    default_registry,
    planner_label,
)
from robot_sf.benchmark.figures.style import (
    figure_size,
    metric_label,
    planner_color,
    planner_palette,
    publication_style,
    semantics_sha256,
)
from robot_sf.benchmark.figures.thumbnails import ThumbMeta, save_montage, save_scenario_thumbnails

__all__ = [
    "SemanticsRegistry",
    "ThumbMeta",
    "build_caption_fragment",
    "build_provenance",
    "default_registry",
    "figure_size",
    "generate_force_field_figure",
    "metric_label",
    "planner_color",
    "planner_label",
    "planner_palette",
    "publication_style",
    "save_montage",
    "save_publication_figure",
    "save_scenario_thumbnails",
    "semantics_sha256",
    "write_caption_fragment",
    "write_provenance",
]
