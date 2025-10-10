"""Benchmark figures generation modules.

This package contains modules for generating various figures used in
benchmark analysis and reporting.
"""

from robot_sf.benchmark.figures.force_field import generate_force_field_figure
from robot_sf.benchmark.figures.thumbnails import ThumbMeta, save_montage, save_scenario_thumbnails

__all__ = [
    "ThumbMeta",
    "generate_force_field_figure",
    "save_montage",
    "save_scenario_thumbnails",
]
