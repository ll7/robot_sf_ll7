"""Render module for visualization and recording utilities.

This module provides tools for rendering simulations, capturing frames,
and managing video recording outputs.
"""

from robot_sf.render.helper_catalog import capture_frames, ensure_output_dir

__all__ = [
    "capture_frames",
    "ensure_output_dir",
]
