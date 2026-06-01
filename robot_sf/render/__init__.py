"""Render module for visualization and recording utilities.

This module provides tools for rendering simulations, capturing frames,
and managing video recording outputs.
"""

from robot_sf.render.helper_catalog import capture_frames, ensure_output_dir
from robot_sf.render.trace_viewer import (
    TRACE_VIEWER_SCENE_VERSION,
    TraceViewerResult,
    build_trace_scene,
    export_trace_viewer,
)

__all__ = [
    "TRACE_VIEWER_SCENE_VERSION",
    "TraceViewerResult",
    "build_trace_scene",
    "capture_frames",
    "ensure_output_dir",
    "export_trace_viewer",
]
