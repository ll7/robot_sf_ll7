"""Module test_contract_frame_shape auto-generated docstring."""

import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.visualization import frame_shape_from_map


def make_svg(content: str, suffix: str = ".svg") -> Path:
    """Make svg.

    Args:
        content: Auto-generated placeholder description.
        suffix: Auto-generated placeholder description.

    Returns:
        Path: Auto-generated placeholder description.
    """
    # Use NamedTemporaryFile with delete=False so the file persists for the
    # duration of the test run. The OS temp directory will be cleaned later.
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8")
    tf.write(content)
    tf.flush()
    tf.close()
    return Path(tf.name)


def test_frame_shape_from_map_width_height():
    """Test frame shape from map width height.

    Returns:
        Any: Auto-generated placeholder description.
    """
    svg = """<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg"></svg>"""
    p = make_svg(svg)
    w, h = frame_shape_from_map(str(p))
    assert (w, h) == (800, 600)


def test_frame_shape_from_map_viewbox():
    """Test frame shape from map viewbox.

    Returns:
        Any: Auto-generated placeholder description.
    """
    svg = """<svg viewBox="0 0 1024 768" xmlns="http://www.w3.org/2000/svg"></svg>"""
    p = make_svg(svg)
    w, h = frame_shape_from_map(str(p))
    assert (w, h) == (1024, 768)


def test_frame_shape_from_map_invalid():
    """Test frame shape from map invalid.

    Returns:
        Any: Auto-generated placeholder description.
    """
    svg = """<svg></svg>"""
    p = make_svg(svg)
    with pytest.raises(ValueError):
        frame_shape_from_map(str(p))
