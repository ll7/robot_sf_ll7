"""Contract tests for ``frame_shape_from_map`` SVG dimension parsing."""

import tempfile
from pathlib import Path

import pytest

from robot_sf.benchmark.visualization import frame_shape_from_map


def make_svg(content: str, suffix: str = ".svg") -> Path:
    # Use NamedTemporaryFile with delete=False so the file persists for the
    # duration of the test run. The OS temp directory will be cleaned later.
    """Write SVG content to a persistent temporary file for parsing.

    Args:
        content: SVG document text to write.
        suffix: File suffix controlling the temporary file extension.

    Returns:
        Path to the written temporary SVG file.
    """
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode="w", encoding="utf-8")
    tf.write(content)
    tf.flush()
    tf.close()
    return Path(tf.name)


def test_frame_shape_from_map_width_height():
    """Verify width/height attributes are used when viewBox is absent."""
    svg = """<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg"></svg>"""
    p = make_svg(svg)
    w, h = frame_shape_from_map(str(p))
    assert (w, h) == (800, 600)


def test_frame_shape_from_map_viewbox():
    """Verify the viewBox fallback yields its width and height."""
    svg = """<svg viewBox="0 0 1024 768" xmlns="http://www.w3.org/2000/svg"></svg>"""
    p = make_svg(svg)
    w, h = frame_shape_from_map(str(p))
    assert (w, h) == (1024, 768)


@pytest.mark.parametrize(
    ("viewbox", "expected"),
    [
        ("0,0,100,200", (100, 200)),
        ("0, 0, 1024, 768", (1024, 768)),
    ],
)
def test_frame_shape_from_map_viewbox_with_commas(viewbox: str, expected: tuple[int, int]):
    """Parse comma-separated SVG viewBox dimensions."""
    svg = f"""<svg viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg"></svg>"""
    p = make_svg(svg)
    w, h = frame_shape_from_map(str(p))
    assert (w, h) == expected


def test_frame_shape_from_map_invalid():
    """Verify an SVG lacking both dimensions raises ValueError."""
    svg = """<svg></svg>"""
    p = make_svg(svg)
    with pytest.raises(ValueError):
        frame_shape_from_map(str(p))
