"""Contract tests for ``overlay_text`` canvas text drawing."""

import pytest

from robot_sf.benchmark.visualization import overlay_text


class DummyCanvas:
    """Canvas double that records draw_text calls for assertions."""

    def __init__(self):
        """Initialize an empty call log."""
        self.calls = []

    def draw_text(self, text, pos, font=None):
        """Record the text, position, and font passed by overlay_text.

        Args:
            text: Text drawn by the overlay.
            pos: Pixel position of the text.
            font: Optional font name.
        """
        self.calls.append((text, pos, font))


def test_overlay_text_calls_draw_text():
    """Verify overlay_text forwards text, position, and font to the canvas."""
    c = DummyCanvas()
    overlay_text(c, "hello", (10, 20), font="Arial")
    assert c.calls == [("hello", (10, 20), "Arial")]


class BadCanvas:
    """Canvas double without a draw_text method."""

    pass


def test_overlay_text_missing_draw_text():
    """Verify a canvas lacking draw_text raises TypeError."""
    bad = BadCanvas()
    with pytest.raises(TypeError):
        overlay_text(bad, "x", (0, 0))
