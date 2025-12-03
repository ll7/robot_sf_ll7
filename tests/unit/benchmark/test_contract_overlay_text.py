"""TODO docstring. Document this module."""

import pytest

from robot_sf.benchmark.visualization import overlay_text


class DummyCanvas:
    """TODO docstring. Document this class."""

    def __init__(self):
        """TODO docstring. Document this function."""
        self.calls = []

    def draw_text(self, text, pos, font=None):
        """TODO docstring. Document this function.

        Args:
            text: TODO docstring.
            pos: TODO docstring.
            font: TODO docstring.
        """
        self.calls.append((text, pos, font))


def test_overlay_text_calls_draw_text():
    """TODO docstring. Document this function."""
    c = DummyCanvas()
    overlay_text(c, "hello", (10, 20), font="Arial")
    assert c.calls == [("hello", (10, 20), "Arial")]


class BadCanvas:
    """TODO docstring. Document this class."""

    pass


def test_overlay_text_missing_draw_text():
    """TODO docstring. Document this function."""
    bad = BadCanvas()
    with pytest.raises(TypeError):
        overlay_text(bad, "x", (0, 0))
