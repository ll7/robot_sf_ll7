"""Module test_contract_overlay_text auto-generated docstring."""

import pytest

from robot_sf.benchmark.visualization import overlay_text


class DummyCanvas:
    """DummyCanvas class."""

    def __init__(self):
        """Init.

        Returns:
            Any: Auto-generated placeholder description.
        """
        self.calls = []

    def draw_text(self, text, pos, font=None):
        """Draw text.

        Args:
            text: Auto-generated placeholder description.
            pos: Auto-generated placeholder description.
            font: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        self.calls.append((text, pos, font))


def test_overlay_text_calls_draw_text():
    """Test overlay text calls draw text.

    Returns:
        Any: Auto-generated placeholder description.
    """
    c = DummyCanvas()
    overlay_text(c, "hello", (10, 20), font="Arial")
    assert c.calls == [("hello", (10, 20), "Arial")]


class BadCanvas:
    """BadCanvas class."""

    pass


def test_overlay_text_missing_draw_text():
    """Test overlay text missing draw text.

    Returns:
        Any: Auto-generated placeholder description.
    """
    bad = BadCanvas()
    with pytest.raises(TypeError):
        overlay_text(bad, "x", (0, 0))
