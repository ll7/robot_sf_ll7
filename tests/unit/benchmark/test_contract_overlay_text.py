import pytest

from robot_sf.benchmark.visualization import overlay_text


class DummyCanvas:
    def __init__(self):
        self.calls = []

    def draw_text(self, text, pos, font=None):
        self.calls.append((text, pos, font))


def test_overlay_text_calls_draw_text():
    c = DummyCanvas()
    overlay_text(c, "hello", (10, 20), font="Arial")
    assert c.calls == [("hello", (10, 20), "Arial")]


class BadCanvas:
    pass


def test_overlay_text_missing_draw_text():
    bad = BadCanvas()
    with pytest.raises(TypeError):
        overlay_text(bad, "x", (0, 0))
