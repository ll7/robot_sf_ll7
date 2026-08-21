"""Fast regression tests for the SimulationView robot-goal marker grammar."""

from __future__ import annotations

import numpy as np
import pygame

from robot_sf.render.sim_view import ROBOT_GOAL_COLOR, SimulationView


def test_robot_goal_is_an_outline_while_entities_stay_filled(monkeypatch) -> None:
    """The goal ring must be distinct from the filled robot and pedestrian marks."""
    draw_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _spy_circle(*args: object, **kwargs: object) -> None:
        """Record circle drawing calls without changing their call shape."""
        draw_calls.append((args, kwargs))

    monkeypatch.setattr(pygame.draw, "circle", _spy_circle)
    rendered = SimulationView(record_video=True, scaling=10, goal_radius=1.5)
    rendered._draw_robot(((1.0, 1.0), 0.0))
    rendered._draw_pedestrians(np.array([[2.0, 2.0]]))
    rendered._augment_goal_position((3.0, 3.0))

    assert len(draw_calls) == 3
    robot_call, ped_call, goal_call = draw_calls
    assert "width" not in robot_call[1]
    assert "width" not in ped_call[1]
    assert 1 <= goal_call[1]["width"] <= 3
    assert goal_call[0][1] == ROBOT_GOAL_COLOR
    assert goal_call[0][2] == rendered._scale_tuple((3.0, 3.0))
    assert goal_call[0][3] == rendered.goal_radius * rendered.scaling


def test_robot_goal_ring_is_visible_at_minimum_scale() -> None:
    """A small-scale headless render keeps the goal centre open and ring visible."""
    rendered = SimulationView(
        width=64,
        height=64,
        scaling=1,
        goal_radius=4,
        record_video=True,
    )
    rendered.screen.fill((0, 0, 0))
    rendered._augment_goal_position((32.0, 32.0))

    pixels = pygame.surfarray.array3d(rendered.screen)
    goal_color = np.asarray(ROBOT_GOAL_COLOR)
    goal_pixels = np.all(pixels == goal_color, axis=2)
    assert not np.array_equal(pixels[32, 32], goal_color)
    assert int(goal_pixels.sum()) > 0
