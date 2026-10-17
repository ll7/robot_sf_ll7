"""Regression tests for the simulation view's event-queue dispatch."""

from types import SimpleNamespace

import pygame

from robot_sf.render.sim_view import SimulationView


def test_event_queue_quit_requests_exit_and_abortion(monkeypatch) -> None:
    """Dispatch window events through real handlers without opening a display."""
    view = object.__new__(SimulationView)
    view.is_exit_requested = False
    view.is_abortion_requested = False
    view.record_video = False
    view.size_changed = False
    view.display_text = False
    ticks = []
    view.clock = SimpleNamespace(tick=ticks.append)
    events = [
        pygame.event.Event(pygame.VIDEORESIZE, {"w": 140, "h": 100}),
        pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_t}),
        pygame.event.Event(pygame.QUIT),
    ]
    monkeypatch.setattr(pygame.event, "get", lambda: events)
    monkeypatch.setattr(pygame.key, "get_mods", lambda: 0)

    view._process_event_queue()

    assert view.is_exit_requested is True
    assert view.is_abortion_requested is True
    assert view.size_changed is True
    assert (view.width, view.height) == (140, 100)
    assert view.display_text is True
    assert ticks == [30]
