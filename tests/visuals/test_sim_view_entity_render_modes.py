"""Tests for SimulationView entity render modes (circle vs sprite)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame

from robot_sf.render.sim_view import SimulationView

if TYPE_CHECKING:
    from pathlib import Path


def _write_sprite(path: Path) -> None:
    """Write a tiny RGBA sprite file for rendering tests."""
    pygame.init()
    surface = pygame.Surface((8, 8), pygame.SRCALPHA)
    surface.fill((255, 0, 0, 255))
    pygame.image.save(surface, str(path))


def test_robot_circle_mode_draws_circle(monkeypatch) -> None:
    """Robot circle mode should use pygame circle drawing path."""
    draw_calls: list[tuple] = []

    def _spy_circle(*args: object, **kwargs: object) -> None:
        draw_calls.append((args, kwargs))

    monkeypatch.setattr(pygame.draw, "circle", _spy_circle)
    view = SimulationView(record_video=True, robot_render_mode="circle")
    view._draw_robot(((1.0, 1.0), 0.0))
    assert draw_calls


def test_robot_sprite_mode_uses_rotation(monkeypatch, tmp_path: Path) -> None:
    """Robot sprite mode should rotate sprite based on heading."""
    sprite_path = tmp_path / "robot_sprite.png"
    _write_sprite(sprite_path)
    rotate_calls: list[float] = []
    orig_rotate = pygame.transform.rotate

    def _spy_rotate(surface, angle):
        rotate_calls.append(float(angle))
        return orig_rotate(surface, angle)

    monkeypatch.setattr(pygame.transform, "rotate", _spy_rotate)
    view = SimulationView(
        record_video=True,
        robot_render_mode="sprite",
        robot_sprite_path=str(sprite_path),
    )
    view._draw_robot(((1.0, 1.0), 0.5))
    assert rotate_calls


def test_robot_sprite_missing_path_falls_back_to_circle(monkeypatch) -> None:
    """Missing robot sprite should fall back to circle rendering."""
    draw_calls: list[tuple] = []

    def _spy_circle(*args: object, **kwargs: object) -> None:
        draw_calls.append((args, kwargs))

    monkeypatch.setattr(pygame.draw, "circle", _spy_circle)
    view = SimulationView(
        record_video=True,
        robot_render_mode="sprite",
        robot_sprite_path="/tmp/definitely_missing_robot_sprite.png",
    )
    view._draw_robot(((1.0, 1.0), 0.0))
    assert draw_calls


def test_pedestrian_sprite_mode_uses_sprite_branch(monkeypatch, tmp_path: Path) -> None:
    """Pedestrian sprite mode should call sprite draw helper with no rotation."""
    sprite_path = tmp_path / "ped_sprite.png"
    _write_sprite(sprite_path)
    sprite_calls: list[float | None] = []
    view = SimulationView(
        record_video=True,
        ped_render_mode="sprite",
        ped_sprite_path=str(sprite_path),
    )

    def _spy_draw_sprite(
        sprite: object,
        center: object,
        radius_px: object,
        theta: float | None = None,
    ) -> None:
        del sprite, center, radius_px
        sprite_calls.append(theta)

    monkeypatch.setattr(view, "_draw_sprite", _spy_draw_sprite)
    view._draw_pedestrians(np.array([[1.0, 1.0]]))
    assert sprite_calls == [None]
