"""Contract tests for the opt-in presentation style preset (issue #9367)."""

from __future__ import annotations

import os
from dataclasses import fields

import numpy as np

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from robot_sf.render import sim_view as sim_view_module
from robot_sf.render.presentation_style import (
    PRESENTATION_FLAT,
    legend_entries,
    render_legend_panel,
    render_title_block,
    sim_view_kwargs,
    title_lines,
    validate_style,
)
from robot_sf.render.sim_view import SimulationView


def test_preset_validates_clean() -> None:
    """The shipped preset meets its own non-color/background/minimum contract."""
    assert validate_style(PRESENTATION_FLAT) == []


def test_spec_carries_no_radii_or_planner_identity() -> None:
    """Radii and planner names cannot leak into the style spec by construction."""
    field_names = {item.name for item in fields(PRESENTATION_FLAT)}
    assert "radius" not in "".join(field_names).lower()
    assert "planner" not in "".join(field_names).lower()
    roles_blob = " ".join(PRESENTATION_FLAT.roles)
    assert "planner" not in roles_blob
    assert "algo" not in roles_blob
    for entry in PRESENTATION_FLAT.roles.values():
        assert entry.encoding, "every role needs a non-color encoding"


def test_defaults_untouched_without_overrides() -> None:
    """Legacy rendering without overrides resolves every module constant."""
    view = SimulationView()

    assert view.color_overrides is None
    assert view._style_color("robot", sim_view_module.ROBOT_COLOR) == sim_view_module.ROBOT_COLOR
    assert (
        view._style_color("background", sim_view_module.BACKGROUND_COLOR)
        == sim_view_module.BACKGROUND_COLOR
    )


def test_color_overrides_reach_entity_draws(monkeypatch) -> None:
    """The override hook changes entity colors without touching draw logic."""
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _spy_circle(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(pygame.draw, "circle", _spy_circle)
    view = SimulationView(color_overrides={"robot": (0, 114, 178)})
    view._draw_robot(((1.0, 1.0), 0.0))

    assert calls, "expected a robot circle draw call"
    assert calls[0][0][1] == (0, 114, 178)


def test_kwargs_leave_radii_and_physics_untouched() -> None:
    """The kwargs adapter sets only presentation knobs, never geometry."""
    kwargs = sim_view_kwargs(PRESENTATION_FLAT, caption="Doorway specimen")

    assert not any("radius" in key for key in kwargs)
    assert not any("physics" in key or "force" in key for key in kwargs)
    view = SimulationView(**kwargs)
    plain = SimulationView()

    assert view.robot_radius == plain.robot_radius
    assert view.ped_radius == plain.ped_radius
    assert view.goal_radius == plain.goal_radius
    assert view.display_help is False
    assert view.show_lidar is False
    assert set(kwargs["color_overrides"]) >= {"robot", "pedestrian", "background"}


def test_legend_text_unclipped_at_slide_sizes() -> None:
    """Legend text fits its panel; title blocks fit 1080p and 720p surfaces."""
    _panel, bounds = render_legend_panel(PRESENTATION_FLAT)
    for _label, (x, y, width, height) in bounds:
        assert width > 0 and height > 0

    lines = title_lines("Classic doorway, high seed 112", "Robot holds; nearest clearance 0.42 m")
    assert len(lines) == 2
    for width, height in ((1920, 1080), (1280, 720)):
        _overlay, rects = render_title_block(lines, width, height)
        assert rects, "title block must emit measurable text"
        for x, y, rect_width, rect_height in rects:
            assert 0 <= x and 0 <= y
            assert x + rect_width <= width
            assert y + rect_height <= height


def test_matched_frames_differ_only_by_style() -> None:
    """Same recorded state renders deterministically different pixels per style."""
    kwargs = sim_view_kwargs(PRESENTATION_FLAT, width=320, height=200)
    styled = SimulationView(scaling=10, **kwargs)
    plain = SimulationView(width=320, height=200, scaling=10)
    pose = ((5.0, 5.0), 0.0)
    peds = np.array([[8.0, 5.0]])

    styled.screen.fill((0, 0, 0))
    styled._draw_robot(pose)
    styled._draw_pedestrians(peds)
    first = pygame.surfarray.array3d(styled.screen).copy()

    styled.screen.fill((0, 0, 0))
    styled._draw_robot(pose)
    styled._draw_pedestrians(peds)
    second = pygame.surfarray.array3d(styled.screen)

    plain.screen.fill((0, 0, 0))
    plain._draw_robot(pose)
    plain._draw_pedestrians(peds)
    baseline = pygame.surfarray.array3d(plain.screen)

    assert np.array_equal(first, second), "preset rendering must be deterministic"
    assert not np.array_equal(first, baseline), "preset must change rendered pixels"
    # Recorded inputs are untouched by styling.
    assert pose == ((5.0, 5.0), 0.0)
    assert peds.tolist() == [[8.0, 5.0]]


def test_legend_uses_no_metric_units_and_no_outcome_rows() -> None:
    """Legend rows name roles only; unknown outcome is never rendered as success."""
    rows = legend_entries(PRESENTATION_FLAT)

    assert rows, "legend must not be empty"
    blob = " ".join(label for _glyph, label, _description in rows).lower()
    assert "success" not in blob
    assert "collision" not in blob
    assert all("planner" not in label.lower() for _glyph, label, _description in rows)


def _zone_map():
    """Map with one obstacle and one spawn/goal zone triple for hook coverage."""
    from robot_sf.nav.map_config import MapDefinition
    from robot_sf.nav.obstacle import Obstacle

    zone = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))
    far_zone = ((6.0, 6.0), (8.0, 6.0), (8.0, 8.0))
    return MapDefinition(
        width=10.0,
        height=8.0,
        obstacles=[Obstacle([(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0)])],
        robot_spawn_zones=[zone],
        robot_goal_zones=[far_zone],
        ped_spawn_zones=[zone],
        ped_goal_zones=[far_zone],
        ped_crowded_zones=[],
        bounds=[],
        robot_routes=[],
        ped_routes=[],
    )


def _override_view():
    """View with every hook color overridden to a unique sentinel."""
    return SimulationView(
        width=64,
        height=64,
        scaling=4,
        map_def=_zone_map(),
        color_overrides={
            "background": (11, 12, 13),
            "obstacle": (21, 22, 23),
            "ped_spawn_zone": (31, 32, 33),
            "ped_goal_zone": (41, 42, 43),
        },
    )


def test_override_colors_reach_background_zones_and_obstacles():
    """Background fill, obstacle, and zone hooks execute with overrides."""
    view = _override_view()
    view.clear()
    view._draw_obstacles()
    view._draw_spawn_zones()
    view._draw_goal_zones()

    pixels = pygame.surfarray.array3d(view.screen)
    for sentinel in ((11, 12, 13), (21, 22, 23), (31, 32, 33), (41, 42, 43)):
        assert (np.all(pixels == sentinel, axis=2)).any(), sentinel


def test_prepare_frame_executes_background_fill_with_overrides():
    """The full-frame path fills through the background hook."""
    from robot_sf.render.sim_view import VisualizableAction, VisualizableSimState

    view = _override_view()
    state = VisualizableSimState(
        timestep=0,
        robot_action=VisualizableAction(pose=((1.0, 1.0), 0.0), action=(0.0, 0.0), goal=(4.0, 4.0)),
        robot_pose=((1.0, 1.0), 0.0),
        pedestrian_positions=np.array([[2.0, 2.0]]),
        ray_vecs=np.array([]),
        ped_actions=np.array([]),
        planned_path=[],
        time_per_step_in_secs=0.1,
    )
    view._prepare_frame(state)

    pixels = pygame.surfarray.array3d(view.screen)
    assert (np.all(pixels == (11, 12, 13), axis=2)).any()
