"""Opt-in presentation styling for simulator views (issue #9367).

This module owns the ``presentation-flat`` preset: a restrained visual
hierarchy for recorded episodes that keeps the default renderer untouched.
It carries no radii, no planner identity, and no metric semantics -- roles
are actor roles only, and every role pairs its color with a non-color
encoding so the view stays legible without color alone.

The preset applies through existing ``SimulationView`` knobs plus the
``color_overrides`` hook; legacy rendering with ``color_overrides=None`` is
byte-identical to before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PRESENTATION_FLAT_VERSION = "presentation-flat.v1"

_ROLE_NAMES = (
    "background",
    "obstacle",
    "robot",
    "robot_action",
    "pedestrian",
    "pedestrian_action",
    "ego_ped",
    "robot_goal",
    "robot_route",
    "ped_route",
    "ped_spawn_zone",
    "ped_goal_zone",
)


@dataclass(frozen=True, slots=True)
class RoleStyle:
    """Color plus non-color encoding for one presentation role."""

    color: tuple[int, ...]
    encoding: str
    note: str = ""


@dataclass(frozen=True, slots=True)
class PresentationStyle:
    """Versioned, immutable presentation preset (no radii, no planner identity)."""

    name: str
    version: str
    roles: dict[str, RoleStyle]
    title_font_px: int
    legend_font_px: int
    min_font_px: int
    layers: dict[str, bool] = field(default_factory=dict)


PRESENTATION_FLAT = PresentationStyle(
    name="presentation-flat",
    version=PRESENTATION_FLAT_VERSION,
    roles={
        # Quiet near-white canvas; light slate walls recede behind actors.
        "background": RoleStyle((248, 249, 250), "solid fill", "quiet canvas"),
        "obstacle": RoleStyle((170, 182, 196), "thin outline polygons", "light walls"),
        # Robot: Wong blue, filled disc plus white ring and heading tick.
        "robot": RoleStyle((0, 114, 178), "filled disc + white ring + heading tick", ""),
        "robot_action": RoleStyle((255, 255, 255), "heading tick line", "white on blue"),
        # Ordinary pedestrians: neutral gray, plain discs -- never danger red.
        "pedestrian": RoleStyle((150, 150, 150), "plain filled disc", ""),
        "pedestrian_action": RoleStyle(
            (100, 140, 180), "short motion tick", "muted, thinner than robot tick"
        ),
        # Ego pedestrian: Wong reddish purple; diamond glyph in legend/diagrams.
        "ego_ped": RoleStyle((204, 121, 167), "filled disc; diamond glyph in legend", ""),
        # Goal: green ring outline, never a filled disc.
        "robot_goal": RoleStyle((0, 158, 115), "ring outline only", ""),
        "robot_route": RoleStyle((0, 80, 160), "width-2 polyline", ""),
        "ped_route": RoleStyle((120, 160, 200), "width-1 polyline", ""),
        "ped_spawn_zone": RoleStyle((232, 236, 241), "very light fill", "hidden by default"),
        "ped_goal_zone": RoleStyle((220, 235, 225), "very light fill", "hidden by default"),
    },
    title_font_px=28,
    legend_font_px=18,
    min_font_px=14,
    layers={
        "help_text": False,
        "lidar": False,
        "spawn_zones": False,
        "telemetry_panel": False,
    },
)


def validate_style(style: PresentationStyle = PRESENTATION_FLAT) -> list[str]:
    """Return human-readable violations of the presentation contract.

    Checks: every role has a non-color encoding; role colors differ from the
    background; text sizes meet the minimum; no planner identity or radius
    data leaks into the spec (the dataclass carries neither field).

    Returns:
        Violation strings; empty when the preset is contract-clean.
    """
    violations: list[str] = []
    background = style.roles.get("background")
    for role in _ROLE_NAMES:
        entry = style.roles.get(role)
        if entry is None:
            violations.append(f"missing role: {role}")
            continue
        if not entry.encoding:
            violations.append(f"role without non-color encoding: {role}")
        if background is not None and entry.color == background.color and role != "background":
            violations.append(f"role indistinguishable from background: {role}")
        if "planner" in role or "algo" in role:
            violations.append(f"planner identity leaked into role: {role}")
    for label, size in (
        ("title", style.title_font_px),
        ("legend", style.legend_font_px),
    ):
        if size < style.min_font_px:
            violations.append(f"{label} font below minimum: {size}")
    return violations


def sim_view_kwargs(
    style: PresentationStyle = PRESENTATION_FLAT,
    *,
    width: int = 1280,
    height: int = 720,
    caption: str = "Robot SF — presentation",
) -> dict[str, Any]:
    """Return ``SimulationView`` constructor overrides for one preset.

    Only existing knobs are set; radii, physics, and defaults elsewhere are
    untouched. Debug help and lidar stay off in this preset.

    Returns:
        Keyword arguments for ``SimulationView``.
    """
    return {
        "width": width,
        "height": height,
        "caption": caption,
        "robot_render_mode": "circle",
        "ped_render_mode": "circle",
        "ego_ped_render_mode": "circle",
        "manual_view_mode": "fixed_map",
        "focus_on_robot": True,
        "display_help": False,
        "show_lidar": False,
        "show_telemetry_panel": False,
        "color_overrides": {role: entry.color for role, entry in style.roles.items()},
    }


def legend_entries(
    style: PresentationStyle = PRESENTATION_FLAT,
) -> list[tuple[str, str, str]]:
    """Return locale-ready ``(glyph, label, description)`` legend rows.

    Glyphs name shapes (``disc``, ``disc+tick``, ``diamond``, ``ring``,
    ``line``); no metric labels or units are invented here.

    Returns:
        Legend rows in stable display order.
    """
    _ = style
    return [
        ("disc+tick", "Robot", "ego robot with heading tick"),
        ("disc", "Pedestrian", "ordinary pedestrian, neutral"),
        ("diamond", "Focal pedestrian", "interaction focus where labelled"),
        ("ring", "Goal", "robot goal outline"),
        ("line", "Robot path", "robot route polyline"),
        ("line", "Pedestrian path", "pedestrian route polyline"),
    ]


def title_lines(scenario_title: str, annotation: str = "") -> list[str]:
    """Return title-block lines: scenario title plus one short annotation.

    Returns:
        One or two locale-ready strings; empty annotation yields one line.
    """
    lines = [scenario_title]
    if annotation:
        lines.append(annotation)
    return lines


def _require_pygame() -> object:
    """Import pygame lazily so spec and validation stay dependency-free.

    Returns:
        The pygame module.
    """
    try:
        import pygame  # noqa: PLC0415 - deliberate lazy import keeps spec import-light.
    except ImportError as exc:
        raise ImportError("presentation legend rendering requires pygame") from exc
    return pygame


def render_legend_panel(
    style: PresentationStyle = PRESENTATION_FLAT,
) -> tuple[object, list[tuple[str, tuple[int, int, int, int]]]]:
    """Render the legend panel and report per-row text bounds.

    Returns:
        ``(surface, [(label, rect), ...])`` for clipping assertions.
    """
    pygame = _require_pygame()
    font = pygame.font.Font(None, style.legend_font_px)
    rows = legend_entries(style)
    row_height = style.legend_font_px + 12
    glyph_width = 44
    widest = 0
    rendered: list[tuple[object, str, tuple[int, int]]] = []
    for glyph, label, _description in rows:
        text = font.render(label, True, (30, 30, 30))
        widest = max(widest, text.get_width())
        rendered.append((text, glyph, label))
    panel = pygame.Surface((glyph_width + widest + 32, row_height * len(rows) + 24))
    panel.fill((255, 255, 255))
    bounds: list[tuple[str, tuple[int, int, int, int]]] = []
    dark = (30, 30, 30)
    for index, (text, glyph, label) in enumerate(rendered):
        baseline = 12 + index * row_height
        panel.blit(text, (glyph_width + 16, baseline))
        bounds.append((label, (glyph_width + 16, baseline, text.get_width(), text.get_height())))
        center = (22, baseline + text.get_height() // 2)
        if glyph == "diamond":
            radius = 8
            pygame.draw.polygon(
                panel,
                dark,
                [
                    (center[0], center[1] - radius),
                    (center[0] + radius, center[1]),
                    (center[0], center[1] + radius),
                    (center[0] - radius, center[1]),
                ],
            )
        elif glyph == "ring":
            pygame.draw.circle(panel, dark, center, 9, width=2)
        elif glyph == "line":
            pygame.draw.line(panel, dark, (8, center[1]), (36, center[1]), width=2)
        else:
            pygame.draw.circle(panel, dark, center, 8)
            if glyph == "disc+tick":
                pygame.draw.line(panel, dark, center, (center[0] + 14, center[1]), width=2)
    return panel, bounds


def render_title_block(
    lines: list[str],
    width: int,
    height: int,
    style: PresentationStyle = PRESENTATION_FLAT,
) -> tuple[object, list[tuple[int, int, int, int]]]:
    """Render title lines onto a transparent overlay with bounds for clipping checks.

    Returns:
        ``(surface, [rect, ...])``; all rects must fit inside ``width`` x ``height``.
    """
    pygame = _require_pygame()
    overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    rects: list[tuple[int, int, int, int]] = []
    y = 16
    for position, line in enumerate(lines):
        size = style.title_font_px if position == 0 else style.legend_font_px
        font = pygame.font.Font(None, size)
        text = font.render(line, True, (30, 30, 30))
        if text.get_width() + 32 > width:
            shortened = line
            while shortened and font.size(shortened + "…")[0] + 32 > width:
                shortened = shortened[:-1]
            text = font.render(shortened + "…", True, (30, 30, 30))
        overlay.blit(text, (16, y))
        rects.append((16, y, text.get_width(), text.get_height()))
        y += text.get_height() + 6
    return overlay, rects
