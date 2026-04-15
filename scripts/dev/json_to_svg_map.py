"""Historical JSON-to-SVG migration tool used for issue #532.

This script converts a legacy JSON map definition to an SVG file that, when
parsed by ``robot_sf.nav.svg_map_parser.convert_map``, produces a
``MapDefinition`` equivalent to the one produced by
``robot_sf.nav.map_config.serialize_map``.

Key mapping decisions
---------------------
- Obstacles (polygon point lists) → ``<path d="M ... Z">`` with
  ``inkscape:label="obstacle"``.
- Zones (3-point tuples) → ``<rect>`` with the appropriate label.
  The bounding box is derived from the min/max of the 3 points.
- Robot routes → both the original AND reversed route are written as
  ``<path>`` elements so that the SVG-parsed ``MapDefinition`` has the same
  doubled set of routes that ``serialize_map`` produces.
- Pedestrian routes → ``<path>`` with ``inkscape:label="ped_route_<s>_<g>"``.
- Crowded zones → ``<rect inkscape:label="ped_crowded_zone">``.

Coordinate normalisation
------------------------
The JSON stores absolute world coordinates offset by ``x_margin`` / ``y_margin``.
``serialize_map`` translates them so that (min_x, min_y) maps to (0, 0).
The generated SVG uses the same translated coordinate space so
``convert_map`` can parse it without any additional offset.

Usage::

    uv run python scripts/dev/json_to_svg_map.py \\
        path/to/legacy_map.json \\
        robot_sf/maps/uni_campus_big.svg

    # Verify parity after generating the SVG:
    uv run python -c "
    from robot_sf.nav.svg_map_parser import convert_map
    m = convert_map('robot_sf/maps/uni_campus_big.svg')
    print('obstacles:', len(m.obstacles))
    print('robot_spawn_zones:', len(m.robot_spawn_zones))
    print('robot_routes:', len(m.robot_routes))
    "
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

Vec2D = tuple[float, float]


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _translate(pt: list[float], min_x: float, min_y: float) -> Vec2D:
    """Translate a point from JSON absolute coordinates to SVG (0-based) space."""
    return (pt[0] - min_x, pt[1] - min_y)


def _zone_rect(
    raw_zone: list[list[float]], min_x: float, min_y: float
) -> tuple[float, float, float, float]:
    """Return (x, y, width, height) of the bounding rect for a 3-point zone."""
    pts = [_translate(p, min_x, min_y) for p in raw_zone]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x = min(xs)
    y = min(ys)
    w = max(xs) - x
    h = max(ys) - y
    return x, y, w, h


def _path_d(waypoints: list[Vec2D]) -> str:
    """Build an absolute SVG path ``d`` attribute from a list of (x, y) waypoints."""
    if not waypoints:
        return ""
    parts = [f"M {waypoints[0][0]:.4f},{waypoints[0][1]:.4f}"]
    for pt in waypoints[1:]:
        parts.append(f"L {pt[0]:.4f},{pt[1]:.4f}")
    return " ".join(parts)


def _obstacle_d(polygon: list[Vec2D]) -> str:
    """Build a closed SVG path ``d`` attribute for an obstacle polygon."""
    return _path_d(polygon) + " Z"


# ---------------------------------------------------------------------------
# SVG XML string builders
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    """Escape XML attribute values."""
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def _rect_elem(
    rect: tuple[float, float, float, float],
    label: str,
    elem_id: str,
    fill: str = "none",
    stroke: str = "black",
    stroke_width: float = 0.5,
) -> str:
    x, y, w, h = rect
    return (
        f'    <rect id="{_esc(elem_id)}" inkscape:label="{_esc(label)}"'
        f' x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}"'
        f' style="fill:{fill};stroke:{stroke};stroke-width:{stroke_width}" />'
    )


def _path_elem(
    d: str,
    label: str,
    elem_id: str,
    fill: str = "none",
    stroke: str = "black",
    stroke_width: float = 0.5,
) -> str:
    return (
        f'    <path id="{_esc(elem_id)}" inkscape:label="{_esc(label)}"'
        f' d="{_esc(d)}"'
        f' style="fill:{fill};stroke:{stroke};stroke-width:{stroke_width}" />'
    )


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


_SUPPORTED_JSON_KEYS = frozenset(
    {
        "x_margin",
        "y_margin",
        "obstacles",
        "robot_spawn_zones",
        "robot_goal_zones",
        "ped_spawn_zones",
        "ped_goal_zones",
        "ped_crowded_zones",
        "ped_routes",
        "robot_routes",
    }
)
"""JSON map keys that this converter can represent in SVG.

``serialize_map`` also handles ``single_pedestrians`` (individual ped
trajectories) and any future extensions.  These cannot be expressed in the
current SVG schema and are silently dropped unless explicitly detected here.
"""


def convert_json_to_svg(map_data: dict[str, Any], output_path: Path) -> None:
    """Convert a JSON map dictionary to an SVG file at *output_path*.

    Args:
        map_data: Parsed JSON map dictionary (as loaded from a ``.json`` map file).
        output_path: Destination ``.svg`` file path.

    Raises:
        SystemExit: If the map contains keys not supported by this converter
            (e.g. ``single_pedestrians``).  Use ``--force`` to convert anyway
            and accept that those fields will be dropped.
    """
    unsupported = sorted(set(map_data.keys()) - _SUPPORTED_JSON_KEYS)
    if unsupported:
        print(
            f"ERROR: the source map contains fields that cannot be represented "
            f"in SVG and would be silently dropped:\n  {unsupported}\n"
            f"Aborting conversion to prevent data loss.  Remove or migrate those "
            f"fields before converting, or add SVG support for them.",
            file=sys.stderr,
        )
        sys.exit(1)

    min_x, max_x = map_data["x_margin"]
    min_y, max_y = map_data["y_margin"]
    width = max_x - min_x
    height = max_y - min_y

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"'
        f' width="{width:.4f}" height="{height:.4f}"'
        f' viewBox="0 0 {width:.4f} {height:.4f}">'
    )

    # ------------------------------------------------------------------
    # Layer 1: Obstacles
    # ------------------------------------------------------------------
    lines.append('  <g id="obstacles_layer" inkscape:label="Obstacles" inkscape:groupmode="layer">')
    for i, raw_obs in enumerate(map_data.get("obstacles", [])):
        pts = [_translate(p, min_x, min_y) for p in raw_obs]
        lines.append(
            _path_elem(
                _obstacle_d(pts),
                label="obstacle",
                elem_id=f"obstacle_{i}",
                fill="#333333",
                stroke="#000000",
                stroke_width=0.3,
            )
        )
    lines.append("  </g>")

    # ------------------------------------------------------------------
    # Layer 2: Robot (spawn zones, goal zones, routes)
    # ------------------------------------------------------------------
    lines.append('  <g id="robot_layer" inkscape:label="Robot" inkscape:groupmode="layer">')

    for i, zone in enumerate(map_data.get("robot_spawn_zones", [])):
        lines.append(
            _rect_elem(
                _zone_rect(zone, min_x, min_y),
                label="robot_spawn_zone",
                elem_id=f"robot_spawn_zone_{i}",
                fill="#ffdf00",
                stroke="none",
            )
        )

    for i, zone in enumerate(map_data.get("robot_goal_zones", [])):
        lines.append(
            _rect_elem(
                _zone_rect(zone, min_x, min_y),
                label="robot_goal_zone",
                elem_id=f"robot_goal_zone_{i}",
                fill="#ff6c00",
                stroke="none",
            )
        )

    # Robot routes: write both the original and the reversed variant so that
    # convert_map produces the same doubled route list that serialize_map does.
    route_id = 0
    for route in map_data.get("robot_routes", []):
        spawn = route["spawn_id"]
        goal = route["goal_id"]
        wps = route["waypoints"]

        fwd_pts = [_translate(p, min_x, min_y) for p in wps]
        lines.append(
            _path_elem(
                _path_d(fwd_pts),
                label=f"robot_route_{spawn}_{goal}",
                elem_id=f"robot_route_{route_id}",
                stroke="#0000cc",
                stroke_width=0.4,
            )
        )
        route_id += 1

        rev_pts = list(reversed(fwd_pts))
        lines.append(
            _path_elem(
                _path_d(rev_pts),
                label=f"robot_route_{goal}_{spawn}",
                elem_id=f"robot_route_{route_id}",
                stroke="#0000cc",
                stroke_width=0.4,
            )
        )
        route_id += 1

    lines.append("  </g>")

    # ------------------------------------------------------------------
    # Layer 3: Pedestrian
    # ------------------------------------------------------------------
    lines.append('  <g id="ped_layer" inkscape:label="Pedestrian" inkscape:groupmode="layer">')

    for i, zone in enumerate(map_data.get("ped_spawn_zones", [])):
        lines.append(
            _rect_elem(
                _zone_rect(zone, min_x, min_y),
                label="ped_spawn_zone",
                elem_id=f"ped_spawn_zone_{i}",
                fill="#23ff00",
                stroke="none",
            )
        )

    for i, zone in enumerate(map_data.get("ped_goal_zones", [])):
        lines.append(
            _rect_elem(
                _zone_rect(zone, min_x, min_y),
                label="ped_goal_zone",
                elem_id=f"ped_goal_zone_{i}",
                fill="#107400",
                stroke="none",
            )
        )

    for i, zone in enumerate(map_data.get("ped_crowded_zones", [])):
        lines.append(
            _rect_elem(
                _zone_rect(zone, min_x, min_y),
                label="ped_crowded_zone",
                elem_id=f"ped_crowded_zone_{i}",
                fill="#ff0000",
                stroke="none",
                stroke_width=0.2,
            )
        )

    route_id = 0
    for route in map_data.get("ped_routes", []):
        spawn = route["spawn_id"]
        goal = route["goal_id"]
        pts = [_translate(p, min_x, min_y) for p in route["waypoints"]]
        lines.append(
            _path_elem(
                _path_d(pts),
                label=f"ped_route_{spawn}_{goal}",
                elem_id=f"ped_route_{route_id}",
                stroke="#cc0000",
                stroke_width=0.4,
            )
        )
        route_id += 1

    lines.append("  </g>")
    lines.append("</svg>")

    output_path.write_text("\n".join(lines), encoding="utf-8")

    n_robot_routes = len(map_data.get("robot_routes", []))
    print(f"SVG written to {output_path}")
    print(f"  Map size: {width:.1f} x {height:.1f}")
    print(f"  Obstacles:         {len(map_data.get('obstacles', []))}")
    print(f"  Robot spawn zones: {len(map_data.get('robot_spawn_zones', []))}")
    print(f"  Robot goal zones:  {len(map_data.get('robot_goal_zones', []))}")
    print(
        f"  Robot routes:      {n_robot_routes} (+ {n_robot_routes} reversed = {2 * n_robot_routes} total)"
    )
    print(f"  Ped spawn zones:   {len(map_data.get('ped_spawn_zones', []))}")
    print(f"  Ped goal zones:    {len(map_data.get('ped_goal_zones', []))}")
    print(f"  Ped crowded zones: {len(map_data.get('ped_crowded_zones', []))}")
    print(f"  Ped routes:        {len(map_data.get('ped_routes', []))}")


def main() -> None:
    """Entry point for the JSON→SVG map converter."""
    parser = argparse.ArgumentParser(
        description="Convert a legacy JSON map definition to SVG format."
    )
    parser.add_argument("json_path", type=Path, help="Input JSON map file.")
    parser.add_argument("svg_path", type=Path, help="Output SVG file path.")
    args = parser.parse_args()

    if not args.json_path.is_file():
        print(f"Error: JSON file not found: {args.json_path}", file=sys.stderr)
        sys.exit(1)

    with open(args.json_path, encoding="utf-8") as f:
        map_data = json.load(f)

    args.svg_path.parent.mkdir(parents=True, exist_ok=True)
    convert_json_to_svg(map_data, args.svg_path)


if __name__ == "__main__":
    main()
