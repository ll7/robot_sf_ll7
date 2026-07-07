#!/usr/bin/env python3
"""Minimal MAPF oracle diagnostic for issue #4795.

Discretizes an SVG map into a coarse occupancy grid and runs a
single-agent A* search to check route feasibility and produce
oracle path metrics.  This is a diagnostic tool, not a benchmark
claim or production planner.

Usage:
    uv run python scripts/tools/mapf_oracle_diagnostic.py \
        maps/svg_maps/classic_crossing.svg \
        --start 1 1 --goal 38 38 --grid-size 40

Upstream provenance:
    Algorithm inspired by SIPP (Safe Interval Path Planning, Li et al.
    ICRA 2011, DOI: 10.1109/ICRA.2011.5980306) from
    atb033/multi_agent_path_planning under MIT license.
    This is a clean-room reimplementation, not a copy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SVG obstacle extraction
# ---------------------------------------------------------------------------


def _parse_svg_obstacles(
    svg_path: Path, width: float, height: float
) -> list[tuple[float, float, float, float]]:
    """Return list of (x, y, w, h) for obstacle rectangles in the SVG."""
    text = svg_path.read_text(encoding="utf-8")

    # Match <rect ... /> elements that carry inkscape:label="obstacle"
    # We handle both inline and block-style rect elements.
    rect_pattern = re.compile(
        r"<rect\s(.*?)\s*/?>",
        re.DOTALL,
    )
    attrs: dict[str, str] = {}
    obstacles: list[tuple[float, float, float, float]] = []

    for match in rect_pattern.finditer(text):
        raw = match.group(1)
        attrs.clear()
        for kv in re.findall(r'(\S+?)="([^"]*)"', raw):
            attrs[kv[0]] = kv[1]

        # Only count rects with inkscape:label="obstacle"
        if attrs.get("inkscape:label") != "obstacle":
            continue

        x = float(attrs.get("x", 0))
        y = float(attrs.get("y", 0))
        w = float(attrs.get("width", 0))
        h = float(attrs.get("height", 0))

        if w > 0 and h > 0:
            obstacles.append((x, y, w, h))

    return obstacles


def _build_occupancy_grid(
    obstacles: list[tuple[float, float, float, float]],
    map_width: float,
    map_height: float,
    grid_size: int,
) -> list[list[int]]:
    """Build a grid_size x grid_size occupancy grid (0=free, 1=occupied)."""
    cell_w = map_width / grid_size
    cell_h = map_height / grid_size

    grid: list[list[int]] = [[0] * grid_size for _ in range(grid_size)]

    for ox, oy, ow, oh in obstacles:
        for row in range(grid_size):
            for col in range(grid_size):
                # Cell top-left and bottom-right in SVG coords
                c_left = col * cell_w
                c_top = row * cell_h
                c_right = c_left + cell_w
                c_bottom = c_top + cell_h

                # Check overlap between cell and obstacle rectangle
                if ox < c_right and ox + ow > c_left and oy < c_bottom and oy + oh > c_top:
                    grid[row][col] = 1

    return grid


# ---------------------------------------------------------------------------
# A* pathfinding (static-obstacle variant of SIPP)
# ---------------------------------------------------------------------------


def _heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
    """Manhattan distance heuristic."""
    return abs(r1 - r2) + abs(c1 - c2)


def astar_search(  # noqa: C901
    grid: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """A* search on a binary occupancy grid. Returns path or None."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    sr, sc = start
    gr, gc = goal

    # Bounds and obstacle checks
    if not (0 <= sr < rows and 0 <= sc < cols):
        return None
    if not (0 <= gr < rows and 0 <= gc < cols):
        return None
    if grid[sr][sc] == 1 or grid[gr][gc] == 1:
        return None

    # Priority queue via sorted list (small enough for diagnostic use)
    import heapq

    open_set: list[tuple[float, int, int]] = []
    heapq.heappush(open_set, (_heuristic(sr, sc, gr, gc), sr, sc))

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    # 4-connected moves
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))

    while open_set:
        _f, cr, cc = heapq.heappop(open_set)

        if (cr, cc) == goal:
            # Reconstruct path
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = goal
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path

        for dr, dc in moves:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue

            tentative_g = g_score[(cr, cc)] + 1.0
            if tentative_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative_g
                f_score = tentative_g + _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_set, (f_score, nr, nc))
                came_from[(nr, nc)] = (cr, cc)

    return None  # No path found


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "map_file",
        type=Path,
        help="Path to an SVG map file (e.g. maps/svg_maps/classic_crossing.svg).",
    )
    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Start position in SVG coordinates. Defaults to (0,0) corner if omitted.",
    )
    parser.add_argument(
        "--goal",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Goal position in SVG coordinates. Defaults to opposite corner if omitted.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=40,
        help="Grid dimensions (grid_size x grid_size). Default: 40.",
    )
    return parser


def _svg_dimensions(svg_path: Path) -> tuple[float, float]:
    """Extract width and height from the SVG root element."""
    text = svg_path.read_text(encoding="utf-8")
    w_match = re.search(r'<svg[^>]*\bwidth="([^"]*)"', text)
    h_match = re.search(r'<svg[^>]*\bheight="([^"]*)"', text)
    width = float(w_match.group(1)) if w_match else 40.0
    height = float(h_match.group(1)) if h_match else 40.0
    return width, height


def main(argv: list[str] | None = None) -> int:
    """Entry point for the MAPF oracle diagnostic CLI."""
    args = _build_parser().parse_args(argv)

    if not args.map_file.exists():
        print(f"Error: map file not found: {args.map_file}", file=sys.stderr)
        return 1

    width, height = _svg_dimensions(args.map_file)
    obstacles = _parse_svg_obstacles(args.map_file, width, height)

    if not obstacles:
        print(
            f"Warning: no obstacles found in {args.map_file}. The map may use a different format.",
            file=sys.stderr,
        )

    grid_size = args.grid_size
    grid = _build_occupancy_grid(obstacles, width, height, grid_size)

    # Compute cell size for coordinate conversion
    cell_w = width / grid_size
    cell_h = height / grid_size

    # Default start/goal to corners
    if args.start is None:
        sx, sy = 0.0, 0.0
    else:
        sx, sy = args.start

    if args.goal is None:
        gx, gy = width, height
    else:
        gx, gy = args.goal

    # Convert SVG coords to grid coords (row, col)
    start_col = min(max(int(sx / cell_w), 0), grid_size - 1)
    start_row = min(max(int(sy / cell_h), 0), grid_size - 1)
    goal_col = min(max(int(gx / cell_w), 0), grid_size - 1)
    goal_row = min(max(int(gy / cell_h), 0), grid_size - 1)

    start_pos = (start_row, start_col)
    goal_pos = (goal_row, goal_col)

    # Run A*
    path = astar_search(grid, start_pos, goal_pos)

    occupied = sum(cell for row in grid for cell in row)
    total = grid_size * grid_size

    # Build diagnostic output
    diagnostic: dict[str, Any] = {
        "tool": "mapf_oracle_diagnostic",
        "issue": 4795,
        "map_file": args.map_file.as_posix(),
        "map_dimensions": {"width": width, "height": height},
        "grid_dimensions": {"rows": grid_size, "cols": grid_size},
        "occupancy_ratio": round(occupied / total, 4) if total > 0 else 0.0,
        "obstacle_count": len(obstacles),
        "start_grid": {"row": start_row, "col": start_col},
        "goal_grid": {"row": goal_row, "col": goal_col},
    }

    if path is None:
        diagnostic["mapf_feasible"] = False
        diagnostic["diagnostic_status"] = "infeasible"
        diagnostic["oracle_path_length"] = None
        diagnostic["oracle_path"] = None
        diagnostic["oracle_wait_steps"] = 0
    else:
        diagnostic["mapf_feasible"] = True
        diagnostic["diagnostic_status"] = "feasible"
        diagnostic["oracle_path_length"] = len(path)
        diagnostic["oracle_path"] = [{"row": r, "col": c} for r, c in path]
        diagnostic["oracle_wait_steps"] = 0  # No waiting in static variant

    output = json.dumps(diagnostic, indent=2, sort_keys=False) + "\n"
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
