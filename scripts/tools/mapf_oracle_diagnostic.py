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
import heapq
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
# Dynamic obstacle loading and SIPP search
# ---------------------------------------------------------------------------


def _load_dynamic_obstacles(
    json_path: Path,
) -> list[dict[str, Any]]:
    """Load dynamic obstacle trajectories from a JSON file.

    Expected format::

        {
            "dynamic_obstacles": [
                {"id": 0, "trajectory": [[3, 5], [3, 6], [3, 7], ...]},
                {"id": 1, "trajectory": [[10, 10], [10, 11], ...]},
            ]
        }

    Each trajectory entry is a list of [row, col] positions indexed by
    time step.  Validation errors raise ``ValueError``.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "dynamic_obstacles" not in data:
        raise ValueError(f"Missing 'dynamic_obstacles' key in {json_path}")
    obs = data["dynamic_obstacles"]
    if not isinstance(obs, list):
        raise ValueError("'dynamic_obstacles' must be a list")
    for entry in obs:
        if "id" not in entry or "trajectory" not in entry:
            raise ValueError("Each obstacle needs 'id' and 'trajectory'")
        traj = entry["trajectory"]
        if not isinstance(traj, list) or len(traj) == 0:
            raise ValueError(f"Obstacle {entry['id']}: trajectory must be non-empty")
        for pos in traj:
            if not isinstance(pos, list) or len(pos) != 2:
                raise ValueError(f"Obstacle {entry['id']}: each position must be [row, col]")
    return obs


def _build_time_blocked(
    obstacles: list[dict[str, Any]],
) -> dict[int, set[tuple[int, int]]]:
    """Convert obstacle trajectories to time-indexed blocked positions."""
    result: dict[int, set[tuple[int, int]]] = {}
    for entry in obstacles:
        for t, pos in enumerate(entry["trajectory"]):
            key = (pos[0], pos[1])
            if t not in result:
                result[t] = set()
            result[t].add(key)
    return result


def _build_time_edges_blocked(
    obstacles: list[dict[str, Any]],
) -> dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]:
    """Convert obstacle trajectories to time-indexed traversed edges.

    Keyed by the departure time ``t``; each entry holds the ``(from, to)``
    cell transitions that obstacles make between ``t`` and ``t + 1``.  Used to
    detect swap collisions against the *specific* moving obstacle rather than
    inferring one from two independent occupancy facts.
    """
    result: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
    for entry in obstacles:
        traj = entry["trajectory"]
        for t in range(len(traj) - 1):
            frm = (traj[t][0], traj[t][1])
            to = (traj[t + 1][0], traj[t + 1][1])
            if frm == to:
                continue
            result.setdefault(t, set()).add((frm, to))
    return result


def _has_collision(
    nr: int,
    nc: int,
    cr: int,
    cc: int,
    ct: int,
    next_t: int,
    time_blocked: dict[int, set[tuple[int, int]]],
    time_edges_blocked: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] | None = None,
) -> bool:
    """Check vertex and edge (swap) collisions for a SIPP move.

    Vertex collision: the destination cell is occupied at arrival time.
    Edge collision: a single obstacle traverses the reverse edge in the same
    interval (obstacle ``(nr, nc) -> (cr, cc)`` while the robot moves
    ``(cr, cc) -> (nr, nc)``), i.e. a true position swap.
    """
    blocked_next = time_blocked.get(next_t, set())
    if (nr, nc) in blocked_next:
        return True
    if time_edges_blocked is not None:
        if ((nr, nc), (cr, cc)) in time_edges_blocked.get(ct, set()):
            return True
    return False


def sipp_search(  # noqa: C901
    grid: list[list[int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    time_blocked: dict[int, set[tuple[int, int]]],
    max_time: int = 200,
    time_edges_blocked: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] | None = None,
) -> list[tuple[int, int, int]] | None:
    """SIPP-style A* search with dynamic obstacle time-windows.

    Returns a list of (row, col, time) or None if no path exists.

    Collision checks:
    - Vertex collision: the destination cell must not be occupied by a
      dynamic obstacle at the arrival time.
    - Edge collision: a dynamic obstacle must not swap positions with the
      robot (moving from the destination to the current position at the
      same time step).
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < rows and 0 <= sc < cols):
        return None
    if not (0 <= gr < rows and 0 <= gc < cols):
        return None
    if grid[sr][sc] == 1 or grid[gr][gc] == 1:
        return None

    start_state = (sr, sc, 0)
    goal_pos = (gr, gc)

    open_set: list[tuple[float, int, int, int]] = []
    heapq.heappush(open_set, (_heuristic(sr, sc, gr, gc), 0, sr, sc))

    g_score: dict[tuple[int, int, int], float] = {start_state: 0.0}
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {
        start_state: None,
    }

    # 4-connected moves plus wait
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))

    while open_set:
        _f, ct, cr, cc = heapq.heappop(open_set)

        if (cr, cc) == goal_pos:
            path: list[tuple[int, int, int]] = []
            state: tuple[int, int, int] | None = (cr, cc, ct)
            while state is not None:
                path.append(state)
                state = came_from[state]
            path.reverse()
            return path

        if ct >= max_time:
            continue

        next_t = ct + 1
        for dr, dc in moves:
            nr, nc = cr + dr, cc + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue
            if _has_collision(nr, nc, cr, cc, ct, next_t, time_blocked, time_edges_blocked):
                continue

            new_state = (nr, nc, next_t)
            tentative_g = g_score[(cr, cc, ct)] + 1.0
            if tentative_g < g_score.get(new_state, float("inf")):
                g_score[new_state] = tentative_g
                f_score = tentative_g + _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_set, (f_score, next_t, nr, nc))
                came_from[new_state] = (cr, cc, ct)

    return None


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
    parser.add_argument(
        "--dynamic-obstacles",
        type=Path,
        default=None,
        help="JSON file with dynamic obstacle trajectories for SIPP search.",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=200,
        help="Maximum time horizon for SIPP search. Default: 200.",
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


def _base_diagnostic(
    map_file: Path,
    width: float,
    height: float,
    grid_size: int,
    obstacles: list[tuple[float, float, float, float]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> dict[str, Any]:
    """Build the common diagnostic fields shared by both search modes."""
    occupied = sum(
        cell for row in _build_occupancy_grid(obstacles, width, height, grid_size) for cell in row
    )
    total = grid_size * grid_size
    return {
        "tool": "mapf_oracle_diagnostic",
        "issue": 4795,
        "map_file": map_file.as_posix(),
        "map_dimensions": {"width": width, "height": height},
        "grid_dimensions": {"rows": grid_size, "cols": grid_size},
        "occupancy_ratio": round(occupied / total, 4) if total > 0 else 0.0,
        "obstacle_count": len(obstacles),
        "start_grid": {"row": start[0], "col": start[1]},
        "goal_grid": {"row": goal[0], "col": goal[1]},
    }


def _run_static_search(
    diagnostic: dict[str, Any],
    grid: list[list[int]],
    start_pos: tuple[int, int],
    goal_pos: tuple[int, int],
) -> dict[str, Any]:
    """Run static A* and fill diagnostic fields."""
    path = astar_search(grid, start_pos, goal_pos)
    diagnostic["search_method"] = "astar_static"
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
        diagnostic["oracle_wait_steps"] = 0
    return diagnostic


def _run_sipp_search(
    diagnostic: dict[str, Any],
    grid: list[list[int]],
    start_pos: tuple[int, int],
    goal_pos: tuple[int, int],
    time_blocked: dict[int, set[tuple[int, int]]],
    dyn_obstacles: list[dict[str, Any]],
    max_time: int,
) -> dict[str, Any]:
    """Run SIPP search and fill diagnostic fields."""
    time_edges_blocked = _build_time_edges_blocked(dyn_obstacles)
    sipp_path = sipp_search(
        grid,
        start_pos,
        goal_pos,
        time_blocked,
        max_time,
        time_edges_blocked=time_edges_blocked,
    )
    diagnostic["search_method"] = "sipp"
    diagnostic["dynamic_obstacle_count"] = len(dyn_obstacles)
    diagnostic["max_time"] = max_time
    if sipp_path is None:
        diagnostic["mapf_feasible"] = False
        diagnostic["diagnostic_status"] = "infeasible"
        diagnostic["oracle_path_length"] = None
        diagnostic["oracle_path"] = None
        diagnostic["oracle_arrival_time"] = None
        diagnostic["oracle_wait_steps"] = 0
    else:
        wait_steps = sum(
            1
            for i in range(1, len(sipp_path))
            if sipp_path[i][0] == sipp_path[i - 1][0] and sipp_path[i][1] == sipp_path[i - 1][1]
        )
        diagnostic["mapf_feasible"] = True
        diagnostic["diagnostic_status"] = "feasible"
        diagnostic["oracle_path_length"] = len(sipp_path)
        diagnostic["oracle_path"] = [{"row": r, "col": c, "time": t} for r, c, t in sipp_path]
        diagnostic["oracle_arrival_time"] = sipp_path[-1][2]
        diagnostic["oracle_wait_steps"] = wait_steps
    return diagnostic


def main(argv: list[str] | None = None) -> int:
    """Entry point for the MAPF oracle diagnostic CLI."""
    args = _build_parser().parse_args(argv)

    if not args.map_file.exists():
        print(f"Error: map file not found: {args.map_file}", file=sys.stderr)
        return 1

    # Load dynamic obstacles if provided
    dyn_obstacles: list[dict[str, Any]] | None = None
    time_blocked: dict[int, set[tuple[int, int]]] = {}
    if args.dynamic_obstacles is not None:
        if not args.dynamic_obstacles.exists():
            print(
                f"Error: dynamic obstacles file not found: {args.dynamic_obstacles}",
                file=sys.stderr,
            )
            return 1
        try:
            dyn_obstacles = _load_dynamic_obstacles(args.dynamic_obstacles)
            time_blocked = _build_time_blocked(dyn_obstacles)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
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

    cell_w = width / grid_size
    cell_h = height / grid_size

    sx, sy = (0.0, 0.0) if args.start is None else args.start
    gx, gy = (width, height) if args.goal is None else args.goal

    start_col = min(max(int(sx / cell_w), 0), grid_size - 1)
    start_row = min(max(int(sy / cell_h), 0), grid_size - 1)
    goal_col = min(max(int(gx / cell_w), 0), grid_size - 1)
    goal_row = min(max(int(gy / cell_h), 0), grid_size - 1)

    start_pos = (start_row, start_col)
    goal_pos = (goal_row, goal_col)

    diagnostic = _base_diagnostic(
        args.map_file,
        width,
        height,
        grid_size,
        obstacles,
        start_pos,
        goal_pos,
    )

    if dyn_obstacles is not None:
        diagnostic = _run_sipp_search(
            diagnostic,
            grid,
            start_pos,
            goal_pos,
            time_blocked,
            dyn_obstacles,
            args.max_time,
        )
    else:
        diagnostic = _run_static_search(diagnostic, grid, start_pos, goal_pos)

    output = json.dumps(diagnostic, indent=2, sort_keys=False) + "\n"
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
