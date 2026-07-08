#!/usr/bin/env python3
"""Minimal MAPF oracle diagnostic for issue #4795.

Discretizes an SVG map into a coarse occupancy grid and runs
single-agent A* / SIPP or multi-agent CBS to check route feasibility
and produce oracle path metrics.  Optional TPG (Temporal Plan Graph)
post-processing computes schedule slack and temporal dependency
information.  This is a diagnostic tool, not a benchmark claim or
production planner.

Usage (single-agent):
    uv run python scripts/tools/mapf_oracle_diagnostic.py \
        maps/svg_maps/classic_crossing.svg \
        --start 1 1 --goal 38 38 --grid-size 40

Usage (multi-agent CBS):
    uv run python scripts/tools/mapf_oracle_diagnostic.py \
        maps/svg_maps/classic_crossing.svg \
        --grid-size 40 --agents agents.json

Usage (CBS + TPG schedule post-processing):
    uv run python scripts/tools/mapf_oracle_diagnostic.py \
        maps/svg_maps/classic_crossing.svg \
        --grid-size 40 --agents agents.json --tpg

Upstream provenance:
    Algorithms inspired by SIPP (Safe Interval Path Planning, Li et al.
    ICRA 2011, DOI: 10.1109/ICRA.2011.5980306), CBS (Conflict-Based
    Search, Sharon et al., AAAI 2015), and TPG (Temporal Plan Graph,
    Phillips & Likhachev, ICAPS 2011) from
    atb033/multi_agent_path_planning under MIT license.
    This is a clean-room reimplementation, not a copy.
"""

from __future__ import annotations

import argparse
import dataclasses
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
    constraints: dict[int, set[tuple[int, int]]] | None = None,
    edge_constraints: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] | None = None,
) -> list[tuple[int, int, int]] | None:
    """SIPP-style A* search with dynamic obstacle time-windows.

    Returns a list of (row, col, time) or None if no path exists.

    Collision checks:
    - Vertex collision: the destination cell must not be occupied by a
      dynamic obstacle at the arrival time.
    - Edge collision: a dynamic obstacle must not swap positions with the
      robot (moving from the destination to the current position at the
      same time step).
    - CBS constraints: agent-specific cell-time constraints (used by
      multi-agent CBS to avoid other agents).
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
            if constraints is not None and (nr, nc) in constraints.get(next_t, set()):
                continue
            if edge_constraints is not None and ((cr, cc), (nr, nc)) in edge_constraints.get(
                ct, set()
            ):
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
# Multi-agent CBS (Conflict-Based Search)
# ---------------------------------------------------------------------------


def _load_agents(
    json_path: Path,
) -> list[dict[str, Any]]:
    """Load multi-agent definitions from a JSON file.

    Expected format::

        {
            "agents": [
                {"id": 0, "start": [0, 0], "goal": [5, 5]},
                {"id": 1, "start": [0, 5], "goal": [5, 0]},
            ]
        }

    Each agent has an id, start [row, col], and goal [row, col].
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if "agents" not in data:
        raise ValueError(f"Missing 'agents' key in {json_path}")
    agents = data["agents"]
    if not isinstance(agents, list) or len(agents) == 0:
        raise ValueError("'agents' must be a non-empty list")
    seen_ids: set[int] = set()
    for entry in agents:
        if "id" not in entry or "start" not in entry or "goal" not in entry:
            raise ValueError("Each agent needs 'id', 'start', and 'goal'")
        agent_id = int(entry["id"])
        if agent_id in seen_ids:
            raise ValueError(f"Duplicate agent id: {agent_id}")
        seen_ids.add(agent_id)
        for key in ("start", "goal"):
            pos = entry[key]
            if not isinstance(pos, list) or len(pos) != 2:
                raise ValueError(f"Agent {entry['id']}: {key} must be [row, col]")
    return agents


@dataclasses.dataclass(frozen=True)
class _CBSNode:
    """A node in the CBS constraint tree."""

    solution: dict[int, list[tuple[int, int, int]]]
    constraints: dict[int, dict[int, set[tuple[int, int]]]]
    edge_constraints: dict[int, dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]]
    cost: float
    _counter: int


@dataclasses.dataclass(frozen=True)
class _Conflict:
    """A conflict between two agents."""

    agent_a: int
    agent_b: int
    time: int
    cell: tuple[int, int]
    is_edge: bool
    cell_a: tuple[int, int] | None = None
    cell_b: tuple[int, int] | None = None


def _path_cost(path: list[tuple[int, int, int]]) -> float:
    """Sum of step costs (each move = 1.0)."""
    return float(len(path) - 1) if path else 0.0


def _find_conflict(
    solution: dict[int, list[tuple[int, int, int]]],
) -> _Conflict | None:
    """Return the first conflict between any pair of agents, or None.

    A finished agent is treated as resting on its goal cell for every timestep
    after it arrives (standard MAPF semantics: agents remain at their goal
    indefinitely). Without this, an agent whose path crosses another agent's
    goal *after* that agent has arrived would be missed, so ``_find_conflict``
    could report a physically colliding solution as conflict-free (fail-open).
    """
    agent_ids = sorted(solution.keys())
    for i, a_id in enumerate(agent_ids):
        path_a = solution[a_id]
        times_a = {t: (r, c) for r, c, t in path_a}
        last_a = max(times_a) if times_a else 0
        goal_a = times_a.get(last_a)

        for b_id in agent_ids[i + 1 :]:
            path_b = solution[b_id]
            times_b = {t: (r, c) for r, c, t in path_b}
            last_b = max(times_b) if times_b else 0
            goal_b = times_b.get(last_b)

            # Compare over the full makespan; after an agent's last recorded
            # step it holds position at its goal cell.
            horizon = max(last_a, last_b)
            for t in range(horizon + 1):
                pa = times_a.get(t, goal_a if t > last_a else None)
                pb = times_b.get(t, goal_b if t > last_b else None)
                if pa is not None and pb is not None and pa == pb:
                    return _Conflict(a_id, b_id, t, pa, False)

                # Edge (swap) conflicts only apply while both agents are still
                # moving; a resting agent has pa_now == pa_next and is excluded
                # by the pa_now != pa_next guard below.
                if t + 1 in times_a and t + 1 in times_b:
                    pa_now = times_a.get(t)
                    pa_next = times_a.get(t + 1)
                    pb_now = times_b.get(t)
                    pb_next = times_b.get(t + 1)
                    if (
                        pa_now is not None
                        and pa_next is not None
                        and pb_now is not None
                        and pb_next is not None
                        and pa_now == pb_next
                        and pa_next == pb_now
                        and pa_now != pa_next
                    ):
                        return _Conflict(a_id, b_id, t, pa_now, True, pa_now, pa_next)

    return None


def _merge_constraints(
    parent_constraints: dict[int, dict[int, set[tuple[int, int]]]],
    agent_id: int,
    time: int,
    cell: tuple[int, int],
) -> dict[int, dict[int, set[tuple[int, int]]]]:
    """Create a new constraint set with one added constraint."""
    new: dict[int, dict[int, set[tuple[int, int]]]] = {}
    for aid, c_map in parent_constraints.items():
        new[aid] = {t: set(cells) for t, cells in c_map.items()}
    if agent_id not in new:
        new[agent_id] = {}
    if time not in new[agent_id]:
        new[agent_id][time] = set()
    new[agent_id][time].add(cell)
    return new


def _build_agent_constraints(
    cbs_constraints: dict[int, dict[int, set[tuple[int, int]]]],
    agent_id: int,
) -> dict[int, set[tuple[int, int]]]:
    """Extract the constraint set for a single agent."""
    raw = cbs_constraints.get(agent_id, {})
    return {t: set(cells) for t, cells in raw.items()}


def _merge_edge_constraints(
    parent_edge_constraints: dict[int, dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]],
    agent_id: int,
    time: int,
    edge: tuple[tuple[int, int], tuple[int, int]],
) -> dict[int, dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]]:
    """Create a new edge constraint set with one added edge constraint."""
    new: dict[int, dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]] = {}
    for aid, c_map in parent_edge_constraints.items():
        new[aid] = {t: set(edges) for t, edges in c_map.items()}
    if agent_id not in new:
        new[agent_id] = {}
    if time not in new[agent_id]:
        new[agent_id][time] = set()
    new[agent_id][time].add(edge)
    return new


def _build_agent_edge_constraints(
    cbs_edge_constraints: dict[int, dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]],
    agent_id: int,
) -> dict[int, set[tuple[tuple[int, int], tuple[int, int]]]]:
    """Extract the edge constraint set for a single agent."""
    raw = cbs_edge_constraints.get(agent_id, {})
    return {t: set(edges) for t, edges in raw.items()}


def cbs_search(  # noqa: C901
    grid: list[list[int]],
    agents: list[dict[str, Any]],
    time_blocked: dict[int, set[tuple[int, int]]] | None = None,
    max_time: int = 200,
    max_nodes: int = 500,
    time_edges_blocked: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] | None = None,
) -> dict[str, Any] | None:
    """CBS (Conflict-Based Search) for multi-agent path finding.

    Returns a dict with solution info or None if infeasible.

    Algorithm: Sharon et al., "Conflict-Based Search for Optimal
    Multi-Agent Path Finding", AAAI 2015.

    Provenance: clean-room reimplementation, not a copy of upstream code.
    """
    if time_blocked is None:
        time_blocked = {}

    agent_ids: list[int] = [int(a["id"]) for a in agents]
    agent_map: dict[int, dict[str, Any]] = {int(a["id"]): a for a in agents}

    solution: dict[int, list[tuple[int, int, int]]] = {}
    for aid in agent_ids:
        a = agent_map[aid]
        start = (int(a["start"][0]), int(a["start"][1]))
        goal = (int(a["goal"][0]), int(a["goal"][1]))
        path = sipp_search(
            grid,
            start,
            goal,
            time_blocked,
            max_time,
            time_edges_blocked=time_edges_blocked,
        )
        if path is None:
            return None
        solution[aid] = path

    counter = 0
    root = _CBSNode(
        solution=solution,
        constraints={aid: {} for aid in agent_ids},
        edge_constraints={aid: {} for aid in agent_ids},
        cost=sum(_path_cost(p) for p in solution.values()),
        _counter=counter,
    )

    open_list: list[tuple[float, int, _CBSNode]] = []
    heapq.heappush(open_list, (root.cost, root._counter, root))

    nodes_expanded = 0

    while open_list:
        if nodes_expanded >= max_nodes:
            return None

        _c, _cnt, node = heapq.heappop(open_list)
        nodes_expanded += 1

        conflict = _find_conflict(node.solution)
        if conflict is None:
            return {
                "solution": node.solution,
                "cost": node.cost,
                "nodes_expanded": nodes_expanded,
                "agent_ids": agent_ids,
            }

        for agent_id in (conflict.agent_a, conflict.agent_b):
            if conflict.is_edge:
                if agent_id == conflict.agent_a:
                    edge = (conflict.cell_a, conflict.cell_b)
                else:
                    edge = (conflict.cell_b, conflict.cell_a)
                new_constraints = node.constraints
                new_edge_constraints = _merge_edge_constraints(
                    node.edge_constraints,
                    agent_id,
                    conflict.time,
                    edge,
                )
            else:
                new_constraints = _merge_constraints(
                    node.constraints,
                    agent_id,
                    conflict.time,
                    conflict.cell,
                )
                new_edge_constraints = node.edge_constraints

            agent_constraints = _build_agent_constraints(new_constraints, agent_id)
            agent_edge_constraints = _build_agent_edge_constraints(new_edge_constraints, agent_id)
            a = agent_map[agent_id]
            start = (int(a["start"][0]), int(a["start"][1]))
            goal = (int(a["goal"][0]), int(a["goal"][1]))

            new_path = sipp_search(
                grid,
                start,
                goal,
                time_blocked,
                max_time,
                time_edges_blocked=time_edges_blocked,
                constraints=agent_constraints,
                edge_constraints=agent_edge_constraints,
            )
            if new_path is None:
                continue

            new_solution = dict(node.solution)
            new_solution[agent_id] = new_path

            counter += 1
            new_cost = sum(_path_cost(p) for p in new_solution.values())
            child = _CBSNode(
                solution=new_solution,
                constraints=new_constraints,
                edge_constraints=new_edge_constraints,
                cost=new_cost,
                _counter=counter,
            )
            heapq.heappush(open_list, (child.cost, child._counter, child))

    return None


def _run_cbs_search(
    diagnostic: dict[str, Any],
    grid: list[list[int]],
    agents: list[dict[str, Any]],
    time_blocked: dict[int, set[tuple[int, int]]],
    max_time: int,
    max_nodes: int,
    dyn_obstacles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run CBS multi-agent search and fill diagnostic fields."""
    time_edges_blocked = _build_time_edges_blocked(dyn_obstacles) if dyn_obstacles else None
    result = cbs_search(
        grid,
        agents,
        time_blocked,
        max_time,
        max_nodes,
        time_edges_blocked=time_edges_blocked,
    )

    diagnostic["search_method"] = "cbs"
    diagnostic["agent_count"] = len(agents)
    diagnostic["max_time"] = max_time
    diagnostic["cbs_max_nodes"] = max_nodes

    if result is None:
        diagnostic["multi_agent_feasible"] = False
        diagnostic["diagnostic_status"] = "infeasible"
        diagnostic["agent_paths"] = None
        diagnostic["total_path_cost"] = None
        diagnostic["cbs_nodes_expanded"] = None
        return diagnostic

    diagnostic["multi_agent_feasible"] = True
    diagnostic["diagnostic_status"] = "feasible"
    diagnostic["total_path_cost"] = result["cost"]
    diagnostic["cbs_nodes_expanded"] = result["nodes_expanded"]

    agent_paths: dict[str, Any] = {}
    for aid in result["agent_ids"]:
        path = result["solution"][aid]
        agent_paths[str(aid)] = {
            "path_length": len(path),
            "path": [{"row": r, "col": c, "time": t} for r, c, t in path],
        }
    diagnostic["agent_paths"] = agent_paths
    return diagnostic


# ---------------------------------------------------------------------------
# TPG (Temporal Plan Graph) schedule post-processing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _TPGEdge:
    """A temporal dependency edge in the TPG."""

    from_agent: int
    to_agent: int
    time: int
    cell: tuple[int, int]


def _build_tpg_graph(
    solution: dict[int, list[tuple[int, int, int]]],
) -> list[_TPGEdge]:
    """Build a Temporal Plan Graph from a CBS solution.

    For each pair of agents that share a cell (but at different times),
    an edge is created from the earlier agent to the later one.  This
    captures the temporal ordering constraint: the later agent must not
    arrive before the earlier one leaves.

    Provenance: inspired by Temporal Plan Graph from
    Phillips & Likhachev, ICAPS 2011.  Clean-room reimplementation.
    """
    edges: list[_TPGEdge] = []
    agent_ids = sorted(solution.keys())

    # Build position -> time mapping per agent
    agent_pos_times: dict[int, dict[tuple[int, int], list[int]]] = {}
    for aid in agent_ids:
        pos_times: dict[tuple[int, int], list[int]] = {}
        for r, c, t in solution[aid]:
            pos_times.setdefault((r, c), []).append(t)
        agent_pos_times[aid] = pos_times

    for i, a_id in enumerate(agent_ids):
        for b_id in agent_ids[i + 1 :]:
            a_pos = agent_pos_times[a_id]
            b_pos = agent_pos_times[b_id]
            # Find shared cells
            shared = set(a_pos.keys()) & set(b_pos.keys())
            for cell in shared:
                for t_a in a_pos[cell]:
                    for t_b in b_pos[cell]:
                        if t_a < t_b:
                            edges.append(_TPGEdge(a_id, b_id, t_a, cell))
                        elif t_b < t_a:
                            edges.append(_TPGEdge(b_id, a_id, t_b, cell))
    return edges


def _compute_schedule_slack(
    solution: dict[int, list[tuple[int, int, int]]],
    tpg_edges: list[_TPGEdge],
    time_blocked: dict[int, set[tuple[int, int]]] | None = None,
) -> dict[int, float]:
    """Compute per-agent schedule slack from the TPG.

    Slack is the maximum number of time steps an agent can be uniformly
    delayed before it would violate a temporal dependency (edge) in the
    TPG or collide with a dynamic obstacle.  Higher slack means more
    flexibility for execution under kinematic imperfections.

    Returns a dict mapping agent_id -> slack (may be inf for agents
    with no downstream dependencies).
    """
    if time_blocked is None:
        time_blocked = {}

    agent_ids = sorted(solution.keys())
    makespan = max(path[-1][2] for path in solution.values()) if solution else 0

    downstream: dict[int, list[_TPGEdge]] = {aid: [] for aid in agent_ids}
    for edge in tpg_edges:
        downstream[edge.from_agent].append(edge)

    slack: dict[int, float] = {}
    for aid in agent_ids:
        path = solution[aid]
        path_times = {t: (r, c) for r, c, t in path}
        min_delay = _downstream_slack(solution, downstream, aid)
        min_delay = _obstacle_slack(path_times, time_blocked, min_delay)
        if min_delay == float("inf"):
            slack[aid] = float(makespan)
        else:
            slack[aid] = max(0.0, min_delay)

    return slack


def _downstream_slack(
    solution: dict[int, list[tuple[int, int, int]]],
    downstream: dict[int, list[_TPGEdge]],
    aid: int,
) -> float:
    """Compute the minimum delay constraint from downstream TPG edges."""
    min_delay = float("inf")
    for edge in downstream[aid]:
        to_path = solution[edge.to_agent]
        to_times = {t: (r, c) for r, c, t in to_path}
        for t_to, pos in to_times.items():
            if pos == edge.cell and t_to > edge.time:
                delay_gap = t_to - edge.time - 1
                min_delay = min(min_delay, delay_gap)
                break
    return min_delay


def _obstacle_slack(
    path_times: dict[int, tuple[int, int]],
    time_blocked: dict[int, set[tuple[int, int]]],
    current_min: float,
) -> float:
    """Update min_delay if any path position collides with a dynamic obstacle."""
    last_time = max(path_times) if path_times else 0
    for t in range(last_time + 1):
        blocked = time_blocked.get(t, set())
        pos = path_times.get(t)
        if pos is not None and pos in blocked:
            return 0.0
    return current_min


def tpg_post_process(
    solution: dict[int, list[tuple[int, int, int]]],
    time_blocked: dict[int, set[tuple[int, int]]] | None = None,
) -> dict[str, Any]:
    """Run TPG post-processing on a CBS solution and return diagnostic metrics.

    Returns a dict with:
    - ``tpg_makespan``: latest arrival time across all agents.
    - ``tpg_total_slack``: sum of per-agent schedule slack.
    - ``tpg_min_slack``: minimum slack across agents (bottleneck).
    - ``tpg_bottleneck_agent``: agent with the least slack.
    - ``tpg_slack_per_agent``: per-agent slack values.
    - ``tpg_dependency_edges``: number of temporal dependency edges.
    - ``tpg_dependency_pairs``: unique agent-pair count with dependencies.

    Provenance: inspired by Temporal Plan Graph (Phillips & Likhachev,
    ICAPS 2011).  Clean-room reimplementation for diagnostic use.
    """
    tpg_edges = _build_tpg_graph(solution)
    slack = _compute_schedule_slack(solution, tpg_edges, time_blocked)

    makespan = max(path[-1][2] for path in solution.values()) if solution else 0

    # Dependency pair count (unique unordered pairs)
    dep_pairs: set[tuple[int, int]] = set()
    for edge in tpg_edges:
        pair = (min(edge.from_agent, edge.to_agent), max(edge.from_agent, edge.to_agent))
        dep_pairs.add(pair)

    min_slack = min(slack.values()) if slack else 0.0
    bottleneck = min(slack, key=lambda a: slack[a]) if slack else None

    return {
        "tpg_enabled": True,
        "tpg_makespan": makespan,
        "tpg_total_slack": round(sum(slack.values()), 2),
        "tpg_min_slack": round(min_slack, 2),
        "tpg_bottleneck_agent": bottleneck,
        "tpg_slack_per_agent": {str(a): round(s, 2) for a, s in sorted(slack.items())},
        "tpg_dependency_edges": len(tpg_edges),
        "tpg_dependency_pairs": len(dep_pairs),
    }


def _run_cbs_search(
    diagnostic: dict[str, Any],
    grid: list[list[int]],
    agents: list[dict[str, Any]],
    time_blocked: dict[int, set[tuple[int, int]]],
    max_time: int,
    max_nodes: int,
    dyn_obstacles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run CBS multi-agent search and fill diagnostic fields."""
    time_edges_blocked = _build_time_edges_blocked(dyn_obstacles) if dyn_obstacles else None
    result = cbs_search(
        grid,
        agents,
        time_blocked,
        max_time,
        max_nodes,
        time_edges_blocked=time_edges_blocked,
    )

    diagnostic["search_method"] = "cbs"
    diagnostic["agent_count"] = len(agents)
    diagnostic["max_time"] = max_time
    diagnostic["cbs_max_nodes"] = max_nodes

    if result is None:
        diagnostic["multi_agent_feasible"] = False
        diagnostic["diagnostic_status"] = "infeasible"
        diagnostic["agent_paths"] = None
        diagnostic["total_path_cost"] = None
        diagnostic["cbs_nodes_expanded"] = None
        return diagnostic

    diagnostic["multi_agent_feasible"] = True
    diagnostic["diagnostic_status"] = "feasible"
    diagnostic["total_path_cost"] = result["cost"]
    diagnostic["cbs_nodes_expanded"] = result["nodes_expanded"]

    agent_paths: dict[str, Any] = {}
    for aid in result["agent_ids"]:
        path = result["solution"][aid]
        agent_paths[str(aid)] = {
            "path_length": len(path),
            "path": [{"row": r, "col": c, "time": t} for r, c, t in path],
        }
    diagnostic["agent_paths"] = agent_paths
    return diagnostic


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
        help="Maximum time horizon for SIPP/CBS search. Default: 200.",
    )
    parser.add_argument(
        "--agents",
        type=Path,
        default=None,
        help="JSON file with multi-agent definitions for CBS search.",
    )
    parser.add_argument(
        "--cbs-max-nodes",
        type=int,
        default=500,
        help="Maximum CBS constraint-tree nodes to expand. Default: 500.",
    )
    parser.add_argument(
        "--tpg",
        action="store_true",
        default=False,
        help="Run TPG (Temporal Plan Graph) schedule post-processing on CBS solutions.",
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


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]] | None,
    list[dict[str, Any]] | None,
    dict[int, set[tuple[int, int]]],
]:
    """Load dynamic obstacles and agents from CLI args. Return (dyn_obstacles, agents, time_blocked) or raise SystemExit."""
    dyn_obstacles: list[dict[str, Any]] | None = None
    time_blocked: dict[int, set[tuple[int, int]]] = {}
    if args.dynamic_obstacles is not None:
        if not args.dynamic_obstacles.exists():
            print(
                f"Error: dynamic obstacles file not found: {args.dynamic_obstacles}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            dyn_obstacles = _load_dynamic_obstacles(args.dynamic_obstacles)
            time_blocked = _build_time_blocked(dyn_obstacles)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)

    agents: list[dict[str, Any]] | None = None
    if args.agents is not None:
        if not args.agents.exists():
            print(
                f"Error: agents file not found: {args.agents}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        try:
            agents = _load_agents(args.agents)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1)

    return dyn_obstacles, agents, time_blocked


def main(argv: list[str] | None = None) -> int:
    """Entry point for the MAPF oracle diagnostic CLI."""
    args = _build_parser().parse_args(argv)

    if not args.map_file.exists():
        print(f"Error: map file not found: {args.map_file}", file=sys.stderr)
        return 1

    try:
        dyn_obstacles, agents, time_blocked = _load_inputs(args)
    except SystemExit as exc:
        return int(exc.code)

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

    if agents is not None:
        diagnostic = _run_cbs_search(
            diagnostic,
            grid,
            agents,
            time_blocked,
            args.max_time,
            args.cbs_max_nodes,
            dyn_obstacles=dyn_obstacles,
        )
        if args.tpg and diagnostic.get("multi_agent_feasible") and diagnostic.get("agent_paths"):
            # Build solution from diagnostic output for TPG post-processing
            solution: dict[int, list[tuple[int, int, int]]] = {}
            for aid_str, info in diagnostic["agent_paths"].items():
                solution[int(aid_str)] = [
                    (step["row"], step["col"], step["time"]) for step in info["path"]
                ]
            tpg_result = tpg_post_process(solution, time_blocked)
            diagnostic.update(tpg_result)
        elif args.tpg and not diagnostic.get("multi_agent_feasible"):
            diagnostic["tpg_enabled"] = False
            diagnostic["tpg_skipped_reason"] = "infeasible_cbs_solution"
    elif dyn_obstacles is not None:
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
