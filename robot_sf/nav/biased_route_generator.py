"""Deterministic neutral/left/right route-condition generator (issue #8033).

This module owns pure, deterministic generation of route-condition variants on
canonical multi-homotopy grids (two-corridor and doorway walls): the
``neutral`` condition plans on the unmodified grid, while the ``left`` and
``right`` conditions plan with the opposite side's free cells masked so the
route is forced through the named side.

Every generated route is verified with the merged
``route_choice_observability.v1`` contract (#7890): the classified side must
match the requested condition and the topological identity must be stable.
The contract boundary carries over unchanged: planner-route observability
only, no pedestrian-preference or social-compliance claim, and no change to
default navigation route selection.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.benchmark.route_choice_observability import (
    DIAGNOSTIC_CLAIM_BOUNDARY,
    classify_route_side,
    homotopy_identity,
)

#: Versioned record emitted by :func:`route_condition_report`.
ROUTE_CONDITION_SCHEMA_VERSION = "route_condition_report.v1"

#: Bounded route-condition vocabulary.
ROUTE_CONDITIONS = ("neutral", "left", "right")

_NEIGHBORS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


@dataclass(frozen=True)
class RouteConditionReport:
    """Deterministic verification report for the three generated variants."""

    status: str
    reason: str | None
    conditions: dict[str, dict[str, Any]]
    claim_boundary: str = DIAGNOSTIC_CLAIM_BOUNDARY

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-ready record."""
        return {
            "schema": ROUTE_CONDITION_SCHEMA_VERSION,
            "status": self.status,
            "reason": self.reason,
            "conditions": dict(self.conditions),
            "claim_boundary": self.claim_boundary,
        }


def _plan_grid(
    blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """Deterministic 8-connected A* over ``blocked`` in ``(row, col)`` order.

    Ties break by ``(f, heuristic, row, col, axis-distance)`` so identical
    inputs always produce the identical path, and equal-cost candidates hug
    the start-to-goal axis.

    Returns:
        The grid-cell path from start to goal, or ``None`` when the goal is
        unreachable.
    """
    rows, cols = blocked.shape
    for row, col in (start, goal):
        if not (0 <= row < rows and 0 <= col < cols) or blocked[row, col]:
            return None

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap: list[tuple[float, float, int, int, int]] = [
        (heuristic(start), 0, start[0], start[1], 0)
    ]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}
    while open_heap:
        _f, _h, row, col, _axis = heapq.heappop(open_heap)
        cell = (row, col)
        if cell == goal:
            path: list[tuple[int, int]] = [cell]
            while cell in came_from:
                cell = came_from[cell]
                path.append(cell)
            path.reverse()
            return path
        for d_row, d_col, step_cost in _NEIGHBORS:
            nxt = (row + d_row, col + d_col)
            n_row, n_col = nxt
            if not (0 <= n_row < rows and 0 <= n_col < cols) or blocked[n_row, n_col]:
                continue
            new_cost = cost_so_far[cell] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                came_from[nxt] = cell
                axis_distance = abs(_signed_side(nxt, start, goal))
                heapq.heappush(
                    open_heap,
                    (new_cost + heuristic(nxt), heuristic(nxt), n_row, n_col, axis_distance),
                )
    return None


def _signed_side(cell: tuple[int, int], start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Return the sign of the cell's position relative to the start-to-goal axis.

    Uses the same left-hand positive convention as the observability contract:
    positive is left of the directed axis, negative is right, zero is on the
    axis.
    """
    axis = (goal[0] - start[0], goal[1] - start[1])
    offset = (cell[0] - start[0], cell[1] - start[1])
    cross = axis[0] * offset[1] - axis[1] * offset[0]
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def _masked_grid(
    blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int], condition: str
) -> np.ndarray:
    """Return ``blocked`` with every non-condition side cell masked out.

    ``left`` blocks axis and right-side cells, ``right`` blocks axis and
    left-side cells, and ``neutral`` blocks both strict sides. The shared
    start and goal cells always stay free so every variant can connect.
    """
    allowed_sign = {"left": 1, "right": -1, "neutral": 0}.get(condition)
    if allowed_sign is None:
        return blocked
    masked = blocked.copy()
    rows, cols = masked.shape
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if masked[row, col] or cell in (start, goal):
                continue
            if _signed_side(cell, start, goal) != allowed_sign:
                masked[row, col] = True
    return masked


def generate_route_conditions(
    blocked: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> dict[str, list[tuple[int, int]] | None]:
    """Generate deterministic ``neutral``, ``left``, and ``right`` route variants.

    Args:
        blocked: Boolean occupancy grid in ``(row, col)`` order where ``True``
            is blocked.
        start: Grid start cell.
        goal: Grid goal cell.

    Returns:
        A mapping from condition name to the planned grid-cell path, or
        ``None`` when that condition has no feasible route on the given map.
    """
    if blocked.ndim != 2 or blocked.dtype != np.dtype(bool):
        raise ValueError("blocked must be a 2-D boolean numpy array")
    variants: dict[str, list[tuple[int, int]] | None] = {}
    for condition in ROUTE_CONDITIONS:
        variants[condition] = _plan_grid(_masked_grid(blocked, start, goal, condition), start, goal)
    return variants


def route_condition_report(
    blocked: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    world_start: tuple[float, float] | None = None,
    world_goal: tuple[float, float] | None = None,
) -> RouteConditionReport:
    """Generate the three variants and verify them with the #7890 contract.

    The grid ``(row, col)`` cells double as world coordinates for the side
    classification unless explicit world coordinates are supplied for the
    start/goal axis. A condition passes when its classified side matches the
    requested condition (neutral allows ``neutral``) and its topological
    identity is stable under a deterministic cell-repetition replan.

    Returns:
        A :class:`RouteConditionReport` with per-condition verification and
        the inherited observability claim boundary.
    """
    if world_start is None:
        world_start = (float(start[0]), float(start[1]))
    if world_goal is None:
        world_goal = (float(goal[0]), float(goal[1]))
    variants = generate_route_conditions(blocked, start, goal)
    conditions: dict[str, dict[str, Any]] = {}
    for condition in ROUTE_CONDITIONS:
        path = variants.get(condition)
        if path is None:
            conditions[condition] = {
                "status": "unavailable",
                "reason": "no_feasible_route",
                "side": None,
                "identity": None,
            }
            continue
        world_path = [(float(row), float(col)) for row, col in path]
        side_report = classify_route_side(world_path, start=world_start, goal=world_goal)
        observation = homotopy_identity(path, blocked)
        replan = [(row, col) for row, col in path for _ in range(2)]
        replan_observation = homotopy_identity(replan, blocked)
        identity_stable = (
            observation.identity is not None and observation.identity == replan_observation.identity
        )
        expected_side = "neutral" if condition == "neutral" else condition
        verified = side_report.side == expected_side and identity_stable
        conditions[condition] = {
            "status": "verified" if verified else "failed",
            "reason": None
            if verified
            else (
                f"side_mismatch:{side_report.side}"
                if side_report.side != expected_side
                else "identity_unstable"
            ),
            "side": side_report.side,
            "identity": observation.identity,
        }
    status = "verified" if all(c["status"] == "verified" for c in conditions.values()) else "failed"
    return RouteConditionReport(status=status, reason=None, conditions=conditions)


def corridor_map() -> np.ndarray:
    """Return the canonical three-corridor grid.

    Barrier segments at columns 6-8 leave three east-west corridors: rows 0-1
    (left of the start-to-goal axis), row 4 (on-axis neutral), and rows 7-8
    (right). All three route conditions are feasible.
    """
    grid = np.zeros((9, 15), dtype=bool)
    grid[2:4, 6:9] = True
    grid[5:7, 6:9] = True
    return grid


def doorway_map() -> np.ndarray:
    """Return the canonical doorway grid (vertical wall, three openings)."""
    grid = np.zeros((9, 15), dtype=bool)
    grid[:, 7] = True
    grid[1, 7] = False
    grid[4, 7] = False
    grid[7, 7] = False
    return grid
