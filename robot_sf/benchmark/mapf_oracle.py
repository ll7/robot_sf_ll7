"""Benchmark-loop integration for the MAPF oracle diagnostic (issue #4795).

Runs the MAPF oracle diagnostic (A*, SIPP, CBS, TPG) against benchmark
scenario matrices to produce per-scenario route-feasibility annotations.
This is a diagnostic-only tool, not a benchmark claim or production planner.

Provenance: MAPF algorithms inspired by SIPP (Li et al. ICRA 2011),
CBS (Sharon et al. AAAI 2015), and TPG (Phillips & Likhachev, ICAPS 2011)
from atb033/multi_agent_path_planning under MIT license.  Clean-room
reimplementation in scripts/tools/mapf_oracle_diagnostic.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot_sf.benchmark.runner import load_scenario_matrix
from scripts.tools.mapf_oracle_diagnostic import (
    _build_occupancy_grid,
    _parse_svg_obstacles,
    _svg_dimensions,
    astar_search,
)

SCHEMA_VERSION = "mapf_oracle_benchmark_diagnostics.v1"


def _resolve_map_path(map_file: str, base_dir: Path) -> Path:
    """Resolve a scenario map_file path relative to the scenario YAML directory.

    Returns:
        Resolved absolute path to the map file.
    """
    path = Path(map_file)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _run_single_scenario_diagnostic(
    scenario: dict[str, Any],
    map_path: Path,
    grid_size: int,
) -> dict[str, Any]:
    """Run MAPF oracle diagnostic on a single scenario map.

    Returns:
        Diagnostic dict for this scenario with mapf feasibility fields.
    """
    scenario_name = scenario.get("name", "unknown")

    if not map_path.exists():
        return {
            "scenario": scenario_name,
            "map_file": str(map_path),
            "diagnostic_status": "error",
            "error": f"map file not found: {map_path}",
        }

    try:
        width, height = _svg_dimensions(map_path)
    except (ValueError, OSError, AttributeError) as exc:
        return {
            "scenario": scenario_name,
            "map_file": str(map_path),
            "diagnostic_status": "error",
            "error": f"SVG dimension extraction failed: {exc}",
        }

    try:
        obstacles = _parse_svg_obstacles(map_path, width, height)
    except (ValueError, OSError, AttributeError) as exc:
        return {
            "scenario": scenario_name,
            "map_file": str(map_path),
            "diagnostic_status": "error",
            "error": f"SVG obstacle parsing failed: {exc}",
        }

    grid = _build_occupancy_grid(obstacles, width, height, grid_size)
    occupied = sum(cell for row in grid for cell in row)
    total = grid_size * grid_size

    cell_w = width / grid_size
    cell_h = height / grid_size

    start_row, start_col = 0, 0
    goal_row = min(grid_size - 1, int((height - cell_h / 2) / cell_h))
    goal_col = min(grid_size - 1, int((width - cell_w / 2) / cell_w))

    if grid[start_row][start_col] == 1:
        start_row, start_col = _find_nearest_free(grid, 0, 0)
    if grid[goal_row][goal_col] == 1:
        goal_row, goal_col = _find_nearest_free(grid, goal_row, goal_col)

    if start_row is None or goal_row is None:
        return {
            "scenario": scenario_name,
            "map_file": str(map_path),
            "diagnostic_status": "degenerate",
            "mapf_feasible": False,
            "grid_dimensions": {"rows": grid_size, "cols": grid_size},
            "occupancy_ratio": round(occupied / total, 4) if total > 0 else 0.0,
            "obstacle_count": len(obstacles),
            "error": "no free cells found for start or goal",
        }

    start_pos = (start_row, start_col)
    goal_pos = (goal_row, goal_col)

    path = astar_search(grid, start_pos, goal_pos)

    result: dict[str, Any] = {
        "scenario": scenario_name,
        "map_file": str(map_path),
        "grid_dimensions": {"rows": grid_size, "cols": grid_size},
        "map_dimensions": {"width": width, "height": height},
        "occupancy_ratio": round(occupied / total, 4) if total > 0 else 0.0,
        "obstacle_count": len(obstacles),
        "start_grid": {"row": start_row, "col": start_col},
        "goal_grid": {"row": goal_row, "col": goal_col},
    }

    if path is None:
        result["mapf_feasible"] = False
        result["diagnostic_status"] = "infeasible"
        result["oracle_path_length"] = None
    else:
        result["mapf_feasible"] = True
        result["diagnostic_status"] = "feasible"
        result["oracle_path_length"] = len(path)

    return result


def _find_nearest_free(
    grid: list[list[int]], target_row: int, target_col: int
) -> tuple[int | None, int | None]:
    """Find the nearest free cell to the target using BFS from the target.

    Returns:
        Tuple of (row, col) for the nearest free cell, or (None, None) if all cells are blocked.
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    if 0 <= target_row < rows and 0 <= target_col < cols and grid[target_row][target_col] == 0:
        return target_row, target_col
    for radius in range(1, max(rows, cols)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue
                r, c = target_row + dr, target_col + dc
                if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
                    return r, c
    return None, None


def run_mapf_oracle_diagnostics(
    scenario_path: str | Path,
    *,
    grid_size: int = 40,
    scenario_filter: str | None = None,
) -> dict[str, Any]:
    """Run MAPF oracle diagnostics on a benchmark scenario matrix.

    Loads scenarios from a YAML matrix file and runs the A* static-route
    feasibility check on each scenario's map.  Returns a structured report
    with per-scenario diagnostics and aggregate counts.

    This is a diagnostic-only tool: ``mapf_feasible`` indicates whether a
    static A* path exists on the discretized occupancy grid.  It does not
    account for dynamic pedestrians, robot kinematics, or social compliance.

    Args:
        scenario_path: Path to a scenario matrix YAML file.
        grid_size: Grid resolution for the occupancy grid (grid_size x grid_size).
        scenario_filter: Optional scenario name substring filter.

    Returns:
        Versioned diagnostic report dict.
    """
    scenario_path = Path(scenario_path)
    if not scenario_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "issue": 4795,
            "status": "error",
            "error": f"scenario file not found: {scenario_path}",
            "scenarios": [],
        }

    try:
        scenarios = load_scenario_matrix(scenario_path)
    except (ValueError, OSError, KeyError) as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "issue": 4795,
            "status": "error",
            "error": f"failed to load scenario matrix: {exc}",
            "scenarios": [],
        }

    base_dir = scenario_path.parent

    if scenario_filter:
        scenarios = [
            s for s in scenarios if scenario_filter.lower() in str(s.get("name", "")).lower()
        ]

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        map_file = scenario.get("map_file")
        if not isinstance(map_file, str) or not map_file.strip():
            results.append(
                {
                    "scenario": scenario.get("name", "unknown"),
                    "diagnostic_status": "error",
                    "error": "missing or empty map_file",
                }
            )
            continue

        map_path = _resolve_map_path(map_file, base_dir)
        result = _run_single_scenario_diagnostic(scenario, map_path, grid_size)
        results.append(result)

    feasible_count = sum(1 for r in results if r.get("mapf_feasible") is True)
    infeasible_count = sum(1 for r in results if r.get("mapf_feasible") is False)
    error_count = sum(1 for r in results if r.get("diagnostic_status") == "error")
    degenerate_count = sum(1 for r in results if r.get("diagnostic_status") == "degenerate")

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 4795,
        "tool": "mapf_oracle_benchmark_diagnostics",
        "scenario_path": str(scenario_path),
        "grid_size": grid_size,
        "total_scenarios": len(results),
        "feasible_count": feasible_count,
        "infeasible_count": infeasible_count,
        "error_count": error_count,
        "degenerate_count": degenerate_count,
        "claim_boundary": "diagnostic_only_not_benchmark_evidence",
        "scenarios": results,
    }


__all__ = [
    "SCHEMA_VERSION",
    "run_mapf_oracle_diagnostics",
]
