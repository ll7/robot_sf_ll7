"""Tests for scripts/tools/mapf_oracle_diagnostic.py.

These tests validate the MAPF oracle diagnostic tool using in-memory
grids and temporary SVG files. They do not require the Robot SF runtime,
CARLA, or GPU access.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.tools.mapf_oracle_diagnostic import (
    _build_occupancy_grid,
    _build_time_blocked,
    _load_dynamic_obstacles,
    _parse_svg_obstacles,
    _svg_dimensions,
    astar_search,
    main,
    sipp_search,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_svg(width: float, height: float, rects: list[dict]) -> Path:
    """Write a minimal SVG with obstacle rects and return the path."""
    lines = [
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">',
    ]
    for r in rects:
        label = r.get("label", "")
        attrs = f'x="{r["x"]}" y="{r["y"]}" width="{r["w"]}" height="{r["h"]}"'
        if label:
            attrs += f' inkscape:label="{label}"'
        lines.append(f"  <rect {attrs} />")
    lines.append("</svg>")

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".svg", delete=False, encoding="utf-8")
    tmp.write("\n".join(lines))
    tmp.close()
    return Path(tmp.name)


def _make_dynamic_obstacles_json(obstacles: list[dict]) -> Path:
    """Write a dynamic obstacles JSON file and return the path."""
    data = {"dynamic_obstacles": obstacles}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# SVG parsing
# ---------------------------------------------------------------------------


class TestSvgDimensions:
    """SVG dimension extraction tests."""

    def test_basic_dimensions(self) -> None:
        path = _make_svg(40.0, 40.0, [])
        try:
            w, h = _svg_dimensions(path)
            assert w == 40.0
            assert h == 40.0
        finally:
            path.unlink()

    def test_non_square(self) -> None:
        path = _make_svg(100.0, 50.0, [])
        try:
            w, h = _svg_dimensions(path)
            assert w == 100.0
            assert h == 50.0
        finally:
            path.unlink()

    def test_defaults_when_missing(self) -> None:
        """Missing width/height defaults to 40."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".svg", delete=False, encoding="utf-8")
        tmp.write("<svg></svg>")
        tmp.close()
        path = Path(tmp.name)
        try:
            w, h = _svg_dimensions(path)
            assert w == 40.0
            assert h == 40.0
        finally:
            path.unlink()


class TestParseSvgObstacles:
    """SVG obstacle parsing tests."""

    def test_no_obstacles(self) -> None:
        path = _make_svg(40.0, 40.0, [])
        try:
            result = _parse_svg_obstacles(path, 40.0, 40.0)
            assert result == []
        finally:
            path.unlink()

    def test_single_obstacle(self) -> None:
        path = _make_svg(
            40.0,
            40.0,
            [
                {"x": 10, "y": 10, "w": 5, "h": 5, "label": "obstacle"},
            ],
        )
        try:
            result = _parse_svg_obstacles(path, 40.0, 40.0)
            assert len(result) == 1
            assert result[0] == (10.0, 10.0, 5.0, 5.0)
        finally:
            path.unlink()

    def test_ignores_non_obstacle_rects(self) -> None:
        path = _make_svg(
            40.0,
            40.0,
            [
                {"x": 0, "y": 0, "w": 10, "h": 10, "label": "obstacle"},
                {"x": 5, "y": 5, "w": 3, "h": 3},  # no inkscape:label
            ],
        )
        try:
            result = _parse_svg_obstacles(path, 40.0, 40.0)
            assert len(result) == 1
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------


class TestBuildOccupancyGrid:
    """Occupancy grid construction tests."""

    def test_empty_grid(self) -> None:
        grid = _build_occupancy_grid([], 10.0, 10.0, 10)
        assert len(grid) == 10
        assert all(cell == 0 for row in grid for cell in row)

    def test_full_obstacle(self) -> None:
        """One obstacle covering the entire map."""
        obstacles = [(0.0, 0.0, 10.0, 10.0)]
        grid = _build_occupancy_grid(obstacles, 10.0, 10.0, 5)
        assert all(cell == 1 for row in grid for cell in row)

    def test_corner_obstacle(self) -> None:
        """Small obstacle in the top-left corner."""
        obstacles = [(0.0, 0.0, 2.0, 2.0)]
        grid = _build_occupancy_grid(obstacles, 10.0, 10.0, 10)
        # Cells (0,0), (0,1), (1,0), (1,1) should be occupied
        assert grid[0][0] == 1
        assert grid[0][1] == 1
        assert grid[1][0] == 1
        assert grid[1][1] == 1
        assert grid[2][2] == 0  # Outside obstacle


# ---------------------------------------------------------------------------
# A* pathfinding
# ---------------------------------------------------------------------------


class TestAstarSearch:
    """A* pathfinding tests."""

    def test_trivial_path(self) -> None:
        grid = [[0, 0], [0, 0]]
        path = astar_search(grid, (0, 0), (1, 1))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (1, 1)
        assert len(path) == 3  # (0,0) -> (0,1) -> (1,1) or similar

    def test_blocked_path(self) -> None:
        grid = [[0, 1, 0], [0, 1, 0], [0, 0, 0]]
        path = astar_search(grid, (0, 0), (0, 2))
        # Must go around the wall
        assert path is not None
        assert len(path) == 7  # (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
        # Actually: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
        assert len(path) == 7

    def test_completely_blocked(self) -> None:
        grid = [[0, 1], [0, 1]]
        path = astar_search(grid, (0, 0), (0, 1))
        assert path is None  # Goal is obstacle

    def test_start_on_obstacle(self) -> None:
        grid = [[1, 0], [0, 0]]
        path = astar_search(grid, (0, 0), (1, 1))
        assert path is None

    def test_goal_on_obstacle(self) -> None:
        grid = [[0, 0], [0, 1]]
        path = astar_search(grid, (0, 0), (1, 1))
        assert path is None

    def test_out_of_bounds_start(self) -> None:
        grid = [[0, 0], [0, 0]]
        path = astar_search(grid, (5, 5), (1, 1))
        assert path is None

    def test_same_start_goal(self) -> None:
        grid = [[0, 0], [0, 0]]
        path = astar_search(grid, (0, 0), (0, 0))
        assert path is not None
        assert path == [(0, 0)]

    def test_longer_maze(self) -> None:
        """L-shaped corridor."""
        grid = [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ]
        path = astar_search(grid, (0, 0), (2, 2))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert len(path) == 5  # (0,0)->(1,0)->(2,0)->(2,1)->(2,2)


# ---------------------------------------------------------------------------
# Dynamic obstacle loading
# ---------------------------------------------------------------------------


class TestLoadDynamicObstacles:
    """Tests for _load_dynamic_obstacles."""

    def test_valid_file(self) -> None:
        path = _make_dynamic_obstacles_json(
            [
                {"id": 0, "trajectory": [[3, 5], [3, 6], [3, 7]]},
                {"id": 1, "trajectory": [[10, 10]]},
            ]
        )
        try:
            result = _load_dynamic_obstacles(path)
            assert len(result) == 2
            assert result[0]["id"] == 0
            assert len(result[0]["trajectory"]) == 3
        finally:
            path.unlink()

    def test_missing_key(self) -> None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"wrong_key": []}, tmp)
        tmp.close()
        path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="Missing 'dynamic_obstacles'"):
                _load_dynamic_obstacles(path)
        finally:
            path.unlink()

    def test_not_a_list(self) -> None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        json.dump({"dynamic_obstacles": "not_a_list"}, tmp)
        tmp.close()
        path = Path(tmp.name)
        try:
            with pytest.raises(ValueError, match="must be a list"):
                _load_dynamic_obstacles(path)
        finally:
            path.unlink()

    def test_missing_id(self) -> None:
        path = _make_dynamic_obstacles_json([{"trajectory": [[0, 0]]}])
        try:
            with pytest.raises(ValueError, match="needs 'id' and 'trajectory'"):
                _load_dynamic_obstacles(path)
        finally:
            path.unlink()

    def test_empty_trajectory(self) -> None:
        path = _make_dynamic_obstacles_json([{"id": 0, "trajectory": []}])
        try:
            with pytest.raises(ValueError, match="non-empty"):
                _load_dynamic_obstacles(path)
        finally:
            path.unlink()

    def test_invalid_position_format(self) -> None:
        path = _make_dynamic_obstacles_json([{"id": 0, "trajectory": [[1]]}])
        try:
            with pytest.raises(ValueError, match="each position must be"):
                _load_dynamic_obstacles(path)
        finally:
            path.unlink()


# ---------------------------------------------------------------------------
# Time-blocked set construction
# ---------------------------------------------------------------------------


class TestBuildTimeBlocked:
    """Tests for _build_time_blocked."""

    def test_single_obstacle(self) -> None:
        obstacles = [{"id": 0, "trajectory": [[3, 5], [3, 6], [3, 7]]}]
        tb = _build_time_blocked(obstacles)
        assert tb[0] == {(3, 5)}
        assert tb[1] == {(3, 6)}
        assert tb[2] == {(3, 7)}

    def test_multiple_obstacles(self) -> None:
        obstacles = [
            {"id": 0, "trajectory": [[0, 0], [0, 1]]},
            {"id": 1, "trajectory": [[1, 1], [1, 0]]},
        ]
        tb = _build_time_blocked(obstacles)
        assert tb[0] == {(0, 0), (1, 1)}
        assert tb[1] == {(0, 1), (1, 0)}

    def test_empty_obstacles(self) -> None:
        tb = _build_time_blocked([])
        assert tb == {}


# ---------------------------------------------------------------------------
# SIPP search
# ---------------------------------------------------------------------------


class TestSippSearch:
    """Tests for sipp_search with dynamic obstacles."""

    def test_no_dynamic_obstacles(self) -> None:
        """With empty time_blocked, SIPP should behave like A*."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        path = sipp_search(grid, (0, 0), (2, 2), {}, max_time=50)
        assert path is not None
        assert path[0] == (0, 0, 0)
        assert path[-1][0] == 2 and path[-1][1] == 2

    def test_static_obstacle_waits(self) -> None:
        """Obstacle blocks direct path at t=0, agent waits then passes."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Obstacle at (0,1) at t=0 only
        time_blocked = {0: {(0, 1)}}
        path = sipp_search(grid, (0, 0), (0, 2), time_blocked, max_time=50)
        assert path is not None
        assert path[0] == (0, 0, 0)
        assert path[-1][0] == 0 and path[-1][1] == 2
        # Must not be at (0,1) at t=0
        positions_at_t0 = [(s[0], s[1]) for s in path if s[2] == 0]
        assert (0, 1) not in positions_at_t0

    def test_dynamic_obstacle_along_path(self) -> None:
        """Obstacle moves along the corridor, forcing wait or detour."""
        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        # Obstacle moves right along row 0: (0,1)@t0, (0,2)@t1, (0,3)@t2
        time_blocked = {0: {(0, 1)}, 1: {(0, 2)}, 2: {(0, 3)}}
        path = sipp_search(grid, (0, 0), (0, 3), time_blocked, max_time=50)
        assert path is not None
        # Path must avoid obstacle positions at their times
        for r, c, t in path:
            blocked = time_blocked.get(t, set())
            assert (r, c) not in blocked

    def test_completely_blocked_by_dynamic(self) -> None:
        """Dynamic obstacle blocks goal cell for all reachable time steps."""
        grid = [[0, 0], [0, 0]]
        # Obstacle blocks goal (1,1) for all time steps up to max_time
        time_blocked = {t: {(1, 1)} for t in range(100)}
        path = sipp_search(grid, (0, 0), (1, 1), time_blocked, max_time=10)
        assert path is None

    def test_vertex_collision_avoidance(self) -> None:
        """Robot must not step onto a cell occupied by a dynamic obstacle."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Obstacle at (1,1) at t=1
        time_blocked = {1: {(1, 1)}}
        path = sipp_search(grid, (0, 0), (2, 2), time_blocked, max_time=50)
        assert path is not None
        for r, c, t in path:
            if t == 1:
                assert (r, c) != (1, 1)

    def test_edge_collision_avoidance(self) -> None:
        """Robot must not swap positions with a dynamic obstacle."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Obstacle moves from (1,0) at t=0 to (0,0) at t=1 (swap with robot)
        time_blocked = {0: {(1, 0)}, 1: {(0, 0)}}
        path = sipp_search(grid, (0, 0), (1, 0), time_blocked, max_time=50)
        assert path is not None
        # If robot goes (0,0)@t0 -> (1,0)@t1, that's a swap with obstacle
        # going (1,0)@t0 -> (0,0)@t1. The search should avoid this.
        if len(path) >= 2 and path[0] == (0, 0, 0) and path[1] == (1, 0, 1):
            # If it did take this path, the edge collision check should have
            # prevented it — this would be a test failure
            pytest.fail("Edge collision not detected: robot swapped with obstacle")

    def test_wait_steps_counted(self) -> None:
        """Waiting in place counts as wait steps when bottleneck is blocked."""
        # True bottleneck: only (1,1) connects left and right halves
        grid = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ]
        # Obstacle blocks the bottleneck (1,1) at t=0 and t=1
        time_blocked = {0: {(1, 1)}, 1: {(1, 1)}}
        path = sipp_search(grid, (0, 0), (0, 2), time_blocked, max_time=50)
        assert path is not None
        # The path must be longer than the Manhattan distance because
        # the bottleneck is temporally blocked at t=0,1.
        manhattan = abs(0 - 0) + abs(0 - 2)
        assert len(path) > manhattan

    def test_start_on_obstacle(self) -> None:
        grid = [[1, 0], [0, 0]]
        path = sipp_search(grid, (0, 0), (1, 1), {}, max_time=50)
        assert path is None

    def test_out_of_bounds(self) -> None:
        grid = [[0, 0], [0, 0]]
        path = sipp_search(grid, (5, 5), (1, 1), {}, max_time=50)
        assert path is None

    def test_max_time_limit(self) -> None:
        """Search terminates at max_time."""
        grid = [[0, 0], [0, 0]]
        # Obstacle blocks goal forever
        time_blocked = {t: {(1, 1)} for t in range(100)}
        path = sipp_search(grid, (0, 0), (1, 1), time_blocked, max_time=5)
        assert path is None

    def test_obstacle_clears_later(self) -> None:
        """Path becomes available after obstacle clears."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        # Obstacle blocks (0,1) only at t=0
        time_blocked = {0: {(0, 1)}}
        path = sipp_search(grid, (0, 0), (0, 2), time_blocked, max_time=50)
        assert path is not None
        # Should arrive at goal
        assert path[-1][0] == 0 and path[-1][1] == 2
        # Should not be at (0,1) at t=0
        for r, c, t in path:
            if t == 0:
                assert (r, c) != (0, 1)


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------


class TestCliMain:
    """End-to-end CLI tests."""

    def test_real_svg_map(self) -> None:
        """Run against the classic_crossing.svg map."""
        map_file = Path("maps/svg_maps/classic_crossing.svg")
        if not map_file.exists():
            pytest.skip("classic_crossing.svg not found in this environment")

        # Capture stdout
        import sys as _sys
        from io import StringIO

        old_stdout = _sys.stdout
        _sys.stdout = buf = StringIO()
        try:
            rc = main(
                [
                    str(map_file),
                    "--start",
                    "1",
                    "1",
                    "--goal",
                    "38",
                    "38",
                    "--grid-size",
                    "40",
                ]
            )
        finally:
            _sys.stdout = old_stdout

        assert rc == 0
        output = buf.getvalue().strip()
        diagnostic = json.loads(output)

        assert diagnostic["tool"] == "mapf_oracle_diagnostic"
        assert diagnostic["issue"] == 4795
        assert diagnostic["grid_dimensions"] == {"rows": 40, "cols": 40}
        assert "mapf_feasible" in diagnostic
        assert "diagnostic_status" in diagnostic
        assert diagnostic["diagnostic_status"] in ("feasible", "infeasible")
        if diagnostic["mapf_feasible"]:
            assert isinstance(diagnostic["oracle_path_length"], int)
            assert diagnostic["oracle_path_length"] > 0
        assert 0.0 <= diagnostic["occupancy_ratio"] <= 1.0

    def test_synthetic_svg(self) -> None:
        """Run against a synthetic SVG with a clear path."""
        path = _make_svg(
            10.0,
            10.0,
            [
                {"x": 4, "y": 4, "w": 2, "h": 2, "label": "obstacle"},
            ],
        )
        try:
            import sys as _sys
            from io import StringIO

            old_stdout = _sys.stdout
            _sys.stdout = buf = StringIO()
            try:
                rc = main(
                    [
                        str(path),
                        "--start",
                        "0",
                        "0",
                        "--goal",
                        "10",
                        "10",
                        "--grid-size",
                        "10",
                    ]
                )
            finally:
                _sys.stdout = old_stdout

            assert rc == 0
            diagnostic = json.loads(buf.getvalue().strip())
            assert diagnostic["mapf_feasible"] is True
        finally:
            path.unlink()

    def test_missing_file(self) -> None:
        rc = main(["/nonexistent/path.svg"])
        assert rc == 1

    def test_sipp_mode_synthetic(self) -> None:
        """SIPP mode with dynamic obstacles on a synthetic SVG."""
        svg_path = _make_svg(10.0, 10.0, [])
        obs_path = _make_dynamic_obstacles_json(
            [
                {"id": 0, "trajectory": [[2, 2], [3, 3], [4, 4]]},
            ]
        )
        try:
            import sys as _sys
            from io import StringIO

            old_stdout = _sys.stdout
            _sys.stdout = buf = StringIO()
            try:
                rc = main(
                    [
                        str(svg_path),
                        "--start",
                        "0",
                        "0",
                        "--goal",
                        "9",
                        "9",
                        "--grid-size",
                        "10",
                        "--dynamic-obstacles",
                        str(obs_path),
                        "--max-time",
                        "50",
                    ]
                )
            finally:
                _sys.stdout = old_stdout

            assert rc == 0
            diagnostic = json.loads(buf.getvalue().strip())
            assert diagnostic["search_method"] == "sipp"
            assert diagnostic["dynamic_obstacle_count"] == 1
            assert diagnostic["max_time"] == 50
            assert "mapf_feasible" in diagnostic
        finally:
            svg_path.unlink()
            obs_path.unlink()

    def test_sipp_mode_feasible_path_avoids_obstacle(self) -> None:
        """SIPP finds path that avoids a stationary dynamic obstacle."""
        svg_path = _make_svg(10.0, 10.0, [])
        # Obstacle stays at (0,1) for all time steps
        obs_path = _make_dynamic_obstacles_json(
            [
                {"id": 0, "trajectory": [[0, 1]] * 20},
            ]
        )
        try:
            import sys as _sys
            from io import StringIO

            old_stdout = _sys.stdout
            _sys.stdout = buf = StringIO()
            try:
                rc = main(
                    [
                        str(svg_path),
                        "--start",
                        "0",
                        "0",
                        "--goal",
                        "0",
                        "2",
                        "--grid-size",
                        "10",
                        "--dynamic-obstacles",
                        str(obs_path),
                        "--max-time",
                        "20",
                    ]
                )
            finally:
                _sys.stdout = old_stdout

            assert rc == 0
            diagnostic = json.loads(buf.getvalue().strip())
            assert diagnostic["mapf_feasible"] is True
            # Verify no step lands on the obstacle
            for step in diagnostic["oracle_path"]:
                assert not (step["row"] == 0 and step["col"] == 1)
        finally:
            svg_path.unlink()
            obs_path.unlink()

    def test_sipp_mode_infeasible(self) -> None:
        """SIPP reports infeasible when obstacle blocks goal forever."""
        svg_path = _make_svg(5.0, 5.0, [])
        # Obstacle at grid cell (4,4) for all time steps beyond max_time
        obs_path = _make_dynamic_obstacles_json(
            [
                {"id": 0, "trajectory": [[4, 4]] * 200},
            ]
        )
        try:
            import sys as _sys
            from io import StringIO

            old_stdout = _sys.stdout
            _sys.stdout = buf = StringIO()
            try:
                rc = main(
                    [
                        str(svg_path),
                        "--start",
                        "0",
                        "0",
                        "--goal",
                        "5",
                        "5",
                        "--grid-size",
                        "5",
                        "--dynamic-obstacles",
                        str(obs_path),
                        "--max-time",
                        "10",
                    ]
                )
            finally:
                _sys.stdout = old_stdout

            assert rc == 0
            diagnostic = json.loads(buf.getvalue().strip())
            assert diagnostic["mapf_feasible"] is False
            assert diagnostic["diagnostic_status"] == "infeasible"
        finally:
            svg_path.unlink()
            obs_path.unlink()

    def test_sipp_missing_obstacles_file(self) -> None:
        """SIPP mode fails gracefully when obstacles file missing."""
        svg_path = _make_svg(5.0, 5.0, [])
        try:
            rc = main(
                [
                    str(svg_path),
                    "--dynamic-obstacles",
                    "/nonexistent/obstacles.json",
                ]
            )
            assert rc == 1
        finally:
            svg_path.unlink()

    def test_sipp_invalid_obstacles_json(self) -> None:
        """SIPP mode fails gracefully on invalid JSON."""
        svg_path = _make_svg(5.0, 5.0, [])
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        tmp.write("{invalid json")
        tmp.close()
        obs_path = Path(tmp.name)
        try:
            rc = main(
                [
                    str(svg_path),
                    "--dynamic-obstacles",
                    str(obs_path),
                ]
            )
            assert rc == 1
        finally:
            svg_path.unlink()
            obs_path.unlink()
