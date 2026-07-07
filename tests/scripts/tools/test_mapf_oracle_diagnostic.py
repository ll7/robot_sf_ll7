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
    _parse_svg_obstacles,
    _svg_dimensions,
    astar_search,
    main,
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
