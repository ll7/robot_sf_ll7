"""Tests for robot_sf/benchmark/mapf_oracle.py.

Validates the benchmark-loop integration for the MAPF oracle diagnostic.
All tests use synthetic SVG maps and in-memory scenario YAML files.
No Robot SF runtime, CARLA, or GPU access required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.mapf_oracle import (
    _find_nearest_free,
    _resolve_map_path,
    _run_single_scenario_diagnostic,
    run_mapf_oracle_diagnostics,
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


def _make_scenario_yaml(scenarios: list[dict], tmp_dir: Path) -> Path:
    """Write a scenario matrix YAML file and return its path."""
    data = {"scenarios": scenarios}
    path = tmp_dir / "scenarios.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# _resolve_map_path
# ---------------------------------------------------------------------------


class TestResolveMapPath:
    """Tests for _resolve_map_path."""

    def test_relative_path(self) -> None:
        base = Path("/repo/configs/scenarios")
        result = _resolve_map_path("../../../maps/foo.svg", base)
        assert result == (base / "../../../maps/foo.svg").resolve()

    def test_absolute_path(self) -> None:
        result = _resolve_map_path("/absolute/path/map.svg", Path("/base"))
        assert result == Path("/absolute/path/map.svg")

    def test_simple_relative(self) -> None:
        base = Path("/repo/configs")
        result = _resolve_map_path("maps/foo.svg", base)
        assert result == Path("/repo/configs/maps/foo.svg")


# ---------------------------------------------------------------------------
# _find_nearest_free
# ---------------------------------------------------------------------------


class TestFindNearestFree:
    """Tests for _find_nearest_free."""

    def test_already_free(self) -> None:
        grid = [[0, 0], [0, 0]]
        assert _find_nearest_free(grid, 0, 0) == (0, 0)

    def test_target_blocked(self) -> None:
        grid = [[1, 0], [0, 0]]
        result = _find_nearest_free(grid, 0, 0)
        assert result is not None
        r, c = result
        assert grid[r][c] == 0

    def test_all_blocked(self) -> None:
        grid = [[1, 1], [1, 1]]
        assert _find_nearest_free(grid, 0, 0) == (None, None)

    def test_corner_blocked(self) -> None:
        grid = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
        r, c = _find_nearest_free(grid, 0, 0)
        assert r is not None and c is not None
        assert grid[r][c] == 0


# ---------------------------------------------------------------------------
# _run_single_scenario_diagnostic
# ---------------------------------------------------------------------------


class TestRunSingleScenarioDiagnostic:
    """Tests for _run_single_scenario_diagnostic."""

    def test_feasible_map(self) -> None:
        svg_path = _make_svg(10.0, 10.0, [])
        try:
            scenario = {"name": "test_open"}
            result = _run_single_scenario_diagnostic(scenario, svg_path, 10)
            assert result["scenario"] == "test_open"
            assert result["mapf_feasible"] is True
            assert result["diagnostic_status"] == "feasible"
            assert result["oracle_path_length"] is not None
            assert result["oracle_path_length"] > 0
            assert result["grid_dimensions"] == {"rows": 10, "cols": 10}
        finally:
            svg_path.unlink()

    def test_infeasible_map(self) -> None:
        """Wall blocking the entire middle row."""
        svg_path = _make_svg(
            10.0,
            10.0,
            [{"x": 0, "y": 4.5, "w": 10, "h": 1, "label": "obstacle"}],
        )
        try:
            scenario = {"name": "test_wall"}
            result = _run_single_scenario_diagnostic(scenario, svg_path, 10)
            assert result["mapf_feasible"] is False
            assert result["diagnostic_status"] == "infeasible"
            assert result["oracle_path_length"] is None
        finally:
            svg_path.unlink()

    def test_missing_map_file(self) -> None:
        scenario = {"name": "test_missing"}
        result = _run_single_scenario_diagnostic(scenario, Path("/nonexistent.svg"), 10)
        assert result["diagnostic_status"] == "error"
        assert "not found" in result["error"]

    def test_obstacle_count_reported(self) -> None:
        svg_path = _make_svg(
            10.0,
            10.0,
            [
                {"x": 2, "y": 2, "w": 1, "h": 1, "label": "obstacle"},
                {"x": 5, "y": 5, "w": 1, "h": 1, "label": "obstacle"},
            ],
        )
        try:
            scenario = {"name": "test_obs"}
            result = _run_single_scenario_diagnostic(scenario, svg_path, 10)
            assert result["obstacle_count"] == 2
            assert 0.0 < result["occupancy_ratio"] < 1.0
        finally:
            svg_path.unlink()


# ---------------------------------------------------------------------------
# run_mapf_oracle_diagnostics
# ---------------------------------------------------------------------------


class TestRunMapfOracleDiagnostics:
    """Tests for the full benchmark-loop integration function."""

    def test_single_feasible_scenario(self, tmp_path: Path) -> None:
        svg_path = _make_svg(10.0, 10.0, [])
        try:
            scenario_yaml = _make_scenario_yaml(
                [{"name": "open_field", "map_file": str(svg_path)}],
                tmp_path,
            )
            report = run_mapf_oracle_diagnostics(scenario_yaml, grid_size=10)
            assert report["schema_version"] == "mapf_oracle_benchmark_diagnostics.v1"
            assert report["issue"] == 4795
            assert report["total_scenarios"] == 1
            assert report["feasible_count"] == 1
            assert report["infeasible_count"] == 0
            assert report["error_count"] == 0
            assert report["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
            assert report["scenarios"][0]["mapf_feasible"] is True
        finally:
            svg_path.unlink()

    def test_multiple_scenarios(self, tmp_path: Path) -> None:
        svg_open = _make_svg(10.0, 10.0, [])
        svg_wall = _make_svg(10.0, 10.0, [{"x": 0, "y": 4.5, "w": 10, "h": 1, "label": "obstacle"}])
        try:
            scenario_yaml = _make_scenario_yaml(
                [
                    {"name": "open", "map_file": str(svg_open)},
                    {"name": "walled", "map_file": str(svg_wall)},
                ],
                tmp_path,
            )
            report = run_mapf_oracle_diagnostics(scenario_yaml, grid_size=10)
            assert report["total_scenarios"] == 2
            assert report["feasible_count"] == 1
            assert report["infeasible_count"] == 1
            names = [s["scenario"] for s in report["scenarios"]]
            assert "open" in names
            assert "walled" in names
        finally:
            svg_open.unlink()
            svg_wall.unlink()

    def test_missing_map_file(self, tmp_path: Path) -> None:
        scenario_yaml = _make_scenario_yaml(
            [{"name": "bad", "map_file": "/nonexistent/map.svg"}],
            tmp_path,
        )
        report = run_mapf_oracle_diagnostics(scenario_yaml, grid_size=10)
        assert report["total_scenarios"] == 1
        assert report["error_count"] == 1
        assert report["scenarios"][0]["diagnostic_status"] == "error"

    def test_missing_map_file_field(self, tmp_path: Path) -> None:
        scenario_yaml = _make_scenario_yaml(
            [{"name": "no_map_field"}],
            tmp_path,
        )
        report = run_mapf_oracle_diagnostics(scenario_yaml, grid_size=10)
        assert report["error_count"] == 1
        assert "missing" in report["scenarios"][0]["error"]

    def test_scenario_filter(self, tmp_path: Path) -> None:
        svg1 = _make_svg(10.0, 10.0, [])
        svg2 = _make_svg(10.0, 10.0, [])
        try:
            scenario_yaml = _make_scenario_yaml(
                [
                    {"name": "bottleneck_low", "map_file": str(svg1)},
                    {"name": "crossing_high", "map_file": str(svg2)},
                ],
                tmp_path,
            )
            report = run_mapf_oracle_diagnostics(
                scenario_yaml, grid_size=10, scenario_filter="bottleneck"
            )
            assert report["total_scenarios"] == 1
            assert report["scenarios"][0]["scenario"] == "bottleneck_low"
        finally:
            svg1.unlink()
            svg2.unlink()

    def test_nonexistent_scenario_file(self) -> None:
        report = run_mapf_oracle_diagnostics("/nonexistent/scenarios.yaml")
        assert report["status"] == "error"
        assert report["scenarios"] == []

    def test_real_svg_map(self, tmp_path: Path) -> None:
        """Run against a real map file if available."""
        repo_root = Path(__file__).resolve().parents[2]
        map_file = repo_root / "maps/svg_maps/classic_crossing.svg"
        if not map_file.exists():
            pytest.skip("classic_crossing.svg not found")

        scenario_yaml = _make_scenario_yaml(
            [{"name": "crossing", "map_file": str(map_file.resolve())}],
            tmp_path,
        )
        report = run_mapf_oracle_diagnostics(scenario_yaml, grid_size=40)
        assert report["total_scenarios"] == 1
        assert report["scenarios"][0]["diagnostic_status"] in ("feasible", "infeasible")

    def test_default_grid_size(self, tmp_path: Path) -> None:
        svg_path = _make_svg(40.0, 40.0, [])
        try:
            scenario_yaml = _make_scenario_yaml(
                [{"name": "test", "map_file": str(svg_path)}],
                tmp_path,
            )
            report = run_mapf_oracle_diagnostics(scenario_yaml)
            assert report["grid_size"] == 40
            assert report["scenarios"][0]["grid_dimensions"] == {"rows": 40, "cols": 40}
        finally:
            svg_path.unlink()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestMapfOracleCli:
    """Tests for the benchmark CLI mapf-oracle subcommand."""

    def test_cli_subcommand_exists(self) -> None:
        from robot_sf.benchmark.cli import get_parser

        parser = get_parser()
        # Verify the subcommand is registered by parsing known args
        args = parser.parse_args(["mapf-oracle", "dummy.yaml"])
        assert args.cmd == "mapf-oracle"
        assert args.matrix == "dummy.yaml"
        assert args.grid_size == 40
        assert args.filter is None

    def test_cli_with_custom_grid_size(self) -> None:
        from robot_sf.benchmark.cli import get_parser

        parser = get_parser()
        args = parser.parse_args(["mapf-oracle", "matrix.yaml", "--grid-size", "20"])
        assert args.grid_size == 20

    def test_cli_with_filter(self) -> None:
        from robot_sf.benchmark.cli import get_parser

        parser = get_parser()
        args = parser.parse_args(["mapf-oracle", "matrix.yaml", "--filter", "bottleneck"])
        assert args.filter == "bottleneck"

    def test_cli_end_to_end(self, tmp_path: Path) -> None:
        from robot_sf.benchmark.cli import cli_main

        svg_path = _make_svg(10.0, 10.0, [])
        try:
            scenario_yaml = _make_scenario_yaml(
                [{"name": "e2e_test", "map_file": str(svg_path)}],
                tmp_path,
            )
            import sys as _sys
            from io import StringIO

            old_stdout = _sys.stdout
            _sys.stdout = buf = StringIO()
            try:
                rc = cli_main(["mapf-oracle", str(scenario_yaml), "--grid-size", "10"])
            finally:
                _sys.stdout = old_stdout

            assert rc == 0
            output = buf.getvalue().strip()
            report = json.loads(output)
            assert report["total_scenarios"] == 1
            assert report["scenarios"][0]["mapf_feasible"] is True
        finally:
            svg_path.unlink()
