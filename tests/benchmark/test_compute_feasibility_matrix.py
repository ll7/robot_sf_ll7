"""Tests for compute-feasibility matrix runner (issue #5525)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.latency_stress import classify_feasibility

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "benchmark" / "run_compute_feasibility_matrix.py"


def _import_runner() -> Any:
    """Import runner module for unit testing."""
    import importlib

    return importlib.import_module("scripts.benchmark.run_compute_feasibility_matrix")


class TestConfigValidation:
    """Config loading must reject malformed YAML."""

    def test_rejects_wrong_schema(self, tmp_path: Path) -> None:
        """Config with wrong schema version must fail."""
        bad_yaml = {"schema_version": "unknown_v1", "planners": []}
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(yaml.dump(bad_yaml), encoding="utf-8")

        mod = _import_runner()
        with pytest.raises(ValueError, match="compute_feasibility_matrix.v1"):
            mod._load_config(bad_path)

    def test_accepts_valid_config(self) -> None:
        """Canonical config must load without error."""
        config_path = REPO_ROOT / "configs" / "benchmarks" / "compute_feasibility_matrix_v1.yaml"
        mod = _import_runner()
        config = mod._load_config(config_path)

        assert config["schema_version"] == "compute_feasibility_matrix.v1"
        assert len(config["planners"]) >= 2
        assert len(config["maps"]) >= 1
        assert len(config["pedestrian_counts"]) >= 1
        assert len(config["control_deadlines_ms"]) >= 1
        assert len(config["seeds"]) >= 1


class TestMatrixExpansion:
    """Matrix cell expansion must produce correct Cartesian product."""

    def test_cell_count(self) -> None:
        """Cell count must equal planners × maps × peds × deadlines × seeds."""
        config_path = REPO_ROOT / "configs" / "benchmarks" / "compute_feasibility_matrix_v1.yaml"
        mod = _import_runner()
        config = mod._load_config(config_path)
        cells = mod._expand_cells(config)

        expected = (
            len(config["planners"])
            * len(config["maps"])
            * len(config["pedestrian_counts"])
            * len(config["control_deadlines_ms"])
            * len(config["seeds"])
        )
        assert len(cells) == expected

    def test_cell_uniqueness(self) -> None:
        """Each cell tuple must appear exactly once."""
        config_path = REPO_ROOT / "configs" / "benchmarks" / "compute_feasibility_matrix_v1.yaml"
        mod = _import_runner()
        config = mod._load_config(config_path)
        cells = mod._expand_cells(config)

        tuples = {
            (
                c.planner_key,
                c.map_label,
                c.pedestrian_count,
                int(c.deadline_ms),
                c.seed,
            )
            for c in cells
        }
        assert len(tuples) == len(cells)

    def test_cell_fields_populated(self) -> None:
        """Each cell must carry all required fields."""
        config_path = REPO_ROOT / "configs" / "benchmarks" / "compute_feasibility_matrix_v1.yaml"
        mod = _import_runner()
        config = mod._load_config(config_path)
        cells = mod._expand_cells(config)

        for cell in cells:
            assert cell.planner_key
            assert cell.algo
            assert cell.map_label
            assert cell.map_path
            assert cell.pedestrian_count >= 0
            assert cell.deadline_ms > 0
            assert isinstance(cell.seed, int)


class TestDryRunMode:
    """Dry-run must print cells without executing episodes."""

    def test_dry_run_exits_zero(self, tmp_path: Path) -> None:
        """Dry-run should exit 0 and produce no episode output."""
        mini_config = {
            "schema_version": "compute_feasibility_matrix.v1",
            "planners": [{"key": "goal", "algo": "goal", "benchmark_profile": "baseline-safe"}],
            "maps": [
                {
                    "path": str(REPO_ROOT / "maps" / "svg_maps" / "atomic_empty_frame_test.svg"),
                    "label": "empty_8x8",
                }
            ],
            "pedestrian_counts": [0],
            "control_deadlines_ms": [100],
            "seeds": [42],
            "horizon": 10,
            "dt": 0.1,
        }
        config_path = tmp_path / "mini.yaml"
        config_path.write_text(yaml.dump(mini_config), encoding="utf-8")

        out_dir = tmp_path / "out"
        result = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--config",
                str(config_path),
                "--out",
                str(out_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0
        assert "Dry run: 1 cells" in result.stdout
        assert not out_dir.exists()


class TestSummaryGeneration:
    """Summary builder must aggregate cell rows correctly."""

    def test_empty_rows(self) -> None:
        """Empty input must produce valid summary."""
        mod = _import_runner()
        summary = mod._build_summary([])
        assert summary["total_cells"] == 0
        assert summary["non_native_cells"] == 0

    def test_native_cell_aggregation(self) -> None:
        """Native cells must aggregate p95 latency values."""
        rows = [
            {
                "cell": {
                    "planner_key": "goal",
                    "deadline_ms": 100.0,
                },
                "status": "native",
                "latency": {
                    "steady_state_latency_p95_ms": 45.0,
                },
            },
            {
                "cell": {
                    "planner_key": "goal",
                    "deadline_ms": 100.0,
                },
                "status": "native",
                "latency": {
                    "steady_state_latency_p95_ms": 50.0,
                },
            },
            {
                "cell": {
                    "planner_key": "goal",
                    "deadline_ms": 50.0,
                },
                "status": "native",
                "latency": {
                    "steady_state_latency_p95_ms": 55.0,
                },
            },
            {
                "cell": {
                    "planner_key": "mppi_social",
                    "deadline_ms": 100.0,
                },
                "status": "native",
                "latency": {
                    "steady_state_latency_p95_ms": 80.0,
                },
            },
        ]

        mod = _import_runner()
        summary = mod._build_summary(rows)

        assert summary["total_cells"] == 4
        assert summary["non_native_cells"] == 0

        goal_summary = summary["goal"]
        assert goal_summary["planner_key"] == "goal"

        goal_100 = goal_summary["p95_ms_deadline_100"]
        assert goal_100["n_cells"] == 2
        assert abs(goal_100["mean_ms"] - 47.5) < 0.01
        assert goal_100["max_ms"] == 50.0

        goal_50 = goal_summary["p95_ms_deadline_50"]
        assert goal_50["n_cells"] == 1
        assert goal_50["mean_ms"] == 55.0

        mppi_summary = summary["mppi_social"]
        mppi_100 = mppi_summary["p95_ms_deadline_100"]
        assert mppi_100["n_cells"] == 1
        assert mppi_100["mean_ms"] == 80.0

    def test_failed_cell_skipped_from_summary(self) -> None:
        """Non-native cells must not pollute p95 aggregation."""
        rows = [
            {
                "cell": {
                    "planner_key": "goal",
                    "deadline_ms": 100.0,
                },
                "status": "failed",
                "latency": {"failure_reason": "timeout"},
            },
            {
                "cell": {
                    "planner_key": "goal",
                    "deadline_ms": 100.0,
                },
                "status": "native",
                "latency": {
                    "steady_state_latency_p95_ms": 40.0,
                },
            },
        ]

        mod = _import_runner()
        summary = mod._build_summary(rows)

        assert summary["non_native_cells"] == 1
        goal_summary = summary["goal"]
        goal_100 = goal_summary["p95_ms_deadline_100"]
        assert goal_100["n_cells"] == 1
        assert goal_100["mean_ms"] == 40.0


class TestMatrixCell:
    """MatrixCell dataclass must carry expected fields."""

    def test_cell_construction(self) -> None:
        """Cell should be constructible with all fields."""
        import importlib

        mod = importlib.import_module("scripts.benchmark.run_compute_feasibility_matrix")
        cell = mod.MatrixCell(
            planner_key="goal",
            algo="goal",
            algo_config=None,
            map_label="test",
            map_path="maps/svg_maps/test.svg",
            pedestrian_count=5,
            deadline_ms=100.0,
            seed=42,
        )

        assert cell.planner_key == "goal"
        assert cell.algo == "goal"
        assert cell.algo_config is None
        assert cell.pedestrian_count == 5
        assert cell.deadline_ms == 100.0
        assert cell.seed == 42

    def test_cell_with_algo_config(self) -> None:
        """Cell should carry algo_config when provided."""
        import importlib

        mod = importlib.import_module("scripts.benchmark.run_compute_feasibility_matrix")
        cell = mod.MatrixCell(
            planner_key="mppi_social",
            algo="mppi_social",
            algo_config={"horizon_steps": 5, "sample_count": 8},
            map_label="test",
            map_path="maps/svg_maps/test.svg",
            pedestrian_count=0,
            deadline_ms=50.0,
            seed=111,
        )

        assert cell.algo_config["horizon_steps"] == 5
        assert cell.algo_config["sample_count"] == 8


class TestFeasibilityClassification:
    """Classification logic must match harness contract."""

    def test_meets_budget(self) -> None:
        """All latencies under deadline means feasible."""
        result = classify_feasibility(
            steady_state_latencies=[10.0, 20.0, 30.0, 40.0],
            deadline_ms=50.0,
        )
        assert result == "meets_budget_on_measured_host"

    def test_misses_budget(self) -> None:
        """Any latency over deadline means infeasible."""
        result = classify_feasibility(
            steady_state_latencies=[10.0, 60.0, 30.0, 40.0],
            deadline_ms=50.0,
        )
        assert result == "misses_budget_on_measured_host"

    def test_target_unmeasured(self) -> None:
        """Missing measured host identity with target hardware means unmeasured."""
        result = classify_feasibility(
            steady_state_latencies=[10.0, 20.0],
            deadline_ms=50.0,
            target_hardware="jetson",
            measured_host_identity=None,
        )
        assert result == "target_hardware_unmeasured"

    def test_empty_latencies(self) -> None:
        """Empty latencies means unmeasured."""
        result = classify_feasibility(
            steady_state_latencies=[],
            deadline_ms=50.0,
        )
        assert result == "target_hardware_unmeasured"
