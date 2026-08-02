"""Tests for robot_sf.benchmark.map_runner_batch_runner — job execution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.map_runner_batch_runner import (
    BatchExecutionResult,
    _initial_feasibility_totals,
    execute_map_jobs,
)

if TYPE_CHECKING:
    from pathlib import Path


def _noop_run_map_job(args: tuple) -> dict[str, Any]:
    """A stub run_map_job that returns a minimal episode record."""
    scenario, seed, _fixed = args
    return {
        "scenario_id": scenario.get("name", "unknown"),
        "seed": seed,
        "metrics": {"success": 1.0},
    }


def _noop_write(handle: Any, schema: dict, record: dict) -> None:
    """A stub write_validated_to_handle that writes nothing."""


class _NoopBridgeUpdate:
    """Minimal bridge update stub."""

    adapter_requested_seen = False
    adapter_native_steps = 0
    adapter_adapted_steps = 0
    runtime_algorithm_contract: dict[str, Any] = {}


def _noop_bridge(
    rec: dict, *, feasibility_totals: dict, runtime_algorithm_contract: dict | None
) -> _NoopBridgeUpdate:
    """A stub apply_worker_metadata_bridge."""
    return _NoopBridgeUpdate()


def _scenario_id(scenario: dict) -> str:
    """Extract scenario id for logging."""
    return scenario.get("name", "unknown")


class TestInitialFeasibilityTotals:
    """Tests for _initial_feasibility_totals."""

    def test_all_keys_present(self) -> None:
        """The initial totals must contain all expected accumulator keys."""
        totals = _initial_feasibility_totals()
        expected_keys = {
            "commands_evaluated",
            "infeasible_native_count",
            "projected_count",
            "sum_abs_delta_linear",
            "sum_abs_delta_angular",
            "max_abs_delta_linear",
            "max_abs_delta_angular",
            "ammv_commands_evaluated",
            "ammv_episode_count",
            "ammv_feasible_episode_count",
            "ammv_tip_over_episode_count",
            "ammv_curvature_violation_count",
            "ammv_min_stability_margin",
        }
        assert expected_keys.issubset(set(totals.keys()))

    def test_initial_values_zero(self) -> None:
        """Counters must start at zero (or inf for min_stability_margin)."""
        totals = _initial_feasibility_totals()
        assert totals["commands_evaluated"] == 0
        assert totals["infeasible_native_count"] == 0
        assert totals["projected_count"] == 0
        assert totals["ammv_min_stability_margin"] == float("inf")


class TestBatchExecutionResult:
    """Tests for BatchExecutionResult NamedTuple."""

    def test_fields_accessible(self) -> None:
        """All NamedTuple fields must be accessible."""
        result = BatchExecutionResult(
            wrote=5,
            episode_records=[],
            failures=[],
            adapter_native_steps=10,
            adapter_adapted_steps=2,
            adapter_samples_seen=True,
            runtime_algorithm_contract=None,
            feasibility_totals={},
            batch_runtime_sec=1.5,
            abort_metadata=None,
        )
        assert result.wrote == 5
        assert result.adapter_native_steps == 10
        assert result.adapter_samples_seen is True
        assert result.batch_runtime_sec == 1.5
        assert result.abort_metadata is None


class TestExecuteMapJobsSerial:
    """Tests for execute_map_jobs in serial mode (workers=1)."""

    def test_serial_execution_writes_all(self, tmp_path: Path) -> None:
        """Serial execution must write all successful jobs."""
        jobs = [({"name": f"sc-{i}"}, i) for i in range(3)]
        out_path = tmp_path / "episodes.jsonl"
        result = execute_map_jobs(
            jobs=jobs,
            fixed_params={},
            out_path=out_path,
            schema={},
            workers=1,
            run_map_job=_noop_run_map_job,
            write_validated_to_handle=_noop_write,
            apply_worker_metadata_bridge=_noop_bridge,
            scenario_id=_scenario_id,
            executor_cls=None,
            as_completed_fn=None,
        )
        assert result.wrote == 3
        assert len(result.episode_records) == 3
        assert result.failures == []
        assert result.batch_runtime_sec >= 0.0

    def test_serial_execution_records_failures(self, tmp_path: Path) -> None:
        """Serial execution must record failures without crashing."""

        def failing_job(args: tuple) -> dict:
            raise RuntimeError("sim failed")

        jobs = [({"name": "sc-0"}, 0)]
        out_path = tmp_path / "episodes.jsonl"
        result = execute_map_jobs(
            jobs=jobs,
            fixed_params={},
            out_path=out_path,
            schema={},
            workers=1,
            run_map_job=failing_job,
            write_validated_to_handle=_noop_write,
            apply_worker_metadata_bridge=_noop_bridge,
            scenario_id=_scenario_id,
            executor_cls=None,
            as_completed_fn=None,
        )
        assert result.wrote == 0
        assert len(result.failures) == 1
        assert "sim failed" in result.failures[0]["error"]

    def test_empty_jobs_list(self, tmp_path: Path) -> None:
        """An empty job list must produce zero writes."""
        out_path = tmp_path / "episodes.jsonl"
        result = execute_map_jobs(
            jobs=[],
            fixed_params={},
            out_path=out_path,
            schema={},
            workers=1,
            run_map_job=_noop_run_map_job,
            write_validated_to_handle=_noop_write,
            apply_worker_metadata_bridge=_noop_bridge,
            scenario_id=_scenario_id,
            executor_cls=None,
            as_completed_fn=None,
        )
        assert result.wrote == 0
        assert result.episode_records == []

    def test_feasibility_totals_initialized(self, tmp_path: Path) -> None:
        """Feasibility totals must be initialized in the result."""
        out_path = tmp_path / "episodes.jsonl"
        result = execute_map_jobs(
            jobs=[],
            fixed_params={},
            out_path=out_path,
            schema={},
            workers=1,
            run_map_job=_noop_run_map_job,
            write_validated_to_handle=_noop_write,
            apply_worker_metadata_bridge=_noop_bridge,
            scenario_id=_scenario_id,
            executor_cls=None,
            as_completed_fn=None,
        )
        assert result.feasibility_totals["commands_evaluated"] == 0
