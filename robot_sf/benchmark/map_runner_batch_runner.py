"""Job execution and writeback helpers for map-based benchmark batches."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, NamedTuple

from loguru import logger

FeasibilityTotals = dict[str, float | int]


class BatchExecutionResult(NamedTuple):
    """Counters and metadata accumulated while executing map-runner jobs."""

    wrote: int
    episode_records: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    adapter_native_steps: int
    adapter_adapted_steps: int
    adapter_samples_seen: bool
    runtime_algorithm_contract: dict[str, Any] | None
    feasibility_totals: FeasibilityTotals
    batch_runtime_sec: float


def _initial_feasibility_totals() -> FeasibilityTotals:
    """Return the mutable feasibility accumulator used by batch summaries."""
    return {
        "commands_evaluated": 0,
        "infeasible_native_count": 0,
        "projected_count": 0,
        "sum_abs_delta_linear": 0.0,
        "sum_abs_delta_angular": 0.0,
        "max_abs_delta_linear": 0.0,
        "max_abs_delta_angular": 0.0,
        "ammv_commands_evaluated": 0,
        "ammv_episode_count": 0,
        "ammv_feasible_episode_count": 0,
        "ammv_tip_over_episode_count": 0,
        "ammv_curvature_violation_count": 0,
        "ammv_min_stability_margin": float("inf"),
    }


def _fold_worker_metadata(
    rec: dict[str, Any],
    *,
    apply_worker_metadata_bridge,
    feasibility_totals: FeasibilityTotals,
    runtime_algorithm_contract: dict[str, Any] | None,
    adapter_samples_seen: bool,
    adapter_native_steps: int,
    adapter_adapted_steps: int,
) -> tuple[bool, int, int, dict[str, Any]]:
    """Fold one episode record into batch-level adapter and contract counters.

    Returns:
        Updated adapter-sample flag, native/adapted step counters, and runtime contract.
    """
    bridge_update = apply_worker_metadata_bridge(
        rec,
        feasibility_totals=feasibility_totals,
        runtime_algorithm_contract=runtime_algorithm_contract,
    )
    return (
        adapter_samples_seen or bridge_update.adapter_requested_seen,
        adapter_native_steps + bridge_update.adapter_native_steps,
        adapter_adapted_steps + bridge_update.adapter_adapted_steps,
        bridge_update.runtime_algorithm_contract,
    )


def _serial_execute_map_jobs(  # noqa: PLR0913
    *,
    jobs: list[tuple[dict[str, Any], int]],
    fixed_params: dict[str, Any],
    out_path: Path,
    schema: dict[str, Any],
    run_map_job,
    write_validated_to_handle,
    apply_worker_metadata_bridge,
    scenario_id,
    feasibility_totals: FeasibilityTotals,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], bool, int, int, dict[str, Any] | None]:
    """Execute map-runner jobs serially and append successful records.

    Returns:
        Write count, failures, adapter counters, and runtime algorithm contract.
    """
    wrote = 0
    episode_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    adapter_native_steps = 0
    adapter_adapted_steps = 0
    adapter_samples_seen = False
    runtime_algorithm_contract: dict[str, Any] | None = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for scenario, seed in jobs:
            try:
                rec = run_map_job((scenario, seed, fixed_params))
                (
                    adapter_samples_seen,
                    adapter_native_steps,
                    adapter_adapted_steps,
                    runtime_algorithm_contract,
                ) = _fold_worker_metadata(
                    rec,
                    apply_worker_metadata_bridge=apply_worker_metadata_bridge,
                    feasibility_totals=feasibility_totals,
                    runtime_algorithm_contract=runtime_algorithm_contract,
                    adapter_samples_seen=adapter_samples_seen,
                    adapter_native_steps=adapter_native_steps,
                    adapter_adapted_steps=adapter_adapted_steps,
                )
                write_validated_to_handle(handle, schema, rec)
                wrote += 1
                episode_records.append(rec)
            except Exception as exc:  # pragma: no cover - error path
                logger.exception(
                    "Map batch worker failed in serial execution: scenario={} seed={}",
                    scenario_id(scenario),
                    seed,
                )
                failures.append(
                    {
                        "scenario_id": scenario.get("name", "unknown"),
                        "seed": seed,
                        "error": repr(exc),
                    }
                )
    return (
        wrote,
        episode_records,
        failures,
        adapter_samples_seen,
        adapter_native_steps,
        adapter_adapted_steps,
        runtime_algorithm_contract,
    )


def _parallel_execute_map_jobs(  # noqa: PLR0913
    *,
    jobs: list[tuple[dict[str, Any], int]],
    fixed_params: dict[str, Any],
    out_path: Path,
    schema: dict[str, Any],
    run_map_job,
    write_validated_to_handle,
    apply_worker_metadata_bridge,
    scenario_id,
    feasibility_totals: FeasibilityTotals,
    workers: int,
    executor_cls,
    as_completed_fn,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], bool, int, int, dict[str, Any] | None]:
    """Execute map-runner jobs in parallel and append records in job order.

    Returns:
        Write count, failures, adapter counters, and runtime algorithm contract.
    """
    wrote = 0
    episode_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    adapter_native_steps = 0
    adapter_adapted_steps = 0
    adapter_samples_seen = False
    runtime_algorithm_contract: dict[str, Any] | None = None
    results_by_idx: dict[int, dict[str, Any]] = {}
    with executor_cls(max_workers=int(workers)) as ex:
        future_to_job: dict[Any, tuple[int, dict[str, Any], int]] = {}
        for idx, (scenario, seed) in enumerate(jobs, start=1):
            fut = ex.submit(run_map_job, (scenario, seed, fixed_params))
            future_to_job[fut] = (idx, scenario, seed)
        for fut in as_completed_fn(future_to_job):
            idx, scenario, seed = future_to_job[fut]
            try:
                rec = fut.result()
                (
                    adapter_samples_seen,
                    adapter_native_steps,
                    adapter_adapted_steps,
                    runtime_algorithm_contract,
                ) = _fold_worker_metadata(
                    rec,
                    apply_worker_metadata_bridge=apply_worker_metadata_bridge,
                    feasibility_totals=feasibility_totals,
                    runtime_algorithm_contract=runtime_algorithm_contract,
                    adapter_samples_seen=adapter_samples_seen,
                    adapter_native_steps=adapter_native_steps,
                    adapter_adapted_steps=adapter_adapted_steps,
                )
                results_by_idx[idx] = rec
            except Exception as exc:  # pragma: no cover
                logger.exception(
                    "Map batch worker failed in parallel execution: scenario={} seed={}",
                    scenario_id(scenario),
                    seed,
                )
                failures.append(
                    {
                        "scenario_id": scenario.get("name", "unknown"),
                        "seed": seed,
                        "error": repr(exc),
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for idx in sorted(results_by_idx):
            try:
                write_validated_to_handle(handle, schema, results_by_idx[idx])
                wrote += 1
                episode_records.append(results_by_idx[idx])
            except Exception as exc:  # pragma: no cover - write/validate path
                rec = results_by_idx[idx]
                logger.exception(
                    (
                        "Map batch write/validation failed for scenario={} seed={}; "
                        "preserving fail-closed batch status."
                    ),
                    rec.get("scenario_id", "unknown"),
                    rec.get("seed", -1),
                )
                failures.append(
                    {
                        "scenario_id": rec.get("scenario_id")
                        or rec.get("scenario", {}).get("name", "unknown"),
                        "seed": rec.get("seed", -1),
                        "error": repr(exc),
                    }
                )
    return (
        wrote,
        episode_records,
        failures,
        adapter_samples_seen,
        adapter_native_steps,
        adapter_adapted_steps,
        runtime_algorithm_contract,
    )


def execute_map_jobs(  # noqa: PLR0913
    *,
    jobs: list[tuple[dict[str, Any], int]],
    fixed_params: dict[str, Any],
    out_path: str | Path,
    schema: dict[str, Any],
    workers: int,
    run_map_job,
    write_validated_to_handle,
    apply_worker_metadata_bridge,
    scenario_id,
    executor_cls,
    as_completed_fn,
) -> BatchExecutionResult:
    """Execute map-runner jobs and append validated JSONL records.

    Returns:
        Batch execution counters and metadata accumulators.
    """
    out_path = Path(out_path)
    feasibility_totals = _initial_feasibility_totals()
    batch_started = time.perf_counter()
    if workers <= 1:
        (
            wrote,
            episode_records,
            failures,
            adapter_samples_seen,
            adapter_native_steps,
            adapter_adapted_steps,
            runtime_algorithm_contract,
        ) = _serial_execute_map_jobs(
            jobs=jobs,
            fixed_params=fixed_params,
            out_path=out_path,
            schema=schema,
            run_map_job=run_map_job,
            write_validated_to_handle=write_validated_to_handle,
            apply_worker_metadata_bridge=apply_worker_metadata_bridge,
            scenario_id=scenario_id,
            feasibility_totals=feasibility_totals,
        )
    else:
        (
            wrote,
            episode_records,
            failures,
            adapter_samples_seen,
            adapter_native_steps,
            adapter_adapted_steps,
            runtime_algorithm_contract,
        ) = _parallel_execute_map_jobs(
            jobs=jobs,
            fixed_params=fixed_params,
            out_path=out_path,
            schema=schema,
            run_map_job=run_map_job,
            write_validated_to_handle=write_validated_to_handle,
            apply_worker_metadata_bridge=apply_worker_metadata_bridge,
            scenario_id=scenario_id,
            feasibility_totals=feasibility_totals,
            workers=workers,
            executor_cls=executor_cls,
            as_completed_fn=as_completed_fn,
        )
    return BatchExecutionResult(
        wrote=wrote,
        episode_records=episode_records,
        failures=failures,
        adapter_native_steps=adapter_native_steps,
        adapter_adapted_steps=adapter_adapted_steps,
        adapter_samples_seen=adapter_samples_seen,
        runtime_algorithm_contract=runtime_algorithm_contract,
        feasibility_totals=feasibility_totals,
        batch_runtime_sec=float(max(time.perf_counter() - batch_started, 0.0)),
    )
