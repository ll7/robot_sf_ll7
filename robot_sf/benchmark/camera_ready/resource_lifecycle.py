"""Resource lifecycle management for camera-ready benchmark campaigns.

This module provides subprocess isolation for planner/kinematics arms to ensure
GPU memory is fully released between campaign iterations (issue #4826).

The subprocess isolation approach:
1. Parent process spawns one subprocess per planner/kinematics arm
2. Subprocess runs a single arm and writes summary.json + episodes.jsonl
3. Subprocess exit fully releases GPU memory via OS reclamation
4. Parent process reads the summary and continues to the next arm

This is more robust than in-process cleanup because it guarantees release of
all PyTorch/TensorFlow/SB3 global state regardless of implementation details.
"""

from __future__ import annotations

import gc
import json

# Set PyTorch allocator policy for subprocess isolation (defense-in-depth)
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class _SubprocessArmParams:
    """Parameters for running a single planner arm in subprocess isolation."""

    planner_key: str
    planner_algo: str
    planner_human_model_variant: str | None
    planner_human_model_source: str | None
    planner_group: str
    benchmark_profile: str
    socnav_missing_prereq_policy: str
    adapter_impact_eval: str
    kinematics: str
    observation_mode: str
    workers: int
    horizon: int | None
    dt: float | None
    scenario_matrix_path: Path
    episodes_path: Path
    summary_path: Path
    record_forces: bool
    record_planner_decision_trace: bool
    record_simulation_step_trace: bool
    observation_noise: dict[str, Any] | None
    synthetic_actuation_profile: dict[str, Any] | None
    latency_stress_profile: dict[str, Any] | None
    snqi_weights: dict[str, Any] | None
    snqi_baseline: dict[str, Any] | None
    algo_config_path: Path | None
    # Fully-prepared scenario list serialized by the parent (see
    # _run_campaign_planner_variant_subprocess). When set, the worker consumes it
    # verbatim instead of re-deriving scenarios from scenario_matrix_path: the
    # parent's preparation applies map_file normalization, seed overrides,
    # scenario-candidate filtering, AMV overrides, horizon schedules, and the
    # configured holonomic_command_mode — none of which a bare matrix re-load
    # reproduces (Slurm jobs 13372/13373: every episode failed on unresolvable
    # relative map_file, and arm seeds diverged from the campaign plan).
    scoped_scenarios_path: Path | None = None


# Path-typed fields on _SubprocessArmParams. dataclasses.asdict() returns these as
# PosixPath objects, which json.dumps cannot serialize, so the serializer below
# str-converts them first. The subprocess worker (_main_subprocess_worker)
# re-parses str->Path for these same fields. This is the single serialization
# point for the subprocess-isolation parent->worker handoff (issue #4826 / #4957);
# tests must exercise THIS function rather than a hand-converted copy so a
# regression here is caught.
_SUBPROCESS_ARM_PATH_FIELDS: tuple[str, ...] = (
    "scenario_matrix_path",
    "episodes_path",
    "summary_path",
    "algo_config_path",
    "scoped_scenarios_path",
)


def _serialize_subprocess_arm_params(params: _SubprocessArmParams) -> str:
    """Serialize arm params to JSON for handoff to the subprocess worker.

    ``_SubprocessArmParams`` holds ``pathlib.Path`` fields
    (``scenario_matrix_path``, ``episodes_path``, ``summary_path``,
    ``algo_config_path``). ``dataclasses.asdict`` returns those as ``PosixPath``
    objects, which ``json.dumps`` cannot serialize (this was the crash at
    ``campaign.py`` that prevented ``--arm-isolation subprocess`` from ever
    running end-to-end — issue #4957). This helper str-converts the path fields
    first so the parent->worker handoff is JSON-safe.

    Args:
        params: Arm execution parameters built by the parent campaign process.

    Returns:
        JSON string consumable by ``_main_subprocess_worker``.
    """
    params_dict = asdict(params)
    for field_name in _SUBPROCESS_ARM_PATH_FIELDS:
        value = params_dict.get(field_name)
        if value is not None:
            params_dict[field_name] = str(value)
    return json.dumps(params_dict)


def _cleanup_gpu_memory_before_exit(
    *,
    planner_key: str,
    kinematics: str,
) -> dict[str, Any]:
    """Clean up GPU memory before subprocess exit.

    Forces garbage collection and explicitly clears CUDA cache. This is
    defense-in-depth; the main benefit of subprocess isolation is that OS
    reclaims all resources on exit regardless.

    Args:
        planner_key: Identifier for the planner that just completed.
        kinematics: Kinematics variant that just completed.

    Returns:
        Memory metrics dict with allocated/freed stats and high-water mark.
    """
    memory_metrics: dict[str, Any] = {
        "planner_key": planner_key,
        "kinematics": kinematics,
        "torch_available": False,
        "cuda_available": False,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "high_water_mark_mb": 0.0,
        "allocated_freed_mb": 0.0,
        "reserved_freed_mb": 0.0,
    }

    if "torch" in sys.modules:
        import torch  # noqa: PLC0415

        memory_metrics["torch_available"] = True
        if torch.cuda.is_available():
            memory_metrics["cuda_available"] = True
            # Measure before gc/empty_cache so diagnostics capture what cleanup freed.
            memory_metrics["high_water_mark_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            allocated_before = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_before = torch.cuda.memory_reserved() / 1024 / 1024

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            allocated_after = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_after = torch.cuda.memory_reserved() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()

            memory_metrics["allocated_mb"] = allocated_after
            memory_metrics["reserved_mb"] = reserved_after
            memory_metrics["allocated_freed_mb"] = max(0, allocated_before - allocated_after)
            memory_metrics["reserved_freed_mb"] = max(0, reserved_before - reserved_after)

    return memory_metrics


def _run_single_arm_subprocess(params: _SubprocessArmParams) -> dict[str, Any]:
    """Run a single planner/kinematics arm in subprocess isolation.

    This is the main entrypoint for subprocess-isolated arm execution. It:
    1. Loads scenarios from the matrix
    2. Runs the batch for this planner/kinematics variant
    3. Writes episodes.jsonl and summary.json
    4. Returns the summary for parent process aggregation

    Args:
        params: Arm execution parameters from parent process.

    Returns:
        Summary dict compatible with campaign orchestration expectations.
    """

    from robot_sf.benchmark.camera_ready._config import (  # noqa: PLC0415
        _scenario_with_kinematics,
    )
    from robot_sf.benchmark.camera_ready._util import (  # noqa: PLC0415
        _latency_stress_metadata,
        _synthetic_actuation_metadata,
        _utc_now,
    )
    from robot_sf.benchmark.fallback_policy import (  # noqa: PLC0415
        availability_payload,
        summarize_benchmark_availability,
    )
    from robot_sf.benchmark.runner import load_scenario_matrix, run_batch  # noqa: PLC0415

    if params.scoped_scenarios_path is not None:
        # Preferred handoff: consume the parent's fully-prepared scenario list so
        # the subprocess arm executes EXACTLY what the in-process path would have.
        # Re-deriving scenarios from the matrix here loses the campaign loader's
        # map_file normalization (relative map paths then fail to resolve from the
        # worker cwd — Slurm jobs 13372/13373 lost all 147 episodes to this), plus
        # seed overrides, candidate filtering, AMV overrides, horizon schedules,
        # and the configured holonomic_command_mode.
        with params.scoped_scenarios_path.open("r", encoding="utf-8") as f:
            scoped_scenarios = json.load(f)
    else:
        # Legacy fallback for callers that predate scoped_scenarios_path.
        # NOTE (2026-07-11): this previously imported `robot_sf.benchmark.scenario_matrix`
        # — a module that has NEVER existed — and read `.scenarios` off an object that
        # was really a list (observed live in Slurm job 13364). load_scenario_matrix
        # lives in runner and returns the scenario list directly.
        scenarios = load_scenario_matrix(params.scenario_matrix_path)
        scoped_scenarios = [
            _scenario_with_kinematics(
                sc,
                kinematics=params.kinematics,
                holonomic_command_mode="differential_drive",  # Default from campaign
            )
            for sc in scenarios
        ]

    # Run the batch
    status = "ok"
    warnings: list[str] = []
    try:
        summary = run_batch(
            scoped_scenarios,
            out_path=params.episodes_path,
            schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
            horizon=params.horizon if params.horizon is not None else 0,
            dt=params.dt if params.dt is not None else 0.0,
            record_forces=params.record_forces,
            record_planner_decision_trace=params.record_planner_decision_trace,
            record_simulation_step_trace=params.record_simulation_step_trace,
            snqi_weights=params.snqi_weights,
            snqi_baseline=params.snqi_baseline,
            algo=params.planner_algo,
            algo_config_path=(
                str(params.algo_config_path) if params.algo_config_path is not None else None
            ),
            benchmark_profile=params.benchmark_profile,
            socnav_missing_prereq_policy=params.socnav_missing_prereq_policy,
            adapter_impact_eval=params.adapter_impact_eval,
            observation_mode=params.observation_mode,
            observation_noise=params.observation_noise,
            synthetic_actuation_profile=_synthetic_actuation_metadata(
                params.synthetic_actuation_profile
            ),
            latency_stress_profile=_latency_stress_metadata(
                params.latency_stress_profile,
                dt=params.dt,
            ),
            workers=params.workers,
            resume=False,  # Subprocess runs fresh
        )
        availability = summarize_benchmark_availability(summary)
        if availability.availability_status == "not_available":
            status = "not_available"
        elif availability.availability_status == "partial-failure":
            status = "partial-failure"
        elif availability.availability_status == "failed":
            status = "failed"
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        status = "failed"
        summary = {
            "status": "failed",
            "error": repr(exc),
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 0,
            "failures": [],
        }
        warnings.append(f"Arm failed: {exc}")

    # Add metadata to summary
    import time  # noqa: PLC0415

    planner_started_at_utc = _utc_now()
    planner_start = time.perf_counter()

    planner_finished_at_utc = _utc_now()
    runtime_sec = float(max(1e-9, time.perf_counter() - planner_start))
    episodes_written = int(summary.get("written", 0))

    summary["status"] = status
    summary["started_at_utc"] = planner_started_at_utc
    summary["finished_at_utc"] = planner_finished_at_utc
    summary["runtime_sec"] = runtime_sec
    summary["episodes_per_second"] = (episodes_written / runtime_sec) if runtime_sec > 0 else 0.0
    summary["kinematics"] = params.kinematics
    summary["benchmark_availability"] = availability_payload(summary)
    summary["episodes_total"] = episodes_written

    # Write summary.json
    params.summary_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(params.summary_path, summary)

    # GPU cleanup before exit (defense-in-depth)
    cleanup_metrics = _cleanup_gpu_memory_before_exit(
        planner_key=params.planner_key,
        kinematics=params.kinematics,
    )

    # Return summary with cleanup metrics for parent process
    result = {
        "summary": summary,
        "cleanup_metrics": cleanup_metrics,
        "warnings": warnings,
        "episodes_total": episodes_written,
    }

    return result


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _main_subprocess_worker() -> int:
    """Entrypoint for subprocess-isolated arm execution.

    Expects JSON-encoded _SubprocessArmParams via stdin. Writes result to stdout.

    Returns:
        0 on success, 1 on failure.
    """
    import json  # noqa: PLC0415

    from loguru import logger  # noqa: PLC0415

    # Read parameters from stdin
    params_json = sys.stdin.read()
    params_dict = json.loads(params_json)
    # Re-parse str -> Path using the same field list the serializer uses, so a
    # field added to one side cannot silently miss the other.
    for field_name in _SUBPROCESS_ARM_PATH_FIELDS:
        field_value = params_dict.get(field_name)
        if field_value:
            params_dict[field_name] = Path(field_value)

    params = _SubprocessArmParams(**params_dict)

    # Configure minimal logging for subprocess
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    logger.info(
        "Subprocess arm execution: planner={} algo={} kinematics={}",
        params.planner_key,
        params.planner_algo,
        params.kinematics,
    )

    try:
        result = _run_single_arm_subprocess(params)
        # Write result to stdout as JSON for parent process
        print(json.dumps(result))  # noqa: T201
        return 0
    except (json.JSONDecodeError, TypeError, ValueError, RuntimeError, OSError, ImportError) as exc:
        logger.error("Subprocess arm failed: {}", exc)
        print(  # noqa: T201
            json.dumps(
                {
                    "summary": {
                        "status": "failed",
                        "error": repr(exc),
                        "total_jobs": 0,
                        "written": 0,
                        "failed_jobs": 0,
                        "failures": [],
                    },
                    "cleanup_metrics": {},
                    "warnings": [str(exc)],
                    "episodes_total": 0,
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(_main_subprocess_worker())
