"""Camera-ready benchmark campaign orchestration (extracted ``run_campaign``).

The heavy ``run_campaign`` orchestrator was moved here from
``robot_sf.benchmark.camera_ready_campaign`` for the #3385 decomposition so the legacy
module can act as a thin compatibility facade.

``run_campaign`` accepts its filesystem-/subprocess-touching collaborators
(``prepare_campaign_preflight``, ``run_batch``, ``compute_aggregates_with_ci`` and
``export_publication_bundle``) as optional injected callables, mirroring the pattern
already used by ``camera_ready/_preflight.prepare_campaign_preflight``. The facade injects
its own module-level bindings so existing tests that monkeypatch
``robot_sf.benchmark.camera_ready_campaign.<name>`` keep working unchanged. When no override
is supplied the canonical implementations are imported lazily, keeping this module free of a
circular import back onto the facade.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.assurance_fragment import (
    build_assurance_fragment,
    validate_assurance_fragment,
    write_assurance_fragment,
)
from robot_sf.benchmark.camera_ready._artifacts import (
    _write_actuation_envelope_artifacts,
    _write_json,
    _write_seed_episode_rows_artifact,
    _write_seed_variability_artifacts,
    _write_snqi_diagnostics_artifacts,
    _write_statistical_sufficiency_artifact,
    _write_table_artifacts,
)
from robot_sf.benchmark.camera_ready._config import _sanitize_name, _scenario_with_kinematics
from robot_sf.benchmark.camera_ready._reporting import (
    _build_breakdown_rows,
    _build_scenario_amv_lookup,
    _planner_report_row,
    build_campaign_credibility_scorecard,
    write_campaign_report,
)
from robot_sf.benchmark.camera_ready._run_state import _campaign_success_counters
from robot_sf.benchmark.camera_ready._summaries import (
    _SEED_VARIABILITY_METRICS,
    _build_actuation_envelope_summary,
    _build_seed_variability_payload,
    _build_statistical_sufficiency_payload,
)
from robot_sf.benchmark.camera_ready._util import (
    _kinematics_matrix_or_default,
    _latency_stress_metadata,
    _repo_relative,
    _sha256_file,
    _sha256_payload,
    _synthetic_actuation_metadata,
    _utc_now,
)
from robot_sf.benchmark.fallback_policy import (
    availability_payload,
    classify_planner_row_status,
    summarize_benchmark_availability,
    summarize_campaign_outcome,
    summarize_campaign_status_axes,
)
from robot_sf.benchmark.latency_stress import not_available_latency_metrics
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.seed_variance import build_seed_episode_rows
from robot_sf.benchmark.snqi.campaign_contract import (
    SnqiContractThresholds,
    build_positioning_recommendation,
    calibrate_weights,
    collect_episodes_from_campaign_runs,
    compute_baseline_stats_from_episodes,
    compute_component_correlations,
    compute_component_dominance,
    compute_planner_snqi_ordering,
    compute_weight_sensitivity,
    evaluate_snqi_contract,
    resolve_weight_mapping,
    sanitize_baseline_stats,
    soft_contract_warning_active,
    validate_snqi_normalized_inputs,
)
from robot_sf.benchmark.utils import load_optional_json
from robot_sf.common.artifact_paths import get_artifact_category_path, get_repository_root

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec


CAMPAIGN_SCHEMA_VERSION = "benchmark-camera-ready-campaign.v1"
DEFAULT_EPISODE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")


@dataclass(frozen=True)
class _CampaignRuntimeDependencies:
    prepare_campaign_preflight: Callable[..., dict[str, Any]]
    run_batch: Callable[..., dict[str, Any]]
    compute_aggregates_with_ci: Callable[..., dict[str, Any]]
    export_publication_bundle: Callable[..., Any]


def _resolve_campaign_runtime_dependencies(
    *,
    prepare_campaign_preflight: Callable[..., dict[str, Any]] | None = None,
    run_batch: Callable[..., dict[str, Any]] | None = None,
    compute_aggregates_with_ci: Callable[..., dict[str, Any]] | None = None,
    export_publication_bundle: Callable[..., Any] | None = None,
) -> _CampaignRuntimeDependencies:
    if prepare_campaign_preflight is None:
        from robot_sf.benchmark.camera_ready._preflight import (  # noqa: PLC0415
            prepare_campaign_preflight,
        )
    if run_batch is None:
        from robot_sf.benchmark.runner import run_batch  # noqa: PLC0415
    if compute_aggregates_with_ci is None:
        from robot_sf.benchmark.aggregate import compute_aggregates_with_ci  # noqa: PLC0415
    if export_publication_bundle is None:
        from robot_sf.benchmark.artifact_publication import (  # noqa: PLC0415
            export_publication_bundle,
        )
    return _CampaignRuntimeDependencies(
        prepare_campaign_preflight=prepare_campaign_preflight,
        run_batch=run_batch,
        compute_aggregates_with_ci=compute_aggregates_with_ci,
        export_publication_bundle=export_publication_bundle,
    )


def run_campaign(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    skip_publication_bundle: bool = False,
    invoked_command: str | None = None,
    prepare_campaign_preflight: Callable[..., dict[str, Any]] | None = None,
    run_batch: Callable[..., dict[str, Any]] | None = None,
    compute_aggregates_with_ci: Callable[..., dict[str, Any]] | None = None,
    export_publication_bundle: Callable[..., Any] | None = None,
    arm_isolation: str | None = None,
) -> dict[str, Any]:
    """Execute a camera-ready planner campaign and emit campaign artifacts.

    The ``prepare_campaign_preflight``, ``run_batch``, ``compute_aggregates_with_ci`` and
    ``export_publication_bundle`` collaborators are injected so the legacy
    ``camera_ready_campaign`` facade can pass its own monkeypatchable bindings; when omitted
    the canonical implementations are imported lazily.

    Args:
        cfg: Campaign configuration.
        output_root: Optional campaign base output directory.
        label: Optional label suffix embedded into campaign_id.
        campaign_id: Optional exact campaign directory id for resume.
        skip_publication_bundle: Skip publication bundle export even if enabled in config.
        invoked_command: Full command line that invoked this run.
        prepare_campaign_preflight: Optional preflight collaborator override.
        run_batch: Optional batch run collaborator override.
        compute_aggregates_with_ci: Optional aggregates collaborator override.
        export_publication_bundle: Optional publication bundle collaborator override.
        arm_isolation: Optional override for arm isolation mode ("in_process" or "subprocess").
            If None, uses cfg.arm_isolation (issue #4826).

    Returns:
        Campaign execution summary with output paths and high-level counters.

    Raises:
        OrcaRvo2PreflightError: When enabled ORCA-dependent planners require ``rvo2`` but it is
            not importable.
        RouteClearanceError: When any scenario route centerline lies closer to a static obstacle
            than the robot radius, making the route geometrically impossible to follow without
            collision.
    """
    dependencies = _resolve_campaign_runtime_dependencies(
        prepare_campaign_preflight=prepare_campaign_preflight,
        run_batch=run_batch,
        compute_aggregates_with_ci=compute_aggregates_with_ci,
        export_publication_bundle=export_publication_bundle,
    )
    return _run_campaign_orchestrator(
        cfg,
        output_root=output_root,
        label=label,
        campaign_id=campaign_id,
        skip_publication_bundle=skip_publication_bundle,
        invoked_command=invoked_command,
        dependencies=dependencies,
        arm_isolation=arm_isolation,
    )


@dataclass(frozen=True)
class _CampaignPlannerRunResults:
    run_entries: list[dict[str, Any]]
    planner_rows: list[dict[str, Any]]
    warnings: list[str]
    seed_variability_records: list[dict[str, Any]]


@dataclass(frozen=True)
class _CampaignPlannerMatrixContext:
    cfg: CampaignConfig
    scenarios: list[Any]
    snqi_weights: dict[str, Any] | None
    snqi_baseline: dict[str, Any] | None
    runs_dir: Path
    dependencies: _CampaignRuntimeDependencies


@dataclass(frozen=True)
class _CampaignPlannerVariantResult:
    run_entries: list[dict[str, Any]]
    planner_rows: list[dict[str, Any]]
    warnings: list[str]
    seed_variability_records: list[dict[str, Any]]
    stop_requested: bool


@dataclass(frozen=True)
class _CampaignPlannerVariantRun:
    kinematics: str
    active_observation_mode: str
    planner_dir: Path
    episodes_path: Path
    effective_workers: int
    effective_horizon: int | None
    effective_dt: float | None
    scoped_scenarios: list[Any]


@dataclass(frozen=True)
class _CampaignPlannerBatchResult:
    status: str
    summary: dict[str, Any]
    warnings: list[str]


def _checkpoint_fallback_detected(summary: dict[str, Any]) -> bool:
    """Return whether preflight or runtime metadata proves checkpoint fallback occurred."""
    preflight = summary.get("preflight")
    contract = summary.get("algorithm_metadata_contract")
    checkpoint = contract.get("checkpoint_provenance") if isinstance(contract, dict) else None
    preflight_fallback = isinstance(preflight, dict) and (
        preflight.get("status") == "fallback"
        or preflight.get("planner_metadata_status") == "fallback"
    )
    return preflight_fallback or (
        isinstance(checkpoint, dict) and checkpoint.get("fallback_triggered") is True
    )


def _cleanup_gpu_memory_between_arms(
    *,
    planner_key: str,
    kinematics: str,
) -> dict[str, Any]:
    """Clean up GPU memory between campaign arms to prevent VRAM leaks.

    Forces garbage collection and explicitly clears CUDA cache after each
    planner/kinematics variant completes. Logs high-water marks for
    diagnostics.

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
            # Measure before empty_cache so diagnostics capture what cleanup freed.
            memory_metrics["high_water_mark_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
            allocated_before = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_before = torch.cuda.memory_reserved() / 1024 / 1024

            # Capture the allocated/reserved baseline before collecting Python
            # references so cleanup telemetry includes memory freed by gc.collect().
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

            logger.info(
                "GPU cleanup after planner={} kinematics={}: "
                "allocated {:.2f}→{:.2f} MiB ({:.2f} freed), "
                "reserved {:.2f}→{:.2f} MiB ({:.2f} freed), "
                "high-water {:.2f} MiB",
                planner_key,
                kinematics,
                allocated_before,
                allocated_after,
                memory_metrics["allocated_freed_mb"],
                reserved_before,
                reserved_after,
                memory_metrics["reserved_freed_mb"],
                memory_metrics["high_water_mark_mb"],
            )
        else:
            # CPU-only nodes still need Python-level cleanup between arms.
            gc.collect()
    else:
        # Keep no-torch environments from accumulating completed-arm objects.
        gc.collect()
    return memory_metrics


def _execute_campaign_planner_batch(
    context: _CampaignPlannerMatrixContext,
    planner: PlannerSpec,
    run: _CampaignPlannerVariantRun,
) -> _CampaignPlannerBatchResult:
    cfg = context.cfg
    dependencies = context.dependencies
    status = "ok"
    warnings: list[str] = []
    try:
        summary = dependencies.run_batch(
            run.scoped_scenarios,
            out_path=run.episodes_path,
            schema_path=DEFAULT_EPISODE_SCHEMA_PATH,
            horizon=run.effective_horizon if run.effective_horizon is not None else 0,
            dt=run.effective_dt if run.effective_dt is not None else 0.0,
            record_forces=cfg.record_forces,
            record_planner_decision_trace=cfg.record_planner_decision_trace,
            record_simulation_step_trace=cfg.record_simulation_step_trace,
            snqi_weights=context.snqi_weights,
            snqi_baseline=context.snqi_baseline,
            algo=planner.algo,
            algo_config_path=(
                str(planner.algo_config_path) if planner.algo_config_path is not None else None
            ),
            benchmark_profile=planner.benchmark_profile,
            socnav_missing_prereq_policy=(
                "fail-fast"
                if cfg.checkpoint_provenance_enforcement == "error"
                else planner.socnav_missing_prereq_policy
            ),
            adapter_impact_eval=planner.adapter_impact_eval,
            observation_mode=run.active_observation_mode,
            observation_noise=cfg.observation_noise,
            synthetic_actuation_profile=_synthetic_actuation_metadata(
                cfg.synthetic_actuation_profile
            ),
            latency_stress_profile=_latency_stress_metadata(
                cfg.latency_stress_profile,
                dt=run.effective_dt,
            ),
            workers=run.effective_workers,
            resume=cfg.resume,
        )
        availability = summarize_benchmark_availability(summary)
        if availability.availability_status == "not_available":
            status = "not_available"
        elif availability.availability_status == "partial-failure":
            status = "partial-failure"
        elif availability.availability_status == "failed":
            status = "failed"
    except Exception as exc:
        status = "failed"
        summary = {
            "status": "failed",
            "error": repr(exc),
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 0,
            "failures": [],
        }
        warnings.append(f"Planner '{planner.key}' failed for kinematics '{run.kinematics}': {exc}")
    if cfg.checkpoint_provenance_enforcement == "error":
        if _checkpoint_fallback_detected(summary):
            raise RuntimeError(
                "checkpoint_provenance_enforcement='error' blocked planner fallback for "
                f"arm '{planner.key}' ({run.kinematics})"
            )
    return _CampaignPlannerBatchResult(status=status, summary=summary, warnings=warnings)


def _prepare_campaign_planner_variant_run(
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    kinematics: str,
    active_observation_mode: str,
    log_run: bool = True,
) -> _CampaignPlannerVariantRun:
    cfg = context.cfg
    planner_run_key = f"{_sanitize_name(planner.key)}__{_sanitize_name(kinematics)}"
    planner_dir = context.runs_dir / planner_run_key
    planner_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = planner_dir / "episodes.jsonl"
    effective_workers = (
        planner.workers_override if planner.workers_override is not None else cfg.workers
    )
    effective_horizon = (
        planner.horizon_override if planner.horizon_override is not None else cfg.horizon
    )
    effective_dt = planner.dt_override if planner.dt_override is not None else cfg.dt
    if log_run:
        logger.info(
            "Running campaign planner key={} algo={} kinematics={} profile={} workers={}",
            planner.key,
            planner.algo,
            kinematics,
            planner.benchmark_profile,
            effective_workers,
        )
    scoped_scenarios = [
        _scenario_with_kinematics(
            sc,
            kinematics=kinematics,
            holonomic_command_mode=cfg.holonomic_command_mode,
        )
        for sc in context.scenarios
    ]
    return _CampaignPlannerVariantRun(
        kinematics=kinematics,
        active_observation_mode=active_observation_mode,
        planner_dir=planner_dir,
        episodes_path=episodes_path,
        effective_workers=effective_workers,
        effective_horizon=effective_horizon,
        effective_dt=effective_dt,
        scoped_scenarios=scoped_scenarios,
    )


def _dependency_gated_planner_summary(
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    run: _CampaignPlannerVariantRun,
) -> dict[str, Any]:
    readiness = get_algorithm_readiness(planner.algo)
    reason = str(planner.fail_closed_reason or "").strip() or (
        f"{planner.key} blocked by availability_gate={planner.availability_gate!r}"
    )
    logger.info(
        "Skipping dependency-gated planner key={} algo={} kinematics={} reason={}",
        planner.key,
        planner.algo,
        run.kinematics,
        reason,
    )
    return {
        "status": "not_available",
        "total_jobs": 0,
        "written": 0,
        "successful_jobs": 0,
        "failed_jobs": 0,
        "skipped_jobs": len(run.scoped_scenarios),
        "failures": [],
        "out_path": str(run.episodes_path),
        "algorithm_readiness": {
            "name": readiness.canonical_name if readiness is not None else planner.algo,
            "tier": readiness.tier if readiness is not None else "unknown",
            "profile": planner.benchmark_profile,
        },
        "algorithm_metadata_contract": {"planner_kinematics": {"execution_mode": "unknown"}},
        "preflight": {
            "status": "skipped",
            "compatibility_status": "dependency_gated",
            "compatibility_reason": reason,
            "availability_gate": planner.availability_gate,
            "learned_policy_contract": {"status": "not_applicable"},
        },
        "latency_stress_profile": (
            _latency_stress_metadata(
                context.cfg.latency_stress_profile,
                dt=run.effective_dt,
            )
            if context.cfg.latency_stress_profile is not None
            else None
        ),
        "latency_stress_metrics": (
            not_available_latency_metrics()
            if context.cfg.latency_stress_profile is not None
            else None
        ),
    }


def _run_campaign_planner_variant(
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    kinematics: str,
    active_observation_mode: str,
) -> _CampaignPlannerVariantResult:
    cfg = context.cfg
    run_entries: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seed_variability_records: list[dict[str, Any]] = []
    stop_requested = False
    run = _prepare_campaign_planner_variant_run(
        context,
        planner=planner,
        kinematics=kinematics,
        active_observation_mode=active_observation_mode,
        log_run=planner.availability_gate != "dependency_gated",
    )

    planner_started_at_utc = _utc_now()
    planner_start = time.perf_counter()
    if planner.availability_gate == "dependency_gated":
        status = "not_available"
        summary = _dependency_gated_planner_summary(context, planner=planner, run=run)
    else:
        batch_result = _execute_campaign_planner_batch(context, planner, run)
        status = batch_result.status
        summary = batch_result.summary
        warnings.extend(batch_result.warnings)
    aggregates: dict[str, Any] | None = None

    planner_finished_at_utc = _utc_now()
    runtime_sec = float(max(1e-9, time.perf_counter() - planner_start))
    episodes_written = int(summary.get("written", 0))
    summary["status"] = status
    summary["started_at_utc"] = planner_started_at_utc
    summary["finished_at_utc"] = planner_finished_at_utc
    summary["runtime_sec"] = runtime_sec
    summary["episodes_per_second"] = (episodes_written / runtime_sec) if runtime_sec > 0 else 0.0
    summary["kinematics"] = kinematics
    summary["benchmark_availability"] = availability_payload(summary)
    _write_json(run.planner_dir / "summary.json", summary)

    records: list[dict[str, Any]] = []
    if run.episodes_path.exists() and run.episodes_path.stat().st_size > 0:
        records = read_jsonl(str(run.episodes_path))
        summary["episodes_total"] = len(records)
        if status == "ok":
            for record in records:
                annotated = dict(record)
                annotated["planner_key"] = planner.key
                annotated["planner_group"] = planner.planner_group
                annotated["benchmark_profile"] = planner.benchmark_profile
                annotated["kinematics"] = kinematics
                seed_variability_records.append(annotated)
        try:
            aggregates = context.dependencies.compute_aggregates_with_ci(
                records,
                group_by="scenario_params.algo",
                bootstrap_samples=cfg.bootstrap_samples,
                bootstrap_confidence=cfg.bootstrap_confidence,
                bootstrap_seed=cfg.bootstrap_seed,
            )
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            warnings.append(
                f"Aggregation failed for planner '{planner.key}' ({kinematics}): {exc}",
            )

    row = _planner_report_row(
        planner,
        summary,
        aggregates,
        kinematics=kinematics,
        synthetic_actuation_profile=cfg.synthetic_actuation_profile,
        records=records,
    )
    planner_rows.append(row)

    if status in {"failed", "partial-failure"}:
        reason = str(row.get("most_likely_failure_reason", "")).strip() or "unspecified"
        warnings.append(
            "Planner failure recorded: "
            f"planner='{planner.key}' kinematics='{kinematics}' status='{status}' "
            f"most_likely_reason='{reason}'"
        )
    elif classify_planner_row_status(status) == "accepted_unavailable":
        reason = str(row.get("availability_reason", "")).strip() or "unspecified"
        warnings.append(
            "Accepted unavailable planner row recorded: "
            f"planner='{planner.key}' kinematics='{kinematics}' status='{status}' "
            f"availability_reason='{reason}'"
        )

    run_entries.append(
        {
            "planner": {
                "key": planner.key,
                "algo": planner.algo,
                "human_model_variant": planner.human_model_variant,
                "human_model_source": planner.human_model_source,
                "planner_group": planner.planner_group,
                "benchmark_profile": planner.benchmark_profile,
                "kinematics": kinematics,
                "algo_config_path": (
                    _repo_relative(planner.algo_config_path)
                    if planner.algo_config_path is not None
                    else None
                ),
                "socnav_missing_prereq_policy": planner.socnav_missing_prereq_policy,
                "adapter_impact_eval": planner.adapter_impact_eval,
                "observation_mode": active_observation_mode,
                "workers": run.effective_workers,
                "horizon": run.effective_horizon,
                "dt": run.effective_dt,
            },
            "status": status,
            "started_at_utc": planner_started_at_utc,
            "finished_at_utc": planner_finished_at_utc,
            "runtime_sec": runtime_sec,
            "episodes_path": _repo_relative(run.episodes_path),
            "summary_path": _repo_relative(run.planner_dir / "summary.json"),
            "summary": summary,
            "aggregates": aggregates,
        },
    )

    if classify_planner_row_status(status) == "unexpected_failure" and cfg.stop_on_failure:
        logger.warning(
            "Campaign stop_on_failure triggered: planner key={} kinematics={} status={} "
            "(halting remaining planners).",
            planner.key,
            kinematics,
            status,
        )
        if status == "partial-failure":
            warnings.append(
                (
                    "Campaign halted early: planner "
                    f"'{planner.key}' ({kinematics}) had partial failures "
                    f"({int(summary.get('failed_jobs', 0))} failed jobs); "
                    "stop_on_failure=true"
                ),
            )
        stop_requested = True
    return _CampaignPlannerVariantResult(
        run_entries=run_entries,
        planner_rows=planner_rows,
        warnings=warnings,
        seed_variability_records=seed_variability_records,
        stop_requested=stop_requested,
    )


def _run_campaign_planner_variant_subprocess(
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    kinematics: str,
    active_observation_mode: str,
) -> _CampaignPlannerVariantResult:
    """Run a single planner/kinematics arm via subprocess isolation.

    This variant spawns a subprocess to execute one arm. When the subprocess
    exits, the OS reclaims all GPU memory regardless of planner implementation
    details. This is the robust fix for issue #4826.

    Args:
        context: Campaign matrix context with config and dependencies.
        planner: Planner specification.
        kinematics: Kinematics variant to run.
        active_observation_mode: Observation mode for this run.

    Returns:
        Campaign variant result with run_entries, planner_rows, etc.
    """
    cfg = context.cfg

    # Import subprocess worker module
    from robot_sf.benchmark.camera_ready.resource_lifecycle import (  # noqa: PLC0415
        _serialize_subprocess_arm_params,
        _SubprocessArmParams,
    )

    run = _prepare_campaign_planner_variant_run(
        context,
        planner=planner,
        kinematics=kinematics,
        active_observation_mode=active_observation_mode,
        log_run=True,
    )

    # Hand the worker the fully-prepared scenario list rather than letting it
    # re-load the matrix: the preparation above (campaign loader + kinematics
    # scoping) carries map_file normalization, seed overrides, candidate
    # filtering, AMV overrides, horizon schedules, and holonomic_command_mode,
    # and the worker must execute exactly the same episodes the in-process
    # path would (Slurm jobs 13372/13373 failed every episode without this).
    scoped_scenarios_path = run.planner_dir / "scoped_scenarios.json"
    scoped_scenarios_path.write_text(
        json.dumps(run.scoped_scenarios, default=str),
        encoding="utf-8",
    )

    # Build subprocess parameters
    arm_params = _SubprocessArmParams(
        planner_key=planner.key,
        planner_algo=planner.algo,
        planner_human_model_variant=planner.human_model_variant,
        planner_human_model_source=planner.human_model_source,
        planner_group=planner.planner_group,
        benchmark_profile=planner.benchmark_profile,
        socnav_missing_prereq_policy=(
            "fail-fast"
            if cfg.checkpoint_provenance_enforcement == "error"
            else planner.socnav_missing_prereq_policy
        ),
        adapter_impact_eval=planner.adapter_impact_eval,
        kinematics=kinematics,
        observation_mode=active_observation_mode,
        workers=run.effective_workers,
        horizon=run.effective_horizon,
        dt=run.effective_dt,
        scenario_matrix_path=cfg.scenario_matrix_path,
        episodes_path=run.episodes_path,
        summary_path=run.planner_dir / "summary.json",
        record_forces=cfg.record_forces,
        record_planner_decision_trace=cfg.record_planner_decision_trace,
        record_simulation_step_trace=cfg.record_simulation_step_trace,
        observation_noise=cfg.observation_noise,
        synthetic_actuation_profile=cfg.synthetic_actuation_profile,
        latency_stress_profile=cfg.latency_stress_profile,
        snqi_weights=context.snqi_weights,
        snqi_baseline=context.snqi_baseline,
        algo_config_path=planner.algo_config_path,
        scoped_scenarios_path=scoped_scenarios_path,
    )

    # Serialize parameters for subprocess. The Path-typed fields on
    # _SubprocessArmParams must be str-converted before json.dumps or the
    # handoff crashes (issue #4957); _serialize_subprocess_arm_params is the
    # single serialization point and is covered by a regression test.
    arm_params_json = _serialize_subprocess_arm_params(arm_params)
    proc = subprocess.run(
        [sys.executable, "-m", "robot_sf.benchmark.camera_ready.resource_lifecycle"],
        input=arm_params_json,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        logger.error(
            "Subprocess arm failed: planner={} kinematics={} returncode={} stderr={}",
            planner.key,
            kinematics,
            proc.returncode,
            proc.stderr,
        )
        # Create a failed result
        status = "failed"
        summary = {
            "status": "failed",
            "error": f"Subprocess failed with return code {proc.returncode}: {proc.stderr}",
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 0,
            "failures": [],
        }
        warnings = [f"Subprocess failed: {proc.stderr}"]
        planner_rows = []
        seed_variability_records = []
        stop_requested = cfg.stop_on_failure and status in {"failed", "partial-failure"}

        run_entries = [
            {
                "planner": {
                    "key": planner.key,
                    "algo": planner.algo,
                    "human_model_variant": planner.human_model_variant,
                    "human_model_source": planner.human_model_source,
                    "planner_group": planner.planner_group,
                    "benchmark_profile": planner.benchmark_profile,
                    "kinematics": kinematics,
                    "algo_config_path": (
                        _repo_relative(planner.algo_config_path)
                        if planner.algo_config_path is not None
                        else None
                    ),
                    "socnav_missing_prereq_policy": planner.socnav_missing_prereq_policy,
                    "adapter_impact_eval": planner.adapter_impact_eval,
                    "observation_mode": active_observation_mode,
                    "workers": run.effective_workers,
                    "horizon": run.effective_horizon,
                    "dt": run.effective_dt,
                },
                "status": status,
                "started_at_utc": _utc_now(),
                "finished_at_utc": _utc_now(),
                "runtime_sec": 0.0,
                "episodes_path": _repo_relative(run.episodes_path),
                "summary_path": _repo_relative(run.planner_dir / "summary.json"),
                "summary": summary,
                "aggregates": None,
                "subprocess_isolation": True,
            }
        ]

        return _CampaignPlannerVariantResult(
            run_entries=run_entries,
            planner_rows=planner_rows,
            warnings=warnings,
            seed_variability_records=seed_variability_records,
            stop_requested=stop_requested,
        )

    warnings: list[str] = []

    # Parse subprocess result
    try:
        subprocess_result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse subprocess output: {}", exc)
        status = "failed"
        summary = {
            "status": "failed",
            "error": f"Failed to parse subprocess output: {exc}",
            "total_jobs": 0,
            "written": 0,
            "failed_jobs": 0,
            "failures": [],
        }
        warnings = [f"Subprocess output parse failed: {exc}"]
        subprocess_result = {
            "summary": summary,
            "cleanup_metrics": {},
            "warnings": warnings,
            "episodes_total": 0,
        }

    summary = subprocess_result.get("summary", {})
    cleanup_metrics = subprocess_result.get("cleanup_metrics", {})
    warnings.extend(subprocess_result.get("warnings", []))
    status = summary.get("status", "unknown")

    # Build run entry with subprocess isolation marker
    planner_started_at_utc = summary.get("started_at_utc", _utc_now())
    planner_finished_at_utc = summary.get("finished_at_utc", _utc_now())
    runtime_sec = summary.get("runtime_sec", 0.0)

    # Read episodes for aggregates
    records: list[dict[str, Any]] = []
    seed_variability_records: list[dict[str, Any]] = []
    if run.episodes_path.exists() and run.episodes_path.stat().st_size > 0:
        records = read_jsonl(str(run.episodes_path))
        if status == "ok":
            for record in records:
                annotated = dict(record)
                annotated["planner_key"] = planner.key
                annotated["planner_group"] = planner.planner_group
                annotated["benchmark_profile"] = planner.benchmark_profile
                annotated["kinematics"] = kinematics
                seed_variability_records.append(annotated)

    # Compute aggregates
    aggregates: dict[str, Any] | None = None
    if records and status == "ok":
        try:
            aggregates = context.dependencies.compute_aggregates_with_ci(
                records,
                group_by="scenario_params.algo",
                bootstrap_samples=cfg.bootstrap_samples,
                bootstrap_confidence=cfg.bootstrap_confidence,
                bootstrap_seed=cfg.bootstrap_seed,
            )
        except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
            warnings.append(
                f"Aggregation failed for planner '{planner.key}' ({kinematics}): {exc}",
            )

    # Build planner row
    row = _planner_report_row(
        planner,
        summary,
        aggregates,
        kinematics=kinematics,
        synthetic_actuation_profile=cfg.synthetic_actuation_profile,
        records=records,
    )
    planner_rows = [row]

    # Build run entry
    run_entries = [
        {
            "planner": {
                "key": planner.key,
                "algo": planner.algo,
                "human_model_variant": planner.human_model_variant,
                "human_model_source": planner.human_model_source,
                "planner_group": planner.planner_group,
                "benchmark_profile": planner.benchmark_profile,
                "kinematics": kinematics,
                "algo_config_path": (
                    _repo_relative(planner.algo_config_path)
                    if planner.algo_config_path is not None
                    else None
                ),
                "socnav_missing_prereq_policy": planner.socnav_missing_prereq_policy,
                "adapter_impact_eval": planner.adapter_impact_eval,
                "observation_mode": active_observation_mode,
                "workers": run.effective_workers,
                "horizon": run.effective_horizon,
                "dt": run.effective_dt,
            },
            "status": status,
            "started_at_utc": planner_started_at_utc,
            "finished_at_utc": planner_finished_at_utc,
            "runtime_sec": runtime_sec,
            "episodes_path": _repo_relative(run.episodes_path),
            "summary_path": _repo_relative(run.planner_dir / "summary.json"),
            "summary": summary,
            "aggregates": aggregates,
            "subprocess_isolation": True,
            "gpu_cleanup": cleanup_metrics,
        }
    ]

    # Check for stop_on_failure
    stop_requested = False
    if classify_planner_row_status(status) == "unexpected_failure" and cfg.stop_on_failure:
        logger.warning(
            "Campaign stop_on_failure triggered: planner key={} kinematics={} status={} "
            "(halting remaining planners).",
            planner.key,
            kinematics,
            status,
        )
        warnings.append(
            f"Campaign halted by subprocess arm failure: planner='{planner.key}' "
            f"kinematics='{kinematics}' status='{status}'"
        )
        stop_requested = True

    return _CampaignPlannerVariantResult(
        run_entries=run_entries,
        planner_rows=planner_rows,
        warnings=warnings,
        seed_variability_records=seed_variability_records,
        stop_requested=stop_requested,
    )


def _run_campaign_planner_matrix(
    *,
    cfg: CampaignConfig,
    scenarios: list[Any],
    snqi_weights: dict[str, Any] | None,
    snqi_baseline: dict[str, Any] | None,
    runs_dir: Path,
    dependencies: _CampaignRuntimeDependencies,
    arm_isolation: str | None = None,
) -> _CampaignPlannerRunResults:
    """Run the planner matrix with optional arm isolation override.

    Args:
        cfg: Campaign configuration.
        scenarios: Scenario list to run.
        snqi_weights: Optional SNQI weights dict.
        snqi_baseline: Optional SNQI baseline dict.
        runs_dir: Output directory for run results.
        dependencies: Runtime dependency collaborators.
        arm_isolation: Optional override for arm isolation mode ("in_process" or "subprocess").
            If None, uses cfg.arm_isolation (issue #4826).

    Returns:
        Campaign planner run results with run_entries, planner_rows, warnings, and
        seed_variability_records.
    """
    run_entries: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seed_variability_records: list[dict[str, Any]] = []
    kinematics_matrix = _kinematics_matrix_or_default(cfg.kinematics_matrix)
    # Use arm_isolation override if provided, otherwise use cfg value
    effective_arm_isolation = arm_isolation if arm_isolation is not None else cfg.arm_isolation
    context = _CampaignPlannerMatrixContext(
        cfg=cfg,
        scenarios=scenarios,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        runs_dir=runs_dir,
        dependencies=dependencies,
    )
    stop_requested = False

    for planner in cfg.planners:
        if not planner.enabled:
            continue
        active_observation_mode = planner.observation_mode or cfg.observation_mode
        for kinematics in kinematics_matrix:
            # Dispatch based on arm_isolation mode (issue #4826)
            use_subprocess = effective_arm_isolation == "subprocess"

            if use_subprocess:
                logger.info(
                    "Running arm with subprocess isolation: planner={} kinematics={}",
                    planner.key,
                    kinematics,
                )
                variant_result = _run_campaign_planner_variant_subprocess(
                    context,
                    planner=planner,
                    kinematics=kinematics,
                    active_observation_mode=active_observation_mode,
                )
            else:
                try:
                    variant_result = _run_campaign_planner_variant(
                        context,
                        planner=planner,
                        kinematics=kinematics,
                        active_observation_mode=active_observation_mode,
                    )
                finally:
                    # Clean up GPU memory and Python refs after each arm to prevent
                    # VRAM/RSS leaks across campaign iterations (issue #4826).
                    # Runs even if the arm raised — keeps the next arm from inheriting
                    # leaked CUDA allocations.
                    memory_metrics = _cleanup_gpu_memory_between_arms(
                        planner_key=planner.key,
                        kinematics=kinematics,
                    )
                # Attach diagnostics to the run entry created by this variant.
                if variant_result.run_entries:
                    variant_result.run_entries[-1]["gpu_cleanup"] = memory_metrics

            run_entries.extend(variant_result.run_entries)
            planner_rows.extend(variant_result.planner_rows)
            warnings.extend(variant_result.warnings)
            seed_variability_records.extend(variant_result.seed_variability_records)

            if variant_result.stop_requested:
                stop_requested = True
                break
        if stop_requested:
            break
    return _CampaignPlannerRunResults(
        run_entries=run_entries,
        planner_rows=planner_rows,
        warnings=warnings,
        seed_variability_records=seed_variability_records,
    )


def _build_skipped_combo_rows(run_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skipped_combo_rows: list[dict[str, Any]] = []
    for entry in run_entries:
        summary = entry.get("summary", {})
        if not isinstance(summary, dict):
            continue
        preflight = summary.get("preflight")
        if not isinstance(preflight, dict):
            continue
        if str(preflight.get("status", "")).lower() != "skipped":
            continue
        skipped_combo_rows.append(
            {
                "planner_key": str((entry.get("planner") or {}).get("key", "unknown")),
                "algo": str((entry.get("planner") or {}).get("algo", "unknown")),
                "kinematics": str((entry.get("planner") or {}).get("kinematics", "unknown")),
                "reason": str(
                    preflight.get("compatibility_reason")
                    or preflight.get("error")
                    or "unspecified skip reason"
                ),
            }
        )
    return skipped_combo_rows


def _runtime_checkpoint_record(
    entry: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any] | None:
    """Return one kinematics-specific runtime checkpoint record when observable."""
    summary = entry.get("summary")
    planner_entry = entry.get("planner")
    if not isinstance(summary, dict) or not isinstance(planner_entry, dict):
        return None
    contract = summary.get("algorithm_metadata_contract")
    runtime = contract.get("checkpoint_provenance") if isinstance(contract, dict) else None
    if not isinstance(runtime, dict):
        preflight = summary.get("preflight")
        if isinstance(preflight, dict) and preflight.get("status") == "fallback":
            runtime = {
                "model_id": provenance.get("model_id"),
                "checkpoint_sha256": provenance.get("checkpoint_sha256"),
                "load_succeeded": False,
                "fallback_triggered": True,
                "load_status": "fallback",
                "load_error": preflight.get("error"),
            }
        else:
            return None
    return {
        **runtime,
        "kinematics": planner_entry.get("kinematics"),
        "run_status": entry.get("status"),
    }


def _summarize_checkpoint_runtime(provenance: dict[str, Any]) -> None:
    """Summarize kinematics-specific load results on the planner-level block."""
    runtime_records = provenance.get("runtime")
    if not isinstance(runtime_records, list) or not runtime_records:
        return
    load_values = [
        item.get("load_succeeded")
        for item in runtime_records
        if isinstance(item, dict) and isinstance(item.get("load_succeeded"), bool)
    ]
    fallback_values = [
        item.get("fallback_triggered")
        for item in runtime_records
        if isinstance(item, dict) and isinstance(item.get("fallback_triggered"), bool)
    ]
    provenance["load_succeeded"] = all(load_values) if load_values else None
    provenance["fallback_triggered"] = any(fallback_values) if fallback_values else None
    runtime_hashes = {
        str(item["checkpoint_sha256"])
        for item in runtime_records
        if isinstance(item, dict) and item.get("checkpoint_sha256") is not None
    }
    if len(runtime_hashes) == 1:
        provenance["checkpoint_sha256"] = runtime_hashes.pop()
    if provenance["fallback_triggered"] is True:
        provenance["status"] = "fallback"
    elif provenance["load_succeeded"] is True:
        provenance["status"] = "loaded"
    elif provenance["load_succeeded"] is False:
        provenance["status"] = "load_failed"
    else:
        provenance["status"] = "runtime_not_observed"


def _finalize_checkpoint_provenance(
    manifest_payload: dict[str, Any], run_entries: list[dict[str, Any]]
) -> None:
    """Fold runtime load/fallback diagnostics into each manifest planner arm in place."""
    planners = manifest_payload.get("planners")
    if not isinstance(planners, list):
        return
    by_key = {str(planner.get("key")): planner for planner in planners if isinstance(planner, dict)}
    for entry in run_entries:
        planner_entry = entry.get("planner")
        if not isinstance(planner_entry, dict):
            continue
        planner = by_key.get(str(planner_entry.get("key")))
        provenance = planner.get("checkpoint_provenance") if isinstance(planner, dict) else None
        if not isinstance(provenance, dict) or provenance.get("status") == "not_applicable":
            continue
        runtime = _runtime_checkpoint_record(entry, provenance)
        if not isinstance(runtime, dict):
            continue
        provenance.setdefault("runtime", []).append(runtime)
        planner_entry["checkpoint_provenance"] = runtime

    for planner in planners:
        provenance = planner.get("checkpoint_provenance") if isinstance(planner, dict) else None
        if isinstance(provenance, dict):
            _summarize_checkpoint_runtime(provenance)


def _run_campaign_orchestrator(  # noqa: C901, PLR0912, PLR0915
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    skip_publication_bundle: bool = False,
    invoked_command: str | None = None,
    dependencies: _CampaignRuntimeDependencies,
    arm_isolation: str | None = None,
) -> dict[str, Any]:
    """Execute the campaign orchestrator with optional arm isolation override.

    Args:
        cfg: Campaign configuration.
        output_root: Optional campaign base output directory.
        label: Optional label suffix embedded into campaign_id.
        campaign_id: Optional exact campaign directory id for resume.
        skip_publication_bundle: Skip publication bundle export even if enabled in config.
        invoked_command: Full command line that invoked this run.
        dependencies: Runtime dependency collaborators.
        arm_isolation: Optional override for arm isolation mode ("in_process" or "subprocess").
            If None, uses cfg.arm_isolation (issue #4826).

    Returns:
        Campaign execution summary with output paths and counters.
    """
    start = time.perf_counter()
    prepared = dependencies.prepare_campaign_preflight(
        cfg,
        output_root=output_root,
        label=label,
        campaign_id=campaign_id,
        invoked_command=invoked_command,
    )
    campaign_id = str(prepared["campaign_id"])
    campaign_root = Path(prepared["campaign_root"])
    reports_dir = Path(prepared["reports_dir"])
    validate_config_path = Path(prepared["validate_config_path"])
    preview_scenarios_path = Path(prepared["preview_scenarios_path"])
    matrix_summary_json_path = Path(prepared["matrix_summary_json_path"])
    matrix_summary_csv_path = Path(prepared["matrix_summary_csv_path"])
    amv_coverage_json_path = Path(prepared["amv_coverage_json_path"])
    amv_coverage_md_path = Path(prepared["amv_coverage_md_path"])
    comparability_json_path = (
        Path(path) if (path := prepared.get("comparability_json_path")) else None
    )
    comparability_md_path = Path(path) if (path := prepared.get("comparability_md_path")) else None
    manifest_payload = dict(prepared["manifest_payload"])
    amv_summary = dict(prepared["amv_summary"])
    campaign_started_at_utc = str(prepared["created_at_utc"])
    scenarios = list(prepared["scenarios"])
    resolved_seeds = list(prepared["resolved_seeds"])
    scenario_hash = str(prepared["scenario_hash"])
    git_meta = dict(prepared["git_meta"])
    config_hash = str(prepared["config_hash"])
    snqi_weights = load_optional_json(str(cfg.snqi_weights_path) if cfg.snqi_weights_path else None)
    snqi_baseline = load_optional_json(
        str(cfg.snqi_baseline_path) if cfg.snqi_baseline_path else None
    )

    runs_dir = campaign_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    kinematics_matrix = _kinematics_matrix_or_default(cfg.kinematics_matrix)
    planner_run_results = _run_campaign_planner_matrix(
        cfg=cfg,
        scenarios=scenarios,
        snqi_weights=snqi_weights,
        snqi_baseline=snqi_baseline,
        runs_dir=runs_dir,
        dependencies=dependencies,
        arm_isolation=arm_isolation,
    )
    run_entries = planner_run_results.run_entries
    planner_rows = planner_run_results.planner_rows
    warnings = planner_run_results.warnings
    seed_variability_records = planner_run_results.seed_variability_records
    _finalize_checkpoint_provenance(manifest_payload, run_entries)

    planner_rows.sort(
        key=lambda row: (row.get("snqi_mean", "nan") == "nan", row.get("planner_key"))
    )

    summary_json_path = reports_dir / "campaign_summary.json"
    report_md_path = reports_dir / "campaign_report.md"
    credibility_scorecard_json_path = reports_dir / "campaign_credibility_scorecard.json"

    csv_path, md_table_path = _write_table_artifacts(
        reports_dir,
        "campaign_table",
        planner_rows,
        headers=(
            "planner_key",
            "algo",
            "human_model_variant",
            "human_model_source",
            "planner_group",
            "kinematics",
            "execution_mode",
            "readiness_status",
            "availability_status",
            "benchmark_success",
            "most_likely_failure_reason",
            "availability_reason",
            "readiness_tier",
            "preflight_status",
            "learned_policy_contract_status",
            "socnav_prereq_policy",
            "status",
            "episodes",
            "commands_evaluated",
            "projection_rate",
            "infeasible_rate",
            "success_mean",
            "collisions_mean",
            "ped_collision_count_mean",
            "obstacle_collision_count_mean",
            "total_collision_count_mean",
            "near_misses_mean",
            "time_to_goal_norm_mean",
            "path_efficiency_mean",
            "comfort_exposure_mean",
            "jerk_mean",
            "snqi_mean",
        ),
    )
    if cfg.paper_facing:
        core_rows = [row for row in planner_rows if str(row.get("planner_group")) == "core"]
        experimental_rows = [row for row in planner_rows if str(row.get("planner_group")) != "core"]
    else:
        core_rows = [
            row for row in planner_rows if str(row.get("readiness_tier")) == "baseline-ready"
        ]
        experimental_rows = [
            row for row in planner_rows if str(row.get("readiness_tier")) != "baseline-ready"
        ]
    core_csv_path, core_md_path = _write_table_artifacts(
        reports_dir,
        "campaign_table_core",
        core_rows,
        headers=(
            "planner_key",
            "algo",
            "human_model_variant",
            "human_model_source",
            "planner_group",
            "kinematics",
            "readiness_tier",
            "status",
            "episodes",
            "success_mean",
            "collisions_mean",
            "ped_collision_count_mean",
            "obstacle_collision_count_mean",
            "total_collision_count_mean",
            "snqi_mean",
        ),
    )
    experimental_csv_path, experimental_md_path = _write_table_artifacts(
        reports_dir,
        "campaign_table_experimental",
        experimental_rows,
        headers=(
            "planner_key",
            "algo",
            "human_model_variant",
            "human_model_source",
            "planner_group",
            "kinematics",
            "readiness_tier",
            "status",
            "episodes",
            "success_mean",
            "collisions_mean",
            "ped_collision_count_mean",
            "obstacle_collision_count_mean",
            "total_collision_count_mean",
            "snqi_mean",
        ),
    )
    scenario_amv_lookup = _build_scenario_amv_lookup(scenarios)
    scenario_rows, family_rows = _build_breakdown_rows(
        run_entries,
        scenario_amv_lookup=scenario_amv_lookup,
    )
    scenario_csv_path, scenario_md_path = _write_table_artifacts(
        reports_dir,
        "scenario_breakdown",
        scenario_rows,
        headers=(
            "planner_key",
            "algo",
            "scenario_family",
            "scenario_id",
            "use_case",
            "context",
            "speed_regime",
            "maneuver_type",
            "episodes",
            "success_mean",
            "collisions_mean",
            "ped_collision_count_mean",
            "obstacle_collision_count_mean",
            "total_collision_count_mean",
            "near_misses_mean",
            "time_to_goal_norm_mean",
            "path_efficiency_mean",
            "comfort_exposure_mean",
            "jerk_mean",
            "snqi_mean",
        ),
    )
    family_csv_path, family_md_path = _write_table_artifacts(
        reports_dir,
        "scenario_family_breakdown",
        family_rows,
        headers=(
            "planner_key",
            "algo",
            "scenario_family",
            "use_case",
            "context",
            "speed_regime",
            "maneuver_type",
            "episodes",
            "success_mean",
            "collisions_mean",
            "ped_collision_count_mean",
            "obstacle_collision_count_mean",
            "total_collision_count_mean",
            "near_misses_mean",
            "time_to_goal_norm_mean",
            "path_efficiency_mean",
            "comfort_exposure_mean",
            "jerk_mean",
            "snqi_mean",
        ),
    )
    parity_rows = sorted(
        [
            {
                "planner_key": str(row.get("planner_key", "")),
                "algo": str(row.get("algo", "")),
                "human_model_variant": str(row.get("human_model_variant", "")),
                "human_model_source": str(row.get("human_model_source", "")),
                "planner_group": str(row.get("planner_group", "experimental")),
                "kinematics": str(row.get("kinematics", "")),
                "execution_mode": str(row.get("execution_mode", "unknown")),
                "status": str(row.get("status", "unknown")),
                "episodes": int(row.get("episodes", 0)),
                "success_mean": str(row.get("success_mean", "nan")),
                "success_ci_low": str(row.get("success_ci_low", "nan")),
                "success_ci_high": str(row.get("success_ci_high", "nan")),
                "collisions_mean": str(row.get("collisions_mean", "nan")),
                "ped_collision_count_mean": str(row.get("ped_collision_count_mean", "nan")),
                "obstacle_collision_count_mean": str(
                    row.get("obstacle_collision_count_mean", "nan")
                ),
                "total_collision_count_mean": str(row.get("total_collision_count_mean", "nan")),
                "collision_ci_low": str(row.get("collision_ci_low", "nan")),
                "collision_ci_high": str(row.get("collision_ci_high", "nan")),
                "near_misses_mean": str(row.get("near_misses_mean", "nan")),
                "comfort_exposure_mean": str(row.get("comfort_exposure_mean", "nan")),
                "snqi_mean": str(row.get("snqi_mean", "nan")),
                "snqi_ci_low": str(row.get("snqi_ci_low", "nan")),
                "snqi_ci_high": str(row.get("snqi_ci_high", "nan")),
                "projection_rate": str(row.get("projection_rate", "0.0000")),
                "infeasible_rate": str(row.get("infeasible_rate", "0.0000")),
            }
            for row in planner_rows
        ],
        key=lambda row: (row["algo"], row["kinematics"], row["planner_key"]),
    )
    parity_csv_path, parity_md_path = _write_table_artifacts(
        reports_dir,
        "kinematics_parity_table",
        parity_rows,
        headers=(
            "planner_key",
            "algo",
            "human_model_variant",
            "human_model_source",
            "planner_group",
            "kinematics",
            "execution_mode",
            "status",
            "episodes",
            "success_mean",
            "success_ci_low",
            "success_ci_high",
            "collisions_mean",
            "ped_collision_count_mean",
            "obstacle_collision_count_mean",
            "total_collision_count_mean",
            "collision_ci_low",
            "collision_ci_high",
            "near_misses_mean",
            "comfort_exposure_mean",
            "snqi_mean",
            "snqi_ci_low",
            "snqi_ci_high",
            "projection_rate",
            "infeasible_rate",
        ),
    )
    skipped_combo_rows = _build_skipped_combo_rows(run_entries)
    skipped_csv_path, skipped_md_path = _write_table_artifacts(
        reports_dir,
        "kinematics_skipped_combinations",
        skipped_combo_rows,
        headers=("planner_key", "algo", "kinematics", "reason"),
    )

    campaign_finished_at_utc = _utc_now()
    runtime_sec = float(max(1e-9, time.perf_counter() - start))
    total_episodes = sum(
        int(
            entry.get("summary", {}).get(
                "episodes_total",
                entry.get("summary", {}).get("written", 0),
            )
        )
        for entry in run_entries
    )
    campaign_outcome = summarize_campaign_outcome(
        {"runs": run_entries, "planner_rows": planner_rows}
    )
    successful_runs = campaign_outcome.successful_runs
    expected_total_runs = len([planner for planner in cfg.planners if planner.enabled]) * len(
        kinematics_matrix
    )
    expected_core_runs = sum(
        1 for planner in cfg.planners if planner.enabled and planner.planner_group == "core"
    )
    campaign_status_axes = summarize_campaign_status_axes(
        {"runs": run_entries, "planner_rows": planner_rows},
        expected_total_runs=expected_total_runs,
    )
    row_status_summary = asdict(campaign_status_axes.row_status_summary)
    success_counters = _campaign_success_counters(
        run_entries, expected_core_runs=expected_core_runs * len(kinematics_matrix)
    )
    benchmark_success = bool(
        success_counters["benchmark_success"] and campaign_status_axes.evidence_status == "valid"
    )
    confidence_settings = {
        "method": "bootstrap_mean_over_seed_means",
        "confidence": float(cfg.bootstrap_confidence),
        "bootstrap_samples": int(cfg.bootstrap_samples),
        "bootstrap_seed": int(cfg.bootstrap_seed),
    }
    successful_seed_run_entries = [
        entry
        for entry in run_entries
        if str(entry.get("status", "")) == "ok" and str(entry.get("episodes_path", "")).strip()
    ]
    seed_source_paths = {
        "campaign_manifest_path": _repo_relative(campaign_root / "campaign_manifest.json"),
        "run_meta_path": _repo_relative(campaign_root / "run_meta.json"),
        "episodes_paths": [
            _repo_relative(campaign_root / str(entry.get("episodes_path", "")))
            for entry in successful_seed_run_entries
        ],
    }
    seed_variability_payload = _build_seed_variability_payload(
        seed_variability_records,
        campaign_id=campaign_id,
        generated_at_utc=campaign_finished_at_utc,
        config_hash=config_hash,
        git_hash=git_meta.get("commit", "unknown"),
        seed_policy={
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "resolved_seeds": list(resolved_seeds),
        },
        confidence_settings=confidence_settings,
        source_paths=seed_source_paths,
    )
    seed_variability_json_path, seed_variability_csv_path = _write_seed_variability_artifacts(
        reports_dir,
        seed_variability_payload,
    )
    seed_episode_rows = build_seed_episode_rows(seed_variability_records)
    seed_episode_rows_csv_path = _write_seed_episode_rows_artifact(reports_dir, seed_episode_rows)
    statistical_sufficiency_payload = _build_statistical_sufficiency_payload(
        campaign_id=campaign_id,
        generated_at_utc=campaign_finished_at_utc,
        seed_variability_payload=seed_variability_payload,
    )
    statistical_sufficiency_json_path = _write_statistical_sufficiency_artifact(
        reports_dir,
        statistical_sufficiency_payload,
    )
    actuation_envelope_payload: dict[str, Any] | None = None
    actuation_envelope_json_path: Path | None = None
    actuation_envelope_md_path: Path | None = None
    if cfg.synthetic_actuation_profile is not None:
        actuation_envelope_payload = _build_actuation_envelope_summary(
            campaign_id=campaign_id,
            generated_at_utc=campaign_finished_at_utc,
            profile=cfg.synthetic_actuation_profile,
            planner_rows=planner_rows,
            amv_summary=amv_summary,
        )
        actuation_envelope_json_path, actuation_envelope_md_path = (
            _write_actuation_envelope_artifacts(reports_dir, actuation_envelope_payload)
        )
    release_tag_value = cfg.release_tag
    expected_archive_name = f"{campaign_id}_publication_bundle.tar.gz"
    repository_url = cfg.repository_url.rstrip("/")
    release_url = f"{repository_url}/releases/tag/{release_tag_value}"
    release_asset_url = (
        f"{repository_url}/releases/download/{release_tag_value}/{expected_archive_name}"
    )
    doi_url = f"https://doi.org/{cfg.doi}"
    episodes = collect_episodes_from_campaign_runs(run_entries, repo_root=get_repository_root())
    configured_weights = resolve_weight_mapping(snqi_weights)
    if snqi_baseline is None:
        baseline_source = "derived_from_campaign_episodes"
        baseline_for_eval, baseline_warnings = compute_baseline_stats_from_episodes(episodes)
        baseline_adjustments = len(baseline_warnings)
        warnings.extend(baseline_warnings)
    else:
        baseline_source = "config_file"
        baseline_for_eval, baseline_warnings = sanitize_baseline_stats(snqi_baseline)
        baseline_adjustments = len(baseline_warnings)
        warnings.extend(baseline_warnings)
    if cfg.paper_facing and cfg.snqi_contract.enabled:
        normalized_input_issues = validate_snqi_normalized_inputs(
            episodes=episodes,
            baseline=baseline_for_eval,
        )
        if normalized_input_issues:
            raise RuntimeError(
                "SNQI sensitivity preflight failed: "
                + "; ".join(sorted(set(normalized_input_issues)))
            )

    thresholds = SnqiContractThresholds(
        rank_alignment_warn=cfg.snqi_contract.rank_alignment_warn_threshold,
        rank_alignment_fail=cfg.snqi_contract.rank_alignment_fail_threshold,
        outcome_separation_warn=cfg.snqi_contract.outcome_separation_warn_threshold,
        outcome_separation_fail=cfg.snqi_contract.outcome_separation_fail_threshold,
        max_component_dominance_warn=cfg.snqi_contract.max_component_dominance_warn_threshold,
        max_component_dominance_fail=cfg.snqi_contract.max_component_dominance_fail_threshold,
    )
    contract_eval = evaluate_snqi_contract(
        planner_rows,
        episodes,
        weights=configured_weights,
        baseline=baseline_for_eval,
        thresholds=thresholds,
    )
    calibration = calibrate_weights(
        planner_rows,
        episodes,
        baseline=baseline_for_eval,
        seed=cfg.snqi_contract.calibration_seed,
        trials=cfg.snqi_contract.calibration_trials,
    )
    component_dominance = compute_component_dominance(
        episodes,
        weights=configured_weights,
        baseline=baseline_for_eval,
    )
    component_correlations = compute_component_correlations(
        episodes,
        weights=configured_weights,
        baseline=baseline_for_eval,
    )
    planner_ordering = compute_planner_snqi_ordering(
        episodes,
        weights=configured_weights,
        baseline=baseline_for_eval,
    )
    weight_sensitivity = compute_weight_sensitivity(
        episodes,
        weights=configured_weights,
        baseline=baseline_for_eval,
    )
    positioning = build_positioning_recommendation(
        component_correlations,
        planner_ordering,
        weight_sensitivity,
    )
    weights_path = (
        _repo_relative(cfg.snqi_weights_path) if cfg.snqi_weights_path is not None else None
    )
    baseline_path = (
        _repo_relative(cfg.snqi_baseline_path) if cfg.snqi_baseline_path is not None else None
    )
    weights_sha256 = (
        _sha256_file(cfg.snqi_weights_path)
        if cfg.snqi_weights_path is not None
        else _sha256_payload(configured_weights)
    )
    baseline_sha256 = (
        _sha256_file(cfg.snqi_baseline_path)
        if cfg.snqi_baseline_path is not None
        else _sha256_payload(baseline_for_eval)
    )
    snqi_diagnostics_payload = {
        "schema_version": "benchmark-snqi-diagnostics.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": campaign_finished_at_utc,
        "contract_enabled": bool(cfg.snqi_contract.enabled),
        "contract_enforcement": cfg.snqi_contract.enforcement,
        "contract_status": contract_eval.status,
        "rank_alignment_spearman": contract_eval.rank_alignment_spearman,
        "outcome_separation": contract_eval.outcome_separation,
        "objective_score": contract_eval.objective_score,
        "dominant_component": contract_eval.dominant_component,
        "dominant_component_mean_abs": contract_eval.dominant_component_mean_abs,
        "thresholds": {
            "rank_alignment_warn": cfg.snqi_contract.rank_alignment_warn_threshold,
            "rank_alignment_fail": cfg.snqi_contract.rank_alignment_fail_threshold,
            "outcome_separation_warn": cfg.snqi_contract.outcome_separation_warn_threshold,
            "outcome_separation_fail": cfg.snqi_contract.outcome_separation_fail_threshold,
            "max_component_dominance_warn": cfg.snqi_contract.max_component_dominance_warn_threshold,
            "max_component_dominance_fail": cfg.snqi_contract.max_component_dominance_fail_threshold,
        },
        "weights_path": weights_path,
        "weights_version": (
            cfg.snqi_weights_path.stem if cfg.snqi_weights_path is not None else "default"
        ),
        "weights_sha256": weights_sha256,
        "baseline_path": baseline_path,
        "baseline_version": (
            cfg.snqi_baseline_path.stem if cfg.snqi_baseline_path is not None else "derived"
        ),
        "baseline_sha256": baseline_sha256,
        "baseline_source": baseline_source,
        "baseline_adjustments": baseline_adjustments,
        "baseline_for_eval": baseline_for_eval,
        "configured_weights": configured_weights,
        "calibrated_weights": calibration.get("weights"),
        "calibration": calibration,
        "component_dominance": component_dominance,
        "component_correlations": component_correlations,
        "planner_ordering": planner_ordering,
        "weight_sensitivity": weight_sensitivity,
        "positioning": positioning,
    }
    snqi_diagnostics_json_path, snqi_diagnostics_md_path, snqi_sensitivity_csv_path = (
        _write_snqi_diagnostics_artifacts(reports_dir, snqi_diagnostics_payload)
    )
    snqi_hard_fail = (
        cfg.paper_facing
        and cfg.snqi_contract.enabled
        and cfg.snqi_contract.enforcement in {"error", "enforce"}
        and contract_eval.status == "fail"
    )
    # Issue #5240: a soft contract warning (enforcement=warn with status warn/fail) must NOT
    # change the exit code. It is surfaced in the summary as ``soft_contract_warning: true``
    # plus a ``warnings[]`` entry, and the campaign still counts as benchmark_success when all
    # planner rows succeeded. Only hard enforcement levels stay fatal (handled by snqi_hard_fail).
    soft_contract_warning = bool(
        cfg.paper_facing
        and cfg.snqi_contract.enabled
        and soft_contract_warning_active(cfg.snqi_contract.enforcement, contract_eval.status)
    )
    if snqi_hard_fail:
        warnings.append(
            "SNQI contract status=fail with "
            f"snqi_contract.enforcement={cfg.snqi_contract.enforcement}; "
            "campaign marked with hard contract warning."
        )
    elif soft_contract_warning:
        warnings.append(
            "SNQI contract status="
            f"{contract_eval.status} with snqi_contract.enforcement=warn; campaign marked with soft contract warning."
        )

    campaign_summary = {
        "campaign": {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "campaign_id": campaign_id,
            "name": cfg.name,
            "created_at_utc": campaign_started_at_utc,
            "started_at_utc": campaign_started_at_utc,
            "finished_at_utc": campaign_finished_at_utc,
            "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
            "scenario_matrix_hash": scenario_hash,
            "git_hash": git_meta.get("commit", "unknown"),
            "invoked_command": invoked_command,
            "runtime_sec": runtime_sec,
            "episodes_per_second": (total_episodes / runtime_sec) if runtime_sec > 0 else 0.0,
            "total_episodes": total_episodes,
            "successful_runs": successful_runs,
            "total_runs": len(run_entries),
            "non_success_runs": campaign_outcome.non_success_runs,
            "accepted_unavailable_runs": campaign_outcome.accepted_unavailable_runs,
            "unexpected_failed_runs": campaign_outcome.unexpected_failed_runs,
            "campaign_execution_status": campaign_status_axes.campaign_execution_status,
            "evidence_status": campaign_status_axes.evidence_status,
            "row_status_summary": row_status_summary,
            "benchmark_success": benchmark_success,
            "status": campaign_outcome.status,
            "status_reason": campaign_outcome.status_reason,
            "exit_code": campaign_outcome.exit_code,
            "benchmark_success_basis": success_counters["benchmark_success_basis"],
            "core_successful_runs": success_counters["core_successful_runs"],
            "core_total_runs": success_counters["core_total_runs"],
            "paper_interpretation_profile": cfg.paper_interpretation_profile,
            "kinematics_matrix": list(kinematics_matrix),
            "holonomic_command_mode": cfg.holonomic_command_mode,
            "paper_facing": bool(cfg.paper_facing),
            "paper_profile_version": cfg.paper_profile_version,
            "observation_noise": normalize_observation_noise_spec(cfg.observation_noise),
            "observation_noise_hash": observation_noise_hash(
                normalize_observation_noise_spec(cfg.observation_noise)
            ),
            "amv_profile_name": cfg.amv_profile.name,
            "amv_contract_version": cfg.amv_profile.contract_version,
            "amv_coverage_enforcement": cfg.amv_profile.coverage_enforcement,
            "amv_coverage_status": str(
                (manifest_payload or {}).get("amv_coverage_status", "unknown")
            ),
            "scenario_amv_overrides": {
                scenario_name: dict(values)
                for scenario_name, values in sorted(cfg.scenario_amv_overrides.items())
            },
            "scenario_candidates": list(cfg.scenario_candidates.names),
            "scenario_candidates_selection_name": cfg.scenario_candidates.selection_name,
            "synthetic_actuation_profile": _synthetic_actuation_metadata(
                cfg.synthetic_actuation_profile
            ),
            "latency_stress_profile": _latency_stress_metadata(
                cfg.latency_stress_profile,
                dt=cfg.dt,
            ),
            "latency_stress_metrics": (
                not_available_latency_metrics() if cfg.latency_stress_profile is not None else None
            ),
            "comparability_mapping_path": manifest_payload.get("comparability_mapping_path"),
            "comparability_mapping_version": manifest_payload.get("comparability_mapping_version"),
            "comparability_mapping_hash": manifest_payload.get("comparability_mapping_hash"),
            "repository_url": cfg.repository_url,
            "release_tag": release_tag_value,
            "doi": cfg.doi,
            "release_url": release_url,
            "release_asset_url": release_asset_url,
            "doi_url": doi_url,
            "snqi_weights_version": (
                cfg.snqi_weights_path.stem if cfg.snqi_weights_path is not None else "default"
            ),
            "snqi_weights_sha256": weights_sha256,
            "snqi_baseline_version": (
                cfg.snqi_baseline_path.stem if cfg.snqi_baseline_path is not None else "derived"
            ),
            "snqi_baseline_sha256": baseline_sha256,
            "snqi_contract_status": contract_eval.status,
            "snqi_contract_rank_alignment_spearman": contract_eval.rank_alignment_spearman,
            "snqi_contract_outcome_separation": contract_eval.outcome_separation,
            "snqi_contract_dominant_component": contract_eval.dominant_component,
            "snqi_contract_dominant_component_mean_abs": (
                contract_eval.dominant_component_mean_abs
            ),
            "snqi_positioning_recommendation": positioning.get("recommendation"),
            "snqi_positioning_claim_scope": positioning.get("claim_scope"),
        },
        "planner_rows": planner_rows,
        "runs": run_entries,
        "warnings": warnings,
        "soft_contract_warning": soft_contract_warning,
        "artifacts": {
            "campaign_manifest": _repo_relative(campaign_root / "campaign_manifest.json"),
            "campaign_summary_json": _repo_relative(summary_json_path),
            "campaign_credibility_scorecard_json": _repo_relative(credibility_scorecard_json_path),
            "campaign_table_csv": _repo_relative(csv_path),
            "campaign_table_md": _repo_relative(md_table_path),
            "campaign_table_core_csv": _repo_relative(core_csv_path),
            "campaign_table_core_md": _repo_relative(core_md_path),
            "campaign_table_experimental_csv": _repo_relative(experimental_csv_path),
            "campaign_table_experimental_md": _repo_relative(experimental_md_path),
            "kinematics_parity_csv": _repo_relative(parity_csv_path),
            "kinematics_parity_md": _repo_relative(parity_md_path),
            "kinematics_skipped_combinations_csv": _repo_relative(skipped_csv_path),
            "kinematics_skipped_combinations_md": _repo_relative(skipped_md_path),
            "matrix_summary_json": _repo_relative(matrix_summary_json_path),
            "matrix_summary_csv": _repo_relative(matrix_summary_csv_path),
            "amv_coverage_json": _repo_relative(amv_coverage_json_path),
            "amv_coverage_md": _repo_relative(amv_coverage_md_path),
            "comparability_json": (
                _repo_relative(comparability_json_path) if comparability_json_path else None
            ),
            "comparability_md": (
                _repo_relative(comparability_md_path) if comparability_md_path else None
            ),
            "seed_variability_json": _repo_relative(seed_variability_json_path),
            "seed_variability_csv": _repo_relative(seed_variability_csv_path),
            "seed_episode_rows_csv": _repo_relative(seed_episode_rows_csv_path),
            "statistical_sufficiency_json": _repo_relative(statistical_sufficiency_json_path),
            "actuation_envelope_json": (
                _repo_relative(actuation_envelope_json_path)
                if actuation_envelope_json_path is not None
                else None
            ),
            "actuation_envelope_md": (
                _repo_relative(actuation_envelope_md_path)
                if actuation_envelope_md_path is not None
                else None
            ),
            "preflight_validate_config": _repo_relative(validate_config_path),
            "preflight_preview_scenarios": _repo_relative(preview_scenarios_path),
            "scenario_breakdown_csv": _repo_relative(scenario_csv_path),
            "scenario_breakdown_md": _repo_relative(scenario_md_path),
            "scenario_family_breakdown_csv": _repo_relative(family_csv_path),
            "scenario_family_breakdown_md": _repo_relative(family_md_path),
            "campaign_report_md": _repo_relative(report_md_path),
            "expected_release_archive": expected_archive_name,
            "release_url": release_url,
            "release_asset_url": release_asset_url,
            "doi_url": doi_url,
            "snqi_diagnostics_json": _repo_relative(snqi_diagnostics_json_path),
            "snqi_diagnostics_md": _repo_relative(snqi_diagnostics_md_path),
            "snqi_sensitivity_csv": _repo_relative(snqi_sensitivity_csv_path),
            "assurance_fragment_json": _repo_relative(reports_dir / "assurance_fragment.json"),
            "assurance_fragment_md": _repo_relative(reports_dir / "assurance_fragment.md"),
            "assurance_fragment_svg": _repo_relative(reports_dir / "assurance_fragment.svg"),
        },
    }

    # Write run-level files and the final campaign_manifest.json before the publication
    # bundle export so the bundle copies the fully-evaluated manifest (including
    # snqi_positioning_recommendation) rather than the placeholder written at campaign start.
    run_meta = {
        "repo": {
            "remote": git_meta.get("remote", "unknown"),
            "branch": git_meta.get("branch", "unknown"),
            "commit": git_meta.get("commit", "unknown"),
        },
        "matrix_path": _repo_relative(cfg.scenario_matrix_path),
        "scenario_matrix_hash": scenario_hash,
        "latency_stress_profile": _latency_stress_metadata(
            cfg.latency_stress_profile,
            dt=cfg.dt,
        ),
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": resolved_seeds,
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
        "preflight_artifacts": {
            "validate_config": _repo_relative(validate_config_path),
            "preview_scenarios": _repo_relative(preview_scenarios_path),
            "amv_coverage_json": _repo_relative(amv_coverage_json_path),
            "amv_coverage_md": _repo_relative(amv_coverage_md_path),
            "comparability_json": (
                _repo_relative(comparability_json_path) if comparability_json_path else None
            ),
            "comparability_md": (
                _repo_relative(comparability_md_path) if comparability_md_path else None
            ),
            "seed_variability_json": _repo_relative(seed_variability_json_path),
            "seed_variability_csv": _repo_relative(seed_variability_csv_path),
            "seed_episode_rows_csv": _repo_relative(seed_episode_rows_csv_path),
            "statistical_sufficiency_json": _repo_relative(statistical_sufficiency_json_path),
            "actuation_envelope_json": (
                _repo_relative(actuation_envelope_json_path)
                if actuation_envelope_json_path is not None
                else None
            ),
            "actuation_envelope_md": (
                _repo_relative(actuation_envelope_md_path)
                if actuation_envelope_md_path is not None
                else None
            ),
        },
        "synthetic_actuation_artifacts": {
            "json": (
                _repo_relative(actuation_envelope_json_path)
                if actuation_envelope_json_path is not None
                else None
            ),
            "md": (
                _repo_relative(actuation_envelope_md_path)
                if actuation_envelope_md_path is not None
                else None
            ),
        },
        "snqi_artifacts": {
            "diagnostics_json": _repo_relative(snqi_diagnostics_json_path),
            "diagnostics_md": _repo_relative(snqi_diagnostics_md_path),
            "sensitivity_csv": _repo_relative(snqi_sensitivity_csv_path),
        },
        "seed_variability_artifacts": {
            "json": _repo_relative(seed_variability_json_path),
            "csv": _repo_relative(seed_variability_csv_path),
            "seed_episode_rows_csv": _repo_relative(seed_episode_rows_csv_path),
            "statistical_sufficiency_json": _repo_relative(statistical_sufficiency_json_path),
        },
        "seed_variability": {
            "metrics": list(_SEED_VARIABILITY_METRICS),
            "row_count": int(seed_variability_payload.get("row_count", 0)),
            "bootstrap_method": str(
                seed_variability_payload.get("confidence", {}).get("method", "")
            ),
            "bootstrap_level": float(
                seed_variability_payload.get("confidence", {}).get("confidence", 0.0) or 0.0
            ),
            "bootstrap_samples": int(
                seed_variability_payload.get("confidence", {}).get("bootstrap_samples", 0) or 0
            ),
            "seed": int(
                seed_variability_payload.get("confidence", {}).get("bootstrap_seed", 0) or 0
            ),
        },
        "campaign_id": campaign_id,
        "started_at_utc": campaign_started_at_utc,
        "finished_at_utc": campaign_finished_at_utc,
        "invoked_command": invoked_command,
        "runtime_sec": runtime_sec,
        "episodes_per_second": (total_episodes / runtime_sec) if runtime_sec > 0 else 0.0,
    }
    run_manifest = {
        "git_hash": git_meta.get("commit", "unknown"),
        "scenario_matrix_hash": scenario_hash,
        "runtime_sec": runtime_sec,
        "episodes_per_second": (total_episodes / runtime_sec) if runtime_sec > 0 else 0.0,
    }
    _write_json(campaign_root / "run_meta.json", run_meta)
    _write_json(campaign_root / "manifest.json", run_manifest)
    _write_json(
        campaign_root / "campaign_manifest.json",
        {
            **manifest_payload,
            "runtime_sec": runtime_sec,
            "finished_at_utc": campaign_finished_at_utc,
            "snqi_contract_status": contract_eval.status,
            "snqi_positioning_recommendation": positioning.get("recommendation"),
            "snqi_positioning_claim_scope": positioning.get("claim_scope"),
            "artifacts": {
                **dict(manifest_payload.get("artifacts") or {}),
                "seed_variability_json": _repo_relative(seed_variability_json_path),
                "seed_variability_csv": _repo_relative(seed_variability_csv_path),
                "seed_episode_rows_csv": _repo_relative(seed_episode_rows_csv_path),
                "statistical_sufficiency_json": _repo_relative(statistical_sufficiency_json_path),
                "actuation_envelope_json": (
                    _repo_relative(actuation_envelope_json_path)
                    if actuation_envelope_json_path is not None
                    else None
                ),
                "actuation_envelope_md": (
                    _repo_relative(actuation_envelope_md_path)
                    if actuation_envelope_md_path is not None
                    else None
                ),
                "snqi_diagnostics_json": _repo_relative(snqi_diagnostics_json_path),
                "snqi_diagnostics_md": _repo_relative(snqi_diagnostics_md_path),
                "snqi_sensitivity_csv": _repo_relative(snqi_sensitivity_csv_path),
                "assurance_fragment_json": _repo_relative(reports_dir / "assurance_fragment.json"),
                "assurance_fragment_md": _repo_relative(reports_dir / "assurance_fragment.md"),
                "assurance_fragment_svg": _repo_relative(reports_dir / "assurance_fragment.svg"),
            },
            "seed_variability": {
                **dict(run_meta.get("seed_variability") or {}),
            },
        },
    )

    publication_payload: dict[str, Any] | None = None
    if (
        cfg.export_publication_bundle
        and not skip_publication_bundle
        and not snqi_hard_fail
        and benchmark_success
    ):
        publication_dir = get_artifact_category_path("benchmarks") / "publication"
        bundle_name = f"{campaign_id}_publication_bundle"
        try:
            bundle = dependencies.export_publication_bundle(
                campaign_root,
                publication_dir,
                bundle_name=bundle_name,
                include_videos=cfg.include_videos_in_publication,
                repository_url=cfg.repository_url,
                release_tag=cfg.release_tag,
                doi=cfg.doi,
                overwrite=cfg.overwrite_publication_bundle,
            )
            publication_payload = {
                "bundle_dir": _repo_relative(bundle.bundle_dir),
                "archive_path": _repo_relative(bundle.archive_path),
                "manifest_path": _repo_relative(bundle.manifest_path),
                "checksums_path": _repo_relative(bundle.checksums_path),
                "file_count": bundle.file_count,
                "total_bytes": bundle.total_bytes,
            }
            campaign_summary["publication_bundle"] = publication_payload
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            warnings.append(f"Publication bundle export failed: {exc}")
    elif (
        cfg.export_publication_bundle
        and not skip_publication_bundle
        and not snqi_hard_fail
        and not benchmark_success
    ):
        warnings.append("Publication bundle export skipped because benchmark_success=false.")

    campaign_summary["credibility_scorecard"] = build_campaign_credibility_scorecard(
        campaign_summary
    )
    _write_json(credibility_scorecard_json_path, campaign_summary["credibility_scorecard"])
    _write_json(summary_json_path, campaign_summary)
    write_campaign_report(report_md_path, campaign_summary)

    # Export assurance fragment
    try:
        release_gate_report = None
        for gate_report_path in reports_dir.glob("*release_gate*.json"):
            try:
                with gate_report_path.open("r", encoding="utf-8") as f:
                    release_gate_report = json.load(f)
                break
            except (OSError, json.JSONDecodeError):
                continue

        fragment = build_assurance_fragment(
            campaign_summary,
            repo_root=get_repository_root(),
            release_gate_report=release_gate_report,
        )
        validate_assurance_fragment(fragment)
        write_assurance_fragment(reports_dir, fragment, repo_root=get_repository_root())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        warnings.append(f"Assurance fragment export failed: {exc}")

    if snqi_hard_fail:
        raise RuntimeError(
            f"SNQI contract failed with enforcement={cfg.snqi_contract.enforcement}; "
            f"rank_alignment={contract_eval.rank_alignment_spearman:.4f}, "
            f"outcome_separation={contract_eval.outcome_separation:.4f}. "
            f"See diagnostics: {_repo_relative(snqi_diagnostics_json_path)}"
        )

    logger.info(
        "Camera-ready campaign finished id={} runs={} episodes={} out={}",
        campaign_id,
        len(run_entries),
        total_episodes,
        campaign_root,
    )

    return {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "summary_json": str(summary_json_path),
        "table_csv": str(csv_path),
        "table_md": str(md_table_path),
        "report_md": str(report_md_path),
        "snqi_diagnostics_json": str(snqi_diagnostics_json_path),
        "snqi_diagnostics_md": str(snqi_diagnostics_md_path),
        "snqi_sensitivity_csv": str(snqi_sensitivity_csv_path),
        "assurance_fragment_json": str(reports_dir / "assurance_fragment.json"),
        "assurance_fragment_md": str(reports_dir / "assurance_fragment.md"),
        "assurance_fragment_svg": str(reports_dir / "assurance_fragment.svg"),
        "matrix_summary_json": str(matrix_summary_json_path),
        "matrix_summary_csv": str(matrix_summary_csv_path),
        "seed_variability_json": str(seed_variability_json_path),
        "seed_variability_csv": str(seed_variability_csv_path),
        "seed_episode_rows_csv": str(seed_episode_rows_csv_path),
        "statistical_sufficiency_json": str(statistical_sufficiency_json_path),
        "actuation_envelope_json": (
            str(actuation_envelope_json_path) if actuation_envelope_json_path is not None else None
        ),
        "actuation_envelope_md": (
            str(actuation_envelope_md_path) if actuation_envelope_md_path is not None else None
        ),
        "total_runs": len(run_entries),
        "successful_runs": successful_runs,
        "non_success_runs": campaign_outcome.non_success_runs,
        "accepted_unavailable_runs": campaign_outcome.accepted_unavailable_runs,
        "unexpected_failed_runs": campaign_outcome.unexpected_failed_runs,
        "campaign_execution_status": campaign_status_axes.campaign_execution_status,
        "evidence_status": campaign_status_axes.evidence_status,
        "row_status_summary": row_status_summary,
        "benchmark_success": benchmark_success,
        "status": campaign_outcome.status,
        "status_reason": campaign_outcome.status_reason,
        "exit_code": campaign_outcome.exit_code,
        "benchmark_success_basis": success_counters["benchmark_success_basis"],
        "core_successful_runs": success_counters["core_successful_runs"],
        "core_total_runs": success_counters["core_total_runs"],
        "total_episodes": total_episodes,
        "runtime_sec": runtime_sec,
        "publication_bundle": publication_payload,
        "warnings": warnings,
        "soft_contract_warning": soft_contract_warning,
    }
