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
from robot_sf.benchmark.camera_ready._resume_plan import (
    ArmResumeVerdict,
    build_resume_plan,
    emit_resume_plan_log,
    verify_resume_context,
    write_resume_plan,
)
from robot_sf.benchmark.camera_ready._run_state import (
    _build_arm_rollup,
    _campaign_success_counters,
    validate_campaign_integrity,
)
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
from robot_sf.benchmark.fairness_contract import build_fairness_report, emit_fairness_annotations
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
from robot_sf.benchmark.result_provenance import build_execution_context_provenance
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
    """Injected campaign collaborators (preflight, batch runner, aggregates, publication bundle)."""

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
    """Return the four campaign runtime collaborators.

    Lazily imports the canonical implementation for any collaborator that was not injected.
    """

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
    """Collected output for one planner: run entries, planner rows, warnings, seed-variability records."""

    run_entries: list[dict[str, Any]]
    planner_rows: list[dict[str, Any]]
    warnings: list[str]
    seed_variability_records: list[dict[str, Any]]


@dataclass(frozen=True)
class _CampaignPlannerMatrixContext:
    """Shared matrix context iterated across planner variants."""

    cfg: CampaignConfig
    scenarios: list[Any]
    snqi_weights: dict[str, Any] | None
    snqi_baseline: dict[str, Any] | None
    runs_dir: Path
    dependencies: _CampaignRuntimeDependencies


@dataclass(frozen=True)
class _CampaignPlannerVariantResult:
    """Result of one planner variant, extending run results with a stop-requested flag."""

    run_entries: list[dict[str, Any]]
    planner_rows: list[dict[str, Any]]
    warnings: list[str]
    seed_variability_records: list[dict[str, Any]]
    stop_requested: bool


@dataclass(frozen=True)
class _CampaignPlannerVariantRun:
    """Resolved parameters for one planner/kinematics batch run."""

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
    """Outcome of one batch execution: status string, runner summary, and warnings."""

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


def _resolve_arm_safety_wrapper(
    *, cfg: CampaignConfig, planner: PlannerSpec
) -> dict[str, Any] | None:
    """Resolve the effective safety-wrapper mapping for one campaign arm.

    The per-arm ``PlannerSpec.safety_wrapper`` overrides the campaign-level
    ``CampaignConfig.safety_wrapper`` default (issue #3501 / #4830). The factorial
    ``planner x {wrapper_off, wrapper_on}`` design is encoded as distinct arms whose
    per-arm value wins, while a campaign can still pin a default for arms that leave
    the wrapper unset. ``None`` (the default) keeps the wrapper off at runtime.

    Returns:
        The resolved safety-wrapper mapping, or ``None`` when neither arm nor campaign
        declares one (wrapper stays off).
    """
    if planner.safety_wrapper is not None:
        return dict(planner.safety_wrapper)
    if cfg.safety_wrapper is not None:
        return dict(cfg.safety_wrapper)
    return None


def _execute_campaign_planner_batch(
    context: _CampaignPlannerMatrixContext,
    planner: PlannerSpec,
    run: _CampaignPlannerVariantRun,
) -> _CampaignPlannerBatchResult:
    """Return the batch result of executing one planner/kinematics run via the injected runner.

    Classifies availability from the summary and fails closed on checkpoint
    fallback when ``error`` enforcement is configured.
    """

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
            safety_wrapper=_resolve_arm_safety_wrapper(cfg=cfg, planner=planner),
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
    """Return one planner/kinematics variant's resolved output paths and runtime settings.

    Applies per-planner overrides for workers, horizon, and dt, and scopes
    scenarios to the variant's kinematics.
    """

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
    """Return a ``not_available`` summary block for a dependency-gated planner.

    Records the skip reason and algorithm-readiness tier without executing the arm.
    """

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


def _resolve_campaign_planner_batch_result(
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    run: _CampaignPlannerVariantRun,
    resume_verdict: ArmResumeVerdict | None,
) -> _CampaignPlannerBatchResult:
    """Return an executed, gated, or completed-resume arm result."""
    if resume_verdict is not None:
        if resume_verdict.verdict != "skip-complete":
            raise ValueError(
                "resume_verdict must be skip-complete when bypassing arm execution, "
                f"got {resume_verdict.verdict!r}"
            )
        if resume_verdict.episodes_path != run.episodes_path:
            raise ValueError(
                "resume-plan episodes path does not match scheduler arm path: "
                f"{resume_verdict.episodes_path} != {run.episodes_path}"
            )
        logger.info(
            "Skipping completed campaign arm: planner={} kinematics={} episodes={}/{}",
            planner.key,
            run.kinematics,
            resume_verdict.episodes_found,
            resume_verdict.expected_total,
        )
        if not isinstance(resume_verdict.prior_summary, dict):
            raise ValueError(
                "completed resume arm requires a JSON-object summary.json: "
                f"{run.planner_dir / 'summary.json'}"
            )
        summary = dict(resume_verdict.prior_summary)
        summary.setdefault("total_jobs", resume_verdict.expected_total)
        summary.setdefault("written", resume_verdict.episodes_found)
        summary.setdefault("failed_jobs", 0)
        summary.setdefault("failures", [])
        summary.setdefault("out_path", str(run.episodes_path))
        return _CampaignPlannerBatchResult(
            status=str(summary.get("status", "ok")),
            summary=summary,
            warnings=[],
        )
    if planner.availability_gate == "dependency_gated":
        return _CampaignPlannerBatchResult(
            status="not_available",
            summary=_dependency_gated_planner_summary(context, planner=planner, run=run),
            warnings=[],
        )
    return _execute_campaign_planner_batch(context, planner, run)


def _run_campaign_planner_variant(  # noqa: PLR0915
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    kinematics: str,
    active_observation_mode: str,
    resume_verdict: ArmResumeVerdict | None = None,
) -> _CampaignPlannerVariantResult:
    """Return one planner variant's end-to-end result.

    Prepares, executes or resumes, and aggregates one variant, returning its run
    entries, planner rows, warnings, and seed-variability records.
    """

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
        log_run=resume_verdict is None and planner.availability_gate != "dependency_gated",
    )

    planner_started_at_utc = _utc_now()
    planner_start = time.perf_counter()
    batch_result = _resolve_campaign_planner_batch_result(
        context,
        planner=planner,
        run=run,
        resume_verdict=resume_verdict,
    )
    status = batch_result.status
    summary = batch_result.summary
    warnings.extend(batch_result.warnings)
    aggregates: dict[str, Any] | None = None

    if resume_verdict is None:
        planner_finished_at_utc = _utc_now()
        runtime_sec = float(max(1e-9, time.perf_counter() - planner_start))
        episodes_written = int(summary.get("written", 0))
    else:
        planner_started_at_utc = str(summary.get("started_at_utc") or planner_started_at_utc)
        planner_finished_at_utc = str(summary.get("finished_at_utc") or planner_started_at_utc)
        runtime_sec = float(summary.get("runtime_sec", 0.0))
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


def _prepare_subprocess_arm_run(
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    kinematics: str,
    active_observation_mode: str,
) -> tuple[_CampaignPlannerVariantRun, str]:
    """Prepare run directory, write scoped scenarios, build and serialize arm params.

    Returns:
        Tuple of (run, arm_params_json) where arm_params_json is the JSON string
        to pass to the subprocess worker via stdin.
    """
    cfg = context.cfg

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

    scoped_scenarios_path = run.planner_dir / "scoped_scenarios.json"
    scoped_scenarios_path.write_text(
        json.dumps(run.scoped_scenarios, default=str),
        encoding="utf-8",
    )

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
        resume=cfg.resume,
        scoped_scenarios_path=scoped_scenarios_path,
        safety_wrapper=_resolve_arm_safety_wrapper(cfg=cfg, planner=planner),
    )

    arm_params_json = _serialize_subprocess_arm_params(arm_params)
    return run, arm_params_json


def _execute_subprocess_arm_and_parse(
    arm_params_json: str,
    *,
    planner: PlannerSpec,
    kinematics: str,
) -> dict[str, Any]:
    """Execute the subprocess worker and parse its JSON output.

    Returns:
        Parsed subprocess result dict with keys: summary, cleanup_metrics,
        warnings, episodes_total.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "robot_sf.benchmark.camera_ready.resource_lifecycle"],
        input=arm_params_json,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode != 0:
        logger.error(
            "Subprocess arm exited non-zero: planner={} kinematics={} returncode={} stderr={}; "
            "attempting to parse its structured result",
            planner.key,
            kinematics,
            proc.returncode,
            proc.stderr,
        )

    warnings: list[str] = []
    try:
        subprocess_result = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse subprocess output: {}", exc)
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
    return subprocess_result


def _build_subprocess_episode_data(
    run: _CampaignPlannerVariantRun,
    status: str,
    *,
    planner: PlannerSpec,
    kinematics: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read episodes from disk and build seed-variability records."""  # noqa: DOC201
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
    return records, seed_variability_records


def _compute_subprocess_variant_aggregates(
    context: _CampaignPlannerMatrixContext,
    records: list[dict[str, Any]],
    *,
    planner: PlannerSpec,
    kinematics: str,
    status: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Compute aggregates with confidence intervals for a subprocess variant."""  # noqa: DOC201
    warnings: list[str] = []
    aggregates: dict[str, Any] | None = None
    if records and status == "ok":
        cfg = context.cfg
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
    return aggregates, warnings


def _build_subprocess_variant_output(  # noqa: PLR0913
    context: _CampaignPlannerMatrixContext,
    *,
    planner: PlannerSpec,
    kinematics: str,
    active_observation_mode: str,
    run: _CampaignPlannerVariantRun,
    status: str,
    summary: dict[str, Any],
    aggregates: dict[str, Any] | None,
    records: list[dict[str, Any]],
    cleanup_metrics: dict[str, Any],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, list[str]]:
    """Build run entry, planner rows, and check stop_on_failure."""  # noqa: DOC201
    cfg = context.cfg
    row = _planner_report_row(
        planner,
        summary,
        aggregates,
        kinematics=kinematics,
        synthetic_actuation_profile=cfg.synthetic_actuation_profile,
        records=records,
    )
    planner_rows = [row]
    planner_started_at_utc = summary.get("started_at_utc", _utc_now())
    planner_finished_at_utc = summary.get("finished_at_utc", _utc_now())
    runtime_sec = summary.get("runtime_sec", 0.0)
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

    return run_entries, planner_rows, stop_requested, warnings


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

    Returns:
        Campaign variant result with run_entries, planner_rows, etc.
    """
    run, arm_params_json = _prepare_subprocess_arm_run(
        context,
        planner=planner,
        kinematics=kinematics,
        active_observation_mode=active_observation_mode,
    )
    subprocess_result = _execute_subprocess_arm_and_parse(
        arm_params_json, planner=planner, kinematics=kinematics
    )
    summary = subprocess_result.get("summary", {})
    cleanup_metrics = subprocess_result.get("cleanup_metrics", {})
    warnings: list[str] = []
    warnings.extend(subprocess_result.get("warnings", []))
    status = summary.get("status", "unknown")

    records, seed_variability_records = _build_subprocess_episode_data(
        run, status, planner=planner, kinematics=kinematics
    )
    aggregates, agg_warnings = _compute_subprocess_variant_aggregates(
        context, records, planner=planner, kinematics=kinematics, status=status
    )
    warnings.extend(agg_warnings)
    run_entries, planner_rows, stop_requested, warnings = _build_subprocess_variant_output(
        context,
        planner=planner,
        kinematics=kinematics,
        active_observation_mode=active_observation_mode,
        run=run,
        status=status,
        summary=summary,
        aggregates=aggregates,
        records=records,
        cleanup_metrics=cleanup_metrics,
        warnings=warnings,
    )
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
    resume_verdicts: list[ArmResumeVerdict] | None = None,
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
        resume_verdicts: Per-arm resume plan used to bypass completed arms.

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
    resume_by_arm = {
        (verdict.planner_key, verdict.kinematics): verdict for verdict in (resume_verdicts or [])
    }
    stop_requested = False

    for planner in cfg.planners:
        if not planner.enabled:
            continue
        active_observation_mode = planner.observation_mode or cfg.observation_mode
        for kinematics in kinematics_matrix:
            resume_verdict = resume_by_arm.get((planner.key, kinematics))
            # Dispatch based on arm_isolation mode (issue #4826)
            use_subprocess = effective_arm_isolation == "subprocess"

            if resume_verdict is not None and resume_verdict.verdict == "skip-complete":
                variant_result = _run_campaign_planner_variant(
                    context,
                    planner=planner,
                    kinematics=kinematics,
                    active_observation_mode=active_observation_mode,
                    resume_verdict=resume_verdict,
                )
            elif use_subprocess:
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
    """Return one row per run entry whose preflight was skipped.

    Records planner key, algo, kinematics, and the skip reason for reporting.
    """

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


def _emit_resume_plan_preflight(
    *,
    cfg: CampaignConfig,
    campaign_id: str,
    config_hash: str,
    campaign_root: Path,
    runs_dir: Path,
    scenarios: list[Any],
) -> list[ArmResumeVerdict]:
    """Emit a resume plan before a resumed campaign executes (issue #5392).

    When ``cfg.resume`` is enabled and any arm directory already exists, this
    verifies the campaign context matches and writes ``resume_plan.json`` so the
    operator can sanity-check projected walltime.

    Returns:
        Per-arm resume verdicts, or an empty list when no prior arms exist.

    Raises:
        ResumeMismatchError: If the campaign-id or config-hash on disk does not
            match the current invocation.
    """
    if not cfg.resume:
        return []

    has_prior_arms = any(r.is_dir() for r in runs_dir.iterdir()) if runs_dir.exists() else False
    if not has_prior_arms:
        return []

    # Fail-closed context check before any work begins.
    verify_resume_context(
        campaign_root,
        campaign_id=campaign_id,
        config_hash=config_hash,
    )

    # Build plan from existing arm directories.
    planners = [
        {
            "key": planner.key,
            "enabled": planner.enabled,
        }
        for planner in cfg.planners
    ]
    kinematics = list(cfg.kinematics_matrix) or ("differential_drive",)
    verdicts = build_resume_plan(
        runs_dir,
        planners=planners,
        kinematics_matrix=list(kinematics),
        scenarios=scenarios,
    )

    emit_resume_plan_log(verdicts)
    write_resume_plan(
        campaign_root,
        config_hash=config_hash,
        campaign_id=campaign_id,
        verdicts=verdicts,
    )
    return verdicts


_CAMPAIGN_TABLE_HEADERS = (
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
    "fairness_mismatch_flags",
    "fairness_in_ranking_subset",
)


def _write_campaign_matrix_tables(
    reports_dir: Path,
    planner_rows: list[dict[str, Any]],
    cfg: CampaignConfig,
) -> dict[str, Path]:
    """Write campaign-level table artifacts (full, core, experimental).

    Returns:
        Dict with keys: csv_path, md_table_path, core_csv_path, core_md_path,
        experimental_csv_path, experimental_md_path.
    """
    csv_path, md_table_path = _write_table_artifacts(
        reports_dir,
        "campaign_table",
        planner_rows,
        headers=_CAMPAIGN_TABLE_HEADERS,
    )
    core_rows, experimental_rows = _split_campaign_table_rows(planner_rows, cfg)
    core_headers = (
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
    )
    core_csv_path, core_md_path = _write_table_artifacts(
        reports_dir, "campaign_table_core", core_rows, headers=core_headers
    )
    experimental_csv_path, experimental_md_path = _write_table_artifacts(
        reports_dir,
        "campaign_table_experimental",
        experimental_rows,
        headers=core_headers,
    )
    return {
        "csv_path": csv_path,
        "md_table_path": md_table_path,
        "core_csv_path": core_csv_path,
        "core_md_path": core_md_path,
        "experimental_csv_path": experimental_csv_path,
        "experimental_md_path": experimental_md_path,
    }


def _split_campaign_table_rows(
    planner_rows: list[dict[str, Any]], cfg: CampaignConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate core and experimental rows using the active report policy."""  # noqa: DOC201
    field, core_value = (
        ("planner_group", "core") if cfg.paper_facing else ("readiness_tier", "baseline-ready")
    )
    core_rows = [row for row in planner_rows if str(row.get(field)) == core_value]
    experimental_rows = [row for row in planner_rows if str(row.get(field)) != core_value]
    return core_rows, experimental_rows


def _write_scenario_breakdown_tables(
    reports_dir: Path,
    run_entries: list[dict[str, Any]],
    scenarios: list[Any],
) -> dict[str, Path]:
    """Write scenario-breakdown and family-breakdown table artifacts.

    Returns:
        Dict with keys: scenario_csv_path, scenario_md_path, family_csv_path,
        family_md_path, scenario_amv_lookup.
    """
    scenario_amv_lookup = _build_scenario_amv_lookup(scenarios)
    scenario_rows, family_rows = _build_breakdown_rows(
        run_entries,
        scenario_amv_lookup=scenario_amv_lookup,
    )
    breakdown_headers = (
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
    )
    scenario_csv_path, scenario_md_path = _write_table_artifacts(
        reports_dir, "scenario_breakdown", scenario_rows, headers=breakdown_headers
    )
    family_csv_path, family_md_path = _write_table_artifacts(
        reports_dir,
        "scenario_family_breakdown",
        family_rows,
        headers=breakdown_headers,
    )
    return {
        "scenario_csv_path": scenario_csv_path,
        "scenario_md_path": scenario_md_path,
        "family_csv_path": family_csv_path,
        "family_md_path": family_md_path,
        "scenario_amv_lookup": scenario_amv_lookup,
    }


def _write_parity_and_skipped_tables(
    reports_dir: Path,
    planner_rows: list[dict[str, Any]],
    run_entries: list[dict[str, Any]],
) -> dict[str, Path]:
    """Write kinematics parity table and skipped-combinations table.

    Returns:
        Dict with keys: parity_csv_path, parity_md_path, skipped_csv_path,
        skipped_md_path.
    """
    parity_rows = _build_kinematics_parity_rows(planner_rows)
    parity_headers = (
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
    )
    parity_csv_path, parity_md_path = _write_table_artifacts(
        reports_dir, "kinematics_parity_table", parity_rows, headers=parity_headers
    )
    skipped_combo_rows = _build_skipped_combo_rows(run_entries)
    skipped_csv_path, skipped_md_path = _write_table_artifacts(
        reports_dir,
        "kinematics_skipped_combinations",
        skipped_combo_rows,
        headers=("planner_key", "algo", "kinematics", "reason"),
    )
    return {
        "parity_csv_path": parity_csv_path,
        "parity_md_path": parity_md_path,
        "skipped_csv_path": skipped_csv_path,
        "skipped_md_path": skipped_md_path,
    }


def _build_kinematics_parity_rows(planner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and sort planner rows for the kinematics parity table."""  # noqa: DOC201
    fields = (
        ("planner_key", ""),
        ("algo", ""),
        ("human_model_variant", ""),
        ("human_model_source", ""),
        ("planner_group", "experimental"),
        ("kinematics", ""),
        ("execution_mode", "unknown"),
        ("status", "unknown"),
        ("success_mean", "nan"),
        ("success_ci_low", "nan"),
        ("success_ci_high", "nan"),
        ("collisions_mean", "nan"),
        ("ped_collision_count_mean", "nan"),
        ("obstacle_collision_count_mean", "nan"),
        ("total_collision_count_mean", "nan"),
        ("collision_ci_low", "nan"),
        ("collision_ci_high", "nan"),
        ("near_misses_mean", "nan"),
        ("comfort_exposure_mean", "nan"),
        ("snqi_mean", "nan"),
        ("snqi_ci_low", "nan"),
        ("snqi_ci_high", "nan"),
        ("projection_rate", "0.0000"),
        ("infeasible_rate", "0.0000"),
    )
    return sorted(
        [
            {key: str(row.get(key, default)) for key, default in fields}
            | {"episodes": int(row.get("episodes", 0))}
            for row in planner_rows
        ],
        key=lambda row: (row["algo"], row["kinematics"], row["planner_key"]),
    )


@dataclass(frozen=True)
class _CampaignStatusMetrics:
    """Aggregated campaign outcome, status, and success counters."""

    campaign_finished_at_utc: str
    runtime_sec: float
    total_episodes: int
    campaign_outcome: Any
    successful_runs: int
    expected_total_runs: int
    expected_core_runs: int
    campaign_status_axes: Any
    row_status_summary: dict[str, Any]
    success_counters: Any
    campaign_evidence_status: str
    campaign_status: str
    campaign_status_reason: str
    campaign_exit_code: int
    benchmark_success: bool
    confidence_settings: dict[str, Any]
    seed_source_paths: dict[str, Any]


def _compute_campaign_status_metrics(
    cfg: CampaignConfig,
    run_entries: list[dict[str, Any]],
    planner_rows: list[dict[str, Any]],
    kinematics_matrix: tuple[str, ...],
    campaign_integrity: dict[str, Any],
    campaign_root: Path,
    orchestrator_started_at: float,
) -> _CampaignStatusMetrics:
    """Compute campaign outcome, status axes, and benchmark success.

    Returns:
        _CampaignStatusMetrics with outcome, status, and success counters.
    """
    campaign_finished_at_utc = _utc_now()
    runtime_sec = float(max(1e-9, time.perf_counter() - orchestrator_started_at))

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
    expected_total_runs = len([p for p in cfg.planners if p.enabled]) * len(kinematics_matrix)
    expected_core_runs = sum(1 for p in cfg.planners if p.enabled and p.planner_group == "core")
    campaign_status_axes = summarize_campaign_status_axes(
        {"runs": run_entries, "planner_rows": planner_rows},
        expected_total_runs=expected_total_runs,
    )
    row_status_summary = asdict(campaign_status_axes.row_status_summary)
    success_counters = _campaign_success_counters(
        run_entries, expected_core_runs=expected_core_runs * len(kinematics_matrix)
    )
    (
        campaign_evidence_status,
        campaign_status,
        campaign_status_reason,
        campaign_exit_code,
        benchmark_success,
    ) = _resolve_campaign_status_values(
        campaign_status_axes, campaign_outcome, success_counters, campaign_integrity
    )
    confidence_settings = {
        "method": "bootstrap_mean_over_seed_means",
        "confidence": float(cfg.bootstrap_confidence),
        "bootstrap_samples": int(cfg.bootstrap_samples),
        "bootstrap_seed": int(cfg.bootstrap_seed),
    }
    seed_source_paths = _campaign_seed_source_paths(run_entries, campaign_root)
    return _CampaignStatusMetrics(
        campaign_finished_at_utc=campaign_finished_at_utc,
        runtime_sec=runtime_sec,
        total_episodes=total_episodes,
        campaign_outcome=campaign_outcome,
        successful_runs=successful_runs,
        expected_total_runs=expected_total_runs,
        expected_core_runs=expected_core_runs,
        campaign_status_axes=campaign_status_axes,
        row_status_summary=row_status_summary,
        success_counters=success_counters,
        campaign_evidence_status=campaign_evidence_status,
        campaign_status=campaign_status,
        campaign_status_reason=campaign_status_reason,
        campaign_exit_code=campaign_exit_code,
        benchmark_success=benchmark_success,
        confidence_settings=confidence_settings,
        seed_source_paths=seed_source_paths,
    )


def _resolve_campaign_status_values(
    campaign_status_axes: Any,
    campaign_outcome: Any,
    success_counters: Any,
    campaign_integrity: dict[str, Any],
) -> tuple[str, str, str, int, bool]:
    """Apply integrity gating to the campaign status and benchmark-success result."""  # noqa: DOC201
    evidence_status = campaign_status_axes.evidence_status
    status, reason, exit_code = (
        campaign_outcome.status,
        campaign_outcome.status_reason,
        campaign_outcome.exit_code,
    )
    if (
        not campaign_integrity["benchmark_success_allowed"]
        and success_counters["benchmark_success"]
        and evidence_status == "valid"
    ):
        evidence_status, status, reason, exit_code = (
            "invalid",
            "integrity_failed",
            "aggregate integrity validation failed",
            1,
        )
    benchmark_success = bool(
        success_counters["benchmark_success"]
        and evidence_status == "valid"
        and campaign_integrity["benchmark_success_allowed"]
    )
    return evidence_status, status, reason, exit_code, benchmark_success


def _campaign_seed_source_paths(
    run_entries: list[dict[str, Any]], campaign_root: Path
) -> dict[str, Any]:
    """Return durable paths for successful seed-level campaign records."""
    successful_entries = [
        entry
        for entry in run_entries
        if str(entry.get("status", "")) == "ok" and str(entry.get("episodes_path", "")).strip()
    ]
    return {
        "campaign_manifest_path": _repo_relative(campaign_root / "campaign_manifest.json"),
        "run_meta_path": _repo_relative(campaign_root / "run_meta.json"),
        "episodes_paths": [
            _repo_relative(campaign_root / str(entry["episodes_path"]))
            for entry in successful_entries
        ],
    }


def _build_and_write_seed_variability(  # noqa: PLR0913
    reports_dir: Path,
    seed_variability_records: list[dict[str, Any]],
    campaign_id: str,
    campaign_finished_at_utc: str,
    config_hash: str,
    git_meta: dict[str, Any],
    resolved_seeds: list[Any],
    confidence_settings: dict[str, Any],
    seed_source_paths: dict[str, Any],
    seed_policy: Any,
) -> dict[str, Path]:
    """Build and write seed-variability and statistical-sufficiency artifacts.

    Returns:
        Dict with keys: seed_variability_json_path, seed_variability_csv_path,
        seed_episode_rows_csv_path, statistical_sufficiency_json_path,
        seed_variability_payload.
    """
    seed_variability_payload = _build_seed_variability_payload(
        seed_variability_records,
        campaign_id=campaign_id,
        generated_at_utc=campaign_finished_at_utc,
        config_hash=config_hash,
        git_hash=git_meta.get("commit", "unknown"),
        seed_policy={
            "mode": seed_policy.mode,
            "seed_set": seed_policy.seed_set,
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
    return {
        "seed_variability_json_path": seed_variability_json_path,
        "seed_variability_csv_path": seed_variability_csv_path,
        "seed_episode_rows_csv_path": seed_episode_rows_csv_path,
        "statistical_sufficiency_json_path": statistical_sufficiency_json_path,
        "seed_variability_payload": seed_variability_payload,
    }


def _build_and_write_actuation_envelope(
    reports_dir: Path,
    cfg: CampaignConfig,
    campaign_id: str,
    campaign_finished_at_utc: str,
    planner_rows: list[dict[str, Any]],
    amv_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build and write actuation-envelope artifacts when synthetic profile is configured.

    Returns:
        Dict with keys: actuation_envelope_payload, actuation_envelope_json_path,
        actuation_envelope_md_path (all None when no profile configured).
    """
    if cfg.synthetic_actuation_profile is None:
        return {
            "actuation_envelope_payload": None,
            "actuation_envelope_json_path": None,
            "actuation_envelope_md_path": None,
        }
    actuation_envelope_payload = _build_actuation_envelope_summary(
        campaign_id=campaign_id,
        generated_at_utc=campaign_finished_at_utc,
        profile=cfg.synthetic_actuation_profile,
        planner_rows=planner_rows,
        amv_summary=amv_summary,
    )
    actuation_envelope_json_path, actuation_envelope_md_path = _write_actuation_envelope_artifacts(
        reports_dir, actuation_envelope_payload
    )
    return {
        "actuation_envelope_payload": actuation_envelope_payload,
        "actuation_envelope_json_path": actuation_envelope_json_path,
        "actuation_envelope_md_path": actuation_envelope_md_path,
    }


def _resolve_snqi_baseline_and_evaluate(
    cfg: CampaignConfig,
    snqi_weights: dict[str, Any] | None,
    snqi_baseline: dict[str, Any] | None,
    run_entries: list[dict[str, Any]],
    planner_rows: list[dict[str, Any]],
    campaign_id: str,
    campaign_finished_at_utc: str,
) -> dict[str, Any]:
    """Resolve SNQI inputs, evaluate them, and return diagnostics."""  # noqa: DOC201
    episodes = collect_episodes_from_campaign_runs(run_entries, repo_root=get_repository_root())
    configured_weights = resolve_weight_mapping(snqi_weights)
    baseline_source, baseline_for_eval, baseline_adjustments, warnings = _resolve_snqi_baseline(
        snqi_baseline, episodes
    )
    _validate_snqi_normalized_inputs(cfg, episodes, baseline_for_eval)
    analysis = _evaluate_snqi_analysis(
        cfg, planner_rows, episodes, configured_weights, baseline_for_eval
    )
    diagnostics = _build_snqi_diagnostics_payload(
        cfg,
        campaign_id,
        campaign_finished_at_utc,
        baseline_source,
        baseline_adjustments,
        configured_weights,
        baseline_for_eval,
        analysis,
    )
    contract_eval = analysis["contract_eval"]
    return {
        "snqi_diagnostics_payload": diagnostics,
        "snqi_hard_fail": bool(
            cfg.paper_facing
            and cfg.snqi_contract.enabled
            and cfg.snqi_contract.enforcement in {"error", "enforce"}
            and contract_eval.status == "fail"
        ),
        "soft_contract_warning": bool(
            cfg.paper_facing
            and cfg.snqi_contract.enabled
            and soft_contract_warning_active(cfg.snqi_contract.enforcement, contract_eval.status)
        ),
        **analysis,
        "configured_weights": configured_weights,
        "baseline_for_eval": baseline_for_eval,
        "baseline_source": baseline_source,
        "baseline_adjustments": baseline_adjustments,
        "warnings": warnings,
    }


def _resolve_snqi_baseline(
    snqi_baseline: dict[str, Any] | None, episodes: list[dict[str, Any]]
) -> tuple[str, dict[str, Any], int, list[str]]:
    """Derive or sanitize the baseline used for SNQI evaluation."""  # noqa: DOC201
    if snqi_baseline is None:
        baseline, warnings = compute_baseline_stats_from_episodes(episodes)
        return "derived_from_campaign_episodes", baseline, len(warnings), list(warnings)
    baseline, warnings = sanitize_baseline_stats(snqi_baseline)
    return "config_file", baseline, len(warnings), list(warnings)


def _validate_snqi_normalized_inputs(
    cfg: CampaignConfig, episodes: list[dict[str, Any]], baseline: dict[str, Any]
) -> None:
    """Fail the paper-facing path when normalized SNQI inputs are invalid."""
    if cfg.paper_facing and cfg.snqi_contract.enabled:
        issues = validate_snqi_normalized_inputs(episodes=episodes, baseline=baseline)
        if issues:
            raise RuntimeError(
                "SNQI sensitivity preflight failed: " + "; ".join(sorted(set(issues)))
            )


def _evaluate_snqi_analysis(
    cfg: CampaignConfig,
    planner_rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    weights: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compute the SNQI contract, sensitivity, and positioning analyses."""  # noqa: DOC201
    thresholds = SnqiContractThresholds(
        rank_alignment_warn=cfg.snqi_contract.rank_alignment_warn_threshold,
        rank_alignment_fail=cfg.snqi_contract.rank_alignment_fail_threshold,
        outcome_separation_warn=cfg.snqi_contract.outcome_separation_warn_threshold,
        outcome_separation_fail=cfg.snqi_contract.outcome_separation_fail_threshold,
        max_component_dominance_warn=cfg.snqi_contract.max_component_dominance_warn_threshold,
        max_component_dominance_fail=cfg.snqi_contract.max_component_dominance_fail_threshold,
    )
    contract_eval = evaluate_snqi_contract(
        planner_rows, episodes, weights=weights, baseline=baseline, thresholds=thresholds
    )
    calibration = calibrate_weights(
        planner_rows,
        episodes,
        baseline=baseline,
        seed=cfg.snqi_contract.calibration_seed,
        trials=cfg.snqi_contract.calibration_trials,
    )
    component_dominance = compute_component_dominance(episodes, weights=weights, baseline=baseline)
    component_correlations = compute_component_correlations(
        episodes, weights=weights, baseline=baseline
    )
    planner_ordering = compute_planner_snqi_ordering(episodes, weights=weights, baseline=baseline)
    weight_sensitivity = compute_weight_sensitivity(episodes, weights=weights, baseline=baseline)
    return {
        "contract_eval": contract_eval,
        "positioning": build_positioning_recommendation(
            component_correlations, planner_ordering, weight_sensitivity
        ),
        "calibration": calibration,
        "component_dominance": component_dominance,
        "component_correlations": component_correlations,
        "planner_ordering": planner_ordering,
        "weight_sensitivity": weight_sensitivity,
    }


def _build_snqi_diagnostics_payload(
    cfg: CampaignConfig,
    campaign_id: str,
    generated_at_utc: str,
    baseline_source: str,
    baseline_adjustments: int,
    configured_weights: dict[str, Any],
    baseline: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Build the persisted SNQI diagnostics payload from evaluated inputs."""  # noqa: DOC201
    contract_eval = analysis["contract_eval"]
    weights_path = _repo_relative(cfg.snqi_weights_path) if cfg.snqi_weights_path else None
    baseline_path = _repo_relative(cfg.snqi_baseline_path) if cfg.snqi_baseline_path else None
    weights_sha256 = (
        _sha256_file(cfg.snqi_weights_path)
        if cfg.snqi_weights_path
        else _sha256_payload(configured_weights)
    )
    baseline_sha256 = (
        _sha256_file(cfg.snqi_baseline_path)
        if cfg.snqi_baseline_path
        else _sha256_payload(baseline)
    )
    return {
        "schema_version": "benchmark-snqi-diagnostics.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": generated_at_utc,
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
        "weights_version": cfg.snqi_weights_path.stem if cfg.snqi_weights_path else "default",
        "weights_sha256": weights_sha256,
        "baseline_path": baseline_path,
        "baseline_version": cfg.snqi_baseline_path.stem if cfg.snqi_baseline_path else "derived",
        "baseline_sha256": baseline_sha256,
        "baseline_source": baseline_source,
        "baseline_adjustments": baseline_adjustments,
        "baseline_for_eval": baseline,
        "configured_weights": configured_weights,
        "calibrated_weights": analysis["calibration"].get("weights"),
        "calibration": analysis["calibration"],
        "component_dominance": analysis["component_dominance"],
        "component_correlations": analysis["component_correlations"],
        "planner_ordering": analysis["planner_ordering"],
        "weight_sensitivity": analysis["weight_sensitivity"],
        "positioning": analysis["positioning"],
    }


def _build_campaign_summary_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Build the full campaign summary from completed execution state."""  # noqa: DOC201
    return {
        "fairness_contract": state["fairness_report"].to_dict(),
        "campaign": _build_campaign_summary_metadata(state),
        "planner_rows": state["planner_rows"],
        "arm_rollup": state["arm_rollup"],
        "runs": state["run_entries"],
        "campaign_integrity": state["campaign_integrity"],
        "warnings": state["warnings"],
        "soft_contract_warning": state["snqi_result"].get("soft_contract_warning", False),
        "artifacts": _build_campaign_summary_artifacts(state),
    }


def _build_campaign_summary_metadata(state: dict[str, Any]) -> dict[str, Any]:
    """Build campaign identity, execution-status, and release metadata."""  # noqa: DOC201
    cfg, metrics, snqi = state["cfg"], state["metrics"], state["snqi_result"]
    noise = normalize_observation_noise_spec(cfg.observation_noise)
    contract = snqi["contract_eval"]
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": state["campaign_id"],
        "name": cfg.name,
        "created_at_utc": state["campaign_started_at_utc"],
        "started_at_utc": state["campaign_started_at_utc"],
        "finished_at_utc": metrics.campaign_finished_at_utc,
        "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
        "scenario_matrix_hash": state["scenario_hash"],
        "git_hash": state["git_meta"].get("commit", "unknown"),
        "invoked_command": state["invoked_command"],
        "runtime_sec": metrics.runtime_sec,
        "episodes_per_second": metrics.total_episodes / metrics.runtime_sec
        if metrics.runtime_sec > 0
        else 0.0,
        "total_episodes": metrics.total_episodes,
        "successful_runs": metrics.successful_runs,
        "total_runs": len(state["run_entries"]),
        "seed_count": len(metrics.seed_source_paths.get("episodes_paths", [])),
        "non_success_runs": metrics.campaign_outcome.non_success_runs,
        "accepted_unavailable_runs": metrics.campaign_outcome.accepted_unavailable_runs,
        "unexpected_failed_runs": metrics.campaign_outcome.unexpected_failed_runs,
        "campaign_execution_status": metrics.campaign_status_axes.campaign_execution_status,
        "evidence_status": metrics.campaign_evidence_status,
        "row_status_summary": metrics.row_status_summary,
        "benchmark_success": metrics.benchmark_success,
        "status": metrics.campaign_status,
        "status_reason": metrics.campaign_status_reason,
        "exit_code": metrics.campaign_exit_code,
        "benchmark_success_basis": metrics.success_counters["benchmark_success_basis"],
        "core_successful_runs": metrics.success_counters["core_successful_runs"],
        "core_total_runs": metrics.success_counters["core_total_runs"],
        **_build_campaign_summary_configuration_metadata(state, noise),
        "snqi_weights_version": cfg.snqi_weights_path.stem if cfg.snqi_weights_path else "default",
        "snqi_weights_sha256": snqi.get("weights_sha256"),
        "snqi_baseline_version": cfg.snqi_baseline_path.stem
        if cfg.snqi_baseline_path
        else "derived",
        "snqi_baseline_sha256": snqi.get("baseline_sha256"),
        "snqi_contract_status": contract.status,
        "snqi_contract_rank_alignment_spearman": contract.rank_alignment_spearman,
        "snqi_contract_outcome_separation": contract.outcome_separation,
        "snqi_contract_dominant_component": contract.dominant_component,
        "snqi_contract_dominant_component_mean_abs": contract.dominant_component_mean_abs,
        "snqi_positioning_recommendation": snqi["positioning"].get("recommendation"),
        "snqi_positioning_claim_scope": snqi["positioning"].get("claim_scope"),
    }


def _build_campaign_summary_configuration_metadata(
    state: dict[str, Any], noise: dict[str, Any]
) -> dict[str, Any]:
    """Build configuration and release fields for a campaign summary."""  # noqa: DOC201
    cfg, manifest = state["cfg"], state["manifest_payload"]
    return {
        "paper_interpretation_profile": cfg.paper_interpretation_profile,
        "kinematics_matrix": list(state["kinematics_matrix"]),
        "holonomic_command_mode": cfg.holonomic_command_mode,
        "paper_facing": bool(cfg.paper_facing),
        "paper_profile_version": cfg.paper_profile_version,
        "observation_noise": noise,
        "observation_noise_hash": observation_noise_hash(noise),
        "amv_profile_name": cfg.amv_profile.name,
        "amv_contract_version": cfg.amv_profile.contract_version,
        "amv_coverage_enforcement": cfg.amv_profile.coverage_enforcement,
        "amv_coverage_status": str((manifest or {}).get("amv_coverage_status", "unknown")),
        "scenario_amv_overrides": {
            name: dict(values) for name, values in sorted(cfg.scenario_amv_overrides.items())
        },
        "scenario_candidates": list(cfg.scenario_candidates.names),
        "scenario_candidates_selection_name": cfg.scenario_candidates.selection_name,
        "synthetic_actuation_profile": _synthetic_actuation_metadata(
            cfg.synthetic_actuation_profile
        ),
        "latency_stress_profile": _latency_stress_metadata(cfg.latency_stress_profile, dt=cfg.dt),
        "latency_stress_metrics": not_available_latency_metrics()
        if cfg.latency_stress_profile
        else None,
        "comparability_mapping_path": manifest.get("comparability_mapping_path"),
        "comparability_mapping_version": manifest.get("comparability_mapping_version"),
        "comparability_mapping_hash": manifest.get("comparability_mapping_hash"),
        "repository_url": cfg.repository_url,
        "release_tag": state["release_tag_value"],
        "doi": cfg.doi,
        "release_url": state["release_url"],
        "release_asset_url": state["release_asset_url"],
        "doi_url": state["doi_url"],
    }


def _build_campaign_summary_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build all campaign-summary artifact references."""  # noqa: DOC201
    reports_dir, campaign_root = state["reports_dir"], state["campaign_root"]
    return {
        "campaign_manifest": _repo_relative(campaign_root / "campaign_manifest.json"),
        "campaign_summary_json": _repo_relative(reports_dir / "campaign_summary.json"),
        "campaign_credibility_scorecard_json": _repo_relative(
            reports_dir / "campaign_credibility_scorecard.json"
        ),
        **_build_campaign_table_artifacts(state),
        **_build_campaign_support_artifacts(state),
    }


def _build_campaign_table_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build table and scenario-breakdown artifact references."""  # noqa: DOC201
    tables, scenarios, parity = (
        state["table_paths"],
        state["scenario_table_paths"],
        state["parity_table_paths"],
    )
    return {
        "campaign_table_csv": _repo_relative(tables["csv_path"]),
        "campaign_table_md": _repo_relative(tables["md_table_path"]),
        "campaign_table_core_csv": _repo_relative(tables["core_csv_path"]),
        "campaign_table_core_md": _repo_relative(tables["core_md_path"]),
        "campaign_table_experimental_csv": _repo_relative(tables["experimental_csv_path"]),
        "campaign_table_experimental_md": _repo_relative(tables["experimental_md_path"]),
        "kinematics_parity_csv": _repo_relative(parity["parity_csv_path"]),
        "kinematics_parity_md": _repo_relative(parity["parity_md_path"]),
        "kinematics_skipped_combinations_csv": _repo_relative(parity["skipped_csv_path"]),
        "kinematics_skipped_combinations_md": _repo_relative(parity["skipped_md_path"]),
        "scenario_breakdown_csv": _repo_relative(scenarios["scenario_csv_path"]),
        "scenario_breakdown_md": _repo_relative(scenarios["scenario_md_path"]),
        "scenario_family_breakdown_csv": _repo_relative(scenarios["family_csv_path"]),
        "scenario_family_breakdown_md": _repo_relative(scenarios["family_md_path"]),
    }


def _build_campaign_support_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build preflight, variability, SNQI, and release artifact references."""  # noqa: DOC201
    seed, actuation, diagnostics, reports = (
        state["seed_var_paths"],
        state["actuation_paths"],
        state["snqi_diagnostics_paths"],
        state["reports_dir"],
    )
    relative = _repo_relative
    return {
        "matrix_summary_json": relative(state["matrix_summary_json_path"]),
        "matrix_summary_csv": relative(state["matrix_summary_csv_path"]),
        "amv_coverage_json": relative(state["amv_coverage_json_path"]),
        "amv_coverage_md": relative(state["amv_coverage_md_path"]),
        "comparability_json": relative(state["comparability_json_path"])
        if state["comparability_json_path"]
        else None,
        "comparability_md": relative(state["comparability_md_path"])
        if state["comparability_md_path"]
        else None,
        "seed_variability_json": relative(seed["seed_variability_json_path"]),
        "seed_variability_csv": relative(seed["seed_variability_csv_path"]),
        "seed_episode_rows_csv": relative(seed["seed_episode_rows_csv_path"]),
        "statistical_sufficiency_json": relative(seed["statistical_sufficiency_json_path"]),
        "actuation_envelope_json": relative(actuation["actuation_envelope_json_path"])
        if actuation["actuation_envelope_json_path"]
        else None,
        "actuation_envelope_md": relative(actuation["actuation_envelope_md_path"])
        if actuation["actuation_envelope_md_path"]
        else None,
        "preflight_validate_config": relative(state["validate_config_path"]),
        "preflight_preview_scenarios": relative(state["preview_scenarios_path"]),
        "campaign_report_md": relative(reports / "campaign_report.md"),
        "campaign_integrity_json": relative(reports / "campaign_integrity.json"),
        "expected_release_archive": f"{state['campaign_id']}_publication_bundle.tar.gz",
        "release_url": state["release_url"],
        "release_asset_url": state["release_asset_url"],
        "doi_url": state["doi_url"],
        "snqi_diagnostics_json": relative(diagnostics["snqi_diagnostics_json_path"]),
        "snqi_diagnostics_md": relative(diagnostics["snqi_diagnostics_md_path"]),
        "snqi_sensitivity_csv": relative(diagnostics["snqi_sensitivity_csv_path"]),
        "assurance_fragment_json": relative(reports / "assurance_fragment.json"),
        "assurance_fragment_md": relative(reports / "assurance_fragment.md"),
        "assurance_fragment_svg": relative(reports / "assurance_fragment.svg"),
    }


def _write_campaign_output_files(
    state: dict[str, Any],
) -> None:
    """Write run_meta.json, manifest.json, and campaign_manifest.json."""
    run_meta = _build_campaign_run_meta(state)
    metrics, campaign_root = state["metrics"], state["campaign_root"]
    run_manifest = {
        "git_hash": state["git_meta"].get("commit", "unknown"),
        "scenario_matrix_hash": state["scenario_hash"],
        "runtime_sec": metrics.runtime_sec,
        "episodes_per_second": (metrics.total_episodes / metrics.runtime_sec)
        if metrics.runtime_sec > 0
        else 0.0,
    }
    _write_json(campaign_root / "run_meta.json", run_meta)
    _write_json(campaign_root / "manifest.json", run_manifest)
    _write_json(
        campaign_root / "campaign_manifest.json", _build_campaign_manifest_payload(state, run_meta)
    )


def _build_campaign_run_meta(state: dict[str, Any]) -> dict[str, Any]:
    """Build durable provenance and execution metadata for a campaign."""  # noqa: DOC201
    cfg, metrics, git_meta = state["cfg"], state["metrics"], state["git_meta"]
    return {
        "repo": {key: git_meta.get(key, "unknown") for key in ("remote", "branch", "commit")},
        "execution_context": build_execution_context_provenance(),
        "matrix_path": _repo_relative(cfg.scenario_matrix_path),
        "scenario_matrix_hash": state["scenario_hash"],
        "latency_stress_profile": _latency_stress_metadata(cfg.latency_stress_profile, dt=cfg.dt),
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": list(state["resolved_seeds"]),
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
        "preflight_artifacts": _build_campaign_preflight_artifacts(state),
        "synthetic_actuation_artifacts": _build_campaign_actuation_artifacts(state),
        "snqi_artifacts": _build_campaign_snqi_artifacts(state),
        "seed_variability_artifacts": _build_campaign_seed_artifacts(state),
        "seed_variability": _build_seed_variability_metadata(state["seed_var_paths"]),
        "campaign_id": cfg.name,
        "started_at_utc": state["campaign_started_at_utc"],
        "finished_at_utc": metrics.campaign_finished_at_utc,
        "invoked_command": state["invoked_command"] or "",
        "runtime_sec": metrics.runtime_sec,
        "episodes_per_second": metrics.total_episodes / metrics.runtime_sec
        if metrics.runtime_sec > 0
        else 0.0,
    }


def _build_campaign_preflight_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build preflight artifact pointers for run metadata."""  # noqa: DOC201
    seed, actuation, relative = state["seed_var_paths"], state["actuation_paths"], _repo_relative
    return {
        "validate_config": relative(state["validate_config_path"]),
        "preview_scenarios": relative(state["preview_scenarios_path"]),
        "amv_coverage_json": relative(state["amv_coverage_json_path"]),
        "amv_coverage_md": relative(state["amv_coverage_md_path"]),
        "comparability_json": relative(state["comparability_json_path"])
        if state["comparability_json_path"]
        else None,
        "comparability_md": relative(state["comparability_md_path"])
        if state["comparability_md_path"]
        else None,
        "seed_variability_json": relative(seed["seed_variability_json_path"]),
        "seed_variability_csv": relative(seed["seed_variability_csv_path"]),
        "seed_episode_rows_csv": relative(seed["seed_episode_rows_csv_path"]),
        "statistical_sufficiency_json": relative(seed["statistical_sufficiency_json_path"]),
        "actuation_envelope_json": relative(actuation["actuation_envelope_json_path"])
        if actuation["actuation_envelope_json_path"]
        else None,
        "actuation_envelope_md": relative(actuation["actuation_envelope_md_path"])
        if actuation["actuation_envelope_md_path"]
        else None,
    }


def _build_campaign_actuation_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build synthetic-actuation artifact pointers."""  # noqa: DOC201
    paths = state["actuation_paths"]
    return {
        "json": _repo_relative(paths["actuation_envelope_json_path"])
        if paths["actuation_envelope_json_path"]
        else None,
        "md": _repo_relative(paths["actuation_envelope_md_path"])
        if paths["actuation_envelope_md_path"]
        else None,
    }


def _build_campaign_snqi_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build SNQI diagnostic artifact pointers."""  # noqa: DOC201
    paths = state["snqi_diagnostics_paths"]
    return {
        "diagnostics_json": _repo_relative(paths["snqi_diagnostics_json_path"]),
        "diagnostics_md": _repo_relative(paths["snqi_diagnostics_md_path"]),
        "sensitivity_csv": _repo_relative(paths["snqi_sensitivity_csv_path"]),
    }


def _build_campaign_seed_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build seed-variability artifact pointers."""  # noqa: DOC201
    paths = state["seed_var_paths"]
    return {
        "json": _repo_relative(paths["seed_variability_json_path"]),
        "csv": _repo_relative(paths["seed_variability_csv_path"]),
        "seed_episode_rows_csv": _repo_relative(paths["seed_episode_rows_csv_path"]),
        "statistical_sufficiency_json": _repo_relative(paths["statistical_sufficiency_json_path"]),
    }


def _build_seed_variability_metadata(paths: dict[str, Any]) -> dict[str, Any]:
    """Extract compact variability settings from the generated payload."""  # noqa: DOC201
    payload, confidence = (
        paths.get("seed_variability_payload", {}),
        paths.get("seed_variability_payload", {}).get("confidence", {}),
    )
    return {
        "metrics": list(_SEED_VARIABILITY_METRICS),
        "row_count": int(payload.get("row_count", 0)),
        "bootstrap_method": str(confidence.get("method", "")),
        "bootstrap_level": float(confidence.get("confidence", 0.0) or 0.0),
        "bootstrap_samples": int(confidence.get("bootstrap_samples", 0) or 0),
        "seed": int(confidence.get("bootstrap_seed", 0) or 0),
    }


def _build_campaign_manifest_payload(
    state: dict[str, Any], run_meta: dict[str, Any]
) -> dict[str, Any]:
    """Build the final campaign manifest with derived artifact references."""  # noqa: DOC201
    metrics, snqi, manifest = state["metrics"], state["snqi_result"], state["manifest_payload"]
    return {
        **manifest,
        "runtime_sec": metrics.runtime_sec,
        "finished_at_utc": metrics.campaign_finished_at_utc,
        "snqi_contract_status": snqi["contract_eval"].status,
        "snqi_positioning_recommendation": snqi["positioning"].get("recommendation"),
        "snqi_positioning_claim_scope": snqi["positioning"].get("claim_scope"),
        "artifacts": {
            **dict(manifest.get("artifacts") or {}),
            **_build_campaign_manifest_artifacts(state),
        },
        "seed_variability": {**dict(run_meta.get("seed_variability") or {})},
    }


def _build_campaign_manifest_artifacts(state: dict[str, Any]) -> dict[str, Any]:
    """Build campaign-manifest artifact references after reporting completes."""  # noqa: DOC201
    seed, actuation, diagnostics, reports = (
        state["seed_var_paths"],
        state["actuation_paths"],
        state["snqi_diagnostics_paths"],
        state["reports_dir"],
    )
    return {
        "seed_variability_json": _repo_relative(seed["seed_variability_json_path"]),
        "seed_variability_csv": _repo_relative(seed["seed_variability_csv_path"]),
        "seed_episode_rows_csv": _repo_relative(seed["seed_episode_rows_csv_path"]),
        "statistical_sufficiency_json": _repo_relative(seed["statistical_sufficiency_json_path"]),
        "actuation_envelope_json": _repo_relative(actuation["actuation_envelope_json_path"])
        if actuation["actuation_envelope_json_path"]
        else None,
        "actuation_envelope_md": _repo_relative(actuation["actuation_envelope_md_path"])
        if actuation["actuation_envelope_md_path"]
        else None,
        "snqi_diagnostics_json": _repo_relative(diagnostics["snqi_diagnostics_json_path"]),
        "snqi_diagnostics_md": _repo_relative(diagnostics["snqi_diagnostics_md_path"]),
        "snqi_sensitivity_csv": _repo_relative(diagnostics["snqi_sensitivity_csv_path"]),
        "assurance_fragment_json": _repo_relative(reports / "assurance_fragment.json"),
        "assurance_fragment_md": _repo_relative(reports / "assurance_fragment.md"),
        "assurance_fragment_svg": _repo_relative(reports / "assurance_fragment.svg"),
    }


def _export_publication_bundle_if_configured(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    skip_publication_bundle: bool,
    snqi_hard_fail: bool,
    benchmark_success: bool,
    campaign_id: str,
    campaign_root: Path,
    dependencies: _CampaignRuntimeDependencies,
    campaign_summary: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any] | None:
    """Export publication bundle when configured and eligible."""  # noqa: DOC201
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
    return publication_payload


def _prepare_campaign_execution_context(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    output_root: Path | None,
    label: str | None,
    campaign_id: str | None,
    invoked_command: str | None,
    dependencies: _CampaignRuntimeDependencies,
    skip_publication_bundle: bool,
    arm_isolation: str | None,
    orchestrator_started_at: float,
) -> dict[str, Any]:
    """Prepare immutable preflight state for a campaign execution."""  # noqa: DOC201
    prepared = dependencies.prepare_campaign_preflight(
        cfg,
        output_root=output_root,
        label=label,
        campaign_id=campaign_id,
        invoked_command=invoked_command,
    )
    return {
        "cfg": cfg,
        "dependencies": dependencies,
        "skip_publication_bundle": skip_publication_bundle,
        "invoked_command": invoked_command,
        "arm_isolation": arm_isolation,
        "orchestrator_started_at": orchestrator_started_at,
        "campaign_id": str(prepared["campaign_id"]),
        "campaign_root": Path(prepared["campaign_root"]),
        "reports_dir": Path(prepared["reports_dir"]),
        "validate_config_path": Path(prepared["validate_config_path"]),
        "preview_scenarios_path": Path(prepared["preview_scenarios_path"]),
        "matrix_summary_json_path": Path(prepared["matrix_summary_json_path"]),
        "matrix_summary_csv_path": Path(prepared["matrix_summary_csv_path"]),
        "amv_coverage_json_path": Path(prepared["amv_coverage_json_path"]),
        "amv_coverage_md_path": Path(prepared["amv_coverage_md_path"]),
        "comparability_json_path": (
            Path(value) if (value := prepared.get("comparability_json_path")) else None
        ),
        "comparability_md_path": (
            Path(value) if (value := prepared.get("comparability_md_path")) else None
        ),
        "manifest_payload": dict(prepared["manifest_payload"]),
        "amv_summary": dict(prepared["amv_summary"]),
        "campaign_started_at_utc": str(prepared["created_at_utc"]),
        "scenarios": list(prepared["scenarios"]),
        "resolved_seeds": list(prepared["resolved_seeds"]),
        "scenario_hash": str(prepared["scenario_hash"]),
        "git_meta": dict(prepared["git_meta"]),
        "config_hash": str(prepared["config_hash"]),
        "snqi_weights": load_optional_json(
            str(cfg.snqi_weights_path) if cfg.snqi_weights_path else None
        ),
        "snqi_baseline": load_optional_json(
            str(cfg.snqi_baseline_path) if cfg.snqi_baseline_path else None
        ),
    }


def _run_campaign_matrix_and_validate(state: dict[str, Any]) -> None:
    """Run planner arms and attach validated, fairness-annotated results to state."""
    cfg = state["cfg"]
    campaign_root = state["campaign_root"]
    runs_dir = campaign_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    resume_verdicts = _emit_resume_plan_preflight(
        cfg=cfg,
        campaign_id=state["campaign_id"],
        config_hash=state["config_hash"],
        campaign_root=campaign_root,
        runs_dir=runs_dir,
        scenarios=state["scenarios"],
    )
    state["kinematics_matrix"] = _kinematics_matrix_or_default(cfg.kinematics_matrix)
    planner_run_results = _run_campaign_planner_matrix(
        cfg=cfg,
        scenarios=state["scenarios"],
        snqi_weights=state["snqi_weights"],
        snqi_baseline=state["snqi_baseline"],
        runs_dir=runs_dir,
        dependencies=state["dependencies"],
        arm_isolation=state["arm_isolation"],
        resume_verdicts=resume_verdicts,
    )
    state["run_entries"] = planner_run_results.run_entries
    state["planner_rows"] = planner_run_results.planner_rows
    state["warnings"] = planner_run_results.warnings
    state["seed_variability_records"] = planner_run_results.seed_variability_records
    _finalize_checkpoint_provenance(state["manifest_payload"], state["run_entries"])
    state["arm_rollup"] = _build_arm_rollup(state["run_entries"])
    campaign_integrity = validate_campaign_integrity(
        state["run_entries"],
        scenarios=state["scenarios"],
        resolved_seeds=state["resolved_seeds"],
        campaign_root=campaign_root,
        campaign_manifest=state["manifest_payload"],
    )
    state["campaign_integrity"] = campaign_integrity
    for blocker in campaign_integrity["blockers"]:
        state["warnings"].append(
            "Aggregate integrity blocker: "
            f"arm='{blocker['arm']}' invariant='{blocker['invariant']}'"
        )
    state["planner_rows"].sort(
        key=lambda row: (row.get("snqi_mean", "nan") == "nan", row.get("planner_key"))
    )
    fairness_report = build_fairness_report(
        [
            {
                "algo": planner.algo,
                "observation_mode": planner.observation_mode or cfg.observation_mode,
                "tuning": asdict(planner.tuning) if planner.tuning is not None else {},
            }
            for planner in cfg.planners
            if planner.enabled
        ]
    )
    emit_fairness_annotations(fairness_report, state["planner_rows"])
    state["fairness_report"] = fairness_report


def _build_campaign_reporting_artifacts(state: dict[str, Any]) -> None:
    """Write table and variability artifacts, then calculate campaign status metrics."""
    cfg = state["cfg"]
    reports_dir = state["reports_dir"]
    state["summary_json_path"] = reports_dir / "campaign_summary.json"
    state["report_md_path"] = reports_dir / "campaign_report.md"
    state["credibility_scorecard_json_path"] = reports_dir / "campaign_credibility_scorecard.json"
    state["table_paths"] = _write_campaign_matrix_tables(reports_dir, state["planner_rows"], cfg)
    state["scenario_table_paths"] = _write_scenario_breakdown_tables(
        reports_dir, state["run_entries"], state["scenarios"]
    )
    state["parity_table_paths"] = _write_parity_and_skipped_tables(
        reports_dir, state["planner_rows"], state["run_entries"]
    )
    metrics = _compute_campaign_status_metrics(
        cfg,
        state["run_entries"],
        state["planner_rows"],
        state["kinematics_matrix"],
        state["campaign_integrity"],
        state["campaign_root"],
        state["orchestrator_started_at"],
    )
    state["metrics"] = metrics
    state["seed_var_paths"] = _build_and_write_seed_variability(
        reports_dir,
        state["seed_variability_records"],
        state["campaign_id"],
        metrics.campaign_finished_at_utc,
        state["config_hash"],
        state["git_meta"],
        state["resolved_seeds"],
        metrics.confidence_settings,
        metrics.seed_source_paths,
        cfg.seed_policy,
    )
    state["actuation_paths"] = _build_and_write_actuation_envelope(
        reports_dir,
        cfg,
        state["campaign_id"],
        metrics.campaign_finished_at_utc,
        state["planner_rows"],
        state["amv_summary"],
    )


def _evaluate_campaign_snqi(state: dict[str, Any]) -> None:
    """Evaluate SNQI, persist diagnostics, and record contract warnings in state."""
    cfg = state["cfg"]
    metrics = state["metrics"]
    campaign_id = state["campaign_id"]
    repository_url = cfg.repository_url.rstrip("/")
    release_tag_value = cfg.release_tag
    expected_archive_name = f"{campaign_id}_publication_bundle.tar.gz"
    state.update(
        {
            "release_tag_value": release_tag_value,
            "repository_url": repository_url,
            "release_url": f"{repository_url}/releases/tag/{release_tag_value}",
            "release_asset_url": (
                f"{repository_url}/releases/download/{release_tag_value}/{expected_archive_name}"
            ),
            "doi_url": f"https://doi.org/{cfg.doi}",
        }
    )
    snqi_result = _resolve_snqi_baseline_and_evaluate(
        cfg,
        state["snqi_weights"],
        state["snqi_baseline"],
        state["run_entries"],
        state["planner_rows"],
        campaign_id,
        metrics.campaign_finished_at_utc,
    )
    state["snqi_result"] = snqi_result
    state["warnings"].extend(snqi_result.get("warnings", []))
    diagnostics_paths = _write_snqi_diagnostics_artifacts(
        state["reports_dir"], snqi_result["snqi_diagnostics_payload"]
    )
    state["snqi_diagnostics_paths"] = {
        "snqi_diagnostics_json_path": diagnostics_paths[0],
        "snqi_diagnostics_md_path": diagnostics_paths[1],
        "snqi_sensitivity_csv_path": diagnostics_paths[2],
    }
    state["snqi_hard_fail"] = snqi_result["snqi_hard_fail"]
    state["soft_contract_warning"] = snqi_result["soft_contract_warning"]
    if state["snqi_hard_fail"]:
        state["warnings"].append(
            "SNQI contract status=fail with "
            f"snqi_contract.enforcement={cfg.snqi_contract.enforcement}; "
            "campaign marked with hard contract warning."
        )
    elif state["soft_contract_warning"]:
        state["warnings"].append(
            "SNQI contract status="
            f"{snqi_result['contract_eval'].status} with snqi_contract.enforcement=warn; "
            "campaign marked with soft contract warning."
        )


def _build_campaign_summary_from_state(state: dict[str, Any]) -> None:
    """Build the campaign summary from already-produced execution state."""
    state["campaign_summary"] = _build_campaign_summary_dict(state)


def _write_campaign_state_outputs(state: dict[str, Any]) -> None:
    """Persist run metadata and the fully evaluated campaign manifest."""
    _write_campaign_output_files(state)


def _write_campaign_assurance_artifacts(state: dict[str, Any]) -> None:
    """Write final reports and best-effort assurance artifacts."""
    campaign_summary = state["campaign_summary"]
    reports_dir = state["reports_dir"]
    campaign_summary["credibility_scorecard"] = build_campaign_credibility_scorecard(
        campaign_summary
    )
    _write_json(state["credibility_scorecard_json_path"], campaign_summary["credibility_scorecard"])
    _write_json(reports_dir / "campaign_integrity.json", state["campaign_integrity"])
    _write_json(state["summary_json_path"], campaign_summary)
    write_campaign_report(state["report_md_path"], campaign_summary)
    try:
        release_gate_report = None
        for gate_report_path in reports_dir.glob("*release_gate*.json"):
            try:
                with gate_report_path.open("r", encoding="utf-8") as stream:
                    release_gate_report = json.load(stream)
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
        state["warnings"].append(f"Assurance fragment export failed: {exc}")


def _build_campaign_result(state: dict[str, Any]) -> dict[str, Any]:
    """Return the public campaign result payload from completed state."""
    metrics = state["metrics"]
    seed_var_paths = state["seed_var_paths"]
    actuation_paths = state["actuation_paths"]
    diagnostics_paths = state["snqi_diagnostics_paths"]
    reports_dir = state["reports_dir"]
    return {
        "campaign_id": state["campaign_id"],
        "campaign_root": str(state["campaign_root"]),
        "summary_json": str(state["summary_json_path"]),
        "table_csv": str(state["table_paths"]["csv_path"]),
        "table_md": str(state["table_paths"]["md_table_path"]),
        "report_md": str(state["report_md_path"]),
        "snqi_diagnostics_json": str(diagnostics_paths["snqi_diagnostics_json_path"]),
        "snqi_diagnostics_md": str(diagnostics_paths["snqi_diagnostics_md_path"]),
        "snqi_sensitivity_csv": str(diagnostics_paths["snqi_sensitivity_csv_path"]),
        "assurance_fragment_json": str(reports_dir / "assurance_fragment.json"),
        "assurance_fragment_md": str(reports_dir / "assurance_fragment.md"),
        "assurance_fragment_svg": str(reports_dir / "assurance_fragment.svg"),
        "matrix_summary_json": str(state["matrix_summary_json_path"]),
        "matrix_summary_csv": str(state["matrix_summary_csv_path"]),
        "seed_variability_json": str(seed_var_paths["seed_variability_json_path"]),
        "seed_variability_csv": str(seed_var_paths["seed_variability_csv_path"]),
        "seed_episode_rows_csv": str(seed_var_paths["seed_episode_rows_csv_path"]),
        "statistical_sufficiency_json": str(seed_var_paths["statistical_sufficiency_json_path"]),
        "actuation_envelope_json": (
            str(actuation_paths["actuation_envelope_json_path"])
            if actuation_paths["actuation_envelope_json_path"] is not None
            else None
        ),
        "actuation_envelope_md": (
            str(actuation_paths["actuation_envelope_md_path"])
            if actuation_paths["actuation_envelope_md_path"] is not None
            else None
        ),
        "total_runs": len(state["run_entries"]),
        "successful_runs": metrics.successful_runs,
        "non_success_runs": metrics.campaign_outcome.non_success_runs,
        "accepted_unavailable_runs": metrics.campaign_outcome.accepted_unavailable_runs,
        "unexpected_failed_runs": metrics.campaign_outcome.unexpected_failed_runs,
        "campaign_execution_status": metrics.campaign_status_axes.campaign_execution_status,
        "evidence_status": metrics.campaign_evidence_status,
        "row_status_summary": metrics.row_status_summary,
        "benchmark_success": metrics.benchmark_success,
        "status": metrics.campaign_status,
        "status_reason": metrics.campaign_status_reason,
        "exit_code": metrics.campaign_exit_code,
        "benchmark_success_basis": metrics.success_counters["benchmark_success_basis"],
        "core_successful_runs": metrics.success_counters["core_successful_runs"],
        "core_total_runs": metrics.success_counters["core_total_runs"],
        "total_episodes": metrics.total_episodes,
        "runtime_sec": metrics.runtime_sec,
        "publication_bundle": state["publication_payload"],
        "campaign_integrity": state["campaign_integrity"],
        "warnings": state["warnings"],
        "soft_contract_warning": state["soft_contract_warning"],
    }


def _finalize_campaign_execution(state: dict[str, Any]) -> dict[str, Any]:
    """Export optional publication data, final artifacts, and the public result."""  # noqa: DOC201
    metrics = state["metrics"]
    publication_payload = _export_publication_bundle_if_configured(
        state["cfg"],
        skip_publication_bundle=state["skip_publication_bundle"],
        snqi_hard_fail=state["snqi_hard_fail"],
        benchmark_success=metrics.benchmark_success,
        campaign_id=state["campaign_id"],
        campaign_root=state["campaign_root"],
        dependencies=state["dependencies"],
        campaign_summary=state["campaign_summary"],
        warnings=state["warnings"],
    )
    state["publication_payload"] = publication_payload
    if publication_payload:
        state["campaign_summary"]["publication_bundle"] = publication_payload
    _write_campaign_assurance_artifacts(state)
    if state["snqi_hard_fail"]:
        contract_eval = state["snqi_result"]["contract_eval"]
        raise RuntimeError(
            f"SNQI contract failed with enforcement={state['cfg'].snqi_contract.enforcement}; "
            f"rank_alignment={contract_eval.rank_alignment_spearman:.4f}, "
            f"outcome_separation={contract_eval.outcome_separation:.4f}. "
            "See diagnostics: "
            f"{_repo_relative(state['snqi_diagnostics_paths']['snqi_diagnostics_json_path'])}"
        )
    logger.info(
        "Camera-ready campaign finished id={} runs={} episodes={} out={}",
        state["campaign_id"],
        len(state["run_entries"]),
        metrics.total_episodes,
        state["campaign_root"],
    )
    return _build_campaign_result(state)


def _run_campaign_orchestrator(
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
    """Execute the campaign as a thin coordinator over focused phases."""  # noqa: DOC201
    state = _prepare_campaign_execution_context(
        cfg,
        output_root=output_root,
        label=label,
        campaign_id=campaign_id,
        invoked_command=invoked_command,
        dependencies=dependencies,
        skip_publication_bundle=skip_publication_bundle,
        arm_isolation=arm_isolation,
        orchestrator_started_at=time.perf_counter(),
    )
    _run_campaign_matrix_and_validate(state)
    _build_campaign_reporting_artifacts(state)
    _evaluate_campaign_snqi(state)
    _build_campaign_summary_from_state(state)
    _write_campaign_state_outputs(state)
    return _finalize_campaign_execution(state)
