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

import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.benchmark.aggregate import read_jsonl
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
    validate_snqi_normalized_inputs,
)
from robot_sf.benchmark.utils import load_optional_json
from robot_sf.common.artifact_paths import get_artifact_category_path, get_repository_root

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig

CAMPAIGN_SCHEMA_VERSION = "benchmark-camera-ready-campaign.v1"
DEFAULT_EPISODE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")


def run_campaign(  # noqa: C901, PLR0912, PLR0913, PLR0915
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
) -> dict[str, Any]:
    """Execute a camera-ready planner campaign and emit campaign artifacts.

    The ``prepare_campaign_preflight``, ``run_batch``, ``compute_aggregates_with_ci`` and
    ``export_publication_bundle`` collaborators are injected so the legacy
    ``camera_ready_campaign`` facade can pass its own monkeypatchable bindings; when omitted
    the canonical implementations are imported lazily.

    Returns:
        Campaign execution summary with output paths and high-level counters.

    Raises:
        OrcaRvo2PreflightError: When enabled ORCA-dependent planners require ``rvo2`` but it is
            not importable.
        RouteClearanceError: When any scenario route centerline lies closer to a static obstacle
            than the robot radius, making the route geometrically impossible to follow without
            collision.
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

    start = time.perf_counter()
    prepared = prepare_campaign_preflight(
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

    run_entries: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seed_variability_records: list[dict[str, Any]] = []
    kinematics_matrix = _kinematics_matrix_or_default(cfg.kinematics_matrix)
    stop_requested = False

    for planner in cfg.planners:
        if not planner.enabled:
            continue
        active_observation_mode = planner.observation_mode or cfg.observation_mode
        for kinematics in kinematics_matrix:
            planner_run_key = f"{_sanitize_name(planner.key)}__{_sanitize_name(kinematics)}"
            planner_dir = runs_dir / planner_run_key
            planner_dir.mkdir(parents=True, exist_ok=True)
            episodes_path = planner_dir / "episodes.jsonl"

            effective_workers = (
                planner.workers_override if planner.workers_override is not None else cfg.workers
            )
            effective_horizon = (
                planner.horizon_override if planner.horizon_override is not None else cfg.horizon
            )
            effective_dt = planner.dt_override if planner.dt_override is not None else cfg.dt

            logger.info(
                "Running campaign planner key={} algo={} kinematics={} profile={} workers={}",
                planner.key,
                planner.algo,
                kinematics,
                planner.benchmark_profile,
                effective_workers,
            )

            planner_started_at_utc = _utc_now()
            planner_start = time.perf_counter()
            status = "ok"
            summary: dict[str, Any]
            aggregates: dict[str, Any] | None = None
            scoped_scenarios = [
                _scenario_with_kinematics(
                    sc,
                    kinematics=kinematics,
                    holonomic_command_mode=cfg.holonomic_command_mode,
                )
                for sc in scenarios
            ]

            try:
                summary = run_batch(
                    scoped_scenarios,
                    out_path=episodes_path,
                    schema_path=DEFAULT_EPISODE_SCHEMA_PATH,
                    horizon=effective_horizon if effective_horizon is not None else 0,
                    dt=effective_dt if effective_dt is not None else 0.0,
                    record_forces=cfg.record_forces,
                    record_planner_decision_trace=cfg.record_planner_decision_trace,
                    record_simulation_step_trace=cfg.record_simulation_step_trace,
                    snqi_weights=snqi_weights,
                    snqi_baseline=snqi_baseline,
                    algo=planner.algo,
                    algo_config_path=(
                        str(planner.algo_config_path)
                        if planner.algo_config_path is not None
                        else None
                    ),
                    benchmark_profile=planner.benchmark_profile,
                    socnav_missing_prereq_policy=planner.socnav_missing_prereq_policy,
                    adapter_impact_eval=planner.adapter_impact_eval,
                    observation_mode=active_observation_mode,
                    observation_noise=cfg.observation_noise,
                    synthetic_actuation_profile=_synthetic_actuation_metadata(
                        cfg.synthetic_actuation_profile
                    ),
                    latency_stress_profile=_latency_stress_metadata(
                        cfg.latency_stress_profile,
                        dt=effective_dt,
                    ),
                    workers=effective_workers,
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
                warnings.append(
                    f"Planner '{planner.key}' failed for kinematics '{kinematics}': {exc}"
                )

            planner_finished_at_utc = _utc_now()
            runtime_sec = float(max(1e-9, time.perf_counter() - planner_start))
            episodes_written = int(summary.get("written", 0))
            summary["status"] = status
            summary["started_at_utc"] = planner_started_at_utc
            summary["finished_at_utc"] = planner_finished_at_utc
            summary["runtime_sec"] = runtime_sec
            summary["episodes_per_second"] = (
                (episodes_written / runtime_sec) if runtime_sec > 0 else 0.0
            )
            summary["kinematics"] = kinematics
            summary["benchmark_availability"] = availability_payload(summary)
            _write_json(planner_dir / "summary.json", summary)

            records: list[dict[str, Any]] = []
            if episodes_path.exists() and episodes_path.stat().st_size > 0:
                records = read_jsonl(str(episodes_path))
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
                    aggregates = compute_aggregates_with_ci(
                        records,
                        group_by="scenario_params.algo",
                        bootstrap_samples=cfg.bootstrap_samples,
                        bootstrap_confidence=cfg.bootstrap_confidence,
                        bootstrap_seed=cfg.bootstrap_seed,
                    )
                except Exception as exc:
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
                        "workers": effective_workers,
                        "horizon": effective_horizon,
                        "dt": effective_dt,
                    },
                    "status": status,
                    "started_at_utc": planner_started_at_utc,
                    "finished_at_utc": planner_finished_at_utc,
                    "runtime_sec": runtime_sec,
                    "episodes_path": _repo_relative(episodes_path),
                    "summary_path": _repo_relative(planner_dir / "summary.json"),
                    "summary": summary,
                    "aggregates": aggregates,
                },
            )

            if classify_planner_row_status(status) == "unexpected_failure" and cfg.stop_on_failure:
                logger.warning(
                    "Campaign stop_on_failure triggered: planner key={} kinematics={} status={} (halting remaining planners).",
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
                break
        if stop_requested:
            break

    planner_rows.sort(
        key=lambda row: (row.get("snqi_mean", "nan") == "nan", row.get("planner_key"))
    )

    summary_json_path = reports_dir / "campaign_summary.json"
    report_md_path = reports_dir / "campaign_report.md"

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
    if snqi_hard_fail:
        warnings.append(
            "SNQI contract status=fail with "
            f"snqi_contract.enforcement={cfg.snqi_contract.enforcement}; "
            "campaign marked with hard contract warning."
        )
    elif (
        cfg.paper_facing
        and cfg.snqi_contract.enabled
        and cfg.snqi_contract.enforcement == "warn"
        and contract_eval.status in {"warn", "fail"}
    ):
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
        "artifacts": {
            "campaign_manifest": _repo_relative(campaign_root / "campaign_manifest.json"),
            "campaign_summary_json": _repo_relative(summary_json_path),
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
            bundle = export_publication_bundle(
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
        except Exception as exc:
            warnings.append(f"Publication bundle export failed: {exc}")
    elif (
        cfg.export_publication_bundle
        and not skip_publication_bundle
        and not snqi_hard_fail
        and not benchmark_success
    ):
        warnings.append("Publication bundle export skipped because benchmark_success=false.")

    _write_json(summary_json_path, campaign_summary)
    write_campaign_report(report_md_path, campaign_summary)

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
    }
