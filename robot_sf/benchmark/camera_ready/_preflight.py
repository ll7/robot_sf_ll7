"""Preflight payload helpers for camera-ready benchmark campaigns."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from robot_sf.benchmark.camera_ready._artifacts import (
    _write_amv_coverage_artifacts,
    _write_comparability_artifacts,
    _write_json,
    _write_matrix_summary_artifacts,
)
from robot_sf.benchmark.camera_ready._config import (
    _assert_radius_sweep_preflight_ready,
    _load_campaign_scenarios,
    _radius_binding_metadata,
    _resolved_seed_inventory,
    _scenario_horizon_summary,
)
from robot_sf.benchmark.camera_ready._config_types import (
    TUNING_SOURCE_DECLARED,
)
from robot_sf.benchmark.camera_ready._resume_plan import verify_resume_context
from robot_sf.benchmark.camera_ready._route_clearance import (
    _assert_route_clearance_feasible,
    _build_route_clearance_warnings,
    _load_route_clearance_certifications,
    _route_clearance_warning_summary,
)
from robot_sf.benchmark.camera_ready._run_state import _git_context, _resolve_campaign_id
from robot_sf.benchmark.camera_ready._summaries import (
    _build_amv_coverage_summary,
    _build_comparability_summary,
    _build_matrix_summary_rows,
)
from robot_sf.benchmark.camera_ready._util import (
    _hash_payload,
    _jsonable_repo_relative,
    _latency_stress_metadata,
    _repo_relative,
    _synthetic_actuation_metadata,
    _utc_now,
)
from robot_sf.benchmark.campaign_checkpoint_preflight import (
    check_campaign_arm_checkpoints_preflight,
)
from robot_sf.benchmark.campaign_runtime_preflight import (
    check_campaign_arm_policy_dependencies_preflight,
    check_campaign_scenario_maps_preflight,
)
from robot_sf.benchmark.latency_stress import not_available_latency_metrics
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.orca_preflight import check_orca_rvo2_preflight
from robot_sf.benchmark.tuning_run_provenance import (
    aggregate_tuning_records,
    build_launch_records,
)
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path

CAMPAIGN_SCHEMA_VERSION = "benchmark-camera-ready-campaign.v1"

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec

CheckpointPreflightMode = Literal["metadata_only", "enforced_staged"]
_CHECKPOINT_PREFLIGHT_REPORT_NAME: dict[str, str] = {
    "metadata_only": "checkpoint_resolvability.json",
    "enforced_staged": "checkpoint_staging.json",
}

# Honest label for an arm whose tuning effort has not been recorded or reconstructed yet. This is
# distinct from a declared ``backfilled`` source: it makes the cross-arm asymmetry visible in the
# manifest without inventing tuning parameters the author never recorded (issue #5143).
_TUNING_BACKFILL_PENDING = "backfill_pending"
_SCENARIO_FILE_REFERENCE_FIELDS = ("map_file", "route_overrides_file")


def _campaign_config_provenance(cfg: CampaignConfig) -> dict[str, str]:
    """Return portable immutable-config provenance for preflight artifacts.

    Campaign configs loaded from the repository retain their source path.  A
    preflight packet must name that repository-relative path and its full file
    checksum so evidence-registry tooling can verify the packet against the
    producing commit.  Programmatically constructed or external configs have
    no portable repository provenance, so this helper intentionally omits the
    fields rather than serializing an absolute local path.
    """
    if cfg.source_config_path is None or not cfg.source_config_sha256:
        return {}
    config_path = Path(cfg.source_config_path).resolve()
    portable_path = _repo_relative(config_path)
    if Path(portable_path).is_absolute():
        return {}
    return {
        "config_path": portable_path,
        "config_sha256": cfg.source_config_sha256,
    }


def _verify_existing_resume_context(
    cfg: CampaignConfig,
    *,
    campaign_root: Path,
    campaign_id: str,
    config_hash: str,
) -> None:
    """Verify an existing resumed campaign before refreshing its manifest."""
    if not cfg.resume:
        return
    runs_dir = campaign_root / "runs"
    has_prior_arms = any(r.is_dir() for r in runs_dir.iterdir()) if runs_dir.exists() else False
    if has_prior_arms:
        # Verify the on-disk context before preflight refresh overwrites the manifest. Otherwise a
        # fixed-ID resume could normalize an incompatible campaign into the current config and
        # bypass the resume-plan fail-closed contract (issue #5538).
        verify_resume_context(
            campaign_root,
            campaign_id=campaign_id,
            config_hash=config_hash,
        )


def _checkpoint_provenance_block(
    planner: PlannerSpec,
    checkpoint_preflight_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the preflight checkpoint-provenance block for one campaign arm.

    Returns:
        JSON-serializable checkpoint identity and not-yet-run status.
    """
    references = [
        dict(item)
        for item in checkpoint_preflight_summary.get("arms", [])
        if isinstance(item, dict) and item.get("planner_key") == planner.key
    ]
    if not references:
        return {
            "status": "not_applicable",
            "model_id": None,
            "checkpoint_sha256": None,
            "load_succeeded": None,
            "fallback_triggered": None,
            "references": [],
            "runtime": [],
        }
    model_ids = sorted(
        {str(item["model_id"]) for item in references if item.get("model_id") is not None}
    )
    hashes = sorted(
        {
            str(item["checkpoint_sha256"])
            for item in references
            if item.get("checkpoint_sha256") is not None
        }
    )
    resolved_statuses = {"present_local", "staged", "stageable_remote"}
    unresolved = [item for item in references if item.get("status") not in resolved_statuses]
    return {
        "status": "resolution_failed" if unresolved else "not_run",
        "model_id": model_ids[0] if len(model_ids) == 1 else None,
        "checkpoint_sha256": hashes[0] if len(hashes) == 1 else None,
        "load_succeeded": None,
        "fallback_triggered": None,
        "references": references,
        "runtime": [],
    }


def _tuning_effort_block(planner: PlannerSpec) -> dict[str, Any]:
    """Return the per-arm tuning-effort manifest block (issue #5143).

    When an arm declares no ``tuning`` block, synthesize a best-effort ``backfill_pending`` entry
    so the under-tuning asymmetry is always visible in campaign artifacts rather than silent. A
    declared block is emitted verbatim.

    Returns:
        JSON-serializable tuning-effort manifest block for one arm.
    """
    tuning = planner.tuning
    if tuning is None:
        return {
            "parameters_touched": [],
            "tuning_scenario_ids": [],
            "eval_set_disjoint": None,
            "budget_runs": None,
            "budget_hours": None,
            "tuned_by": None,
            "tuned_at_utc": None,
            "source": _TUNING_BACKFILL_PENDING,
            "note": (
                "No tuning block declared for this arm; tuning effort is unrecorded. "
                "Synthesized as backfill_pending so the cross-arm asymmetry is visible."
            ),
        }
    block: dict[str, Any] = {
        "parameters_touched": list(tuning.parameters_touched),
        "tuning_scenario_ids": list(tuning.tuning_scenario_ids),
        "eval_set_disjoint": tuning.eval_set_disjoint,
        "budget_runs": tuning.budget_runs,
        "budget_hours": tuning.budget_hours,
        "tuned_by": tuning.tuned_by,
        "tuned_at_utc": tuning.tuned_at_utc,
        "source": tuning.source,
    }
    if tuning.source != TUNING_SOURCE_DECLARED:
        block["note"] = (
            "Tuning block is recorded as best-effort reconstruction or unknown; not author-declared."
        )
    return block


def _tuning_effort_summary(planners: tuple[PlannerSpec, ...]) -> dict[str, Any]:
    """Summarize per-arm tuning-effort coverage for the manifest (issue #5143).

    Returns:
        JSON-serializable summary of declared vs backfill-pending arm counts.
    """
    enabled = [planner for planner in planners if planner.enabled]
    declared = [planner for planner in enabled if planner.tuning is not None]
    backfill_pending = [planner for planner in enabled if planner.tuning is None]
    by_source: dict[str, int] = {}
    for planner in declared:
        source = planner.tuning.source if planner.tuning is not None else _TUNING_BACKFILL_PENDING
        by_source[source] = by_source.get(source, 0) + 1
    for planner in backfill_pending:
        by_source[_TUNING_BACKFILL_PENDING] = by_source.get(_TUNING_BACKFILL_PENDING, 0) + 1
    return {
        "enabled_arm_count": len(enabled),
        "declared_count": len(declared),
        "backfill_pending_count": len(backfill_pending),
        "arms_missing_tuning": sorted(planner.key for planner in backfill_pending),
        "by_source": by_source,
    }


def _build_tuning_provenance_ledger(
    cfg: CampaignConfig,
    *,
    campaign_id: str,
    created_at_utc: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Emit and aggregate one prospective launch record per enabled planner arm.

    The generated ledger is a launch/ingestion receipt, not benchmark evidence.  A
    missing machine counter remains ``null`` in the record and in the aggregate.

    Returns:
        JSON-compatible tuning ledger with records, policy, and capture metadata.
    """
    planner_parameters = {
        planner.key: (planner.tuning.parameters_touched if planner.tuning is not None else None)
        for planner in cfg.planners
        if planner.enabled
    }
    git_meta = metadata.get("git_meta") if isinstance(metadata.get("git_meta"), dict) else {}
    records = build_launch_records(
        cfg.tuning_run_provenance,
        campaign_id=campaign_id,
        source_commit=(str(git_meta.get("commit")) if git_meta.get("commit") else None),
        config_hash=metadata.get("config_hash"),
        planner_parameters=planner_parameters,
        recorded_at_utc=created_at_utc,
        provenance={
            "campaign_name": cfg.name,
            "tuning_effort_enforcement": cfg.tuning_effort_enforcement,
        },
        strict=cfg.tuning_effort_enforcement == "error",
    )
    ledger = aggregate_tuning_records(records)
    ledger["capture"] = {
        "mode": "camera_ready_preflight",
        "campaign_id": campaign_id,
        "source_commit": git_meta.get("commit"),
        "config_hash": metadata.get("config_hash"),
        "recorded_at_utc": created_at_utc,
        "strict_validation": cfg.tuning_effort_enforcement == "error",
    }
    return ledger


def _scenario_display_name(scenario: dict[str, Any]) -> str:
    """Return the stable scenario identifier used in preflight payloads."""
    return str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "")


def _portable_scenario_file_reference(value: str | Path) -> str:
    """Return a repository-portable representation of a scenario file reference."""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    return _repo_relative(path)


def _portable_preview_route_override(value: str | Path) -> str:
    """Return portable route-override provenance without leaking a worktree path."""
    normalized = _portable_scenario_file_reference(value)
    return Path(normalized).name if Path(normalized).is_absolute() else normalized


def _preview_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Return one preview scenario with repository-portable route provenance."""
    preview = _jsonable_repo_relative(scenario)
    route_override = scenario.get("route_overrides_file")
    if isinstance(route_override, (str, Path)):
        preview["route_overrides_file"] = _portable_preview_route_override(route_override)
    return preview


def _scenario_hash_payload(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize resolved scenario file references before deriving matrix identity.

    Returns:
        JSON-serializable scenarios with repository-portable file references.
    """
    payload: list[dict[str, Any]] = []
    for scenario in scenarios:
        normalized = dict(_jsonable_repo_relative(scenario))
        for field_name in _SCENARIO_FILE_REFERENCE_FIELDS:
            value = scenario.get(field_name)
            if isinstance(value, (str, Path)):
                normalized[field_name] = _portable_scenario_file_reference(value)
        payload.append(normalized)
    return payload


def _scenario_matrix_hash(scenarios: list[dict[str, Any]]) -> str:
    """Return the stable hash for resolved scenarios independent of their worktree path."""
    return _hash_payload(_scenario_hash_payload(scenarios))


def _build_preflight_validate_payload(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    campaign_id: str,
    created_at_utc: str,
    scenarios: list[dict[str, Any]],
    resolved_seeds: list[int],
    scenario_horizons_summary: dict[str, Any] | None,
    route_clearance_warnings: list[dict[str, Any]],
    route_clearance_warning_summary: dict[str, Any],
    noise_spec: dict[str, Any],
    noise_hash: str,
    checkpoint_preflight_summary: dict[str, Any],
    checkpoint_preflight_mode: CheckpointPreflightMode,
) -> dict[str, Any]:
    """Build the ``validate_config.json`` preflight artifact payload.

    Returns:
        JSON-serializable preflight validation artifact payload.
    """
    payload: dict[str, Any] = {
        "schema_version": "benchmark-preflight-validate-config.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": created_at_utc,
        **_campaign_config_provenance(cfg),
        "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
        "radius_binding": _radius_binding_metadata(cfg.radius_sweep),
        "scenario_count": len(scenarios),
        "scenario_candidates": {
            "requested": list(cfg.scenario_candidates.names),
            "resolved": [_scenario_display_name(scenario) for scenario in scenarios],
        },
        "scenario_amv_overrides": {
            scenario_name: dict(values)
            for scenario_name, values in sorted(cfg.scenario_amv_overrides.items())
        },
        "planner_count": len([planner for planner in cfg.planners if planner.enabled]),
        "workers": cfg.workers,
        "horizon": cfg.horizon,
        "dt": cfg.dt,
        "resume": cfg.resume,
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": resolved_seeds,
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
        "checkpoint_preflight": {
            "mode": checkpoint_preflight_mode,
            "stage": bool(checkpoint_preflight_summary.get("stage")),
            "checked": int(checkpoint_preflight_summary.get("checked", 0)),
            "resolved": int(checkpoint_preflight_summary.get("resolved", 0)),
            "submit_safe": bool(checkpoint_preflight_summary.get("submit_safe")),
            "arms": list(checkpoint_preflight_summary.get("arms", [])),
        },
        "amv_profile": {
            "name": cfg.amv_profile.name,
            "contract_version": cfg.amv_profile.contract_version,
            "coverage_enforcement": cfg.amv_profile.coverage_enforcement,
            "required_dimensions": {
                key: list(values) for key, values in cfg.amv_profile.required_dimensions.items()
            },
        },
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
        "comparability_mapping": (
            _repo_relative(cfg.comparability_mapping_path)
            if cfg.comparability_mapping_path is not None
            else None
        ),
        "retained_metric_contract_path": (
            _repo_relative(cfg.retained_metric_contract_path)
            if cfg.retained_metric_contract_path is not None
            else None
        ),
        "snqi_contract": {
            "enabled": bool(cfg.snqi_contract.enabled),
            "enforcement": cfg.snqi_contract.enforcement,
            "rank_alignment_warn_threshold": cfg.snqi_contract.rank_alignment_warn_threshold,
            "rank_alignment_fail_threshold": cfg.snqi_contract.rank_alignment_fail_threshold,
            "outcome_separation_warn_threshold": cfg.snqi_contract.outcome_separation_warn_threshold,
            "outcome_separation_fail_threshold": cfg.snqi_contract.outcome_separation_fail_threshold,
            "max_component_dominance_warn_threshold": (
                cfg.snqi_contract.max_component_dominance_warn_threshold
            ),
            "max_component_dominance_fail_threshold": (
                cfg.snqi_contract.max_component_dominance_fail_threshold
            ),
            "calibration_seed": cfg.snqi_contract.calibration_seed,
            "calibration_trials": cfg.snqi_contract.calibration_trials,
        },
        "snqi_weights_path": (
            _repo_relative(cfg.snqi_weights_path) if cfg.snqi_weights_path is not None else None
        ),
        "snqi_baseline_path": (
            _repo_relative(cfg.snqi_baseline_path) if cfg.snqi_baseline_path is not None else None
        ),
        "route_clearance_warnings": route_clearance_warnings,
        "route_clearance_warning_count": len(route_clearance_warnings),
        "route_clearance_warning_summary": route_clearance_warning_summary,
        "route_clearance_certifications_path": (
            _repo_relative(cfg.route_clearance_certifications_path)
            if cfg.route_clearance_certifications_path is not None
            else None
        ),
        "observation_noise": noise_spec,
        "observation_noise_hash": noise_hash,
    }
    if scenario_horizons_summary is not None:
        payload["scenario_horizons"] = scenario_horizons_summary
    return payload


def _build_preflight_preview_payload(
    cfg: CampaignConfig,
    *,
    campaign_id: str,
    created_at_utc: str,
    scenarios: list[dict[str, Any]],
    route_clearance_warnings: list[dict[str, Any]],
    route_clearance_warning_summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``preview_scenarios.json`` preflight artifact payload.

    Returns:
        JSON-serializable scenario preview artifact payload.
    """
    preview_limit = max(0, int(cfg.preview_scenario_limit))
    payload: dict[str, Any] = {
        "schema_version": "benchmark-preflight-preview-scenarios.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": created_at_utc,
        "radius_binding": _radius_binding_metadata(cfg.radius_sweep),
        **_campaign_config_provenance(cfg),
        "scenario_count": len(scenarios),
        "preview_limit": preview_limit,
        "scenario_candidates": list(cfg.scenario_candidates.names),
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
        "route_clearance_warnings": route_clearance_warnings,
        "route_clearance_warning_count": len(route_clearance_warnings),
        "route_clearance_warning_summary": route_clearance_warning_summary,
        "route_clearance_certifications_path": (
            _repo_relative(cfg.route_clearance_certifications_path)
            if cfg.route_clearance_certifications_path is not None
            else None
        ),
    }
    if len(scenarios) > preview_limit:
        payload["truncated"] = True
        payload["total_scenarios"] = len(scenarios)
        payload["scenarios"] = [
            {
                "name": _scenario_display_name(scenario),
                "map_file": scenario.get("map_file"),
                "seeds": scenario.get("seeds"),
                "metadata": scenario.get("metadata"),
            }
            for scenario in scenarios[:preview_limit]
        ]
    else:
        payload["truncated"] = False
        payload["scenarios"] = [_preview_scenario(scenario) for scenario in scenarios]
    return payload


# ---------------------------------------------------------------------------
# Focused sub-functions extracted from prepare_campaign_preflight (issue #6537)
# ---------------------------------------------------------------------------


def _run_preflight_checks(
    cfg: CampaignConfig,
    *,
    checkpoint_preflight_mode: CheckpointPreflightMode,
    checkpoint_cache_dir: Path | None,
    checkpoint_registry_path: str | Path | None,
) -> dict[str, Any]:
    """Run ORCA, policy-dependency, and arm-checkpoint preflight checks.

    Returns:
        Checkpoint preflight report from the arm-checkpoint preflight gate.
    """
    check_orca_rvo2_preflight(cfg)
    check_campaign_arm_policy_dependencies_preflight(cfg)
    # Fail fast when an enabled arm names a checkpoint that cannot be resolved (unknown/mistyped
    # model_id, local_only-missing, or a missing model_path file) before any scenario loads. There
    # are two modes (issue #4613/#4663):
    #   * metadata_only (default, cheap, network-free): accept present_local OR stageable_remote.
    #     Safe to leave always-on. NOT submit-safe when any arm is stageable_remote.
    #   * enforced_staged: actually download+checksum-verify each registry artifact so the compute
    #     node loads a validated file. The submit wrapper must run this before sbatch.
    checkpoint_preflight_stage = checkpoint_preflight_mode == "enforced_staged"
    return check_campaign_arm_checkpoints_preflight(
        cfg,
        stage=checkpoint_preflight_stage,
        registry_path=checkpoint_registry_path,
        cache_dir=checkpoint_cache_dir,
        fail_closed_implicit=cfg.checkpoint_provenance_enforcement == "error",
    )


def _setup_campaign_directories(
    cfg: CampaignConfig,
    *,
    output_root: Path | None,
    label: str | None,
    campaign_id: str | None,
) -> tuple[str, Path, Path, Path]:
    """Resolve campaign identity and create output directories.

    Returns:
        Tuple of ``(campaign_id, campaign_root, reports_dir, preflight_dir)``.
    """
    ensure_canonical_tree(categories=("benchmarks",))
    campaign_id = _resolve_campaign_id(cfg, label=label, campaign_id=campaign_id)
    base_dir = (
        output_root.resolve()
        if output_root
        else (get_artifact_category_path("benchmarks") / "camera_ready")
    )
    campaign_root = (base_dir / campaign_id).resolve()
    reports_dir = campaign_root / "reports"
    preflight_dir = campaign_root / "preflight"
    reports_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir.mkdir(parents=True, exist_ok=True)
    return campaign_id, campaign_root, reports_dir, preflight_dir


def _validate_and_setup_campaign(
    cfg: CampaignConfig,
    *,
    checkpoint_preflight_mode: CheckpointPreflightMode,
    checkpoint_cache_dir: Path | None,
    checkpoint_registry_path: str | Path | None,
    output_root: Path | None,
    label: str | None,
    campaign_id: str | None,
) -> tuple[dict[str, Any], str, Path, Path, Path]:
    """Run preflight checks and set up campaign directories.

    Returns:
        Tuple of ``(checkpoint_report, campaign_id, campaign_root, reports_dir, preflight_dir)``.
    """
    ckpt_report = _run_preflight_checks(
        cfg,
        checkpoint_preflight_mode=checkpoint_preflight_mode,
        checkpoint_cache_dir=checkpoint_cache_dir,
        checkpoint_registry_path=checkpoint_registry_path,
    )
    campaign_id, campaign_root, reports_dir, preflight_dir = _setup_campaign_directories(
        cfg,
        output_root=output_root,
        label=label,
        campaign_id=campaign_id,
    )
    return ckpt_report, campaign_id, campaign_root, reports_dir, preflight_dir


def _load_scenarios_and_route_clearance(
    cfg: CampaignConfig,
    build_route_clearance_warnings: Callable[..., list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load scenarios, validate maps, and build route-clearance warnings.

    Returns:
        Tuple of ``(scenarios, route_clearance_warnings, route_clearance_warning_summary)``.

    Raises:
        CampaignScenarioMapPreflightError: When a scenario ``map_file`` cannot resolve.
        RouteClearanceError: When a route centerline is geometrically infeasible.
    """
    scenarios = _load_campaign_scenarios(cfg)
    check_campaign_scenario_maps_preflight(scenarios)
    route_clearance_certifications = _load_route_clearance_certifications(
        cfg.route_clearance_certifications_path
    )
    route_clearance_warnings = build_route_clearance_warnings(
        scenarios,
        certifications=route_clearance_certifications,
    )
    # Fail closed before producing any preflight artifact: a route whose centerline is closer to a
    # static obstacle than the robot radius is geometrically impossible to follow without
    # collision, so the benchmark must refuse to run it rather than emit a silent warning
    # (issue #3628).
    _assert_route_clearance_feasible(route_clearance_warnings)
    route_clearance_warning_summary = _route_clearance_warning_summary(route_clearance_warnings)
    return scenarios, route_clearance_warnings, route_clearance_warning_summary


def _compute_campaign_metadata(
    cfg: CampaignConfig,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute campaign metadata hashes and summaries.

    Returns:
        Dict with ``resolved_seeds``, ``scenario_hash``, ``scenario_horizons_summary``,
        ``git_meta``, ``config_hash``, ``noise_spec``, and ``noise_hash``.
    """
    resolved_seeds = _resolved_seed_inventory(scenarios)
    scenario_hash = _scenario_matrix_hash(scenarios)
    scenario_horizons_summary = _scenario_horizon_summary(
        scenarios,
        schedule_path=cfg.scenario_horizons_path,
    )
    git_meta = _git_context()
    config_payload = asdict(cfg)
    if cfg.tuning_run_provenance is None:
        # Preserve hashes for legacy configs that predate the optional prospective block.
        config_payload.pop("tuning_run_provenance", None)
    config_hash = _config_hash(_jsonable_repo_relative(config_payload))
    noise_spec = normalize_observation_noise_spec(cfg.observation_noise)
    noise_hash = observation_noise_hash(noise_spec)
    return {
        "resolved_seeds": resolved_seeds,
        "scenario_hash": scenario_hash,
        "scenario_horizons_summary": scenario_horizons_summary,
        "git_meta": git_meta,
        "config_hash": config_hash,
        "noise_spec": noise_spec,
        "noise_hash": noise_hash,
    }


def _load_and_prepare_scenarios(
    cfg: CampaignConfig,
    build_route_clearance_warnings: Callable[..., list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load scenarios, validate route clearance, and compute campaign metadata.

    Returns:
        Tuple of ``(scenarios, route_clearance_warnings, route_clearance_warning_summary,
        metadata)``.
    """
    scenarios, rc_warnings, rc_summary = _load_scenarios_and_route_clearance(
        cfg,
        build_route_clearance_warnings,
    )
    metadata = _compute_campaign_metadata(cfg, scenarios)
    return scenarios, rc_warnings, rc_summary, metadata


def _write_preflight_artifacts(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    preflight_dir: Path,
    campaign_id: str,
    created_at_utc: str,
    scenarios: list[dict[str, Any]],
    metadata: dict[str, Any],
    route_clearance_warnings: list[dict[str, Any]],
    route_clearance_warning_summary: dict[str, Any],
    checkpoint_preflight_report: dict[str, Any],
    checkpoint_preflight_mode: CheckpointPreflightMode,
) -> tuple[Path, Path, Path]:
    """Build and write preflight validation, preview, and checkpoint artifacts.

    Returns:
        Tuple of ``(validate_config_path, preview_scenarios_path,
        checkpoint_preflight_report_path)``.
    """
    validate_config_path = preflight_dir / "validate_config.json"
    preview_scenarios_path = preflight_dir / "preview_scenarios.json"
    validate_payload = _build_preflight_validate_payload(
        cfg,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
        scenarios=scenarios,
        resolved_seeds=metadata["resolved_seeds"],
        scenario_horizons_summary=metadata["scenario_horizons_summary"],
        route_clearance_warnings=route_clearance_warnings,
        route_clearance_warning_summary=route_clearance_warning_summary,
        noise_spec=metadata["noise_spec"],
        noise_hash=metadata["noise_hash"],
        checkpoint_preflight_summary=checkpoint_preflight_report,
        checkpoint_preflight_mode=checkpoint_preflight_mode,
    )
    preview_payload = _build_preflight_preview_payload(
        cfg,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
        scenarios=scenarios,
        route_clearance_warnings=route_clearance_warnings,
        route_clearance_warning_summary=route_clearance_warning_summary,
    )
    _write_json(validate_config_path, validate_payload)
    _write_json(preview_scenarios_path, preview_payload)
    # Persist the per-arm checkpoint preflight summary next to the other preflight artifacts so the
    # submit path can record `submit_safe`/staging-status alongside the requeue packet (issue
    # #4613/#4663). `enforced_staged` writes `checkpoint_staging.json`; the cheap metadata-only
    # mode writes `checkpoint_resolvability.json` and clearly labels itself non-submit-safe when
    # any arm is only `stageable_remote`.
    checkpoint_preflight_report_path = (
        preflight_dir / _CHECKPOINT_PREFLIGHT_REPORT_NAME[checkpoint_preflight_mode]
    )
    _write_json(
        checkpoint_preflight_report_path,
        {
            "mode": checkpoint_preflight_mode,
            "stage": bool(checkpoint_preflight_report.get("stage")),
            "checked": int(checkpoint_preflight_report.get("checked", 0)),
            "resolved": int(checkpoint_preflight_report.get("resolved", 0)),
            "submit_safe": bool(checkpoint_preflight_report.get("submit_safe")),
            "arms": list(checkpoint_preflight_report.get("arms", [])),
        },
    )
    return validate_config_path, preview_scenarios_path, checkpoint_preflight_report_path


def _write_matrix_and_amv_artifacts(
    cfg: CampaignConfig,
    *,
    reports_dir: Path,
    scenarios: list[dict[str, Any]],
    resolved_seeds: list[int],
    scenario_hash: str,
    git_meta: dict[str, Any],
    campaign_id: str,
    created_at_utc: str,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    """Build and write matrix summary and AMV coverage artifacts.

    Returns:
        Tuple of ``(matrix_json_path, matrix_csv_path, amv_json_path, amv_md_path,
        amv_summary)``.

    Raises:
        ValueError: When paper-facing AMV coverage enforcement fails.
    """
    matrix_rows = _build_matrix_summary_rows(
        cfg,
        scenarios,
        resolved_seeds,
        scenario_hash=scenario_hash,
        git_meta=git_meta,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
    )
    matrix_summary_json_path, matrix_summary_csv_path = _write_matrix_summary_artifacts(
        reports_dir,
        matrix_rows,
    )
    amv_summary = _build_amv_coverage_summary(
        cfg,
        scenarios,
        campaign_id=campaign_id,
        generated_at_utc=created_at_utc,
    )
    amv_coverage_json_path, amv_coverage_md_path = _write_amv_coverage_artifacts(
        reports_dir,
        amv_summary,
    )
    if (
        cfg.paper_facing
        and amv_summary.get("status") == "fail"
        and cfg.amv_profile.coverage_enforcement == "error"
    ):
        raise ValueError(
            "AMV coverage contract validation failed: missing required AMV dimensions "
            "(coverage_enforcement=error)."
        )
    return (
        matrix_summary_json_path,
        matrix_summary_csv_path,
        amv_coverage_json_path,
        amv_coverage_md_path,
        amv_summary,
    )


def _build_comparability_artifacts_if_configured(
    cfg: CampaignConfig,
    *,
    reports_dir: Path,
    scenarios: list[dict[str, Any]],
    campaign_id: str,
    created_at_utc: str,
) -> tuple[dict[str, Any] | None, Path | None, Path | None, Path | None]:
    """Build comparability artifacts when a mapping path is configured.

    Returns:
        Tuple of ``(comparability_summary, comparability_json_path,
        comparability_md_path, comparability_mapping_path)``.
    """
    if cfg.comparability_mapping_path is None:
        return None, None, None, None
    comparability_summary: dict[str, Any] | None = None
    comparability_json_path: Path | None = None
    comparability_md_path: Path | None = None
    comparability_mapping_path: Path | None = None
    try:
        comparability_summary, comparability_mapping_path = _build_comparability_summary(
            cfg,
            scenarios,
            campaign_id=campaign_id,
            generated_at_utc=created_at_utc,
        )
        comparability_json_path, comparability_md_path = _write_comparability_artifacts(
            reports_dir,
            comparability_summary,
        )
    except (ValueError, FileNotFoundError, yaml.YAMLError):
        if cfg.paper_facing:
            raise
    return (
        comparability_summary,
        comparability_json_path,
        comparability_md_path,
        comparability_mapping_path,
    )


def _write_summary_artifacts(
    cfg: CampaignConfig,
    *,
    reports_dir: Path,
    scenarios: list[dict[str, Any]],
    metadata: dict[str, Any],
    campaign_id: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Write matrix summary, AMV coverage, and comparability artifacts.

    Returns:
        Dict with artifact paths and summaries keyed by artifact name.
    """
    mx_json, mx_csv, amv_json, amv_md, amv_summary = _write_matrix_and_amv_artifacts(
        cfg,
        reports_dir=reports_dir,
        scenarios=scenarios,
        resolved_seeds=metadata["resolved_seeds"],
        scenario_hash=metadata["scenario_hash"],
        git_meta=metadata["git_meta"],
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
    )
    comp_sum, comp_json, comp_md, comp_map = _build_comparability_artifacts_if_configured(
        cfg,
        reports_dir=reports_dir,
        scenarios=scenarios,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
    )
    return {
        "matrix_summary_json_path": mx_json,
        "matrix_summary_csv_path": mx_csv,
        "amv_coverage_json_path": amv_json,
        "amv_coverage_md_path": amv_md,
        "amv_summary": amv_summary,
        "comparability_summary": comp_sum,
        "comparability_json_path": comp_json,
        "comparability_md_path": comp_md,
        "comparability_mapping_path": comp_map,
    }


def _write_campaign_artifacts(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    preflight_dir: Path,
    reports_dir: Path,
    campaign_id: str,
    created_at_utc: str,
    scenarios: list[dict[str, Any]],
    metadata: dict[str, Any],
    route_clearance_warnings: list[dict[str, Any]],
    route_clearance_warning_summary: dict[str, Any],
    checkpoint_preflight_report: dict[str, Any],
    checkpoint_preflight_mode: CheckpointPreflightMode,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Write preflight and report artifacts.

    Returns:
        Tuple of preflight paths followed by the summary-artifact mapping.
    """
    validate_config_path, preview_scenarios_path, checkpoint_report_path = (
        _write_preflight_artifacts(
            cfg,
            preflight_dir=preflight_dir,
            campaign_id=campaign_id,
            created_at_utc=created_at_utc,
            scenarios=scenarios,
            metadata=metadata,
            route_clearance_warnings=route_clearance_warnings,
            route_clearance_warning_summary=route_clearance_warning_summary,
            checkpoint_preflight_report=checkpoint_preflight_report,
            checkpoint_preflight_mode=checkpoint_preflight_mode,
        )
    )
    summary_artifacts = _write_summary_artifacts(
        cfg,
        reports_dir=reports_dir,
        scenarios=scenarios,
        metadata=metadata,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
    )
    tuning_ledger = _build_tuning_provenance_ledger(
        cfg,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
        metadata=metadata,
    )
    tuning_ledger_path = reports_dir / "tuning_ledger.json"
    _write_json(tuning_ledger_path, tuning_ledger)
    summary_artifacts["tuning_ledger"] = tuning_ledger
    summary_artifacts["tuning_ledger_path"] = tuning_ledger_path
    return validate_config_path, preview_scenarios_path, checkpoint_report_path, summary_artifacts


def _build_manifest_planner_entries(
    cfg: CampaignConfig,
    checkpoint_preflight_report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build per-planner manifest entries with tuning and checkpoint provenance.

    Returns:
        List of JSON-serializable planner manifest entries.
    """
    return [
        {
            "key": planner.key,
            "algo": planner.algo,
            "human_model_variant": planner.human_model_variant,
            "human_model_source": planner.human_model_source,
            "planner_group": planner.planner_group,
            "benchmark_profile": planner.benchmark_profile,
            "algo_config_path": (
                _repo_relative(planner.algo_config_path)
                if planner.algo_config_path is not None
                else None
            ),
            "availability_gate": planner.availability_gate,
            "fail_closed_reason": planner.fail_closed_reason,
            "status": (
                "not_available" if planner.availability_gate == "dependency_gated" else "ok"
            ),
            "observation_mode": planner.observation_mode,
            "enabled": planner.enabled,
            "tuning": _tuning_effort_block(planner),
            "checkpoint_provenance": _checkpoint_provenance_block(
                planner, checkpoint_preflight_report
            ),
        }
        for planner in cfg.planners
    ]


def _build_manifest_artifact_block(  # noqa: PLR0913
    *,
    validate_config_path: Path,
    preview_scenarios_path: Path,
    checkpoint_preflight_report_path: Path,
    matrix_summary_json_path: Path,
    matrix_summary_csv_path: Path,
    amv_coverage_json_path: Path,
    amv_coverage_md_path: Path,
    comparability_json_path: Path | None,
    comparability_md_path: Path | None,
    tuning_ledger_path: Path,
) -> dict[str, Any]:
    """Build the artifact-path block for the campaign manifest.

    Returns:
        JSON-serializable artifact path mapping.
    """
    return {
        "preflight_validate_config": _repo_relative(validate_config_path),
        "preflight_preview_scenarios": _repo_relative(preview_scenarios_path),
        "preflight_checkpoint_provisioning": _repo_relative(checkpoint_preflight_report_path),
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
        "snqi_diagnostics_json": None,
        "snqi_diagnostics_md": None,
        "snqi_sensitivity_csv": None,
        "tuning_ledger": _repo_relative(tuning_ledger_path),
    }


def _build_manifest_context_block(
    cfg: CampaignConfig,
    *,
    campaign_id: str,
    created_at_utc: str,
    metadata: dict[str, Any],
    invoked_command: str | None,
) -> dict[str, Any]:
    """Build identity, scenario, and provenance fields for the campaign manifest.

    Returns:
        JSON-serializable manifest fields for campaign identity and provenance.
    """
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "name": cfg.name,
        "created_at_utc": created_at_utc,
        "started_at_utc": created_at_utc,
        "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
        "scenario_matrix_hash": metadata["scenario_hash"],
        "radius_binding": _radius_binding_metadata(cfg.radius_sweep),
        "scenario_candidates": list(cfg.scenario_candidates.names),
        "scenario_amv_overrides": {
            scenario_name: dict(values)
            for scenario_name, values in sorted(cfg.scenario_amv_overrides.items())
        },
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": metadata["resolved_seeds"],
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
        "git": metadata["git_meta"],
        "config_hash": metadata["config_hash"],
        "invoked_command": invoked_command,
        "paper_facing": bool(cfg.paper_facing),
        "paper_profile_version": cfg.paper_profile_version,
        "amv_profile_name": cfg.amv_profile.name,
        "amv_contract_version": cfg.amv_profile.contract_version,
    }


def _build_manifest_contract_block(
    cfg: CampaignConfig,
    *,
    metadata: dict[str, Any],
    route_clearance_warnings: list[dict[str, Any]],
    route_clearance_warning_summary: dict[str, Any],
    amv_summary: dict[str, Any],
    comparability_summary: dict[str, Any] | None,
    comparability_mapping_path: Path | None,
) -> dict[str, Any]:
    """Build benchmark-contract and evidence fields for the campaign manifest.

    Returns:
        JSON-serializable manifest fields for benchmark contract metadata.
    """
    return {
        "amv_coverage_enforcement": cfg.amv_profile.coverage_enforcement,
        "amv_coverage_status": amv_summary.get("status", "unknown"),
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
        "comparability_mapping_path": (
            _repo_relative(comparability_mapping_path) if comparability_mapping_path else None
        ),
        "comparability_mapping_version": (
            comparability_summary.get("mapping_version") if comparability_summary else None
        ),
        "comparability_mapping_hash": (
            comparability_summary.get("mapping_hash") if comparability_summary else None
        ),
        "route_clearance_warnings": route_clearance_warnings,
        "route_clearance_warning_count": len(route_clearance_warnings),
        "route_clearance_warning_summary": route_clearance_warning_summary,
        "route_clearance_certifications_path": (
            _repo_relative(cfg.route_clearance_certifications_path)
            if cfg.route_clearance_certifications_path is not None
            else None
        ),
        "scenario_horizons": metadata["scenario_horizons_summary"],
        "observation_noise": metadata["noise_spec"],
        "observation_noise_hash": metadata["noise_hash"],
        "snqi_weights_path": (
            _repo_relative(cfg.snqi_weights_path) if cfg.snqi_weights_path is not None else None
        ),
        "snqi_baseline_path": (
            _repo_relative(cfg.snqi_baseline_path) if cfg.snqi_baseline_path is not None else None
        ),
        "retained_metric_contract_path": (
            _repo_relative(cfg.retained_metric_contract_path)
            if cfg.retained_metric_contract_path is not None
            else None
        ),
        "snqi_contract_enabled": bool(cfg.snqi_contract.enabled),
        "snqi_contract_enforcement": cfg.snqi_contract.enforcement,
        "snqi_contract_status": "not_evaluated",
        "snqi_positioning_recommendation": "not_evaluated",
        "snqi_positioning_claim_scope": "benchmark aggregate, not a universal ground-truth utility",
    }


def _build_manifest_execution_block(
    cfg: CampaignConfig,
    planner_entries: list[dict[str, Any]],
    *,
    tuning_ledger: dict[str, Any],
    tuning_ledger_path: Path,
) -> dict[str, Any]:
    """Build planner, tuning, and execution metadata for the campaign manifest.

    Returns:
        JSON-serializable manifest fields for planner and execution metadata.
    """
    return {
        "planners": planner_entries,
        "tuning_effort_enforcement": cfg.tuning_effort_enforcement,
        "tuning_effort_summary": _tuning_effort_summary(cfg.planners),
        "tuning_run_provenance": {
            "run_class": (
                cfg.tuning_run_provenance.run_class
                if cfg.tuning_run_provenance is not None
                else "debug"
            ),
            "ledger_schema_version": tuning_ledger["schema_version"],
            "record_schema_version": tuning_ledger["record_schema_version"],
            "ledger_sha256": tuning_ledger["ledger_sha256"],
            "record_count": len(tuning_ledger["records"]),
            "summary": tuning_ledger["summary"],
            "ledger_path": _repo_relative(tuning_ledger_path),
            "policy": tuning_ledger["policy"],
        },
        "checkpoint_provenance_enforcement": cfg.checkpoint_provenance_enforcement,
        "kinematics_matrix": list(cfg.kinematics_matrix),
        "holonomic_command_mode": cfg.holonomic_command_mode,
        "observation_mode": cfg.observation_mode,
        "repository_url": cfg.repository_url,
        "release_tag": cfg.release_tag,
        "doi": cfg.doi,
    }


def _build_campaign_manifest_payload(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    campaign_id: str,
    created_at_utc: str,
    metadata: dict[str, Any],
    invoked_command: str | None,
    route_clearance_warnings: list[dict[str, Any]],
    route_clearance_warning_summary: dict[str, Any],
    amv_summary: dict[str, Any],
    comparability_summary: dict[str, Any] | None,
    comparability_mapping_path: Path | None,
    planner_entries: list[dict[str, Any]],
    artifact_block: dict[str, Any],
    tuning_ledger: dict[str, Any],
    tuning_ledger_path: Path,
) -> dict[str, Any]:
    """Build the complete JSON-serializable campaign manifest payload.

    Returns:
        Complete JSON-serializable campaign manifest payload.
    """
    return {
        **_build_manifest_context_block(
            cfg,
            campaign_id=campaign_id,
            created_at_utc=created_at_utc,
            metadata=metadata,
            invoked_command=invoked_command,
        ),
        **_build_manifest_contract_block(
            cfg,
            metadata=metadata,
            route_clearance_warnings=route_clearance_warnings,
            route_clearance_warning_summary=route_clearance_warning_summary,
            amv_summary=amv_summary,
            comparability_summary=comparability_summary,
            comparability_mapping_path=comparability_mapping_path,
        ),
        **_build_manifest_execution_block(
            cfg,
            planner_entries,
            tuning_ledger=tuning_ledger,
            tuning_ledger_path=tuning_ledger_path,
        ),
        "artifacts": artifact_block,
    }


def _finalize_campaign_preflight(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    campaign_id: str,
    campaign_root: Path,
    reports_dir: Path,
    preflight_dir: Path,
    created_at_utc: str,
    metadata: dict[str, Any],
    invoked_command: str | None,
    scenarios: list[dict[str, Any]],
    route_clearance_warnings: list[dict[str, Any]],
    route_clearance_warning_summary: dict[str, Any],
    checkpoint_preflight_report: dict[str, Any],
    validate_config_path: Path,
    preview_scenarios_path: Path,
    checkpoint_preflight_report_path: Path,
    summary_artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Build manifest, verify resume context, write manifest, and assemble result.

    Returns:
        Paths and metadata required by preflight-only workflows and full runs.

    Raises:
        OrcaRvo2PreflightError: When ORCA planners require ``rvo2`` but it is missing.
        RouteClearanceError: When a route centerline is geometrically infeasible.
        CampaignCheckpointPreflightError: When a checkpoint cannot be resolved.
        CampaignPolicyDependencyPreflightError: When a policy dependency is missing.
        CampaignScenarioMapPreflightError: When a scenario map file cannot resolve.
    """
    manifest_payload = _build_campaign_manifest_payload(
        cfg,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
        metadata=metadata,
        invoked_command=invoked_command,
        route_clearance_warnings=route_clearance_warnings,
        route_clearance_warning_summary=route_clearance_warning_summary,
        amv_summary=summary_artifacts["amv_summary"],
        comparability_summary=summary_artifacts["comparability_summary"],
        comparability_mapping_path=summary_artifacts["comparability_mapping_path"],
        planner_entries=_build_manifest_planner_entries(cfg, checkpoint_preflight_report),
        artifact_block=_build_manifest_artifact_block(
            validate_config_path=validate_config_path,
            preview_scenarios_path=preview_scenarios_path,
            checkpoint_preflight_report_path=checkpoint_preflight_report_path,
            matrix_summary_json_path=summary_artifacts["matrix_summary_json_path"],
            matrix_summary_csv_path=summary_artifacts["matrix_summary_csv_path"],
            amv_coverage_json_path=summary_artifacts["amv_coverage_json_path"],
            amv_coverage_md_path=summary_artifacts["amv_coverage_md_path"],
            comparability_json_path=summary_artifacts["comparability_json_path"],
            comparability_md_path=summary_artifacts["comparability_md_path"],
            tuning_ledger_path=summary_artifacts["tuning_ledger_path"],
        ),
        tuning_ledger=summary_artifacts["tuning_ledger"],
        tuning_ledger_path=summary_artifacts["tuning_ledger_path"],
    )
    _verify_existing_resume_context(
        cfg,
        campaign_root=campaign_root,
        campaign_id=campaign_id,
        config_hash=metadata["config_hash"],
    )
    _write_json(campaign_root / "campaign_manifest.json", manifest_payload)
    return {
        "campaign_id": campaign_id,
        "campaign_root": campaign_root,
        "reports_dir": reports_dir,
        "preflight_dir": preflight_dir,
        "validate_config_path": validate_config_path,
        "preview_scenarios_path": preview_scenarios_path,
        "checkpoint_preflight_report_path": checkpoint_preflight_report_path,
        "checkpoint_preflight_summary": checkpoint_preflight_report,
        "matrix_summary_json_path": summary_artifacts["matrix_summary_json_path"],
        "matrix_summary_csv_path": summary_artifacts["matrix_summary_csv_path"],
        "amv_coverage_json_path": summary_artifacts["amv_coverage_json_path"],
        "amv_coverage_md_path": summary_artifacts["amv_coverage_md_path"],
        "amv_summary": summary_artifacts["amv_summary"],
        "comparability_json_path": summary_artifacts["comparability_json_path"],
        "comparability_md_path": summary_artifacts["comparability_md_path"],
        "tuning_ledger_path": summary_artifacts["tuning_ledger_path"],
        "tuning_ledger": summary_artifacts["tuning_ledger"],
        "manifest_payload": manifest_payload,
        "created_at_utc": created_at_utc,
        "scenarios": scenarios,
        "resolved_seeds": metadata["resolved_seeds"],
        "scenario_hash": metadata["scenario_hash"],
        "git_meta": metadata["git_meta"],
        "config_hash": metadata["config_hash"],
    }


def prepare_campaign_preflight(  # noqa: PLR0913
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    invoked_command: str | None = None,
    validate_campaign_config: Callable[[CampaignConfig], None] | None = None,
    build_route_clearance_warnings: Callable[..., list[dict[str, Any]]] | None = None,
    checkpoint_preflight_mode: CheckpointPreflightMode = "metadata_only",
    checkpoint_cache_dir: Path | None = None,
    checkpoint_registry_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare campaign preflight artifacts and matrix-definition summary.

    Returns:
        Paths and metadata required by preflight-only workflows and full runs.
    """
    if validate_campaign_config is None:
        from robot_sf.benchmark.camera_ready_campaign import (  # noqa: PLC0415
            _validate_campaign_config as validate_campaign_config,
        )
    if build_route_clearance_warnings is None:
        build_route_clearance_warnings = _build_route_clearance_warnings
    validate_campaign_config(cfg)
    _assert_radius_sweep_preflight_ready(cfg.radius_sweep)
    ckpt_report, campaign_id, campaign_root, reports_dir, preflight_dir = (
        _validate_and_setup_campaign(
            cfg,
            checkpoint_preflight_mode=checkpoint_preflight_mode,
            checkpoint_cache_dir=checkpoint_cache_dir,
            checkpoint_registry_path=checkpoint_registry_path,
            output_root=output_root,
            label=label,
            campaign_id=campaign_id,
        )
    )
    created_at_utc = _utc_now()
    scenarios, rc_warnings, rc_summary, meta = _load_and_prepare_scenarios(
        cfg,
        build_route_clearance_warnings,
    )
    vc_path, ps_path, ckpt_path, summary_artifacts = _write_campaign_artifacts(
        cfg,
        preflight_dir=preflight_dir,
        reports_dir=reports_dir,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
        scenarios=scenarios,
        metadata=meta,
        route_clearance_warnings=rc_warnings,
        route_clearance_warning_summary=rc_summary,
        checkpoint_preflight_report=ckpt_report,
        checkpoint_preflight_mode=checkpoint_preflight_mode,
    )
    return _finalize_campaign_preflight(
        cfg,
        campaign_id=campaign_id,
        campaign_root=campaign_root,
        reports_dir=reports_dir,
        preflight_dir=preflight_dir,
        created_at_utc=created_at_utc,
        metadata=meta,
        invoked_command=invoked_command,
        scenarios=scenarios,
        route_clearance_warnings=rc_warnings,
        route_clearance_warning_summary=rc_summary,
        checkpoint_preflight_report=ckpt_report,
        validate_config_path=vc_path,
        preview_scenarios_path=ps_path,
        checkpoint_preflight_report_path=ckpt_path,
        summary_artifacts=summary_artifacts,
    )
