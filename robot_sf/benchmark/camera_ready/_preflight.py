"""Preflight payload helpers for camera-ready benchmark campaigns."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.camera_ready._artifacts import (
    _write_amv_coverage_artifacts,
    _write_comparability_artifacts,
    _write_json,
    _write_matrix_summary_artifacts,
)
from robot_sf.benchmark.camera_ready._config import (
    _load_campaign_scenarios,
    _resolved_seed_inventory,
    _scenario_horizon_summary,
)
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
from robot_sf.benchmark.latency_stress import not_available_latency_metrics
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.orca_preflight import check_orca_rvo2_preflight
from robot_sf.benchmark.utils import _config_hash
from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path

CAMPAIGN_SCHEMA_VERSION = "benchmark-camera-ready-campaign.v1"

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from robot_sf.benchmark.camera_ready_campaign_config import CampaignConfig


def _scenario_display_name(scenario: dict[str, Any]) -> str:
    """Return the stable scenario identifier used in preflight payloads."""
    return str(scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "")


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
) -> dict[str, Any]:
    """Build the ``validate_config.json`` preflight artifact payload.

    Returns:
        JSON-serializable preflight validation artifact payload.
    """
    payload: dict[str, Any] = {
        "schema_version": "benchmark-preflight-validate-config.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": created_at_utc,
        "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
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
        payload["scenarios"] = scenarios
    return payload


def prepare_campaign_preflight(
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    invoked_command: str | None = None,
    validate_campaign_config: Callable[[CampaignConfig], None] | None = None,
    build_route_clearance_warnings: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Prepare campaign preflight artifacts and matrix-definition summary.

    Returns:
        Paths and metadata required by preflight-only workflows and full runs.

    Raises:
        OrcaRvo2PreflightError: When enabled ORCA-dependent planners require ``rvo2`` but it is
            not importable.
        RouteClearanceError: When any scenario route centerline lies closer to a static obstacle
            than the robot radius, making the route geometrically impossible to follow without
            collision.
    """
    if validate_campaign_config is None:
        from robot_sf.benchmark.camera_ready_campaign import (  # noqa: PLC0415
            _validate_campaign_config as validate_campaign_config,
        )
    if build_route_clearance_warnings is None:
        build_route_clearance_warnings = _build_route_clearance_warnings

    validate_campaign_config(cfg)
    check_orca_rvo2_preflight(cfg)
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

    created_at_utc = _utc_now()
    scenarios = _load_campaign_scenarios(cfg)
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
    resolved_seeds = _resolved_seed_inventory(scenarios)
    scenario_hash = _hash_payload(scenarios)
    scenario_horizons_summary = _scenario_horizon_summary(
        scenarios,
        schedule_path=cfg.scenario_horizons_path,
    )
    git_meta = _git_context()
    config_hash = _config_hash(_jsonable_repo_relative(asdict(cfg)))
    noise_spec = normalize_observation_noise_spec(cfg.observation_noise)
    noise_hash = observation_noise_hash(noise_spec)

    validate_config_path = preflight_dir / "validate_config.json"
    preview_scenarios_path = preflight_dir / "preview_scenarios.json"
    validate_payload = _build_preflight_validate_payload(
        cfg,
        campaign_id=campaign_id,
        created_at_utc=created_at_utc,
        scenarios=scenarios,
        resolved_seeds=resolved_seeds,
        scenario_horizons_summary=scenario_horizons_summary,
        route_clearance_warnings=route_clearance_warnings,
        route_clearance_warning_summary=route_clearance_warning_summary,
        noise_spec=noise_spec,
        noise_hash=noise_hash,
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

    comparability_summary: dict[str, Any] | None = None
    comparability_json_path: Path | None = None
    comparability_md_path: Path | None = None
    comparability_mapping_path: Path | None = None
    if cfg.comparability_mapping_path is not None:
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

    manifest_payload: dict[str, Any] = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "name": cfg.name,
        "created_at_utc": created_at_utc,
        "started_at_utc": created_at_utc,
        "scenario_matrix": _repo_relative(cfg.scenario_matrix_path),
        "scenario_matrix_hash": scenario_hash,
        "scenario_candidates": list(cfg.scenario_candidates.names),
        "scenario_amv_overrides": {
            scenario_name: dict(values)
            for scenario_name, values in sorted(cfg.scenario_amv_overrides.items())
        },
        "seed_policy": {
            "mode": cfg.seed_policy.mode,
            "seed_set": cfg.seed_policy.seed_set,
            "seeds": list(cfg.seed_policy.seeds),
            "resolved_seeds": resolved_seeds,
            "seed_sets_path": _repo_relative(cfg.seed_policy.seed_sets_path),
        },
        "git": git_meta,
        "config_hash": config_hash,
        "invoked_command": invoked_command,
        "paper_facing": bool(cfg.paper_facing),
        "paper_profile_version": cfg.paper_profile_version,
        "amv_profile_name": cfg.amv_profile.name,
        "amv_contract_version": cfg.amv_profile.contract_version,
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
        "scenario_horizons": scenario_horizons_summary,
        "observation_noise": noise_spec,
        "observation_noise_hash": noise_hash,
        "snqi_weights_path": (
            _repo_relative(cfg.snqi_weights_path) if cfg.snqi_weights_path is not None else None
        ),
        "snqi_baseline_path": (
            _repo_relative(cfg.snqi_baseline_path) if cfg.snqi_baseline_path is not None else None
        ),
        "snqi_contract_enabled": bool(cfg.snqi_contract.enabled),
        "snqi_contract_enforcement": cfg.snqi_contract.enforcement,
        "snqi_contract_status": "not_evaluated",
        "snqi_positioning_recommendation": "not_evaluated",
        "snqi_positioning_claim_scope": "benchmark aggregate, not a universal ground-truth utility",
        "planners": [
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
                "observation_mode": planner.observation_mode,
                "enabled": planner.enabled,
            }
            for planner in cfg.planners
        ],
        "kinematics_matrix": list(cfg.kinematics_matrix),
        "holonomic_command_mode": cfg.holonomic_command_mode,
        "observation_mode": cfg.observation_mode,
        "repository_url": cfg.repository_url,
        "release_tag": cfg.release_tag,
        "doi": cfg.doi,
        "artifacts": {
            "preflight_validate_config": _repo_relative(validate_config_path),
            "preflight_preview_scenarios": _repo_relative(preview_scenarios_path),
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
        },
    }
    _write_json(campaign_root / "campaign_manifest.json", manifest_payload)
    return {
        "campaign_id": campaign_id,
        "campaign_root": campaign_root,
        "reports_dir": reports_dir,
        "preflight_dir": preflight_dir,
        "validate_config_path": validate_config_path,
        "preview_scenarios_path": preview_scenarios_path,
        "matrix_summary_json_path": matrix_summary_json_path,
        "matrix_summary_csv_path": matrix_summary_csv_path,
        "amv_coverage_json_path": amv_coverage_json_path,
        "amv_coverage_md_path": amv_coverage_md_path,
        "amv_summary": amv_summary,
        "comparability_json_path": comparability_json_path,
        "comparability_md_path": comparability_md_path,
        "manifest_payload": manifest_payload,
        "created_at_utc": created_at_utc,
        "scenarios": scenarios,
        "resolved_seeds": resolved_seeds,
        "scenario_hash": scenario_hash,
        "git_meta": git_meta,
        "config_hash": config_hash,
    }
