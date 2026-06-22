"""Preflight payload helpers for camera-ready benchmark campaigns."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.camera_ready._util import _repo_relative
from robot_sf.benchmark.latency_stress import not_available_latency_metrics

if TYPE_CHECKING:
    from robot_sf.benchmark.camera_ready_campaign_config import CampaignConfig
    from robot_sf.benchmark.latency_stress import LatencyStressProfile
    from robot_sf.benchmark.synthetic_actuation import SyntheticActuationProfile


def _synthetic_actuation_metadata(
    profile: SyntheticActuationProfile | None,
) -> dict[str, Any] | None:
    """Return a JSON-safe synthetic-actuation metadata payload when configured."""
    if profile is None:
        return None
    return profile.to_metadata()


def _latency_stress_metadata(
    profile: LatencyStressProfile | None,
    *,
    dt: float | None = None,
) -> dict[str, Any] | None:
    """Return a JSON-safe latency-stress metadata payload when configured."""
    if profile is None:
        return None
    return profile.to_metadata(dt=dt)


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
