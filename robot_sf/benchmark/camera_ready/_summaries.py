"""Pure summary builders for camera-ready benchmark campaigns."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import yaml

from robot_sf.benchmark.camera_ready._util import (
    _hash_payload,
    _jsonable_repo_relative,
    _kinematics_matrix_or_default,
    _latency_stress_metadata,
    _repo_relative,
    _synthetic_actuation_metadata,
)
from robot_sf.benchmark.camera_ready_campaign_config import _AMV_DIMENSIONS, CampaignConfig
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.seed_variance import (
    build_seed_variability_rows,
    build_statistical_sufficiency_rows,
)
from robot_sf.benchmark.synthetic_actuation import CALIBRATED_ACTUATION_CLAIM_SCOPE
from robot_sf.benchmark.utils import _config_hash

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.benchmark.synthetic_actuation import SyntheticActuationProfile

_SEED_VARIABILITY_METRICS: tuple[str, ...] = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "snqi",
)
_ACTUATION_REPORT_METRICS: tuple[str, ...] = (
    "success",
    "total_collision_count",
    "near_misses",
    "min_clearance",
    "time_to_collision_min",
    "time_to_goal_norm",
    "failure_to_progress",
    "stalled_time",
    "velocity_max",
    "acceleration_max",
    "jerk_mean",
    "jerk_max",
    "curvature_mean",
    "energy",
    "command_clip_fraction",
    "yaw_rate_saturation_fraction",
    "signed_braking_peak_m_s2",
)


def _build_matrix_summary_rows(
    cfg: CampaignConfig,
    scenarios: list[dict[str, Any]],
    resolved_seeds: list[int],
    *,
    scenario_hash: str,
    git_meta: dict[str, str],
    campaign_id: str,
    created_at_utc: str,
) -> list[dict[str, Any]]:
    """Build planner/kinematics matrix-definition summary rows.

    Returns:
        Deterministically ordered matrix rows for CSV/JSON artifacts.
    """
    matrix_path = _repo_relative(cfg.scenario_matrix_path)
    config_hash = _config_hash(_jsonable_repo_relative(asdict(cfg)))
    noise_spec = normalize_observation_noise_spec(cfg.observation_noise)
    repeats = len(resolved_seeds)
    horizon_mode = "scenario_horizons" if cfg.scenario_horizons_path is not None else "fixed"
    scenario_horizons_path = (
        _repo_relative(cfg.scenario_horizons_path) if cfg.scenario_horizons_path is not None else ""
    )
    rows: list[dict[str, Any]] = []
    normalized_kinematics = _kinematics_matrix_or_default(cfg.kinematics_matrix)
    for planner in cfg.planners:
        if not planner.enabled:
            continue
        active_observation_mode = planner.observation_mode or cfg.observation_mode
        for kinematics in normalized_kinematics:
            rows.append(
                {
                    "scenario_matrix": matrix_path,
                    "scenario_matrix_hash": scenario_hash,
                    "scenario_count": len(scenarios),
                    "scenario_candidates": list(cfg.scenario_candidates.names),
                    "synthetic_actuation_profile": _synthetic_actuation_metadata(
                        cfg.synthetic_actuation_profile
                    ),
                    "latency_stress_profile": _latency_stress_metadata(
                        cfg.latency_stress_profile,
                        dt=cfg.dt,
                    ),
                    "planner_key": planner.key,
                    "algo": planner.algo,
                    "human_model_variant": planner.human_model_variant,
                    "human_model_source": planner.human_model_source,
                    "planner_group": planner.planner_group,
                    "benchmark_profile": planner.benchmark_profile,
                    "observation_mode": active_observation_mode,
                    "kinematics": kinematics,
                    "seed_policy.mode": cfg.seed_policy.mode,
                    "seed_policy.seed_set": cfg.seed_policy.seed_set,
                    "resolved_seeds": list(resolved_seeds),
                    "repeats": repeats,
                    "horizon_mode": horizon_mode,
                    "horizon": cfg.horizon,
                    "scenario_horizons_path": scenario_horizons_path,
                    "paper_facing": bool(cfg.paper_facing),
                    "paper_profile_version": cfg.paper_profile_version,
                    "observation_noise.profile": noise_spec["profile"],
                    "observation_noise.enabled": bool(noise_spec["enabled"]),
                    "observation_noise_hash": observation_noise_hash(noise_spec),
                    "config_hash": config_hash,
                    "git_commit": git_meta.get("commit", "unknown"),
                    "campaign_id": campaign_id,
                    "created_at_utc": created_at_utc,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row.get("planner_group", "")),
            str(row.get("planner_key", "")),
            str(row.get("kinematics", "")),
        )
    )
    return rows


def _build_seed_variability_payload(
    records: list[dict[str, Any]],
    *,
    campaign_id: str,
    generated_at_utc: str,
    config_hash: str,
    git_hash: str,
    seed_policy: dict[str, Any],
    confidence_settings: dict[str, Any],
    source_paths: dict[str, Any],
) -> dict[str, Any]:
    """Build paper-facing seed-variability export payload from campaign episodes.

    Returns:
        JSON-serializable payload for ``seed_variability_by_scenario.json``.
    """
    rows = build_seed_variability_rows(
        records,
        metrics=_SEED_VARIABILITY_METRICS,
        campaign_id=campaign_id,
        config_hash=config_hash,
        git_hash=git_hash,
        seed_policy=seed_policy,
        confidence_settings=confidence_settings,
    )
    return {
        "schema_version": "benchmark-seed-variability-by-scenario.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": generated_at_utc,
        "metrics": list(_SEED_VARIABILITY_METRICS),
        "confidence": dict(confidence_settings),
        "source": source_paths,
        "row_count": len(rows),
        "rows": rows,
    }


def _build_statistical_sufficiency_payload(
    *,
    campaign_id: str,
    generated_at_utc: str,
    seed_variability_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build a thin statistical-sufficiency summary from seed-variability rows.

    Returns:
        JSON-serializable statistical sufficiency payload.
    """
    rows = build_statistical_sufficiency_rows(
        seed_variability_payload.get("rows") or [],
        metrics=seed_variability_payload.get("metrics") or [],
    )
    return {
        "schema_version": "benchmark-seed-statistical-sufficiency.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": generated_at_utc,
        "confidence": dict(seed_variability_payload.get("confidence") or {}),
        "row_count": len(rows),
        "rows": rows,
    }


def _scenario_family_from_scenario(scenario: dict[str, Any]) -> str:
    """Resolve scenario-family/archetype label from scenario metadata.

    Returns:
        str: Best-effort scenario-family label.
    """
    metadata = scenario.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("archetype", "scenario_family", "family"):
        value = metadata.get(key) or scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("scenario_id", "name", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _extract_amv_taxonomy(scenario: dict[str, Any]) -> dict[str, str]:
    """Extract AMV taxonomy fields from one scenario definition.

    Returns:
        dict[str, str]: AMV taxonomy values keyed by dimension name.
    """
    amv = scenario.get("amv")
    if not isinstance(amv, dict):
        metadata = scenario.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("amv"), dict):
            amv = metadata["amv"]
        else:
            amv = {}
    resolved: dict[str, str] = {}
    for dimension in _AMV_DIMENSIONS:
        value = amv.get(dimension)
        if isinstance(value, str) and value.strip():
            resolved[dimension] = value.strip()
    return resolved


def _build_amv_coverage_summary(
    cfg: CampaignConfig,
    scenarios: list[dict[str, Any]],
    *,
    campaign_id: str,
    generated_at_utc: str,
) -> dict[str, Any]:
    """Build AMV scope coverage summary for preflight/report artifacts.

    Returns:
        dict[str, Any]: JSON-serializable AMV coverage summary payload.
    """
    observed_by_dimension: dict[str, set[str]] = {dimension: set() for dimension in _AMV_DIMENSIONS}
    by_scenario: list[dict[str, Any]] = []
    for scenario in scenarios:
        taxonomy = _extract_amv_taxonomy(scenario)
        for dimension, value in taxonomy.items():
            observed_by_dimension[dimension].add(value)
        by_scenario.append(
            {
                "name": scenario.get("name") or scenario.get("scenario_id") or scenario.get("id"),
                "scenario_family": _scenario_family_from_scenario(scenario),
                "amv": taxonomy,
            }
        )

    required = {
        dimension: sorted(
            str(v).strip() for v in cfg.amv_profile.required_dimensions.get(dimension, ()) if v
        )
        for dimension in _AMV_DIMENSIONS
    }
    observed = {dimension: sorted(values) for dimension, values in observed_by_dimension.items()}
    missing = {
        dimension: [value for value in required[dimension] if value not in observed[dimension]]
        for dimension in _AMV_DIMENSIONS
    }
    has_missing = any(missing_values for missing_values in missing.values())
    enforcement = cfg.amv_profile.coverage_enforcement
    status = "pass"
    if has_missing:
        status = "fail" if enforcement == "error" else "warn"

    return {
        "schema_version": "benchmark-amv-coverage-summary.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": generated_at_utc,
        "profile_name": cfg.amv_profile.name,
        "contract_version": cfg.amv_profile.contract_version,
        "coverage_enforcement": enforcement,
        "status": status,
        "required_dimensions": required,
        "observed_dimensions": observed,
        "missing_dimensions": missing,
        "scenario_count": len(scenarios),
        "scenario_rows": by_scenario,
    }


def _build_actuation_envelope_summary(
    *,
    campaign_id: str,
    generated_at_utc: str,
    profile: SyntheticActuationProfile,
    planner_rows: list[dict[str, Any]],
    amv_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a synthetic actuation-envelope diagnostic summary payload.

    Returns:
        JSON-safe actuation-envelope summary payload.
    """
    rows: list[dict[str, Any]] = []
    for planner_row in planner_rows:
        planner_command_space = str(planner_row.get("planner_command_space", "unknown"))
        benchmark_command_space = str(planner_row.get("benchmark_command_space", "unknown"))
        projection_policy = str(planner_row.get("projection_policy", "unknown"))
        saturation = {
            "command_clip_fraction": str(
                planner_row.get("command_clip_fraction_mean", "not_available")
            ),
            "yaw_rate_saturation_fraction": str(
                planner_row.get("yaw_rate_saturation_fraction_mean", "not_available")
            ),
            "signed_braking_peak_m_s2": str(
                planner_row.get("signed_braking_peak_m_s2_mean", "not_available")
            ),
        }
        rows.append(
            {
                "planner_key": str(planner_row.get("planner_key", "")),
                "algo": str(planner_row.get("algo", "")),
                "planner_group": str(planner_row.get("planner_group", "")),
                "kinematics": str(planner_row.get("kinematics", "")),
                "status": str(planner_row.get("status", "")),
                "readiness_status": str(planner_row.get("readiness_status", "")),
                "availability_status": str(planner_row.get("availability_status", "")),
                "benchmark_success": str(planner_row.get("benchmark_success", "")),
                "execution_mode": str(planner_row.get("execution_mode", "unknown")),
                "execution_detail": str(planner_row.get("execution_detail", "unspecified")),
                "planner_command_space": planner_command_space,
                "benchmark_command_space": benchmark_command_space,
                "projection_policy": projection_policy,
                "projection_metadata_status": (
                    "explicit"
                    if all(
                        value and value != "unknown"
                        for value in (
                            planner_command_space,
                            benchmark_command_space,
                            projection_policy,
                        )
                    )
                    else "unknown"
                ),
                "metric_means": {
                    metric: str(planner_row.get(f"{metric}_mean", "nan"))
                    for metric in _ACTUATION_REPORT_METRICS
                    if metric not in saturation
                },
                "saturation_metrics": saturation,
            }
        )
    scenario_amv_rows: list[dict[str, Any]] = []
    amv_coverage_status = "unknown"
    if isinstance(amv_summary, dict):
        amv_coverage_status = str(amv_summary.get("status", "unknown"))
        raw_rows = amv_summary.get("scenario_rows")
        if isinstance(raw_rows, list):
            for row in raw_rows:
                if not isinstance(row, dict):
                    continue
                amv = row.get("amv")
                scenario_amv_rows.append(
                    {
                        "name": str(row.get("name", "")),
                        "scenario_family": str(row.get("scenario_family", "")),
                        "amv": dict(amv) if isinstance(amv, dict) else {},
                    }
                )
    return {
        "schema_version": "benchmark-actuation-envelope-summary.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": generated_at_utc,
        "paper_facing": False,
        "claim_boundary": profile.claim_boundary,
        "actuation_profile_type": (
            "calibrated_amv_actuation"
            if profile.claim_scope == CALIBRATED_ACTUATION_CLAIM_SCOPE
            else "synthetic_diagnostic"
        ),
        "synthetic_actuation_profile": profile.to_metadata(),
        "amv_coverage_status": amv_coverage_status,
        "scenario_amv_rows": scenario_amv_rows,
        "row_count": len(rows),
        "rows": rows,
    }


def _load_comparability_mapping(path: Path) -> dict[str, Any]:
    """Load and validate Alyassi comparability mapping configuration.

    Returns:
        dict[str, Any]: Parsed and validated comparability mapping payload.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Comparability mapping must be a mapping: {path}")
    mapping_version = payload.get("mapping_version")
    if not isinstance(mapping_version, str) or not mapping_version.strip():
        raise ValueError("Comparability mapping requires non-empty 'mapping_version'")
    family_map = payload.get("scenario_family_mapping")
    if not isinstance(family_map, dict):
        raise ValueError("Comparability mapping requires 'scenario_family_mapping' mapping")
    _validate_family_map(family_map)

    metric_map = payload.get("metric_comparability")
    if not isinstance(metric_map, dict):
        raise ValueError("Comparability mapping requires 'metric_comparability' mapping")
    _validate_metric_map(metric_map)

    planner_key_map = payload.get("planner_key_mapping")
    if planner_key_map is not None:
        if not isinstance(planner_key_map, dict):
            raise ValueError("Comparability mapping 'planner_key_mapping' must be a mapping")
        _validate_planner_key_map(planner_key_map)

    extensions = payload.get("amv_specific_extensions", [])
    if not isinstance(extensions, list):
        raise ValueError("Comparability mapping 'amv_specific_extensions' must be a list")
    return payload


def _validate_family_map(family_map: dict[str, Any]) -> None:
    """Validate scenario-family mapping payload."""
    for key, value in family_map.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError("scenario_family_mapping entries must be non-empty string->string")


def _validate_metric_map(metric_map: dict[str, Any]) -> None:
    """Validate metric comparability mapping payload."""
    for metric, config in metric_map.items():
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError("metric_comparability keys must be non-empty strings")
        if not isinstance(config, dict):
            raise ValueError(f"metric_comparability[{metric}] must be a mapping")
        classification = config.get("classification")
        if classification not in {"comparable", "proxy", "amv_specific"}:
            raise ValueError(
                f"metric_comparability[{metric}] classification must be one of comparable|proxy|amv_specific"
            )


def _validate_planner_key_map(planner_key_map: dict[str, Any]) -> None:
    """Validate planner-key mapping payload."""
    for key, value in planner_key_map.items():
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
        ):
            raise ValueError("planner_key_mapping entries must be non-empty string->string")


def _build_comparability_summary(
    cfg: CampaignConfig,
    scenarios: list[dict[str, Any]],
    *,
    campaign_id: str,
    generated_at_utc: str,
) -> tuple[dict[str, Any], Path]:
    """Build comparability report payload from mapping and scenario taxonomy.

    Returns:
        tuple[dict[str, Any], Path]: Summary payload and resolved mapping path.
    """
    mapping_path = cfg.comparability_mapping_path
    if mapping_path is None:
        raise ValueError("No comparability mapping configured for campaign")
    mapping = _load_comparability_mapping(mapping_path)
    family_map = mapping["scenario_family_mapping"]
    metric_map = mapping["metric_comparability"]
    planner_key_map = mapping.get("planner_key_mapping")
    if cfg.paper_facing:
        if not isinstance(planner_key_map, dict):
            raise ValueError(
                "Comparability mapping requires 'planner_key_mapping' mapping for paper-facing campaigns"
            )
        missing_planner_keys = [
            planner.key
            for planner in cfg.planners
            if planner.enabled and planner.key not in planner_key_map
        ]
        if missing_planner_keys:
            missing = ", ".join(sorted(missing_planner_keys))
            raise ValueError(
                "Comparability mapping is missing planner_key_mapping entries for enabled planners: "
                f"{missing}"
            )

    family_counts: dict[str, int] = {}
    for scenario in scenarios:
        family = _scenario_family_from_scenario(scenario)
        family_counts[family] = family_counts.get(family, 0) + 1

    overlap_rows: list[dict[str, Any]] = []
    for family, count in sorted(family_counts.items()):
        mapped = family_map.get(family)
        overlap_rows.append(
            {
                "robot_sf_family": family,
                "scenario_count": count,
                "alyassi_category": mapped or "unmapped",
                "overlap": "overlap" if mapped else "amv_extension",
            }
        )

    metric_rows: list[dict[str, Any]] = []
    for metric, config in sorted(metric_map.items()):
        metric_rows.append(
            {
                "metric": metric,
                "classification": str(config.get("classification", "")),
                "alyassi_metric": str(config.get("alyassi_metric", "n/a")),
                "rationale": str(config.get("rationale", "")),
            }
        )

    payload = {
        "schema_version": "benchmark-comparability-summary.v1",
        "campaign_id": campaign_id,
        "generated_at_utc": generated_at_utc,
        "mapping_path": _repo_relative(mapping_path),
        "mapping_version": str(mapping.get("mapping_version")),
        "mapping_hash": _hash_payload(mapping),
        "coverage_overlap_rows": overlap_rows,
        "amv_specific_extensions": list(mapping.get("amv_specific_extensions", [])),
        "metric_comparability_rows": metric_rows,
    }
    return payload, mapping_path
