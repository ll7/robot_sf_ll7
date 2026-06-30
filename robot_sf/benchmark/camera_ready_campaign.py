"""Camera-ready benchmark campaign orchestration.

This module provides a config-driven workflow to run a planner matrix over a
scenario manifest, generate campaign-level reports, and export a publication
bundle for archival/release pipelines.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

import robot_sf.benchmark.camera_ready._route_clearance as _route_clearance_module
from robot_sf.benchmark.aggregate import compute_aggregates_with_ci, read_jsonl
from robot_sf.benchmark.artifact_publication import export_publication_bundle
from robot_sf.benchmark.camera_ready._artifacts import (  # noqa: F401 - re-exported for back-compat
    _escape_markdown_cell,
    _markdown_rows_from_mapping_rows,
    _write_actuation_envelope_artifacts,
    _write_amv_coverage_artifacts,
    _write_comparability_artifacts,
    _write_csv,
    _write_json,
    _write_markdown_table,
    _write_matrix_summary_artifacts,
    _write_seed_episode_rows_artifact,
    _write_seed_variability_artifacts,
    _write_snqi_diagnostics_artifacts,
    _write_statistical_sufficiency_artifact,
    _write_table_artifacts,
)
from robot_sf.benchmark.camera_ready._config import (  # noqa: F401 - re-exported for back-compat
    _AMV_COVERAGE_ENFORCEMENT,
    _PAPER_KINEMATICS_BY_PROFILE,
    _PLANNER_GROUPS,
    _SNQI_CONTRACT_ENFORCEMENT,
    _apply_scenario_amv_overrides,
    _apply_scenario_horizon_schedule,
    _campaign_scenario_id,
    _filter_scenario_candidates,
    _load_campaign_scenarios,
    _load_scenario_horizon_schedule,
    _load_seed_sets,
    _normalize_kinematics_matrix,
    _normalize_observation_mode,
    _optional_synthetic_actuation_profile_mapping,
    _resolve_seed_override,
    _resolved_seed_inventory,
    _sanitize_name,
    _scenario_horizon_summary,
    _scenario_name_key,
    _scenario_with_kinematics,
    _validate_scenario_amv_override_keys,
)
from robot_sf.benchmark.camera_ready._preflight import (
    _build_preflight_preview_payload,
    _build_preflight_validate_payload,
    _latency_stress_metadata,
    _synthetic_actuation_metadata,
)
from robot_sf.benchmark.camera_ready._reporting import (  # noqa: F401 - re-exported for back-compat
    _REPORT_METRICS,
    _build_breakdown_rows,
    _build_scenario_amv_lookup,
    _episode_metric_ci,
    _episode_metric_mean,
    _metric_ci,
    _metric_mean,
    _normalized_algorithm_metadata_contract,
    _planner_report_row,
    _safe_float,
    _scenario_family,
    _strict_vs_fallback_comparisons,
)
from robot_sf.benchmark.camera_ready._route_clearance import (  # noqa: F401 - re-exported for back-compat
    _ROUTE_CLEARANCE_CERTIFICATION_STATUSES,
    _ROUTE_CLEARANCE_FAIL_MARGIN_M,
    _ROUTE_CLEARANCE_WARN_THRESHOLD_M,
    RouteClearanceError,
    _assert_route_clearance_feasible,
    _load_route_clearance_certifications,
    _map_route_clearance_center_min_m,
    _resolve_map_path_for_scenario,
    _route_clearance_warning_summary,
    _scenario_robot_radius_m,
    _scenario_route_lines,
    _valid_obstacle_polygons,
    _valid_route_lines,
)
from robot_sf.benchmark.camera_ready._route_clearance import (
    _build_route_clearance_warnings as _build_route_clearance_warnings_impl,
)
from robot_sf.benchmark.camera_ready._run_state import (  # noqa: F401 - re-exported for back-compat
    _campaign_id,
    _campaign_success_counters,
    _git_context,
    _resolve_campaign_id,
    _resolve_execution_mode,
    _resolve_observation_noise,
    _resolve_path,
    _sanitize_git_remote,
)
from robot_sf.benchmark.camera_ready._summaries import (  # noqa: F401 - re-exported for back-compat
    _ACTUATION_REPORT_METRICS,
    _SEED_VARIABILITY_METRICS,
    _build_actuation_envelope_summary,
    _build_amv_coverage_summary,
    _build_comparability_summary,
    _build_matrix_summary_rows,
    _build_seed_variability_payload,
    _build_statistical_sufficiency_payload,
    _extract_amv_taxonomy,
    _load_comparability_mapping,
    _scenario_family_from_scenario,
    _validate_family_map,
    _validate_metric_map,
    _validate_planner_key_map,
)
from robot_sf.benchmark.camera_ready._util import (  # noqa: F401 - re-exported for back-compat
    _hash_payload,
    _jsonable,
    _jsonable_repo_relative,
    _kinematics_matrix_or_default,
    _repo_relative,
    _sanitize_csv_cell,
    _sha256_file,
    _sha256_payload,
    _utc_now,
)
from robot_sf.benchmark.camera_ready_campaign_config import (  # re-exported for back-compat
    _AMV_DIMENSIONS,
    DEFAULT_SEED_SETS_PATH,
    AmvProfileConfig,
    CampaignConfig,
    PlannerSpec,
    ScenarioCandidateSelection,
    SeedPolicy,
    SnqiContractConfig,
)
from robot_sf.benchmark.fallback_policy import (
    availability_payload,
    classify_planner_row_status,
    summarize_benchmark_availability,
    summarize_campaign_outcome,
    summarize_campaign_status_axes,
)
from robot_sf.benchmark.latency_stress import (
    load_latency_stress_profile,
    not_available_latency_metrics,
    validate_latency_stress_profile,
)
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)
from robot_sf.benchmark.orca_preflight import check_orca_rvo2_preflight
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.seed_variance import (
    build_seed_episode_rows,
)
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
from robot_sf.benchmark.synthetic_actuation import (
    SYNTHETIC_ACTUATION_CLAIM_SCOPE,
    SyntheticActuationProfile,
    validate_actuation_profile_claim_boundary,
    validate_synthetic_actuation_profile,
)
from robot_sf.benchmark.utils import (
    _config_hash,
    load_optional_json,
)
from robot_sf.common.artifact_paths import (
    ensure_canonical_tree,
    get_artifact_category_path,
    get_repository_root,
)
from robot_sf.nav.svg_map_parser import convert_map

CAMPAIGN_SCHEMA_VERSION = "benchmark-camera-ready-campaign.v1"
DEFAULT_EPISODE_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/episode.schema.v1.json")
_normalized_kinematics_matrix = _kinematics_matrix_or_default


def _build_route_clearance_warnings(
    scenarios: list[dict[str, Any]],
    *,
    certifications: dict[str, dict[str, Any]] | None = None,
    margin_warn_threshold_m: float = _ROUTE_CLEARANCE_WARN_THRESHOLD_M,
) -> list[dict[str, Any]]:
    """Build route-clearance warnings via the extracted helper.

    ``convert_map`` remains a module global for compatibility with tests and
    downstream callers that monkeypatch the old ``camera_ready_campaign`` import
    surface.

    Returns:
        List of warning dictionaries for scenarios with low route-obstacle clearance.
    """
    original_convert_map = _route_clearance_module.convert_map
    _route_clearance_module.convert_map = convert_map
    try:
        return _build_route_clearance_warnings_impl(
            scenarios,
            certifications=certifications,
            margin_warn_threshold_m=margin_warn_threshold_m,
        )
    finally:
        _route_clearance_module.convert_map = original_convert_map


def _validate_campaign_config(cfg: CampaignConfig) -> None:  # noqa: C901, PLR0912, PLR0915
    """Validate campaign-level invariants after config parsing."""
    if cfg.scenario_horizons_path is not None and not cfg.scenario_horizons_path.is_file():
        raise FileNotFoundError(
            f"Scenario horizon schedule not found: {cfg.scenario_horizons_path}"
        )
    if (
        cfg.route_clearance_certifications_path is not None
        and not cfg.route_clearance_certifications_path.is_file()
    ):
        raise FileNotFoundError(
            "Route-clearance certification file not found: "
            f"{cfg.route_clearance_certifications_path}"
        )
    if cfg.scenario_horizons_path is not None:
        if cfg.horizon is not None:
            raise ValueError("scenario_horizons cannot be combined with fixed horizon")
        planners_with_horizon_override = [
            planner.key
            for planner in cfg.planners
            if planner.enabled and planner.horizon_override is not None
        ]
        if planners_with_horizon_override:
            names = ", ".join(sorted(planners_with_horizon_override))
            raise ValueError(
                f"scenario_horizons cannot be combined with per-planner horizon overrides: {names}"
            )
    enforcement = cfg.amv_profile.coverage_enforcement
    if enforcement not in _AMV_COVERAGE_ENFORCEMENT:
        known = ", ".join(sorted(_AMV_COVERAGE_ENFORCEMENT))
        raise ValueError(f"Unsupported amv_profile.coverage_enforcement '{enforcement}'. {known}")
    for key, values in cfg.amv_profile.required_dimensions.items():
        for value in values:
            if not str(value).strip():
                raise ValueError(f"AMV required dimension '{key}' contains an empty value")
    if cfg.scenario_candidates.names and any(
        not str(name).strip() for name in cfg.scenario_candidates.names
    ):
        raise ValueError("scenario_candidates must not contain empty names")
    for scenario_name, amv_override in cfg.scenario_amv_overrides.items():
        if not str(scenario_name).strip():
            raise ValueError("scenario_amv_overrides keys must be non-empty scenario names")
        if not amv_override:
            raise ValueError(
                "scenario_amv_overrides entries must include at least one AMV taxonomy dimension"
            )
    if cfg.synthetic_actuation_profile is not None:
        if cfg.synthetic_actuation_profile.claim_scope == SYNTHETIC_ACTUATION_CLAIM_SCOPE:
            validate_synthetic_actuation_profile(cfg.synthetic_actuation_profile)
        if cfg.paper_facing:
            raise ValueError("synthetic_actuation_profile requires paper_facing=false")
        normalized_kinematics = tuple(str(value).strip().lower() for value in cfg.kinematics_matrix)
        if normalized_kinematics != ("differential_drive",):
            raise ValueError(
                "synthetic_actuation_profile requires kinematics_matrix=['differential_drive']"
            )
    if cfg.latency_stress_profile is not None:
        validate_latency_stress_profile(cfg.latency_stress_profile)
        if cfg.paper_facing:
            raise ValueError("latency_stress_profile requires paper_facing=false")
        if cfg.latency_stress_profile.action_delay_steps > 0:
            normalized_kinematics = tuple(
                str(value).strip().lower() for value in cfg.kinematics_matrix
            )
            if normalized_kinematics != ("differential_drive",):
                raise ValueError(
                    "latency_stress_profile.action_delay_steps requires "
                    "kinematics_matrix=['differential_drive']"
                )
    if cfg.snqi_contract.enforcement not in _SNQI_CONTRACT_ENFORCEMENT:
        known = ", ".join(sorted(_SNQI_CONTRACT_ENFORCEMENT))
        raise ValueError(
            f"Unsupported snqi_contract.enforcement '{cfg.snqi_contract.enforcement}'. {known}"
        )
    threshold_values = {
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
    }
    for field_name, value in threshold_values.items():
        if not math.isfinite(value):
            raise ValueError(f"snqi_contract.{field_name} must be a finite float")
    if (
        cfg.snqi_contract.rank_alignment_fail_threshold
        > cfg.snqi_contract.rank_alignment_warn_threshold
    ):
        raise ValueError(
            "snqi_contract.rank_alignment_fail_threshold must be <= rank_alignment_warn_threshold"
        )
    if (
        cfg.snqi_contract.outcome_separation_fail_threshold
        > cfg.snqi_contract.outcome_separation_warn_threshold
    ):
        raise ValueError(
            "snqi_contract.outcome_separation_fail_threshold must be <= outcome_separation_warn_threshold"
        )
    if (
        cfg.snqi_contract.max_component_dominance_fail_threshold
        < cfg.snqi_contract.max_component_dominance_warn_threshold
    ):
        raise ValueError(
            "snqi_contract.max_component_dominance_fail_threshold must be >= "
            "max_component_dominance_warn_threshold"
        )

    if cfg.paper_facing:
        if not cfg.paper_profile_version or not str(cfg.paper_profile_version).strip():
            raise ValueError("paper_facing=true requires non-empty paper_profile_version")
        paper_profile = str(cfg.paper_profile_version).strip()
        expected_kinematics = _PAPER_KINEMATICS_BY_PROFILE.get(paper_profile)
        if expected_kinematics is None:
            known_profiles = ", ".join(sorted(_PAPER_KINEMATICS_BY_PROFILE))
            raise ValueError(
                f"Unsupported paper_profile_version '{paper_profile}'. Expected one of: "
                f"{known_profiles}"
            )
        normalized_kinematics = tuple(str(value).strip().lower() for value in cfg.kinematics_matrix)
        if normalized_kinematics != expected_kinematics:
            raise ValueError(
                "paper_facing=true requires kinematics_matrix="
                f"{list(expected_kinematics)!r} for paper_profile_version='{paper_profile}'",
            )
        for planner in cfg.planners:
            if not planner.enabled:
                continue
            if not planner.planner_group_explicit:
                raise ValueError(
                    "paper_facing=true requires explicit planner_group for each enabled planner",
                )
        if cfg.comparability_mapping_path is None:
            raise ValueError("paper_facing=true requires comparability_mapping path")


def load_campaign_config(path: Path) -> CampaignConfig:  # noqa: C901, PLR0912, PLR0915
    """Load and validate a camera-ready benchmark campaign YAML config.

    Returns:
        Parsed campaign configuration dataclass.
    """
    config_path = path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Campaign config must be a mapping: {config_path}")

    name = str(payload.get("name") or config_path.stem)
    matrix_raw = payload.get("scenario_matrix")
    if not isinstance(matrix_raw, str) or not matrix_raw.strip():
        raise ValueError("Campaign config requires a non-empty 'scenario_matrix' string")
    scenario_matrix_path = _resolve_path(matrix_raw, base_dir=config_path.parent)
    if scenario_matrix_path is None:
        raise FileNotFoundError(
            f"Could not resolve scenario_matrix '{matrix_raw}' from config '{config_path}'.",
        )

    planners_raw = payload.get("planners")
    if not isinstance(planners_raw, list) or not planners_raw:
        raise ValueError("Campaign config requires a non-empty 'planners' list")

    planner_specs: list[PlannerSpec] = []
    for entry in planners_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each planners entry must be a mapping")
        key = str(entry.get("key") or entry.get("algo") or "").strip()
        algo = str(entry.get("algo") or "").strip()
        if not key or not algo:
            raise ValueError("Planner entry requires non-empty key and algo")
        planner_group_explicit = "planner_group" in entry
        planner_group = str(entry.get("planner_group", "experimental")).strip().lower()
        if planner_group not in _PLANNER_GROUPS:
            raise ValueError(
                f"Unsupported planner_group '{planner_group}'. Expected one of: core|experimental."
            )
        planner_specs.append(
            PlannerSpec(
                key=key,
                algo=algo,
                human_model_variant=_normalize_observation_mode(
                    entry.get("human_model_variant"),
                    label="Planner entry 'human_model_variant'",
                ),
                human_model_source=_normalize_observation_mode(
                    entry.get("human_model_source"),
                    label="Planner entry 'human_model_source'",
                ),
                benchmark_profile=str(entry.get("benchmark_profile", "baseline-safe")),
                algo_config_path=_resolve_path(
                    entry.get("algo_config"), base_dir=config_path.parent
                ),
                socnav_missing_prereq_policy=str(
                    entry.get("socnav_missing_prereq_policy", "fail-fast"),
                ),
                adapter_impact_eval=bool(entry.get("adapter_impact_eval", False)),
                observation_mode=_normalize_observation_mode(
                    entry.get("observation_mode"),
                    label="Planner entry 'observation_mode'",
                ),
                workers_override=(
                    int(entry["workers"]) if entry.get("workers") is not None else None
                ),
                horizon_override=(
                    int(entry["horizon"]) if entry.get("horizon") is not None else None
                ),
                dt_override=(float(entry["dt"]) if entry.get("dt") is not None else None),
                enabled=bool(entry.get("enabled", True)),
                planner_group=planner_group,
                planner_group_explicit=planner_group_explicit,
            ),
        )

    seed_policy_raw = (
        payload.get("seed_policy") if isinstance(payload.get("seed_policy"), dict) else {}
    )
    mode = str(seed_policy_raw.get("mode", "scenario-default"))
    seed_set = seed_policy_raw.get("seed_set")
    seeds = seed_policy_raw.get("seeds") if isinstance(seed_policy_raw.get("seeds"), list) else []
    seed_sets_path_raw = seed_policy_raw.get("seed_sets_path")
    seed_sets_path = (
        _resolve_path(str(seed_sets_path_raw), base_dir=config_path.parent)
        if isinstance(seed_sets_path_raw, str) and seed_sets_path_raw.strip()
        else None
    )
    if seed_sets_path is None:
        seed_sets_path = (get_repository_root() / DEFAULT_SEED_SETS_PATH).resolve()

    snqi_weights = _resolve_path(payload.get("snqi_weights"), base_dir=config_path.parent)
    snqi_baseline = _resolve_path(payload.get("snqi_baseline"), base_dir=config_path.parent)
    scenario_horizons = _resolve_path(
        payload.get("scenario_horizons"),
        base_dir=config_path.parent,
    )
    route_clearance_certifications_path = _resolve_path(
        payload.get("route_clearance_certifications"),
        base_dir=config_path.parent,
    )
    comparability_mapping_path = _resolve_path(
        payload.get("comparability_mapping"),
        base_dir=config_path.parent,
    )
    if comparability_mapping_path is None:
        default_mapping_path = (
            get_repository_root() / "configs/benchmarks/alyassi_comparability_map_v1.yaml"
        ).resolve()
        if default_mapping_path.exists():
            comparability_mapping_path = default_mapping_path

    amv_raw = payload.get("amv_profile") if isinstance(payload.get("amv_profile"), dict) else {}
    snqi_contract_raw = (
        payload.get("snqi_contract") if isinstance(payload.get("snqi_contract"), dict) else {}
    )
    required_raw = (
        amv_raw.get("required_dimensions")
        if isinstance(amv_raw.get("required_dimensions"), dict)
        else {}
    )
    for key in required_raw:
        if key not in _AMV_DIMENSIONS:
            known = ", ".join(_AMV_DIMENSIONS)
            raise ValueError(
                f"Unsupported amv_profile.required_dimensions key '{key}'. Expected: {known}"
            )
    required_dimensions: dict[str, tuple[str, ...]] = {}
    for dimension in _AMV_DIMENSIONS:
        values = required_raw.get(dimension, [])
        if isinstance(values, (str, int, float)):
            normalized = (str(values).strip(),) if str(values).strip() else ()
        elif isinstance(values, list):
            normalized = tuple(str(value).strip() for value in values if str(value).strip())
        else:
            normalized = ()
        required_dimensions[dimension] = normalized
    scenario_candidates_raw = payload.get("scenario_candidates", [])
    if isinstance(scenario_candidates_raw, (str, int, float)):
        scenario_candidates = (str(scenario_candidates_raw).strip(),)
    elif isinstance(scenario_candidates_raw, list):
        if any(not isinstance(value, (str, int, float)) for value in scenario_candidates_raw):
            raise TypeError("scenario_candidates entries must be scalar names")
        scenario_candidates = tuple(
            str(value).strip() for value in scenario_candidates_raw if str(value).strip()
        )
    elif "scenario_candidates" in payload:
        raise TypeError("scenario_candidates must be a scalar name or list of scalar names")
    else:
        scenario_candidates = ()
    scenario_amv_overrides_raw = payload.get("scenario_amv_overrides")
    if scenario_amv_overrides_raw is None:
        scenario_amv_overrides: dict[str, dict[str, str]] = {}
    elif not isinstance(scenario_amv_overrides_raw, dict):
        raise TypeError(
            "scenario_amv_overrides must be a mapping of scenario names to AMV mappings"
        )
    else:
        scenario_amv_overrides = {}
        for raw_scenario_name, raw_taxonomy in scenario_amv_overrides_raw.items():
            scenario_name = str(raw_scenario_name).strip()
            if not scenario_name:
                raise ValueError("scenario_amv_overrides keys must be non-empty scenario names")
            if not isinstance(raw_taxonomy, dict):
                raise TypeError(
                    "scenario_amv_overrides entries must be mappings keyed by AMV dimension"
                )
            taxonomy: dict[str, str] = {}
            for raw_dimension, raw_value in raw_taxonomy.items():
                dimension = str(raw_dimension).strip()
                if dimension not in _AMV_DIMENSIONS:
                    known = ", ".join(_AMV_DIMENSIONS)
                    raise ValueError(
                        f"Unsupported scenario_amv_overrides dimension '{dimension}'. "
                        f"Expected: {known}"
                    )
                if raw_value is None:
                    raise ValueError(
                        "scenario_amv_overrides values must be non-empty strings when provided"
                    )
                value = str(raw_value).strip()
                if not value:
                    raise ValueError(
                        "scenario_amv_overrides values must be non-empty strings when provided"
                    )
                taxonomy[dimension] = value
            if not taxonomy:
                raise ValueError(
                    "scenario_amv_overrides entries must include at least one AMV taxonomy dimension"
                )
            scenario_amv_overrides[scenario_name] = taxonomy
    synthetic_actuation_raw = payload.get("synthetic_actuation_profile")
    if synthetic_actuation_raw is not None and not isinstance(synthetic_actuation_raw, dict):
        raise TypeError("synthetic_actuation_profile must be a mapping when provided")
    if synthetic_actuation_raw is not None:
        validate_actuation_profile_claim_boundary(synthetic_actuation_raw)
    latency_stress_raw = payload.get("latency_stress_profile")
    kinematics_matrix = _normalize_kinematics_matrix(
        payload.get("kinematics_matrix", ["differential_drive"])
    )

    cfg = CampaignConfig(
        name=name,
        scenario_matrix_path=scenario_matrix_path,
        planners=tuple(planner_specs),
        scenario_candidates=ScenarioCandidateSelection(names=scenario_candidates),
        scenario_amv_overrides=scenario_amv_overrides,
        scenario_horizons_path=scenario_horizons,
        seed_policy=SeedPolicy(
            mode=mode,
            seed_set=str(seed_set) if seed_set is not None else None,
            seeds=tuple(int(seed) for seed in seeds),
            seed_sets_path=seed_sets_path,
        ),
        workers=int(payload.get("workers", 1)),
        horizon=(int(payload["horizon"]) if payload.get("horizon") is not None else None),
        dt=(float(payload["dt"]) if payload.get("dt") is not None else None),
        record_forces=bool(payload.get("record_forces", True)),
        resume=bool(payload.get("resume", True)),
        bootstrap_samples=int(payload.get("bootstrap_samples", 400)),
        bootstrap_confidence=float(payload.get("bootstrap_confidence", 0.95)),
        bootstrap_seed=int(payload.get("bootstrap_seed", 123)),
        snqi_weights_path=snqi_weights,
        snqi_baseline_path=snqi_baseline,
        stop_on_failure=bool(payload.get("stop_on_failure", False)),
        export_publication_bundle=bool(payload.get("export_publication_bundle", True)),
        include_videos_in_publication=bool(payload.get("include_videos_in_publication", False)),
        overwrite_publication_bundle=bool(payload.get("overwrite_publication_bundle", True)),
        repository_url=str(payload.get("repository_url", "https://github.com/ll7/robot_sf_ll7")),
        release_tag=str(payload.get("release_tag", "{release_tag}")),
        doi=str(payload.get("doi", "10.5281/zenodo.<record-id>")),
        paper_interpretation_profile=str(
            payload.get("paper_interpretation_profile", "baseline-ready-core")
        ),
        preview_scenario_limit=int(payload.get("preview_scenario_limit", 100)),
        kinematics_matrix=kinematics_matrix,
        holonomic_command_mode=str(payload.get("holonomic_command_mode", "vx_vy")).strip(),
        observation_mode=_normalize_observation_mode(
            payload.get("observation_mode"),
            label="Campaign 'observation_mode'",
        ),
        paper_facing=bool(payload.get("paper_facing", False)),
        paper_profile_version=(
            str(payload.get("paper_profile_version")).strip()
            if payload.get("paper_profile_version") is not None
            else None
        ),
        amv_profile=AmvProfileConfig(
            name=str(amv_raw.get("name", "amv-paper-v1")).strip() or "amv-paper-v1",
            contract_version=str(amv_raw.get("contract_version", "1")).strip() or "1",
            coverage_enforcement=(
                str(amv_raw.get("coverage_enforcement", "warn")).strip().lower() or "warn"
            ),
            required_dimensions=required_dimensions,
        ),
        synthetic_actuation_profile=(
            SyntheticActuationProfile(
                name=str(synthetic_actuation_raw.get("name", "")).strip(),
                profile_version=(
                    str(synthetic_actuation_raw.get("profile_version", "v0")).strip() or "v0"
                ),
                claim_scope=(
                    str(synthetic_actuation_raw.get("claim_scope", "synthetic-only")).strip()
                    or "synthetic-only"
                ),
                claim_boundary=str(synthetic_actuation_raw.get("claim_boundary", "")).strip(),
                max_linear_accel_m_s2=float(
                    synthetic_actuation_raw.get("max_linear_accel_m_s2", 0.0)
                ),
                max_linear_decel_m_s2=float(
                    synthetic_actuation_raw.get("max_linear_decel_m_s2", 0.0)
                ),
                max_yaw_rate_rad_s=float(synthetic_actuation_raw.get("max_yaw_rate_rad_s", 0.0)),
                max_angular_accel_rad_s2=float(
                    synthetic_actuation_raw.get("max_angular_accel_rad_s2", 0.0)
                ),
                latency_mode=(
                    str(synthetic_actuation_raw.get("latency_mode", "zero-step-delay"))
                    .strip()
                    .lower()
                ),
                update_mode=(
                    str(synthetic_actuation_raw.get("update_mode", "10hz-matched")).strip().lower()
                ),
                variability_distribution=_optional_synthetic_actuation_profile_mapping(
                    synthetic_actuation_raw,
                    "variability_distribution",
                ),
                variability_sample=_optional_synthetic_actuation_profile_mapping(
                    synthetic_actuation_raw,
                    "variability_sample",
                ),
                provenance=_optional_synthetic_actuation_profile_mapping(
                    synthetic_actuation_raw,
                    "provenance",
                ),
            )
            if synthetic_actuation_raw is not None
            else None
        ),
        latency_stress_profile=load_latency_stress_profile(latency_stress_raw),
        comparability_mapping_path=comparability_mapping_path,
        route_clearance_certifications_path=route_clearance_certifications_path,
        snqi_contract=SnqiContractConfig(
            enabled=bool(snqi_contract_raw.get("enabled", True)),
            enforcement=(
                str(snqi_contract_raw.get("enforcement", "warn")).strip().lower() or "warn"
            ),
            rank_alignment_warn_threshold=float(
                snqi_contract_raw.get("rank_alignment_warn_threshold", 0.5)
            ),
            rank_alignment_fail_threshold=float(
                snqi_contract_raw.get("rank_alignment_fail_threshold", 0.3)
            ),
            outcome_separation_warn_threshold=float(
                snqi_contract_raw.get("outcome_separation_warn_threshold", 0.05)
            ),
            outcome_separation_fail_threshold=float(
                snqi_contract_raw.get("outcome_separation_fail_threshold", 0.0)
            ),
            max_component_dominance_warn_threshold=float(
                snqi_contract_raw.get("max_component_dominance_warn_threshold", 0.24)
            ),
            max_component_dominance_fail_threshold=float(
                snqi_contract_raw.get("max_component_dominance_fail_threshold", 0.27)
            ),
            calibration_seed=int(snqi_contract_raw.get("calibration_seed", 123)),
            calibration_trials=int(snqi_contract_raw.get("calibration_trials", 3000)),
        ),
        observation_noise=_resolve_observation_noise(
            payload.get("observation_noise"),
            base_dir=config_path.parent,
        ),
    )
    _validate_campaign_config(cfg)
    return cfg


def prepare_campaign_preflight(
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    invoked_command: str | None = None,
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
    _validate_campaign_config(cfg)
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
    route_clearance_warnings = _build_route_clearance_warnings(
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


def write_campaign_report(  # noqa: C901, PLR0912, PLR0915
    path: Path, payload: dict[str, Any]
) -> None:
    """Write a human-readable campaign report in Markdown."""
    campaign = payload.get("campaign", {})
    rows = payload.get("planner_rows", [])
    warnings = payload.get("warnings", [])
    accepted_unavailable_rows = [
        row
        for row in rows
        if classify_planner_row_status(str(row.get("status", ""))) == "accepted_unavailable"
    ]
    unexpected_failed_rows = [
        row
        for row in rows
        if classify_planner_row_status(str(row.get("status", ""))) == "unexpected_failure"
    ]

    lines = [
        "# Camera-Ready Benchmark Campaign Report",
        "",
        f"- Campaign ID: `{campaign.get('campaign_id', 'unknown')}`",
        f"- Name: `{campaign.get('name', 'unknown')}`",
        f"- Created (UTC): `{campaign.get('created_at_utc', 'unknown')}`",
        f"- Scenario matrix: `{campaign.get('scenario_matrix', 'unknown')}`",
        f"- Scenario matrix hash: `{campaign.get('scenario_matrix_hash', 'unknown')}`",
        f"- Git commit: `{campaign.get('git_hash', 'unknown')}`",
        f"- Runtime sec: `{campaign.get('runtime_sec', 0.0)}`",
        f"- Episodes/sec: `{campaign.get('episodes_per_second', 0.0)}`",
        f"- Campaign status: `{campaign.get('status', 'unknown')}`",
        f"- Campaign execution status: `{campaign.get('campaign_execution_status', 'unknown')}`",
        f"- Evidence status: `{campaign.get('evidence_status', 'unknown')}`",
        f"- Status reason: `{campaign.get('status_reason', 'unknown')}`",
        f"- Benchmark success: `{campaign.get('benchmark_success', False)}`",
        f"- Successful rows: `{campaign.get('successful_runs', 0)}` / `{campaign.get('total_runs', 0)}`",
        f"- Accepted unavailable/excluded rows: `{campaign.get('accepted_unavailable_runs', 0)}`",
        f"- Unexpected failed rows: `{campaign.get('unexpected_failed_runs', 0)}`",
        (f"- Row status summary: `{campaign.get('row_status_summary', {})}`"),
        f"- Interpretation profile: `{campaign.get('paper_interpretation_profile', 'unknown')}`",
        f"- Command: `{campaign.get('invoked_command', 'unknown')}`",
        "",
        "## Planner Summary",
        "",
    ]

    if rows:
        lines.extend(
            [
                "| planner | algo | planner group | kinematics | status | started (UTC) | runtime (s) | episodes | eps/s | success | collisions | snqi | proj_rate | infeasible_rate |",
                "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ],
        )
        for row in rows:
            lines.append(
                "| "
                f"{_escape_markdown_cell(row.get('planner_key'))} | "
                f"{_escape_markdown_cell(row.get('algo'))} | "
                f"{_escape_markdown_cell(row.get('planner_group'))} | "
                f"{_escape_markdown_cell(row.get('kinematics'))} | "
                f"{_escape_markdown_cell(row.get('status'))} | "
                f"{_escape_markdown_cell(row.get('started_at_utc'))} | "
                f"{_escape_markdown_cell(row.get('runtime_sec'))} | "
                f"{_escape_markdown_cell(row.get('episodes'))} | "
                f"{_escape_markdown_cell(row.get('episodes_per_second'))} | "
                f"{_escape_markdown_cell(row.get('success_mean'))} | "
                f"{_escape_markdown_cell(row.get('collisions_mean'))} | "
                f"{_escape_markdown_cell(row.get('snqi_mean'))} | "
                f"{_escape_markdown_cell(row.get('projection_rate'))} | "
                f"{_escape_markdown_cell(row.get('infeasible_rate'))} |",
            )
    else:
        lines.append("No planner rows were produced.")
    fallback_rows = [
        row for row in rows if str(row.get("readiness_status", "")) in {"fallback", "degraded"}
    ]
    lines.extend(["", "## Readiness & Degraded/Fallback Status", ""])
    if rows:
        lines.append(
            "| planner | planner group | execution mode | execution detail | planner cmd | benchmark cmd | projection policy | readiness status | tier | preflight | learned contract | run status |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                "| "
                f"{_escape_markdown_cell(row.get('planner_key'))} | "
                f"{_escape_markdown_cell(row.get('planner_group'))} | "
                f"{_escape_markdown_cell(row.get('execution_mode'))} | "
                f"{_escape_markdown_cell(row.get('execution_detail'))} | "
                f"{_escape_markdown_cell(row.get('planner_command_space'))} | "
                f"{_escape_markdown_cell(row.get('benchmark_command_space'))} | "
                f"{_escape_markdown_cell(row.get('projection_policy'))} | "
                f"{_escape_markdown_cell(row.get('readiness_status'))} | "
                f"{_escape_markdown_cell(row.get('readiness_tier'))} | "
                f"{_escape_markdown_cell(row.get('preflight_status'))} | "
                f"{_escape_markdown_cell(row.get('learned_policy_contract_status'))} | "
                f"{_escape_markdown_cell(row.get('status'))} |"
            )
    if fallback_rows:
        lines.append("")
        lines.append("Planners in fallback/degraded mode:")
        for row in fallback_rows:
            lines.append(
                f"- `{row.get('planner_key')}`: readiness={row.get('readiness_status')}, "
                f"preflight={row.get('preflight_status')}, tier={row.get('readiness_tier')}"
            )
    else:
        lines.append("")
        lines.append("- No fallback/degraded planners detected.")

    lines.extend(["", "## SocNav Strict-vs-Fallback Disclosure", ""])
    if rows:
        lines.append(
            "| planner | algo | planner group | prereq policy | preflight status | readiness status |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in rows:
            lines.append(
                "| "
                f"{_escape_markdown_cell(row.get('planner_key'))} | "
                f"{_escape_markdown_cell(row.get('algo'))} | "
                f"{_escape_markdown_cell(row.get('planner_group'))} | "
                f"{_escape_markdown_cell(row.get('socnav_prereq_policy'))} | "
                f"{_escape_markdown_cell(row.get('preflight_status'))} | "
                f"{_escape_markdown_cell(row.get('readiness_status'))} |"
            )
        comparisons = _strict_vs_fallback_comparisons(rows)
        if comparisons:
            lines.append("")
            lines.append("Strict-vs-fallback comparisons (where both modes are present):")
            for line in comparisons:
                lines.append(f"- {line}")
        else:
            lines.append("")
            lines.append(
                "- No within-campaign strict-vs-fallback pair available for direct comparison."
            )

    scenario_path = (payload.get("artifacts") or {}).get("scenario_breakdown_csv")
    family_path = (payload.get("artifacts") or {}).get("scenario_family_breakdown_csv")
    if isinstance(scenario_path, str) or isinstance(family_path, str):
        lines.extend(["", "## Scenario Diagnostics", ""])
        if isinstance(scenario_path, str):
            lines.append(f"- Per-scenario breakdown: `{scenario_path}`")
        if isinstance(family_path, str):
            lines.append(f"- Per-family breakdown: `{family_path}`")
    parity_path = (payload.get("artifacts") or {}).get("kinematics_parity_csv")
    skipped_path = (payload.get("artifacts") or {}).get("kinematics_skipped_combinations_csv")
    if isinstance(parity_path, str) or isinstance(skipped_path, str):
        lines.extend(["", "## Kinematics Parity", ""])
        if isinstance(parity_path, str):
            lines.append(f"- Planner x kinematics parity table: `{parity_path}`")
        if isinstance(skipped_path, str):
            lines.append(f"- Skipped planner/kinematics combinations: `{skipped_path}`")
    amv_json = (payload.get("artifacts") or {}).get("amv_coverage_json")
    amv_md = (payload.get("artifacts") or {}).get("amv_coverage_md")
    if isinstance(amv_json, str) or isinstance(amv_md, str):
        lines.extend(["", "## AMV Coverage Contract", ""])
        if isinstance(amv_json, str):
            lines.append(f"- Coverage JSON: `{amv_json}`")
        if isinstance(amv_md, str):
            lines.append(f"- Coverage Markdown: `{amv_md}`")
        lines.append(
            f"- Coverage status: `{campaign.get('amv_coverage_status', 'unknown')}` "
            f"(enforcement: `{campaign.get('amv_coverage_enforcement', 'warn')}`)"
        )
    comparability_json = (payload.get("artifacts") or {}).get("comparability_json")
    comparability_md = (payload.get("artifacts") or {}).get("comparability_md")
    if isinstance(comparability_json, str) or isinstance(comparability_md, str):
        lines.extend(["", "## Alyassi Comparability", ""])
        if isinstance(comparability_json, str):
            lines.append(f"- Comparability JSON: `{comparability_json}`")
        if isinstance(comparability_md, str):
            lines.append(f"- Comparability Markdown: `{comparability_md}`")
        lines.append(
            f"- Mapping version: `{campaign.get('comparability_mapping_version', 'unknown')}`"
        )
    snqi_diag_json = (payload.get("artifacts") or {}).get("snqi_diagnostics_json")
    snqi_diag_md = (payload.get("artifacts") or {}).get("snqi_diagnostics_md")
    snqi_sensitivity = (payload.get("artifacts") or {}).get("snqi_sensitivity_csv")
    if isinstance(snqi_diag_json, str) or isinstance(snqi_diag_md, str):
        lines.extend(["", "## SNQI Contract", ""])
        lines.append(f"- Contract status: `{campaign.get('snqi_contract_status', 'unknown')}`")
        lines.append(
            f"- Rank alignment (Spearman): `{campaign.get('snqi_contract_rank_alignment_spearman', 'nan')}`"
        )
        lines.append(
            f"- Outcome separation: `{campaign.get('snqi_contract_outcome_separation', 'nan')}`"
        )
        lines.append(
            f"- Positioning recommendation: `{campaign.get('snqi_positioning_recommendation', 'unknown')}`"
        )
        lines.append(f"- Weights version: `{campaign.get('snqi_weights_version', 'unknown')}`")
        lines.append(f"- Baseline version: `{campaign.get('snqi_baseline_version', 'unknown')}`")
        if isinstance(snqi_diag_json, str):
            lines.append(f"- Diagnostics JSON: `{snqi_diag_json}`")
        if isinstance(snqi_diag_md, str):
            lines.append(f"- Diagnostics Markdown: `{snqi_diag_md}`")
        if isinstance(snqi_sensitivity, str):
            lines.append(f"- Sensitivity CSV: `{snqi_sensitivity}`")

    lines.extend(["", "## Accepted Unavailable/Excluded Planners", ""])
    if accepted_unavailable_rows:
        lines.append("| planner | status | availability reason |")
        lines.append("|---|---|---|")
        for row in accepted_unavailable_rows:
            lines.append(
                "| "
                f"{_escape_markdown_cell(row.get('planner_key'))} | "
                f"{_escape_markdown_cell(row.get('status'))} | "
                f"{_escape_markdown_cell(row.get('availability_reason') or row.get('most_likely_failure_reason') or 'unspecified')} |"
            )
    else:
        lines.append("- No accepted unavailable/excluded planners.")

    lines.extend(["", "## Unexpected Failed/Partial Planners", ""])
    if unexpected_failed_rows:
        lines.append("| planner | status | most likely reason |")
        lines.append("|---|---|---|")
        for row in unexpected_failed_rows:
            lines.append(
                "| "
                f"{_escape_markdown_cell(row.get('planner_key'))} | "
                f"{_escape_markdown_cell(row.get('status'))} | "
                f"{_escape_markdown_cell(row.get('most_likely_failure_reason') or row.get('availability_reason') or 'unspecified')} |"
            )
    else:
        lines.append("- No unexpected failed/partial planners.")

    lines.extend(["", "## Campaign Warnings", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No campaign-level warnings.")

    publication = payload.get("publication_bundle")
    if isinstance(publication, dict):
        lines.extend(
            [
                "",
                "## Publication Bundle",
                "",
                f"- Bundle dir: `{publication.get('bundle_dir', 'unknown')}`",
                f"- Archive: `{publication.get('archive_path', 'unknown')}`",
                f"- Manifest: `{publication.get('manifest_path', 'unknown')}`",
                f"- Checksums: `{publication.get('checksums_path', 'unknown')}`",
            ],
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(  # noqa: C901, PLR0912, PLR0915
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    skip_publication_bundle: bool = False,
    invoked_command: str | None = None,
) -> dict[str, Any]:
    """Execute a camera-ready planner campaign and emit campaign artifacts.

    Returns:
        Campaign execution summary with output paths and high-level counters.

    Raises:
        OrcaRvo2PreflightError: When enabled ORCA-dependent planners require ``rvo2`` but it is
            not importable.
        RouteClearanceError: When any scenario route centerline lies closer to a static obstacle
            than the robot radius, making the route geometrically impossible to follow without
            collision.
    """
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
        and cfg.snqi_contract.enforcement == "error"
        and contract_eval.status == "fail"
    )
    if snqi_hard_fail:
        warnings.append(
            "SNQI contract status=fail with snqi_contract.enforcement=error; campaign marked with hard contract warning."
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
            "SNQI contract failed with enforcement=error; "
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


__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "AmvProfileConfig",
    "CampaignConfig",
    "PlannerSpec",
    "RouteClearanceError",
    "SeedPolicy",
    "SnqiContractConfig",
    "load_campaign_config",
    "prepare_campaign_preflight",
    "run_campaign",
    "write_campaign_report",
]
