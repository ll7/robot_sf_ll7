"""Camera-ready benchmark campaign orchestration.

This module provides a config-driven workflow to run a planner matrix over a
scenario manifest, generate campaign-level reports, and export a publication
bundle for archival/release pipelines.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

import robot_sf.benchmark.camera_ready._route_clearance as _route_clearance_module
from robot_sf.benchmark.aggregate import compute_aggregates_with_ci
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
from robot_sf.benchmark.camera_ready._preflight import (  # noqa: F401 - re-exported back-compat
    _build_preflight_preview_payload,
    _build_preflight_validate_payload,
    _latency_stress_metadata,
    _synthetic_actuation_metadata,
)
from robot_sf.benchmark.camera_ready._preflight import (
    prepare_campaign_preflight as _prepare_campaign_preflight_impl,
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
    write_campaign_report,
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
from robot_sf.benchmark.camera_ready.campaign import run_campaign as _run_campaign_impl
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
from robot_sf.benchmark.latency_stress import (
    load_latency_stress_profile,
    validate_latency_stress_profile,
)
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.synthetic_actuation import (
    SYNTHETIC_ACTUATION_CLAIM_SCOPE,
    SyntheticActuationProfile,
    validate_actuation_profile_claim_boundary,
    validate_synthetic_actuation_profile,
)
from robot_sf.benchmark.utils import (  # noqa: F401 - re-exported for back-compat / test patch surface
    load_optional_json,
)
from robot_sf.common.artifact_paths import get_repository_root
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
    """Prepare campaign preflight artifacts via the extracted preflight module.

    Returns:
        Paths and metadata required by preflight-only workflows and full runs.
    """
    return _prepare_campaign_preflight_impl(
        cfg,
        output_root=output_root,
        label=label,
        campaign_id=campaign_id,
        invoked_command=invoked_command,
        validate_campaign_config=_validate_campaign_config,
        build_route_clearance_warnings=_build_route_clearance_warnings,
    )


def run_campaign(
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    skip_publication_bundle: bool = False,
    invoked_command: str | None = None,
) -> dict[str, Any]:
    """Execute a camera-ready planner campaign via the extracted campaign module.

    The heavy orchestration lives in ``robot_sf.benchmark.camera_ready.campaign``. This facade
    wrapper injects the module-level ``prepare_campaign_preflight``, ``run_batch``,
    ``compute_aggregates_with_ci`` and ``export_publication_bundle`` bindings so existing tests
    that monkeypatch ``robot_sf.benchmark.camera_ready_campaign.<name>`` keep working unchanged.

    Returns:
        Campaign execution summary with output paths and high-level counters.

    Raises:
        OrcaRvo2PreflightError: When enabled ORCA-dependent planners require ``rvo2`` but it is
            not importable.
        RouteClearanceError: When any scenario route centerline lies closer to a static obstacle
            than the robot radius, making the route geometrically impossible to follow without
            collision.
    """
    return _run_campaign_impl(
        cfg,
        output_root=output_root,
        label=label,
        campaign_id=campaign_id,
        skip_publication_bundle=skip_publication_bundle,
        invoked_command=invoked_command,
        prepare_campaign_preflight=prepare_campaign_preflight,
        run_batch=run_batch,
        compute_aggregates_with_ci=compute_aggregates_with_ci,
        export_publication_bundle=export_publication_bundle,
    )


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
