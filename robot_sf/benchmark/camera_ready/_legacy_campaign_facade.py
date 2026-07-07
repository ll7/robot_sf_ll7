"""Legacy camera-ready campaign compatibility facade.

This module preserves the old ``robot_sf.benchmark.camera_ready_campaign`` import
surface while the package-owned implementations live under
``robot_sf.benchmark.camera_ready``. Keep old-module monkeypatch-sensitive
wrappers here until downstream callers have moved to package imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    _validate_campaign_config,
    _validate_scenario_amv_override_keys,
    load_campaign_config,
)
from robot_sf.benchmark.camera_ready._config_types import (  # noqa: F401 - re-exported for back-compat
    _AMV_DIMENSIONS,
    DEFAULT_SEED_SETS_PATH,
    AmvProfileConfig,
    CampaignConfig,
    PlannerSpec,
    ScenarioCandidateSelection,
    SeedPolicy,
    SnqiContractConfig,
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
    build_campaign_credibility_scorecard,
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
from robot_sf.benchmark.camera_ready.campaign import (  # noqa: F401 - re-exported for back-compat
    CAMPAIGN_SCHEMA_VERSION,
    DEFAULT_EPISODE_SCHEMA_PATH,
)
from robot_sf.benchmark.camera_ready.campaign import run_campaign as _run_campaign_impl
from robot_sf.benchmark.runner import run_batch
from robot_sf.benchmark.utils import (  # noqa: F401 - re-exported for back-compat / test patch surface
    load_optional_json,
)
from robot_sf.nav.svg_map_parser import convert_map

if TYPE_CHECKING:
    from pathlib import Path

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
    # Concurrency: temporarily rebinds the ``_route_clearance`` module global
    # ``convert_map`` and restores it in ``finally``. Assumes single-process,
    # non-concurrent execution (matches the campaign runner's ``workers: 1``
    # dispatch); do not call from multiple threads sharing this module.
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


def prepare_campaign_preflight(
    cfg: CampaignConfig,
    *,
    output_root: Path | None = None,
    label: str | None = None,
    campaign_id: str | None = None,
    invoked_command: str | None = None,
    checkpoint_preflight_mode: str = "metadata_only",
    checkpoint_cache_dir: Path | None = None,
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
        checkpoint_preflight_mode=checkpoint_preflight_mode,
        checkpoint_cache_dir=checkpoint_cache_dir,
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
    "build_campaign_credibility_scorecard",
    "load_campaign_config",
    "prepare_campaign_preflight",
    "run_campaign",
    "write_campaign_report",
]
