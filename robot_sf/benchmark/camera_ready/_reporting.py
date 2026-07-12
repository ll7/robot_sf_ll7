"""Reporting row builders for camera-ready benchmark campaigns.

Extracted from ``robot_sf.benchmark.camera_ready_campaign`` for the #3385
decomposition. The legacy module re-exports these private helpers to preserve
existing imports.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.camera_ready._artifacts import _escape_markdown_cell
from robot_sf.benchmark.camera_ready._config_types import _AMV_DIMENSIONS, PlannerSpec
from robot_sf.benchmark.camera_ready._summaries import _extract_amv_taxonomy
from robot_sf.benchmark.fairness_contract import (
    build_fairness_report,
    emit_fairness_annotations,
)
from robot_sf.benchmark.fallback_policy import (
    classify_planner_row_status,
    summarize_benchmark_availability,
)
from robot_sf.benchmark.synthetic_actuation import (
    SyntheticActuationProfile,
    not_available_saturation_metrics,
)
from robot_sf.benchmark.utils import episode_metric_value
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path

_REPORT_METRICS: tuple[str, ...] = (
    "success",
    "collisions",
    "ped_collision_count",
    "obstacle_collision_count",
    "total_collision_count",
    "near_misses",
    "time_to_goal_norm",
    "path_efficiency",
    "comfort_exposure",
    "jerk_mean",
    "snqi",
)

_CREDIBILITY_SCORECARD_SCHEMA = "campaign_credibility_scorecard.v1"
_CREDIBILITY_SCORECARD_FACTORS: tuple[dict[str, str], ...] = (
    {
        "factor_id": "verification",
        "factor": "Verification",
        "description": "Repository and campaign checks supporting correct implementation.",
    },
    {
        "factor_id": "validation",
        "factor": "Validation",
        "description": "Comparison against trusted real-world or independent reference data.",
    },
    {
        "factor_id": "input_pedigree",
        "factor": "Input pedigree",
        "description": "Pinned inputs, hashes, manifests, and source provenance.",
    },
    {
        "factor_id": "uncertainty_characterization",
        "factor": "Uncertainty characterization",
        "description": "Seed repetition, confidence intervals, or variance analysis.",
    },
    {
        "factor_id": "results_robustness",
        "factor": "Results robustness",
        "description": "Coverage across planners, scenarios, and non-success classifications.",
    },
    {
        "factor_id": "use_history",
        "factor": "Use history",
        "description": "Prior accepted use of this campaign configuration or equivalent evidence.",
    },
)
_CREDIBILITY_STATUS_BY_SCORE = {
    0: "weak",
    1: "weak",
    2: "partial",
    3: "strong",
}


def _not_assessed_factor(spec: dict[str, str]) -> dict[str, Any]:
    """Build one fail-closed credibility factor row.

    Returns:
        Factor row with ``not_assessed`` status and a null score.
    """
    return {
        **spec,
        "status": "not_assessed",
        "score": None,
        "scale": "not_assessed | weak | partial | strong | not_applicable",
        "justification": "No campaign artifact supplied enough evidence for this factor.",
        "evidence": [],
    }


def build_campaign_credibility_scorecard(payload: dict[str, Any]) -> dict[str, Any]:
    """Build NASA-STD-7009-style per-campaign credibility scorecard.

    The scorecard is deliberately conservative: every expected factor is present, and factors
    without direct campaign evidence remain ``not_assessed`` instead of being silently omitted.

    Returns:
        Scorecard payload for JSON and Markdown campaign reports.
    """
    campaign = payload.get("campaign", {}) if isinstance(payload.get("campaign"), dict) else {}
    artifacts = payload.get("artifacts", {}) if isinstance(payload.get("artifacts"), dict) else {}
    rows = payload.get("planner_rows", []) if isinstance(payload.get("planner_rows"), list) else []
    factors = {
        spec["factor_id"]: _not_assessed_factor(spec) for spec in _CREDIBILITY_SCORECARD_FACTORS
    }

    if campaign.get("git_hash") and campaign.get("scenario_matrix_hash"):
        factors["input_pedigree"].update(
            {
                "status": "partial",
                "score": 2,
                "justification": (
                    "Campaign records git commit and scenario matrix hash; external input "
                    "lineage remains limited to recorded artifacts."
                ),
                "evidence": [
                    f"git_hash={campaign.get('git_hash')}",
                    f"scenario_matrix_hash={campaign.get('scenario_matrix_hash')}",
                    f"campaign_manifest={artifacts.get('campaign_manifest')}",
                ],
            }
        )

    seed_count = campaign.get("seed_count")
    try:
        seed_count_int = int(seed_count)
    except (TypeError, ValueError):
        seed_count_int = 0
    if seed_count_int > 1 or artifacts.get("seed_variability_json"):
        score = 2 if seed_count_int > 1 else 1
        factors["uncertainty_characterization"].update(
            {
                "status": _CREDIBILITY_STATUS_BY_SCORE[score],
                "score": score,
                "justification": (
                    f"Campaign records {seed_count_int} seed(s) and seed-variability artifacts "
                    "when available; no claim beyond campaign-level uncertainty."
                ),
                "evidence": [
                    f"seed_count={seed_count_int}",
                    f"seed_variability_json={artifacts.get('seed_variability_json')}",
                    f"statistical_sufficiency_json={artifacts.get('statistical_sufficiency_json')}",
                ],
            }
        )

    if artifacts.get("campaign_summary_json") and artifacts.get("campaign_table_csv"):
        factors["verification"].update(
            {
                "status": "weak",
                "score": 1,
                "justification": (
                    "Report was generated from structured campaign summary/table artifacts; "
                    "test-suite evidence is not embedded in the campaign artifact."
                ),
                "evidence": [
                    f"campaign_summary_json={artifacts.get('campaign_summary_json')}",
                    f"campaign_table_csv={artifacts.get('campaign_table_csv')}",
                ],
            }
        )

    total_runs = len(rows)
    successful_runs = sum(1 for row in rows if str(row.get("status")) == "ok")
    if total_runs:
        factors["results_robustness"].update(
            {
                "status": _CREDIBILITY_STATUS_BY_SCORE[1 if successful_runs else 0],
                "score": 1 if successful_runs else 0,
                "justification": (
                    f"Campaign reports {successful_runs}/{total_runs} successful planner row(s); "
                    "fallback/degraded rows remain caveats, not success evidence."
                ),
                "evidence": [
                    f"total_runs={total_runs}",
                    f"successful_runs={successful_runs}",
                    f"row_status_summary={campaign.get('row_status_summary', {})}",
                ],
            }
        )

    assessed_scores = [
        int(factor["score"])
        for factor in factors.values()
        if factor.get("status") not in {"not_assessed", "not_applicable"}
        and factor.get("score") is not None
    ]
    overall_score = (
        round(sum(assessed_scores) / len(assessed_scores), 2) if assessed_scores else None
    )
    return {
        "schema_version": _CREDIBILITY_SCORECARD_SCHEMA,
        "standard_reference": "NASA-STD-7009B-inspired credibility assessment",
        "campaign_id": campaign.get("campaign_id", "unknown"),
        "overall_score": overall_score,
        "overall_status": (
            _CREDIBILITY_STATUS_BY_SCORE[round(overall_score)]
            if overall_score is not None
            else "not_assessed"
        ),
        "claim_boundary": (
            "Scorecard is reporting metadata for campaign credibility dimensions; it is not "
            "benchmark proof, paper evidence, or real-world validation."
        ),
        "factors": [factors[spec["factor_id"]] for spec in _CREDIBILITY_SCORECARD_FACTORS],
    }


def _metric_mean(block: dict[str, Any], metric: str) -> float:
    """Extract aggregate mean value for one metric.

    Returns:
        Mean metric value, or ``nan`` when unavailable.
    """
    metric_block = block.get(metric)
    if not isinstance(metric_block, dict):
        return float("nan")
    value = metric_block.get("mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _metric_ci(block: dict[str, Any], metric: str) -> tuple[float, float]:
    """Extract mean confidence interval values for one metric.

    Returns:
        Tuple ``(low, high)`` with ``nan`` values when unavailable.
    """
    metric_block = block.get(metric)
    if not isinstance(metric_block, dict):
        return (float("nan"), float("nan"))
    ci = metric_block.get("mean_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        return (float("nan"), float("nan"))
    try:
        return (float(ci[0]), float(ci[1]))
    except (TypeError, ValueError):
        return (float("nan"), float("nan"))


def _episode_metric_mean(records: list[dict[str, Any]], metric: str) -> float:
    """Return a mean metric value directly from episode records."""
    values: list[float] = []
    for record in records:
        value = episode_metric_value(record, metric)
        if value is None or not math.isfinite(value):
            continue
        values.append(value)
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _episode_metric_min(records: list[dict[str, Any]], metric: str) -> float:
    """Return the campaign-wide minimum of a per-episode metric.

    Used for worst-case safety fields (e.g. ``min_clearance_m``) where the
    smallest per-episode value across the whole campaign is the fail-closed
    quantity a safety-floor release gate should evaluate. Returns ``nan`` when
    no finite per-episode value is present, so downstream gates report
    ``not_evaluable`` rather than a misleading value.

    Returns:
        Minimum finite metric value across episodes, or ``nan`` when unavailable.
    """
    values: list[float] = []
    for record in records:
        value = episode_metric_value(record, metric)
        if value is None or not math.isfinite(value):
            continue
        values.append(value)
    if not values:
        return float("nan")
    return float(min(values))


def _episode_metric_ci(records: list[dict[str, Any]], metric: str) -> tuple[float, float]:
    """Return CI placeholders for metrics recomputed directly from episode records.

    The aggregate summary block can contain stale CI values when means are
    corrected from per-episode termination semantics. Emit ``nan`` CI bounds in
    that case rather than pairing corrected means with misleading intervals.
    """
    _ = records, metric
    return (float("nan"), float("nan"))


def _safe_float(value: float | None) -> str:
    """Format a float for report tables with NaN handling.

    Returns:
        Fixed-precision string or ``\"nan\"``.
    """
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _normalized_algorithm_metadata_contract(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the algorithm metadata contract as a dictionary."""
    contract = summary.get("algorithm_metadata_contract")
    if isinstance(contract, dict):
        return contract
    return {}


def _planner_report_row(  # noqa: C901, PLR0912, PLR0915
    planner: PlannerSpec,
    summary: dict[str, Any],
    aggregates: dict[str, Any] | None,
    *,
    kinematics: str,
    synthetic_actuation_profile: SyntheticActuationProfile | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one campaign table row for a planner run.

    Returns:
        Flattened row payload for CSV/Markdown export.
    """
    if aggregates:
        groups = [name for name in aggregates.keys() if name != "_meta"]
        metric_block = aggregates.get(groups[0], {}) if groups else {}
    else:
        metric_block = {}

    success_ci = _metric_ci(metric_block, "success")
    collision_ci = _metric_ci(metric_block, "collisions")
    snqi_ci = _metric_ci(metric_block, "snqi")

    algorithm_metadata_contract = _normalized_algorithm_metadata_contract(summary)
    availability = summarize_benchmark_availability(summary)
    execution_mode = availability.execution_mode
    planner_kinematics = algorithm_metadata_contract.get("planner_kinematics")
    if not isinstance(planner_kinematics, dict):
        planner_kinematics = {}
    preflight_status = str((summary.get("preflight") or {}).get("status", "unknown"))
    learned_policy_contract = (summary.get("preflight") or {}).get("learned_policy_contract")
    contract_status = "not_applicable"
    contract_critical = 0
    contract_warnings = 0
    if isinstance(learned_policy_contract, dict):
        contract_status = str(learned_policy_contract.get("status", "not_applicable"))
        critical_list = learned_policy_contract.get("critical_mismatches")
        warning_list = learned_policy_contract.get("warnings")
        if isinstance(critical_list, list):
            contract_critical = len(critical_list)
        if isinstance(warning_list, list):
            contract_warnings = len(warning_list)
    status = str(summary.get("status", "unknown"))
    readiness_status = availability.readiness_status

    resolved_metrics = {
        "success_mean": _metric_mean(metric_block, "success"),
        "collisions_mean": _metric_mean(metric_block, "collisions"),
        "ped_collision_count_mean": _metric_mean(metric_block, "ped_collision_count"),
        "obstacle_collision_count_mean": _metric_mean(metric_block, "obstacle_collision_count"),
        "total_collision_count_mean": _metric_mean(metric_block, "total_collision_count"),
        "near_misses_mean": _metric_mean(metric_block, "near_misses"),
        "min_clearance_mean": _metric_mean(metric_block, "min_clearance"),
        "time_to_collision_min_mean": _metric_mean(metric_block, "time_to_collision_min"),
        "time_to_goal_norm_mean": _metric_mean(metric_block, "time_to_goal_norm"),
        "failure_to_progress_mean": _metric_mean(metric_block, "failure_to_progress"),
        "stalled_time_mean": _metric_mean(metric_block, "stalled_time"),
        "path_efficiency_mean": _metric_mean(metric_block, "path_efficiency"),
        "velocity_max_mean": _metric_mean(metric_block, "velocity_max"),
        "acceleration_max_mean": _metric_mean(metric_block, "acceleration_max"),
        "comfort_exposure_mean": _metric_mean(metric_block, "comfort_exposure"),
        "jerk_mean": _metric_mean(metric_block, "jerk_mean"),
        "jerk_max_mean": _metric_mean(metric_block, "jerk_max"),
        "curvature_mean_mean": _metric_mean(metric_block, "curvature_mean"),
        "energy_mean": _metric_mean(metric_block, "energy"),
        "command_clip_fraction_mean": _metric_mean(metric_block, "command_clip_fraction"),
        "yaw_rate_saturation_fraction_mean": _metric_mean(
            metric_block, "yaw_rate_saturation_fraction"
        ),
        "signed_braking_peak_m_s2_mean": _metric_mean(metric_block, "signed_braking_peak_m_s2"),
        "snqi_mean": _metric_mean(metric_block, "snqi"),
        # Dedicated release-gate retention fields (issue #4326). These use the
        # exact names the release-gate evaluator consumes so FUTURE campaigns
        # become evaluable for the clearance/proxemic gates. min_clearance_m is
        # the campaign-wide worst-case (minimum) clearance, distinct from the
        # mean-of-per-episode-minimums already retained as min_clearance_mean;
        # the aggregate CI block carries no minimum, so it defaults to nan and is
        # filled from episode records below. proxemic_intrusion_rate is the mean
        # per-episode personal-space intrusion fraction. Campaigns that never
        # recorded these stay nan -> not_evaluable (fail-closed); no backfill.
        "min_clearance_m": float("nan"),
        "proxemic_intrusion_rate": _metric_mean(metric_block, "social_proxemic_intrusion_frac"),
    }
    if records:
        metric_sources = {
            "success_mean": "success",
            "collisions_mean": "collisions",
            "ped_collision_count_mean": "ped_collision_count",
            "obstacle_collision_count_mean": "obstacle_collision_count",
            "total_collision_count_mean": "total_collision_count",
            "near_misses_mean": "near_misses",
            "min_clearance_mean": "min_clearance",
            "time_to_collision_min_mean": "time_to_collision_min",
            "time_to_goal_norm_mean": "time_to_goal_norm",
            "failure_to_progress_mean": "failure_to_progress",
            "stalled_time_mean": "stalled_time",
            "path_efficiency_mean": "path_efficiency",
            "velocity_max_mean": "velocity_max",
            "acceleration_max_mean": "acceleration_max",
            "comfort_exposure_mean": "comfort_exposure",
            "jerk_mean": "jerk_mean",
            "jerk_max_mean": "jerk_max",
            "curvature_mean_mean": "curvature_mean",
            "energy_mean": "energy",
            "command_clip_fraction_mean": "command_clip_fraction",
            "yaw_rate_saturation_fraction_mean": "yaw_rate_saturation_fraction",
            "signed_braking_peak_m_s2_mean": "signed_braking_peak_m_s2",
            "snqi_mean": "snqi",
            # Release-gate proxemic field (issue #4326): mean per-episode
            # personal-space intrusion fraction, retained under its gate name.
            "proxemic_intrusion_rate": "social_proxemic_intrusion_frac",
        }
        for field_name, metric_name in metric_sources.items():
            resolved_metrics[field_name] = _episode_metric_mean(records, metric_name)
            if field_name == "success_mean":
                success_ci = _episode_metric_ci(records, metric_name)
            elif field_name == "collisions_mean":
                collision_ci = _episode_metric_ci(records, metric_name)
            elif field_name == "snqi_mean":
                snqi_ci = _episode_metric_ci(records, metric_name)
        # Release-gate clearance floor (issue #4326): worst-case (minimum)
        # per-episode clearance across the campaign, not a mean, so the safety
        # floor gate evaluates the strictest observed value.
        resolved_metrics["min_clearance_m"] = _episode_metric_min(records, "min_clearance")
    episode_count = (
        len(records)
        if records is not None
        else int(summary.get("episodes_total", summary.get("written", 0)))
    )

    row = {
        "planner_key": planner.key,
        "algo": planner.algo,
        "human_model_variant": planner.human_model_variant or "",
        "human_model_source": planner.human_model_source or "",
        "planner_group": planner.planner_group,
        "kinematics": kinematics,
        "status": status,
        "episodes": int(episode_count),
        "started_at_utc": str(summary.get("started_at_utc", "unknown")),
        "finished_at_utc": str(summary.get("finished_at_utc", "unknown")),
        "runtime_sec": _safe_float(summary.get("runtime_sec")),
        "episodes_per_second": _safe_float(summary.get("episodes_per_second")),
        "failed_jobs": int(summary.get("failed_jobs", 0)),
        "success_mean": _safe_float(resolved_metrics["success_mean"]),
        "collisions_mean": _safe_float(resolved_metrics["collisions_mean"]),
        "ped_collision_count_mean": _safe_float(resolved_metrics["ped_collision_count_mean"]),
        "obstacle_collision_count_mean": _safe_float(
            resolved_metrics["obstacle_collision_count_mean"]
        ),
        "total_collision_count_mean": _safe_float(resolved_metrics["total_collision_count_mean"]),
        "near_misses_mean": _safe_float(resolved_metrics["near_misses_mean"]),
        "min_clearance_mean": _safe_float(resolved_metrics["min_clearance_mean"]),
        # Dedicated release-gate safety field (issue #4326): worst-case clearance.
        "min_clearance_m": _safe_float(resolved_metrics["min_clearance_m"]),
        "time_to_collision_min_mean": _safe_float(resolved_metrics["time_to_collision_min_mean"]),
        "time_to_goal_norm_mean": _safe_float(resolved_metrics["time_to_goal_norm_mean"]),
        "failure_to_progress_mean": _safe_float(resolved_metrics["failure_to_progress_mean"]),
        "stalled_time_mean": _safe_float(resolved_metrics["stalled_time_mean"]),
        "path_efficiency_mean": _safe_float(resolved_metrics["path_efficiency_mean"]),
        "velocity_max_mean": _safe_float(resolved_metrics["velocity_max_mean"]),
        "acceleration_max_mean": _safe_float(resolved_metrics["acceleration_max_mean"]),
        "comfort_exposure_mean": _safe_float(resolved_metrics["comfort_exposure_mean"]),
        # Dedicated release-gate comfort field (issue #4326): proxemic-intrusion rate.
        "proxemic_intrusion_rate": _safe_float(resolved_metrics["proxemic_intrusion_rate"]),
        "jerk_mean": _safe_float(resolved_metrics["jerk_mean"]),
        "jerk_max_mean": _safe_float(resolved_metrics["jerk_max_mean"]),
        "curvature_mean_mean": _safe_float(resolved_metrics["curvature_mean_mean"]),
        "energy_mean": _safe_float(resolved_metrics["energy_mean"]),
        "command_clip_fraction_mean": _safe_float(resolved_metrics["command_clip_fraction_mean"]),
        "yaw_rate_saturation_fraction_mean": _safe_float(
            resolved_metrics["yaw_rate_saturation_fraction_mean"]
        ),
        "signed_braking_peak_m_s2_mean": _safe_float(
            resolved_metrics["signed_braking_peak_m_s2_mean"]
        ),
        "snqi_mean": _safe_float(resolved_metrics["snqi_mean"]),
        "success_ci_low": _safe_float(success_ci[0]),
        "success_ci_high": _safe_float(success_ci[1]),
        "collision_ci_low": _safe_float(collision_ci[0]),
        "collision_ci_high": _safe_float(collision_ci[1]),
        "snqi_ci_low": _safe_float(snqi_ci[0]),
        "snqi_ci_high": _safe_float(snqi_ci[1]),
        "execution_mode": execution_mode,
        "execution_detail": str(planner_kinematics.get("execution_detail", "unspecified")),
        "planner_command_space": str(planner_kinematics.get("planner_command_space", "unknown")),
        "benchmark_command_space": str(
            planner_kinematics.get("benchmark_command_space", "unknown")
        ),
        "projection_policy": str(planner_kinematics.get("projection_policy", "unknown")),
        "readiness_status": readiness_status,
        "availability_status": availability.availability_status,
        "benchmark_success": str(availability.benchmark_success).lower(),
        "availability_reason": availability.availability_reason or "",
        "most_likely_failure_reason": (
            (availability.availability_reason or summary.get("error") or "")
            if availability.availability_status != "available" or summary.get("status") == "failed"
            else ""
        ),
        "readiness_tier": str((summary.get("algorithm_readiness") or {}).get("tier", "unknown")),
        "preflight_status": preflight_status,
        "socnav_prereq_policy": planner.socnav_missing_prereq_policy,
        "learned_policy_contract_status": contract_status,
        "learned_policy_contract_critical": contract_critical,
        "learned_policy_contract_warnings": contract_warnings,
    }
    if synthetic_actuation_profile is not None:
        row["synthetic_actuation_profile_name"] = synthetic_actuation_profile.name
        row["synthetic_actuation_profile_version"] = synthetic_actuation_profile.profile_version
        row["synthetic_actuation_latency_mode"] = synthetic_actuation_profile.latency_mode
        row["synthetic_actuation_update_mode"] = synthetic_actuation_profile.update_mode
        row["synthetic_actuation_max_linear_accel_m_s2"] = _safe_float(
            synthetic_actuation_profile.max_linear_accel_m_s2
        )
        row["synthetic_actuation_max_linear_decel_m_s2"] = _safe_float(
            synthetic_actuation_profile.max_linear_decel_m_s2
        )
        row["synthetic_actuation_max_yaw_rate_rad_s"] = _safe_float(
            synthetic_actuation_profile.max_yaw_rate_rad_s
        )
        row["synthetic_actuation_max_angular_accel_rad_s2"] = _safe_float(
            synthetic_actuation_profile.max_angular_accel_rad_s2
        )
    else:
        for key, value in not_available_saturation_metrics().items():
            row[f"{key}_mean"] = value
    feasibility = algorithm_metadata_contract.get("kinematics_feasibility")
    if isinstance(feasibility, dict):
        row["commands_evaluated"] = int(feasibility.get("commands_evaluated", 0) or 0)
        row["projection_rate"] = _safe_float(float(feasibility.get("projection_rate", 0.0) or 0.0))
        row["infeasible_rate"] = _safe_float(float(feasibility.get("infeasible_rate", 0.0) or 0.0))
    else:
        row["commands_evaluated"] = 0
        row["projection_rate"] = _safe_float(0.0)
        row["infeasible_rate"] = _safe_float(0.0)
    return row


def _scenario_family(record: dict[str, Any]) -> str:
    """Resolve scenario-family/archetype label from episode record metadata.

    Returns:
        Best-effort scenario family label.
    """
    scenario_params = record.get("scenario_params")
    if not isinstance(scenario_params, dict):
        scenario_params = {}
    metadata = scenario_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("archetype", "scenario_family", "family"):
        value = metadata.get(key) or scenario_params.get(key) or record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    scenario_id = record.get("scenario_id")
    if isinstance(scenario_id, str) and scenario_id.strip():
        return scenario_id.split("_", 1)[0]
    return "unknown"


def _campaign_scenario_id(scenario: dict[str, Any]) -> str:
    """Return the stable identifier used to join scenario metadata to campaign sidecars."""
    for key in ("name", "scenario_id", "id"):
        value = scenario.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _build_scenario_amv_lookup(
    scenarios: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Build a lookup from scenario identifier to AMV taxonomy dimensions.

    Returns:
        dict[str, dict[str, str]]: Mapping from scenario name/id to AMV taxonomy
        values keyed by dimension name. Scenarios without AMV data map to empty
        dicts so downstream code can distinguish absent from empty.
    """
    lookup: dict[str, dict[str, str]] = {}
    for scenario in scenarios:
        scenario_id = _campaign_scenario_id(scenario)
        taxonomy = _extract_amv_taxonomy(scenario)
        lookup[scenario_id] = taxonomy
    return lookup


def _build_breakdown_rows(  # noqa: C901
    run_entries: list[dict[str, Any]],
    *,
    scenario_amv_lookup: dict[str, dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build per-scenario and per-family campaign diagnostic rows.

    When ``scenario_amv_lookup`` is provided, AMV taxonomy columns
    (``use_case``, ``context``, ``speed_regime``, ``maneuver_type``) are
    merged into per-scenario rows from the matching scenario identifier.
    Per-family rows receive the union of AMV dimension values observed
    across contributing scenarios, joined by semicolons. Scenarios or
    families without AMV data emit empty-string placeholders rather than
    absent columns, so downstream consumers never mistake a missing column
    for an unavailable taxonomy dimension.

    Returns:
        Tuple of per-scenario rows and per-family rows.
    """
    amv_lookup = scenario_amv_lookup if scenario_amv_lookup is not None else {}
    per_scenario: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    per_family: dict[tuple[str, str, str], dict[str, Any]] = {}

    family_amv_values: defaultdict[tuple[str, str, str], defaultdict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    def _add_metric(bucket: dict[str, Any], metric: str, value: float | None) -> None:
        """Append one finite metric sample to an aggregation bucket."""
        if value is None:
            return
        bucket.setdefault(metric, []).append(value)

    def _mean(values: list[float]) -> str:
        """Return the report-formatted arithmetic mean for a metric bucket.

        Returns:
            str: Formatted finite mean, or ``"nan"`` for an empty bucket.
        """
        # Return a display-formatted value via _safe_float for table rendering.
        if not values:
            return "nan"
        return _safe_float(float(sum(values) / len(values)))

    for entry in run_entries:
        planner = entry.get("planner") if isinstance(entry.get("planner"), dict) else {}
        planner_key = str(planner.get("key", "unknown"))
        algo = str(planner.get("algo", "unknown"))
        episodes_path = entry.get("episodes_path")
        if not isinstance(episodes_path, str):
            continue
        candidate = get_repository_root() / episodes_path
        if not candidate.exists():
            continue
        for record in read_jsonl(str(candidate)):
            if not isinstance(record, dict):
                continue
            scenario_id = str(record.get("scenario_id", "unknown"))
            family = _scenario_family(record)
            scenario_key = (planner_key, algo, scenario_id, family)
            family_key = (planner_key, algo, family)

            scenario_bucket = per_scenario.setdefault(
                scenario_key,
                {
                    "planner_key": planner_key,
                    "algo": algo,
                    "scenario_id": scenario_id,
                    "scenario_family": family,
                    "episodes": 0,
                },
            )
            family_bucket = per_family.setdefault(
                family_key,
                {
                    "planner_key": planner_key,
                    "algo": algo,
                    "scenario_family": family,
                    "episodes": 0,
                },
            )
            scenario_bucket["episodes"] += 1
            family_bucket["episodes"] += 1
            for metric in _REPORT_METRICS:
                value = episode_metric_value(record, metric)
                if value is not None and not math.isfinite(value):
                    value = None
                _add_metric(scenario_bucket, metric, value)
                _add_metric(family_bucket, metric, value)

            scenario_amv = amv_lookup.get(scenario_id, {})
            for dimension in _AMV_DIMENSIONS:
                scenario_bucket.setdefault(dimension, scenario_amv.get(dimension, ""))

            for dimension in _AMV_DIMENSIONS:
                dim_value = scenario_amv.get(dimension, "")
                if dim_value:
                    family_amv_values[family_key][dimension].add(dim_value)

    def _finalize(row: dict[str, Any]) -> dict[str, Any]:
        """Replace metric sample lists with report-ready mean fields.

        Returns:
            dict[str, Any]: Finalized per-scenario or per-family summary row.
        """
        finalized = dict(row)
        for metric in _REPORT_METRICS:
            values = finalized.pop(metric, [])
            if not isinstance(values, list):
                values = []
            finalized[f"{metric}_mean"] = _mean(values)
        return finalized

    scenario_rows = sorted(
        (_finalize(row) for row in per_scenario.values()),
        key=lambda row: (
            row.get("planner_key", ""),
            row.get("scenario_id", ""),
            row.get("scenario_family", ""),
        ),
    )
    family_rows_data: list[dict[str, Any]] = []
    for family_key, family_row in per_family.items():
        finalized = _finalize(family_row)
        family_dims = family_amv_values.get(family_key, {})
        for dimension in _AMV_DIMENSIONS:
            values = family_dims.get(dimension)
            if values:
                finalized[dimension] = ";".join(sorted(values))
            else:
                finalized[dimension] = ""
        family_rows_data.append(finalized)

    family_rows_data.sort(
        key=lambda row: (
            row.get("planner_key", ""),
            row.get("scenario_family", ""),
        ),
    )
    return scenario_rows, family_rows_data


def _strict_vs_fallback_comparisons(rows: list[dict[str, Any]]) -> list[str]:
    """Build strict-vs-fallback comparison summaries when both modes are present.

    Returns:
        Human-readable comparison lines.
    """
    by_algo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        algo = str(row.get("algo", "unknown"))
        by_algo.setdefault(algo, []).append(row)

    lines: list[str] = []
    for algo, algo_rows in sorted(by_algo.items()):
        strict = [row for row in algo_rows if str(row.get("socnav_prereq_policy")) == "fail-fast"]
        fallback = [row for row in algo_rows if str(row.get("socnav_prereq_policy")) == "fallback"]
        if not strict or not fallback:
            continue
        strict_row = strict[0]
        fallback_row = fallback[0]
        lines.append(
            f"`{algo}`: strict preflight={strict_row.get('preflight_status')}, "
            f"fallback preflight={fallback_row.get('preflight_status')}, "
            f"strict success={strict_row.get('success_mean')}, "
            f"fallback success={fallback_row.get('success_mean')}"
        )
    return lines


def write_campaign_report(  # noqa: C901, PLR0912, PLR0915
    path: Path, payload: dict[str, Any]
) -> None:
    """Write a human-readable campaign report in Markdown."""
    campaign = payload.get("campaign", {})
    rows = payload.get("planner_rows", [])
    warnings = payload.get("warnings", [])
    scorecard = payload.get("credibility_scorecard")
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
    if not isinstance(scorecard, dict):
        scorecard = build_campaign_credibility_scorecard(payload)
    factors = scorecard.get("factors", []) if isinstance(scorecard.get("factors"), list) else []
    lines.extend(
        [
            "",
            "## Credibility Scorecard",
            "",
            (
                "NASA-STD-7009B-inspired campaign credibility metadata. "
                "Unscored factors are shown as `not_assessed`."
            ),
            "",
            f"- Schema: `{scorecard.get('schema_version', 'unknown')}`",
            f"- Overall status: `{scorecard.get('overall_status', 'not_assessed')}`",
            f"- Overall score: `{scorecard.get('overall_score')}`",
            f"- Claim boundary: `{scorecard.get('claim_boundary', 'unknown')}`",
            "",
            "| factor | status | score | justification |",
            "|---|---|---:|---|",
        ]
    )
    for factor in factors:
        lines.append(
            "| "
            f"{_escape_markdown_cell(factor.get('factor'))} | "
            f"{_escape_markdown_cell(factor.get('status'))} | "
            f"{_escape_markdown_cell(factor.get('score'))} | "
            f"{_escape_markdown_cell(factor.get('justification'))} |"
        )

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

    lines.extend(["", "## Fairness Contract", ""])
    fairness_report = build_fairness_report(rows)
    emit_fairness_annotations(fairness_report, rows)
    fairness_verdict = fairness_report.ranking_claim_allowed
    hard_mismatches = [m for m in fairness_report.mismatches if m.severity == "hard"]
    soft_mismatches = [m for m in fairness_report.mismatches if m.severity == "soft"]

    lines.append(f"- Ranking claim allowed: `{fairness_verdict}`")
    lines.append(f"- Fair subset size: `{len(fairness_report.fair_subset)}`")
    lines.append(f"- Excluded planners: `{len(fairness_report.excluded_planners)}`")
    lines.append(f"- Hard mismatches: `{len(hard_mismatches)}`")
    lines.append(f"- Soft mismatches (caveats): `{len(soft_mismatches)}`")

    if fairness_report.fair_subset:
        lines.append("")
        lines.append("Fair comparison subset:")
        for name in fairness_report.fair_subset:
            lines.append(f"- `{name}`")

    if fairness_report.excluded_planners:
        lines.append("")
        lines.append("Excluded planners:")
        for name in fairness_report.excluded_planners:
            lines.append(f"- `{name}`")

    if hard_mismatches:
        lines.append("")
        lines.append("Hard mismatches (block ranking claims):")
        for m in hard_mismatches:
            lines.append(f"- **{m.dimension}**: {m.planner_a} vs {m.planner_b} — {m.description}")

    if soft_mismatches:
        lines.append("")
        lines.append("Soft mismatches (caveats):")
        for m in soft_mismatches:
            lines.append(f"- **{m.dimension}**: {m.planner_a} vs {m.planner_b} — {m.description}")

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
