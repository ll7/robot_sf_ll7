"""Artifact-driven scenario difficulty analysis for camera-ready benchmark campaigns."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path

_PRIMARY_PROXY_METRICS: tuple[tuple[str, bool, float], ...] = (
    ("success_mean", False, 0.40),
    ("collisions_mean", True, 0.25),
    ("near_misses_mean", True, 0.15),
    ("time_to_goal_norm_mean", True, 0.20),
)
_SUPPORTING_METRICS: tuple[str, ...] = (
    "success_mean",
    "collisions_mean",
    "near_misses_mean",
    "time_to_goal_norm_mean",
    "snqi_mean",
)
_SEED_METRICS: tuple[str, ...] = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "snqi",
)
_CORE_PLANNER_GROUP = "core"
_READINESS_EXCLUDED_STATUSES = frozenset({"fallback", "degraded"})
_PREFLIGHT_EXCLUDED_STATUSES = frozenset({"fallback"})
_BENCHMARK_EXCLUDED_STATUSES = frozenset(
    {"failed", "partial-failure", "not_available", "not-available"}
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def _max(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(max(finite))


def _metric_range(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(max(finite) - min(finite))


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    bounded_q = min(1.0, max(0.0, float(q)))
    rank = round((len(ordered) - 1) * bounded_q)
    return ordered[rank]


def _scenario_keys(scenario: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []
    for field in ("scenario_id", "name", "id"):
        value = scenario.get(field)
        if isinstance(value, str) and value.strip() and value.strip() not in keys:
            keys.append(value.strip())
    return keys


def _scenario_family(row: Mapping[str, Any]) -> str:
    for field in ("scenario_family", "archetype", "family"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    scenario_id = row.get("scenario_id")
    if isinstance(scenario_id, str) and scenario_id.strip():
        return scenario_id.strip()
    return "unknown"


def _normalized_ranks(
    values_by_key: Mapping[str, float],
    *,
    higher_is_harder: bool,
) -> dict[str, float]:
    if not values_by_key:
        return {}
    if higher_is_harder:
        ordered = sorted(values_by_key.items(), key=lambda item: (item[1], item[0]))
    else:
        ordered = sorted(values_by_key.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 0.0}

    out: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        normalized = float(average_rank / (len(ordered) - 1))
        for key, _value in ordered[start:end]:
            out[key] = normalized
        start = end
    return out


def _planner_row_index(planner_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in planner_rows:
        planner_key = row.get("planner_key")
        if isinstance(planner_key, str) and planner_key.strip():
            out[planner_key.strip()] = dict(row)
    return out


def _is_benchmark_success(row: Mapping[str, Any]) -> bool:
    value = row.get("benchmark_success")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"false", "0", "no"}


def _is_consensus_planner(row: Mapping[str, Any]) -> bool:
    planner_group = str(row.get("planner_group", "")).strip().lower()
    if not planner_group:
        return False
    if planner_group != _CORE_PLANNER_GROUP:
        return False
    if str(row.get("readiness_status", "")).strip().lower() in _READINESS_EXCLUDED_STATUSES:
        return False
    if str(row.get("preflight_status", "")).strip().lower() in _PREFLIGHT_EXCLUDED_STATUSES:
        return False
    if str(row.get("status", "")).strip().lower() in _BENCHMARK_EXCLUDED_STATUSES:
        return False
    return _is_benchmark_success(row)


def _build_seed_index(
    seed_payload: Mapping[str, Any] | None,
) -> dict[tuple[str, str], dict[str, Any]]:
    rows = seed_payload.get("rows") if isinstance(seed_payload, Mapping) else None
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        scenario_id = row.get("scenario_id")
        planner_key = row.get("planner_key")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            continue
        if not isinstance(planner_key, str) or not planner_key.strip():
            continue
        out[(scenario_id.strip(), planner_key.strip())] = dict(row)
    return out


def _seed_field(
    seed_row: Mapping[str, Any] | None,
    metric: str,
    field: str,
) -> float | None:
    if not isinstance(seed_row, Mapping):
        return None
    summary = seed_row.get("summary")
    if not isinstance(summary, Mapping):
        return None
    metric_summary = summary.get(metric)
    if not isinstance(metric_summary, Mapping):
        return None
    return _safe_float(metric_summary.get(field))


def _preview_metadata_lookup(  # noqa: C901
    preview_payload: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], bool]:
    if not isinstance(preview_payload, Mapping):
        return {}, False
    warnings = preview_payload.get("route_clearance_warnings")
    warning_index: dict[str, dict[str, Any]] = {}
    if isinstance(warnings, list):
        for warning in warnings:
            if not isinstance(warning, dict):
                continue
            scenario_name = warning.get("scenario")
            if isinstance(scenario_name, str) and scenario_name.strip():
                warning_index[scenario_name.strip()] = dict(warning)

    lookup: dict[str, dict[str, Any]] = {}
    scenarios = preview_payload.get("scenarios")
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            metadata = scenario.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            simulation_config = scenario.get("simulation_config")
            if not isinstance(simulation_config, dict):
                simulation_config = {}
            route_warning = None
            for key in _scenario_keys(scenario):
                route_warning = warning_index.get(key)
                if route_warning is not None:
                    break
            row = {
                "archetype": metadata.get("archetype"),
                "flow": metadata.get("flow"),
                "behavior": metadata.get("behavior"),
                "primary_capability": metadata.get("primary_capability"),
                "target_failure_mode": metadata.get("target_failure_mode"),
                "determinism": metadata.get("determinism"),
                "ped_density": _safe_float(simulation_config.get("ped_density")),
                "map_file": scenario.get("map_file"),
                "route_clearance_warning": route_warning is not None,
                "route_clearance_scope": (
                    route_warning.get("warning_scope") if isinstance(route_warning, dict) else None
                ),
                "min_clearance_margin_m": (
                    _safe_float(route_warning.get("min_clearance_margin_m"))
                    if isinstance(route_warning, dict)
                    else None
                ),
            }
            for key in _scenario_keys(scenario):
                lookup[key] = row
    return lookup, bool(preview_payload.get("truncated", False))


def _load_verified_simple_ids(path: Path | None) -> tuple[set[str], str | None]:
    if path is None or not path.exists():
        return set(), None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = payload.get("scenarios") if isinstance(payload, Mapping) else None
    if not isinstance(scenarios, list):
        return set(), str(path)
    scenario_ids: set[str] = set()
    for scenario in scenarios:
        if not isinstance(scenario, Mapping):
            continue
        for key in _scenario_keys(scenario):
            scenario_ids.add(key)
    return scenario_ids, str(path)


def _difficulty_weighted_score(
    scenario_id: str,
    component_ranks: Mapping[str, Mapping[str, float]],
) -> tuple[float | None, dict[str, float]]:
    components: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, _higher_is_harder, weight in _PRIMARY_PROXY_METRICS:
        metric_rank = component_ranks.get(metric_name, {}).get(scenario_id)
        if metric_rank is None:
            continue
        components[metric_name] = metric_rank
        weighted_sum += float(metric_rank) * weight
        total_weight += weight
    if total_weight <= 0.0:
        return None, components
    return float(weighted_sum / total_weight), components


def _planner_quality_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        planner_key = row.get("planner_key")
        if isinstance(planner_key, str) and planner_key.strip():
            grouped[planner_key.strip()].append(row)

    output: list[dict[str, Any]] = []
    for planner_key, planner_rows in grouped.items():
        template = planner_rows[0]
        output.append(
            {
                "planner_key": planner_key,
                "algo": str(template.get("algo") or "unknown").strip() or "unknown",
                "planner_group": str(template.get("planner_group") or "unknown").strip()
                or "unknown",
                "success_mean": _mean(_safe_float(row.get("success_mean")) for row in planner_rows),
                "collisions_mean": _mean(
                    _safe_float(row.get("collisions_mean")) for row in planner_rows
                ),
                "near_misses_mean": _mean(
                    _safe_float(row.get("near_misses_mean")) for row in planner_rows
                ),
                "time_to_goal_norm_mean": _mean(
                    _safe_float(row.get("time_to_goal_norm_mean")) for row in planner_rows
                ),
                "scenario_count": len(planner_rows),
            }
        )
    output.sort(
        key=lambda row: (
            -(row.get("success_mean") if row.get("success_mean") is not None else float("-inf")),
            row.get("collisions_mean") if row.get("collisions_mean") is not None else float("inf"),
            row.get("near_misses_mean")
            if row.get("near_misses_mean") is not None
            else float("inf"),
            row.get("time_to_goal_norm_mean")
            if row.get("time_to_goal_norm_mean") is not None
            else float("inf"),
            str(row.get("planner_key", "")),
        )
    )
    return output


def _spearman_rank_correlation(
    full_rows: Sequence[Mapping[str, Any]],
    subset_rows: Sequence[Mapping[str, Any]],
) -> float | None:
    full_ranks = {
        str(row.get("planner_key")): index + 1
        for index, row in enumerate(full_rows)
        if isinstance(row.get("planner_key"), str)
    }
    subset_ranks = {
        str(row.get("planner_key")): index + 1
        for index, row in enumerate(subset_rows)
        if isinstance(row.get("planner_key"), str)
    }
    common = sorted(set(full_ranks) & set(subset_ranks))
    n = len(common)
    if n < 2:
        return None
    d_squared = sum((full_ranks[key] - subset_ranks[key]) ** 2 for key in common)
    return float(1.0 - ((6.0 * d_squared) / (n * (n * n - 1))))


def _planner_selection_rows(
    scenario_rows: Sequence[Mapping[str, Any]],
    planner_index: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    core_keys = {key for key, row in planner_index.items() if _is_consensus_planner(row)}
    selected = [
        dict(row)
        for row in scenario_rows
        if isinstance(row.get("planner_key"), str) and row.get("planner_key") in core_keys
    ]
    if selected:
        return selected, "core benchmark-success planners"
    return [dict(row) for row in scenario_rows], "all planners (fallback: no eligible core set)"


def _verified_simple_assessment(
    scenario_rows: Sequence[Mapping[str, Any]],
    planner_index: Mapping[str, Mapping[str, Any]],
    *,
    manifest_path: Path | None,
) -> dict[str, Any]:
    subset_ids, manifest_path_text = _load_verified_simple_ids(manifest_path)
    if not subset_ids:
        return {
            "status": "manifest_missing",
            "manifest_path": manifest_path_text,
            "subset_size": 0,
            "matched_scenario_count": 0,
            "matched_scenarios": [],
            "comparison_planner_selection": None,
            "rank_correlation": None,
            "worth_adding": None,
            "recommendation": (
                "Verified-simple candidate manifest is unavailable; keep the subset decision open "
                "until the candidate scenarios are defined in-repo."
            ),
        }

    subset_rows = [row for row in scenario_rows if row.get("scenario_id") in subset_ids]
    matched_scenarios = sorted({str(row.get("scenario_id")) for row in subset_rows})
    if not matched_scenarios:
        return {
            "status": "rerun_required",
            "manifest_path": manifest_path_text,
            "subset_size": len(subset_ids),
            "matched_scenario_count": 0,
            "matched_scenarios": [],
            "comparison_planner_selection": None,
            "rank_correlation": None,
            "worth_adding": None,
            "recommendation": (
                "The current camera-ready campaign does not include the verified-simple candidate "
                "scenarios. Keep the subset as a debugging or promotion gate for now and run one "
                "bounded pilot before treating it as a calibration set."
            ),
        }

    selected_full_rows, selection_reason = _planner_selection_rows(scenario_rows, planner_index)
    selected_subset_rows = [
        row for row in selected_full_rows if row.get("scenario_id") in set(matched_scenarios)
    ]
    full_planner_rows = _planner_quality_rows(selected_full_rows)
    subset_planner_rows = _planner_quality_rows(selected_subset_rows)
    rank_correlation = _spearman_rank_correlation(full_planner_rows, subset_planner_rows)

    full_noise = _mean(
        _safe_float(row.get("seed_success_ci_half_width")) for row in selected_full_rows
    )
    subset_noise = _mean(
        _safe_float(row.get("seed_success_ci_half_width")) for row in selected_subset_rows
    )
    full_by_planner = {str(row.get("planner_key")): row for row in full_planner_rows}
    subset_by_planner = {str(row.get("planner_key")): row for row in subset_planner_rows}
    mean_abs_success_shift = _mean(
        abs(
            float(full_by_planner[key]["success_mean"])
            - float(subset_by_planner[key]["success_mean"])
        )
        for key in sorted(set(full_by_planner) & set(subset_by_planner))
        if full_by_planner[key].get("success_mean") is not None
        and subset_by_planner[key].get("success_mean") is not None
    )

    worth_adding: bool | None
    status: str
    recommendation: str
    if rank_correlation is None:
        worth_adding = None
        status = "insufficient_planner_overlap"
        recommendation = (
            "Verified-simple scenarios are present, but there are not enough comparable planner "
            "rows to judge ordering stability. Keep the subset as a gate until a broader pilot is "
            "available."
        )
    elif (
        rank_correlation >= 0.8
        and subset_noise is not None
        and full_noise is not None
        and subset_noise <= full_noise * 1.15
    ):
        worth_adding = True
        status = "candidate_supported"
        recommendation = (
            "The verified-simple subset broadly preserves planner ordering while keeping seed noise "
            "comparable to the full campaign. Use it as a calibration aid, not as a replacement "
            "benchmark."
        )
    elif rank_correlation < 0.5 or (
        subset_noise is not None and full_noise is not None and subset_noise > full_noise * 1.35
    ):
        worth_adding = False
        status = "candidate_noisy"
        recommendation = (
            "The verified-simple subset materially reorders planners or increases seed noise. Keep "
            "it as a debugging or promotion gate rather than a benchmark calibration set."
        )
    elif rank_correlation >= 0.8:
        worth_adding = None
        status = "mixed_signal"
        recommendation = (
            "The verified-simple subset preserves planner ordering, but this campaign does not "
            "include enough seed-variability evidence to show that the subset is comparably "
            "stable. Use it only as a secondary calibration view until a bounded pilot fills in "
            "the noise data."
        )
    else:
        worth_adding = None
        status = "mixed_signal"
        recommendation = (
            "The verified-simple subset shifts absolute planner scores, but the ordering and noise "
            "signal are mixed. Use it only as a secondary calibration view until a bounded pilot "
            "produces clearer evidence."
        )

    return {
        "status": status,
        "manifest_path": manifest_path_text,
        "subset_size": len(subset_ids),
        "matched_scenario_count": len(matched_scenarios),
        "matched_scenarios": matched_scenarios,
        "comparison_planner_selection": selection_reason,
        "rank_correlation": rank_correlation,
        "worth_adding": worth_adding,
        "mean_absolute_success_shift": mean_abs_success_shift,
        "full_seed_success_ci_half_width_mean": full_noise,
        "subset_seed_success_ci_half_width_mean": subset_noise,
        "full_planner_order": [row["planner_key"] for row in full_planner_rows],
        "subset_planner_order": [row["planner_key"] for row in subset_planner_rows],
        "recommendation": recommendation,
    }


def build_scenario_difficulty_analysis(  # noqa: C901, PLR0912, PLR0915
    *,
    planner_rows: Sequence[Mapping[str, Any]],
    scenario_breakdown_rows: Sequence[Mapping[str, Any]],
    seed_variability_payload: Mapping[str, Any] | None = None,
    preview_payload: Mapping[str, Any] | None = None,
    verified_simple_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build scenario-difficulty diagnostics from existing campaign artifacts.

    Returns:
        JSON-serializable scenario difficulty payload.
    """
    if not scenario_breakdown_rows:
        return {
            "schema_version": "benchmark-scenario-difficulty-analysis.v1",
            "status": "unavailable",
            "primary_proxy": {
                "name": "consensus_outcome_rank_v1",
                "eligible_planner_selection": "core benchmark-success planners",
            },
            "scenario_rows": [],
            "family_rows": [],
            "planner_family_rows": [],
            "planner_residual_rows": [],
            "planner_summary_rows": [],
            "verified_simple_assessment": _verified_simple_assessment(
                [],
                _planner_row_index(planner_rows),
                manifest_path=verified_simple_manifest_path,
            ),
            "findings": [],
        }

    planner_index = _planner_row_index(planner_rows)
    configured_consensus_planners = {
        planner_key for planner_key, row in planner_index.items() if _is_consensus_planner(row)
    }
    consensus_planners = set(configured_consensus_planners)
    consensus_selection = "core benchmark-success planners"
    findings: list[str] = []
    if not consensus_planners:
        consensus_planners = {
            str(row.get("planner_key"))
            for row in scenario_breakdown_rows
            if isinstance(row.get("planner_key"), str) and str(row.get("planner_key")).strip()
        }
        consensus_selection = "all planners (fallback: no eligible core set)"
        findings.append(
            "No eligible core benchmark-success planners were available; difficulty consensus fell "
            "back to all planners in the scenario breakdown."
        )

    seed_index = _build_seed_index(seed_variability_payload)
    metadata_lookup, preview_truncated = _preview_metadata_lookup(preview_payload)
    if preview_truncated:
        findings.append(
            "Preflight preview is truncated, so static metadata may be missing for some scenarios."
        )

    normalized_rows: list[dict[str, Any]] = []
    scenario_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_row in scenario_breakdown_rows:
        planner_key = str(raw_row.get("planner_key", "unknown"))
        scenario_id = str(raw_row.get("scenario_id", "unknown"))
        family = _scenario_family(raw_row)
        planner_meta = planner_index.get(planner_key, {})
        seed_row = seed_index.get((scenario_id, planner_key))
        metadata = metadata_lookup.get(scenario_id, {})
        normalized = {
            "planner_key": planner_key,
            "algo": str(raw_row.get("algo") or planner_meta.get("algo") or "unknown"),
            "planner_group": str(planner_meta.get("planner_group", "unknown")),
            "scenario_id": scenario_id,
            "scenario_family": family,
            "episodes": int(_safe_float(raw_row.get("episodes")) or 0),
            "success_mean": _safe_float(raw_row.get("success_mean")),
            "collisions_mean": _safe_float(raw_row.get("collisions_mean")),
            "near_misses_mean": _safe_float(raw_row.get("near_misses_mean")),
            "time_to_goal_norm_mean": _safe_float(raw_row.get("time_to_goal_norm_mean")),
            "snqi_mean": _safe_float(raw_row.get("snqi_mean")),
            "seed_success_ci_half_width": _seed_field(seed_row, "success", "ci_half_width"),
            "seed_success_cv": _seed_field(seed_row, "success", "cv"),
            "seed_time_to_goal_ci_half_width": _seed_field(
                seed_row, "time_to_goal_norm", "ci_half_width"
            ),
            "seed_time_to_goal_cv": _seed_field(seed_row, "time_to_goal_norm", "cv"),
            "seed_snqi_ci_half_width": _seed_field(seed_row, "snqi", "ci_half_width"),
            "seed_snqi_cv": _seed_field(seed_row, "snqi", "cv"),
            "seed_row_count": int(_safe_float(seed_row.get("seed_count")) or 0)
            if isinstance(seed_row, Mapping)
            else 0,
            "archetype": metadata.get("archetype"),
            "flow": metadata.get("flow"),
            "behavior": metadata.get("behavior"),
            "primary_capability": metadata.get("primary_capability"),
            "target_failure_mode": metadata.get("target_failure_mode"),
            "determinism": metadata.get("determinism"),
            "ped_density": _safe_float(metadata.get("ped_density")),
            "route_clearance_warning": bool(metadata.get("route_clearance_warning", False)),
            "route_clearance_scope": metadata.get("route_clearance_scope"),
            "min_clearance_margin_m": _safe_float(metadata.get("min_clearance_margin_m")),
        }
        normalized_rows.append(normalized)
        scenario_groups[scenario_id].append(normalized)

    consensus_rows: list[dict[str, Any]] = []
    for scenario_id, grouped_rows in scenario_groups.items():
        eligible_rows = [
            row for row in grouped_rows if row.get("planner_key") in consensus_planners
        ]
        if not eligible_rows:
            eligible_rows = list(grouped_rows)
        template = eligible_rows[0]
        consensus_row = {
            "scenario_id": scenario_id,
            "scenario_family": str(template.get("scenario_family", "unknown")),
            "eligible_planner_count": len(eligible_rows),
            "total_planner_count": len(grouped_rows),
            "consensus_success_mean": _mean(
                _safe_float(row.get("success_mean")) for row in eligible_rows
            ),
            "consensus_collisions_mean": _mean(
                _safe_float(row.get("collisions_mean")) for row in eligible_rows
            ),
            "consensus_near_misses_mean": _mean(
                _safe_float(row.get("near_misses_mean")) for row in eligible_rows
            ),
            "consensus_time_to_goal_norm_mean": _mean(
                _safe_float(row.get("time_to_goal_norm_mean")) for row in eligible_rows
            ),
            "consensus_snqi_mean": _mean(
                _safe_float(row.get("snqi_mean")) for row in eligible_rows
            ),
            "planner_success_range": _metric_range(
                _safe_float(row.get("success_mean")) for row in eligible_rows
            ),
            "planner_collision_range": _metric_range(
                _safe_float(row.get("collisions_mean")) for row in eligible_rows
            ),
            "planner_near_miss_range": _metric_range(
                _safe_float(row.get("near_misses_mean")) for row in eligible_rows
            ),
            "planner_time_to_goal_range": _metric_range(
                _safe_float(row.get("time_to_goal_norm_mean")) for row in eligible_rows
            ),
            "seed_success_ci_half_width_mean": _mean(
                _safe_float(row.get("seed_success_ci_half_width")) for row in eligible_rows
            ),
            "seed_success_cv_mean": _mean(
                _safe_float(row.get("seed_success_cv")) for row in eligible_rows
            ),
            "seed_time_to_goal_ci_half_width_mean": _mean(
                _safe_float(row.get("seed_time_to_goal_ci_half_width")) for row in eligible_rows
            ),
            "seed_time_to_goal_cv_mean": _mean(
                _safe_float(row.get("seed_time_to_goal_cv")) for row in eligible_rows
            ),
            "seed_snqi_ci_half_width_mean": _mean(
                _safe_float(row.get("seed_snqi_ci_half_width")) for row in eligible_rows
            ),
            "seed_snqi_cv_mean": _mean(
                _safe_float(row.get("seed_snqi_cv")) for row in eligible_rows
            ),
            "archetype": template.get("archetype"),
            "flow": template.get("flow"),
            "behavior": template.get("behavior"),
            "primary_capability": template.get("primary_capability"),
            "target_failure_mode": template.get("target_failure_mode"),
            "determinism": template.get("determinism"),
            "ped_density": _safe_float(template.get("ped_density")),
            "route_clearance_warning": bool(template.get("route_clearance_warning", False)),
            "route_clearance_scope": template.get("route_clearance_scope"),
            "min_clearance_margin_m": _safe_float(template.get("min_clearance_margin_m")),
        }
        consensus_rows.append(consensus_row)

    component_ranks: dict[str, dict[str, float]] = {}
    for metric_name, higher_is_harder, _weight in _PRIMARY_PROXY_METRICS:
        values_by_scenario = {}
        field_name = f"consensus_{metric_name}"
        for row in consensus_rows:
            value = _safe_float(row.get(field_name))
            if value is not None:
                values_by_scenario[str(row.get("scenario_id"))] = value
        component_ranks[metric_name] = _normalized_ranks(
            values_by_scenario,
            higher_is_harder=higher_is_harder,
        )

    snqi_rank = _normalized_ranks(
        {
            str(row.get("scenario_id")): value
            for row in consensus_rows
            if (value := _safe_float(row.get("consensus_snqi_mean"))) is not None
        },
        higher_is_harder=False,
    )

    for row in consensus_rows:
        scenario_id = str(row.get("scenario_id"))
        difficulty_score, components = _difficulty_weighted_score(scenario_id, component_ranks)
        row["difficulty_score"] = difficulty_score
        row["difficulty_components"] = components
        row["supporting_snqi_rank"] = snqi_rank.get(scenario_id)

    ranked_rows = sorted(
        consensus_rows,
        key=lambda row: (
            -(row.get("difficulty_score") if row.get("difficulty_score") is not None else -1.0),
            str(row.get("scenario_id", "")),
        ),
    )
    for index, row in enumerate(ranked_rows, start=1):
        row["difficulty_rank"] = index
    scenario_index = {str(row.get("scenario_id")): row for row in ranked_rows}

    residual_rows: list[dict[str, Any]] = []
    for row in normalized_rows:
        scenario_id = str(row.get("scenario_id"))
        consensus = scenario_index.get(scenario_id)
        if consensus is None:
            continue
        success_range = _safe_float(consensus.get("planner_success_range")) or 0.0
        collision_range = _safe_float(consensus.get("planner_collision_range")) or 0.0
        near_miss_range = _safe_float(consensus.get("planner_near_miss_range")) or 0.0
        time_to_goal_range = _safe_float(consensus.get("planner_time_to_goal_range")) or 0.0
        components: list[float] = []
        residual = {
            "planner_key": row.get("planner_key"),
            "algo": row.get("algo"),
            "planner_group": row.get("planner_group"),
            "scenario_id": scenario_id,
            "scenario_family": row.get("scenario_family"),
            "difficulty_score": consensus.get("difficulty_score"),
            "difficulty_rank": consensus.get("difficulty_rank"),
            "success_residual": None,
            "collisions_residual": None,
            "near_misses_residual": None,
            "time_to_goal_norm_residual": None,
            "snqi_residual": None,
        }

        success_value = _safe_float(row.get("success_mean"))
        success_consensus = _safe_float(consensus.get("consensus_success_mean"))
        if success_value is not None and success_consensus is not None:
            residual["success_residual"] = float(success_value - success_consensus)
            components.append((success_consensus - success_value) / max(success_range, 1e-6))

        collision_value = _safe_float(row.get("collisions_mean"))
        collision_consensus = _safe_float(consensus.get("consensus_collisions_mean"))
        if collision_value is not None and collision_consensus is not None:
            residual["collisions_residual"] = float(collision_value - collision_consensus)
            components.append((collision_value - collision_consensus) / max(collision_range, 1e-6))

        near_miss_value = _safe_float(row.get("near_misses_mean"))
        near_miss_consensus = _safe_float(consensus.get("consensus_near_misses_mean"))
        if near_miss_value is not None and near_miss_consensus is not None:
            residual["near_misses_residual"] = float(near_miss_value - near_miss_consensus)
            components.append((near_miss_value - near_miss_consensus) / max(near_miss_range, 1e-6))

        time_value = _safe_float(row.get("time_to_goal_norm_mean"))
        time_consensus = _safe_float(consensus.get("consensus_time_to_goal_norm_mean"))
        if time_value is not None and time_consensus is not None:
            residual["time_to_goal_norm_residual"] = float(time_value - time_consensus)
            components.append((time_value - time_consensus) / max(time_to_goal_range, 1e-6))

        snqi_value = _safe_float(row.get("snqi_mean"))
        snqi_consensus = _safe_float(consensus.get("consensus_snqi_mean"))
        if snqi_value is not None and snqi_consensus is not None:
            residual["snqi_residual"] = float(snqi_value - snqi_consensus)

        residual["residual_score"] = _mean(components)
        residual_rows.append(residual)

    residual_threshold = _percentile(
        [
            float(row["residual_score"])
            for row in residual_rows
            if row.get("residual_score") is not None
        ],
        0.75,
    )
    easy_difficulty_limit = _percentile(
        [
            float(row["difficulty_score"])
            for row in ranked_rows
            if row.get("difficulty_score") is not None
        ],
        0.50,
    )
    for row in residual_rows:
        residual_score = _safe_float(row.get("residual_score"))
        row["easy_scenario_underperformance"] = bool(
            residual_score is not None
            and residual_threshold is not None
            and residual_threshold > 0.0
            and residual_score > 0.0
            and residual_score >= residual_threshold
            and row.get("difficulty_score") is not None
            and easy_difficulty_limit is not None
            and float(row["difficulty_score"]) <= easy_difficulty_limit
        )

    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ranked_rows:
        family_groups[str(row.get("scenario_family", "unknown"))].append(row)
    family_rows: list[dict[str, Any]] = []
    for family, rows in sorted(family_groups.items()):
        family_rows.append(
            {
                "scenario_family": family,
                "scenario_count": len(rows),
                "difficulty_score_mean": _mean(
                    _safe_float(row.get("difficulty_score")) for row in rows
                ),
                "consensus_success_mean": _mean(
                    _safe_float(row.get("consensus_success_mean")) for row in rows
                ),
                "consensus_collisions_mean": _mean(
                    _safe_float(row.get("consensus_collisions_mean")) for row in rows
                ),
                "consensus_near_misses_mean": _mean(
                    _safe_float(row.get("consensus_near_misses_mean")) for row in rows
                ),
                "consensus_time_to_goal_norm_mean": _mean(
                    _safe_float(row.get("consensus_time_to_goal_norm_mean")) for row in rows
                ),
                "seed_success_ci_half_width_mean": _mean(
                    _safe_float(row.get("seed_success_ci_half_width_mean")) for row in rows
                ),
                "seed_time_to_goal_ci_half_width_mean": _mean(
                    _safe_float(row.get("seed_time_to_goal_ci_half_width_mean")) for row in rows
                ),
                "hardest_scenarios": [
                    scenario_row["scenario_id"]
                    for scenario_row in sorted(
                        rows,
                        key=lambda item: (
                            -(
                                item.get("difficulty_score")
                                if item.get("difficulty_score") is not None
                                else -1.0
                            ),
                            str(item.get("scenario_id", "")),
                        ),
                    )[:3]
                ],
            }
        )

    planner_summary_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    planner_family_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in residual_rows:
        planner_key = str(row.get("planner_key", "unknown"))
        planner_summary_groups[planner_key].append(row)
        planner_family_groups[(planner_key, str(row.get("scenario_family", "unknown")))].append(row)

    planner_summary_rows: list[dict[str, Any]] = []
    for planner_key, rows in sorted(planner_summary_groups.items()):
        template = rows[0]
        worst_rows = sorted(
            [row for row in rows if row.get("residual_score") is not None],
            key=lambda item: (
                -(item.get("residual_score") if item.get("residual_score") is not None else -1.0),
                str(item.get("scenario_id", "")),
            ),
        )
        planner_summary_rows.append(
            {
                "planner_key": planner_key,
                "algo": template.get("algo"),
                "planner_group": template.get("planner_group"),
                "scenario_count": len(rows),
                "mean_residual_score": _mean(
                    _safe_float(row.get("residual_score")) for row in rows
                ),
                "max_residual_score": _max(_safe_float(row.get("residual_score")) for row in rows),
                "easy_scenario_underperformance_count": sum(
                    1 for row in rows if row.get("easy_scenario_underperformance")
                ),
                "worst_scenarios": [row.get("scenario_id") for row in worst_rows[:3]],
            }
        )

    family_index = {str(row.get("scenario_family")): row for row in family_rows}
    planner_family_rows: list[dict[str, Any]] = []
    for (planner_key, family), rows in sorted(planner_family_groups.items()):
        template = rows[0]
        family_consensus = family_index.get(family, {})
        planner_family_rows.append(
            {
                "planner_key": planner_key,
                "algo": template.get("algo"),
                "planner_group": template.get("planner_group"),
                "scenario_family": family,
                "scenario_count": len(rows),
                "mean_residual_score": _mean(
                    _safe_float(row.get("residual_score")) for row in rows
                ),
                "easy_scenario_underperformance_count": sum(
                    1 for row in rows if row.get("easy_scenario_underperformance")
                ),
                "family_difficulty_score_mean": _safe_float(
                    family_consensus.get("difficulty_score_mean")
                ),
            }
        )

    verified_simple_ids, _manifest_path_text = _load_verified_simple_ids(
        verified_simple_manifest_path
    )
    for row in ranked_rows:
        row["verified_simple_candidate"] = str(row.get("scenario_id")) in verified_simple_ids

    return {
        "schema_version": "benchmark-scenario-difficulty-analysis.v1",
        "status": "ok",
        "primary_proxy": {
            "name": "consensus_outcome_rank_v1",
            "description": (
                "Weighted consensus score across core benchmark-success planners using success, "
                "collisions, near-misses, and normalized time-to-goal."
            ),
            "eligible_planner_selection": consensus_selection,
            "eligible_planner_count": len(consensus_planners),
            "metric_weights": {
                metric_name: weight
                for metric_name, _higher_is_harder, weight in _PRIMARY_PROXY_METRICS
            },
            "supporting_metric": "snqi_mean",
        },
        "scenario_rows": ranked_rows,
        "family_rows": family_rows,
        "planner_family_rows": planner_family_rows,
        "planner_residual_rows": residual_rows,
        "planner_summary_rows": planner_summary_rows,
        "verified_simple_assessment": _verified_simple_assessment(
            normalized_rows,
            planner_index,
            manifest_path=verified_simple_manifest_path,
        ),
        "findings": findings,
    }


__all__ = ["build_scenario_difficulty_analysis"]
