"""SNQI calibration and normalization robustness helpers."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.snqi.campaign_contract import (
    SnqiContractThresholds,
    compute_component_correlations,
    compute_component_dominance,
    compute_planner_snqi_ordering,
    compute_weight_sensitivity,
    evaluate_snqi_contract,
    sanitize_baseline_stats,
    spearman_correlation,
)
from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES, compute_snqi

_NORMALIZED_METRIC_NAMES = (
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "force_exceed_events",
    "jerk_mean",
)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _quantile(values: Sequence[float], q: float) -> float:
    """Return a deterministic linear-interpolated quantile."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(value) for value in values)
    clamped = max(0.0, min(1.0, float(q)))
    position = (len(ordered) - 1) * clamped
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def load_episode_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load episode records from a JSONL file.

    Returns:
        Decoded episode dictionaries.
    """
    episodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                episodes.append(payload)
    return episodes


def derive_planner_rows_from_episodes(
    episodes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build minimal planner rows needed for contract evaluation from episodes.

    Returns:
        Planner aggregate rows with means for contract target metrics.
    """
    grouped: dict[str, dict[str, Any]] = {}
    metric_names = ("success", "collisions", "near_misses", "comfort_exposure")
    for episode in episodes:
        planner_key = str(episode.get("planner_key", "unknown"))
        kinematics = str(episode.get("kinematics", "unknown"))
        key = f"{planner_key}::{kinematics}"
        bucket = grouped.setdefault(
            key,
            {
                "planner_key": planner_key,
                "kinematics": kinematics,
                "count": 0,
                **{f"{name}_sum": 0.0 for name in metric_names},
            },
        )
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        bucket["count"] = int(bucket["count"]) + 1
        for metric_name in metric_names:
            value = metrics.get(metric_name)
            if _is_finite(value):
                bucket[f"{metric_name}_sum"] = float(bucket[f"{metric_name}_sum"]) + float(value)

    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        count = int(bucket.get("count", 0))
        if count <= 0:
            continue
        rows.append(
            {
                "planner_key": str(bucket["planner_key"]),
                "kinematics": str(bucket["kinematics"]),
                "success_mean": float(bucket["success_sum"]) / count,
                "collisions_mean": float(bucket["collisions_sum"]) / count,
                "near_misses_mean": float(bucket["near_misses_sum"]) / count,
                "comfort_exposure_mean": float(bucket["comfort_exposure_sum"]) / count,
            }
        )
    return rows


def normalization_anchor_variants(
    episodes: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Return v3 and dataset-derived anchor variants for normalized SNQI terms."""
    values_by_metric: dict[str, list[float]] = {name: [] for name in _NORMALIZED_METRIC_NAMES}
    for episode in episodes:
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for metric_name in _NORMALIZED_METRIC_NAMES:
            value = metrics.get(metric_name)
            if _is_finite(value):
                values_by_metric[metric_name].append(float(value))

    raw_variants: dict[str, dict[str, dict[str, float]]] = {
        "v3_fixed": {
            key: {"med": float(value.get("med", 0.0)), "p95": float(value.get("p95", 1.0))}
            for key, value in baseline.items()
            if isinstance(value, Mapping)
        },
        "dataset_median_p95": {},
        "dataset_median_p90": {},
        "dataset_median_max": {},
        "dataset_p25_p75": {},
    }
    for metric_name, values in values_by_metric.items():
        if not values:
            continue
        raw_variants["dataset_median_p95"][metric_name] = {
            "med": _quantile(values, 0.50),
            "p95": _quantile(values, 0.95),
        }
        raw_variants["dataset_median_p90"][metric_name] = {
            "med": _quantile(values, 0.50),
            "p95": _quantile(values, 0.90),
        }
        raw_variants["dataset_median_max"][metric_name] = {
            "med": _quantile(values, 0.50),
            "p95": max(values),
        }
        raw_variants["dataset_p25_p75"][metric_name] = {
            "med": _quantile(values, 0.25),
            "p95": _quantile(values, 0.75),
        }

    variants: dict[str, dict[str, dict[str, float]]] = {}
    for name, raw in raw_variants.items():
        sanitized, _warnings = sanitize_baseline_stats(raw, metric_names=_NORMALIZED_METRIC_NAMES)
        variants[name] = sanitized
    return variants


def weight_variants(
    weights: Mapping[str, float],
    *,
    epsilon: float = 0.15,
) -> dict[str, dict[str, float]]:
    """Return deterministic simplex-preserving local and ablation weight variants."""
    base = {name: float(weights.get(name, 0.0)) for name in WEIGHT_NAMES}
    total = sum(base.values()) or 1.0
    variants: dict[str, dict[str, float]] = {"v3_fixed": dict(base)}

    for name in WEIGHT_NAMES:
        for direction, factor in (("down", 1.0 - epsilon), ("up", 1.0 + epsilon)):
            candidate = dict(base)
            candidate[name] = max(0.0, candidate[name] * factor)
            candidate_total = sum(candidate.values()) or 1.0
            variants[f"local_{name}_{direction}"] = {
                key: value * total / candidate_total for key, value in candidate.items()
            }

    variants["uniform_simplex"] = dict.fromkeys(WEIGHT_NAMES, total / len(WEIGHT_NAMES))
    no_jerk = dict(base)
    no_jerk["w_jerk"] = 0.0
    no_jerk_total = sum(no_jerk.values()) or 1.0
    variants["component_subset_no_jerk"] = {
        key: value * total / no_jerk_total for key, value in no_jerk.items()
    }
    return variants


def _score_episodes(
    episodes: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
) -> list[tuple[Mapping[str, Any], float]]:
    scored: list[tuple[Mapping[str, Any], float]] = []
    for episode in episodes:
        metrics = episode.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        score = compute_snqi(metrics, weights, baseline)
        if _is_finite(score):
            scored.append((episode, float(score)))
    return scored


def _planner_rank_map(ordering: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        f"{row.get('planner_key', 'unknown')}::{row.get('kinematics', 'unknown')}": int(
            row.get("rank", index)
        )
        for index, row in enumerate(ordering, start=1)
    }


def _rank_stability(
    baseline_ordering: Sequence[Mapping[str, Any]],
    variant_ordering: Sequence[Mapping[str, Any]],
) -> tuple[float, float, bool]:
    base_ranks = _planner_rank_map(baseline_ordering)
    variant_ranks = _planner_rank_map(variant_ordering)
    shared = [key for key in base_ranks if key in variant_ranks]
    if len(shared) < 2:
        return 1.0, 0.0, False
    corr = spearman_correlation(
        [float(base_ranks[key]) for key in shared],
        [float(variant_ranks[key]) for key in shared],
    )
    shifts = [abs(float(base_ranks[key] - variant_ranks[key])) for key in shared]
    base_order = sorted(base_ranks, key=base_ranks.get)
    variant_order = sorted(variant_ranks, key=variant_ranks.get)
    return float(corr), float(sum(shifts) / len(shifts)), base_order != variant_order


def _alignment_summary(
    correlations: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int, float]:
    aligned = 0
    variable = 0
    abs_correlations: list[float] = []
    for row in correlations.values():
        if bool(row.get("variable")):
            variable += 1
            if row.get("aligned_with_expected_direction") is True:
                aligned += 1
            spearman = row.get("spearman")
            if _is_finite(spearman):
                abs_correlations.append(abs(float(spearman)))
    mean_abs = sum(abs_correlations) / len(abs_correlations) if abs_correlations else 0.0
    return aligned, variable, mean_abs


def analyze_snqi_calibration(
    episodes: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    baseline: Mapping[str, Mapping[str, float]],
    planner_rows: Sequence[Mapping[str, Any]] | None = None,
    epsilon: float = 0.15,
) -> dict[str, Any]:
    """Compare SNQI v3 with local weight and normalization variants.

    Returns:
        JSON-serializable calibration analysis payload.
    """
    if not episodes:
        raise ValueError("SNQI calibration analysis requires at least one episode.")
    effective_planner_rows = list(planner_rows or derive_planner_rows_from_episodes(episodes))
    base_weights = {name: float(weights.get(name, 0.0)) for name in WEIGHT_NAMES}
    anchor_variants = normalization_anchor_variants(episodes, baseline)
    weight_grid = weight_variants(base_weights, epsilon=epsilon)
    thresholds = SnqiContractThresholds()

    baseline_scores = _score_episodes(
        episodes,
        weights=base_weights,
        baseline=anchor_variants["v3_fixed"],
    )
    baseline_score_values = [score for _episode, score in baseline_scores]
    baseline_ordering = compute_planner_snqi_ordering(
        episodes,
        weights=base_weights,
        baseline=anchor_variants["v3_fixed"],
    )

    rows: list[dict[str, Any]] = []
    variant_specs: list[tuple[str, str, Mapping[str, float], Mapping[str, Mapping[str, float]]]] = [
        ("baseline", "v3_fixed", base_weights, anchor_variants["v3_fixed"])
    ]
    variant_specs.extend(
        ("weight", name, variant_weights, anchor_variants["v3_fixed"])
        for name, variant_weights in weight_grid.items()
        if name != "v3_fixed"
    )
    variant_specs.extend(
        ("anchor", name, base_weights, variant_baseline)
        for name, variant_baseline in anchor_variants.items()
        if name != "v3_fixed"
    )

    for variant_type, name, variant_weights, variant_baseline in variant_specs:
        scored = _score_episodes(episodes, weights=variant_weights, baseline=variant_baseline)
        scores = [score for _episode, score in scored]
        ordering = compute_planner_snqi_ordering(
            episodes,
            weights=variant_weights,
            baseline=variant_baseline,
        )
        planner_corr, mean_shift, order_changed = _rank_stability(baseline_ordering, ordering)
        episode_corr = (
            spearman_correlation(baseline_score_values, scores)
            if len(baseline_score_values) == len(scores) and len(scores) >= 2
            else 1.0
        )
        correlations = compute_component_correlations(
            episodes,
            weights=variant_weights,
            baseline=variant_baseline,
        )
        aligned, variable, mean_abs_component_corr = _alignment_summary(correlations)
        dominance = compute_component_dominance(
            episodes,
            weights=variant_weights,
            baseline=variant_baseline,
        )
        dominant_component = "none"
        dominant_value = 0.0
        if dominance:
            dominant_component, dominant_value = max(dominance.items(), key=lambda item: item[1])
        evaluation = evaluate_snqi_contract(
            effective_planner_rows,
            episodes,
            weights=variant_weights,
            baseline=variant_baseline,
            thresholds=thresholds,
        )
        rows.append(
            {
                "variant": name,
                "variant_type": variant_type,
                "episode_count": len(scores),
                "planner_count": len(ordering),
                "mean_snqi": sum(scores) / len(scores) if scores else 0.0,
                "std_snqi": _stddev(scores),
                "score_range": (max(scores) - min(scores)) if scores else 0.0,
                "episode_rank_correlation_vs_v3": episode_corr,
                "planner_rank_correlation_vs_v3": planner_corr,
                "mean_abs_planner_rank_shift_vs_v3": mean_shift,
                "planner_order_changed_vs_v3": order_changed,
                "aligned_metric_count": aligned,
                "variable_metric_count": variable,
                "alignment_fraction": aligned / variable if variable else 0.0,
                "mean_abs_component_spearman": mean_abs_component_corr,
                "dominant_component": dominant_component,
                "dominant_component_mean_abs": float(dominant_value),
                "contract_status": evaluation.status,
                "rank_alignment_spearman": evaluation.rank_alignment_spearman,
                "outcome_separation": evaluation.outcome_separation,
            }
        )

    sensitivity = _summarize_sensitivity(rows)
    recommendation = _recommend(rows, sensitivity)
    return {
        "schema_version": "snqi-calibration-analysis.v1",
        "weights_epsilon": float(epsilon),
        "episodes": len(episodes),
        "planners": len(baseline_ordering),
        "baseline_ordering": baseline_ordering,
        "variants": rows,
        "weight_ablation": compute_weight_sensitivity(
            episodes,
            weights=base_weights,
            baseline=anchor_variants["v3_fixed"],
        ),
        "sensitivity_summary": sensitivity,
        "recommendation": recommendation,
    }


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _summarize_sensitivity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    weight_rows = [
        row
        for row in rows
        if row.get("variant_type") == "weight" and str(row.get("variant")).startswith("local_")
    ]
    anchor_rows = [row for row in rows if row.get("variant_type") == "anchor"]
    all_nonbaseline = [row for row in rows if row.get("variant_type") != "baseline"]

    def _min(rows_in: Sequence[Mapping[str, Any]], key: str, default: float = 1.0) -> float:
        values = [float(row.get(key, default)) for row in rows_in if _is_finite(row.get(key))]
        return min(values) if values else default

    return {
        "local_weight_min_planner_rank_correlation": _min(
            weight_rows, "planner_rank_correlation_vs_v3"
        ),
        "local_weight_min_episode_rank_correlation": _min(
            weight_rows, "episode_rank_correlation_vs_v3"
        ),
        "anchor_min_planner_rank_correlation": _min(anchor_rows, "planner_rank_correlation_vs_v3"),
        "anchor_min_episode_rank_correlation": _min(anchor_rows, "episode_rank_correlation_vs_v3"),
        "nonbaseline_order_change_count": sum(
            1 for row in all_nonbaseline if bool(row.get("planner_order_changed_vs_v3"))
        ),
        "anchor_order_change_count": sum(
            1 for row in anchor_rows if bool(row.get("planner_order_changed_vs_v3"))
        ),
        "local_weight_order_change_count": sum(
            1 for row in weight_rows if bool(row.get("planner_order_changed_vs_v3"))
        ),
    }


def _recommend(rows: Sequence[Mapping[str, Any]], sensitivity: Mapping[str, Any]) -> dict[str, Any]:
    baseline = next(row for row in rows if row.get("variant_type") == "baseline")
    baseline_alignment = float(baseline.get("alignment_fraction", 0.0))
    baseline_dominance = float(baseline.get("dominant_component_mean_abs", 0.0))

    stable_enough = (
        float(sensitivity.get("local_weight_min_planner_rank_correlation", 1.0)) >= 0.8
        and float(sensitivity.get("anchor_min_planner_rank_correlation", 1.0)) >= 0.8
        and baseline_alignment >= 0.6
    )
    candidates = [
        row
        for row in rows
        if row.get("variant_type") != "baseline"
        and float(row.get("planner_rank_correlation_vs_v3", 0.0)) >= 0.9
        and float(row.get("alignment_fraction", 0.0)) >= baseline_alignment + 0.10
        and float(row.get("dominant_component_mean_abs", 0.0))
        <= max(baseline_dominance * 1.05, baseline_dominance + 1e-9)
    ]
    if not stable_enough:
        decision = "demote_snqi_further"
        rationale = (
            "v3 is locally sensitive enough that SNQI should remain appendix/supporting material "
            "until a revised contract is proven on the frozen bundle set."
        )
    elif candidates:
        decision = "propose_candidate_v4_contract"
        rationale = (
            "at least one variant improves component alignment while preserving v3 planner ranks; "
            "treat it as a future v4 candidate, not a paper-bundle replacement."
        )
    else:
        decision = "keep_v3_fixed"
        rationale = (
            "local weight and anchor variants did not clear the improvement threshold strongly "
            "enough to justify replacing the fixed v3 contract."
        )
    return {
        "decision": decision,
        "rationale": rationale,
        "baseline_alignment_fraction": baseline_alignment,
        "strictly_better_variant_candidates": [str(row.get("variant")) for row in candidates],
        "thresholds": {
            "demote_if_min_planner_rank_correlation_below": 0.8,
            "demote_if_alignment_fraction_below": 0.6,
            "v4_candidate_alignment_gain": 0.10,
            "v4_candidate_min_planner_rank_correlation": 0.9,
        },
    }


def write_calibration_csv(path: Path, payload: Mapping[str, Any]) -> None:
    """Write variant-level calibration rows as CSV."""
    fieldnames = [
        "variant",
        "variant_type",
        "episode_count",
        "planner_count",
        "mean_snqi",
        "std_snqi",
        "score_range",
        "episode_rank_correlation_vs_v3",
        "planner_rank_correlation_vs_v3",
        "mean_abs_planner_rank_shift_vs_v3",
        "planner_order_changed_vs_v3",
        "aligned_metric_count",
        "variable_metric_count",
        "alignment_fraction",
        "mean_abs_component_spearman",
        "dominant_component",
        "dominant_component_mean_abs",
        "contract_status",
        "rank_alignment_spearman",
        "outcome_separation",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload.get("variants", []):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_calibration_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a concise manuscript-facing Markdown summary."""
    recommendation = payload.get("recommendation", {})
    sensitivity = payload.get("sensitivity_summary", {})
    rows = payload.get("variants", [])
    lines = [
        "# SNQI Calibration Analysis",
        "",
        f"- Recommendation: `{recommendation.get('decision', 'unknown')}`",
        f"- Rationale: {recommendation.get('rationale', 'n/a')}",
        f"- Episodes: `{payload.get('episodes', 0)}`",
        f"- Planners: `{payload.get('planners', 0)}`",
        f"- Local weight min planner rank correlation: `{float(sensitivity.get('local_weight_min_planner_rank_correlation', 0.0)):.6f}`",
        f"- Anchor min planner rank correlation: `{float(sensitivity.get('anchor_min_planner_rank_correlation', 0.0)):.6f}`",
        f"- Non-baseline order-change count: `{int(sensitivity.get('nonbaseline_order_change_count', 0))}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Type | Planner rho | Episode rho | Alignment | Order changed | Dominant component |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    if isinstance(rows, list):
        for row in rows:
            lines.append(
                "| {variant} | {variant_type} | {planner:.6f} | {episode:.6f} | {alignment:.3f} | {changed} | {dominant} |".format(
                    variant=row.get("variant", "unknown"),
                    variant_type=row.get("variant_type", "unknown"),
                    planner=float(row.get("planner_rank_correlation_vs_v3", 0.0)),
                    episode=float(row.get("episode_rank_correlation_vs_v3", 0.0)),
                    alignment=float(row.get("alignment_fraction", 0.0)),
                    changed="yes" if row.get("planner_order_changed_vs_v3") else "no",
                    dominant=row.get("dominant_component", "unknown"),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
