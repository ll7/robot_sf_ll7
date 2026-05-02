#!/usr/bin/env python3
"""Compare paper-matrix benchmark campaigns with different seed schedules."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SEED_METRICS: tuple[str, ...] = (
    "success",
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "snqi",
)
_LOWER_IS_BETTER = {
    "collisions",
    "near_misses",
    "time_to_goal_norm",
    "comfort_exposure",
    "jerk",
    "jerk_mean",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{digits}f}"


def _metric_direction(metric: str) -> str:
    return "lower" if metric in _LOWER_IS_BETTER else "higher"


def _metric_summary(row: dict[str, Any], metric: str) -> dict[str, Any]:
    summary = row.get("summary")
    if not isinstance(summary, dict):
        return {}
    metric_summary = summary.get(metric)
    return dict(metric_summary) if isinstance(metric_summary, dict) else {}


def _metric_mean(row: dict[str, Any], metric: str) -> float | None:
    return _safe_float(_metric_summary(row, metric).get("mean"))


def _metric_ci(row: dict[str, Any], metric: str) -> tuple[float | None, float | None]:
    summary = _metric_summary(row, metric)
    low = _safe_float(summary.get("ci_low"))
    high = _safe_float(summary.get("ci_high"))
    return low, high


def _ci_width(row: dict[str, Any], metric: str) -> float | None:
    low, high = _metric_ci(row, metric)
    if low is None or high is None:
        return None
    return float(max(0.0, high - low))


def _ci_overlap(
    base_row: dict[str, Any], candidate_row: dict[str, Any], metric: str
) -> bool | None:
    base_low, base_high = _metric_ci(base_row, metric)
    candidate_low, candidate_high = _metric_ci(candidate_row, metric)
    if None in (base_low, base_high, candidate_low, candidate_high):
        return None
    return bool(base_low <= candidate_high and candidate_low <= base_high)


def _seed_row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("scenario_id", "unknown")),
        str(row.get("planner_key", "unknown")),
        str(row.get("kinematics", "unknown")),
    )


def _planner_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("planner_key", "unknown")),
        str(row.get("kinematics", "unknown")),
    )


def _load_campaign(campaign_root: Path) -> dict[str, Any]:
    summary = _read_json(campaign_root / "reports" / "campaign_summary.json")
    seed_variability = _read_json(campaign_root / "reports" / "seed_variability_by_scenario.json")
    matrix_path = campaign_root / "reports" / "matrix_summary.json"
    matrix_summary = _read_json(matrix_path) if matrix_path.exists() else {}
    return {
        "root": campaign_root,
        "summary": summary,
        "seed_variability": seed_variability,
        "matrix_summary": matrix_summary,
    }


def _campaign_metadata(campaign: dict[str, Any]) -> dict[str, Any]:
    summary = dict(campaign.get("summary") or {})
    campaign_block = dict(summary.get("campaign") or {})
    matrix_rows = (
        (campaign.get("matrix_summary") or {}).get("rows")
        if isinstance(campaign.get("matrix_summary"), dict)
        else None
    )
    first_matrix_row = matrix_rows[0] if isinstance(matrix_rows, list) and matrix_rows else {}
    return {
        "campaign_id": campaign_block.get("campaign_id", campaign["root"].name),
        "campaign_root": str(campaign["root"]),
        "config_name": campaign_block.get("name"),
        "scenario_matrix": campaign_block.get("scenario_matrix"),
        "scenario_matrix_hash": campaign_block.get("scenario_matrix_hash"),
        "paper_profile_version": campaign_block.get("paper_profile_version"),
        "paper_interpretation_profile": campaign_block.get("paper_interpretation_profile"),
        "total_episodes": campaign_block.get("total_episodes"),
        "total_runs": campaign_block.get("total_runs"),
        "successful_runs": campaign_block.get("successful_runs"),
        "benchmark_success": campaign_block.get("benchmark_success"),
        "runtime_sec": campaign_block.get("runtime_sec"),
        "git_hash": campaign_block.get("git_hash"),
        "resolved_seeds": first_matrix_row.get("resolved_seeds"),
        "repeats": first_matrix_row.get("repeats"),
        "seed_policy_mode": first_matrix_row.get("seed_policy.mode"),
        "seed_policy_seed_set": first_matrix_row.get("seed_policy.seed_set"),
    }


def _seed_rows_by_key(campaign: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = campaign.get("seed_variability")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            out[_seed_row_key(row)] = row
    return out


def _planner_rows_by_key(campaign: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    summary = campaign.get("summary")
    rows = summary.get("planner_rows") if isinstance(summary, dict) else None
    if not isinstance(rows, list):
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict):
            out[_planner_key(row)] = row
    return out


def _relative_change(base: float | None, candidate: float | None) -> float | None:
    if base is None or candidate is None or abs(base) <= 1e-12:
        return None
    return float((candidate - base) / abs(base))


def _build_interval_width_rows(
    base_rows: dict[tuple[str, str, str], dict[str, Any]],
    candidate_rows: dict[tuple[str, str, str], dict[str, Any]],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base_rows) & set(candidate_rows)):
        base_row = base_rows[key]
        candidate_row = candidate_rows[key]
        for metric in metrics:
            base_width = _ci_width(base_row, metric)
            candidate_width = _ci_width(candidate_row, metric)
            base_mean = _metric_mean(base_row, metric)
            candidate_mean = _metric_mean(candidate_row, metric)
            rows.append(
                {
                    "scenario_id": key[0],
                    "planner_key": key[1],
                    "kinematics": key[2],
                    "metric": metric,
                    "base_seed_count": base_row.get("seed_count"),
                    "candidate_seed_count": candidate_row.get("seed_count"),
                    "base_mean": base_mean,
                    "candidate_mean": candidate_mean,
                    "mean_delta": (
                        None
                        if base_mean is None or candidate_mean is None
                        else candidate_mean - base_mean
                    ),
                    "base_ci_width": base_width,
                    "candidate_ci_width": candidate_width,
                    "ci_width_delta": (
                        None
                        if base_width is None or candidate_width is None
                        else candidate_width - base_width
                    ),
                    "ci_width_relative_change": _relative_change(base_width, candidate_width),
                    "ci_width_reduction_fraction": (
                        None
                        if base_width is None or candidate_width is None or abs(base_width) <= 1e-12
                        else (base_width - candidate_width) / base_width
                    ),
                    "ci_overlap": _ci_overlap(base_row, candidate_row, metric),
                }
            )
    return rows


def _summarize_interval_widths(
    interval_rows: list[dict[str, Any]],
    metrics: tuple[str, ...],
    *,
    target_reduction: float,
) -> dict[str, Any]:
    by_metric: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        metric_rows = [row for row in interval_rows if row.get("metric") == metric]
        base_widths = [
            width
            for row in metric_rows
            if (width := _safe_float(row.get("base_ci_width"))) is not None
        ]
        candidate_widths = [
            width
            for row in metric_rows
            if (width := _safe_float(row.get("candidate_ci_width"))) is not None
        ]
        base_mean_width = _mean(base_widths)
        candidate_mean_width = _mean(candidate_widths)
        reduction = (
            None
            if base_mean_width is None
            or candidate_mean_width is None
            or abs(base_mean_width) <= 1e-12
            else (base_mean_width - candidate_mean_width) / base_mean_width
        )
        target_met = (
            bool(reduction >= target_reduction)
            if reduction is not None
            else bool(base_mean_width == 0.0 and candidate_mean_width == 0.0)
        )
        by_metric[metric] = {
            "common_rows": len(metric_rows),
            "base_mean_ci_width": base_mean_width,
            "candidate_mean_ci_width": candidate_mean_width,
            "mean_reduction_fraction": reduction,
            "target_relative_reduction": target_reduction,
            "target_met": target_met,
        }
    return by_metric


def _aggregate_metric_means(
    base_rows: dict[tuple[str, str, str], dict[str, Any]],
    candidate_rows: dict[tuple[str, str, str], dict[str, Any]],
    metrics: tuple[str, ...],
    *,
    relative_threshold: float,
    absolute_floor: float,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: {"base": defaultdict(list), "candidate": defaultdict(list)}
    )
    for scenario_id, planner_key, kinematics in sorted(set(base_rows) & set(candidate_rows)):
        planner_group_key = (planner_key, kinematics)
        base_row = base_rows[(scenario_id, planner_key, kinematics)]
        candidate_row = candidate_rows[(scenario_id, planner_key, kinematics)]
        for metric in metrics:
            base_mean = _metric_mean(base_row, metric)
            candidate_mean = _metric_mean(candidate_row, metric)
            if base_mean is not None and candidate_mean is not None:
                grouped[planner_group_key]["base"][metric].append(base_mean)
                grouped[planner_group_key]["candidate"][metric].append(candidate_mean)

    rows: list[dict[str, Any]] = []
    flagged_rows: list[dict[str, Any]] = []
    for (planner_key, kinematics), values in sorted(grouped.items()):
        for metric in metrics:
            base_mean = _mean(list(values["base"].get(metric, [])))
            candidate_mean = _mean(list(values["candidate"].get(metric, [])))
            if base_mean is None or candidate_mean is None:
                continue
            delta = candidate_mean - base_mean
            relative_delta = _relative_change(base_mean, candidate_mean)
            threshold = max(absolute_floor, abs(base_mean) * relative_threshold)
            flagged = abs(delta) > threshold
            row = {
                "planner_key": planner_key,
                "kinematics": kinematics,
                "metric": metric,
                "base_mean": base_mean,
                "candidate_mean": candidate_mean,
                "delta": delta,
                "absolute_delta": abs(delta),
                "relative_delta": relative_delta,
                "flag_threshold": threshold,
                "flagged": flagged,
            }
            rows.append(row)
            if flagged:
                flagged_rows.append(row)
    return {
        "relative_threshold": relative_threshold,
        "absolute_floor": absolute_floor,
        "rows": rows,
        "flagged_rows": flagged_rows,
        "flagged_count": len(flagged_rows),
    }


def _rank_values(
    values: dict[tuple[str, str], float], *, higher_is_better: bool
) -> dict[tuple[str, str], float]:
    ordered = sorted(
        values.items(),
        key=lambda item: (-item[1] if higher_is_better else item[1], item[0][0], item[0][1]),
    )
    ranks: dict[tuple[str, str], float] = {}
    index = 0
    while index < len(ordered):
        value = ordered[index][1]
        tie_end = index + 1
        while tie_end < len(ordered) and ordered[tie_end][1] == value:
            tie_end += 1
        average_rank = (index + 1 + tie_end) / 2.0
        for tied_index in range(index, tie_end):
            ranks[ordered[tied_index][0]] = average_rank
        index = tie_end
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    numerator = sum(x * y for x, y in zip(x_centered, y_centered, strict=True))
    x_den = math.sqrt(sum(x * x for x in x_centered))
    y_den = math.sqrt(sum(y * y for y in y_centered))
    if x_den <= 1e-12 or y_den <= 1e-12:
        return None
    return float(numerator / (x_den * y_den))


def _spearman(
    base_values: dict[tuple[str, str], float],
    candidate_values: dict[tuple[str, str], float],
    *,
    higher_is_better: bool,
) -> float | None:
    common = sorted(set(base_values) & set(candidate_values))
    if len(common) < 2:
        return None
    base_ranks = _rank_values(
        {key: base_values[key] for key in common},
        higher_is_better=higher_is_better,
    )
    candidate_ranks = _rank_values(
        {key: candidate_values[key] for key in common},
        higher_is_better=higher_is_better,
    )
    return _pearson([base_ranks[key] for key in common], [candidate_ranks[key] for key in common])


def _kendall_tau(
    base_values: dict[tuple[str, str], float],
    candidate_values: dict[tuple[str, str], float],
    *,
    higher_is_better: bool,
) -> float | None:
    common = sorted(set(base_values) & set(candidate_values))
    if len(common) < 2:
        return None
    concordant = 0
    discordant = 0
    multiplier = 1.0 if higher_is_better else -1.0
    for index, left in enumerate(common):
        for right in common[index + 1 :]:
            base_delta = multiplier * (base_values[left] - base_values[right])
            candidate_delta = multiplier * (candidate_values[left] - candidate_values[right])
            if abs(base_delta) <= 1e-12 or abs(candidate_delta) <= 1e-12:
                continue
            if base_delta * candidate_delta > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return float((concordant - discordant) / total)


def _ranking_stability(
    base_planner_rows: dict[tuple[str, str], dict[str, Any]],
    candidate_planner_rows: dict[tuple[str, str], dict[str, Any]],
    *,
    metric: str,
    min_tau: float,
) -> dict[str, Any]:
    higher_is_better = _metric_direction(metric.replace("_mean", "")) == "higher"
    common = sorted(set(base_planner_rows) & set(candidate_planner_rows))
    base_values = {
        key: value
        for key in common
        if (value := _safe_float(base_planner_rows[key].get(metric))) is not None
    }
    candidate_values = {
        key: value
        for key in common
        if (value := _safe_float(candidate_planner_rows[key].get(metric))) is not None
    }
    common_metric_keys = sorted(set(base_values) & set(candidate_values))
    base_values = {key: base_values[key] for key in common_metric_keys}
    candidate_values = {key: candidate_values[key] for key in common_metric_keys}
    base_order = [
        {"planner_key": key[0], "kinematics": key[1], "value": base_values[key]}
        for key in sorted(
            common_metric_keys,
            key=lambda item: (
                -base_values[item] if higher_is_better else base_values[item],
                item[0],
                item[1],
            ),
        )
    ]
    candidate_order = [
        {"planner_key": key[0], "kinematics": key[1], "value": candidate_values[key]}
        for key in sorted(
            common_metric_keys,
            key=lambda item: (
                -candidate_values[item] if higher_is_better else candidate_values[item],
                item[0],
                item[1],
            ),
        )
    ]
    tau = _kendall_tau(base_values, candidate_values, higher_is_better=higher_is_better)
    rho = _spearman(base_values, candidate_values, higher_is_better=higher_is_better)
    status = "insufficient_data"
    if tau is not None:
        status = "stable" if tau >= min_tau else "unstable"
    return {
        "metric": metric,
        "direction": "higher" if higher_is_better else "lower",
        "common_planner_count": len(common_metric_keys),
        "spearman_rho": rho,
        "kendall_tau": tau,
        "min_kendall_tau": min_tau,
        "status": status,
        "base_order": base_order,
        "candidate_order": candidate_order,
    }


def _scenario_winner_stability(
    base_rows: dict[tuple[str, str, str], dict[str, Any]],
    candidate_rows: dict[tuple[str, str, str], dict[str, Any]],
    *,
    metric: str,
    change_threshold: float,
) -> dict[str, Any]:
    direction = _metric_direction(metric)
    grouped: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"base": {}, "candidate": {}}
    )
    for scenario_id, planner_key, kinematics in sorted(set(base_rows) & set(candidate_rows)):
        base_mean = _metric_mean(base_rows[(scenario_id, planner_key, kinematics)], metric)
        candidate_mean = _metric_mean(
            candidate_rows[(scenario_id, planner_key, kinematics)],
            metric,
        )
        if base_mean is None or candidate_mean is None:
            continue
        grouped[(scenario_id, kinematics)]["base"][planner_key] = base_mean
        grouped[(scenario_id, kinematics)]["candidate"][planner_key] = candidate_mean

    changed: list[dict[str, Any]] = []
    compared = 0
    for (scenario_id, kinematics), values in sorted(grouped.items()):
        common_planners = sorted(set(values["base"]) & set(values["candidate"]))
        if len(common_planners) < 2:
            continue
        compared += 1
        reverse = direction == "higher"
        base_winner = sorted(
            common_planners,
            key=lambda planner: (
                -values["base"][planner] if reverse else values["base"][planner],
                planner,
            ),
        )[0]
        candidate_winner = sorted(
            common_planners,
            key=lambda planner: (
                -values["candidate"][planner] if reverse else values["candidate"][planner],
                planner,
            ),
        )[0]
        if base_winner != candidate_winner:
            changed.append(
                {
                    "scenario_id": scenario_id,
                    "kinematics": kinematics,
                    "base_winner": base_winner,
                    "candidate_winner": candidate_winner,
                    "base_winner_value": values["base"][base_winner],
                    "candidate_winner_value": values["candidate"][candidate_winner],
                }
            )
    changed_fraction = (len(changed) / compared) if compared else None
    status = "insufficient_data"
    if changed_fraction is not None:
        status = "stable" if changed_fraction <= change_threshold else "unstable"
    return {
        "metric": metric,
        "direction": direction,
        "common_scenario_count": compared,
        "changed_count": len(changed),
        "changed_fraction": changed_fraction,
        "change_threshold": change_threshold,
        "status": status,
        "changed_scenarios": changed,
    }


def _interpretation(
    *,
    interval_summary: dict[str, Any],
    mean_drift: dict[str, Any],
    ranking: dict[str, Any],
    scenario_winners: dict[str, Any],
) -> dict[str, Any]:
    notes: list[str] = []
    ranking_changed = ranking.get("status") == "unstable"
    scenario_changed = scenario_winners.get("status") == "unstable"
    drift_flagged = int(mean_drift.get("flagged_count", 0) or 0) > 0
    interval_misses = [
        metric
        for metric, row in interval_summary.items()
        if isinstance(row, dict) and row.get("target_met") is False
    ]
    if ranking_changed:
        notes.append("Planner ranking stability fell below the configured Kendall tau threshold.")
    if scenario_changed:
        notes.append("Scenario-level winner changes exceeded the configured threshold.")
    if drift_flagged:
        notes.append("Aggregate planner mean drift exceeded the configured drift threshold.")
    if interval_misses:
        notes.append(
            "CI-width reduction target was not met for: " + ", ".join(sorted(interval_misses))
        )
    if not notes:
        notes.append(
            "No ranking, scenario-winner, or aggregate-drift trigger changed the headline interpretation."
        )
    headline_changes = bool(ranking_changed or scenario_changed or drift_flagged)
    return {
        "status": "review" if headline_changes else "stable",
        "headline_interpretation_changes": headline_changes,
        "ci_width_target_missed_metrics": interval_misses,
        "notes": notes,
    }


def compare_seed_schedules(  # noqa: PLR0913
    base_campaign_root: Path,
    candidate_campaign_root: Path,
    *,
    interval_reduction_target: float = 0.20,
    mean_drift_relative_threshold: float = 0.05,
    mean_drift_absolute_floor: float = 0.02,
    ranking_metric: str = "snqi_mean",
    ranking_min_tau: float = 0.80,
    scenario_winner_metric: str = "snqi",
    scenario_winner_change_threshold: float = 0.10,
) -> dict[str, Any]:
    """Compare two camera-ready campaigns that differ only by seed schedule."""
    base = _load_campaign(base_campaign_root.resolve())
    candidate = _load_campaign(candidate_campaign_root.resolve())
    metrics = tuple(
        metric
        for metric in _SEED_METRICS
        if metric in set(base["seed_variability"].get("metrics", []))
        and metric in set(candidate["seed_variability"].get("metrics", []))
    )
    base_seed_rows = _seed_rows_by_key(base)
    candidate_seed_rows = _seed_rows_by_key(candidate)
    base_planner_rows = _planner_rows_by_key(base)
    candidate_planner_rows = _planner_rows_by_key(candidate)

    interval_rows = _build_interval_width_rows(base_seed_rows, candidate_seed_rows, metrics)
    interval_summary = _summarize_interval_widths(
        interval_rows,
        metrics,
        target_reduction=interval_reduction_target,
    )
    mean_drift = _aggregate_metric_means(
        base_seed_rows,
        candidate_seed_rows,
        metrics,
        relative_threshold=mean_drift_relative_threshold,
        absolute_floor=mean_drift_absolute_floor,
    )
    ranking = _ranking_stability(
        base_planner_rows,
        candidate_planner_rows,
        metric=ranking_metric,
        min_tau=ranking_min_tau,
    )
    scenario_winners = _scenario_winner_stability(
        base_seed_rows,
        candidate_seed_rows,
        metric=scenario_winner_metric,
        change_threshold=scenario_winner_change_threshold,
    )
    interpretation = _interpretation(
        interval_summary=interval_summary,
        mean_drift=mean_drift,
        ranking=ranking,
        scenario_winners=scenario_winners,
    )
    return {
        "schema_version": "benchmark-seed-schedule-comparison.v1",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "base_campaign": _campaign_metadata(base),
        "candidate_campaign": _campaign_metadata(candidate),
        "decision_criteria": {
            "interval_reduction_target": interval_reduction_target,
            "mean_drift_relative_threshold": mean_drift_relative_threshold,
            "mean_drift_absolute_floor": mean_drift_absolute_floor,
            "ranking_metric": ranking_metric,
            "ranking_min_kendall_tau": ranking_min_tau,
            "scenario_winner_metric": scenario_winner_metric,
            "scenario_winner_change_threshold": scenario_winner_change_threshold,
        },
        "coverage": {
            "base_seed_rows": len(base_seed_rows),
            "candidate_seed_rows": len(candidate_seed_rows),
            "common_seed_rows": len(set(base_seed_rows) & set(candidate_seed_rows)),
            "missing_in_base": [
                list(key) for key in sorted(set(candidate_seed_rows) - set(base_seed_rows))
            ],
            "missing_in_candidate": [
                list(key) for key in sorted(set(base_seed_rows) - set(candidate_seed_rows))
            ],
        },
        "interval_width": {
            "metrics": list(metrics),
            "aggregate": interval_summary,
            "rows": interval_rows,
        },
        "mean_drift": mean_drift,
        "ranking_stability": ranking,
        "scenario_winner_stability": scenario_winners,
        "interpretation": interpretation,
    }


def _build_markdown(payload: dict[str, Any]) -> str:
    base = payload.get("base_campaign", {})
    candidate = payload.get("candidate_campaign", {})
    interpretation = payload.get("interpretation", {})
    lines = [
        "# Seed Schedule Comparison",
        "",
        f"- Base campaign: `{base.get('campaign_id', 'unknown')}`",
        f"- Candidate campaign: `{candidate.get('campaign_id', 'unknown')}`",
        f"- Base seeds: `{base.get('resolved_seeds', [])}`",
        f"- Candidate seeds: `{candidate.get('resolved_seeds', [])}`",
        f"- Verdict: `{interpretation.get('status', 'unknown')}`",
        f"- Headline interpretation changes: `{interpretation.get('headline_interpretation_changes', 'unknown')}`",
        "",
        "## Decision Criteria",
        "",
    ]
    criteria = payload.get("decision_criteria", {})
    for key, value in criteria.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(["", "## Interpretation", ""])
    for note in interpretation.get("notes", []):
        lines.append(f"- {note}")

    lines.extend(
        [
            "",
            "## CI Width Reduction",
            "",
            "| metric | base mean width | candidate mean width | reduction | target met |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for metric, row in (payload.get("interval_width", {}).get("aggregate") or {}).items():
        lines.append(
            "| "
            f"{metric} | {_fmt(row.get('base_mean_ci_width'))} | "
            f"{_fmt(row.get('candidate_mean_ci_width'))} | "
            f"{_fmt(row.get('mean_reduction_fraction'))} | "
            f"{'yes' if row.get('target_met') else 'no'} |"
        )

    drift = payload.get("mean_drift", {})
    lines.extend(
        [
            "",
            "## Aggregate Mean Drift",
            "",
            f"- Flagged rows: `{drift.get('flagged_count', 0)}`",
            "",
            "| planner | metric | base | candidate | delta | relative delta | flagged |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    drift_rows = drift.get("flagged_rows") or drift.get("rows") or []
    for row in drift_rows[:40]:
        lines.append(
            "| "
            f"{row.get('planner_key')} | {row.get('metric')} | "
            f"{_fmt(row.get('base_mean'))} | {_fmt(row.get('candidate_mean'))} | "
            f"{_fmt(row.get('delta'))} | {_fmt(row.get('relative_delta'))} | "
            f"{'yes' if row.get('flagged') else 'no'} |"
        )
    if len(drift_rows) > 40:
        lines.append(
            "| Table truncated | ... | ... | ... | ... | ... | "
            f"{len(drift_rows) - 40} more rows not shown |"
        )

    ranking = payload.get("ranking_stability", {})
    lines.extend(
        [
            "",
            "## Ranking Stability",
            "",
            f"- Metric: `{ranking.get('metric', 'unknown')}`",
            f"- Kendall tau: `{_fmt(ranking.get('kendall_tau'))}`",
            f"- Spearman rho: `{_fmt(ranking.get('spearman_rho'))}`",
            f"- Status: `{ranking.get('status', 'unknown')}`",
        ]
    )

    winners = payload.get("scenario_winner_stability", {})
    lines.extend(
        [
            "",
            "## Scenario Winner Stability",
            "",
            f"- Metric: `{winners.get('metric', 'unknown')}`",
            f"- Changed scenarios: `{winners.get('changed_count', 0)}` / `{winners.get('common_scenario_count', 0)}`",
            f"- Changed fraction: `{_fmt(winners.get('changed_fraction'))}`",
            f"- Status: `{winners.get('status', 'unknown')}`",
        ]
    )
    changed = winners.get("changed_scenarios") or []
    if changed:
        lines.extend(
            [
                "",
                "| scenario | base winner | candidate winner |",
                "|---|---|---|",
            ]
        )
        for row in changed[:40]:
            lines.append(
                "| "
                f"{row.get('scenario_id')} | {row.get('base_winner')} | "
                f"{row.get('candidate_winner')} |"
            )
        if len(changed) > 40:
            lines.append(
                f"| Table truncated | ... | {len(changed) - 40} more scenarios not shown |"
            )

    lines.append("")
    return "\n".join(lines)


def _resolve_safe_output_path(path: Path, safe_root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(safe_root):
        raise ValueError(f"Unsafe output path outside {safe_root}: {path}")
    return resolved


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-campaign-root", type=Path, required=True)
    parser.add_argument("--candidate-campaign-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--interval-reduction-target", type=float, default=0.20)
    parser.add_argument("--mean-drift-relative-threshold", type=float, default=0.05)
    parser.add_argument("--mean-drift-absolute-floor", type=float, default=0.02)
    parser.add_argument("--ranking-metric", default="snqi_mean")
    parser.add_argument("--ranking-min-tau", type=float, default=0.80)
    parser.add_argument("--scenario-winner-metric", default="snqi")
    parser.add_argument("--scenario-winner-change-threshold", type=float, default=0.10)
    return parser


def main() -> int:
    """CLI entry point for seed-schedule campaign comparison."""
    args = _build_parser().parse_args()
    payload = compare_seed_schedules(
        args.base_campaign_root,
        args.candidate_campaign_root,
        interval_reduction_target=args.interval_reduction_target,
        mean_drift_relative_threshold=args.mean_drift_relative_threshold,
        mean_drift_absolute_floor=args.mean_drift_absolute_floor,
        ranking_metric=args.ranking_metric,
        ranking_min_tau=args.ranking_min_tau,
        scenario_winner_metric=args.scenario_winner_metric,
        scenario_winner_change_threshold=args.scenario_winner_change_threshold,
    )
    safe_root = Path.cwd().resolve()
    output_json = _resolve_safe_output_path(args.output_json, safe_root)
    output_md = _resolve_safe_output_path(args.output_md, safe_root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(_build_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
