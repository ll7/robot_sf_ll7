#!/usr/bin/env python3
"""Analyze seed sufficiency, interval width, and ranking stability across campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

from robot_sf.benchmark.rank_metrics import kendall_tau, rank_order

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SCHEMA_VERSION = "seed_sufficiency_analysis.v1"
DEFAULT_METRICS = ("success", "collisions", "snqi")
LOWER_IS_BETTER = {"collisions", "collision", "near_misses", "time_to_goal", "time_to_goal_norm"}
NON_PROMOTABLE_ROW_STATUSES = frozenset(
    {
        "degraded",
        "failed",
        "fallback",
        "invalid",
        "not-available",
        "not_available",
        "partial-failure",
        "partial_failure",
        "unavailable",
    }
)


def analyze_seed_sufficiency(
    campaign_roots: list[Path],
    output_dir: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    rank_metric: str = "snqi",
    advisory_seed_threshold: int = 2,
    headline_min_seed_budget: int = 20,
    headline_required_durable_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Analyze one or more campaign report folders and write output artifacts."""

    campaigns = [_load_campaign(root, metrics=metrics) for root in campaign_roots]
    campaigns.sort(key=lambda campaign: (campaign["seed_count"], campaign["label"]))

    interval_rows = _build_interval_rows(campaigns, metrics)
    outcome_rows = _build_outcome_counts(campaigns)
    rank_rows = _build_rank_rows(campaigns, rank_metric=rank_metric)
    family_rows = _build_family_instability_rows(campaigns, rank_metric=rank_metric)
    caveats = _build_caveats(
        campaigns,
        rank_rows,
        family_rows,
        advisory_seed_threshold=advisory_seed_threshold,
    )
    headline_contract = _build_headline_contract(
        campaigns,
        metrics=metrics,
        rank_metric=rank_metric,
        min_seed_budget=headline_min_seed_budget,
        required_durable_roots=headline_required_durable_roots,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "rank_metric": rank_metric,
        "metrics": list(metrics),
        "campaigns": [_campaign_public_metadata(campaign) for campaign in campaigns],
        "summary": {
            "campaign_count": len(campaigns),
            "advisory_campaigns": [
                campaign["label"]
                for campaign in campaigns
                if campaign["seed_count"] < advisory_seed_threshold
            ],
            "ranking_instability_rows": sum(1 for row in rank_rows if row["rank_changed"]),
            "scenario_family_winner_changes": sum(
                1 for row in family_rows if row["winner_changed"]
            ),
            "underpowered_or_unstable": bool(caveats),
        },
        "interval_width_rows": interval_rows,
        "planner_rank_stability": rank_rows,
        "scenario_family_instability": family_rows,
        "outcome_counts": outcome_rows,
        "headline_rank_stability_contract": headline_contract,
        "caveats": caveats,
        "artifacts": {
            "json": "seed_sufficiency_analysis.json",
            "headline_contract_json": "headline_rank_stability_contract.json",
            "headline_pairwise_csv": "headline_rank_stability_pairwise.csv",
            "interval_width_csv": "seed_count_interval_width.csv",
            "planner_rank_csv": "planner_rank_stability.csv",
            "scenario_family_csv": "scenario_family_instability.csv",
            "outcome_counts_csv": "outcome_counts.csv",
            "markdown": "seed_sufficiency_summary.md",
            "figure": "fig_seed_interval_width.png",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "seed_sufficiency_analysis.json", payload)
    _write_json(output_dir / "headline_rank_stability_contract.json", headline_contract)
    _write_csv(output_dir / "headline_rank_stability_pairwise.csv", headline_contract["pairwise"])
    _write_csv(output_dir / "seed_count_interval_width.csv", interval_rows)
    _write_csv(output_dir / "planner_rank_stability.csv", rank_rows)
    _write_csv(output_dir / "scenario_family_instability.csv", family_rows)
    _write_csv(output_dir / "outcome_counts.csv", outcome_rows)
    (output_dir / "seed_sufficiency_summary.md").write_text(
        _build_markdown(payload),
        encoding="utf-8",
    )
    _write_interval_figure(output_dir / "fig_seed_interval_width.png", interval_rows, metrics)
    return payload


def resolve_campaign_roots(
    *,
    campaign_roots: list[Path] | None = None,
    campaign_output_roots: list[Path] | None = None,
    campaign_ids: list[str] | None = None,
) -> list[Path]:
    """Resolve direct and container roots into campaign roots with seed reports."""

    roots: list[Path] = []
    roots.extend(campaign_roots or [])
    ids = [campaign_id.strip() for campaign_id in (campaign_ids or []) if campaign_id.strip()]
    for output_root in campaign_output_roots or []:
        roots.extend(_discover_campaign_roots(output_root, campaign_ids=ids))
    roots = _dedupe_paths(roots)
    if not roots:
        raise FileNotFoundError(
            "No campaign roots supplied or discovered. Pass --campaign-root for exact roots, "
            "or --campaign-output-root for a Slurm/output container containing reports/"
            "seed_variability_by_scenario.json."
        )
    return roots


def _discover_campaign_roots(output_root: Path, *, campaign_ids: list[str]) -> list[Path]:
    """Find campaign roots beneath a Slurm/output container, failing closed if absent."""

    if not output_root.exists():
        raise FileNotFoundError(f"Campaign output root does not exist: {output_root}")
    if _has_seed_reports(output_root):
        candidates = [output_root]
    else:
        candidates = sorted(
            {
                path.parent.parent
                for path in output_root.rglob("reports/seed_variability_by_scenario.json")
                if path.is_file()
            }
        )
    if campaign_ids:
        candidates = [
            root
            for root in candidates
            if any(campaign_id in root.name for campaign_id in campaign_ids)
        ]
    if not candidates:
        id_note = f" matching campaign id(s) {campaign_ids}" if campaign_ids else ""
        raise FileNotFoundError(
            f"No seed-sufficiency campaign reports found under {output_root}{id_note}. "
            "Expected reports/seed_variability_by_scenario.json and "
            "reports/seed_episode_rows.csv from the completed S20/S30 campaign."
        )
    return candidates


def _has_seed_reports(root: Path) -> bool:
    """Return whether a path already looks like a campaign root."""

    reports = root / "reports"
    return (reports / "seed_variability_by_scenario.json").is_file()


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate equivalent paths while preserving deterministic order."""

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in sorted(paths, key=str):
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _load_campaign(root: Path, *, metrics: tuple[str, ...]) -> dict[str, Any]:
    """Load the report artifacts used by the analysis."""

    reports = root / "reports"
    seed_payload = _read_json(reports / "seed_variability_by_scenario.json")
    seed_rows = [row for row in seed_payload.get("rows", []) if isinstance(row, dict)]
    episode_rows = _read_csv(reports / "seed_episode_rows.csv")
    sufficiency_path = reports / "statistical_sufficiency.json"
    sufficiency = _read_json(sufficiency_path) if sufficiency_path.exists() else {}
    seed_count = _campaign_seed_count(seed_rows, episode_rows)
    return {
        "root": root,
        "label": root.name,
        "seed_count": seed_count,
        "seed_variability": seed_payload,
        "seed_rows": seed_rows,
        "episode_rows": episode_rows,
        "statistical_sufficiency": sufficiency,
        "metrics": metrics,
    }


def _campaign_public_metadata(campaign: dict[str, Any]) -> dict[str, Any]:
    """Return public campaign metadata for the JSON payload."""

    return {
        "label": campaign["label"],
        "root": str(campaign["root"]),
        "seed_count": campaign["seed_count"],
        "seed_variability_schema": campaign["seed_variability"].get("schema_version"),
        "episode_row_count": len(campaign["episode_rows"]),
        "seed_variability_row_count": len(campaign["seed_rows"]),
        "statistical_sufficiency_available": bool(campaign["statistical_sufficiency"]),
    }


def _campaign_seed_count(
    seed_rows: list[dict[str, Any]], episode_rows: list[dict[str, str]]
) -> int:
    """Resolve a campaign seed count from seed-variability or episode rows."""

    counts = [_safe_int(row.get("seed_count")) for row in seed_rows]
    finite_counts = [count for count in counts if count is not None]
    if finite_counts:
        return max(finite_counts)
    seeds = {row.get("seed") for row in episode_rows if row.get("seed") not in (None, "")}
    return len(seeds)


def _build_interval_rows(
    campaigns: list[dict[str, Any]], metrics: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Build seed-count versus interval-width rows."""

    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        for seed_row in sorted(campaign["seed_rows"], key=_seed_row_sort_key):
            for metric in metrics:
                summary = _metric_summary(seed_row, metric)
                ci_width = _ci_width(summary)
                rows.append(
                    {
                        "campaign": campaign["label"],
                        "seed_count": campaign["seed_count"],
                        "scenario_id": str(seed_row.get("scenario_id", "")),
                        "scenario_family": _scenario_family(seed_row),
                        "planner_key": str(seed_row.get("planner_key", "")),
                        "kinematics": str(seed_row.get("kinematics", "")),
                        "metric": metric,
                        "mean": _safe_float(summary.get("mean")),
                        "ci_width": ci_width,
                        "ci_half_width": _safe_float(summary.get("ci_half_width")),
                        "advisory": campaign["seed_count"] < 2 or ci_width is None,
                    }
                )
    return rows


def _build_outcome_counts(campaigns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve raw success/collision numerator and denominator counts."""

    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for campaign in campaigns:
        for row in campaign["episode_rows"]:
            key = (
                campaign["label"],
                _scenario_family(row),
                row.get("scenario_id") or "unknown",
                row.get("planner_key") or row.get("algo") or "unknown",
            )
            bucket = grouped.setdefault(
                key,
                {
                    "campaign": key[0],
                    "scenario_family": key[1],
                    "scenario_id": key[2],
                    "planner_key": key[3],
                    "episode_count": 0,
                    "success_count": 0,
                    "collision_count": 0,
                },
            )
            bucket["episode_count"] += 1
            bucket["success_count"] += _truthy_int(row.get("success"))
            bucket["collision_count"] += _truthy_int(row.get("collision") or row.get("collisions"))
    return [grouped[key] for key in sorted(grouped)]


def _build_rank_rows(campaigns: list[dict[str, Any]], *, rank_metric: str) -> list[dict[str, Any]]:
    """Build planner rank stability rows across campaigns."""

    previous_ranks: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        values = _planner_metric_values(campaign, rank_metric, eligible_only=False)
        ranks = {
            planner: index + 1
            for index, planner in enumerate(
                rank_order(values, higher_is_better=_higher_is_better(rank_metric))
            )
        }
        for planner_key in sorted(values):
            previous_rank = previous_ranks.get(planner_key)
            rank = ranks[planner_key]
            row = {
                "campaign": campaign["label"],
                "seed_count": campaign["seed_count"],
                "planner_key": planner_key,
                "metric": rank_metric,
                "metric_mean": values[planner_key],
                "rank": rank,
                "previous_rank": previous_rank,
                "rank_delta": None if previous_rank is None else rank - previous_rank,
                "rank_changed": previous_rank is not None and rank != previous_rank,
                "advisory": campaign["seed_count"] < 2,
            }
            rows.append(row)
        previous_ranks = ranks
    return rows


def _build_family_instability_rows(
    campaigns: list[dict[str, Any]], *, rank_metric: str
) -> list[dict[str, Any]]:
    """Build scenario-family winner change rows across campaigns."""

    previous_winners: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        family_values = _family_planner_values(campaign, rank_metric, eligible_only=False)
        winners = {
            family: _winner(values, higher_is_better=_higher_is_better(rank_metric))
            for family, values in family_values.items()
        }
        for family in sorted(winners):
            winner = winners[family]
            previous_winner = previous_winners.get(family)
            rows.append(
                {
                    "campaign": campaign["label"],
                    "seed_count": campaign["seed_count"],
                    "scenario_family": family,
                    "metric": rank_metric,
                    "winning_planner": winner,
                    "previous_winning_planner": previous_winner,
                    "winner_changed": previous_winner is not None and winner != previous_winner,
                    "advisory": campaign["seed_count"] < 2,
                }
            )
        previous_winners = winners
    return rows


def _build_caveats(
    campaigns: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    *,
    advisory_seed_threshold: int,
) -> list[str]:
    """Build human-readable caveats for unstable or underpowered surfaces."""

    caveats: list[str] = []
    for campaign in campaigns:
        if campaign["seed_count"] < advisory_seed_threshold:
            caveats.append(
                f"{campaign['label']}: single-seed or incomplete-seed surface; treat as advisory."
            )
    changed_rank_rows = [row for row in rank_rows if row["rank_changed"]]
    if changed_rank_rows:
        caveats.append(
            f"{len(changed_rank_rows)} planner rank rows changed across seed schedules; "
            "inspect planner_rank_stability.csv before making ranking claims."
        )
    changed_family_rows = [row for row in family_rows if row["winner_changed"]]
    if changed_family_rows:
        caveats.append(
            f"{len(changed_family_rows)} scenario-family winners changed across seed schedules; "
            "scenario-family conclusions remain unstable."
        )
    return caveats


def _build_headline_contract(
    campaigns: list[dict[str, Any]],
    *,
    metrics: tuple[str, ...],
    rank_metric: str,
    min_seed_budget: int,
    required_durable_roots: tuple[Path, ...],
) -> dict[str, Any]:
    """Build the fail-closed headline rank-stability contract payload."""

    max_seed_count = max((campaign["seed_count"] for campaign in campaigns), default=0)
    source_roots = [str(campaign["root"]) for campaign in campaigns]
    missing_durable_roots = [
        str(root) for root in required_durable_roots if not _durable_root_available(root)
    ]
    row_status_exclusions = _row_status_exclusions(campaigns)
    pairwise_rows = _build_headline_pairwise_rows(campaigns, rank_metric=rank_metric)
    rank_flipped = any(row["rank_label"] == "rank_flip" for row in pairwise_rows)
    labels = _headline_labels(
        max_seed_count=max_seed_count,
        min_seed_budget=min_seed_budget,
        missing_durable_roots=missing_durable_roots,
        row_status_exclusions=row_status_exclusions,
        rank_flipped=rank_flipped,
    )
    caveats = _headline_caveats(
        labels=labels,
        max_seed_count=max_seed_count,
        min_seed_budget=min_seed_budget,
        missing_durable_roots=missing_durable_roots,
        row_status_exclusions=row_status_exclusions,
    )
    return {
        "schema_version": "headline-rank-stability-contract.v1",
        "contract_scope": "headline_rank_stability_ci_preflight",
        "label": labels[0],
        "claim_status": _headline_claim_status(labels),
        "labels": labels,
        "source_roots": source_roots,
        "seed_counts": [campaign["seed_count"] for campaign in campaigns],
        "max_seed_count": max_seed_count,
        "min_seed_budget": min_seed_budget,
        "metric_names": list(metrics),
        "rank_metric": rank_metric,
        "required_durable_roots": [str(root) for root in required_durable_roots],
        "missing_durable_roots": missing_durable_roots,
        "row_status_exclusions": row_status_exclusions,
        "pairwise": pairwise_rows,
        "caveats": caveats,
        "promotion_allowed": labels[0] == "stable"
        and "row_status_exclusions_present" not in labels,
    }


def _build_headline_pairwise_rows(
    campaigns: list[dict[str, Any]], *, rank_metric: str
) -> list[dict[str, Any]]:
    """Compare eligible planner rankings between consecutive seed schedules."""

    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for campaign in campaigns:
        values = _planner_metric_values(campaign, rank_metric, eligible_only=True)
        ranking = [
            str(planner)
            for planner in rank_order(values, higher_is_better=_higher_is_better(rank_metric))
        ]
        if previous is not None:
            previous_ranking = previous["ranking"]
            comparable = len(ranking) >= 2 and set(ranking) == set(previous_ranking)
            tau = kendall_tau(previous_ranking, ranking, degenerate=None) if comparable else None
            rank_flip = comparable and ranking != previous_ranking
            rows.append(
                {
                    "from_campaign": previous["label"],
                    "to_campaign": campaign["label"],
                    "from_seed_count": previous["seed_count"],
                    "to_seed_count": campaign["seed_count"],
                    "metric": rank_metric,
                    "from_ranking": "|".join(previous_ranking),
                    "to_ranking": "|".join(ranking),
                    "kendall_tau": tau,
                    "rank_label": "rank_flip"
                    if rank_flip
                    else "stable"
                    if comparable
                    else "not_comparable",
                    "eligible_planner_count": len(ranking),
                }
            )
        previous = {
            "label": campaign["label"],
            "seed_count": campaign["seed_count"],
            "ranking": ranking,
        }
    return rows


def _headline_labels(
    *,
    max_seed_count: int,
    min_seed_budget: int,
    missing_durable_roots: list[str],
    row_status_exclusions: list[dict[str, Any]],
    rank_flipped: bool,
) -> list[str]:
    """Return deterministic headline contract labels, strongest blocker first."""

    labels: list[str] = []
    if max_seed_count < min_seed_budget or missing_durable_roots:
        labels.append("blocked_pending_s20_s30")
    if row_status_exclusions:
        labels.append("row_status_exclusions_present")
    if labels:
        return labels
    return ["rank_flip_detected" if rank_flipped else "stable"]


def _headline_claim_status(labels: list[str]) -> str:
    """Map contract labels to paper-facing claim status."""

    if "blocked_pending_s20_s30" in labels:
        return "blocked_missing_increased_seed_rows"
    if "row_status_exclusions_present" in labels:
        return "blocked_non_promotable_rows"
    if labels[0] == "rank_flip_detected":
        return "not_statistically_distinguishable_budget"
    return "paper_grade"


def _headline_caveats(
    *,
    labels: list[str],
    max_seed_count: int,
    min_seed_budget: int,
    missing_durable_roots: list[str],
    row_status_exclusions: list[dict[str, Any]],
) -> list[str]:
    """Render concise caveats for the headline contract."""

    caveats: list[str] = []
    if "blocked_pending_s20_s30" in labels:
        caveats.append(
            f"Headline ranking remains blocked until seed budget reaches at least S{min_seed_budget} "
            f"and required durable roots are available; observed max seed count is S{max_seed_count}."
        )
    if missing_durable_roots:
        caveats.append("Missing required durable roots: " + ", ".join(missing_durable_roots))
    if row_status_exclusions:
        statuses = sorted({str(row["row_status"]) for row in row_status_exclusions})
        caveats.append(
            "Rows with non-promotable statuses were excluded from headline ranking: "
            + ", ".join(statuses)
        )
    if not caveats:
        caveats.append(
            "Synthetic or local contract output only; durable campaign evidence must still be cited separately."
        )
    return caveats


def _row_status_exclusions(campaigns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that are explicitly ineligible for headline promotion."""

    exclusions: list[dict[str, Any]] = []
    for campaign in campaigns:
        for row in campaign["seed_rows"]:
            status = _row_status(row)
            if status not in NON_PROMOTABLE_ROW_STATUSES:
                continue
            exclusions.append(
                {
                    "campaign": campaign["label"],
                    "source_root": str(campaign["root"]),
                    "scenario_id": str(row.get("scenario_id", "")),
                    "scenario_family": _scenario_family(row),
                    "planner_key": str(row.get("planner_key", "unknown")),
                    "row_status": status,
                }
            )
    return sorted(
        exclusions,
        key=lambda row: (
            row["campaign"],
            row["scenario_id"],
            row["planner_key"],
            row["row_status"],
        ),
    )


def _row_status(row: dict[str, Any]) -> str:
    """Normalize row status/mode fields used to exclude non-promotable results."""

    for key in (
        "row_status",
        "status",
        "result_status",
        "validity_status",
        "readiness_status",
        "availability_status",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip().lower()
    for key in ("execution_mode", "mode", "planner_mode"):
        value = row.get(key)
        if str(value).strip().lower() in {"fallback", "degraded"}:
            return str(value).strip().lower()
    return "valid"


def _durable_root_available(root: Path) -> bool:
    """Return whether a required durable root exists and is non-empty."""

    return root.exists() and (root.is_file() or any(root.iterdir()))


def _planner_metric_values(
    campaign: dict[str, Any], metric: str, *, eligible_only: bool = True
) -> dict[str, float]:
    """Aggregate one metric by planner across scenario rows."""

    values: dict[str, list[float]] = defaultdict(list)
    for row in campaign["seed_rows"]:
        if eligible_only and _row_status(row) in NON_PROMOTABLE_ROW_STATUSES:
            continue
        value = _safe_float(_metric_summary(row, metric).get("mean"))
        if value is not None:
            values[str(row.get("planner_key", "unknown"))].append(value)
    return {planner: _mean(metric_values) for planner, metric_values in sorted(values.items())}


def _family_planner_values(
    campaign: dict[str, Any], metric: str, *, eligible_only: bool = True
) -> dict[str, dict[str, float]]:
    """Aggregate one metric by scenario family and planner."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in campaign["seed_rows"]:
        if eligible_only and _row_status(row) in NON_PROMOTABLE_ROW_STATUSES:
            continue
        value = _safe_float(_metric_summary(row, metric).get("mean"))
        if value is None:
            continue
        grouped[_scenario_family(row)][str(row.get("planner_key", "unknown"))].append(value)
    return {
        family: {planner: _mean(values) for planner, values in sorted(planner_values.items())}
        for family, planner_values in sorted(grouped.items())
    }


def _winner(values: dict[str, float], *, higher_is_better: bool) -> str:
    """Return the deterministic winning planner for a value mapping."""

    return rank_order(values, higher_is_better=higher_is_better)[0]


def _metric_summary(row: dict[str, Any], metric: str) -> dict[str, Any]:
    """Return one metric summary from a seed-variability row."""

    summary = row.get("summary")
    if not isinstance(summary, dict):
        return {}
    metric_summary = summary.get(metric)
    return metric_summary if isinstance(metric_summary, dict) else {}


def _ci_width(summary: dict[str, Any]) -> float | None:
    """Resolve CI width from bounds or half-width."""

    low = _safe_float(summary.get("ci_low"))
    high = _safe_float(summary.get("ci_high"))
    if low is not None and high is not None:
        return max(0.0, high - low)
    half_width = _safe_float(summary.get("ci_half_width"))
    if half_width is not None:
        return max(0.0, 2.0 * half_width)
    return None


def _scenario_family(row: dict[str, Any]) -> str:
    """Resolve scenario family from row metadata with a stable fallback."""

    family = row.get("scenario_family") or row.get("archetype")
    if isinstance(family, str) and family:
        return family
    scenario_id = str(row.get("scenario_id", "unknown"))
    return scenario_id.rsplit("_", 1)[0] if "_" in scenario_id else scenario_id


def _seed_row_sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return deterministic sort key for seed-variability rows."""

    return (
        str(row.get("scenario_id", "")),
        str(row.get("planner_key", "")),
        str(row.get("kinematics", "")),
    )


def _higher_is_better(metric: str) -> bool:
    """Return metric ranking direction."""

    return metric not in LOWER_IS_BETTER


def _mean(values: list[float]) -> float:
    """Compute arithmetic mean for a non-empty list."""

    return float(sum(values) / len(values))


def _safe_float(value: Any) -> float | None:
    """Parse a finite float from CSV/JSON data."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    """Parse an integer from CSV/JSON data."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _truthy_int(value: Any) -> int:
    """Parse a binary outcome value into 0 or 1."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return int(bool(value))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON mapping."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as dictionaries."""

    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with deterministic field order."""

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_markdown(payload: dict[str, Any]) -> str:
    """Render a compact Markdown summary."""

    lines = [
        "# Seed Sufficiency Analysis",
        "",
        f"- Campaigns: `{payload['summary']['campaign_count']}`",
        f"- Rank metric: `{payload['rank_metric']}`",
        f"- Ranking instability rows: `{payload['summary']['ranking_instability_rows']}`",
        f"- Scenario-family winner changes: `{payload['summary']['scenario_family_winner_changes']}`",
        "",
        "## Caveats",
        "",
    ]
    caveats = payload.get("caveats") or []
    if caveats:
        lines.extend(f"- {caveat}" for caveat in caveats)
    else:
        lines.append("- No underpowered or unstable comparisons detected by this diagnostic.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `seed_count_interval_width.csv`",
            "- `headline_rank_stability_contract.json`",
            "- `headline_rank_stability_pairwise.csv`",
            "- `planner_rank_stability.csv`",
            "- `scenario_family_instability.csv`",
            "- `outcome_counts.csv`",
            "- `fig_seed_interval_width.png`",
            "",
            "This diagnostic does not prove scenario coverage or external validity.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_interval_figure(
    path: Path, rows: list[dict[str, Any]], metrics: tuple[str, ...]
) -> None:
    """Write a CI-width by seed-count PNG figure."""

    fig, ax = plt.subplots(figsize=(7, 4))
    has_plots = False
    for metric in metrics:
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            if row["metric"] != metric:
                continue
            width = _safe_float(row.get("ci_width"))
            if width is not None:
                grouped[int(row["seed_count"])].append(width)
        if not grouped:
            continue
        xs = sorted(grouped)
        ys = [_mean(grouped[x]) for x in xs]
        ax.plot(xs, ys, marker="o", label=metric)
        has_plots = True
    ax.set_xlabel("Seed count")
    ax.set_ylabel("Mean CI width")
    ax.set_title("Seed count versus interval width")
    ax.grid(True, alpha=0.25)
    if has_plots:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        action="append",
        default=None,
        help="Campaign root containing reports/seed_* artifacts. Repeat for multiple schedules.",
    )
    parser.add_argument(
        "--campaign-output-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Slurm/output container to scan for campaign roots with reports/"
            "seed_variability_by_scenario.json. Repeatable."
        ),
    )
    parser.add_argument(
        "--campaign-id",
        action="append",
        default=None,
        help="Optional substring filter for campaign roots discovered under --campaign-output-root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output report directory.")
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=DEFAULT_METRICS,
        help="Metric to include. Repeatable. Defaults to success, collisions, and snqi.",
    )
    parser.add_argument("--rank-metric", default="snqi", help="Metric used for ranking stability.")
    parser.add_argument(
        "--advisory-seed-threshold",
        type=int,
        default=2,
        help="Campaigns below this seed count are marked advisory.",
    )
    parser.add_argument(
        "--headline-min-seed-budget",
        type=int,
        default=20,
        help="Minimum seed budget required before headline rank-stability can be unblocked.",
    )
    parser.add_argument(
        "--headline-required-durable-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Required durable root for headline evidence. Repeat for S20/S30 roots; "
            "missing or empty roots emit blocked_pending_s20_s30."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the seed-sufficiency analysis CLI."""

    args = _build_parser().parse_args(argv)
    metrics = tuple(args.metrics) if args.metrics else DEFAULT_METRICS
    campaign_roots = resolve_campaign_roots(
        campaign_roots=args.campaign_root,
        campaign_output_roots=args.campaign_output_root,
        campaign_ids=args.campaign_id,
    )
    payload = analyze_seed_sufficiency(
        campaign_roots,
        args.output_dir,
        metrics=metrics,
        rank_metric=args.rank_metric,
        advisory_seed_threshold=args.advisory_seed_threshold,
        headline_min_seed_budget=args.headline_min_seed_budget,
        headline_required_durable_roots=tuple(args.headline_required_durable_root),
    )
    print(f"wrote seed sufficiency analysis: {args.output_dir}")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
