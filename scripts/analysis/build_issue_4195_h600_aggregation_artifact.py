#!/usr/bin/env python3
"""Build the issue #4195 h600 interpretation aggregation evidence artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

SCHEMA_VERSION = "issue_4195_h600_aggregation.v1"
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_3810_h600_interpretation_2026-07")
DEFAULT_CONFIRM_REPORTS = Path("output/issue3810-h600-longhorizon-confirm-run/13268/reports")
DEFAULT_EXTENDED_REPORTS = Path("output/issue3810-h600-extroster-run/13273/reports")
DEFAULT_H500_S20_REPORTS = Path(
    "docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports"
)
LAUNCH_PACKET = Path("configs/benchmarks/issue_3810_long_horizon_snqi_launch_packet.yaml")
SNQI_WEIGHTS = {
    "w_success": 0.19045845847432735,
    "w_time": 0.09491099136070058,
    "w_collisions": 0.10483542508043969,
    "w_near": 0.30825830332144416,
    "w_comfort": 0.17983060763794978,
    "w_force_exceed": 0.0692114485473155,
    "w_jerk": 0.05249476557782281,
}
SNQI_COMPONENTS = {
    "success": ("w_success", 1.0),
    "collision": ("w_collisions", -1.0),
    "near_miss": ("w_near", -1.0),
    "comfort": ("w_comfort", -1.0),
}
EXPOSURE_REQUIRED_COLUMNS = (
    "interaction_exposure_share",
    "robot_motion_share_before_first_clearance",
    "first_clearance_step",
    "low_exposure_success",
)
EXPOSURE_PROVENANCE_COLUMN = "interaction_exposure_source"


@dataclass(frozen=True)
class MetricSpec:
    """Metric source contract for the h600 aggregation table."""

    name: str
    seed_column: str | None
    summary_mean_field: str
    aggregate_metric: str


METRICS = (
    MetricSpec("snqi", "snqi", "snqi_mean", "snqi"),
    MetricSpec("success", "success", "success_mean", "success"),
    MetricSpec("collision", "collision", "collisions_mean", "collisions"),
    MetricSpec("near_miss", "near_miss", "near_misses_mean", "near_misses"),
    MetricSpec("comfort", None, "comfort_exposure_mean", "comfort_exposure"),
)


def _public_path(path: Path) -> str:
    """Return a repo-public path without local home/worktree prefixes."""
    resolved = path.resolve()
    for anchor in ("docs", "configs", "scripts", "tests", "output"):
        if anchor in resolved.parts:
            index = resolved.parts.index(anchor)
            return str(Path(*resolved.parts[index:]))
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def _json_default(value: Any) -> Any:
    """Serialize non-JSON-native values used by artifact payloads."""
    if isinstance(value, Path):
        return _public_path(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object type {type(value).__name__} is not JSON serializable")


def _scrub_public_paths(value: Any) -> Any:
    """Recursively remove local machine prefixes from path-like strings."""
    if isinstance(value, dict):
        return {key: _scrub_public_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_public_paths(item) for item in value]
    if isinstance(value, str) and "/home/" in value:
        return _public_path(Path(value))
    return value

    if isinstance(value, Path):
        return _public_path(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from ``path`` and fail closed on incompatible payloads."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read seed episode rows as dictionaries."""

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_or_none(value: Any) -> float | None:
    """Parse a finite float, returning ``None`` when the source value is unavailable."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _format_float(value: float | None, *, digits: int = 6) -> str:
    """Format optional floats for stable CSV and Markdown output."""

    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _rank_rows(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Return rows with stable one-based ranks assigned."""

    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row[value_key]) if descending else float(row[value_key]),
            str(row["planner_key"]),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _metric_lookup(rows: list[dict[str, Any]], *, job_id: str) -> dict[str, dict[str, float]]:
    """Return planner -> metric -> value lookup for one generated metric table."""

    lookup: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if str(row["job_id"]) != str(job_id):
            continue
        value = row.get("seed_mean")
        if value is None:
            value = row.get("summary_mean")
        parsed = _float_or_none(value)
        if parsed is not None:
            lookup[str(row["planner_key"])][str(row["metric"])] = parsed
    return lookup


def _normalization(values: list[float]) -> dict[str, float]:
    """Return min/max normalization block for an h600 metric surface."""

    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"min": 0.0, "max": 0.0, "denominator": 1.0}
    minimum = min(finite)
    maximum = max(finite)
    denominator = maximum - minimum
    if abs(denominator) < 1e-12:
        denominator = 1.0
    return {"min": minimum, "max": maximum, "denominator": denominator}


def _component_score(value: float, normalization: dict[str, float], direction: float) -> float:
    """Normalize one component into a desirability or penalty contribution."""

    normalized = (value - normalization["min"]) / normalization["denominator"]
    normalized = min(1.0, max(0.0, normalized))
    if direction > 0:
        return normalized
    return normalized


def _build_h600_recalibration(
    rows: list[dict[str, Any]],
    *,
    h500_reports: Path,
) -> dict[str, Any]:
    """Build analysis-only h600 SNQI recalibration and h500 reversal checks."""

    h600_metrics = _metric_lookup(rows, job_id="13268")
    normalizations = {
        metric: _normalization(
            [
                planner_metrics[metric]
                for planner_metrics in h600_metrics.values()
                if metric in planner_metrics
            ]
        )
        for metric in SNQI_COMPONENTS
    }
    planner_rows: list[dict[str, Any]] = []
    for planner_key, metrics in sorted(h600_metrics.items()):
        missing = [metric for metric in SNQI_COMPONENTS if metric not in metrics]
        original_snqi = metrics.get("snqi")
        recalibrated = None
        if not missing:
            score = SNQI_WEIGHTS["w_success"] * _component_score(
                metrics["success"], normalizations["success"], 1.0
            )
            score -= SNQI_WEIGHTS["w_collisions"] * _component_score(
                metrics["collision"], normalizations["collision"], -1.0
            )
            score -= SNQI_WEIGHTS["w_near"] * _component_score(
                metrics["near_miss"], normalizations["near_miss"], -1.0
            )
            score -= SNQI_WEIGHTS["w_comfort"] * _component_score(
                metrics["comfort"], normalizations["comfort"], -1.0
            )
            recalibrated = score
        planner_rows.append(
            {
                "planner_key": planner_key,
                "original_h600_snqi": original_snqi,
                "recalibrated_h600_snqi": recalibrated,
                "success": metrics.get("success"),
                "collision": metrics.get("collision"),
                "near_miss": metrics.get("near_miss"),
                "comfort": metrics.get("comfort"),
                "status": "ok" if recalibrated is not None else "missing_component_metrics",
                "missing_metrics": missing,
            }
        )

    original_ranked = _rank_rows(
        [row for row in planner_rows if row["original_h600_snqi"] is not None],
        value_key="original_h600_snqi",
    )
    recalibrated_ranked = _rank_rows(
        [row for row in planner_rows if row["recalibrated_h600_snqi"] is not None],
        value_key="recalibrated_h600_snqi",
    )
    original_ranks = {row["planner_key"]: row["rank"] for row in original_ranked}
    recalibrated_ranks = {row["planner_key"]: row["rank"] for row in recalibrated_ranked}
    h500 = _load_h500_rankings(h500_reports)
    h500_snqi_ranks = {
        row["planner_key"]: row["rank"] for row in h500.get("rankings", {}).get("snqi", [])
    }
    comparison_rows = []
    for planner_key in sorted(set(original_ranks) | set(recalibrated_ranks) | set(h500_snqi_ranks)):
        h600_original_rank = original_ranks.get(planner_key)
        h600_recalibrated_rank = recalibrated_ranks.get(planner_key)
        h500_rank = h500_snqi_ranks.get(planner_key)
        status = (
            "ok"
            if h500_rank is not None and h600_recalibrated_rank is not None
            else "not_evaluable"
        )
        h500_delta = (
            h600_recalibrated_rank - h500_rank
            if h500_rank is not None and h600_recalibrated_rank is not None
            else None
        )
        comparison_rows.append(
            {
                "planner_key": planner_key,
                "h500_snqi_rank": h500_rank,
                "h600_original_snqi_rank": h600_original_rank,
                "h600_recalibrated_snqi_rank": h600_recalibrated_rank,
                "original_to_recalibrated_delta": (
                    h600_recalibrated_rank - h600_original_rank
                    if h600_original_rank is not None and h600_recalibrated_rank is not None
                    else None
                ),
                "h500_to_h600_recalibrated_delta": h500_delta,
                "decision_reversal": abs(h500_delta) > 1 if h500_delta is not None else None,
                "h500_to_h600_recalibrated_stability": (
                    "rank_flip"
                    if h500_delta is not None and abs(h500_delta) > 1
                    else "stable"
                    if h500_delta is not None
                    else "not_evaluable"
                ),
                "decision_reversal_status": status,
            }
        )
    return {
        "schema_version": f"{SCHEMA_VERSION}.snqi_recalibration",
        "status": "ok" if planner_rows else "blocked_no_h600_rows",
        "claim_boundary": "diagnostic-only analysis; canonical SNQI weights/baselines are not overwritten",
        "launch_packet": _public_path(LAUNCH_PACKET),
        "source_h600_job_id": "13268",
        "h500_source": h500,
        "weights": SNQI_WEIGHTS,
        "normalization": normalizations,
        "planner_rows": planner_rows,
        "rankings": {
            "original_h600_snqi": original_ranked,
            "recalibrated_h600_snqi": recalibrated_ranked,
        },
        "decision_reversal_rows": comparison_rows,
    }


def _planner_row_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return planner rows keyed by planner key."""

    rows = summary.get("planner_rows")
    if not isinstance(rows, list):
        raise ValueError("campaign_summary.json missing planner_rows list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        planner_key = str(row.get("planner_key") or row.get("algo") or "")
        if planner_key:
            result[planner_key] = row
    return result


def _aggregate_metric_blocks(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return per-planner aggregate metric blocks from campaign run summaries."""

    result: dict[str, dict[str, Any]] = {}
    for run in summary.get("runs") or []:
        if not isinstance(run, dict):
            continue
        planner = run.get("planner") or {}
        planner_key = str(planner.get("key") or planner.get("algo") or "")
        aggregates = run.get("aggregates")
        if not planner_key or not isinstance(aggregates, dict):
            continue
        block = aggregates.get(planner_key)
        if isinstance(block, dict):
            result[planner_key] = block
    return result


def _group_seed_values(
    rows: list[dict[str, str]],
    *,
    planner_key: str,
    metric_column: str,
) -> dict[str, float]:
    """Return per-seed mean values for a planner and metric column."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("planner_key") != planner_key:
            continue
        seed = row.get("seed")
        value = _float_or_none(row.get(metric_column))
        if seed is None or value is None:
            continue
        grouped[str(seed)].append(value)
    return {seed: mean(values) for seed, values in sorted(grouped.items()) if values}


def _bootstrap_ci(
    values: list[float],
    *,
    samples: int,
    confidence: float,
    seed_material: str,
) -> tuple[float | None, float | None]:
    """Return a deterministic percentile bootstrap CI over seed-level means."""

    if not values:
        return (None, None)
    if len(values) == 1:
        return (values[0], values[0])
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    draws: list[float] = []
    count = len(values)
    for _ in range(samples):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        draws.append(mean(sample))
    draws.sort()
    alpha = 1.0 - confidence
    low_index = max(0, int((alpha / 2.0) * samples))
    high_index = min(samples - 1, int((1.0 - alpha / 2.0) * samples))
    return (draws[low_index], draws[high_index])


def _aggregate_ci_from_summary(
    aggregate_blocks: dict[str, dict[str, Any]],
    *,
    planner_key: str,
    metric: MetricSpec,
) -> tuple[float | None, float | None]:
    """Return campaign-summary aggregate CI for metrics absent from seed rows."""

    metric_block = aggregate_blocks.get(planner_key, {}).get(metric.aggregate_metric)
    if not isinstance(metric_block, dict):
        return (None, None)
    ci = metric_block.get("mean_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        return (None, None)
    return (_float_or_none(ci[0]), _float_or_none(ci[1]))


def _build_rows_for_run(
    *,
    job_id: str,
    run_label: str,
    reports_dir: Path,
    bootstrap_samples: int,
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build metric rows and metadata for one h600 campaign report directory."""

    summary_path = reports_dir / "campaign_summary.json"
    seed_rows_path = reports_dir / "seed_episode_rows.csv"
    summary = _read_json(summary_path)
    seed_rows = _read_csv_rows(seed_rows_path)
    campaign = summary.get("campaign")
    if not isinstance(campaign, dict):
        raise ValueError(f"{summary_path} missing campaign object")
    planner_rows = _planner_row_map(summary)
    aggregate_blocks = _aggregate_metric_blocks(summary)
    seed_columns = set(seed_rows[0].keys()) if seed_rows else set()

    rows: list[dict[str, Any]] = []
    for planner_key in sorted(planner_rows):
        planner_summary = planner_rows[planner_key]
        planner_episode_count = sum(1 for row in seed_rows if row.get("planner_key") == planner_key)
        for metric in METRICS:
            summary_mean = _float_or_none(planner_summary.get(metric.summary_mean_field))
            seed_values: dict[str, float] = {}
            ci_low: float | None
            ci_high: float | None
            source = "seed_episode_rows"
            status = "ok"
            if metric.seed_column and metric.seed_column in seed_columns:
                seed_values = _group_seed_values(
                    seed_rows,
                    planner_key=planner_key,
                    metric_column=metric.seed_column,
                )
                ci_low, ci_high = _bootstrap_ci(
                    list(seed_values.values()),
                    samples=bootstrap_samples,
                    confidence=confidence,
                    seed_material=f"{job_id}:{planner_key}:{metric.name}",
                )
                if not seed_values:
                    status = "unavailable_no_seed_values"
            else:
                ci_low, ci_high = _aggregate_ci_from_summary(
                    aggregate_blocks,
                    planner_key=planner_key,
                    metric=metric,
                )
                source = "campaign_summary_aggregate"
                status = "no_seed_episode_column"

            seed_mean = mean(seed_values.values()) if seed_values else None
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "job_id": job_id,
                    "run_label": run_label,
                    "planner_key": planner_key,
                    "metric": metric.name,
                    "scenario_matrix_hash": campaign.get("scenario_matrix_hash"),
                    "comparability_mapping_hash": campaign.get("comparability_mapping_hash"),
                    "evidence_status": campaign.get("evidence_status"),
                    "benchmark_success": campaign.get("benchmark_success"),
                    "episodes": planner_episode_count or planner_summary.get("episodes"),
                    "seed_count": len(seed_values),
                    "source": source,
                    "value_status": status,
                    "summary_mean": summary_mean,
                    "seed_mean": seed_mean,
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "seed_values": seed_values,
                }
            )

    metadata = {
        "job_id": job_id,
        "run_label": run_label,
        "reports_dir": reports_dir,
        "campaign_summary": summary_path,
        "seed_episode_rows": seed_rows_path,
        "campaign": {
            "campaign_id": campaign.get("campaign_id"),
            "evidence_status": campaign.get("evidence_status"),
            "benchmark_success": campaign.get("benchmark_success"),
            "scenario_matrix": campaign.get("scenario_matrix"),
            "scenario_matrix_hash": campaign.get("scenario_matrix_hash"),
            "comparability_mapping_hash": campaign.get("comparability_mapping_hash"),
            "created_at_utc": campaign.get("created_at_utc"),
            "git_hash": campaign.get("git_hash"),
        },
        "source_sha256": {
            "campaign_summary.json": _sha256(summary_path),
            "seed_episode_rows.csv": _sha256(seed_rows_path),
        },
        "planner_keys": sorted(planner_rows),
        "seed_episode_columns": sorted(seed_columns),
    }
    return rows, metadata


def _comparability_report(metadatas: list[dict[str, Any]]) -> dict[str, Any]:
    """Build confirm-vs-extended comparability report for shared planner arms."""

    by_job = {str(item["job_id"]): item for item in metadatas}
    if len(by_job) != 2:
        raise ValueError("comparability report expects exactly two jobs")
    first, second = metadatas
    shared_planners = sorted(set(first["planner_keys"]) & set(second["planner_keys"]))
    first_campaign = first["campaign"]
    second_campaign = second["campaign"]
    matrix_match = first_campaign.get("scenario_matrix_hash") == second_campaign.get(
        "scenario_matrix_hash"
    )
    mapping_match = first_campaign.get("comparability_mapping_hash") == second_campaign.get(
        "comparability_mapping_hash"
    )
    shared_rows = [
        {
            "planner_key": planner,
            "job_ids": [first["job_id"], second["job_id"]],
            "scenario_matrix_hash_match": matrix_match,
            "comparability_mapping_hash_match": mapping_match,
            "status": "comparable" if matrix_match and mapping_match else "not_comparable",
        }
        for planner in shared_planners
    ]
    return {
        "schema_version": f"{SCHEMA_VERSION}.comparability",
        "status": "pass" if shared_rows and matrix_match and mapping_match else "fail",
        "shared_planner_count": len(shared_planners),
        "shared_planners": shared_planners,
        "job_campaigns": {str(item["job_id"]): item["campaign"] for item in metadatas},
        "checks": {
            "scenario_matrix_hash_match": matrix_match,
            "comparability_mapping_hash_match": mapping_match,
        },
        "rows": shared_rows,
    }


def _load_h500_rankings(reports_dir: Path) -> dict[str, Any]:
    """Load the best tracked h500 comparison surface, if available."""

    summary_path = reports_dir / "campaign_summary.json"
    if not summary_path.exists():
        return {
            "status": "missing",
            "reports_dir": _public_path(reports_dir),
            "reason": "campaign_summary.json not present; h500/S20 reversal checks unavailable",
            "rankings": {},
        }
    summary = _read_json(summary_path)
    campaign = summary.get("campaign") or {}
    planner_rows = [
        row
        for row in summary.get("planner_rows") or []
        if isinstance(row, dict) and row.get("planner_key")
    ]
    metric_fields = {
        "snqi": ("snqi_mean", True),
        "success": ("success_mean", True),
        "collision": ("collisions_mean", False),
        "near_miss": ("near_misses_mean", False),
    }
    rankings: dict[str, list[dict[str, Any]]] = {}
    for metric, (field, descending) in metric_fields.items():
        metric_rows = []
        for row in planner_rows:
            value = _float_or_none(row.get(field))
            if value is None:
                continue
            metric_rows.append({"planner_key": str(row["planner_key"]), "value": value})
        rankings[metric] = _rank_rows(metric_rows, value_key="value", descending=descending)
    return {
        "status": "available_s10_h500_not_s20",
        "reports_dir": _public_path(reports_dir),
        "campaign_id": campaign.get("campaign_id"),
        "scenario_matrix": campaign.get("scenario_matrix"),
        "scenario_matrix_hash": campaign.get("scenario_matrix_hash"),
        "seed_budget": "S10",
        "requested_comparison": "h500/S20",
        "caveat": (
            "Tracked h500 source is issue #1454 S10/h500 candidate evidence; no tracked "
            "S20 h500 result bundle was available in this checkout."
        ),
        "rankings": rankings,
    }


def _build_horizon_sensitivity(
    rows: list[dict[str, Any]],
    *,
    h500_reports: Path,
) -> dict[str, Any]:
    """Compare h600 rankings with tracked h500 rankings where possible."""

    h600_metrics = _metric_lookup(rows, job_id="13268")
    h500 = _load_h500_rankings(h500_reports)
    h600_rankings: dict[str, list[dict[str, Any]]] = {}
    for metric, descending in {
        "snqi": True,
        "success": True,
        "collision": False,
        "near_miss": False,
    }.items():
        metric_rows = [
            {"planner_key": planner_key, "value": values[metric]}
            for planner_key, values in h600_metrics.items()
            if metric in values
        ]
        h600_rankings[metric] = _rank_rows(metric_rows, value_key="value", descending=descending)

    comparison_rows = []
    for metric, h600_ranked in h600_rankings.items():
        h600_ranks = {row["planner_key"]: row["rank"] for row in h600_ranked}
        h500_ranked = h500.get("rankings", {}).get(metric, [])
        h500_ranks = {row["planner_key"]: row["rank"] for row in h500_ranked}
        for planner_key in sorted(set(h600_ranks) & set(h500_ranks)):
            delta = h600_ranks[planner_key] - h500_ranks[planner_key]
            comparison_rows.append(
                {
                    "metric": metric,
                    "planner_key": planner_key,
                    "h500_rank": h500_ranks[planner_key],
                    "h600_rank": h600_ranks[planner_key],
                    "rank_delta": delta,
                    "stability": "stable" if abs(delta) <= 1 else "rank_flip",
                }
            )
    return {
        "schema_version": f"{SCHEMA_VERSION}.horizon_sensitivity",
        "status": "ok" if comparison_rows else "blocked_no_shared_rankings",
        "claim_boundary": (
            "diagnostic-only ranking comparison; not a causal horizon-only ablation because "
            "scenario hash, seed budget, planner roster, and SNQI calibration differ"
        ),
        "h600_source_job_id": "13268",
        "h500_source": h500,
        "h600_rankings": h600_rankings,
        "comparison_rows": comparison_rows,
    }


def _build_exposure_diagnostics(metadatas: list[dict[str, Any]]) -> dict[str, Any]:
    """Return exposure diagnostics or a fail-closed missing-fields report."""

    runs = []
    overall_missing: set[str] = set()
    for metadata in metadatas:
        columns = set(metadata.get("seed_episode_columns") or [])
        missing = [column for column in EXPOSURE_REQUIRED_COLUMNS if column not in columns]
        overall_missing.update(missing)
        seed_rows_path = Path(metadata["seed_episode_rows"])
        seed_rows = _read_csv_rows(seed_rows_path) if seed_rows_path.exists() else []
        derivable_rows = 0
        not_derivable_rows = 0
        for row in seed_rows:
            has_required_values = all(
                str(row.get(column, "")) for column in EXPOSURE_REQUIRED_COLUMNS
            )
            if has_required_values:
                derivable_rows += 1
            else:
                not_derivable_rows += 1
        provenance_values = sorted(
            {
                str(row.get(EXPOSURE_PROVENANCE_COLUMN, ""))
                for row in seed_rows
                if str(row.get(EXPOSURE_PROVENANCE_COLUMN, ""))
            }
        )
        runs.append(
            {
                "job_id": metadata["job_id"],
                "run_label": metadata["run_label"],
                "seed_episode_rows": metadata["seed_episode_rows"],
                "available_columns": sorted(columns),
                "required_columns": list(EXPOSURE_REQUIRED_COLUMNS),
                "missing_required_fields": missing,
                "backfill_policy": "derive_from_retained_episode_rows_only_no_imputation",
                "derivable_episode_rows": derivable_rows,
                "not_derivable_episode_rows": not_derivable_rows,
                "exposure_provenance_values": provenance_values,
                "status": "blocked_missing_required_fields"
                if missing
                else "ready_for_episode_level_scan",
            }
        )
    return {
        "schema_version": f"{SCHEMA_VERSION}.interaction_exposure",
        "status": "blocked_missing_required_fields"
        if overall_missing
        else "ready_for_episode_level_scan",
        "claim_boundary": (
            "interaction-exposure coverage is diagnostic-only and must be computed from episode-level "
            "fields; absent fields are not imputed"
        ),
        "required_fields": list(EXPOSURE_REQUIRED_COLUMNS),
        "missing_required_fields": sorted(overall_missing),
        "backfill_policy": "derive_from_retained_episode_rows_only_no_imputation",
        "runs": runs,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the metric table CSV."""

    columns = [
        "job_id",
        "run_label",
        "planner_key",
        "metric",
        "scenario_matrix_hash",
        "comparability_mapping_hash",
        "evidence_status",
        "benchmark_success",
        "episodes",
        "seed_count",
        "source",
        "value_status",
        "summary_mean",
        "seed_mean",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "seed_values_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["seed_values_json"] = json.dumps(row["seed_values"], sort_keys=True)
            for field in ("summary_mean", "seed_mean", "bootstrap_ci_low", "bootstrap_ci_high"):
                out[field] = _format_float(out.get(field))
            writer.writerow({column: out.get(column, "") for column in columns})


def _markdown_escape(value: Any) -> str:
    """Escape simple Markdown table cells."""

    return str(value).replace("|", "\\|")


def _write_metric_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact Markdown view of the metric rows."""

    columns = [
        "job_id",
        "planner_key",
        "metric",
        "source",
        "value_status",
        "summary_mean",
        "seed_mean",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "seed_values",
    ]
    lines = [
        "# H600 Planner Metric Aggregation",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cell_values = []
        for column in columns:
            if column == "seed_values":
                value = json.dumps(row["seed_values"], sort_keys=True)
            elif column in {"summary_mean", "seed_mean", "bootstrap_ci_low", "bootstrap_ci_high"}:
                value = _format_float(row.get(column))
            else:
                value = row.get(column, "")
            cell_values.append(_markdown_escape(value))
        lines.append("| " + " | ".join(cell_values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_comparability_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write comparability report as Markdown."""

    lines = [
        "# H600 Confirm vs Extended-Roster Comparability",
        "",
        f"- Status: `{report['status']}`",
        f"- Shared planner arms: {report['shared_planner_count']}",
        f"- Scenario matrix hash match: `{report['checks']['scenario_matrix_hash_match']}`",
        f"- Comparability mapping hash match: "
        f"`{report['checks']['comparability_mapping_hash_match']}`",
        "",
        "| planner_key | scenario_matrix_hash_match | comparability_mapping_hash_match | status |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            "| {planner_key} | {scenario_matrix_hash_match} | "
            "{comparability_mapping_hash_match} | {status} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_snqi_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write SNQI recalibration Markdown summary."""

    lines = [
        "# H600 SNQI Recalibration Bundle",
        "",
        "- Evidence status: `diagnostic-only`.",
        f"- Status: `{report['status']}`.",
        "- Canonical camera-ready SNQI weights and baselines are not overwritten.",
        f"- H500 comparison source status: `{report['h500_source']['status']}`.",
        "",
        "| planner_key | original_h600_rank | recalibrated_h600_rank | h500_rank | rank_delta | stability | status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["decision_reversal_rows"]:
        lines.append(
            "| {planner_key} | {h600_original_snqi_rank} | {h600_recalibrated_snqi_rank} | "
            "{h500_snqi_rank} | {h500_to_h600_recalibrated_delta} | "
            "{h500_to_h600_recalibrated_stability} | {decision_reversal_status} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_horizon_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write horizon-sensitivity Markdown summary."""

    lines = [
        "# H600 vs H500 Horizon-Sensitivity Report",
        "",
        "- Evidence status: `diagnostic-only`.",
        f"- Status: `{report['status']}`.",
        f"- H500 source status: `{report['h500_source']['status']}`.",
        "- Caveat: this is not a causal horizon-only ablation.",
        "",
        "| metric | planner_key | h500_rank | h600_rank | rank_delta | stability |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["comparison_rows"]:
        lines.append(
            "| {metric} | {planner_key} | {h500_rank} | {h600_rank} | {rank_delta} | "
            "{stability} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_exposure_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write interaction-exposure diagnostic Markdown summary."""

    lines = [
        "# Interaction-Exposure Diagnostics",
        "",
        "- Evidence status: `diagnostic-only`.",
        f"- Status: `{report['status']}`.",
        "- Episode-level exposure fields are required; missing values are not imputed.",
        "",
        "| job_id | run_label | status | missing_required_fields |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["runs"]:
        lines.append(
            "| {job_id} | {run_label} | {status} | {missing} |".format(
                job_id=row["job_id"],
                run_label=row["run_label"],
                status=row["status"],
                missing=", ".join(row["missing_required_fields"]) or "none",
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_readme(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    comparability: dict[str, Any],
    snqi_recalibration: dict[str, Any],
    horizon_sensitivity: dict[str, Any],
    exposure_diagnostics: dict[str, Any],
) -> None:
    """Write the claim-boundary README for the evidence directory."""

    jobs = sorted({str(row["job_id"]) for row in rows})
    metric_count = len(rows)
    comfort_rows = [row for row in rows if row["metric"] == "comfort"]
    comfort_note = (
        "Comfort per-seed values are not present in `seed_episode_rows.csv`; those rows preserve "
        "the campaign-summary aggregate mean confidence interval and are marked "
        "`no_seed_episode_column`."
    )
    text = f"""# Issue 4195 h600 Aggregation Artifact

This directory contains diagnostic-only h600 interpretation artifacts for jobs {", ".join(jobs)}.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- Scope: per-planner aggregation plus issue #4195 checklist items 3-5.
- This artifact does not assert benchmark success, dissertation-ready evidence, paper-grade evidence, or a planner ranking claim.
- No full benchmark campaign, Slurm submission, graphics processing unit job, retention decision, or dissertation claim edit was run for this slice.
- SNQI recalibration, horizon-sensitivity, and exposure diagnostics are diagnostic-only interpretation artifacts.
- Comparability is limited to shared planner arms whose `scenario_matrix_hash` and `comparability_mapping_hash` match across the two campaign summaries.

## Contents

- `planner_metric_summary.csv`: one row per job, planner, and metric with per-seed values where available plus bootstrap confidence intervals.
- `planner_metric_summary.md`: Markdown rendering of the same rows.
- `comparability_check.json` and `comparability_check.md`: shared-arm scenario matrix comparability check.
- `snqi_recalibration_bundle.json` and `snqi_recalibration_report.md`: analysis-only h600 recalibration and h500 reversal checks.
- `horizon_sensitivity_report.json` and `horizon_sensitivity_report.md`: h600-vs-h500 rank-stability and rank-flip diagnostic.
- `interaction_exposure_diagnostics.json` and `interaction_exposure_diagnostics.md`: episode-level exposure coverage readiness; fail-closed when required fields are absent.
- `source_manifest.json`: input paths, campaign metadata, and source file SHA-256 digests.
- `SHA256SUMS`: checksums for generated files in this directory.

## Notes

- Metric rows: {metric_count}.
- Shared-arm comparability status: `{comparability["status"]}`.
- SNQI recalibration status: `{snqi_recalibration["status"]}`.
- Horizon-sensitivity status: `{horizon_sensitivity["status"]}`.
- Interaction-exposure status: `{exposure_diagnostics["status"]}`.
- Comfort rows: {len(comfort_rows)}.
- {comfort_note}
"""
    path.write_text(text, encoding="utf-8")


def _write_sha256sums(output_dir: Path) -> None:
    """Write SHA256SUMS for generated files, excluding SHA256SUMS itself."""

    lines = []
    for path in sorted(output_dir.iterdir()):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        lines.append(f"{_sha256(path)}  {_public_path(path)}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_existing_metric_rows(path: Path) -> list[dict[str, Any]]:
    """Read retained #4199 planner metric rows back into internal row shape."""

    rows: list[dict[str, Any]] = []
    for row in _read_csv_rows(path):
        parsed = dict(row)
        for field in ("summary_mean", "seed_mean", "bootstrap_ci_low", "bootstrap_ci_high"):
            parsed[field] = _float_or_none(parsed.get(field))
        parsed["seed_values"] = json.loads(parsed.pop("seed_values_json", "{}") or "{}")
        rows.append(parsed)
    return rows


def extend_existing_artifact(
    *,
    output_dir: Path,
    h500_s20_reports: Path,
) -> dict[str, Any]:
    """Add issue #4195 interpretation outputs to an existing aggregation artifact."""

    metric_path = output_dir / "planner_metric_summary.csv"
    comparability_path = output_dir / "comparability_check.json"
    manifest_path = output_dir / "source_manifest.json"
    rows = _read_existing_metric_rows(metric_path)
    comparability = _read_json(comparability_path)
    manifest = _read_json(manifest_path)
    metadatas = _scrub_public_paths(manifest.get("runs") or [])
    manifest["runs"] = metadatas
    snqi_recalibration = _build_h600_recalibration(rows, h500_reports=h500_s20_reports)
    horizon_sensitivity = _build_horizon_sensitivity(rows, h500_reports=h500_s20_reports)
    exposure_diagnostics = _build_exposure_diagnostics(metadatas)
    outputs = {
        "snqi_recalibration_bundle.json": output_dir / "snqi_recalibration_bundle.json",
        "snqi_recalibration_report.md": output_dir / "snqi_recalibration_report.md",
        "horizon_sensitivity_report.json": output_dir / "horizon_sensitivity_report.json",
        "horizon_sensitivity_report.md": output_dir / "horizon_sensitivity_report.md",
        "interaction_exposure_diagnostics.json": output_dir
        / "interaction_exposure_diagnostics.json",
        "interaction_exposure_diagnostics.md": output_dir / "interaction_exposure_diagnostics.md",
    }
    outputs["snqi_recalibration_bundle.json"].write_text(
        json.dumps(snqi_recalibration, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_snqi_markdown(outputs["snqi_recalibration_report.md"], snqi_recalibration)
    outputs["horizon_sensitivity_report.json"].write_text(
        json.dumps(horizon_sensitivity, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_horizon_markdown(outputs["horizon_sensitivity_report.md"], horizon_sensitivity)
    outputs["interaction_exposure_diagnostics.json"].write_text(
        json.dumps(exposure_diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_exposure_markdown(outputs["interaction_exposure_diagnostics.md"], exposure_diagnostics)
    manifest["h500_s20_reports"] = h500_s20_reports
    manifest["generated_outputs"] = sorted(
        set(manifest.get("generated_outputs") or []) | set(outputs)
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        output_dir / "README.md",
        rows=rows,
        comparability=comparability,
        snqi_recalibration=snqi_recalibration,
        horizon_sensitivity=horizon_sensitivity,
        exposure_diagnostics=exposure_diagnostics,
    )
    _write_sha256sums(output_dir)
    return {
        "status": "ok" if comparability["status"] == "pass" else "comparability_failed",
        "output_dir": output_dir,
        "row_count": len(rows),
        "snqi_recalibration_status": snqi_recalibration["status"],
        "horizon_sensitivity_status": horizon_sensitivity["status"],
        "interaction_exposure_status": exposure_diagnostics["status"],
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }


def build_artifact(
    *,
    confirm_reports: Path,
    extended_reports: Path,
    h500_s20_reports: Path,
    output_dir: Path,
    bootstrap_samples: int,
    confidence: float,
) -> dict[str, Any]:
    """Build all issue #4195 aggregation artifact files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    metadatas: list[dict[str, Any]] = []
    for job_id, label, reports_dir in (
        ("13268", "confirm", confirm_reports),
        ("13273", "extended_roster", extended_reports),
    ):
        rows, metadata = _build_rows_for_run(
            job_id=job_id,
            run_label=label,
            reports_dir=reports_dir,
            bootstrap_samples=bootstrap_samples,
            confidence=confidence,
        )
        all_rows.extend(rows)
        metadatas.append(metadata)

    comparability = _comparability_report(metadatas)
    snqi_recalibration = _build_h600_recalibration(all_rows, h500_reports=h500_s20_reports)
    horizon_sensitivity = _build_horizon_sensitivity(all_rows, h500_reports=h500_s20_reports)
    exposure_diagnostics = _build_exposure_diagnostics(metadatas)
    outputs = {
        "planner_metric_summary.csv": output_dir / "planner_metric_summary.csv",
        "planner_metric_summary.md": output_dir / "planner_metric_summary.md",
        "comparability_check.json": output_dir / "comparability_check.json",
        "comparability_check.md": output_dir / "comparability_check.md",
        "snqi_recalibration_bundle.json": output_dir / "snqi_recalibration_bundle.json",
        "snqi_recalibration_report.md": output_dir / "snqi_recalibration_report.md",
        "horizon_sensitivity_report.json": output_dir / "horizon_sensitivity_report.json",
        "horizon_sensitivity_report.md": output_dir / "horizon_sensitivity_report.md",
        "interaction_exposure_diagnostics.json": output_dir
        / "interaction_exposure_diagnostics.json",
        "interaction_exposure_diagnostics.md": output_dir / "interaction_exposure_diagnostics.md",
        "source_manifest.json": output_dir / "source_manifest.json",
        "README.md": output_dir / "README.md",
    }
    _write_csv(outputs["planner_metric_summary.csv"], all_rows)
    _write_metric_markdown(outputs["planner_metric_summary.md"], all_rows)
    outputs["comparability_check.json"].write_text(
        json.dumps(comparability, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_comparability_markdown(outputs["comparability_check.md"], comparability)
    outputs["snqi_recalibration_bundle.json"].write_text(
        json.dumps(snqi_recalibration, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_snqi_markdown(outputs["snqi_recalibration_report.md"], snqi_recalibration)
    outputs["horizon_sensitivity_report.json"].write_text(
        json.dumps(horizon_sensitivity, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_horizon_markdown(outputs["horizon_sensitivity_report.md"], horizon_sensitivity)
    outputs["interaction_exposure_diagnostics.json"].write_text(
        json.dumps(exposure_diagnostics, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_exposure_markdown(outputs["interaction_exposure_diagnostics.md"], exposure_diagnostics)
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.source_manifest",
        "bootstrap": {"samples": bootstrap_samples, "confidence": confidence},
        "runs": _scrub_public_paths(metadatas),
        "h500_s20_reports": h500_s20_reports,
        "generated_outputs": sorted(outputs),
    }
    outputs["source_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_readme(
        outputs["README.md"],
        rows=all_rows,
        comparability=comparability,
        snqi_recalibration=snqi_recalibration,
        horizon_sensitivity=horizon_sensitivity,
        exposure_diagnostics=exposure_diagnostics,
    )
    _write_sha256sums(output_dir)
    return {
        "status": "ok" if comparability["status"] == "pass" else "comparability_failed",
        "output_dir": output_dir,
        "row_count": len(all_rows),
        "comparability": comparability,
        "snqi_recalibration_status": snqi_recalibration["status"],
        "horizon_sensitivity_status": horizon_sensitivity["status"],
        "interaction_exposure_status": exposure_diagnostics["status"],
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-reports", type=Path, default=DEFAULT_CONFIRM_REPORTS)
    parser.add_argument("--extended-reports", type=Path, default=DEFAULT_EXTENDED_REPORTS)
    parser.add_argument("--h500-s20-reports", type=Path, default=DEFAULT_H500_S20_REPORTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--extend-existing",
        action="store_true",
        help="add items 3-5 outputs from an existing #4199 aggregation artifact",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line artifact builder."""

    args = _parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.extend_existing:
        result = extend_existing_artifact(
            output_dir=args.output_dir,
            h500_s20_reports=args.h500_s20_reports,
        )
    else:
        result = build_artifact(
            confirm_reports=args.confirm_reports,
            extended_reports=args.extended_reports,
            h500_s20_reports=args.h500_s20_reports,
            output_dir=args.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
