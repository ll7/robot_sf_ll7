#!/usr/bin/env python3
"""Build the issue #4239 h600 SNQI weight-set ranking packet.

The builder is deliberately fail-closed: every active Social Navigation Quality Index (SNQI) term
must be present in retained raw rows before rank or agreement artifacts are emitted. Compact
aggregate summaries are recorded as provenance but are not treated as zero-filled substitutes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.rank_metrics import (
    kendall_tau,
    rank_by,
    rank_order,
    spearman_from_rank_maps,
)
from robot_sf.benchmark.snqi.compute import WEIGHT_NAMES, compute_snqi, normalize_metric
from robot_sf.benchmark.snqi.weights_inventory import build_inventory_report
from robot_sf.benchmark.snqi_scalarization_sensitivity import (
    input_file_provenance,
    load_baseline_mapping,
    load_weight_mapping,
)

SCHEMA_VERSION = "h600-snqi-weight-set-ranking.v1"
PREFLIGHT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.preflight"
REPORT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.report"
BLOCKED_MISSING_TERMS = "blocked_missing_snqi_terms"
BLOCKED_MISSING_SOURCE = "blocked_missing_source_files"
BLOCKED_SHARED_MISMATCH = "blocked_shared_planner_mismatch"
READY = "ready"

TERM_BY_WEIGHT = {
    "w_success": "success",
    "w_time": "time_to_goal_norm",
    "w_collisions": "collisions",
    "w_near": "near_misses",
    "w_comfort": "comfort_exposure",
    "w_force_exceed": "force_exceed_events",
    "w_jerk": "jerk_mean",
}
ALIASES = {
    "collisions": ("collisions", "collision"),
    "near_misses": ("near_misses", "near_miss"),
    "comfort_exposure": ("comfort_exposure", "comfort"),
}
COMPARISON_METRICS = tuple(TERM_BY_WEIGHT.values())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rel(path: Path) -> str:
    root = _repo_root()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config at {path} must be a mapping, got {type(data).__name__}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported config schema_version: {data.get('schema_version')!r}")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_value(row: dict[str, str], metric: str) -> float | None:
    for key in ALIASES.get(metric, (metric,)):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _load_weight_sets(config: dict[str, Any]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for spec in config["weight_sets"]:
        source_path = spec.get("path")
        path = _repo_root() / source_path if source_path else None
        weights = load_weight_mapping(path)
        loaded.append(
            {
                "id": spec["id"],
                "kind": spec.get("kind", "unknown"),
                "path": source_path,
                "sha256": _sha256(path) if path is not None else None,
                "weights": weights,
                "required_terms": [
                    TERM_BY_WEIGHT[name]
                    for name in WEIGHT_NAMES
                    if math.isfinite(float(weights.get(name, 0.0)))
                    and float(weights.get(name, 0.0)) != 0.0
                ],
                "caveat": spec.get("caveat") or spec.get("note") or "",
            }
        )
    return loaded


def _manifest_runs(config: dict[str, Any], source_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    requested = {(str(item["job_id"]), item["run_label"]) for item in config["run_inputs"]}
    runs: list[dict[str, Any]] = []
    for run in source_manifest.get("runs", []):
        key = (str(run.get("job_id")), str(run.get("run_label")))
        if key in requested:
            seed_rows = _repo_root() / str(run.get("seed_episode_rows", ""))
            campaign = run.get("campaign") or {}
            runs.append(
                {
                    "job_id": str(run["job_id"]),
                    "run_label": str(run["run_label"]),
                    "seed_episode_rows": seed_rows,
                    "campaign_summary": _repo_root() / str(run.get("campaign_summary", "")),
                    "scenario_matrix_hash": campaign.get("scenario_matrix_hash", ""),
                    "comparability_mapping_hash": campaign.get("comparability_mapping_hash", ""),
                }
            )
    return runs


def _read_rows(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for run in runs:
        path = run["seed_episode_rows"]
        if not path.exists():
            missing.append(
                {
                    "job_id": run["job_id"],
                    "run_label": run["run_label"],
                    "path": _rel(path),
                    "status": "missing",
                }
            )
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["_job_id"] = run["job_id"]
                row["_run_label"] = run["run_label"]
                row["_scenario_matrix_hash"] = str(run["scenario_matrix_hash"])
                row["_comparability_mapping_hash"] = str(run["comparability_mapping_hash"])
                rows.append(row)
    return rows, missing


def _planner_key(row: dict[str, str]) -> str:
    for key in ("planner_key", "planner", "algo"):
        value = row.get(key)
        if value:
            return value
    return "unknown"


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["_run_label"]), _planner_key(row))].append(row)

    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for (run_label, planner), planner_rows in grouped.items():
        metrics: dict[str, float] = {}
        availability: dict[str, bool] = {}
        for metric in COMPARISON_METRICS:
            values = [_metric_value(row, metric) for row in planner_rows]
            clean = [value for value in values if value is not None]
            availability[metric] = len(clean) == len(planner_rows)
            if clean:
                metrics[metric] = float(math.fsum(clean) / len(clean))
        aggregates[(run_label, planner)] = {
            "planner_key": planner,
            "run_label": run_label,
            "job_id": str(planner_rows[0]["_job_id"]),
            "episodes": len(planner_rows),
            "seed_count": len(
                {row.get("seed", "") for row in planner_rows if row.get("seed", "") != ""}
            ),
            "scenario_matrix_hash": str(planner_rows[0]["_scenario_matrix_hash"]),
            "comparability_mapping_hash": str(planner_rows[0]["_comparability_mapping_hash"]),
            "metrics": metrics,
            "availability": availability,
        }
    return aggregates


def _missing_term_issues(
    aggregates: dict[tuple[str, str], dict[str, Any]], weight_sets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for weight_set in weight_sets:
        for (_run_label, planner), aggregate in sorted(aggregates.items()):
            missing = [
                term
                for term in weight_set["required_terms"]
                if not aggregate["availability"].get(term, False)
            ]
            if missing:
                issues.append(
                    {
                        "code": "missing_required_term",
                        "weight_set_id": weight_set["id"],
                        "planner_key": planner,
                        "source_run": aggregate["run_label"],
                        "missing_terms": missing,
                    }
                )
    return issues


def _deduplicate(
    aggregates: dict[tuple[str, str], dict[str, Any]], *, tolerance: float
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_planner: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (run_label, planner), aggregate in aggregates.items():
        by_planner[planner][run_label] = aggregate

    selected: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for planner, run_map in sorted(by_planner.items()):
        confirm = run_map.get("confirm")
        extended = run_map.get("extended_roster")
        if confirm and extended:
            selected[planner] = confirm
            for metric in COMPARISON_METRICS:
                confirm_value = confirm["metrics"].get(metric)
                extended_value = extended["metrics"].get(metric)
                if confirm_value is None or extended_value is None:
                    status = "missing_metric"
                    delta = ""
                else:
                    delta_float = abs(confirm_value - extended_value)
                    delta = f"{delta_float:.12g}"
                    status = "match" if delta_float <= tolerance else "mismatch"
                row = {
                    "planner_key": planner,
                    "metric": metric,
                    "confirm_value": "" if confirm_value is None else f"{confirm_value:.12g}",
                    "extended_value": "" if extended_value is None else f"{extended_value:.12g}",
                    "abs_delta": delta,
                    "tolerance": f"{tolerance:.12g}",
                    "status": status,
                    "selected_source": "confirm_13268",
                }
                audit.append(row)
                if status == "mismatch":
                    mismatches.append(row)
        elif confirm:
            selected[planner] = confirm
            audit.append(_single_source_audit_row(planner, "confirm_13268", tolerance))
        elif extended:
            selected[planner] = extended
            audit.append(_single_source_audit_row(planner, "extended_13273", tolerance))
    return selected, audit, mismatches


def _single_source_audit_row(planner: str, source: str, tolerance: float) -> dict[str, str]:
    return {
        "planner_key": planner,
        "metric": "planner_source",
        "confirm_value": "",
        "extended_value": "",
        "abs_delta": "",
        "tolerance": f"{tolerance:.12g}",
        "status": "single_source",
        "selected_source": source,
    }


def _source_label(aggregate: dict[str, Any]) -> str:
    run_prefix = (
        "extended" if aggregate["run_label"] == "extended_roster" else aggregate["run_label"]
    )
    return f"{run_prefix}_{aggregate['job_id']}"


def _component_contributions(
    metrics: dict[str, float], weights: dict[str, float], baseline: dict[str, dict[str, float]]
) -> dict[str, float]:
    return {
        "w_success": weights["w_success"] * metrics["success"],
        "w_time": -weights["w_time"] * metrics["time_to_goal_norm"],
        "w_collisions": -weights["w_collisions"]
        * normalize_metric("collisions", metrics["collisions"], baseline),
        "w_near": -weights["w_near"]
        * normalize_metric("near_misses", metrics["near_misses"], baseline),
        "w_comfort": -weights["w_comfort"] * metrics["comfort_exposure"],
        "w_force_exceed": -weights["w_force_exceed"]
        * normalize_metric("force_exceed_events", metrics["force_exceed_events"], baseline),
        "w_jerk": -weights["w_jerk"]
        * normalize_metric("jerk_mean", metrics["jerk_mean"], baseline),
    }


def _rank_tables(
    selected: dict[str, dict[str, Any]],
    weight_sets: list[dict[str, Any]],
    baseline: dict[str, dict[str, float]],
    baseline_path: Path,
) -> tuple[
    list[dict[str, Any]], dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]
]:
    rows: list[dict[str, Any]] = []
    scores_by_weight_set: dict[str, dict[str, float]] = {}
    contributions_by_weight_set: dict[str, dict[str, dict[str, float]]] = {}
    baseline_sha = _sha256(baseline_path)
    for weight_set in weight_sets:
        scores: dict[str, float] = {}
        contributions_by_planner: dict[str, dict[str, float]] = {}
        for planner, aggregate in selected.items():
            score = compute_snqi(aggregate["metrics"], weight_set["weights"], baseline)
            scores[planner] = score
            contributions_by_planner[planner] = _component_contributions(
                aggregate["metrics"], weight_set["weights"], baseline
            )
        ranks = rank_by(scores, higher_is_better=True)
        for planner in rank_order(scores, higher_is_better=True):
            aggregate = selected[planner]
            rows.append(
                {
                    "weight_set_id": weight_set["id"],
                    "weight_source_path": weight_set["path"] or "synthetic_uniform_1p0",
                    "weight_sha256": weight_set["sha256"] or "",
                    "baseline_path": _rel(baseline_path),
                    "baseline_sha256": baseline_sha,
                    "planner_key": planner,
                    "rank": f"{ranks[planner]:.6g}",
                    "snqi_score": f"{scores[planner]:.12g}",
                    "source_run": _source_label(aggregate),
                    "job_id": aggregate["job_id"],
                    "episodes": aggregate["episodes"],
                    "seed_count": aggregate["seed_count"],
                    "scenario_matrix_hash": aggregate["scenario_matrix_hash"],
                    "comparability_mapping_hash": aggregate["comparability_mapping_hash"],
                    "evidence_status": "diagnostic-only",
                    "term_availability_status": "complete_raw_terms",
                    "caveat": weight_set["caveat"],
                }
            )
        scores_by_weight_set[weight_set["id"]] = scores
        contributions_by_weight_set[weight_set["id"]] = contributions_by_planner
    return rows, scores_by_weight_set, contributions_by_weight_set


def _agreement_rows(scores_by_weight_set: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids = sorted(scores_by_weight_set)
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1 :]:
            left_scores = scores_by_weight_set[left_id]
            right_scores = scores_by_weight_set[right_id]
            common = sorted(set(left_scores) & set(right_scores))
            left_ranks = rank_by({key: left_scores[key] for key in common}, higher_is_better=True)
            right_ranks = rank_by({key: right_scores[key] for key in common}, higher_is_better=True)
            left_order = rank_order(
                {key: left_scores[key] for key in common}, higher_is_better=True
            )
            right_order = rank_order(
                {key: right_scores[key] for key in common}, higher_is_better=True
            )
            reversals = _pairwise_reversal_count(left_order, right_order)
            total_pairs = len(common) * (len(common) - 1) / 2
            rows.append(
                {
                    "left_weight_set": left_id,
                    "right_weight_set": right_id,
                    "planner_count": len(common),
                    "common_planners": ";".join(common),
                    "spearman": _format_optional(
                        spearman_from_rank_maps(left_ranks, right_ranks, degenerate=None)
                    ),
                    "kendall_tau": _format_optional(
                        kendall_tau(left_order, right_order, degenerate=None)
                    ),
                    "pairwise_reversal_count": reversals,
                    "pairwise_disagreement_rate": _format_optional(
                        (reversals / total_pairs) if total_pairs else None
                    ),
                    "top1_same": bool(
                        left_order and right_order and left_order[0] == right_order[0]
                    ),
                    "top3_jaccard": _format_optional(_top_jaccard(left_order, right_order, n=3)),
                    "caveat": "diagnostic-only h600 three-seed artifact",
                }
            )
    return rows


def _pairwise_reversal_count(left_order: list[str], right_order: list[str]) -> int:
    right_index = {planner: index for index, planner in enumerate(right_order)}
    reversals = 0
    for index, planner in enumerate(left_order):
        for other in left_order[index + 1 :]:
            if right_index[planner] > right_index[other]:
                reversals += 1
    return reversals


def _top_jaccard(left_order: list[str], right_order: list[str], *, n: int) -> float | None:
    left = set(left_order[:n])
    right = set(right_order[:n])
    if not left and not right:
        return None
    return len(left & right) / len(left | right)


def _format_optional(value: float | None) -> str:
    return "" if value is None else f"{value:.12g}"


def _write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _write_markdown_table(
    path: Path, rows: list[dict[str, Any]], headers: list[str], title: str
) -> None:
    lines = [
        f"# {title}",
        "",
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(header, "")) for header in headers) + "|")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_preflight(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preflight_report(
    *,
    status: str,
    config_path: Path,
    config: dict[str, Any],
    issues: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    source_files_missing: list[dict[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "issue": 4239,
        "status": status,
        "claim_boundary": config["claim_boundary"],
        "config": input_file_provenance(config_path),
        "source_manifest": input_file_provenance(_repo_root() / config["source_manifest"]),
        "source_files_missing": source_files_missing,
        "runs": [
            {
                "job_id": run["job_id"],
                "run_label": run["run_label"],
                "seed_episode_rows": _rel(run["seed_episode_rows"]),
                "seed_episode_rows_exists": run["seed_episode_rows"].exists(),
            }
            for run in runs
        ],
        "issues": issues,
        "generated_at": generated_at,
    }


def _write_report(  # noqa: PLR0913
    path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    preflight: dict[str, Any],
    weight_sets: list[dict[str, Any]],
    baseline_path: Path,
    rank_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    contributions: dict[str, dict[str, dict[str, float]]],
    generated_at: str,
) -> None:
    inventory = build_inventory_report(_repo_root())
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "issue": 4239,
        "status": "ok",
        "claim_boundary": config["claim_boundary"],
        "three_seed_caveat": "h600 source artifacts retain three seeds per planner arm.",
        "config": input_file_provenance(config_path),
        "baseline": input_file_provenance(baseline_path),
        "preflight": preflight,
        "weight_sets": [
            {
                "id": item["id"],
                "path": item["path"],
                "sha256": item["sha256"],
                "weights": item["weights"],
                "required_terms": item["required_terms"],
                "caveat": item["caveat"],
            }
            for item in weight_sets
        ],
        "weight_inventory_source_summary": inventory.source_summary,
        "rank_rows": rank_rows,
        "pairwise_agreement": agreement_rows,
        "component_contributions": contributions,
        "generated_at": generated_at,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_diss_snippet(
    path: Path,
    rank_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    evidence_dir: Path,
) -> None:
    top_rows = [row for row in rank_rows if float(row["rank"]) <= 3.0]
    lines = [
        "# diss#331 h600 SNQI Weight-Set Ranking Snippet",
        "",
        "Diagnostic h600 per-weight-set analysis only: this closes the evidence-gap shape for "
        "Social Navigation Quality Index (SNQI) rank preservation on retained jobs 13268 and 13273, "
        "inherits the three-seed caveat, and does not choose canonical weights or edit claims.",
        "",
        "## Top-3 Rows",
        "",
        "|weight_set_id|rank|planner_key|snqi_score|source_run|",
        "|---|---|---|---|---|",
    ]
    for row in top_rows:
        lines.append(
            f"|{row['weight_set_id']}|{row['rank']}|{row['planner_key']}|"
            f"{row['snqi_score']}|{row['source_run']}|"
        )
    lines.extend(
        [
            "",
            "## Pairwise Agreement",
            "",
            "|left_weight_set|right_weight_set|spearman|kendall_tau|pairwise_disagreement_rate|top1_same|top3_jaccard|",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in agreement_rows:
        lines.append(
            f"|{row['left_weight_set']}|{row['right_weight_set']}|{row['spearman']}|"
            f"{row['kendall_tau']}|{row['pairwise_disagreement_rate']}|"
            f"{row['top1_same']}|{row['top3_jaccard']}|"
        )
    lines.extend(
        [
            "",
            "Artifacts: "
            f"`{_rel(evidence_dir / 'snqi_weight_set_h600_rank_table.csv')}`, "
            f"`{_rel(evidence_dir / 'snqi_weight_set_h600_pairwise_agreement.csv')}`, "
            f"`{_rel(evidence_dir / 'snqi_weight_set_h600_report.json')}`, and "
            f"`{_rel(evidence_dir / 'SHA256SUMS')}`.",
            "",
            "Author's canonical-weight ruling remains unchanged until decided separately.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _update_readme(path: Path) -> None:
    text = (
        path.read_text(encoding="utf-8")
        if path.exists()
        else "# Issue #4195 h600 Aggregation Artifact\n"
    )
    marker = "## Issue #4239 SNQI Weight-Set Ranking Packet"
    block = """## Issue #4239 SNQI Weight-Set Ranking Packet

The issue #4239 packet is diagnostic-only h600 Social Navigation Quality Index (SNQI)
weight-set ranking support for jobs `13268` and `13273`. It de-duplicates shared planner arms,
keeps the three-seed caveat, and does not choose canonical weights, edit paper or dissertation
claims, run campaigns, submit Slurm or graphics processing unit jobs, or copy raw output trees.

New compact artifacts, when the fail-closed preflight is ready, use the
`snqi_weight_set_h600_*` prefix and are checksummed in `SHA256SUMS`.
"""
    if marker in text:
        before = text.split(marker, 1)[0].rstrip()
        text = before + "\n\n" + block
    else:
        text = text.rstrip() + "\n\n" + block
    path.write_text(text, encoding="utf-8")


def _update_source_manifest(path: Path, generated_paths: list[Path], *, status: str) -> None:
    data = _load_json(path)
    outputs = list(dict.fromkeys(data.get("generated_outputs") or []))
    for generated_path in generated_paths:
        name = generated_path.name
        if name not in outputs:
            outputs.append(name)
    data["generated_outputs"] = outputs
    data["issue_4239_weight_set_packet"] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "artifacts": [_rel(path) for path in generated_paths],
        "claim_boundary": "diagnostic-only; no canonical weight decision or claim edit",
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _update_sha256s(path: Path, evidence_dir: Path) -> None:
    files = sorted(
        item for item in evidence_dir.iterdir() if item.is_file() and item.name != "SHA256SUMS"
    )
    lines = [f"{_sha256(item)}  {_rel(item)}" for item in files]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_packet(config_path: Path, output_dir: Path, generated_at: str) -> int:  # noqa: PLR0915
    """Build or fail-closed preflight the h600 SNQI weight-set packet."""
    config = _load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = _repo_root() / config["source_manifest"]
    source_manifest = _load_json(source_manifest_path)
    runs = _manifest_runs(config, source_manifest)
    weight_sets = _load_weight_sets(config)
    baseline_path = _repo_root() / config["baseline"]["path"]
    baseline = load_baseline_mapping(baseline_path)
    rows, missing_sources = _read_rows(runs)
    aggregates = _aggregate_rows(rows)
    missing_term_issues = _missing_term_issues(aggregates, weight_sets)

    preflight_path = output_dir / "snqi_weight_set_h600_preflight.json"
    readme_path = output_dir / "README.md"
    sha_path = output_dir / "SHA256SUMS"
    generated_paths = [preflight_path]

    if missing_sources:
        issues = [{"code": BLOCKED_MISSING_SOURCE, **item} for item in missing_sources]
        preflight = _preflight_report(
            status=BLOCKED_MISSING_SOURCE,
            config_path=config_path,
            config=config,
            issues=issues,
            runs=runs,
            source_files_missing=missing_sources,
            generated_at=generated_at,
        )
        _write_preflight(preflight_path, preflight)
        _update_readme(readme_path)
        _update_source_manifest(
            source_manifest_path, generated_paths, status=BLOCKED_MISSING_SOURCE
        )
        _update_sha256s(sha_path, output_dir)
        return 2

    if missing_term_issues:
        preflight = _preflight_report(
            status=BLOCKED_MISSING_TERMS,
            config_path=config_path,
            config=config,
            issues=missing_term_issues,
            runs=runs,
            source_files_missing=[],
            generated_at=generated_at,
        )
        _write_preflight(preflight_path, preflight)
        _update_readme(readme_path)
        _update_source_manifest(source_manifest_path, generated_paths, status=BLOCKED_MISSING_TERMS)
        _update_sha256s(sha_path, output_dir)
        return 2

    dedup_config = config.get("deduplication") or {}
    selected, audit_rows, mismatches = _deduplicate(
        aggregates, tolerance=float(dedup_config.get("tolerance", 1e-9))
    )
    audit_headers = [
        "planner_key",
        "metric",
        "confirm_value",
        "extended_value",
        "abs_delta",
        "tolerance",
        "status",
        "selected_source",
    ]
    audit_csv = output_dir / "snqi_weight_set_h600_deduplication_audit.csv"
    audit_md = output_dir / "snqi_weight_set_h600_deduplication_audit.md"
    _write_csv(audit_csv, audit_rows, audit_headers)
    _write_markdown_table(audit_md, audit_rows, audit_headers, "h600 SNQI Deduplication Audit")
    generated_paths.extend([audit_csv, audit_md])

    if mismatches:
        preflight = _preflight_report(
            status=BLOCKED_SHARED_MISMATCH,
            config_path=config_path,
            config=config,
            issues=[{"code": BLOCKED_SHARED_MISMATCH, **row} for row in mismatches],
            runs=runs,
            source_files_missing=[],
            generated_at=generated_at,
        )
        _write_preflight(preflight_path, preflight)
        _update_readme(readme_path)
        _update_source_manifest(
            source_manifest_path, generated_paths, status=BLOCKED_SHARED_MISMATCH
        )
        _update_sha256s(sha_path, output_dir)
        return 2

    preflight = _preflight_report(
        status=READY,
        config_path=config_path,
        config=config,
        issues=[],
        runs=runs,
        source_files_missing=[],
        generated_at=generated_at,
    )
    _write_preflight(preflight_path, preflight)
    rank_rows, scores, contributions = _rank_tables(selected, weight_sets, baseline, baseline_path)
    agreement_rows = _agreement_rows(scores)

    rank_headers = [
        "weight_set_id",
        "weight_source_path",
        "weight_sha256",
        "baseline_path",
        "baseline_sha256",
        "planner_key",
        "rank",
        "snqi_score",
        "source_run",
        "job_id",
        "episodes",
        "seed_count",
        "scenario_matrix_hash",
        "comparability_mapping_hash",
        "evidence_status",
        "term_availability_status",
        "caveat",
    ]
    agreement_headers = [
        "left_weight_set",
        "right_weight_set",
        "planner_count",
        "common_planners",
        "spearman",
        "kendall_tau",
        "pairwise_reversal_count",
        "pairwise_disagreement_rate",
        "top1_same",
        "top3_jaccard",
        "caveat",
    ]

    rank_csv = output_dir / "snqi_weight_set_h600_rank_table.csv"
    rank_md = output_dir / "snqi_weight_set_h600_rank_table.md"
    agreement_csv = output_dir / "snqi_weight_set_h600_pairwise_agreement.csv"
    agreement_md = output_dir / "snqi_weight_set_h600_pairwise_agreement.md"
    report_json = output_dir / "snqi_weight_set_h600_report.json"
    diss_md = output_dir / "snqi_weight_set_h600_diss331_comment.md"
    _write_csv(rank_csv, rank_rows, rank_headers)
    _write_markdown_table(rank_md, rank_rows, rank_headers, "h600 SNQI Weight-Set Rank Table")
    _write_csv(agreement_csv, agreement_rows, agreement_headers)
    _write_markdown_table(
        agreement_md, agreement_rows, agreement_headers, "h600 SNQI Pairwise Rank Agreement"
    )
    _write_report(
        report_json,
        config=config,
        config_path=config_path,
        preflight=preflight,
        weight_sets=weight_sets,
        baseline_path=baseline_path,
        rank_rows=rank_rows,
        agreement_rows=agreement_rows,
        contributions=contributions,
        generated_at=generated_at,
    )
    _write_diss_snippet(diss_md, rank_rows, agreement_rows, output_dir)
    generated_paths.extend([rank_csv, rank_md, agreement_csv, agreement_md, report_json, diss_md])
    _update_readme(readme_path)
    _update_source_manifest(source_manifest_path, generated_paths, status="generated")
    _update_sha256s(sha_path, output_dir)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--generated-at",
        default="now",
        help="ISO timestamp or 'now'. Tests pass a fixed value for deterministic artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line packet builder."""
    args = _parse_args()
    generated_at = (
        datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        if args.generated_at == "now"
        else args.generated_at
    )
    return build_packet(args.config, args.output_dir, generated_at)


if __name__ == "__main__":
    raise SystemExit(main())
