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
DEFAULT_CONFIRM_REPORTS = Path(
    "/home/luttkule/git/robot_sf_ll7/output/issue3810-h600-longhorizon-confirm-run/13268/reports"
)
DEFAULT_EXTENDED_REPORTS = Path(
    "/home/luttkule/git/robot_sf_ll7/output/issue3810-h600-extroster-run/13273/reports"
)


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


def _json_default(value: Any) -> Any:
    """Serialize non-JSON-native values used by artifact payloads."""

    if isinstance(value, Path):
        return value.as_posix()
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


def _write_readme(path: Path, *, rows: list[dict[str, Any]], comparability: dict[str, Any]) -> None:
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

This directory contains a diagnostic-only aggregation artifact for h600 jobs {", ".join(jobs)}.

## Claim Boundary

- Evidence status: `diagnostic-only`.
- Scope: mechanical per-planner aggregation from retrieved local artifacts for issue #4195 checklist item 1.
- This artifact does not assert benchmark success, dissertation-ready evidence, paper-grade evidence, or a planner ranking claim.
- No full benchmark campaign, Slurm submission, graphics processing unit job, retention decision, SNQI recalibration, horizon-sensitivity synthesis, or dissertation claim edit was run for this slice.
- Comparability is limited to shared planner arms whose `scenario_matrix_hash` and `comparability_mapping_hash` match across the two campaign summaries.

## Contents

- `planner_metric_summary.csv`: one row per job, planner, and metric with per-seed values where available plus bootstrap confidence intervals.
- `planner_metric_summary.md`: Markdown rendering of the same rows.
- `comparability_check.json` and `comparability_check.md`: shared-arm scenario matrix comparability check.
- `source_manifest.json`: input paths, campaign metadata, and source file SHA-256 digests.
- `SHA256SUMS`: checksums for generated files in this directory.

## Notes

- Metric rows: {metric_count}.
- Shared-arm comparability status: `{comparability["status"]}`.
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
        lines.append(f"{_sha256(path)}  {path.as_posix()}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_artifact(
    *,
    confirm_reports: Path,
    extended_reports: Path,
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
    outputs = {
        "planner_metric_summary.csv": output_dir / "planner_metric_summary.csv",
        "planner_metric_summary.md": output_dir / "planner_metric_summary.md",
        "comparability_check.json": output_dir / "comparability_check.json",
        "comparability_check.md": output_dir / "comparability_check.md",
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
    manifest = {
        "schema_version": f"{SCHEMA_VERSION}.source_manifest",
        "bootstrap": {"samples": bootstrap_samples, "confidence": confidence},
        "runs": metadatas,
        "generated_outputs": sorted(outputs),
    }
    outputs["source_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    _write_readme(outputs["README.md"], rows=all_rows, comparability=comparability)
    _write_sha256sums(output_dir)
    return {
        "status": "ok" if comparability["status"] == "pass" else "comparability_failed",
        "output_dir": output_dir,
        "row_count": len(all_rows),
        "comparability": comparability,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-reports", type=Path, default=DEFAULT_CONFIRM_REPORTS)
    parser.add_argument("--extended-reports", type=Path, default=DEFAULT_EXTENDED_REPORTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line artifact builder."""

    args = _parse_args(argv)
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be positive")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    result = build_artifact(
        confirm_reports=args.confirm_reports,
        extended_reports=args.extended_reports,
        output_dir=args.output_dir,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
