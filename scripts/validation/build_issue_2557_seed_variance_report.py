#!/usr/bin/env python3
"""Build the diagnostic issue #2557 seed-variance report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

SCHEMA_VERSION = "issue-2557-seed-variance-report.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = Path(
    "docs/context/evidence/issue_2557_replica_readiness_packet_2026-06-29/packet.json"
)
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2557_seed_variance_2026-07")
DEFAULT_GENERATED_AT = "2026-07-02T00:00:00+02:00"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2557

CLAIM_BOUNDARY = (
    "Diagnostic seed-variance evidence only for issue #2557 / issue #791 leader "
    "fixed-seed queue-fill replicas. This artifact is not a benchmark-success claim, "
    "ranking claim, paper claim, or dissertation claim."
)

RESCUED_TOP_UPS: tuple[dict[str, Any], ...] = (
    {
        "job_id": 12917,
        "seed": 503,
        "wandb_url": "https://wandb.ai/ll7/robot_sf/runs/2uyaerc3",
        "snqi": -0.04095994684513475,
        "success_rate": 0.8285714285714286,
        "collision_rate": 0.17142857142857143,
        "best_snqi": 0.17280387782643425,
        "best_success_rate": 0.8857142857142857,
        "best_collision_rate": 0.11428571428571428,
        "best_eval_step": 7_864_320,
        "lineage": "manifest-incomplete",
        "caveat": (
            "Admitted by maintainer ruling 2026-07-02: full 10M training completed and "
            "final/best metrics were preserved, but post-training manifest serialization "
            "failed because evaluation_scenario_config resolved outside allowed artifact roots."
        ),
    },
    {
        "job_id": 12931,
        "seed": 504,
        "wandb_url": "https://wandb.ai/ll7/robot_sf/runs/5pdxsuvf",
        "snqi": 0.07361100470957616,
        "success_rate": 0.8571428571428571,
        "collision_rate": 0.14285714285714285,
        "best_snqi": 0.1836819570905286,
        "best_success_rate": 0.8857142857142857,
        "best_collision_rate": 0.11428571428571428,
        "best_eval_step": 7_864_320,
        "lineage": "manifest-incomplete",
        "caveat": (
            "Admitted by maintainer ruling 2026-07-02: full 10M training completed and "
            "final/best metrics were preserved, but post-training manifest serialization "
            "failed because evaluation_scenario_config resolved outside allowed artifact roots."
        ),
    },
    {
        "job_id": 12932,
        "seed": 502,
        "wandb_url": "https://wandb.ai/ll7/robot_sf/runs/klqes0h1",
        "snqi": 0.18344208910554777,
        "success_rate": 0.9,
        "collision_rate": 0.1,
        "best_snqi": 0.3380453317359332,
        "best_success_rate": 0.9285714285714286,
        "best_collision_rate": 0.07142857142857142,
        "best_eval_step": 9_961_472,
        "lineage": "manifest-incomplete",
        "caveat": (
            "Admitted by maintainer ruling 2026-07-02: full 10M training completed and "
            "final/best metrics were preserved, but post-training manifest serialization "
            "failed because evaluation_scenario_config resolved outside allowed artifact roots."
        ),
    },
)

METRICS = ("snqi", "success_rate", "collision_rate")


def _finite_float(value: Any, *, field: str, job_id: int) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"job {job_id}: {field} must be a finite number")
    return float(value)


def _clean_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = packet.get("completed_jobs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("packet.completed_jobs must be a non-empty list")

    provenance_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("packet.completed_jobs entries must be objects")
        if "seed" not in row or "job_id" not in row:
            raise ValueError("packet.completed_jobs entries must contain 'seed' and 'job_id'")
    for row in sorted(rows, key=lambda item: (int(item["seed"]), int(item["job_id"]))):
        job_id = int(row["job_id"])
        provenance_rows.append(
            {
                "job_id": job_id,
                "seed": int(row["seed"]),
                "lineage": "clean",
                "admission_status": "admitted_metrics_available",
                "metric_status": "available",
                "snqi": _finite_float(row.get("snqi"), field="snqi", job_id=job_id),
                "success_rate": _finite_float(
                    row.get("success_rate"), field="success_rate", job_id=job_id
                ),
                "collision_rate": _finite_float(
                    row.get("collision_rate"), field="collision_rate", job_id=job_id
                ),
                "best_snqi": None,
                "best_success_rate": None,
                "best_collision_rate": None,
                "best_eval_step": None,
                "wandb_url": row.get("wandb_url"),
                "source": "replica_readiness_packet.completed_jobs",
                "caveat": "",
            }
        )
    return provenance_rows


def _with_rescued_rows(clean_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(clean_rows)
    for rescued in RESCUED_TOP_UPS:
        rows.append(
            {
                "job_id": rescued["job_id"],
                "seed": rescued["seed"],
                "lineage": rescued["lineage"],
                "admission_status": "admitted_metrics_available",
                "metric_status": "available",
                "snqi": rescued["snqi"],
                "success_rate": rescued["success_rate"],
                "collision_rate": rescued["collision_rate"],
                "best_snqi": rescued["best_snqi"],
                "best_success_rate": rescued["best_success_rate"],
                "best_collision_rate": rescued["best_collision_rate"],
                "best_eval_step": rescued["best_eval_step"],
                "wandb_url": rescued["wandb_url"],
                "source": "wandb_summary_recovered_2026-07-02",
                "caveat": rescued["caveat"],
            }
        )
    return sorted(rows, key=lambda item: (int(item["seed"]), int(item["job_id"])))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    boot_means = [
        _mean([values[rng.randrange(len(values))] for _ in range(len(values))])
        for _ in range(samples)
    ]
    boot_means.sort()
    alpha = 1.0 - confidence
    low_index = max(0, math.floor((alpha / 2.0) * samples))
    high_index = min(samples - 1, math.ceil((1.0 - alpha / 2.0) * samples) - 1)
    return [boot_means[low_index], boot_means[high_index]]


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [row for row in rows if row["metric_status"] == "available"]
    summary: dict[str, Any] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in metric_rows]
        if values:
            summary[metric] = {
                "count": len(values),
                "mean": _mean(values),
                "std": _sample_std(values),
                "min": min(values),
                "max": max(values),
                "range": max(values) - min(values),
                "bootstrap_mean_ci95": _bootstrap_mean_ci(values),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        else:
            summary[metric] = {
                "count": 0,
                "mean": float("nan"),
                "std": 0.0,
                "min": float("nan"),
                "max": float("nan"),
                "range": 0.0,
                "bootstrap_mean_ci95": [float("nan"), float("nan")],
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
    return summary


def build_report(
    packet_path: Path | None = None,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the report payload from tracked packet metrics."""
    if packet_path is None:
        packet_path = DEFAULT_PACKET
    if generated_at is None:
        generated_at = DEFAULT_GENERATED_AT
    packet = _load_json(packet_path)
    clean_rows = _clean_rows(packet)
    rows = _with_rescued_rows(clean_rows)
    pending_rows = [row for row in rows if row["metric_status"] != "available"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "issue": 2557,
        "source_packet": str(packet_path),
        "source_packet_schema_version": packet.get("schema_version"),
        "claim_boundary": CLAIM_BOUNDARY,
        "maintainer_ruling": {
            "date": "2026-07-02",
            "admitted_rescued_jobs": [12917, 12931, 12932],
            "excluded_jobs": [12916],
            "no_reruns": True,
            "analysis_rule": (
                "Analyze all 17 admitted rows. Rescued manifest-incomplete top-ups use final "
                "W&B eval/* metrics for seed-variance summaries, with best/* metrics preserved "
                "as per-run provenance fields."
            ),
        },
        "counts": {
            "admitted_runs": len(rows),
            "clean_metric_runs": len(clean_rows),
            "manifest_incomplete_admitted_runs": len(RESCUED_TOP_UPS),
            "metrics_available_runs": len(rows) - len(pending_rows),
            "metrics_pending_runs": len(pending_rows),
        },
        "provenance_rows": rows,
        "seed_variance_summary": _metric_summary(rows),
        "pending_metric_recovery": [
            {
                "job_id": row["job_id"],
                "seed": row["seed"],
                "lineage": row["lineage"],
                "metric_status": row["metric_status"],
                "caveat": row["caveat"],
            }
            for row in pending_rows
        ],
        "recovered_metric_source": {
            "source": "W&B project ll7/robot_sf summary fields",
            "final_metrics": ["eval/snqi", "eval/success_rate", "eval/collision_rate"],
            "best_metrics": [
                "best/snqi",
                "best/success_rate",
                "best/collision_rate",
                "best/eval_step",
            ],
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "job_id",
        "seed",
        "lineage",
        "admission_status",
        "metric_status",
        "snqi",
        "success_rate",
        "collision_rate",
        "best_snqi",
        "best_success_rate",
        "best_collision_rate",
        "best_eval_step",
        "wandb_url",
        "source",
        "caveat",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field) if row.get(field) not in (None, "") else "na"
                    for field in fieldnames
                }
            )


def _readme(report: dict[str, Any]) -> str:
    counts = report["counts"]
    metrics = report["seed_variance_summary"]
    lines = [
        "# Issue #2557 Seed-Variance Diagnostic Report",
        "",
        "This packet summarizes admitted fixed-seed queue-fill replica evidence for issue #2557.",
        "",
        "## Claim Boundary",
        "",
        f"- {report['claim_boundary']}",
        "- No Slurm or GPU job was submitted while producing this packet.",
        "- Job 12916 is excluded, per maintainer ruling on 2026-07-02.",
        "- Jobs 12917, 12931, and 12932 are admitted with explicit `manifest-incomplete` caveats.",
        "",
        "## Counts",
        "",
        f"- Admitted rows: {counts['admitted_runs']}",
        f"- Metric-bearing rows: {counts['metrics_available_runs']}",
        f"- Metric-bearing clean rows: {counts['clean_metric_runs']}",
        f"- Admitted rows with pending metric recovery: {counts['metrics_pending_runs']}",
        "- Rescued top-up rows use final W&B `eval/*` metrics; `best/*` metrics are preserved "
        "in `per_run_provenance.csv` and `report.json`.",
        "",
        "## Metric Summary",
        "",
        "| metric | n | mean | std | range | bootstrap mean CI95 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in METRICS:
        item = metrics[metric]
        ci = item["bootstrap_mean_ci95"]
        label = (
            "[SNQI](../../../glossary.md#metrics-evaluation) (Social Navigation Quality Index)"
            if metric == "snqi"
            else f"`{metric}`"
        )
        lines.append(
            f"| {label} | {item['count']} | {item['mean']:.6f} | "
            f"{item['std']:.6f} | {item['range']:.6f} | "
            f"[{ci[0]:.6f}, {ci[1]:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `report.json`: structured diagnostic report.",
            "- `per_run_provenance.csv`: reviewable per-run provenance table.",
            "- `SHA256SUMS`: checksums for the packet files.",
            "",
            "Generated by:",
            "",
            "```bash",
            "uv run python scripts/validation/build_issue_2557_seed_variance_report.py",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_sha256sums(output_dir: Path, filenames: list[str]) -> None:
    lines = []
    for filename in filenames:
        path = output_dir / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            relative_path = path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative_path = filename
        lines.append(f"{digest}  {relative_path}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_artifact(report: dict[str, Any], output_dir: Path | None = None) -> None:
    """Write report, provenance CSV, README, and checksums."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "report.json", report)
    _write_csv(output_dir / "per_run_provenance.csv", report["provenance_rows"])
    (output_dir / "README.md").write_text(_readme(report), encoding="utf-8")
    _write_sha256sums(output_dir, ["README.md", "per_run_provenance.csv", "report.json"])


def main(argv: list[str] | None = None) -> int:
    """Run the command-line report builder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--generated-at",
        default=DEFAULT_GENERATED_AT,
        help="Stable timestamp for reviewable artifacts; pass 'now' for wall-clock UTC.",
    )
    args = parser.parse_args(argv)

    generated_at = (
        datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if args.generated_at == "now"
        else args.generated_at
    )
    report = build_report(args.packet, generated_at=generated_at)
    write_artifact(report, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
