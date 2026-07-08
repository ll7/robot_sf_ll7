#!/usr/bin/env python3
"""End-to-end Package A pipeline orchestrator (issue #3078).

Runs the seed-sufficiency analysis, held-out-family transfer validation,
and preliminary claim-card generation on supplied campaign data or synthetic
fixtures. Classifies the result under the issue #3078 fail-closed vocabulary.

This is the canonical entry point that assembles the Package A deliverables:

  - seed-sufficiency analysis (interval widths, rank stability, figure)
  - held-out-family partition validity + leakage audit structure
  - baseline table (benchmark-set summary)
  - transfer-delta table (benchmark-set vs held-out-family per planner)
  - transfer-delta figure (PNG)
  - preliminary claim card with issue result classification

When real campaign data is absent the pipeline falls back to synthetic fixture
campaigns and classifies the result as ``diagnostic`` rather than ``benchmark``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SCHEMA_VERSION = "package_a_pipeline_output.v1"
PACKAGE_A_CLASSIFICATION = (
    "benchmark",
    "diagnostic",
    "negative",
    "null",
    "invalid",
    "blocked",
)


# ---------------------------------------------------------------------------
# Fixture campaign generation
# ---------------------------------------------------------------------------


def _create_campaign_fixture(
    root: Path,
    *,
    seed_count: int,
    planners: dict[str, dict[str, float]],
    scenario_families: list[str],
    ci_half_width: float = 0.1,
) -> Path:
    """Create a minimal campaign fixture under ``root`` with ``reports/``.

    Args:
        root: Campaign root directory to create.
        seed_count: Number of seeds in this campaign.
        planners: Mapping of planner key to metric summaries.
        scenario_families: Scenario families to emit rows for.
        ci_half_width: Confidence interval half-width for metrics.
    """
    reports = root / "reports"
    root.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    variability_rows = []
    for family in scenario_families:
        scenario_id = f"{family}_s1"
        for planner_key, metrics in planners.items():
            summary = {
                metric: {
                    "mean": value,
                    "ci_half_width": ci_half_width,
                    "ci_low": value - ci_half_width,
                    "ci_high": value + ci_half_width,
                }
                for metric, value in metrics.items()
            }
            variability_rows.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_family": family,
                    "planner_key": planner_key,
                    "kinematics": "differential_drive",
                    "seed_count": seed_count,
                    "summary": summary,
                }
            )
            successes = int(metrics.get("success", 0.0) * seed_count)
            collisions = int(metrics.get("collisions", 0.0) * seed_count)
            snqi = metrics.get("snqi", 0.0)
            for seed_index in range(seed_count):
                seed_rows.append(
                    {
                        "episode_id": f"{planner_key}-{family}-{seed_index}",
                        "scenario_id": scenario_id,
                        "scenario_family": family,
                        "planner_key": planner_key,
                        "seed": 111 + seed_index,
                        "success": "1" if seed_index < successes else "0",
                        "collision": "1" if seed_index < collisions else "0",
                        "snqi": f"{snqi:.4f}",
                    }
                )

    (reports / "seed_variability_by_scenario.json").write_text(
        json.dumps(
            {
                "schema_version": "benchmark-seed-variability-by-scenario.v1",
                "metrics": ["success", "collisions", "snqi"],
                "rows": variability_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_seed_episode_rows(reports / "seed_episode_rows.csv", seed_rows)
    (reports / "statistical_sufficiency.json").write_text(
        json.dumps({"bootstrap": {"seed": 123}}) + "\n",
        encoding="utf-8",
    )
    return root


def _write_seed_episode_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Write seed episode rows CSV."""
    fieldnames = [
        "episode_id",
        "scenario_id",
        "scenario_family",
        "planner_key",
        "seed",
        "success",
        "collision",
        "snqi",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Baseline table
# ---------------------------------------------------------------------------


def _build_baseline_table(
    campaign_roots: list[Path],
    *,
    metrics: tuple[str, ...] = ("success", "collisions", "snqi"),
) -> list[dict[str, Any]]:
    """Build a compact baseline table from campaign roots.

    Returns one row per campaign-seed-count-planner with the requested metric
    means.  The table is deterministic and sorted.
    """
    rows: list[dict[str, Any]] = []
    for campaign_root in campaign_roots:
        payload = json.loads(
            (campaign_root / "reports" / "seed_variability_by_scenario.json").read_text(
                encoding="utf-8"
            )
        )
        for row in payload.get("rows", []):
            planner_key = str(row.get("planner_key", "unknown"))
            summary = row.get("summary") or {}
            baseline_row: dict[str, Any] = {
                "campaign": campaign_root.name,
                "scenario_id": str(row.get("scenario_id", "")),
                "scenario_family": str(row.get("scenario_family", "")),
                "planner_key": planner_key,
                "seed_count": row.get("seed_count", 0),
            }
            for metric in metrics:
                metric_summary = summary.get(metric) or {}
                mean = metric_summary.get("mean")
                ci_low = metric_summary.get("ci_low")
                ci_high = metric_summary.get("ci_high")
                baseline_row[f"{metric}_mean"] = _safe_float(mean)
                baseline_row[f"{metric}_ci_low"] = _safe_float(ci_low)
                baseline_row[f"{metric}_ci_high"] = _safe_float(ci_high)
            rows.append(baseline_row)
    rows.sort(
        key=lambda r: (
            r["campaign"],
            r["scenario_family"],
            r["planner_key"],
        )
    )
    return rows


# ---------------------------------------------------------------------------
# Transfer-delta computation
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Parse a finite float or return None."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _build_transfer_delta(
    benchmark_rows: list[dict[str, Any]],
    heldout_rows: list[dict[str, Any]],
    *,
    metric: str = "snqi",
) -> list[dict[str, Any]]:
    """Compute per-planner transfer deltas between benchmark-set and held-out-family.

    A transfer delta is the difference between the mean metric on benchmark-set
    and the mean metric on the held-out-family.  Positive values mean the
    planner performed better on the benchmark set than on held-out data (expected
    transfer loss when delta > 0 for SNQI-style metrics).
    """
    benchmark_map: dict[str, float | None] = {}
    for row in benchmark_rows:
        key = row["planner_key"]
        value = _safe_float(row.get(f"{metric}_mean"))
        if value is not None:
            benchmark_map[key] = value

    heldout_map: dict[str, float | None] = {}
    for row in heldout_rows:
        key = row["planner_key"]
        value = _safe_float(row.get(f"{metric}_mean"))
        if value is not None:
            heldout_map[key] = value

    planners = sorted(set(benchmark_map) | set(heldout_map))
    deltas: list[dict[str, Any]] = []
    for planner in planners:
        benchmark_val = benchmark_map.get(planner)
        heldout_val = heldout_map.get(planner)
        delta = None
        if benchmark_val is not None and heldout_val is not None:
            delta = benchmark_val - heldout_val
        deltas.append(
            {
                "planner_key": planner,
                f"benchmark_{metric}": benchmark_val,
                f"heldout_{metric}": heldout_val,
                f"transfer_delta_{metric}": delta,
                "transfer_direction": (
                    "positive_transfer"
                    if delta is not None and delta > 0
                    else "negative_transfer"
                    if delta is not None and delta < 0
                    else "identical"
                    if delta == 0
                    else "incomplete"
                ),
            }
        )
    return deltas


# ---------------------------------------------------------------------------
# Transfer-delta figure
# ---------------------------------------------------------------------------


def _write_transfer_delta_figure(
    path: Path,
    deltas: list[dict[str, Any]],
    *,
    metric: str = "snqi",
) -> None:
    """Write a bar-chart PNG of transfer deltas by planner."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    keys = [row["planner_key"] for row in deltas]
    values = [row.get(f"transfer_delta_{metric}") for row in deltas]

    colors = []
    bar_values = []
    for val in values:
        if val is None:
            colors.append("#cccccc")
            bar_values.append(0.0)
        elif val > 0:
            colors.append("#d73027")
            bar_values.append(val)
        elif val < 0:
            colors.append("#1a9850")
            bar_values.append(val)
        else:
            colors.append("#4575b4")
            bar_values.append(val)

    bars = ax.bar(keys, bar_values, color=colors, zorder=3)
    ax.axhline(y=0, color="black", linewidth=0.8, zorder=2)
    ax.set_ylabel(f"Transfer delta ({metric})")
    ax.set_title("Benchmark-set vs held-out-family transfer delta")
    ax.grid(axis="y", alpha=0.25, zorder=1)
    for bar, val in zip(bars, values, strict=False):
        if val is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.005 if val >= 0 else -0.005),
                f"{val:.3f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
            )
        elif val is None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.003,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Preliminary claim card
# ---------------------------------------------------------------------------


def _build_claim_card(
    *,
    classification: str,
    reasons: list[str],
    seed_analysis_path: Path | None,
    baseline_table_path: Path | None,
    transfer_delta_path: Path | None,
    figure_path: Path | None,
    partition_manifests: list[Path] | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Build a preliminary claim-card payload."""
    card: dict[str, Any] = {
        "schema_version": "package_a_claim_card.v1",
        "issue": 3078,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "classification": classification,
        "classification_vocabulary": list(PACKAGE_A_CLASSIFICATION),
        "reasons": reasons,
        "artifacts": {
            "seed_sufficiency_analysis": (str(seed_analysis_path) if seed_analysis_path else None),
            "baseline_table": str(baseline_table_path) if baseline_table_path else None,
            "transfer_delta_table": str(transfer_delta_path) if transfer_delta_path else None,
            "transfer_delta_figure": str(figure_path) if figure_path else None,
        },
        "partition_manifests": [str(p) for p in (partition_manifests or [])],
        "output_dir": str(output_dir),
        "claim_boundary": (
            "Diagnostic evidence only. Claim-card review and real campaign "
            "execution required before benchmark-classification promotion."
        ),
        "promotion_allowed": False,
    }
    return card


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------


def _build_markdown_summary(
    *,
    classification: str,
    reasons: list[str],
    seed_analysis_summary: dict[str, Any] | None,
    deltas: list[dict[str, Any]] | None,
    partition_ok: bool | None,
    output_dir: Path,
) -> str:
    """Render a compact Markdown summary of the Package A pipeline run."""
    lines = [
        "# Package A Pipeline Output",
        "",
        "- Issue: `#3078`",
        f"- Classification: `{classification}`",
        f"- Generated: `{datetime.now(UTC).isoformat()}`",
        "",
        "## Artifacts",
        "",
        "- `seed_sufficiency_analysis.json`",
        "- `baseline_table.csv`",
        "- `transfer_delta.csv`",
        "- `fig_transfer_delta.png`",
        "- `claim_card.json`",
        "",
        "## Seed Sufficiency",
        "",
    ]
    if seed_analysis_summary:
        for key, value in sorted(seed_analysis_summary.items()):
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- No seed-sufficiency analysis available (no campaign data).")

    lines.extend(
        [
            "",
            "## Transfer Delta",
            "",
        ]
    )
    if deltas:
        lines.append("| Planner | Transfer Delta (SNQI) | Direction |")
        lines.append("|---|---|---|")
        for row in deltas:
            delta = row.get("transfer_delta_snqi")
            delta_str = f"{delta:.4f}" if delta is not None else "N/A"
            lines.append(
                f"| {row['planner_key']} | {delta_str} | {row.get('transfer_direction', 'N/A')} |"
            )
    else:
        lines.append("- Transfer delta not computed (insufficient cross-surface data).")

    lines.extend(
        [
            "",
            "## Partition Validation",
            "",
        ]
    )
    if partition_ok is True:
        lines.append("- Held-out-family partition manifest: **valid**")
    elif partition_ok is False:
        lines.append("- Held-out-family partition manifest: **validation errors detected**")
    else:
        lines.append("- Held-out-family partition manifest: **not checked**")

    if reasons:
        lines.extend(
            [
                "",
                "## Reasons for Classification",
                "",
            ]
        )
        for reason in reasons:
            lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "---",
            "",
            "This output is diagnostic evidence. Real campaign execution is required "
            "before benchmark-level classification.",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV rows with deterministic field order."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(  # noqa: PLR0915
    output_dir: Path,
    *,
    campaign_roots: list[Path] | None = None,
    benchmark_families: list[str] | None = None,
    heldout_families: list[str] | None = None,
    partition_manifest: Path | None = None,
    metrics: tuple[str, ...] = ("success", "collisions", "snqi"),
) -> dict[str, Any]:
    """Execute the Package A pipeline and write deliverables to ``output_dir``.

    When ``campaign_roots`` is empty the pipeline generates synthetic fixture
    campaigns and classifies the result as ``diagnostic``.

    Returns:
        Pipeline payload with artifact paths and classification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    is_synthetic = campaign_roots is None or not campaign_roots

    # --- 1. Seed-sufficiency analysis ---
    seed_analysis_path: Path | None = None
    seed_analysis_summary: dict[str, Any] | None = None

    if campaign_roots:
        from scripts.tools.analyze_seed_sufficiency import analyze_seed_sufficiency

        seed_out = output_dir / "seed_sufficiency"
        payload = analyze_seed_sufficiency(
            campaign_roots,
            seed_out,
            metrics=metrics,
        )
        seed_analysis_path = seed_out / "seed_sufficiency_analysis.json"
        seed_analysis_summary = payload.get("summary")
    else:
        # Generate synthetic fixture campaigns
        reasons.append("synthetic_fixture_campaigns_used")
        fixture_dir = output_dir / "fixture_campaigns"
        benchmark_planners = {
            "goal": {"success": 0.9, "collisions": 0.05, "snqi": 0.72},
            "social_force": {"success": 0.75, "collisions": 0.15, "snqi": 0.65},
            "orca": {"success": 0.6, "collisions": 0.25, "snqi": 0.5},
        }
        s5 = _create_campaign_fixture(
            fixture_dir / "s5",
            seed_count=5,
            planners=benchmark_planners,
            scenario_families=benchmark_families or ["classic_bottleneck", "classic_cross_trap"],
            ci_half_width=0.15,
        )
        s10 = _create_campaign_fixture(
            fixture_dir / "s10",
            seed_count=10,
            planners={
                "goal": {"success": 0.92, "collisions": 0.04, "snqi": 0.74},
                "social_force": {"success": 0.78, "collisions": 0.13, "snqi": 0.67},
                "orca": {"success": 0.62, "collisions": 0.22, "snqi": 0.52},
            },
            scenario_families=benchmark_families or ["classic_bottleneck", "classic_cross_trap"],
            ci_half_width=0.08,
        )
        campaign_roots = [s5, s10]

        from scripts.tools.analyze_seed_sufficiency import analyze_seed_sufficiency

        seed_out = output_dir / "seed_sufficiency"
        payload = analyze_seed_sufficiency(
            campaign_roots,
            seed_out,
            metrics=metrics,
        )
        seed_analysis_path = seed_out / "seed_sufficiency_analysis.json"
        seed_analysis_summary = payload.get("summary")

    # --- 2. Baseline table ---
    benchmark_rows = _build_baseline_table(campaign_roots, metrics=metrics)
    baseline_table_path = output_dir / "baseline_table.csv"
    _write_csv_rows(baseline_table_path, benchmark_rows)

    # --- 3. Transfer delta ---
    heldout_rows: list[dict[str, Any]] = []
    if heldout_families:
        reasons.append("synthetic_fixture_heldout_used")
        fixture_dir = output_dir / "fixture_campaigns"
        heldout_planners = {
            "goal": {"success": 0.85, "collisions": 0.08, "snqi": 0.68},
            "social_force": {"success": 0.70, "collisions": 0.18, "snqi": 0.60},
            "orca": {"success": 0.55, "collisions": 0.28, "snqi": 0.45},
        }
        heldout_campaign = _create_campaign_fixture(
            fixture_dir / "heldout_pilot",
            seed_count=5,
            planners=heldout_planners,
            scenario_families=heldout_families,
            ci_half_width=0.2,
        )
        heldout_rows = _build_baseline_table([heldout_campaign], metrics=metrics)

    deltas = _build_transfer_delta(benchmark_rows, heldout_rows, metric="snqi")
    transfer_delta_path = output_dir / "transfer_delta.csv"
    _write_csv_rows(transfer_delta_path, deltas)

    transfer_figure_path = output_dir / "fig_transfer_delta.png"
    _write_transfer_delta_figure(transfer_figure_path, deltas, metric="snqi")

    # --- 4. Partition validation ---
    partition_ok: bool | None = None
    if partition_manifest and partition_manifest.is_file():
        from scripts.tools.validate_heldout_transfer_partitions import (
            validate_partition_manifest,
        )

        errors = validate_partition_manifest(partition_manifest)
        partition_ok = len(errors) == 0
        if not partition_ok:
            reasons.extend(f"partition_error: {e}" for e in errors)

    # --- 5. Claim card ---
    if is_synthetic:
        classification = "diagnostic"
        reasons.append("diagnostic: synthetic fixture data, not real campaign evidence")
    elif partition_ok is False:
        classification = "invalid"
        reasons.append("invalid: held-out-family partition manifest has errors")
    else:
        classification = "diagnostic"
        reasons.append("diagnostic: claim-card review required before promotion")

    claim_card = _build_claim_card(
        classification=classification,
        reasons=reasons,
        seed_analysis_path=seed_analysis_path,
        baseline_table_path=baseline_table_path,
        transfer_delta_path=transfer_delta_path,
        figure_path=transfer_figure_path,
        partition_manifests=[partition_manifest] if partition_manifest else None,
        output_dir=output_dir,
    )
    claim_card_path = output_dir / "claim_card.json"
    _write_json(claim_card_path, claim_card)

    # --- 6. Markdown summary ---
    summary_md = _build_markdown_summary(
        classification=classification,
        reasons=reasons,
        seed_analysis_summary=seed_analysis_summary,
        deltas=deltas,
        partition_ok=partition_ok,
        output_dir=output_dir,
    )
    (output_dir / "package_a_summary.md").write_text(summary_md, encoding="utf-8")

    # --- 7. Top-level payload ---
    output_payload = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "classification_vocabulary": list(PACKAGE_A_CLASSIFICATION),
        "reasons": reasons,
        "artifacts": {
            "seed_sufficiency_analysis": str(seed_analysis_path),
            "baseline_table": str(baseline_table_path),
            "transfer_delta": str(transfer_delta_path),
            "transfer_delta_figure": str(transfer_figure_path),
            "claim_card": str(claim_card_path),
            "summary_markdown": str(output_dir / "package_a_summary.md"),
        },
    }
    _write_json(output_dir / "pipeline_output.json", output_payload)
    return output_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for Package A deliverables.",
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        action="append",
        default=None,
        help="Campaign root (with reports/). Repeat for multiple seed schedules.",
    )
    parser.add_argument(
        "--benchmark-families",
        nargs="+",
        default=["classic_bottleneck", "classic_cross_trap"],
        help="Benchmark-set scenario families (used for synthetic fixtures).",
    )
    parser.add_argument(
        "--heldout-families",
        nargs="+",
        default=["classic_station_platform", "francis2023_intersection_wait"],
        help="Held-out-family scenario families (used for synthetic fixtures).",
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=None,
        help=(
            "Held-out-family transfer partition manifest YAML "
            "(runs validate_heldout_transfer_partitions.py against it)."
        ),
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        default=["success", "collisions", "snqi"],
        help="Metrics to analyze. Defaults to success, collisions, snqi.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    output_dir = args.output_dir
    campaign_roots = list(args.campaign_root) if args.campaign_root else None
    partition_manifest = args.partition_manifest
    metrics = tuple(args.metrics)

    try:
        payload = run_pipeline(
            output_dir,
            campaign_roots=campaign_roots,
            benchmark_families=args.benchmark_families,
            heldout_families=args.heldout_families,
            partition_manifest=partition_manifest,
            metrics=metrics,
        )
        print(f"Package A pipeline output: {output_dir}")
        print(f"Classification: {payload['classification']}")
        for reason in payload.get("reasons", []):
            print(f" reason: {reason}")
        return 0
    except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
