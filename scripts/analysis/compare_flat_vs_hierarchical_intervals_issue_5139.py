#!/usr/bin/env python3
"""Flat-vs-hierarchical bootstrap interval comparison (issue #5139, part 3).

Issue #5139 asks for one comparison artifact documenting the anti-conservatism
magnitude of the existing flat (i.i.d. episode) bootstrap relative to the
hierarchical (scenario-then-episode) cluster bootstrap. This script produces
that artifact.

Claim boundary and evidence status
-----------------------------------
- Evidence status: ``diagnostic-only``. This is NOT benchmark evidence and NOT
  a paper claim. The default synthetic mode preserves the original method
  diagnostic. ``--retained-bundle`` adapts a retained camera-ready episode
  table into planner/kinematics groups while keeping scenario identifiers as
  clusters, producing the bounded real-bundle comparison required by #5139.

Usage::

    uv run python scripts/analysis/compare_flat_vs_hierarchical_intervals_issue_5139.py \\
        --output-dir docs/context/evidence/issue_5139_hierarchical_bootstrap

Outputs (under ``--output-dir``):
  * ``comparison_report.md``  - human-readable comparison table.
  * ``comparison_report.json`` - machine-readable results with provenance.
  * ``synthetic_bundle.jsonl`` - the structured episode bundle used (for audit).
  * ``retained_comparison_report.{md,json}`` - real retained-bundle analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# parent 1 = scripts/analysis, parent 2 = scripts, parent 3 = repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

CLAIM_BOUNDARY = (
    "diagnostic-only: synthetic structured episode bundle characterizing the implemented "
    "flat vs hierarchical bootstrap procedures. NOT benchmark evidence and NOT a paper claim. "
    "No repository campaign bundle matches the (archetype, density, scenario_id, seed) "
    "aggregation schema yet; the pre-registered 30-seed successor campaign this work unblocks "
    "has not run. Re-run on a real campaign bundle to obtain nominal benchmark evidence."
)

EVIDENCE_STATUS = "diagnostic-only"
REVIEW_MARKER = "AI-GENERATED NEEDS-REVIEW"
DEFAULT_RETAINED_BUNDLE = (
    REPO_ROOT
    / "docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/reports/seed_episode_rows.csv"
)
DEFAULT_RETAINED_MANIFEST = (
    REPO_ROOT
    / "docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23/campaign_manifest.json"
)
RETAINED_CLAIM_BOUNDARY = (
    "diagnostic-only, analysis-only: flat versus hierarchical interval widths on the retained "
    "issue #1454 exploratory campaign bundle. This post-hoc reuse was not pre-registered for "
    "issue #5139 and does not establish benchmark, planner-ranking, paper, or dissertation claims."
)


def _stable_cell_offset(archetype: str, density: str) -> int:
    """Return a process-independent offset for a synthetic scenario family."""
    material = f"{archetype}\x1f{density}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


@dataclass
class MetricComparison:
    """Flat-vs-hierarchical width comparison for one (group, metric)."""

    archetype: str
    density: str
    metric: str
    mode: str
    mean: float
    mean_ci: tuple[float, float]
    median_ci: tuple[float, float]
    mean_ci_width: float
    is_rate: bool


@dataclass
class ComparisonReport:
    """Top-level comparison report payload with provenance."""

    claim_boundary: str
    evidence_status: str
    issue: str
    description: str
    bundle_description: str
    config: dict[str, Any]
    comparisons: list[MetricComparison] = field(default_factory=list)
    width_ratios: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetainedMetricComparison:
    """Interval-width comparison for one retained-campaign planner and metric."""

    planner_key: str
    kinematics: str
    metric: str
    mode: str
    mean: float
    mean_ci: tuple[float, float]
    mean_ci_width: float
    is_rate: bool


@dataclass
class RetainedComparisonReport:
    """Real retained-bundle comparison with deterministic provenance."""

    claim_boundary: str
    evidence_status: str
    issue: str
    description: str
    source_provenance: dict[str, Any]
    grouping_contract: dict[str, Any]
    config: dict[str, Any]
    comparisons: list[RetainedMetricComparison] = field(default_factory=list)
    width_ratios: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_synthetic_bundle(seed: int = 20260710) -> list[dict[str, Any]]:
    """Build a structured episode bundle with scenario/seed nesting.

    Structure mirrors the documented campaign nesting
    (family -> scenario cell -> seed -> episode) restricted to a single
    archetype/density so the (archetype, density) aggregation group is
    populated. Within-cell outcomes are generated from a shared cell-level
    success probability, inducing the positive intra-cluster correlation that
    makes the flat i.i.d. bootstrap anti-conservative.

    Returns:
        List of episode record dicts consumed by ``aggregate_metrics``.
    """
    rng = random.Random(seed)
    archetypes = [("crossing", "low"), ("crossing", "high"), ("bottleneck", "low")]
    # Cell-level success/collision rates vary across scenarios to create
    # between-cell heterogeneity (the source of cluster-driven uncertainty).
    cell_rate_grid = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
    n_cells_per_group = 6
    seeds_per_cell = 5  # episodes per scenario cell (one per seed)
    records: list[dict[str, Any]] = []
    ep = 0
    for arch, dens in archetypes:
        for cell_idx in range(n_cells_per_group):
            rate = cell_rate_grid[
                (cell_idx + _stable_cell_offset(arch, dens)) % len(cell_rate_grid)
            ]
            rate = min(0.97, max(0.03, rate + rng.uniform(-0.05, 0.05)))
            scenario_id = f"{arch}_{dens}_scenario_{cell_idx}"
            for seed_value in range(seeds_per_cell):
                collision = 1.0 if rng.random() < rate else 0.0
                success = 1.0 - collision
                # Continuous metric also carries a cell-level offset so it has
                # within-cell correlation too.
                ttg = 10.0 + cell_idx * 1.5 + rng.uniform(-0.5, 0.5)
                records.append(
                    {
                        "episode_id": f"ep{ep}",
                        "scenario_id": scenario_id,
                        "seed": seed_value,
                        "archetype": arch,
                        "density": dens,
                        "metrics": {
                            "collision_rate": collision,
                            "success_rate": success,
                            "time_to_goal": ttg,
                        },
                    }
                )
                ep += 1
    return records


class _Cfg:
    """Minimal config stub for aggregate_metrics."""

    def __init__(self, **kwargs):
        self.bootstrap_samples = 1000
        self.bootstrap_confidence = 0.95
        self.master_seed = 123
        self.smoke = False
        self.bootstrap_mode = "flat"
        self.bootstrap_cluster = "scenario"
        for k, v in kwargs.items():
            setattr(self, k, v)


def _ci_width(ci):
    if ci is None or any(math.isnan(v) for v in ci):
        return 0.0
    return ci[1] - ci[0]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a provenance input."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    """Render a path relative to the repository when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_retained_bundle(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load retained camera-ready episode rows into the aggregation contract.

    The full-classic aggregation API groups on ``(archetype, density)``. For this
    cross-scenario campaign comparison those two adapter fields represent
    ``(planner_key, kinematics)``; the original ``scenario_id`` remains the
    hierarchical cluster. This produces one interval per planner/kinematics
    row over the campaign's scenario cells without mixing planners.
    """
    required = {
        "episode_id",
        "scenario_id",
        "planner_key",
        "kinematics",
        "seed",
        "success",
        "collision",
        "near_miss",
        "time_to_goal",
        "snqi",
    }
    if not path.is_file():
        raise FileNotFoundError(f"retained campaign episode table not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"retained campaign table missing columns: {', '.join(missing)}")
        source_rows = list(reader)
    if not source_rows:
        raise ValueError("retained campaign table contains no episode rows")

    records: list[dict[str, Any]] = []
    for row_number, row in enumerate(source_rows, start=2):
        metrics: dict[str, float] = {}
        for source_name, metric_name in (
            ("success", "success_rate"),
            ("collision", "collision_rate"),
            ("near_miss", "near_miss"),
            ("time_to_goal", "time_to_goal"),
            ("snqi", "snqi"),
        ):
            try:
                value = float(row[source_name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"retained campaign table row {row_number} has invalid {source_name!r}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"retained campaign table row {row_number} has non-finite {source_name!r}"
                )
            metrics[metric_name] = value
        records.append(
            {
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "seed": int(row["seed"]),
                # Adapter contract documented above; output translates these
                # names back to planner_key and kinematics.
                "archetype": row["planner_key"],
                "density": row["kinematics"],
                "metrics": metrics,
            }
        )

    provenance = {
        "episode_table": _repo_relative(path),
        "episode_table_sha256": _sha256(path),
        "row_count": len(records),
        "planner_keys": sorted({row["planner_key"] for row in source_rows}),
        "kinematics": sorted({row["kinematics"] for row in source_rows}),
        "scenario_ids": sorted({row["scenario_id"] for row in source_rows}),
        "seeds": sorted({int(row["seed"]) for row in source_rows}),
    }
    return records, provenance


def collect_retained_comparisons(
    records: list[dict[str, Any]],
    *,
    source_provenance: dict[str, Any],
    manifest_path: Path,
) -> RetainedComparisonReport:
    """Compare flat and scenario-hierarchical intervals on retained rows."""
    from robot_sf.benchmark.full_classic.aggregation import aggregate_metrics

    if not manifest_path.is_file():
        raise FileNotFoundError(f"retained campaign manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = {
        **source_provenance,
        "campaign_manifest": _repo_relative(manifest_path),
        "campaign_manifest_sha256": _sha256(manifest_path),
        "campaign_id": manifest.get("campaign_id"),
        "campaign_schema_version": manifest.get("schema_version"),
        "campaign_git_commit": (manifest.get("git") or {}).get("commit"),
        "campaign_config_hash": manifest.get("config_hash"),
    }
    cfg_common = {
        "bootstrap_samples": 1000,
        "bootstrap_confidence": 0.95,
        "master_seed": 5139,
        "smoke": False,
    }
    flat_groups = aggregate_metrics(
        [dict(record) for record in records],
        _Cfg(bootstrap_mode="flat", **cfg_common),
    )
    hierarchical_groups = aggregate_metrics(
        [dict(record) for record in records],
        _Cfg(bootstrap_mode="hierarchical", bootstrap_cluster="scenario", **cfg_common),
    )

    rate_metrics = {"collision_rate", "success_rate"}
    comparisons: list[RetainedMetricComparison] = []
    for groups, mode in ((flat_groups, "flat"), (hierarchical_groups, "hierarchical_scenario")):
        for group in groups:
            for metric_name, metric in sorted(group.metrics.items()):
                comparisons.append(
                    RetainedMetricComparison(
                        planner_key=group.archetype,
                        kinematics=group.density,
                        metric=metric_name,
                        mode=mode,
                        mean=metric.mean,
                        mean_ci=metric.mean_ci or (math.nan, math.nan),
                        mean_ci_width=_ci_width(metric.mean_ci),
                        is_rate=metric_name in rate_metrics,
                    )
                )

    by_key: dict[tuple[str, str, str], dict[str, RetainedMetricComparison]] = {}
    for comparison in comparisons:
        key = (comparison.planner_key, comparison.kinematics, comparison.metric)
        by_key.setdefault(key, {})[comparison.mode] = comparison
    width_ratios: list[dict[str, Any]] = []
    for (planner_key, kinematics, metric), modes in sorted(by_key.items()):
        flat = modes.get("flat")
        hierarchical = modes.get("hierarchical_scenario")
        if flat is None or hierarchical is None or flat.mean_ci_width <= 0:
            continue
        width_ratios.append(
            {
                "planner_key": planner_key,
                "kinematics": kinematics,
                "metric": metric,
                "flat_width": flat.mean_ci_width,
                "hierarchical_width": hierarchical.mean_ci_width,
                "ratio": hierarchical.mean_ci_width / flat.mean_ci_width,
                "is_rate": hierarchical.is_rate,
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        ratios = sorted(row["ratio"] for row in rows)
        if not ratios:
            return {}
        middle = len(ratios) // 2
        median_ratio = (
            ratios[middle] if len(ratios) % 2 else (ratios[middle - 1] + ratios[middle]) / 2
        )
        return {
            "count": len(ratios),
            "min_ratio": ratios[0],
            "median_ratio": median_ratio,
            "mean_ratio": sum(ratios) / len(ratios),
            "max_ratio": ratios[-1],
        }

    report = RetainedComparisonReport(
        claim_boundary=RETAINED_CLAIM_BOUNDARY,
        evidence_status="diagnostic-only (analysis-only)",
        issue="#5139",
        description=(
            "Flat episode-level intervals versus hierarchical scenario-cluster intervals on one "
            "retained campaign bundle. Binary endpoints compare flat Wilson intervals with the "
            "merged cluster-robust intervals."
        ),
        source_provenance=provenance,
        grouping_contract={
            "analysis_group": ["planner_key", "kinematics"],
            "hierarchical_cluster": "scenario_id",
            "episode_level": "seed_episode_rows.csv row",
            "adapter": {
                "archetype": "planner_key",
                "density": "kinematics",
                "reason": (
                    "aggregate_metrics requires archetype/density group keys; the adapter keeps "
                    "planners separate while pooling retained scenario cells for each planner."
                ),
            },
        },
        config={
            **cfg_common,
            "modes": ["flat", "hierarchical_scenario"],
            "rate_interval_flat": "wilson",
            "rate_interval_hierarchical": "cluster_robust",
        },
    )
    report.comparisons = comparisons
    report.width_ratios = width_ratios
    report.summary = {
        "rate_metrics": summarize([row for row in width_ratios if row["is_rate"]]),
        "non_rate_metrics": summarize([row for row in width_ratios if not row["is_rate"]]),
        "interpretation": (
            "Ratios describe this retained exploratory bundle only. A ratio above one means the "
            "scenario-hierarchical interval is wider; a ratio at or below one is retained rather "
            "than filtered and does not invalidate the method."
        ),
    }
    return report


def _compute_width_ratios(
    comparisons: list[MetricComparison],
) -> list[dict[str, Any]]:
    """Compute hierarchical/flat width ratios per (group, metric)."""
    by_key: dict[tuple[str, str, str], dict[str, MetricComparison]] = {}
    for c in comparisons:
        by_key.setdefault((c.archetype, c.density, c.metric), {})[c.mode] = c
    width_ratios: list[dict[str, Any]] = []
    for (arch, dens, metric), modes in sorted(by_key.items()):
        base = modes.get("flat")
        if base is None or base.mean_ci_width <= 0:
            continue
        for mode in ("hierarchical_scenario", "hierarchical_seed"):
            other = modes.get(mode)
            if other is None:
                continue
            width_ratios.append(
                {
                    "archetype": arch,
                    "density": dens,
                    "metric": metric,
                    "mode": mode,
                    "flat_width": base.mean_ci_width,
                    "hierarchical_width": other.mean_ci_width,
                    "ratio": other.mean_ci_width / base.mean_ci_width,
                    "is_rate": other.is_rate,
                }
            )
    return width_ratios


def collect_comparisons(records: list[dict[str, Any]], *, bundle_seed: int) -> ComparisonReport:
    """Run flat and hierarchical aggregation and collect width comparisons."""
    from robot_sf.benchmark.full_classic.aggregation import aggregate_metrics

    rate_metrics = {"collision_rate", "success_rate"}
    cfg_common = {
        "bootstrap_samples": 1000,
        "bootstrap_confidence": 0.95,
        "master_seed": 123,
        "bundle_seed": bundle_seed,
        "smoke": False,
    }
    flat_groups = aggregate_metrics(
        [dict(r) for r in records],
        _Cfg(bootstrap_mode="flat", **cfg_common),
    )
    hier_groups = aggregate_metrics(
        [dict(r) for r in records],
        _Cfg(bootstrap_mode="hierarchical", bootstrap_cluster="scenario", **cfg_common),
    )
    hier_seed_groups = aggregate_metrics(
        [dict(r) for r in records],
        _Cfg(bootstrap_mode="hierarchical", bootstrap_cluster="seed", **cfg_common),
    )

    comparisons: list[MetricComparison] = []

    def _emit(groups, mode):
        for g in groups:
            for mname, m in g.metrics.items():
                comparisons.append(
                    MetricComparison(
                        archetype=g.archetype,
                        density=g.density,
                        metric=mname,
                        mode=mode,
                        mean=m.mean,
                        mean_ci=m.mean_ci,
                        median_ci=m.median_ci,
                        mean_ci_width=_ci_width(m.mean_ci),
                        is_rate=mname in rate_metrics,
                    )
                )

    _emit(flat_groups, "flat")
    _emit(hier_groups, "hierarchical_scenario")
    _emit(hier_seed_groups, "hierarchical_seed")

    width_ratios = _compute_width_ratios(comparisons)

    # Summary statistics, split by cluster mode so the scenario-cluster
    # anti-conservatism is not diluted by the seed-cluster variant.
    rate_ratios = [r for r in width_ratios if r["is_rate"]]
    cont_ratios = [r for r in width_ratios if not r["is_rate"]]
    rate_scenario = [r for r in rate_ratios if r["mode"] == "hierarchical_scenario"]
    rate_seed = [r for r in rate_ratios if r["mode"] == "hierarchical_seed"]
    cont_scenario = [r for r in cont_ratios if r["mode"] == "hierarchical_scenario"]
    cont_seed = [r for r in cont_ratios if r["mode"] == "hierarchical_seed"]

    def _summarize(rs):
        if not rs:
            return {}
        ratios = [r["ratio"] for r in rs]
        return {
            "count": len(ratios),
            "min_ratio": min(ratios),
            "max_ratio": max(ratios),
            "mean_ratio": sum(ratios) / len(ratios),
            "median_ratio": sorted(ratios)[len(ratios) // 2],
        }

    report = ComparisonReport(
        claim_boundary=CLAIM_BOUNDARY,
        evidence_status=EVIDENCE_STATUS,
        issue="#5139",
        description=(
            "Flat (i.i.d. episode) vs hierarchical (scenario-then-episode cluster) "
            "bootstrap interval-width comparison for the analysis layer."
        ),
        bundle_description=(
            f"Synthetic structured bundle: {len(records)} episodes across "
            f"{len({(r['archetype'], r['density']) for r in records})} (archetype, density) "
            "groups, 6 scenario cells per group, 5 seeds per cell, with cell-level "
            "success/collision probabilities inducing intra-cluster correlation."
        ),
        config={
            **cfg_common,
            "modes": ["flat", "hierarchical_scenario", "hierarchical_seed"],
        },
    )
    report.comparisons = comparisons
    report.width_ratios = width_ratios
    report.summary = {
        "rate_metrics_all_modes": _summarize(rate_ratios),
        "rate_metrics_hierarchical_scenario": _summarize(rate_scenario),
        "rate_metrics_hierarchical_seed": _summarize(rate_seed),
        "continuous_metrics_all_modes": _summarize(cont_ratios),
        "continuous_metrics_hierarchical_scenario": _summarize(cont_scenario),
        "continuous_metrics_hierarchical_seed": _summarize(cont_seed),
        "anti_conservatism_note": (
            "ratio > 1 means the hierarchical interval is wider than the flat "
            "interval on the same records, i.e. the flat interval was "
            "anti-conservative (understated uncertainty) under clustering. The "
            "hierarchical_scenario (scenario-then-episode) mode is the documented "
            "successor-campaign procedure and is the primary anti-conservatism "
            "comparison; hierarchical_seed is the optional seed-level cluster "
            "variant. A seed-mode ratio < 1 can occur when the cell-level rate "
            "already absorbs most between-cell variance, leaving little "
            "between-seed dispersion."
        ),
    }
    return report


def write_markdown(report: ComparisonReport, path: Path) -> None:
    """Write a human-readable Markdown comparison report."""
    lines: list[str] = []
    lines.append(f"<!-- {REVIEW_MARKER} -->")
    lines.append("")
    lines.append("# Flat vs Hierarchical Bootstrap Interval Comparison (issue #5139)")
    lines.append("")
    lines.append(f"**Evidence status:** `{report.evidence_status}`")
    lines.append("")
    lines.append(f"**Claim boundary:** {report.claim_boundary}")
    lines.append("")
    lines.append("## Plain-language summary")
    lines.append("")
    lines.append(
        "The analysis layer previously resampled episodes as if they were all "
        "independent (flat bootstrap). Because episodes are actually grouped inside "
        "scenario cells and seeds, that flat method understates how uncertain a result "
        "really is. This artifact compares the old flat interval widths against the new "
        "hierarchical (scenario-then-episode) cluster bootstrap widths on a structured "
        "synthetic bundle, so the size of that understatement (the "
        "anti-conservatism) is documented. It is a diagnostic characterization, not "
        "benchmark evidence."
    )
    lines.append("")
    lines.append("## Bundle")
    lines.append("")
    lines.append(report.bundle_description)
    lines.append("")
    lines.append(f"- bootstrap_samples: {report.config['bootstrap_samples']}")
    lines.append(f"- bootstrap_confidence: {report.config['bootstrap_confidence']}")
    lines.append(f"- master_seed: {report.config['master_seed']}")
    lines.append(f"- bundle_seed: {report.config['bundle_seed']}")
    lines.append(f"- modes compared: {', '.join(report.config['modes'])}")
    lines.append("")
    lines.append("## Width ratios (hierarchical / flat)")
    lines.append("")
    lines.append(
        "ratio > 1 means the hierarchical interval is wider than flat (flat was anti-conservative)."
    )
    lines.append("")
    lines.append(
        "| archetype | density | metric | mode | flat_width | hierarchical_width | ratio |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(
        report.width_ratios, key=lambda x: (x["archetype"], x["density"], x["metric"], x["mode"])
    ):
        lines.append(
            f"| {r['archetype']} | {r['density']} | {r['metric']} | {r['mode']} | "
            f"{r['flat_width']:.4f} | {r['hierarchical_width']:.4f} | {r['ratio']:.2f} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    def _block(title, s):
        lines.append(f"### {title}")
        lines.append("")
        if not s:
            lines.append("_no entries_")
            lines.append("")
            return
        lines.append(f"- count: {s['count']}")
        lines.append(f"- min ratio: {s['min_ratio']:.2f}")
        lines.append(f"- median ratio: {s['median_ratio']:.2f}")
        lines.append(f"- mean ratio: {s['mean_ratio']:.2f}")
        lines.append(f"- max ratio: {s['max_ratio']:.2f}")
        lines.append("")

    _block(
        "Rate metrics - hierarchical_scenario mode (primary anti-conservatism comparison)",
        report.summary.get("rate_metrics_hierarchical_scenario", {}),
    )
    _block(
        "Rate metrics - hierarchical_seed mode (optional seed-cluster variant)",
        report.summary.get("rate_metrics_hierarchical_seed", {}),
    )
    _block(
        "Continuous metrics - hierarchical_scenario mode",
        report.summary.get("continuous_metrics_hierarchical_scenario", {}),
    )
    _block(
        "Continuous metrics - hierarchical_seed mode",
        report.summary.get("continuous_metrics_hierarchical_seed", {}),
    )
    lines.append(report.summary.get("anti_conservatism_note", ""))
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append(
        "```bash\n"
        "uv run python scripts/analysis/compare_flat_vs_hierarchical_intervals_issue_5139.py \\\n"
        "  --output-dir docs/context/evidence/issue_5139_hierarchical_bootstrap\n"
        "```"
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(report: ComparisonReport, path: Path) -> None:
    """Write a machine-readable JSON report with provenance."""
    payload = {
        "review_marker": REVIEW_MARKER,
        "claim_boundary": report.claim_boundary,
        "evidence_status": report.evidence_status,
        "issue": report.issue,
        "description": report.description,
        "bundle_description": report.bundle_description,
        "config": report.config,
        "comparisons": [asdict(c) for c in report.comparisons],
        "width_ratios": report.width_ratios,
        "summary": report.summary,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_bundle(records: list[dict[str, Any]], path: Path) -> None:
    """Write the synthetic episode bundle as JSONL for audit."""
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**r, "review_marker": REVIEW_MARKER}) + "\n")


def write_retained_markdown(report: RetainedComparisonReport, path: Path) -> None:
    """Write the retained-bundle analysis report."""
    provenance = report.source_provenance
    lines = [
        f"<!-- {REVIEW_MARKER} -->",
        "",
        "# Retained Campaign Flat vs Hierarchical Interval Comparison (issue #5139)",
        "",
        f"**Claim boundary:** {report.claim_boundary}",
        "",
        f"**Evidence status:** `{report.evidence_status}`",
        "",
        "**Major caveat:** this is a post-hoc analysis of an exploratory retained campaign, not "
        "the pre-registered successor campaign and not benchmark-strength evidence.",
        "",
        "## Plain-language summary",
        "",
        "This report compares confidence-interval widths when retained campaign episodes are "
        "treated as independent versus when their scenario grouping is respected. It exercises "
        "the merged scenario-hierarchical bootstrap and cluster-robust binary intervals on real "
        "campaign rows while keeping the result analysis-only.",
        "",
        "## Deterministic provenance",
        "",
        f"- campaign: `{provenance['campaign_id']}`",
        f"- episode table: `{provenance['episode_table']}`",
        f"- episode table SHA-256: `{provenance['episode_table_sha256']}`",
        f"- campaign manifest: `{provenance['campaign_manifest']}`",
        f"- campaign manifest SHA-256: `{provenance['campaign_manifest_sha256']}`",
        f"- campaign commit: `{provenance['campaign_git_commit']}`",
        f"- config hash: `{provenance['campaign_config_hash']}`",
        f"- rows: {provenance['row_count']}",
        f"- planners: {len(provenance['planner_keys'])}",
        f"- scenario cells: {len(provenance['scenario_ids'])}",
        f"- seeds: {len(provenance['seeds'])}",
        "- analysis groups: `(planner_key, kinematics)`; hierarchical cluster: `scenario_id`",
        f"- bootstrap samples: {report.config['bootstrap_samples']}",
        f"- confidence: {report.config['bootstrap_confidence']}",
        f"- master seed: {report.config['master_seed']}",
        "",
        "## Interval-width ratios",
        "",
        "A ratio above 1 means the scenario-hierarchical interval is wider than the flat "
        "interval on the same retained rows. Binary endpoints compare cluster-robust intervals "
        "against flat Wilson intervals.",
        "",
        "| planner | kinematics | metric | flat width | hierarchical width | ratio |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in report.width_ratios:
        lines.append(
            f"| {row['planner_key']} | {row['kinematics']} | {row['metric']} | "
            f"{row['flat_width']:.6f} | {row['hierarchical_width']:.6f} | "
            f"{row['ratio']:.3f} |"
        )
    lines.extend(["", "## Bounded summary", ""])
    for label, key in (
        ("Binary rate metrics", "rate_metrics"),
        ("Non-rate metrics", "non_rate_metrics"),
    ):
        summary = report.summary[key]
        lines.extend(
            [
                f"### {label}",
                "",
                f"- comparisons: {summary.get('count', 0)}",
                f"- minimum ratio: {summary.get('min_ratio', math.nan):.3f}",
                f"- median ratio: {summary.get('median_ratio', math.nan):.3f}",
                f"- mean ratio: {summary.get('mean_ratio', math.nan):.3f}",
                f"- maximum ratio: {summary.get('max_ratio', math.nan):.3f}",
                "",
            ]
        )
    lines.extend(
        [
            report.summary["interpretation"],
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python scripts/analysis/compare_flat_vs_hierarchical_intervals_issue_5139.py \\",
            f"  --retained-bundle {provenance['episode_table']} \\",
            f"  --retained-manifest {provenance['campaign_manifest']}",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_retained_json(report: RetainedComparisonReport, path: Path) -> None:
    """Write the machine-readable retained-bundle report."""
    payload = {
        "review_marker": REVIEW_MARKER,
        "schema_version": "issue_5139.retained_interval_comparison.v1",
        "claim_boundary": report.claim_boundary,
        "evidence_status": report.evidence_status,
        "issue": report.issue,
        "description": report.description,
        "source_provenance": report.source_provenance,
        "grouping_contract": report.grouping_contract,
        "config": report.config,
        "comparisons": [asdict(comparison) for comparison in report.comparisons],
        "width_ratios": report.width_ratios,
        "summary": report.summary,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Generate the flat-vs-hierarchical comparison artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "context" / "evidence" / "issue_5139_hierarchical_bootstrap",
        help="Output directory for the comparison artifact.",
    )
    parser.add_argument("--seed", type=int, default=20260710, help="Bundle RNG seed.")
    parser.add_argument(
        "--retained-bundle",
        type=Path,
        help=(
            "Retained seed_episode_rows.csv to analyze instead of generating the historical "
            "synthetic diagnostic. The canonical retained input is "
            f"{_repo_relative(DEFAULT_RETAINED_BUNDLE)}."
        ),
    )
    parser.add_argument(
        "--retained-manifest",
        type=Path,
        default=DEFAULT_RETAINED_MANIFEST,
        help="Campaign manifest paired with --retained-bundle.",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.retained_bundle is not None:
        records, provenance = load_retained_bundle(args.retained_bundle)
        retained_report = collect_retained_comparisons(
            records,
            source_provenance=provenance,
            manifest_path=args.retained_manifest,
        )
        write_retained_markdown(
            retained_report,
            output_dir / "retained_comparison_report.md",
        )
        write_retained_json(
            retained_report,
            output_dir / "retained_comparison_report.json",
        )
        rate_summary = retained_report.summary.get("rate_metrics", {})
        print(f"[issue_5139] wrote retained-bundle artifact to {output_dir}")
        print(f"[issue_5139] evidence_status={retained_report.evidence_status}")
        print(
            "[issue_5139] rate-metric width ratios (hierarchical_scenario/flat): "
            f"median={rate_summary.get('median_ratio'):.2f} "
            f"mean={rate_summary.get('mean_ratio'):.2f} "
            f"max={rate_summary.get('max_ratio'):.2f}"
        )
    else:
        records = build_synthetic_bundle(seed=args.seed)
        report = collect_comparisons(records, bundle_seed=args.seed)
        write_markdown(report, output_dir / "comparison_report.md")
        write_json(report, output_dir / "comparison_report.json")
        write_bundle(records, output_dir / "synthetic_bundle.jsonl")
        rate_summary = report.summary.get("rate_metrics_hierarchical_scenario", {})
        print(f"[issue_5139] wrote synthetic artifact to {output_dir}")
        print(f"[issue_5139] evidence_status={report.evidence_status}")
        if rate_summary:
            print(
                "[issue_5139] rate-metric width ratios (hierarchical_scenario/flat): "
                f"median={rate_summary.get('median_ratio'):.2f} "
                f"mean={rate_summary.get('mean_ratio'):.2f} "
                f"max={rate_summary.get('max_ratio'):.2f}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
