#!/usr/bin/env python3
"""Flat-vs-hierarchical bootstrap interval comparison (issue #5139, part 3).

Issue #5139 asks for one comparison artifact documenting the anti-conservatism
magnitude of the existing flat (i.i.d. episode) bootstrap relative to the
hierarchical (scenario-then-episode) cluster bootstrap. This script produces
that artifact.

Claim boundary and evidence status
-----------------------------------
- Claim boundary: characterizes the implemented resampling procedures
  (``robot_sf.benchmark.full_classic.aggregation``) on a *synthetic but
  structured* episode bundle that mirrors the documented
  family -> scenario cell -> seed -> episode nesting with controlled
  intra-cluster correlation.
- Evidence status: ``diagnostic-only``. This is NOT benchmark evidence and NOT
  a paper claim. No real campaign bundle in the repository matches the
  (archetype, density, scenario_id, seed) aggregation schema yet; the
  pre-registered 30-seed successor campaign this work unblocks has not run.
  The synthetic bundle exists solely so the flat-vs-hierarchical width
  difference is documented and reproducible. Re-run this script on a real
  campaign bundle once one exists to obtain nominal benchmark evidence.

Usage::

    uv run python scripts/analysis/compare_flat_vs_hierarchical_intervals_issue_5139.py \\
        --output-dir docs/context/evidence/issue_5139_hierarchical_bootstrap

Outputs (under ``--output-dir``):
  * ``comparison_report.md``  - human-readable comparison table.
  * ``comparison_report.json`` - machine-readable results with provenance.
  * ``synthetic_bundle.jsonl`` - the structured episode bundle used (for audit).
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = build_synthetic_bundle(seed=args.seed)
    report = collect_comparisons(records, bundle_seed=args.seed)

    write_markdown(report, output_dir / "comparison_report.md")
    write_json(report, output_dir / "comparison_report.json")
    write_bundle(records, output_dir / "synthetic_bundle.jsonl")

    rate_summary = report.summary.get("rate_metrics_hierarchical_scenario", {})
    print(f"[issue_5139] wrote artifact to {output_dir}")
    print(f"[issue_5139] evidence_status={report.evidence_status}")
    if rate_summary:
        print(
            f"[issue_5139] rate-metric width ratios (hierarchical_scenario/flat): "
            f"median={rate_summary.get('median_ratio'):.2f} "
            f"mean={rate_summary.get('mean_ratio'):.2f} "
            f"max={rate_summary.get('max_ratio'):.2f}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
