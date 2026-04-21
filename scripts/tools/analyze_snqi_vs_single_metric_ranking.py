#!/usr/bin/env python3
"""Issue 821 ablation: SNQI planner ranking vs. single-metric planner rankings.

The camera-ready contract claims SNQI is the headline composite score. This
ablation checks whether that composite reorders planners relative to the
single-metric rankings that a reviewer might look at by default (success,
collisions, near_misses). If SNQI preserves the single-metric order, the
composite is cosmetic; if it reorders, the composite is load-bearing.

Inputs:
    --campaign-root PATH   camera-ready campaign directory

Outputs (written under <campaign-root>/reports/):
    snqi_vs_single_metric_ranking.json
    snqi_vs_single_metric_ranking.md

The script only reads ``reports/campaign_table.csv`` (planner-level aggregates)
and does not alter the benchmark contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.snqi.campaign_contract import spearman_correlation


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ranking ablation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Restrict the ablation to benchmark-success rows (recommended).",
    )
    return parser.parse_args(argv)


def _parse_float(value: str) -> float | None:
    """Parse a CSV cell that may be empty or contain a leading quote guard."""
    if value is None:
        return None
    cleaned = value.strip().lstrip("'").lstrip('"')
    if cleaned == "":
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _load_planner_rows(campaign_root: Path, *, core_only: bool) -> list[dict[str, Any]]:
    """Load planner-level aggregate rows from campaign_table.csv.

    Rows with ``benchmark_success != true`` are excluded when ``core_only`` is
    set; otherwise all rows with finite metric values are retained. This
    matches the fail-closed policy interpretation that non-success rows are
    not benchmark evidence.
    """
    table_path = campaign_root / "reports" / "campaign_table.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"campaign_table.csv not found at {table_path}")
    rows: list[dict[str, Any]] = []
    with table_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if core_only:
                success_flag = str(raw.get("benchmark_success", "")).strip().lower()
                if success_flag != "true":
                    continue
            success_mean = _parse_float(raw.get("success_mean", ""))
            collisions_mean = _parse_float(raw.get("collisions_mean", ""))
            near_misses_mean = _parse_float(raw.get("near_misses_mean", ""))
            snqi_mean = _parse_float(raw.get("snqi_mean", ""))
            if None in (success_mean, collisions_mean, near_misses_mean, snqi_mean):
                continue
            rows.append(
                {
                    "planner_key": str(raw.get("planner_key", "unknown")),
                    "algo": str(raw.get("algo", "unknown")),
                    "planner_group": str(raw.get("planner_group", "unknown")),
                    "kinematics": str(raw.get("kinematics", "unknown")),
                    "success_mean": success_mean,
                    "collisions_mean": collisions_mean,
                    "near_misses_mean": near_misses_mean,
                    "snqi_mean": snqi_mean,
                }
            )
    return rows


def _rank_by(rows: list[dict[str, Any]], *, key: str, ascending: bool) -> tuple[list[str], bool]:
    """Return the planner_key ranking and whether the top raw value is tied.

    ``ascending=True`` means smaller raw value is better (e.g. collisions);
    ``ascending=False`` means larger raw value is better (e.g. success, snqi).
    ``planner_key`` remains the deterministic secondary sort key, but
    ``top_tied`` lets consumers distinguish real winner disagreements from
    alphabetical tie-breaking.
    """
    indexed = list(rows)
    indexed.sort(
        key=lambda row: (row[key] if ascending else -row[key], row["planner_key"]),
    )
    if not indexed:
        return [], False
    best_value = indexed[0][key]
    top_tied = sum(math.isclose(row[key], best_value, abs_tol=1e-12) for row in indexed) > 1
    return [row["planner_key"] for row in indexed], top_tied


def _kendall_tau(order_a: list[str], order_b: list[str]) -> float:
    """Kendall tau between two rankings of the same planner set.

    Computed directly from the rank indices so we avoid depending on SciPy.
    Returns 0.0 when fewer than two planners are present.
    """
    if len(order_a) < 2 or set(order_a) != set(order_b):
        return 0.0
    index_b = {planner: idx for idx, planner in enumerate(order_b)}
    ranks_a = list(range(len(order_a)))
    ranks_b = [index_b[planner] for planner in order_a]
    concordant = 0
    discordant = 0
    n = len(order_a)
    for i in range(n):
        for j in range(i + 1, n):
            diff_a = ranks_a[i] - ranks_a[j]
            diff_b = ranks_b[i] - ranks_b[j]
            product = diff_a * diff_b
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return (concordant - discordant) / total


def _spearman_rank(order_a: list[str], order_b: list[str]) -> float:
    """Spearman rho between two planner orderings."""
    if len(order_a) < 2 or set(order_a) != set(order_b):
        return 0.0
    index_b = {planner: idx for idx, planner in enumerate(order_b)}
    x = list(range(len(order_a)))
    y = [index_b[planner] for planner in order_a]
    return spearman_correlation(x, y)


def _build_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the cross-ranking comparison payload."""
    snqi_order, snqi_top_tied = _rank_by(rows, key="snqi_mean", ascending=False)
    success_order, success_top_tied = _rank_by(rows, key="success_mean", ascending=False)
    collisions_order, collisions_top_tied = _rank_by(rows, key="collisions_mean", ascending=True)
    near_misses_order, near_misses_top_tied = _rank_by(rows, key="near_misses_mean", ascending=True)
    comparisons = {
        "success_mean": {
            "description": "Ranking by mean success (higher is better).",
            "order": success_order,
            "top_tied": success_top_tied,
            "kendall_tau_vs_snqi": _kendall_tau(snqi_order, success_order),
            "spearman_rho_vs_snqi": _spearman_rank(snqi_order, success_order),
            "winner_agrees_with_snqi": success_order[0] == snqi_order[0],
        },
        "collisions_mean": {
            "description": "Ranking by mean collisions (lower is better).",
            "order": collisions_order,
            "top_tied": collisions_top_tied,
            "kendall_tau_vs_snqi": _kendall_tau(snqi_order, collisions_order),
            "spearman_rho_vs_snqi": _spearman_rank(snqi_order, collisions_order),
            "winner_agrees_with_snqi": collisions_order[0] == snqi_order[0],
        },
        "near_misses_mean": {
            "description": "Ranking by mean near-misses (lower is better).",
            "order": near_misses_order,
            "top_tied": near_misses_top_tied,
            "kendall_tau_vs_snqi": _kendall_tau(snqi_order, near_misses_order),
            "spearman_rho_vs_snqi": _spearman_rank(snqi_order, near_misses_order),
            "winner_agrees_with_snqi": near_misses_order[0] == snqi_order[0],
        },
    }
    interpretation = _interpret(comparisons)
    return {
        "snqi_order": snqi_order,
        "snqi_top_tied": snqi_top_tied,
        "comparisons": comparisons,
        "interpretation": interpretation,
        "planner_rows": rows,
    }


def _interpret(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether SNQI reorders planners relative to single metrics.

    Kendall tau of 1.0 means identical ranking; lower values mean SNQI is
    load-bearing in the sense that a reviewer would draw different planner
    conclusions from a single-metric table.
    """
    min_tau = min(c["kendall_tau_vs_snqi"] for c in comparisons.values())
    all_winners_agree = all(c["winner_agrees_with_snqi"] for c in comparisons.values())
    decisive_winner_disagreement = any(
        not c["winner_agrees_with_snqi"] and not c["top_tied"] for c in comparisons.values()
    )
    any_top_tied = any(c["top_tied"] for c in comparisons.values())
    if math.isclose(min_tau, 1.0, abs_tol=1e-9) and all_winners_agree:
        verdict = "snqi_redundant"
        narrative = (
            "SNQI does not reorder planners relative to any tested single metric. "
            "The composite is cosmetic for this matrix; a single-metric table "
            "supports the same planner conclusions."
        )
    elif not decisive_winner_disagreement and (min_tau >= 0.7 or any_top_tied):
        verdict = "snqi_mostly_consistent"
        narrative = (
            "SNQI is broadly consistent with single-metric rankings but shifts "
            "some planner positions. Any tied single-metric winner should be "
            "reported as tied rather than as a decisive SNQI disagreement."
        )
    elif decisive_winner_disagreement:
        verdict = "snqi_changes_winner"
        narrative = (
            "SNQI selects a different top planner than at least one single "
            "metric. The composite is load-bearing and must be justified "
            "explicitly against single-metric alternatives in the paper."
        )
    else:
        verdict = "snqi_reorders_tail"
        narrative = (
            "SNQI preserves the top planner but materially reorders the tail. "
            "Report rankings under each metric so reviewers can audit the "
            "composite's effect on tail conclusions."
        )
    return {
        "verdict": verdict,
        "narrative": narrative,
        "min_kendall_tau_vs_snqi": min_tau,
        "all_single_metric_winners_agree_with_snqi": all_winners_agree,
        "decisive_single_metric_winner_disagreement": decisive_winner_disagreement,
        "any_single_metric_top_tied": any_top_tied,
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """Render the ablation payload as a human-readable Markdown report."""
    lines = [
        "# SNQI vs Single-Metric Ranking Ablation",
        "",
        f"- Verdict: `{payload['interpretation']['verdict']}`",
        f"- Min Kendall tau vs SNQI: `{payload['interpretation']['min_kendall_tau_vs_snqi']:.4f}`",
        f"- All single-metric winners agree with SNQI winner: "
        f"`{payload['interpretation']['all_single_metric_winners_agree_with_snqi']}`",
        f"- Any single-metric top tie: `{payload['interpretation']['any_single_metric_top_tied']}`",
        "",
        "## Narrative",
        "",
        payload["interpretation"]["narrative"],
        "",
        "## SNQI Order (best to worst)",
        "",
    ]
    for idx, planner in enumerate(payload["snqi_order"], start=1):
        lines.append(f"{idx}. `{planner}`")
    lines.append("")
    lines.append("## Single-Metric Comparisons")
    lines.append("")
    lines.append(
        "| Metric | Kendall tau vs SNQI | Spearman rho vs SNQI | Winner agrees | Top tied |"
    )
    lines.append("| --- | ---: | ---: | :---: | :---: |")
    for metric, info in payload["comparisons"].items():
        lines.append(
            f"| `{metric}` | {info['kendall_tau_vs_snqi']:.4f} | "
            f"{info['spearman_rho_vs_snqi']:.4f} | "
            f"{'yes' if info['winner_agrees_with_snqi'] else 'no'} | "
            f"{'yes' if info['top_tied'] else 'no'} |"
        )
    lines.append("")
    for metric, info in payload["comparisons"].items():
        lines.append(f"### Order under `{metric}`")
        lines.append("")
        for idx, planner in enumerate(info["order"], start=1):
            lines.append(f"{idx}. `{planner}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the SNQI-vs-single-metric ranking ablation against a campaign directory."""
    args = _parse_args(argv)
    rows = _load_planner_rows(args.campaign_root, core_only=args.core_only)
    if len(rows) < 2:
        print(
            f"Need at least 2 planner rows to compare rankings, got {len(rows)}.",
            file=sys.stderr,
        )
        return 2
    payload = _build_analysis(rows)
    out_json = args.output_json or (
        args.campaign_root / "reports" / "snqi_vs_single_metric_ranking.json"
    )
    out_md = args.output_md or (args.campaign_root / "reports" / "snqi_vs_single_metric_ranking.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(out_md, payload)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
