#!/usr/bin/env python3
"""Generate ranking tables by metric from benchmark episodes JSONL.

This script aggregates episode metrics per group (default: algorithm), then
builds a ranking table sorted by the chosen metric (default: SNQI, higher is
better). It can export to CSV and/or Markdown.

Typical usage:

    uv run python scripts/ranking_table.py \
      --episodes results/episodes.jsonl \
      --out-csv results/ranking_snqi.csv \
      --out-md results/ranking_snqi.md \
      --metric snqi \
      --group-by scenario_params.algo

Inputs
    - episodes: Episodes JSONL path.
    - group-by: Grouping dotted path (e.g., ``scenario_params.algo``).
    - metric: Metric key inside the flattened metrics (e.g., ``snqi``).

Outputs
    - CSV and/or Markdown table with group, metric stats, and rank.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.aggregate import compute_aggregates, read_jsonl


def to_rank_rows(
    aggregates: dict[str, dict[str, dict[str, float]]],
    metric: str,
) -> list[dict[str, Any]]:
    """Convert aggregate stats into a sorted ranking list of rows.

    Parameters
    ----------
    aggregates
        Mapping group → metric → stats dict with keys like ``mean``, ``median``, ``p95``.
    metric
        The metric key to rank by (e.g., ``snqi``).

    Returns
    -------
    list of dict
        Rows with keys: ``group``, ``{metric}_mean``, ``{metric}_median``, ``{metric}_p95``, ``rank``.
    """
    rows: list[dict[str, Any]] = []
    for group, metrics in aggregates.items():
        m = metrics.get(metric)
        if not m:
            continue
        rows.append(
            {
                "group": group,
                f"{metric}_mean": m.get("mean"),
                f"{metric}_median": m.get("median"),
                f"{metric}_p95": m.get("p95"),
            },
        )
    # Higher is better for SNQI; support reverse flag by caller if needed
    rows.sort(
        key=lambda r: (r.get(f"{metric}_mean") is None, r.get(f"{metric}_mean", -np.inf)),
        reverse=True,
    )
    # Add rank
    rank = 1
    for r in rows:
        r["rank"] = rank
        rank += 1
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: str | Path) -> str:
    """TODO docstring. Document this function.

    Args:
        rows: TODO docstring.
        out_csv: TODO docstring.

    Returns:
        TODO docstring.
    """
    import csv

    if not rows:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        Path(out_csv).write_text("group,rank\n", encoding="utf-8")
        return str(out_csv)

    keys = list(rows[0].keys())
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in keys})
    return str(out_csv)


def write_markdown(rows: list[dict[str, Any]], out_md: str | Path) -> str:
    """TODO docstring. Document this function.

    Args:
        rows: TODO docstring.
        out_md: TODO docstring.

    Returns:
        TODO docstring.
    """
    if not rows:
        Path(out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(out_md).write_text("| group | rank |\n|---|---|\n", encoding="utf-8")
        return str(out_md)

    keys = list(rows[0].keys())
    header = "| " + " | ".join(keys) + " |\n"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |\n"
    body = "\n".join("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |" for r in rows) + "\n"
    Path(out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(out_md).write_text(header + sep + body, encoding="utf-8")
    return str(out_md)


def main():
    """CLI entry point to generate ranking tables by metric."""
    ap = argparse.ArgumentParser(description="Generate ranking table by metric from episodes JSONL")
    ap.add_argument("--episodes", required=True, help="Input episodes JSONL")
    ap.add_argument("--out-csv", help="Output CSV path")
    ap.add_argument("--out-md", help="Output Markdown table path")
    ap.add_argument("--metric", default="snqi", help="Metric to rank by (default: snqi)")
    ap.add_argument("--group-by", default="scenario_params.algo", help="Grouping key (dotted path)")
    args = ap.parse_args()

    records = read_jsonl(args.episodes)
    aggregates = compute_aggregates(records, group_by=args.group_by)
    rows = to_rank_rows(aggregates, args.metric)

    if args.out_csv:
        write_csv(rows, args.out_csv)
    if args.out_md:
        write_markdown(rows, args.out_md)

    # Always print a small summary
    preview = rows[:5]
    print({"rows": len(rows), "preview": preview})


if __name__ == "__main__":
    main()
