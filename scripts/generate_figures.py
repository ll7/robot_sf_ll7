"""Generate benchmark figures from episodes JSONL.

This script produces:
- Pareto front PNG (+ optional PDF) for two chosen metrics
- Distribution plots for selected metrics (PNG + optional PDF)
- Baseline comparison table (Markdown)

Usage (example):
  uv run python scripts/generate_figures.py \
    --episodes results/episodes.jsonl \
    --out-dir docs/figures \
    --pareto-x collisions --pareto-y comfort_exposure --pareto-pdf \
    --dmetrics collisions,comfort_exposure --dists-pdf \
    --table-metrics collisions,comfort_exposure
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.distributions import collect_grouped_values, save_distributions
from robot_sf.benchmark.plots import save_pareto_png
from robot_sf.benchmark.report_table import compute_table, format_markdown


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate benchmark figures from episodes JSONL")
    ap.add_argument("--episodes", required=True, help="Episodes JSONL path")
    ap.add_argument("--out-dir", required=True, help="Output directory for figures")
    ap.add_argument("--group-by", default="scenario_params.algo")
    ap.add_argument("--fallback-group-by", default="scenario_id")
    # Pareto
    ap.add_argument("--pareto-x", default="collisions")
    ap.add_argument("--pareto-y", default="comfort_exposure")
    ap.add_argument("--pareto-agg", choices=["mean", "median"], default="mean")
    ap.add_argument("--pareto-pdf", action="store_true", default=False)
    # Distributions
    ap.add_argument("--dmetrics", default="collisions,comfort_exposure")
    ap.add_argument("--dists-bins", type=int, default=30)
    ap.add_argument("--dists-kde", action="store_true", default=False)
    ap.add_argument("--dists-pdf", action="store_true", default=False)
    # Table
    ap.add_argument("--table-metrics", default="collisions,comfort_exposure")
    args = ap.parse_args()

    episodes = Path(args.episodes)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    records = read_jsonl(episodes)

    # Pareto
    pareto_png = out_dir / "pareto.png"
    pareto_pdf = out_dir / "pareto.pdf" if args.pareto_pdf else None
    save_pareto_png(
        records,
        out_path=str(pareto_png),
        x_metric=args.pareto_x,
        y_metric=args.pareto_y,
        group_by=args.group_by,
        fallback_group_by=args.fallback_group_by,
        agg=args.pareto_agg,
        out_pdf=(str(pareto_pdf) if pareto_pdf else None),
    )

    # Distributions
    dmetrics = [m.strip() for m in str(args.dmetrics).split(",") if m.strip()]
    grouped = collect_grouped_values(
        records, metrics=dmetrics, group_by=args.group_by, fallback_group_by=args.fallback_group_by
    )
    save_distributions(
        grouped,
        out_dir=out_dir,
        bins=int(args.dists_bins),
        kde=bool(args.dists_kde),
        out_pdf=bool(args.dists_pdf),
    )

    # Baseline table (Markdown)
    table_metrics = [m.strip() for m in str(args.table_metrics).split(",") if m.strip()]
    rows = compute_table(
        records,
        metrics=table_metrics,
        group_by=args.group_by,
        fallback_group_by=args.fallback_group_by,
    )
    (out_dir / "baseline_table.md").write_text(
        format_markdown(rows, table_metrics), encoding="utf-8"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
