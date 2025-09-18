"""Generate benchmark figures from episodes JSONL and optional thumbnails.

This script produces Pareto plots, distributions, a baseline comparison table,
and optionally scenario thumbnails plus a montage.

Usage (example):
  uv run python scripts/generate_figures.py \
    --episodes results/episodes.jsonl \
    --out-dir docs/figures \
    --pareto-x collisions --pareto-y comfort_exposure --pareto-pdf \
    --dmetrics collisions,comfort_exposure --dists-pdf \
    --table-metrics collisions,comfort_exposure \
    --thumbs-matrix configs/baselines/example_matrix.yaml --thumbs-montage --thumbs-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

from results.figures.fig_force_field import generate_force_field_figure
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.distributions import collect_grouped_values, save_distributions
from robot_sf.benchmark.plots import save_pareto_png
from robot_sf.benchmark.report_table import compute_table, format_markdown
from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.benchmark.scenario_thumbnails import save_montage, save_scenario_thumbnails


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate benchmark figures from episodes JSONL (and optional thumbnails)"
    )
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
    # Force-field figure
    ap.add_argument(
        "--force-field", action="store_true", default=False, help="Generate force-field figure"
    )
    ap.add_argument(
        "--ff-png",
        default=None,
        help="Output PNG for force-field (default: <out-dir>/fig-force-field.png)",
    )
    ap.add_argument(
        "--ff-pdf",
        default=None,
        help="Output PDF for force-field (default: <out-dir>/fig-force-field.pdf)",
    )
    ap.add_argument("--ff-x-min", type=float, default=-1.0)
    ap.add_argument("--ff-x-max", type=float, default=5.0)
    ap.add_argument("--ff-y-min", type=float, default=-2.0)
    ap.add_argument("--ff-y-max", type=float, default=3.0)
    ap.add_argument("--ff-grid", type=int, default=120)
    ap.add_argument("--ff-quiver-step", type=int, default=5)
    # Table
    ap.add_argument("--table-metrics", default="collisions,comfort_exposure")
    # Thumbnails
    ap.add_argument("--thumbs-matrix", default=None, help="Scenario matrix YAML for thumbnails")
    ap.add_argument("--thumbs-pdf", action="store_true", default=False)
    ap.add_argument("--thumbs-cols", type=int, default=3)
    ap.add_argument(
        "--thumbs-out-dir",
        default=None,
        help="Output directory for thumbnails (default: <out-dir>/scenarios)",
    )
    ap.add_argument("--thumbs-montage", action="store_true", default=False)
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

    # Optional thumbnails
    if args.thumbs_matrix:
        thumbs_out = Path(args.thumbs_out_dir) if args.thumbs_out_dir else (out_dir / "scenarios")
        thumbs_out.mkdir(parents=True, exist_ok=True)
        scenarios = load_scenario_matrix(args.thumbs_matrix)
        metas = save_scenario_thumbnails(
            scenarios, out_dir=thumbs_out, out_pdf=bool(args.thumbs_pdf)
        )
        if bool(args.thumbs_montage):
            save_montage(
                metas,
                out_png=str(thumbs_out / "montage.png"),
                cols=int(args.thumbs_cols),
                out_pdf=(str(thumbs_out / "montage.pdf") if args.thumbs_pdf else None),
            )

    # Optional force-field figure
    if bool(args.force_field):
        ff_png = Path(args.ff_png) if args.ff_png else (out_dir / "fig-force-field.png")
        ff_pdf = (
            str(Path(args.ff_pdf))
            if args.ff_pdf is not None
            else str(out_dir / "fig-force-field.pdf")
        )
        generate_force_field_figure(
            out_png=str(ff_png),
            out_pdf=ff_pdf,
            x_min=float(args.ff_x_min),
            x_max=float(args.ff_x_max),
            y_min=float(args.ff_y_min),
            y_max=float(args.ff_y_max),
            grid=int(args.ff_grid),
            quiver_step=int(args.ff_quiver_step),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
