"""Example: Generate a Pareto plot from episodes JSONL or synthetic fallback.

Usage (JSONL):
  uv run python examples/plot_pareto.py --in results/episodes.jsonl --out results/pareto.png \
    --x-metric collisions --y-metric comfort_exposure

Usage (synthetic fallback):
  uv run python examples/plot_pareto.py --out results/pareto_demo.png --x-metric collisions \
    --y-metric comfort_exposure --synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.plots import save_pareto_png


def _synthetic_records():
    return [
        {
            "scenario_id": "s1",
            "scenario_params": {"algo": "A"},
            "metrics": {"collisions": 1.0, "comfort_exposure": 0.5},
        },
        {
            "scenario_id": "s2",
            "scenario_params": {"algo": "A"},
            "metrics": {"collisions": 1.2, "comfort_exposure": 0.6},
        },
        {
            "scenario_id": "s3",
            "scenario_params": {"algo": "B"},
            "metrics": {"collisions": 0.8, "comfort_exposure": 0.9},
        },
        {
            "scenario_id": "s4",
            "scenario_params": {"algo": "C"},
            "metrics": {"collisions": 1.5, "comfort_exposure": 0.4},
        },
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=None)
    ap.add_argument("--out", default="results/pareto_demo.png")
    ap.add_argument("--x-metric", default="collisions")
    ap.add_argument("--y-metric", default="comfort_exposure")
    ap.add_argument("--group-by", default="scenario_params.algo")
    ap.add_argument("--fallback-group-by", default="scenario_id")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    ap.add_argument("--x-higher-better", action="store_true", default=False)
    ap.add_argument("--y-higher-better", action="store_true", default=False)
    ap.add_argument("--title", default=None)
    ap.add_argument("--synthetic", action="store_true", default=False)
    ap.add_argument("--out-pdf", default=None, help="Optional vector PDF path (LaTeX-ready)")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        records = _synthetic_records()
    else:
        if not args.in_path:
            raise SystemExit("--in is required unless --synthetic is set")
        records = read_jsonl(args.in_path)

    meta = save_pareto_png(
        records,
        out_path=str(out),
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        group_by=args.group_by,
        fallback_group_by=args.fallback_group_by,
        agg=args.agg,
        x_higher_better=bool(args.x_higher_better),
        y_higher_better=bool(args.y_higher_better),
        title=args.title,
        out_pdf=(str(args.out_pdf) if args.out_pdf else None),
    )
    print({"wrote": str(out), **meta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
