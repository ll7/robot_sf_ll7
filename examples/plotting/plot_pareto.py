"""Generate Pareto front plots from benchmark episodes or synthetic data.

Purpose:
    Aggregate episode metrics and produce Pareto scatter plots for model
    comparison with optional PDF export.

Usage (JSONL):
    uv run python examples/plotting/plot_pareto.py --in output/results/episodes.jsonl --out output/results/pareto.png \
    --x-metric collisions --y-metric comfort_exposure

Usage (synthetic fallback):
    uv run python examples/plotting/plot_pareto.py --out output/results/pareto_demo.png --x-metric collisions \
    --y-metric comfort_exposure --synthetic

Usage (deterministic fixture demo):
    uv run python examples/plotting/plot_pareto.py --fixture --out-dir output/example-pareto-smoke

Prerequisites:
    - Episodes JSONL with `metrics` fields (unless `--synthetic` or `--fixture` flag is used)

Expected Output:
    - PNG chart at the path passed via `--out`
    - Optional PDF when `--out-pdf` is supplied
    - Fixture mode: PNG, SVG, and JSON summary inside `--out-dir`

Limitations:
    - CLI expects valid metric names; refer to `robot_sf.benchmark.aggregate` outputs.
    - Fixture output demonstrates plotting mechanics only and is not benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.plots import save_pareto_png
from robot_sf.common.artifact_paths import resolve_artifact_path

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "pareto_dominated_tie.json"


def _synthetic_records():
    """Parse command-line arguments for the pareto plot demo."""
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


def _load_fixture_records() -> list[dict]:
    """Load the tracked deterministic fixture dataset."""
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise SystemExit("fixture dataset is empty or malformed")
    return records


def _validate_fixture_records(records: list[dict], x_metric: str, y_metric: str) -> None:
    """Fail closed on missing/non-finite metrics and duplicate scenario identities."""
    seen: set[str] = set()
    for record in records:
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise SystemExit("fixture record is missing a scenario_id")
        if scenario_id in seen:
            raise SystemExit(f"duplicate scenario_id in fixture: {scenario_id}")
        seen.add(scenario_id)
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            raise SystemExit(f"fixture record {scenario_id} is missing metrics")
        for metric in (x_metric, y_metric):
            value = metrics.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise SystemExit(f"fixture record {scenario_id} has missing or non-finite {metric}")


def main(argv: list[str] | None = None) -> int:
    """Run the pareto plot demo, generating visualization from benchmark data.

    Args:
        argv: Optional command-line arguments (uses sys.argv if None).

    Returns:
        Exit code (0 for success).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default=None)
    ap.add_argument("--out", default="output/results/pareto_demo.png")
    ap.add_argument("--x-metric", default="collisions")
    ap.add_argument("--y-metric", default="comfort_exposure")
    ap.add_argument("--group-by", default="scenario_params.algo")
    ap.add_argument("--fallback-group-by", default="scenario_id")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    ap.add_argument("--x-higher-better", action="store_true", default=False)
    ap.add_argument("--y-higher-better", action="store_true", default=False)
    ap.add_argument("--title", default=None)
    ap.add_argument("--synthetic", action="store_true", default=False)
    ap.add_argument("--fixture", action="store_true", default=False)
    ap.add_argument("--out-pdf", default=None, help="Optional vector PDF path (LaTeX-ready)")
    ap.add_argument("--out-svg", default=None, help="Optional vector SVG path")
    ap.add_argument("--out-dir", default=None, help="Fixture-mode output directory")
    ap.add_argument("--out-json", default=None, help="Fixture-mode JSON summary path")
    args = ap.parse_args(argv)

    if args.fixture or (not args.in_path and not args.synthetic):
        if not args.in_path and not args.synthetic and not args.fixture:
            print("no input source given; running deterministic fixture demo")
        return _run_fixture(args)

    out = resolve_artifact_path(Path(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = resolve_artifact_path(Path(args.out_pdf)) if args.out_pdf else None
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)

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
        out_pdf=(str(out_pdf) if out_pdf else None),
    )
    print({"wrote": str(out), **meta})
    return 0


def _run_fixture(args: argparse.Namespace) -> int:
    """Run the deterministic fixture demo into a caller-owned directory."""
    out_dir = resolve_artifact_path(Path(args.out_dir or "output/example-pareto-smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_fixture_records()
    _validate_fixture_records(records, args.x_metric, args.y_metric)
    out_png = out_dir / "pareto_fixture.png"
    out_svg = out_dir / "pareto_fixture.svg"
    out_json = Path(args.out_json) if args.out_json else out_dir / "summary.json"
    if args.out_json:
        out_json = resolve_artifact_path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
    meta = save_pareto_png(
        records,
        out_path=str(out_png),
        x_metric=args.x_metric,
        y_metric=args.y_metric,
        group_by=args.group_by,
        fallback_group_by=args.fallback_group_by,
        agg=args.agg,
        x_higher_better=bool(args.x_higher_better),
        y_higher_better=bool(args.y_higher_better),
        title=args.title or "Pareto fixture demo (plotting mechanics only)",
        out_svg=str(out_svg),
    )
    summary = {
        "mode": "fixture",
        "fixture": str(FIXTURE_PATH.name),
        "mechanics_only": True,
        **{key: meta[key] for key in ("count", "front_size", "front_labels") if key in meta},
    }
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print({"wrote": [str(out_png), str(out_svg), str(out_json)], **meta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
