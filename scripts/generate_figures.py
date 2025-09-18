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
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from results.figures.fig_force_field import generate_force_field_figure
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.distributions import collect_grouped_values, save_distributions
from robot_sf.benchmark.metrics import snqi as _snqi
from robot_sf.benchmark.plots import save_pareto_png
from robot_sf.benchmark.report_table import compute_table, format_markdown
from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.benchmark.scenario_thumbnails import save_montage, save_scenario_thumbnails

SCHEMA_VERSION = 1  # Fallback schema version; replaced at runtime if inferred from episodes
SCRIPT_VERSION = "0.1.0"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _git_sha_short(length: int = 7) -> str:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", f"--short={length}", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
        return sha or "unknown"
    except Exception:
        return "unknown"


def _compute_auto_out_dir(episodes: Path, base_dir: Path | None) -> Path:
    stem = episodes.stem
    sha = _git_sha_short()
    version = _infer_schema_version(episodes) or SCHEMA_VERSION
    folder = f"{stem}__{sha}__v{version}"
    base = base_dir if base_dir is not None else Path("docs/figures")
    return base / folder


def _write_meta(out_dir: Path, episodes: Path, args: argparse.Namespace) -> None:
    meta = {
        "episodes_path": str(episodes),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha_short(),
        "schema_version": _infer_schema_version(episodes) or SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _infer_schema_version(episodes_path: Path) -> int | None:
    """Best-effort detection of schema_version from the JSONL file.

    Looks for either top-level "schema_version" or nested under "_metadata.schema_version".
    Returns None if not found or unreadable.
    """
    try:
        with episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if "schema_version" in obj and isinstance(obj["schema_version"], int):
                    return int(obj["schema_version"])
                meta = obj.get("_metadata")
                if isinstance(meta, dict) and isinstance(meta.get("schema_version"), int):
                    return int(meta["schema_version"])
                # If we got a valid dict and didn't find it, assume absent for the rest
                break
    except OSError:
        return None
    return None


def _load_snqi_inputs(
    weights_path: Path | None, weights_from: Path | None, baseline_path: Path | None
) -> tuple[dict | None, dict | None]:
    weights = None
    baseline = None
    try:
        if weights_path is not None and Path(weights_path).exists():
            with Path(weights_path).open("r", encoding="utf-8") as f:
                weights = json.load(f)
        elif weights_from is not None and Path(weights_from).exists():
            with Path(weights_from).open("r", encoding="utf-8") as f:
                rep = json.load(f)
            if isinstance(rep, dict):
                weights = (
                    rep.get("results", {}).get("recommended", {}).get("weights")
                    or rep.get("recommended", {}).get("weights")
                    or rep.get("recommended_weights")
                )
        if baseline_path is not None and Path(baseline_path).exists():
            with Path(baseline_path).open("r", encoding="utf-8") as f:
                baseline = json.load(f)
    except Exception:
        # Best-effort; ignore malformed files silently for figure generation
        pass
    return (
        weights if isinstance(weights, dict) else None,
        baseline if isinstance(baseline, dict) else None,
    )


def _inject_snqi(records: list[dict], weights: dict | None, baseline: dict | None) -> None:
    if not isinstance(weights, dict):
        return
    for rec in records:
        m = rec.get("metrics") or {}
        if not isinstance(m, dict):
            continue
        if "snqi" in m:
            continue
        try:
            m["snqi"] = float(_snqi(m, weights, baseline))
            rec["metrics"] = m
        except Exception:
            # Leave record unchanged if computation fails for any reason
            continue


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate benchmark figures from episodes JSONL (and optional thumbnails)"
    )
    ap.add_argument("--episodes", required=True, help="Episodes JSONL path")
    ap.add_argument(
        "--out-dir",
        required=False,
        default=None,
        help="Output directory for figures. If omitted with --auto-out-dir, a canonical folder will be computed under docs/figures.",
    )
    ap.add_argument(
        "--auto-out-dir",
        action="store_true",
        default=False,
        help="Compute canonical output folder name based on episodes stem, git sha, and schema version. Appends to --out-dir when provided, otherwise uses docs/figures as base.",
    )
    ap.add_argument(
        "--set-latest",
        action="store_true",
        default=False,
        help="Update docs/figures/_latest.txt to point to the generated figures folder.",
    )
    ap.add_argument("--group-by", default="scenario_params.algo")
    ap.add_argument("--fallback-group-by", default="scenario_id")
    # SNQI inputs (optional)
    ap.add_argument("--snqi-weights", type=Path, default=None, help="Path to SNQI weights JSON")
    ap.add_argument(
        "--snqi-weights-from",
        type=Path,
        default=None,
        help="Path to JSON report containing recommended weights (fallback when --snqi-weights is not provided)",
    )
    ap.add_argument(
        "--snqi-baseline",
        type=Path,
        default=None,
        help="Path to baseline stats JSON (median/p95 per metric)",
    )
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

    if bool(args.auto_out_dir):
        base = Path(args.out_dir) if args.out_dir else None
        out_dir = _compute_auto_out_dir(episodes, base)
    else:
        if args.out_dir is None:
            raise SystemExit("--out-dir is required unless --auto-out-dir is specified")
        out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    records = read_jsonl(episodes)

    # Load SNQI inputs if provided and inject SNQI per episode
    snqi_weights, snqi_baseline = _load_snqi_inputs(
        args.snqi_weights, args.snqi_weights_from, args.snqi_baseline
    )
    _inject_snqi(records, snqi_weights, snqi_baseline)

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
    # If SNQI was computed and not explicitly requested, append it for convenience
    if any(isinstance(r.get("metrics"), dict) and "snqi" in r["metrics"] for r in records):
        if "snqi" not in table_metrics:
            table_metrics.append("snqi")
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

    # Write meta.json and optionally update _latest.txt
    _write_meta(out_dir, episodes, args)
    if bool(args.set_latest):
        latest_file = Path("docs/figures/_latest.txt")
        latest_file.parent.mkdir(parents=True, exist_ok=True)
        # Write path relative to docs/figures for portability when included in LaTeX/docs
        try:
            rel = out_dir.relative_to(Path("docs/figures"))
            latest_file.write_text(str(rel), encoding="utf-8")
        except ValueError:
            # If not under docs/figures, write the folder name
            latest_file.write_text(out_dir.name, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
