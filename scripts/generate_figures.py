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
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from pathlib import Path

from results.figures.fig_force_field import generate_force_field_figure
from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.distributions import collect_grouped_values, save_distributions
from robot_sf.benchmark.figures.thumbnails import save_montage, save_scenario_thumbnails
from robot_sf.benchmark.metrics import snqi as _snqi
from robot_sf.benchmark.plots import save_pareto_png
from robot_sf.benchmark.report_table import (
    TableRow,
    compute_table,
    format_latex_booktabs,
    format_markdown,
)
from robot_sf.benchmark.runner import load_scenario_matrix

SCHEMA_VERSION = 1  # Fallback schema version; replaced at runtime if inferred from episodes
SCRIPT_VERSION = "0.1.0"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _git_sha_short(length: int = 7) -> str:
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", f"--short={length}", "HEAD"],
                stderr=subprocess.DEVNULL,
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
        "generated_at": datetime.now(UTC).isoformat(),
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
    weights_path: Path | None,
    weights_from: Path | None,
    baseline_path: Path | None,
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
        description="Generate benchmark figures from episodes JSONL (and optional thumbnails)",
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
    ap.add_argument(
        "--no-pareto",
        action="store_true",
        default=False,
        help="Skip Pareto plot generation entirely",
    )
    # Distributions
    ap.add_argument("--dmetrics", default="collisions,comfort_exposure")
    ap.add_argument("--dists-bins", type=int, default=30)
    ap.add_argument("--dists-kde", action="store_true", default=False)
    ap.add_argument("--dists-pdf", action="store_true", default=False)
    # Force-field figure
    ap.add_argument(
        "--force-field",
        action="store_true",
        default=False,
        help="Generate force-field figure",
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
    ap.add_argument(
        "--table-tex",
        action="store_true",
        default=False,
        help="Also emit baseline_table.tex (LaTeX booktabs) alongside Markdown table",
    )
    ap.add_argument(
        "--table-summary",
        type=Path,
        default=None,
        help="Path to pre-computed aggregate summary JSON (group -> metric -> stats) to build table from instead of raw episodes",
    )
    ap.add_argument(
        "--table-stats",
        default="mean",
        help="Comma-separated stats to include from summary when --table-summary is used (e.g. mean,median,p95)",
    )
    ap.add_argument(
        "--table-include-ci",
        action="store_true",
        default=False,
        help="When using --table-summary, include confidence interval columns. Default suffix is 'ci', producing *_ci_low/high.",
    )
    ap.add_argument(
        "--ci-column-suffix",
        default="ci",
        help="Suffix used for CI columns when --table-include-ci is set (e.g. ci -> *_ci_low/high, ci95 -> *_ci95_low/high).",
    )
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

    # SNQI injection (optional)
    snqi_weights, snqi_baseline = _load_snqi_inputs(
        args.snqi_weights,
        args.snqi_weights_from,
        args.snqi_baseline,
    )
    _inject_snqi(records, snqi_weights, snqi_baseline)

    # Pareto (unless disabled)
    if not args.no_pareto:
        _generate_pareto(records, out_dir, args)

    _generate_distributions(records, out_dir, args)
    _generate_table(records, out_dir, args)
    _maybe_thumbnails(out_dir, args)
    _maybe_force_field(out_dir, args)

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


def _generate_pareto(records, out_dir, args) -> None:
    """Generate Pareto plot (factored out to reduce main complexity)."""
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


def _generate_distributions(records, out_dir: Path, args) -> None:
    dmetrics = [m.strip() for m in str(args.dmetrics).split(",") if m.strip()]
    grouped = collect_grouped_values(
        records,
        metrics=dmetrics,
        group_by=args.group_by,
        fallback_group_by=args.fallback_group_by,
    )
    save_distributions(
        grouped,
        out_dir=out_dir,
        bins=int(args.dists_bins),
        kde=bool(args.dists_kde),
        out_pdf=bool(args.dists_pdf),
    )


def _generate_table(records, out_dir: Path, args) -> None:
    if args.table_summary:
        rows, metric_cols = _rows_from_summary(
            summary_path=Path(args.table_summary),
            metrics=[m.strip() for m in str(args.table_metrics).split(",") if m.strip()],
            stats=[s.strip() for s in str(args.table_stats).split(",") if s.strip()],
            include_ci=bool(args.table_include_ci),
            ci_suffix=str(args.ci_column_suffix),
        )
    else:
        metric_cols = [m.strip() for m in str(args.table_metrics).split(",") if m.strip()]
        if any(isinstance(r.get("metrics"), dict) and "snqi" in r["metrics"] for r in records):
            if "snqi" not in metric_cols:
                metric_cols.append("snqi")
        rows = compute_table(
            records,
            metrics=metric_cols,
            group_by=args.group_by,
            fallback_group_by=args.fallback_group_by,
        )
    (out_dir / "baseline_table.md").write_text(format_markdown(rows, metric_cols), encoding="utf-8")
    if bool(getattr(args, "table_tex", False)):
        (out_dir / "baseline_table.tex").write_text(
            format_latex_booktabs(rows, metric_cols),
            encoding="utf-8",
        )


def _maybe_thumbnails(out_dir: Path, args) -> None:
    if not args.thumbs_matrix:
        return
    thumbs_out = Path(args.thumbs_out_dir) if args.thumbs_out_dir else (out_dir / "scenarios")
    thumbs_out.mkdir(parents=True, exist_ok=True)
    scenarios = load_scenario_matrix(args.thumbs_matrix)
    metas = save_scenario_thumbnails(scenarios, out_dir=thumbs_out, out_pdf=bool(args.thumbs_pdf))
    if bool(args.thumbs_montage):
        save_montage(
            metas,
            out_png=str(thumbs_out / "montage.png"),
            cols=int(args.thumbs_cols),
            out_pdf=(str(thumbs_out / "montage.pdf") if args.thumbs_pdf else None),
        )


def _maybe_force_field(out_dir: Path, args) -> None:
    if not bool(args.force_field):
        return
    ff_png = Path(args.ff_png) if args.ff_png else (out_dir / "fig-force-field.png")
    ff_pdf = (
        str(Path(args.ff_pdf)) if args.ff_pdf is not None else str(out_dir / "fig-force-field.pdf")
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


def _summary_build_columns(
    metrics: list[str],
    stats: list[str],
    include_ci: bool,
    ci_suffix: str,
) -> list[str]:
    cols: list[str] = []
    for m in metrics:
        for st in stats:
            base = f"{m}_{st}"
            cols.append(base)
            if include_ci:
                cols.append(f"{base}_{ci_suffix}_low")
                cols.append(f"{base}_{ci_suffix}_high")
    return cols


def _summary_ci_pair(metric_dict: dict, stat: str) -> tuple[float | None, float | None]:
    ci = metric_dict.get(f"{stat}_ci")
    if not (isinstance(ci, list | tuple) and len(ci) == 2):
        return None, None
    a, b = ci
    if not isinstance(a, int | float) or not isinstance(b, int | float):
        return None, None
    return float(a), float(b)


def _summary_extract_row(
    group: str,
    metrics_map: dict,
    metrics: list[str],
    stats: list[str],
    include_ci: bool,
    ci_suffix: str,
    missing: set[str],
) -> TableRow:
    values: dict[str, float] = {}
    for m in metrics:
        mm = metrics_map.get(m)
        if not isinstance(mm, dict):
            if include_ci:
                for st in stats:
                    missing.add(f"{m}:{st}")
            continue
        for st in stats:
            base = f"{m}_{st}"
            val = mm.get(st)
            if isinstance(val, int | float):
                values[base] = float(val)
            if not include_ci:
                continue
            lo, hi = _summary_ci_pair(mm, st)
            if lo is None or hi is None:
                missing.add(f"{m}:{st}")
                continue
            values[f"{base}_{ci_suffix}_low"] = lo
            values[f"{base}_{ci_suffix}_high"] = hi
    return TableRow(group=group, values=values)


def _rows_from_summary(
    summary_path: Path,
    *,
    metrics: list[str],
    stats: list[str],
    include_ci: bool,
    ci_suffix: str,
) -> tuple[list[TableRow], list[str]]:
    """Convert an aggregate summary JSON into TableRow objects.

    Structure expected: {group: {metric: {stat: value, stat_ci: [low, high]}}}.
    Produces columns <metric>_<stat> (+ optional CI low/high with suffix).
    Missing CIs yield blank cells and a consolidated warning.
    """
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    columns = _summary_build_columns(metrics, stats, include_ci, ci_suffix)
    missing: set[str] = set()
    rows: list[TableRow] = []
    for group, metrics_map in raw.items():
        if not isinstance(metrics_map, dict):  # skip malformed groups
            continue
        rows.append(
            _summary_extract_row(
                group,
                metrics_map,
                metrics,
                stats,
                include_ci,
                ci_suffix,
                missing,
            ),
        )
    if include_ci and missing:
        import sys

        print(
            f"[WARN] Missing CI arrays for {len(missing)} metric/stat combos: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
    return rows, columns


if __name__ == "__main__":
    raise SystemExit(main())
