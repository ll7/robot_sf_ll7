"""Benchmark CLI providing unified entrypoints (including SNQI tooling)."""
# ruff: noqa: C901  # complexity acceptable for nested CLI builders/wrappers

# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List

from robot_sf.benchmark.ablation import (
    compute_ablation_summary as _abl_summary,
)
from robot_sf.benchmark.ablation import (
    compute_snqi_ablation as _abl_compute,
)
from robot_sf.benchmark.ablation import (
    format_csv as _abl_format_csv,
)
from robot_sf.benchmark.ablation import (
    format_markdown as _abl_format_md,
)
from robot_sf.benchmark.ablation import (
    to_json as _abl_to_json,
)
from robot_sf.benchmark.aggregate import compute_aggregates as _agg_compute
from robot_sf.benchmark.aggregate import compute_aggregates_with_ci as _agg_compute_ci
from robot_sf.benchmark.aggregate import read_jsonl as _agg_read_jsonl
from robot_sf.benchmark.baseline_stats import run_and_compute_baseline
from robot_sf.benchmark.failure_extractor import extract_failures as _extract_failures
from robot_sf.benchmark.ranking import compute_ranking as _compute_ranking
from robot_sf.benchmark.ranking import format_csv as _rank_format_csv
from robot_sf.benchmark.ranking import format_markdown as _rank_format_md
from robot_sf.benchmark.runner import load_scenario_matrix, run_batch
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.benchmark.seed_variance import compute_seed_variance as _compute_seed_variance
from robot_sf.benchmark.summary import summarize_to_plots

DEFAULT_SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _handle_baseline(args) -> int:
    try:
        progress_cb = _progress_cb_factory(bool(args.quiet))
        stats = run_and_compute_baseline(
            args.matrix,
            out_json=args.out,
            out_jsonl=args.jsonl,
            schema_path=args.schema,
            base_seed=args.base_seed,
            repeats_override=args.repeats,
            horizon=args.horizon,
            dt=args.dt,
            record_forces=args.record_forces,
            algo=args.algo,
            algo_config_path=args.algo_config,
            workers=args.workers,
            resume=(not bool(getattr(args, "no_resume", False))),
            progress_cb=progress_cb,
        )
        # Print brief summary to stdout for convenience
        print(json.dumps({"out": args.out, "keys": sorted(stats.keys())}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_list_algorithms(_args) -> int:
    try:
        # Show built-in simple policy
        algorithms = ["simple_policy"]

        # Try to load baseline algorithms
        try:
            from robot_sf.baselines import list_baselines

            baseline_algos = list_baselines()
            algorithms.extend(baseline_algos)
        except ImportError:
            print("Warning: Could not load baseline algorithms", file=sys.stderr)

        print("Available algorithms:")
        for algo in algorithms:
            print(f"  - {algo}")

        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _load_snqi_inputs(args):
    snqi_weights = None
    snqi_baseline = None
    # Priority: explicit weights JSON
    if getattr(args, "snqi_weights", None):
        with open(args.snqi_weights, "r", encoding="utf-8") as f:
            snqi_weights = json.load(f)
    elif getattr(args, "snqi_weights_from", None):
        with open(args.snqi_weights_from, "r", encoding="utf-8") as f:
            report = json.load(f)
        if isinstance(report, dict):
            snqi_weights = (
                report.get("results", {}).get("recommended", {}).get("weights")
                or report.get("recommended", {}).get("weights")
                or report.get("recommended_weights")
            )
        if not isinstance(snqi_weights, dict):
            raise ValueError("No recommended weights found in report JSON")
    if getattr(args, "snqi_baseline", None):
        with open(args.snqi_baseline, "r", encoding="utf-8") as f:
            snqi_baseline = json.load(f)
    return snqi_weights, snqi_baseline


def _progress_cb_factory(quiet: bool):
    # Try to use tqdm when available
    pbar = None
    if not quiet:
        try:  # pragma: no cover - tqdm optional
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=0, unit="ep", disable=False)
        except Exception:
            pbar = None

    def _cb(i, total, sc, seed, ok, err):
        if quiet:
            return
        status = "ok" if ok else "FAIL"
        sid = sc.get("id", "unknown")
        if pbar is not None:
            try:  # pragma: no cover - tqdm optional
                if pbar.total != total:
                    pbar.reset(total=total)
                pbar.update(1)
                pbar.set_description(f"{sid} seed={seed} {status}")
            except Exception:
                pass
        else:
            msg = f"[{i}/{total}] {sid} seed={seed}: {status}"
            if err:
                msg += f" ({err})"
            print(msg)

    return _cb


def _handle_run(args) -> int:
    try:
        # Optional: load SNQI weights/baseline for inline SNQI computation
        try:
            snqi_weights, snqi_baseline = _load_snqi_inputs(args)
        except Exception as e:  # pragma: no cover - error path
            print(f"Error loading SNQI inputs: {e}", file=sys.stderr)
            return 2

        summary = run_batch(
            scenarios_or_path=args.matrix,
            out_path=args.out,
            schema_path=args.schema,
            base_seed=args.base_seed,
            repeats_override=args.repeats,
            horizon=args.horizon,
            dt=args.dt,
            record_forces=args.record_forces,
            append=args.append,
            fail_fast=args.fail_fast,
            progress_cb=_progress_cb_factory(bool(args.quiet)),
            algo=args.algo,
            algo_config_path=args.algo_config,
            snqi_weights=snqi_weights,
            snqi_baseline=snqi_baseline,
            workers=args.workers,
            resume=(not bool(getattr(args, "no_resume", False))),
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_summary(args) -> int:
    try:
        outs = summarize_to_plots(args.in_path, args.out_dir)
        print(json.dumps({"wrote": outs}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_aggregate(args) -> int:
    try:
        # Optional: load SNQI inputs to compute metrics.snqi during aggregation
        snqi_weights = None
        snqi_baseline = None
        if getattr(args, "snqi_weights", None):
            with open(args.snqi_weights, "r", encoding="utf-8") as f:
                snqi_weights = json.load(f)
        if getattr(args, "snqi_baseline", None):
            with open(args.snqi_baseline, "r", encoding="utf-8") as f:
                snqi_baseline = json.load(f)

        records = _agg_read_jsonl(args.in_path)
        use_ci = int(getattr(args, "bootstrap_samples", 0)) > 0
        if use_ci:
            summary = _agg_compute_ci(
                records,
                group_by=args.group_by,
                fallback_group_by=args.fallback_group_by,
                snqi_weights=snqi_weights,
                snqi_baseline=snqi_baseline,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_confidence=float(args.bootstrap_confidence),
                bootstrap_seed=(
                    int(args.bootstrap_seed) if args.bootstrap_seed is not None else None
                ),
            )
        else:
            summary = _agg_compute(
                records,
                group_by=args.group_by,
                fallback_group_by=args.fallback_group_by,
                snqi_weights=snqi_weights,
                snqi_baseline=snqi_baseline,
            )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({"wrote": str(out_path)}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_snqi_ablate(args) -> int:
    try:
        records = _agg_read_jsonl(args.in_path)
        snqi_weights, snqi_baseline = _load_snqi_inputs(args)
        if not isinstance(snqi_weights, dict) or not isinstance(snqi_baseline, dict):
            print(
                "error: --snqi-weights and --snqi-baseline are required for snqi-ablate",
                file=sys.stderr,
            )
            return 2
        rows = _abl_compute(
            records,
            weights=snqi_weights,  # type: ignore[arg-type]
            baseline=snqi_baseline,  # type: ignore[arg-type]
            group_by=args.group_by,
            fallback_group_by=args.fallback_group_by,
            top=(int(args.top) if args.top is not None else None),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = (args.format or "md").lower()
        if fmt == "md":
            content = _abl_format_md(rows)
            out_path.write_text(content, encoding="utf-8")
        elif fmt == "csv":
            content = _abl_format_csv(rows)
            out_path.write_text(content, encoding="utf-8")
        elif fmt == "json":
            content = json.dumps(_abl_to_json(rows), indent=2)
            out_path.write_text(content + "\n", encoding="utf-8")
        else:
            print(
                f"error: unknown format '{args.format}', expected one of: md,csv,json",
                file=sys.stderr,
            )
            return 2
        # Optional summary JSON for ablation (per-weight impact stats)
        summary_written = None
        if getattr(args, "summary_out", None):
            summary_out = Path(args.summary_out)
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_payload = _abl_summary(rows)
            summary_out.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
            summary_written = str(summary_out)
        print(
            json.dumps(
                {
                    "wrote": str(out_path),
                    "rows": len(rows),
                    **({"summary": summary_written} if summary_written else {}),
                },
                indent=2,
            )
        )
        return 0
    except Exception as e:  # pragma: no cover - defensive
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_seed_variance(args) -> int:
    try:
        records = _agg_read_jsonl(args.in_path)
        metrics = None
        if getattr(args, "metrics", None):
            # Split comma-separated list, strip whitespace
            metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
        summary = _compute_seed_variance(
            records,
            group_by=args.group_by,
            fallback_group_by=args.fallback_group_by,
            metrics=metrics,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({"wrote": str(out_path)}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_extract_failures(args) -> int:
    try:
        records = _agg_read_jsonl(args.in_path)
        failures = _extract_failures(
            records,
            collision_threshold=float(args.collision_threshold),
            comfort_threshold=float(args.comfort_threshold),
            near_miss_threshold=float(args.near_miss_threshold),
            snqi_below=(float(args.snqi_below) if args.snqi_below is not None else None),
            max_count=(int(args.max_count) if args.max_count is not None else None),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if bool(args.ids_only):
            ids = [r.get("episode_id") for r in failures]
            with out_path.open("w", encoding="utf-8") as f:
                json.dump({"episode_ids": ids}, f, indent=2)
        else:
            # Write JSONL with full records
            with out_path.open("w", encoding="utf-8") as f:
                for rec in failures:
                    f.write(json.dumps(rec) + "\n")
        print(json.dumps({"wrote": str(out_path), "count": len(failures)}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_rank(args) -> int:
    try:
        records = _agg_read_jsonl(args.in_path)
        rows = _compute_ranking(
            records,
            group_by=args.group_by,
            fallback_group_by=args.fallback_group_by,
            metric=args.metric,
            ascending=bool(args.ascending),
            top=(int(args.top) if args.top is not None else None),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = str(args.format)
        if fmt == "md":
            content = _rank_format_md(rows, args.metric)
            out_path.write_text(content, encoding="utf-8")
        elif fmt == "csv":
            content = _rank_format_csv(rows, args.metric)
            out_path.write_text(content, encoding="utf-8")
        else:
            # JSON fallback
            payload = [r.__dict__ for r in rows]
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"wrote": str(out_path), "rows": len(rows)}, indent=2))
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_list_scenarios(args) -> int:
    try:
        scenarios = load_scenario_matrix(args.matrix)
        ids = [str(s.get("id", "unknown")) for s in scenarios]
        print("Scenario IDs:")
        for sid in ids:
            print(f"  - {sid}")
        return 0
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _handle_validate_config(args) -> int:
    try:
        scenarios = load_scenario_matrix(args.matrix)
        errors = validate_scenario_list(scenarios)
        warnings = []
        summary = {"num_scenarios": len(scenarios), "errors": errors, "warnings": warnings}
        print(json.dumps(summary, indent=2))
        return 0 if not errors else 2
    except Exception as e:  # pragma: no cover - error path
        print(f"Error: {e}", file=sys.stderr)
        return 2


def _add_baseline_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "baseline",
        help="Run batch and compute baseline med/p95 stats for SNQI",
    )
    p.add_argument(
        "--matrix",
        required=True,
        help="Path to scenario matrix YAML (or JSONL path will be ignored if provided)",
    )
    p.add_argument("--out", required=True, help="Path to write baseline stats JSON")
    p.add_argument(
        "--jsonl",
        default=None,
        help="Optional path to write intermediate episode JSONL (default results/baseline_episodes.jsonl)",
    )
    p.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Schema path for validation")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--record-forces", action="store_true", default=False)
    p.add_argument(
        "--algo",
        default="simple_policy",
        help="Algorithm to use for robot policy (simple_policy, baseline_sf, etc.)",
    )
    p.add_argument("--algo-config", help="Path to algorithm configuration YAML file")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (1=sequential)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (skip detection of already present episodes)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-episode progress output",
    )
    p.set_defaults(cmd="baseline")


def _add_run_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "run",
        help="Run a batch of episodes from a scenario matrix and write JSONL",
    )
    p.add_argument("--matrix", required=True, help="Path to scenario matrix YAML")
    p.add_argument("--out", required=True, help="Path to write episode JSONL")
    p.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Schema path for validation")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=None, help="Override repeats in matrix")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--record-forces", action="store_true", default=False)
    p.add_argument("--append", action="store_true", default=False, help="Append to existing JSONL")
    p.add_argument(
        "--algo",
        default="simple_policy",
        help="Algorithm to use for robot policy (simple_policy, baseline_sf, etc.)",
    )
    p.add_argument("--algo-config", help="Path to algorithm configuration YAML file")
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (1=sequential)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume (skip detection of already present episodes)",
    )
    p.add_argument(
        "--snqi-weights",
        type=str,
        default=None,
        help="Optional path to SNQI weights JSON to compute 'metrics.snqi' during run",
    )
    p.add_argument(
        "--snqi-baseline",
        type=str,
        default=None,
        help="Optional path to baseline stats JSON (median/p95) used for SNQI normalization",
    )
    p.add_argument(
        "--snqi-weights-from",
        type=str,
        default=None,
        help=(
            "Convenience: path to an SNQI optimize/recompute output JSON to extract the "
            "recommended weights. Ignored when --snqi-weights is provided."
        ),
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Stop on first failure instead of collecting errors",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress per-episode progress output",
    )
    p.set_defaults(cmd="run")


def _add_list_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "list-algorithms",
        help="List available baseline algorithms",
    )
    p.set_defaults(cmd="list-algorithms")

    p2 = subparsers.add_parser(
        "list-scenarios",
        help="List scenario IDs from a scenario matrix YAML",
    )
    p2.add_argument("--matrix", required=True, help="Path to scenario matrix YAML")
    p2.set_defaults(cmd="list-scenarios")

    p3 = subparsers.add_parser(
        "validate-config",
        help="Validate a scenario matrix YAML for required fields and duplicates",
    )
    p3.add_argument("--matrix", required=True, help="Path to scenario matrix YAML")
    # optional: later we could add --verbose to print detailed schema errors
    p3.set_defaults(cmd="validate-config")


def _add_summary_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "summary",
        help="Generate simple histograms (min_distance, avg_speed) from episode JSONL",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    p.add_argument("--out-dir", required=True, help="Output directory for PNGs")
    p.set_defaults(cmd="summary")


def _add_aggregate_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "aggregate",
        help=(
            "Aggregate episode metrics by group and optionally attach bootstrap CIs; "
            "writes JSON summary."
        ),
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out", required=True, help="Output JSON summary path")
    p.add_argument(
        "--group-by",
        default="scenario_params.algo",
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default="scenario_id",
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Number of bootstrap resamples for CIs (0 disables)",
    )
    p.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals (e.g., 0.95)",
    )
    p.add_argument("--bootstrap-seed", type=int, default=None)
    p.add_argument(
        "--snqi-weights",
        type=str,
        default=None,
        help="Optional SNQI weights JSON to compute metrics.snqi during aggregation",
    )
    p.add_argument(
        "--snqi-baseline",
        type=str,
        default=None,
        help="Optional baseline stats JSON used for SNQI normalization",
    )
    p.set_defaults(cmd="aggregate")


def _add_snqi_ablate_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "snqi-ablate",
        help="Compute rank shifts from one-at-a-time SNQI component removal",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out", required=True, help="Output path (.md/.csv/.json)")
    p.add_argument(
        "--group-by",
        default="scenario_params.algo",
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default="scenario_id",
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument("--format", choices=["md", "csv", "json"], default="md")
    p.add_argument("--top", type=int, default=None, help="Limit to top-N groups by base ranking")
    p.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="Optional path to write per-weight summary JSON",
    )
    # SNQI inputs
    p.add_argument("--snqi-weights", type=str, default=None, help="SNQI weights JSON path")
    p.add_argument(
        "--snqi-weights-from",
        type=str,
        default=None,
        help="Path to JSON report containing recommended weights",
    )
    p.add_argument(
        "--snqi-baseline",
        type=str,
        default=None,
        help="Baseline stats JSON (median/p95 per metric)",
    )
    p.set_defaults(cmd="snqi-ablate")


def _add_seed_variance_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "seed-variance",
        help=("Compute per-metric variability across seeds for groups; writes JSON summary"),
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out", required=True, help="Output JSON summary path")
    p.add_argument(
        "--group-by",
        default="scenario_id",
        help="Grouping key (dotted path). Default: scenario_id",
    )
    p.add_argument(
        "--fallback-group-by",
        default="scenario_id",
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated list of metric names to include (default: all)",
    )
    p.set_defaults(cmd="seed-variance")


def _add_extract_failures_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "extract-failures",
        help="Filter episodes with collisions/low comfort/near-misses/SNQI threshold",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument(
        "--out", required=True, help="Output path (JSONL by default or JSON if --ids-only)"
    )
    p.add_argument(
        "--collision-threshold",
        type=float,
        default=1.0,
        help="Flag when collisions >= threshold (default 1.0)",
    )
    p.add_argument(
        "--comfort-threshold",
        type=float,
        default=0.2,
        help="Flag when comfort_exposure >= threshold (default 0.2)",
    )
    p.add_argument(
        "--near-miss-threshold",
        type=float,
        default=0.0,
        help="Flag when near_misses > threshold (strictly greater-than; default 0.0)",
    )
    p.add_argument("--snqi-below", type=float, default=None)
    p.add_argument("--max-count", type=int, default=None)
    p.add_argument(
        "--ids-only",
        action="store_true",
        default=False,
        help="Write JSON with episode_ids instead of JSONL records",
    )
    p.set_defaults(cmd="extract-failures")


def _add_rank_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    p = subparsers.add_parser(
        "rank",
        help=("Compute rankings by mean of a metric per group and write as Markdown/CSV/JSON"),
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out", required=True, help="Output path (md/csv/json)")
    p.add_argument(
        "--group-by",
        default="scenario_params.algo",
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default="scenario_id",
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument("--metric", default="collisions", help="Metric name under metrics.<name>")
    sort = p.add_mutually_exclusive_group()
    sort.add_argument("--ascending", action="store_true", default=True)
    sort.add_argument("--descending", dest="ascending", action="store_false")
    p.add_argument("--top", type=int, default=None, help="Limit to top N rows")
    p.add_argument(
        "--format",
        choices=["md", "csv", "json"],
        default="md",
        help="Output format (Markdown table, CSV, or JSON)",
    )
    p.set_defaults(cmd="rank")


def _base_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="robot_sf_bench", description="Social Navigation Benchmark CLI"
    )


def _attach_core_subcommands(parser: argparse.ArgumentParser) -> None:  # noqa: C901
    subparsers = parser.add_subparsers(dest="cmd")
    _add_baseline_subparser(subparsers)
    _add_run_subparser(subparsers)
    _add_summary_subparser(subparsers)
    _add_aggregate_subparser(subparsers)
    _add_seed_variance_subparser(subparsers)
    _add_extract_failures_subparser(subparsers)
    _add_snqi_ablate_subparser(subparsers)
    _add_rank_subparser(subparsers)
    _add_list_subparser(subparsers)
    snqi_parser = subparsers.add_parser(
        "snqi",
        help="SNQI weight tooling (optimize / recompute)",
        description="Social Navigation Quality Index tooling: optimize or recompute weights.",
    )
    snqi_sub = snqi_parser.add_subparsers(dest="snqi_cmd")

    # We replicate the script arguments (kept minimal & aligned with parse_args in scripts) to avoid code duplication.
    # Dynamic loading is used so we don't need to refactor the existing scripts immediately.

    def _add_snqi_optimize(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        p = sp.add_parser("optimize", help="Optimize SNQI weights (grid / evolution)")
        p.add_argument("--episodes", type=Path, required=True, help="Episodes JSONL file")
        p.add_argument("--baseline", type=Path, required=True, help="Baseline stats JSON file")
        p.add_argument("--output", type=Path, required=True, help="Output JSON file")
        p.add_argument(
            "--method",
            choices=["grid", "evolution", "both"],
            default="both",
            help="Optimization method",
        )
        p.add_argument("--grid-resolution", type=int, default=5)
        p.add_argument(
            "--maxiter", type=int, default=30, help="Differential evolution max iterations"
        )
        p.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--validate", action="store_true", help="Validate output schema")
        p.add_argument(
            "--max-grid-combinations",
            type=int,
            default=20000,
            help="Guard threshold for total grid combinations",
        )
        p.add_argument(
            "--initial-weights-file",
            type=Path,
            default=None,
            help="JSON file containing initial weight mapping",
        )
        p.add_argument("--progress", action="store_true", help="Show progress bars (tqdm)")
        p.add_argument(
            "--missing-metric-max-list",
            type=int,
            default=5,
            help="Max example episode IDs per missing baseline metric",
        )
        p.add_argument("--fail-on-missing-metric", action="store_true")
        p.add_argument(
            "--sample", type=int, default=None, help="Deterministically sample N episodes"
        )
        p.add_argument("--simplex", action="store_true", help="Project weights onto simplex")
        p.add_argument(
            "--early-stop-patience",
            type=int,
            default=0,
            help="Early stopping patience (0 disables)",
        )
        p.add_argument(
            "--early-stop-min-delta",
            type=float,
            default=1e-4,
            help="Minimum improvement to reset early stopping",
        )
        p.add_argument(
            "--ci-placeholder", action="store_true", help="Include CI placeholder scaffold"
        )
        p.add_argument(
            "--bootstrap-samples",
            type=int,
            default=0,
            help="Number of bootstrap resamples for stability/CI estimation (0 disables)",
        )
        p.add_argument(
            "--bootstrap-confidence",
            type=float,
            default=0.95,
            help="Confidence level for bootstrap intervals (e.g., 0.95)",
        )
        p.add_argument(
            "--small-dataset-threshold",
            type=int,
            default=20,
            help=(
                "Warn when the number of episodes used is below this threshold "
                "(stability and CIs may be unreliable)."
            ),
        )
        p.set_defaults(cmd="snqi", snqi_cmd="optimize")

    def _add_snqi_recompute(sp: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
        p = sp.add_parser("recompute", help="Recompute SNQI weights via predefined strategies")
        p.add_argument("--episodes", type=Path, required=True, help="Episodes JSONL file")
        p.add_argument("--baseline", type=Path, required=True, help="Baseline stats JSON file")
        p.add_argument(
            "--strategy",
            choices=["default", "balanced", "safety_focused", "efficiency_focused", "pareto"],
            default="default",
        )
        p.add_argument("--output", type=Path, required=True, help="Output JSON file")
        p.add_argument("--compare-normalization", action="store_true")
        p.add_argument("--compare-strategies", action="store_true")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--validate", action="store_true")
        p.add_argument(
            "--external-weights-file",
            type=Path,
            default=None,
            help="Evaluate external weights JSON mapping",
        )
        p.add_argument(
            "--missing-metric-max-list",
            type=int,
            default=5,
            help="Max example episode IDs per missing baseline metric",
        )
        p.add_argument("--fail-on-missing-metric", action="store_true")
        p.add_argument(
            "--sample", type=int, default=None, help="Deterministically sample N episodes"
        )
        p.add_argument("--simplex", action="store_true", help="Project weights onto simplex")
        p.add_argument(
            "--bootstrap-samples",
            type=int,
            default=0,
            help="Number of bootstrap resamples for stability/CI estimation (0 disables)",
        )
        p.add_argument(
            "--bootstrap-confidence",
            type=float,
            default=0.95,
            help="Confidence level for bootstrap intervals (e.g., 0.95)",
        )
        p.add_argument(
            "--small-dataset-threshold",
            type=int,
            default=20,
            help=(
                "Warn when the number of episodes used is below this threshold "
                "(stability and CIs may be unreliable)."
            ),
        )
        p.set_defaults(cmd="snqi", snqi_cmd="recompute")

    _add_snqi_optimize(snqi_sub)
    _add_snqi_recompute(snqi_sub)

    # ---- dynamic script module loaders ----
    _OPT_MOD = None  # cache
    _RECOMP_MOD = None

    def _load_script(rel: str, name: str):  # noqa: D401 - simple dynamic loader
        """Dynamically load a script module by relative path.

        Uses importlib spec APIs so module metadata (__spec__, __file__) is set.
        """
        from importlib.util import module_from_spec, spec_from_file_location  # local import

        # Defensive: if a prior failed import left a None placeholder, remove it
        existing = sys.modules.get(name)
        if existing is None:
            sys.modules.pop(name, None)

        path = Path(__file__).resolve().parents[2] / rel
        if not path.exists():  # pragma: no cover - defensive
            raise FileNotFoundError(f"SNQI script not found: {path}")
        spec = spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"Unable to load spec for {path}")
        mod = module_from_spec(spec)
        sys.modules[name] = mod  # allow relative imports inside script if any
        spec.loader.exec_module(mod)
        return mod

    def _get_opt_run(mod):  # type: ignore[no-untyped-def]
        return getattr(mod, "run")

    def _get_recompute_run(mod):  # type: ignore[no-untyped-def]
        return getattr(mod, "run")

    def _ensure_snqi_opt_defaults(args: argparse.Namespace) -> None:
        """Ensure optional flags expected by the optimize script exist on args.

        This guards against AttributeError when constructing Namespace instances
        programmatically (e.g., in tests) without all optional flags present.
        """
        defaults = {
            "progress": False,
            "simplex": False,
            "max_grid_combinations": 20000,
            "grid_resolution": 5,
            "maxiter": 30,
            "sensitivity": False,
            "bootstrap_samples": 0,
            "bootstrap_confidence": 0.95,
            "small_dataset_threshold": 20,
        }
        for name, value in defaults.items():
            if not hasattr(args, name):
                setattr(args, name, value)

    def _load_optimize_run_fn():  # type: ignore[no-untyped-def]
        """Load and return the optimize script's run(args) function or None on error."""
        nonlocal _OPT_MOD
        if _OPT_MOD is None:
            try:
                _OPT_MOD = _load_script(
                    "scripts/snqi_weight_optimization.py", "snqi_optimize_script"
                )
            except Exception as e:  # pragma: no cover - load error
                print(f"Error loading optimization script: {e}", file=sys.stderr)
                traceback.print_exc()
                return None
        return _get_opt_run(_OPT_MOD)

    def _invoke_snqi_opt(args: argparse.Namespace) -> int:  # noqa: C901 - thin wrapper delegating to external script
        _ensure_snqi_opt_defaults(args)
        if os.environ.get("ROBOT_SF_SNQI_LIGHT_TEST") == "1":  # pragma: no cover - test helper
            return 0
        run_fn = _load_optimize_run_fn()
        if run_fn is None:
            return 2
        return int(run_fn(args))  # type: ignore[no-any-return]

    def _invoke_snqi_recompute(args: argparse.Namespace) -> int:
        if os.environ.get("ROBOT_SF_SNQI_LIGHT_TEST") == "1":  # pragma: no cover - test helper
            return 0
        nonlocal _RECOMP_MOD
        if _RECOMP_MOD is None:
            try:
                _RECOMP_MOD = _load_script(
                    "scripts/recompute_snqi_weights.py", "snqi_recompute_script"
                )
            except Exception as e:  # pragma: no cover - load error
                print(f"Error loading recompute script: {e}", file=sys.stderr)
                traceback.print_exc()
                return 2
        run_fn = _get_recompute_run(_RECOMP_MOD)
        return int(run_fn(args))  # type: ignore[no-any-return]

    # Public attribute (not underscored) to avoid protected-member lint warning.
    parser.snqi_loader = {  # type: ignore[attr-defined]  # noqa: ANN001 - dynamic injection
        "invoke_optimize": _invoke_snqi_opt,
        "invoke_recompute": _invoke_snqi_recompute,
    }


def _configure_parser() -> argparse.ArgumentParser:
    parser = _base_parser()
    _attach_core_subcommands(parser)
    return parser


def cli_main(argv: List[str] | None = None) -> int:
    parser = _configure_parser()
    args = parser.parse_args(argv)
    # macOS safe start method for multiprocessing
    if getattr(args, "workers", 1) and int(getattr(args, "workers", 1)) > 1:
        try:
            import multiprocessing as _mp

            _mp.set_start_method("spawn", force=False)
        except Exception:
            pass

    # Access dynamic loaders if present
    snqi_loader = getattr(parser, "snqi_loader", {})  # type: ignore[no-any-explicit]
    # Dispatch
    if args.cmd == "snqi":
        if args.snqi_cmd == "optimize":
            return snqi_loader["invoke_optimize"](args)
        if args.snqi_cmd == "recompute":
            return snqi_loader["invoke_recompute"](args)
        print("Specify a snqi subcommand (optimize|recompute)", file=sys.stderr)
        return 2
    handlers = {
        "baseline": _handle_baseline,
        "list-algorithms": _handle_list_algorithms,
        "list-scenarios": _handle_list_scenarios,
        "validate-config": _handle_validate_config,
        "run": _handle_run,
        "summary": _handle_summary,
        "aggregate": _handle_aggregate,
        "seed-variance": _handle_seed_variance,
        "extract-failures": _handle_extract_failures,
        "snqi-ablate": _handle_snqi_ablate,
        "rank": _handle_rank,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


def main() -> None:  # pragma: no cover - thin wrapper
    raise SystemExit(cli_main())


__all__ = ["cli_main", "main"]
