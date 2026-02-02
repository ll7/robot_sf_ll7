"""Benchmark CLI providing unified entrypoints (including SNQI tooling)."""
# ruff: noqa: C901  # complexity acceptable for nested CLI builders/wrappers

# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
import traceback
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.ablation import compute_ablation_summary as _abl_summary
from robot_sf.benchmark.ablation import compute_snqi_ablation as _abl_compute
from robot_sf.benchmark.ablation import format_csv as _abl_format_csv
from robot_sf.benchmark.ablation import format_markdown as _abl_format_md
from robot_sf.benchmark.ablation import to_json as _abl_to_json
from robot_sf.benchmark.aggregate import compute_aggregates as _agg_compute
from robot_sf.benchmark.aggregate import compute_aggregates_with_ci as _agg_compute_ci
from robot_sf.benchmark.aggregate import read_jsonl as _agg_read_jsonl
from robot_sf.benchmark.baseline_stats import run_and_compute_baseline
from robot_sf.benchmark.distributions import collect_grouped_values as _dist_collect
from robot_sf.benchmark.distributions import save_distributions as _dist_save
from robot_sf.benchmark.failure_extractor import extract_failures as _extract_failures
from robot_sf.benchmark.plots import save_pareto_png as _save_pareto_png
from robot_sf.benchmark.ranking import compute_ranking as _compute_ranking
from robot_sf.benchmark.ranking import format_csv as _rank_format_csv
from robot_sf.benchmark.ranking import format_markdown as _rank_format_md
from robot_sf.benchmark.report_table import compute_table as _tbl_compute
from robot_sf.benchmark.report_table import format_csv as _tbl_format_csv
from robot_sf.benchmark.report_table import format_latex_booktabs as _tbl_format_tex
from robot_sf.benchmark.report_table import format_markdown as _tbl_format_md
from robot_sf.benchmark.report_table import to_json as _tbl_to_json
from robot_sf.benchmark.runner import load_scenario_matrix, run_batch
from robot_sf.benchmark.scenario_schema import validate_scenario_list
from robot_sf.benchmark.scenario_thumbnails import save_montage as _thumb_montage
from robot_sf.benchmark.scenario_thumbnails import (
    save_scenario_thumbnails as _thumb_save_all,
)
from robot_sf.benchmark.seed_variance import (
    compute_seed_variance as _compute_seed_variance,
)
from robot_sf.benchmark.summary import summarize_to_plots
from robot_sf.common.seed import get_seed_state_sample as _seed_sample
from robot_sf.common.seed import set_global_seed as _set_seed

DEFAULT_SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _handle_baseline(args) -> int:
    """Execute baseline command to compute baseline statistics.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 2 for error).
    """
    try:
        progress_cb = _progress_cb_factory(bool(args.quiet))
        run_and_compute_baseline(
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
        # Inform the user where outputs were written (INFO-level; suppressed with --quiet)
        try:
            logging.info("Baseline stats written to %s", args.out)
            if getattr(args, "jsonl", None):
                logging.info("Intermediate episodes JSONL written to %s", args.jsonl)
        except Exception:
            # Defensive: do not let logging interfere with success path
            pass
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_list_algorithms(_args) -> int:
    """List available baseline algorithms.

    Args:
        _args: Parsed command-line arguments (unused).

    Returns:
        Exit code (0 for success).
    """
    try:
        # Only advertise algorithms that can be resolved by the baselines registry.
        algorithms = ["simple_policy"]

        try:
            baseline_module = importlib.import_module("robot_sf.baselines")
            baseline_algos = baseline_module.list_baselines()
            algorithms.extend(baseline_algos)
        except ImportError:
            pass

        seen: set[str] = set()

        for algo in algorithms:
            if algo in seen:
                continue
            seen.add(algo)
            print(algo)

        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _load_snqi_inputs(args):
    """Load SNQI weights and baseline from command-line arguments.

    Args:
        args: Parsed command-line arguments with snqi_weights and snqi_baseline attributes.

    Returns:
        Tuple of (snqi_weights dict or None, snqi_baseline dict or None).
    """
    snqi_weights = None
    snqi_baseline = None
    # Priority: explicit weights JSON
    if getattr(args, "snqi_weights", None):
        with open(args.snqi_weights, encoding="utf-8") as f:
            snqi_weights = json.load(f)
    elif getattr(args, "snqi_weights_from", None):
        with open(args.snqi_weights_from, encoding="utf-8") as f:
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
        with open(args.snqi_baseline, encoding="utf-8") as f:
            snqi_baseline = json.load(f)
    return snqi_weights, snqi_baseline


def _progress_cb_factory(quiet: bool):
    """Create progress callback function with optional tqdm progress bar.

    Args:
        quiet: If True, suppress progress output.

    Returns:
        Progress callback function for tracking episode execution.
    """
    pbar = None
    if not quiet:
        try:  # pragma: no cover - tqdm optional
            tqdm_module = importlib.import_module("tqdm")
            tqdm = tqdm_module.tqdm
            pbar = tqdm(total=0, unit="ep", disable=False)
        except Exception:
            logging.debug("tqdm import failed or pbar unavailable", exc_info=True)
            pbar = None

    def _cb(i, total, sc, seed, ok, err):
        """Update progress for a single episode result."""
        if quiet:
            return
        status = "ok" if ok else "FAIL"
        sid = sc.get("id") or sc.get("name") or sc.get("scenario_id") or "unknown"
        if pbar is not None:
            try:  # pragma: no cover - tqdm optional
                if pbar.total != total:
                    pbar.reset(total=total)
                pbar.update(1)
                pbar.set_description(f"{sid} seed={seed} {status}")
            except Exception:
                logging.debug("Progress bar set_description/update failed", exc_info=True)
        else:
            msg = f"[{i}/{total}] {sid} seed={seed}: {status}"
            if err:
                msg += f" ({err})"

    return _cb


def _handle_run(args) -> int:
    """Execute run command to generate episode records.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, 2 for error).
    """
    try:
        # Optional: load SNQI weights/baseline for inline SNQI computation
        try:
            snqi_weights, snqi_baseline = _load_snqi_inputs(args)
        except Exception:  # pragma: no cover - error path
            return 2

        run_batch(
            scenarios_or_path=args.matrix,
            out_path=args.out,
            schema_path=args.schema,
            base_seed=args.base_seed,
            repeats_override=args.repeats,
            horizon=args.horizon,
            dt=args.dt,
            record_forces=args.record_forces,
            video_enabled=not args.no_video and args.video_renderer != "none",
            video_renderer=args.video_renderer if not args.no_video else "none",
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
        try:
            logging.info("Episodes written to %s", args.out)
        except Exception:
            logging.debug("Logging 'Episodes written' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_summary(args) -> int:
    """Generate summary plots from episodes.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        summarize_to_plots(args.in_path, args.out_dir)
        try:
            logging.info("Summary plots written to %s", args.out_dir)
        except Exception:
            logging.debug("Logging 'Summary plots written' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_aggregate(args) -> int:
    """Aggregate episode metrics and write a JSON summary.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        # Optional: load SNQI inputs to compute metrics.snqi during aggregation
        snqi_weights = None
        snqi_baseline = None
        if getattr(args, "snqi_weights", None):
            with open(args.snqi_weights, encoding="utf-8") as f:
                snqi_weights = json.load(f)
        if getattr(args, "snqi_baseline", None):
            with open(args.snqi_baseline, encoding="utf-8") as f:
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
        try:
            logging.info("Aggregated summary written to %s", out_path)
        except Exception:
            logging.debug("Logging 'Aggregated summary' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_snqi_ablate(args) -> int:
    """Compute SNQI ablation tables for episode records.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        records = _agg_read_jsonl(args.in_path)
        snqi_weights, snqi_baseline = _load_snqi_inputs(args)
        if not isinstance(snqi_weights, dict) or not isinstance(snqi_baseline, dict):
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
            return 2
        # Optional summary JSON for ablation (per-weight impact stats)
        if getattr(args, "summary_out", None):
            summary_out = Path(args.summary_out)
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_payload = _abl_summary(rows)
            summary_out.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
            str(summary_out)
        try:
            logging.info("Ablation results written to %s", out_path)
            if getattr(args, "summary_out", None):
                logging.info("Ablation summary written to %s", summary_out)
        except Exception:
            logging.debug("Logging 'Ablation results' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - defensive
        return 2


def _handle_seed_variance(args) -> int:
    """Compute per-metric seed variance and write a JSON summary.

    Returns:
        Exit code (0 success, 2 failure).
    """
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
        try:
            logging.info("Seed-variance summary written to %s", out_path)
        except Exception:
            logging.debug("Logging 'Seed-variance summary' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_extract_failures(args) -> int:
    """Extract failure episodes and write them to disk.

    Returns:
        Exit code (0 success, 2 failure).
    """
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
        try:
            logging.info("Wrote failures to %s", out_path)
        except Exception:
            logging.debug("Logging 'Wrote failures' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_rank(args) -> int:
    """Compute rankings by metric and write a table.

    Returns:
        Exit code (0 success, 2 failure).
    """
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
        try:
            logging.info("Ranking output written to %s", out_path)
        except Exception:
            logging.debug("Logging 'Ranking output' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_table(args) -> int:
    """Compute baseline comparison tables and write output.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        records = _agg_read_jsonl(args.in_path)
        metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
        rows = _tbl_compute(
            records,
            metrics=metrics,
            group_by=args.group_by,
            fallback_group_by=args.fallback_group_by,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = str(args.format)
        if fmt == "md":
            content = _tbl_format_md(rows, metrics)
            out_path.write_text(content, encoding="utf-8")
        elif fmt == "csv":
            content = _tbl_format_csv(rows, metrics)
            out_path.write_text(content, encoding="utf-8")
        elif fmt == "tex":
            content = _tbl_format_tex(rows, metrics)
            out_path.write_text(content, encoding="utf-8")
        else:
            payload = _tbl_to_json(rows)
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            logging.info("Table output written to %s", out_path)
        except Exception:
            logging.debug("Logging 'Table output' failed", exc_info=True)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_debug_seeds(args) -> int:
    """Apply global seeds and print a deterministic sample.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        report = _set_seed(int(args.seed), deterministic=bool(args.deterministic))
        sample = _seed_sample(n=5)
        payload = {"report": report.to_dict(), "sample": sample}
        if getattr(args, "out", None):
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        else:
            # For interactive use, print the debug payload to stdout (machine-friendly)
            print(json.dumps(payload))
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_plot_pareto(args) -> int:
    """Render a Pareto plot and save it to disk.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        records = _agg_read_jsonl(args.in_path)
        _save_pareto_png(
            records,
            out_path=str(args.out),
            x_metric=str(args.x_metric),
            y_metric=str(args.y_metric),
            group_by=str(args.group_by),
            fallback_group_by=str(args.fallback_group_by),
            agg=str(args.agg),
            x_higher_better=bool(args.x_higher_better),
            y_higher_better=bool(args.y_higher_better),
            title=str(args.title) if args.title is not None else None,
            out_pdf=(str(args.out_pdf) if getattr(args, "out_pdf", None) else None),
        )
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_plot_distributions(args) -> int:
    """Render per-metric distribution plots.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        records = _agg_read_jsonl(args.in_path)
        metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
        grouped = _dist_collect(
            records,
            metrics=metrics,
            group_by=str(args.group_by),
            fallback_group_by=str(args.fallback_group_by),
        )
        _dist_save(
            grouped,
            out_dir=str(args.out_dir),
            bins=int(args.bins),
            kde=bool(args.kde),
            out_pdf=bool(args.out_pdf),
            ci=bool(getattr(args, "ci", False)),
            ci_samples=int(getattr(args, "ci_samples", 1000)),
            ci_confidence=float(getattr(args, "ci_confidence", 0.95)),
            ci_seed=(int(args.ci_seed) if getattr(args, "ci_seed", None) is not None else None),
        )
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_plot_scenarios(args) -> int:
    """Render scenario thumbnails and optional montage.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        scenarios = load_scenario_matrix(args.matrix)
        # Ignore repeats for thumbnails by deduping by id order-preserving
        seen = set()
        unique_scenarios = []
        for sc in scenarios:
            sid = str(sc.get("id") or sc.get("name") or sc.get("scenario_id") or "scenario")
            if sid not in seen:
                seen.add(sid)
                unique_scenarios.append(sc)
        metas = _thumb_save_all(
            unique_scenarios,
            out_dir=str(args.out_dir),
            base_seed=int(args.base_seed),
            out_pdf=bool(args.pdf),
            figsize=(float(args.fig_w), float(args.fig_h)),
        )
        wrote = [m.png for m in metas]
        payload = {"wrote": wrote}
        if bool(args.montage):
            out_png = str(Path(args.out_dir) / "montage.png")
            out_pdf = str(Path(args.out_dir) / "montage.pdf") if bool(args.pdf) else None
            meta = _thumb_montage(metas, out_png=out_png, cols=int(args.cols), out_pdf=out_pdf)
            payload.update({"montage": meta})
        print(json.dumps(payload))
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _handle_list_scenarios(args) -> int:
    """List scenarios from a matrix.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        scenarios = load_scenario_matrix(args.matrix)
        ids = [
            str(s.get("id") or s.get("name") or s.get("scenario_id") or "unknown")
            for s in scenarios
        ]
        for sid in ids:
            print(sid)
        return 0
    except Exception:  # pragma: no cover - error path
        return 2


def _extract_matrix_source(matrix_path: str | Path) -> dict[str, object]:
    """Describe the scenario matrix source and include structure.

    Returns:
        dict[str, object]: Source metadata including format and include list.
    """
    path = Path(matrix_path)
    includes: list[str] = []
    format_hint = "unknown"
    try:
        with path.open("r", encoding="utf-8") as handle:
            docs = list(yaml.safe_load_all(handle))
    except Exception:
        return {"path": str(path), "format": format_hint, "includes": includes}
    if len(docs) > 1:
        format_hint = "stream"
        return {"path": str(path), "format": format_hint, "includes": includes}
    if not docs:
        return {"path": str(path), "format": format_hint, "includes": includes}
    doc = docs[0]
    if isinstance(doc, Mapping):
        raw_includes = doc.get("includes") or doc.get("include") or doc.get("scenario_files")
        if isinstance(raw_includes, list):
            includes = [str(entry) for entry in raw_includes]
        elif isinstance(raw_includes, (str, Path)):
            includes = [str(raw_includes)]
        format_hint = "manifest" if includes else "dict"
        if "scenarios" in doc:
            format_hint = "manifest" if includes else "list"
    elif isinstance(doc, list):
        format_hint = "list"
    return {"path": str(path), "format": format_hint, "includes": includes}


def _summarize_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, object]:
    """Summarize scenario counts and basic field coverage.

    Returns:
        dict[str, object]: Summary counts for archetypes/densities/maps and missing fields.
    """

    def _pick_meta_value(scenario: dict[str, Any], key: str) -> str:
        meta = scenario.get("metadata")
        if isinstance(meta, Mapping):
            value = meta.get(key)
            if isinstance(value, str) and value:
                return value
        fallback = scenario.get(key)
        if isinstance(fallback, str) and fallback:
            return fallback
        return "unknown"

    archetypes = Counter(_pick_meta_value(sc, "archetype") for sc in scenarios)
    densities = Counter(_pick_meta_value(sc, "density") for sc in scenarios)
    maps = Counter(str(sc.get("map_file")) for sc in scenarios if sc.get("map_file"))
    missing_map_file = sum(1 for sc in scenarios if not isinstance(sc.get("map_file"), str))
    missing_metadata = sum(1 for sc in scenarios if not isinstance(sc.get("metadata"), Mapping))
    missing_sim = sum(1 for sc in scenarios if not isinstance(sc.get("simulation_config"), Mapping))
    return {
        "archetypes": dict(archetypes),
        "densities": dict(densities),
        "maps": dict(maps),
        "missing": {
            "map_file": missing_map_file,
            "metadata": missing_metadata,
            "simulation_config": missing_sim,
        },
    }


def _handle_validate_config(args) -> int:
    """Validate scenario matrix config against schema rules.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        scenarios = load_scenario_matrix(args.matrix)
        errors = validate_scenario_list(scenarios)
        warnings = []
        summary = _summarize_scenarios(scenarios)
        source = _extract_matrix_source(args.matrix)
        report = {
            "num_scenarios": len(scenarios),
            "errors": errors,
            "warnings": warnings,
            "summary": summary,
            "source": source,
        }
        print(json.dumps(report))
        return 0 if not errors else 2
    except Exception:  # pragma: no cover - error path
        return 2


def _add_baseline_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the baseline subcommand parser."""
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
        help="Optional path to write intermediate episode JSONL (default output/results/baseline_episodes.jsonl)",
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
    # Global --quiet handled at top-level parser
    p.set_defaults(cmd="baseline")


def _add_run_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the run subcommand parser."""
    p = subparsers.add_parser(
        "run",
        help="Run a batch of episodes from a scenario matrix and write JSONL with real plots/videos",
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
    # Video artifact controls (Episode Video Artifacts MVP)
    p.add_argument(
        "--no-video",
        action="store_true",
        default=False,
        help="Disable per-episode video generation (overrides renderer selection)",
    )
    p.add_argument(
        "--video-renderer",
        type=str,
        choices=["synthetic", "sim-view", "none"],
        default="none",
        help=(
            "Frame source for per-episode MP4 videos. 'synthetic' uses a lightweight renderer;\n"
            "'sim-view' (experimental) uses SimulationView when available; 'none' disables.\n"
            "Generates real simulation replays, not placeholders."
        ),
    )
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
    # Global --quiet handled at top-level parser
    p.set_defaults(cmd="run")


def _add_list_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register list-algorithms/list-scenarios/validate-config parsers."""
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
    """Register the summary subcommand parser."""
    p = subparsers.add_parser(
        "summary",
        help="Generate real statistical plots (PDF histograms, distributions) from episode JSONL",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input JSONL path")
    p.add_argument("--out-dir", required=True, help="Output directory for PNGs")
    p.set_defaults(cmd="summary")


def _add_aggregate_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the aggregate subcommand parser."""
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
    """Register the SNQI ablation subcommand parser."""
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
    """Register the seed-variance subcommand parser."""
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
    """Register the extract-failures subcommand parser."""
    p = subparsers.add_parser(
        "extract-failures",
        help="Filter episodes with collisions/low comfort/near-misses/SNQI threshold",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument(
        "--out",
        required=True,
        help="Output path (JSONL by default or JSON if --ids-only)",
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
    """Register the rank subcommand parser."""
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
        choices=["md", "csv", "tex", "json"],
        default="md",
        help="Output format (Markdown, CSV, LaTeX booktabs, or JSON)",
    )
    p.set_defaults(cmd="rank")


def _add_table_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the table subcommand parser."""
    p = subparsers.add_parser(
        "table",
        help=(
            "Generate a baseline comparison table by per-group means for selected metrics "
            "(Markdown/CSV/JSON)"
        ),
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
    p.add_argument(
        "--metrics",
        required=True,
        help="Comma-separated list of metric names under metrics.<name>",
    )
    p.add_argument(
        "--format",
        choices=["md", "csv", "tex", "json"],
        default="md",
        help="Output format (Markdown table, CSV, LaTeX booktabs, or JSON)",
    )
    p.set_defaults(cmd="table")


def _add_debug_seeds_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the debug-seeds subcommand parser."""
    p = subparsers.add_parser(
        "debug-seeds",
        help="Set global seeds (random, numpy, torch) and print a small state sample.",
    )
    p.add_argument("--seed", type=int, default=42, help="Seed to apply (default: 42)")
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Enable deterministic torch settings when available (default: True)",
    )
    p.add_argument("--out", type=str, default=None, help="Optional JSON summary path")
    p.set_defaults(cmd="debug-seeds")


def _add_plot_pareto_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the plot-pareto subcommand parser."""
    p = subparsers.add_parser(
        "plot-pareto",
        help="Plot a Pareto front for two metrics grouped by algo (PNG)",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--x-metric", required=True, help="Metric name for X axis")
    p.add_argument("--y-metric", required=True, help="Metric name for Y axis")
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
    p.add_argument("--agg", choices=["mean", "median"], default="mean")
    p.add_argument("--x-higher-better", action="store_true", default=False)
    p.add_argument("--y-higher-better", action="store_true", default=False)
    p.add_argument("--title", default=None)
    p.add_argument("--out-pdf", default=None, help="Optional path to also export a vector PDF")
    p.set_defaults(cmd="plot-pareto")


def _add_plot_distributions_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the plot-distributions subcommand parser."""
    p = subparsers.add_parser(
        "plot-distributions",
        help="Plot per-metric distributions (histograms, optional KDE) per group",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out-dir", required=True, help="Output directory for plots")
    p.add_argument(
        "--metrics",
        required=True,
        help="Comma-separated metric names under metrics.<name>",
    )
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
    p.add_argument("--bins", type=int, default=30)
    p.add_argument("--kde", action="store_true", default=False, help="Overlay KDE when available")
    p.add_argument("--ci", action="store_true", default=False, help="Overlay bootstrap CI bands")
    p.add_argument("--ci-samples", type=int, default=1000, help="Number of bootstrap samples")
    p.add_argument(
        "--ci-confidence",
        type=float,
        default=0.95,
        help="Confidence level for CI (e.g., 0.95)",
    )
    p.add_argument("--ci-seed", type=int, default=123, help="Seed for bootstrap resampling")
    p.add_argument(
        "--out-pdf",
        action="store_true",
        default=False,
        help="Also export LaTeX-friendly vector PDFs",
    )
    p.set_defaults(cmd="plot-distributions")


def _add_plot_scenarios_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the plot-scenarios subcommand parser."""
    p = subparsers.add_parser(
        "plot-scenarios",
        help="Render per-scenario thumbnails (PNG/PDF) and optional montage",
    )
    p.add_argument("--matrix", required=True, help="Path to scenario matrix YAML")
    p.add_argument("--out-dir", required=True, help="Output directory for thumbnails")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--pdf", action="store_true", default=False, help="Also export PDFs")
    p.add_argument("--montage", action="store_true", default=False, help="Write montage image")
    p.add_argument("--cols", type=int, default=3, help="Montage columns (default: 3)")
    p.add_argument("--fig-w", type=float, default=3.2, help="Single thumbnail width in inches")
    p.add_argument("--fig-h", type=float, default=2.0, help="Single thumbnail height in inches")
    p.set_defaults(cmd="plot-scenarios")


def _base_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="robot_sf_bench",
        description="Social Navigation Benchmark CLI",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress non-essential output (sets WARNING unless CRITICAL explicitly chosen)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("ROBOT_SF_LOG_LEVEL", "INFO"),
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default INFO or $ROBOT_SF_LOG_LEVEL)",
    )
    return parser


def _configure_logging(quiet: bool, level: str) -> None:
    """Configure root logger once.

    If quiet, downgrade to WARNING unless CRITICAL requested.
    """
    desired_level = getattr(logging, level.upper(), logging.INFO)
    if quiet and desired_level < logging.CRITICAL:
        desired_level = logging.WARNING
    root = logging.getLogger()
    # If already configured, just adjust level
    if root.handlers:
        root.setLevel(desired_level)
        return
    logging.basicConfig(
        level=desired_level,
        format="%(levelname)s:%(name)s:%(message)s",
    )


def configure_logging(quiet: bool, level: str) -> None:
    """Public wrapper for logging configuration (used in tests)."""
    _configure_logging(quiet, level)


def _attach_core_subcommands(parser: argparse.ArgumentParser) -> None:
    """Attach core benchmark CLI subcommands."""
    subparsers = parser.add_subparsers(dest="cmd")
    _add_baseline_subparser(subparsers)
    _add_run_subparser(subparsers)
    _add_summary_subparser(subparsers)
    _add_aggregate_subparser(subparsers)
    _add_seed_variance_subparser(subparsers)
    _add_extract_failures_subparser(subparsers)
    _add_snqi_ablate_subparser(subparsers)
    _add_rank_subparser(subparsers)
    _add_table_subparser(subparsers)
    _add_debug_seeds_subparser(subparsers)
    _add_plot_pareto_subparser(subparsers)
    _add_plot_distributions_subparser(subparsers)
    _add_plot_scenarios_subparser(subparsers)
    _add_list_subparser(subparsers)
    snqi_parser = subparsers.add_parser(
        "snqi",
        help="SNQI weight tooling (optimize / recompute)",
        description="Social Navigation Quality Index tooling: optimize or recompute weights.",
    )
    snqi_sub = snqi_parser.add_subparsers(dest="snqi_cmd", required=True)

    # We replicate the script arguments (kept minimal & aligned with parse_args in scripts) to avoid code duplication.
    # Dynamic loading is used so we don't need to refactor the existing scripts immediately.

    def _add_snqi_optimize(
        sp: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """Register the SNQI optimize subcommand parser."""
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
            "--maxiter",
            type=int,
            default=30,
            help="Differential evolution max iterations",
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
            "--sample",
            type=int,
            default=None,
            help="Deterministically sample N episodes",
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
            "--ci-placeholder",
            action="store_true",
            help="Include CI placeholder scaffold",
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

    def _add_snqi_recompute(
        sp: argparse._SubParsersAction[argparse.ArgumentParser],
    ) -> None:
        """Register the SNQI recompute subcommand parser."""
        p = sp.add_parser("recompute", help="Recompute SNQI weights via predefined strategies")
        p.add_argument("--episodes", type=Path, required=True, help="Episodes JSONL file")
        p.add_argument("--baseline", type=Path, required=True, help="Baseline stats JSON file")
        p.add_argument(
            "--strategy",
            choices=[
                "default",
                "balanced",
                "safety_focused",
                "efficiency_focused",
                "pareto",
            ],
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
            "--sample",
            type=int,
            default=None,
            help="Deterministically sample N episodes",
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

    def _load_script(rel: str, name: str):
        """Dynamically load a script module by relative path.

        Uses importlib spec APIs so module metadata (__spec__, __file__) is set.

        Returns:
            The loaded module object.
        """
        # Defensive: if a prior failed import left a None placeholder, remove it
        existing = sys.modules.get(name)
        if existing is None:
            sys.modules.pop(name, None)

        path = Path(__file__).resolve().parents[2] / rel
        if not path.exists():  # pragma: no cover - defensive
            raise FileNotFoundError(f"SNQI script not found: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"Unable to load spec for {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # allow relative imports inside script if any
        spec.loader.exec_module(mod)
        return mod

    def _get_opt_run(mod):  # type: ignore[no-untyped-def]
        """Return the optimize script run() function.

        Returns:
            Callable run function from the optimize module.
        """
        return mod.run

    def _get_recompute_run(mod):  # type: ignore[no-untyped-def]
        """Return the recompute script run() function.

        Returns:
            Callable run function from the recompute module.
        """
        return mod.run

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
        """Load and return the optimize script's run(args) function or None on error.

        Returns:
            The optimize run function or None if loading failed.
        """
        nonlocal _OPT_MOD
        if _OPT_MOD is None:
            try:
                _OPT_MOD = _load_script(
                    "scripts/snqi_weight_optimization.py",
                    "snqi_optimize_script",
                )
            except Exception:  # pragma: no cover - load error
                traceback.print_exc()
                return None
        return _get_opt_run(_OPT_MOD)

    def _invoke_snqi_opt(args: argparse.Namespace) -> int:
        """Invoke the SNQI optimize script with parsed args.

        Returns:
            Exit code from the optimize script.
        """
        _ensure_snqi_opt_defaults(args)
        if os.environ.get("ROBOT_SF_SNQI_LIGHT_TEST") == "1":  # pragma: no cover - test helper
            return 0
        run_fn = _load_optimize_run_fn()
        if run_fn is None:
            return 2
        return int(run_fn(args))  # type: ignore[no-any-return]

    def _invoke_snqi_recompute(args: argparse.Namespace) -> int:
        """Invoke the SNQI recompute script with parsed args.

        Returns:
            Exit code from the recompute script.
        """
        if os.environ.get("ROBOT_SF_SNQI_LIGHT_TEST") == "1":  # pragma: no cover - test helper
            return 0
        nonlocal _RECOMP_MOD
        if _RECOMP_MOD is None:
            try:
                _RECOMP_MOD = _load_script(
                    "scripts/recompute_snqi_weights.py",
                    "snqi_recompute_script",
                )
            except Exception:  # pragma: no cover - load error
                traceback.print_exc()
                return 2
        run_fn = _get_recompute_run(_RECOMP_MOD)
        return int(run_fn(args))  # type: ignore[no-any-return]

    # Public attribute (not underscored) to avoid protected-member lint warning.
    parser.snqi_loader = {  # type: ignore[attr-defined]
        "invoke_optimize": _invoke_snqi_opt,
        "invoke_recompute": _invoke_snqi_recompute,
    }


def _configure_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = _base_parser()
    _attach_core_subcommands(parser)
    # Install intermixed args behavior directly so both get_parser() and cli_main share it.
    if hasattr(parser, "parse_intermixed_args"):
        _orig = parser.parse_args  # type: ignore[assignment]

        def _mixed_parse(args=None, namespace=None):  # type: ignore[override]
            """Parse args while hoisting global flags after subcommands.

            Returns:
                Parsed argparse.Namespace.
            """
            if args is not None:
                # Allow global flags placed after the subcommand by hoisting them before parsing.
                # Pattern observed in tests: [subcommand, --log-level, DEBUG, --quiet]
                # Argparse expects global flags before subcommand; we rewrite the argv sequence.
                try:
                    rewritten: list[str] = []
                    if (
                        args
                        and isinstance(args, list | tuple)
                        and args
                        and not str(args[0]).startswith("-")
                    ):
                        tokens = list(args)
                        subcmd = tokens[0]
                        rest = tokens[1:]
                        i = 0
                        hoisted: list[str] = []
                        while i < len(rest):
                            tok = rest[i]
                            if tok == "--quiet":
                                hoisted.append(tok)
                                i += 1
                                continue
                            if tok == "--log-level":
                                # Expect a value token next
                                if i + 1 < len(rest):
                                    hoisted.extend([tok, rest[i + 1]])
                                    i += 2
                                    continue
                            # Not a global flag → keep in place
                            i += 1
                        # Remove hoisted tokens from rest
                        j = 0
                        filtered: list[str] = []
                        while j < len(rest):
                            if rest[j] == "--quiet":
                                j += 1
                                continue
                            if rest[j] == "--log-level":
                                # Skip flag and its value if we successfully hoisted earlier
                                if j + 1 < len(rest):
                                    j += 2
                                    continue
                            filtered.append(rest[j])
                            j += 1
                        rewritten = [*hoisted, subcmd, *filtered]
                        args = rewritten
                except Exception:  # pragma: no cover - fallback safety
                    pass
            try:  # pragma: no cover (normal success path still covered elsewhere)
                return parser.parse_intermixed_args(args, namespace)  # type: ignore[attr-defined]
            except Exception:
                return _orig(args, namespace)

        parser.parse_args = _mixed_parse  # type: ignore[assignment]
    return parser


def get_parser() -> argparse.ArgumentParser:
    """Return a configured parser (for tests).

    Returns:
        Configured ArgumentParser instance.
    """
    # NOTE: Tests (and some users) supply global flags *after* the subcommand, e.g.:
    #   list-algorithms --log-level DEBUG
    # Vanilla argparse only supports global options before a subcommand. We wrap
    # parse_args to attempt parse_intermixed_args first (Python 3.7+) and fall
    # back silently if unsupported or if parsing fails.
    parser = _configure_parser()
    if hasattr(parser, "parse_intermixed_args"):
        _orig = parser.parse_args  # type: ignore[assignment]

        def _mixed_parse(args=None, namespace=None):  # type: ignore[override]
            """Parse args using intermixed parsing when available.

            Returns:
                Parsed argparse.Namespace.
            """
            try:  # pragma: no cover - fallback path only hit if feature absent/fails
                return parser.parse_intermixed_args(args, namespace)  # type: ignore[attr-defined]
            except Exception:
                return _orig(args, namespace)

        parser.parse_args = _mixed_parse  # type: ignore[assignment]
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for benchmark commands.

    Returns:
        Process exit code.
    """
    parser = _configure_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "quiet", False), getattr(args, "log_level", "INFO"))
    # macOS safe start method for multiprocessing
    if getattr(args, "workers", 1) and int(getattr(args, "workers", 1)) > 1:
        try:
            multiprocessing_module = importlib.import_module("multiprocessing")
            multiprocessing_module.set_start_method("spawn", force=False)
        except Exception:
            logging.debug("Failed to set multiprocessing start method to spawn", exc_info=True)

    # Access dynamic loaders if present
    snqi_loader = getattr(parser, "snqi_loader", {})  # type: ignore[no-any-explicit]
    # Dispatch
    if args.cmd == "snqi":
        if args.snqi_cmd == "optimize":
            return snqi_loader["invoke_optimize"](args)
        if args.snqi_cmd == "recompute":
            return snqi_loader["invoke_recompute"](args)
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
        "table": _handle_table,
        "debug-seeds": _handle_debug_seeds,
        "plot-pareto": _handle_plot_pareto,
        "plot-distributions": _handle_plot_distributions,
        "plot-scenarios": _handle_plot_scenarios,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


def main() -> None:  # pragma: no cover - thin wrapper
    """Entry-point wrapper for console scripts."""
    raise SystemExit(cli_main())


__all__ = ["cli_main", "configure_logging", "get_parser", "main"]
