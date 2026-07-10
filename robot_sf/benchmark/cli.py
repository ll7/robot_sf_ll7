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
import shlex
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
from robot_sf.benchmark.algorithm_readiness import get_algorithm_readiness
from robot_sf.benchmark.baseline_stats import (
    DEFAULT_BASELINE_JSONL_PATH,
    run_and_compute_baseline,
)
from robot_sf.benchmark.benchmark_claim import (
    BenchmarkClaimError,
    build_benchmark_claim,
    write_benchmark_claim,
)
from robot_sf.benchmark.benchmark_row_claim import (
    BenchmarkRowClaimError,
    validate_all_leaderboards,
    validate_leaderboard_claims,
)
from robot_sf.benchmark.canonical_table_export import (
    TABLE_SPECS as _canonical_table_specs,
)
from robot_sf.benchmark.canonical_table_export import (
    export_canonical_table as _export_canonical_table,
)
from robot_sf.benchmark.canonical_table_export import load_rows_json as _load_canonical_rows_json
from robot_sf.benchmark.collision_scenario_similarity import (
    build_collision_scenario_similarity_report,
    write_collision_scenario_similarity_report,
)
from robot_sf.benchmark.distributions import collect_grouped_values as _dist_collect
from robot_sf.benchmark.distributions import save_distributions as _dist_save
from robot_sf.benchmark.doctor import collect_doctor_report, doctor_exit_code
from robot_sf.benchmark.errors import AggregationMetadataError, EpisodeRecordInputError
from robot_sf.benchmark.failure_extractor import extract_failures as _extract_failures
from robot_sf.benchmark.failure_mechanism_classifier import (
    classify_failure_mechanisms_from_jsonl,
)
from robot_sf.benchmark.fallback_policy import availability_payload, benchmark_run_exit_code
from robot_sf.benchmark.grouping import DEFAULT_REPORT_FALLBACK_GROUP_BY, DEFAULT_REPORT_GROUP_BY
from robot_sf.benchmark.metric_layers import build_metric_layer_summary
from robot_sf.benchmark.observation_levels import OBSERVATION_LEVEL_KEYS
from robot_sf.benchmark.observation_noise import load_observation_noise_spec
from robot_sf.benchmark.parquet_export import export_episodes_jsonl_to_parquet
from robot_sf.benchmark.planner_inclusion import (
    DEFAULT_INCLUSION_MATRIX,
    InclusionCriteria,
    run_planner_inclusion_check,
    to_jsonable_payload,
)
from robot_sf.benchmark.planner_tradeoff_plot import (
    save_planner_tradeoff_figure as _save_planner_tradeoff_figure,
)
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
from robot_sf.benchmark.scenario_flakiness import (
    DEFAULT_OBSERVATION_TRACK_MODE as _DEFAULT_FLAKINESS_TRACK_MODE,
)
from robot_sf.benchmark.scenario_flakiness import (
    compute_flakiness_audit as _compute_flakiness_audit,
)
from robot_sf.benchmark.scenario_schema import (
    validate_scenario_list,
    validate_scenario_matrix_metadata,
)
from robot_sf.benchmark.scenario_thumbnails import (
    resolve_scenario_label as _thumb_resolve_label,
)
from robot_sf.benchmark.scenario_thumbnails import save_montage as _thumb_montage
from robot_sf.benchmark.scenario_thumbnails import (
    save_scenario_thumbnails as _thumb_save_all,
)
from robot_sf.benchmark.seed_variance import (
    compute_seed_variance as _compute_seed_variance,
)
from robot_sf.benchmark.stress_uncertainty_coverage import (
    build_stress_uncertainty_coverage_report_from_jsonl,
    load_stress_uncertainty_coverage_payload,
    write_stress_uncertainty_coverage_report,
)
from robot_sf.benchmark.summary import summarize_to_plots
from robot_sf.common.seed import get_seed_state_sample as _seed_sample
from robot_sf.common.seed import set_global_seed as _set_seed
from robot_sf.training.task_bundles import (
    describe_task_bundle_source,
    is_task_bundle_reference,
)

DEFAULT_SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"

# CLI commands translate malformed files, configuration values, and benchmark records into
# their documented nonzero exit codes.  Programmer errors intentionally remain visible.
_CLI_INPUT_ERRORS = (
    OSError,
    json.JSONDecodeError,
    yaml.YAMLError,
    AggregationMetadataError,
    EpisodeRecordInputError,
)
# AttributeError is limited to the optional tqdm API probe below; ordinary CLI
# logging/display calls must not hide programmer mistakes.
_CLI_OPTIONAL_DEPENDENCY_ERRORS = (ImportError, ModuleNotFoundError, AttributeError)
_CLI_LOGGING_ERRORS = (OSError, ValueError)


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
            benchmark_profile=args.benchmark_profile,
            workers=args.workers,
            resume=(not bool(getattr(args, "no_resume", False))),
            progress_cb=progress_cb,
        )
        # Inform the user where outputs were written (INFO-level; suppressed with --quiet)
        try:
            logging.info("Baseline stats written to %s", args.out)
            if getattr(args, "jsonl", None):
                logging.info("Intermediate episodes JSONL written to %s", args.jsonl)
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            # Defensive: do not let logging interfere with success path
            pass
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
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
    except (ImportError, AttributeError, OSError):  # pragma: no cover - optional registry boundary
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
        except _CLI_OPTIONAL_DEPENDENCY_ERRORS:
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
            except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive display boundary
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

    def _emit_structured(event_payload: dict[str, Any]) -> None:
        """Emit machine-readable run events when structured output is enabled."""
        mode = str(getattr(args, "structured_output", "none")).lower()
        if mode not in {"json", "jsonl"}:
            return
        print(json.dumps(event_payload, sort_keys=True))

    try:
        _configure_external_log_noise(
            mode=str(getattr(args, "external_log_noise", "auto")),
            log_level=str(getattr(args, "log_level", "INFO")),
        )
        # Optional: load SNQI weights/baseline for inline SNQI computation
        try:
            snqi_weights, snqi_baseline = _load_snqi_inputs(args)
        except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
            _emit_structured(
                {"event": "benchmark.run.error", "error": "failed_loading_snqi_inputs"},
            )
            return 2

        readiness = get_algorithm_readiness(str(args.algo))
        if readiness is not None:
            logging.info(
                "Algorithm readiness: algo=%s tier=%s profile=%s note=%s",
                readiness.canonical_name,
                readiness.tier,
                getattr(args, "benchmark_profile", "baseline-safe"),
                readiness.note,
            )

        summary = run_batch(
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
            benchmark_profile=args.benchmark_profile,
            socnav_missing_prereq_policy=args.socnav_missing_prereq_policy,
            adapter_impact_eval=bool(getattr(args, "adapter_impact_eval", False)),
            experimental_ped_impact=bool(getattr(args, "experimental_ped_impact", False)),
            ped_impact_radius_m=float(getattr(args, "ped_impact_radius_m", 2.0)),
            ped_impact_window_steps=int(getattr(args, "ped_impact_window_steps", 5)),
            observation_mode=getattr(args, "observation_mode", None),
            observation_level=getattr(args, "observation_level", None),
            benchmark_track=getattr(args, "benchmark_track", None),
            track_schema_version=getattr(args, "track_schema_version", None),
            record_simulation_step_trace=bool(getattr(args, "record_simulation_step_trace", False)),
            observation_noise=(
                load_observation_noise_spec(args.observation_noise)
                if getattr(args, "observation_noise", None)
                else None
            ),
            snqi_weights=snqi_weights,
            snqi_baseline=snqi_baseline,
            workers=args.workers,
            resume=(not bool(getattr(args, "no_resume", False))),
        )
        availability = summary.get("benchmark_availability")
        if not isinstance(availability, dict):
            availability = availability_payload(summary)
            summary["benchmark_availability"] = availability
        total_jobs = int(summary.get("total_jobs", 0))
        written = int(summary.get("written", 0))
        benchmark_exit_code = benchmark_run_exit_code(summary)
        failed = summary.get("failures", [])
        failure_count = (
            len(failed) if isinstance(failed, list) else int(summary.get("failed_jobs", 0))
        )
        try:
            logging.info(
                "Run summary: total_jobs=%s written=%s failed=%s out=%s",
                total_jobs,
                written,
                failure_count,
                args.out,
            )
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Episodes written' failed", exc_info=True)
        if total_jobs > 0 and written == 0:
            try:
                logging.error(
                    "Benchmark run produced zero episodes for %s scheduled jobs (%s failures).",
                    total_jobs,
                    failure_count,
                )
            except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
                logging.debug("Logging zero-episode failure failed", exc_info=True)
            if str(getattr(args, "structured_output", "none")).lower() == "jsonl" and isinstance(
                failed, list
            ):
                for failure in failed:
                    if isinstance(failure, dict):
                        _emit_structured({"event": "benchmark.run.failure", **failure})
            _emit_structured(
                {
                    "event": "benchmark.run.summary",
                    "exit_code": 2,
                    "total_jobs": total_jobs,
                    "written": written,
                    "failed_jobs": failure_count,
                    "out_path": str(summary.get("out_path", args.out)),
                    "benchmark_availability": availability,
                },
            )
            return 2
        if benchmark_exit_code != 0:
            specific_reason = None
            specific_reason = availability.get("availability_reason")
            reason = (
                str(specific_reason)
                if specific_reason is not None
                else "benchmark run did not satisfy the benchmark availability policy"
            )
            try:
                logging.error("Benchmark run marked non-success: %s", reason)
            except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
                logging.debug("Logging benchmark availability failure failed", exc_info=True)
            _emit_structured(
                {
                    "event": "benchmark.run.summary",
                    "exit_code": benchmark_exit_code,
                    "total_jobs": total_jobs,
                    "written": written,
                    "failed_jobs": failure_count,
                    "out_path": str(summary.get("out_path", args.out)),
                    "benchmark_availability": availability,
                },
            )
            return benchmark_exit_code
        _emit_structured(
            {
                "event": "benchmark.run.summary",
                "exit_code": 0,
                "total_jobs": total_jobs,
                "written": written,
                "failed_jobs": failure_count,
                "out_path": str(summary.get("out_path", args.out)),
                "benchmark_availability": summary.get("benchmark_availability"),
            },
        )
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
        _emit_structured({"event": "benchmark.run.error", "error": "unhandled_exception"})
        return 2


def _configure_external_log_noise(*, mode: str, log_level: str) -> None:
    """Apply best-effort external log suppression policy for noisy dependencies."""
    mode_norm = mode.strip().lower()
    if mode_norm not in {"auto", "suppress", "verbose"}:
        mode_norm = "auto"
    should_suppress = mode_norm == "suppress" or (
        mode_norm == "auto" and log_level.strip().upper() != "DEBUG"
    )
    if not should_suppress:
        return
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    for logger_name in (
        "matplotlib",
        "PIL",
        "pygame",
        "moviepy",
        "tensorflow",
        "numba",
        "OpenGL",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _handle_summary(args) -> int:
    """Generate summary plots from episodes.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        summarize_to_plots(args.in_path, args.out_dir)
        try:
            logging.info("Summary plots written to %s", args.out_dir)
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Summary plots written' failed", exc_info=True)
        return 0
    except _CLI_INPUT_ERRORS as exc:  # pragma: no cover - error path
        logging.exception("Summary generation failed: %s", exc)
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
                recompute_snqi=bool(getattr(args, "recompute_snqi", False)),
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_confidence=float(args.bootstrap_confidence),
                bootstrap_seed=(
                    int(args.bootstrap_seed) if args.bootstrap_seed is not None else None
                ),
                observation_track_mode=str(args.observation_track_mode),
            )
        else:
            summary = _agg_compute(
                records,
                group_by=args.group_by,
                fallback_group_by=args.fallback_group_by,
                snqi_weights=snqi_weights,
                snqi_baseline=snqi_baseline,
                recompute_snqi=bool(getattr(args, "recompute_snqi", False)),
                observation_track_mode=str(args.observation_track_mode),
            )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        try:
            logging.info("Aggregated summary written to %s", out_path)
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Aggregated summary' failed", exc_info=True)
        return 0
    except _CLI_INPUT_ERRORS as exc:  # pragma: no cover - error path
        logging.exception("Aggregation failed: %s", exc)
        return 2


def _handle_metric_layers(args) -> int:
    """Build canonical metric-layer summary from episode JSONL records.

    Returns:
        Exit code ``0`` on success, ``2`` on failure.
    """
    try:
        records = _agg_read_jsonl(args.episodes)
        summary = build_metric_layer_summary(
            records,
            group_by=args.group_by,
            fallback_group_by=args.fallback_group_by,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, allow_nan=False)
        return 0
    except (EpisodeRecordInputError, OSError, TypeError, ValueError) as exc:
        logging.exception("Metric-layer summary failed: %s", exc)
        return 2


def _handle_claim(args) -> int:
    """Build a benchmark claim artifact and write a compact JSON payload.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        claim = build_benchmark_claim(
            claim_id=args.claim_id,
            statement=args.statement,
            scenario_matrix_path=Path(args.scenario_matrix),
            scenario_matrix_sha256=args.scenario_matrix_sha256,
            policy_metadata_path=Path(args.policy_metadata),
            training_episodes=[Path(path) for path in (args.training_episodes or [])],
            validation_episodes=[Path(path) for path in (args.validation_episodes or [])],
            final_benchmark_episodes=[Path(path) for path in (args.final_benchmark_episodes or [])],
            aggregate_reports=[Path(path) for path in (args.aggregate_report or [])],
            dependency_group=args.dependency_group,
            container_image_digest=args.container_image_digest,
        )
        output_path = Path(args.output_json)
        write_benchmark_claim(output_path, claim)
        print(
            json.dumps(
                {
                    "claim_path": str(output_path),
                    "schema_version": claim["schema_version"],
                    "claim_id": claim["claim_id"],
                },
                indent=2,
            )
        )
        return 0
    except BenchmarkClaimError as exc:
        print(f"Benchmark claim error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError):  # pragma: no cover - input/output boundary
        logging.exception("Unexpected error during benchmark claim generation")
        return 2


def _handle_validate_row_claims(args) -> int:
    """Validate benchmark row claim sidecar files.

    Returns:
        Exit code (0 when all checked leaderboards are valid, 2 otherwise).
    """
    try:
        if bool(args.all) == bool(args.sidecar):
            raise BenchmarkRowClaimError("pass exactly one of --all or --sidecar")
        if args.all:
            report = validate_all_leaderboards()
        else:
            report = validate_leaderboard_claims(Path(args.sidecar))
        print(json.dumps(report, indent=2))
        return 0 if report.get("valid") or report.get("overall_valid") else 2
    except BenchmarkRowClaimError as exc:
        print(f"Benchmark row claim error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError):  # pragma: no cover - input/output boundary
        logging.exception("Unexpected error during benchmark row claim validation")
        return 2


def _handle_stress_coverage_report(args) -> int:
    """Build or validate a stress/uncertainty coverage report.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        if args.summary_json:
            payload = load_stress_uncertainty_coverage_payload(args.summary_json)
        else:
            payload = build_stress_uncertainty_coverage_report_from_jsonl(
                args.episodes_jsonl,
                report_id=args.report_id,
                campaign_config_hash=args.campaign_config_hash,
                scenario_matrix_hash=args.scenario_matrix_hash,
                schema_mode=args.schema_mode,
                aggregate_mode=args.aggregate_mode,
                availability_status=args.availability_status,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_confidence=args.bootstrap_confidence,
                bootstrap_seed=args.bootstrap_seed,
            )
        out_path = write_stress_uncertainty_coverage_report(payload, args.out)
        logging.info("Stress/uncertainty coverage report written to %s", out_path)
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input boundary exercised by sibling commands
        logging.exception("Stress/uncertainty coverage report failed")
        return 2


def _handle_classify_failure_mechanisms(args) -> int:
    """Classify paired fixed/long-horizon failure mechanisms.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        payload = classify_failure_mechanisms_from_jsonl(
            args.episodes_jsonl,
            scenario_certificates=args.scenario_certificates,
            output_json=args.out_json,
            output_csv=args.out_csv,
            fixed_horizon=args.fixed_horizon,
            long_horizon=args.long_horizon,
        )
        logging.info(
            "Failure mechanism classification wrote %d rows to %s",
            len(payload["rows"]),
            args.out_json,
        )
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input boundary exercised by sibling commands
        logging.exception("Failure mechanism classification failed")
        return 2


def _handle_collision_scenario_similarity(args) -> int:
    """Build collision-scenario similarity report from episode JSONL records.

    Returns:
        Process exit code.
    """
    try:
        report = build_collision_scenario_similarity_report(
            args.episodes_jsonl,
            nearest_k=args.nearest_k,
            group_threshold=args.group_threshold,
            collision_threshold=args.collision_threshold,
            near_miss_threshold=args.near_miss_threshold,
            comfort_threshold=args.comfort_threshold,
            require_trajectory_comparison=args.require_trajectory_comparison,
        )
        write_collision_scenario_similarity_report(
            report,
            args.out_json,
            out_markdown=args.out_markdown,
        )
        logging.info(
            "Collision scenario similarity report wrote %d selected records to %s",
            report["selection"]["selected_count"],
            args.out_json,
        )
        return 0
    except (EpisodeRecordInputError, OSError, TypeError, ValueError):
        logging.exception("Collision scenario similarity report failed")
        return 2


def _handle_export_parquet(args) -> int:
    """Export benchmark episode JSONL records to Parquet analytics tables.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        result = export_episodes_jsonl_to_parquet(
            args.in_path,
            args.out_dir,
            overwrite=bool(args.overwrite),
        )
        logging.info(
            "Parquet analytics export written to %s (%d episode records)",
            result.output_dir,
            result.record_count,
        )
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input boundary exercised by sibling commands
        logging.exception("Parquet analytics export failed")
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
            observation_track_mode=str(args.observation_track_mode),
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
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Ablation results' failed", exc_info=True)
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input/output boundary
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
            observation_track_mode=str(args.observation_track_mode),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        try:
            logging.info("Seed-variance summary written to %s", out_path)
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Seed-variance summary' failed", exc_info=True)
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
        return 2


def _handle_flakiness_audit(args) -> int:
    """Audit scenario outcome flakiness and write a JSON report.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        records = _agg_read_jsonl(args.in_path)
        report = _compute_flakiness_audit(
            records,
            outcome_metric=str(args.outcome_metric),
            group_by=args.group_by,
            fallback_group_by=args.fallback_group_by,
            seed_field=str(args.seed_field),
            stability_threshold=float(args.stability_threshold),
            min_seeds=int(args.min_seeds),
            observation_track_mode=str(args.observation_track_mode),
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        try:
            summary = report["summary"]
            logging.info(
                "Flakiness audit written to %s (%s cells, %s knife-edge, determinism=%s)",
                out_path,
                summary["n_cells"],
                summary["n_knife_edge_cells"],
                report["exact_repeat"]["is_deterministic"],
            )
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Flakiness audit' failed", exc_info=True)
        return 0
    except ValueError:
        # Fail closed on an empty/invalid audit request rather than emit a report
        # that asserts stability without evidence.
        logging.exception("Flakiness audit failed")
        return 2
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
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
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Wrote failures' failed", exc_info=True)
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
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
            observation_track_mode=str(args.observation_track_mode),
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
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Ranking output' failed", exc_info=True)
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
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
            observation_track_mode=str(args.observation_track_mode),
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
        except _CLI_LOGGING_ERRORS:  # pragma: no cover - defensive logging boundary
            logging.debug("Logging 'Table output' failed", exc_info=True)
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
        return 2


def _handle_export_canonical_table(args) -> int:
    """Export named canonical benchmark tables from JSON row fixtures.

    Returns:
        Process exit code; zero indicates all requested outputs were written.
    """
    try:
        rows = _load_canonical_rows_json(args.rows)
        formats = [fmt.strip() for fmt in str(args.formats).split(",") if fmt.strip()]
        result = _export_canonical_table(
            rows,
            table_id=str(args.table_id),
            output_dir=args.out_dir,
            formats=formats,
            precision=int(args.precision),
            source_paths=[Path(path) for path in args.source],
            command=str(getattr(args, "_canonical_command", shlex.join(sys.argv))),
        )
        print(
            json.dumps(
                {
                    "table_id": result.table_id,
                    "row_count": result.row_count,
                    "outputs": {
                        fmt: path.as_posix() for fmt, path in sorted(result.output_paths.items())
                    },
                    "metadata": result.metadata_path.as_posix(),
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, TypeError) as exc:  # pragma: no cover - input/output boundary
        logging.exception("Canonical table export failed: %s", exc)
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
    except (OSError, ValueError, TypeError):  # pragma: no cover - input/output boundary
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
            observation_track_mode=str(args.observation_track_mode),
        )
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input/output boundary
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
            observation_track_mode=str(args.observation_track_mode),
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
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input/output boundary
        return 2


def _handle_plot_planner_tradeoff(args) -> int:
    """Render the paper-style planner safety-efficiency tradeoff plot.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        meta = _save_planner_tradeoff_figure(
            Path(args.bundle_path),
            out_png=Path(args.out),
            out_pdf=(Path(args.out_pdf) if getattr(args, "out_pdf", None) else None),
            bootstrap_samples=int(args.bootstrap_samples),
            ci_confidence=float(args.ci_confidence),
            bootstrap_seed=int(args.bootstrap_seed),
            title=(str(args.title) if args.title is not None else None),
        )
        if getattr(args, "metadata_out", None):
            metadata_path = Path(args.metadata_out)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, TypeError) as exc:  # pragma: no cover - input/output boundary
        logging.exception("Planner tradeoff plot failed: %s", exc)
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
            sid = _thumb_resolve_label(sc)
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
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input/output boundary
        return 2


def _handle_doctor(args: argparse.Namespace) -> int:
    """Run local runtime diagnostics and print JSON output.

    Returns:
        int: Doctor command exit code.
    """
    report = collect_doctor_report(
        artifact_root=args.artifact_root,
        run_env_smoke=not args.skip_env_smoke,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return doctor_exit_code(report)


def _handle_mapf_oracle(args: argparse.Namespace) -> int:
    """Run MAPF oracle diagnostics on a scenario matrix.

    Returns:
        int: Exit code (0 success, 1 error).
    """
    from robot_sf.benchmark.mapf_oracle import run_mapf_oracle_diagnostics  # noqa: PLC0415

    report = run_mapf_oracle_diagnostics(
        args.matrix,
        grid_size=args.grid_size,
        scenario_filter=args.filter,
    )
    print(json.dumps(report, indent=2, sort_keys=False))
    if report.get("status") == "error":
        return 1
    return 0


def _handle_list_scenarios(args) -> int:
    """List scenarios from a matrix.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        scenarios = load_scenario_matrix(args.matrix)
        ids = [_thumb_resolve_label(s) for s in scenarios]
        for sid in ids:
            print(sid)
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - error path
        return 2


def _extract_matrix_source(matrix_path: str | Path) -> dict[str, object]:
    """Describe the scenario matrix source and include structure.

    Returns:
        dict[str, object]: Source metadata including format, include list, and selector list.
    """
    if is_task_bundle_reference(matrix_path):
        return describe_task_bundle_source(matrix_path)

    path = Path(matrix_path)
    includes: list[str] = []
    select_scenarios: list[str] = []
    format_hint = "unknown"
    try:
        with path.open("r", encoding="utf-8") as handle:
            docs = list(yaml.safe_load_all(handle))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):  # pragma: no cover - metadata fallback
        return {
            "path": str(path),
            "format": format_hint,
            "includes": includes,
            "select_scenarios": select_scenarios,
        }
    if len(docs) > 1:
        format_hint = "stream"
        return {
            "path": str(path),
            "format": format_hint,
            "includes": includes,
            "select_scenarios": select_scenarios,
        }
    if not docs:
        return {
            "path": str(path),
            "format": format_hint,
            "includes": includes,
            "select_scenarios": select_scenarios,
        }
    doc = docs[0]
    schema_version: object = None
    if isinstance(doc, Mapping):
        schema_version = doc.get("schema_version")
        raw_includes = doc.get("includes") or doc.get("include") or doc.get("scenario_files")
        if isinstance(raw_includes, list):
            includes = [str(entry) for entry in raw_includes]
        elif isinstance(raw_includes, (str, Path)):
            includes = [str(raw_includes)]
        raw_select = doc.get("select_scenarios")
        if isinstance(raw_select, list):
            select_scenarios = [str(entry) for entry in raw_select]
        elif isinstance(raw_select, (str, Path)):
            select_scenarios = [str(raw_select)]
        format_hint = "manifest" if includes or select_scenarios else "dict"
        if "scenarios" in doc:
            format_hint = "manifest" if includes or select_scenarios else "list"
    elif isinstance(doc, list):
        format_hint = "list"
    return {
        "path": str(path),
        "format": format_hint,
        "includes": includes,
        "select_scenarios": select_scenarios,
        **({"schema_version": schema_version} if isinstance(schema_version, str) else {}),
    }


def _load_matrix_metadata(matrix_path: str | Path) -> object:
    """Load raw top-level YAML metadata for validation without expanding includes.

    Returns:
        object: First YAML document, or ``None`` when unavailable.
    """
    if is_task_bundle_reference(matrix_path):
        return None
    try:
        with Path(matrix_path).open("r", encoding="utf-8") as handle:
            return next(yaml.safe_load_all(handle), None)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):  # pragma: no cover - metadata fallback
        return None


def _summarize_scenarios(scenarios: list[dict[str, Any]]) -> dict[str, object]:
    """Summarize scenario counts and basic field coverage.

    Returns:
        dict[str, object]: Summary counts for archetypes/densities/maps and missing fields.
    """

    def _pick_meta_value(scenario: dict[str, Any], key: str) -> str:
        """Resolve a summary field from metadata first, then scenario top level.

        Returns:
            str: Non-empty metadata/top-level value, or ``"unknown"``.
        """
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


def _collect_scenario_warnings(  # noqa: PLR0912,PLR0915
    scenarios: list[dict[str, Any]],
    *,
    matrix_path: str | Path | None,
) -> list[dict[str, Any]]:
    """Collect warn-only scenario validation messages.

    Returns:
        list[dict[str, Any]]: Warning records (non-fatal).
    """

    def warn(index: int, scenario_id: str, message: str, path: str) -> None:
        """Append one non-fatal scenario validation warning record."""
        warnings.append(
            {
                "index": index,
                "id": scenario_id,
                "warning": message,
                "path": path,
            }
        )

    warnings: list[dict[str, Any]] = []
    base_dir = _resolve_warning_base_dir(matrix_path)
    for idx, scenario in enumerate(scenarios):
        scenario_id = str(
            scenario.get("id") or scenario.get("name") or scenario.get("scenario_id") or idx
        )
        map_file = scenario.get("map_file")
        if not isinstance(map_file, str):
            warn(idx, scenario_id, "map_file missing or not a string", "/map_file")
        else:
            map_path = Path(map_file)
            if not map_path.is_absolute() and base_dir is not None:
                map_path = (base_dir / map_path).resolve()
            if not map_path.exists():
                warn(idx, scenario_id, f"map_file not found at {map_path}", "/map_file")
            elif map_path.suffix.lower() != ".svg":
                warn(
                    idx,
                    scenario_id,
                    f"map_file extension '{map_path.suffix}' not supported; use SVG maps",
                    "/map_file",
                )

        sim_cfg = scenario.get("simulation_config")
        if not isinstance(sim_cfg, Mapping):
            warn(
                idx, scenario_id, "simulation_config missing or not a mapping", "/simulation_config"
            )
            sim_cfg = {}
        max_steps = sim_cfg.get("max_episode_steps")
        if max_steps is None:
            warn(
                idx,
                scenario_id,
                "max_episode_steps missing",
                "/simulation_config/max_episode_steps",
            )
        else:
            try:
                if int(max_steps) <= 0:
                    warn(
                        idx,
                        scenario_id,
                        "max_episode_steps must be > 0",
                        "/simulation_config/max_episode_steps",
                    )
            except (TypeError, ValueError):
                warn(
                    idx,
                    scenario_id,
                    "max_episode_steps must be an integer",
                    "/simulation_config/max_episode_steps",
                )

        metadata = scenario.get("metadata")
        spawn_mode = metadata.get("spawn_mode") if isinstance(metadata, Mapping) else None
        density = sim_cfg.get("ped_density")
        if density is not None:
            try:
                density_val = float(density)
                if density_val < 0:
                    warn(
                        idx,
                        scenario_id,
                        f"ped_density must be >= 0 (got {density_val})",
                        "/simulation_config/ped_density",
                    )
                marker_placeholder_density = density_val == 0 and spawn_mode == "markers"
                if density_val == 0 and not marker_placeholder_density:
                    warn(
                        idx,
                        scenario_id,
                        "ped_density=0.0 means no pedestrians spawn",
                        "/simulation_config/ped_density",
                    )
                if not marker_placeholder_density and not 0.02 <= density_val <= 0.08:
                    warn(
                        idx,
                        scenario_id,
                        (
                            "ped_density outside recommended [0.02, 0.08]; "
                            "unit is peds per m^2 of spawnable area "
                            "(route/zone sidewalk area, not whole-map); "
                            "may reduce benchmark comparability"
                        ),
                        "/simulation_config/ped_density",
                    )
                if not marker_placeholder_density and density_val > 0.15:
                    warn(
                        idx,
                        scenario_id,
                        (
                            f"ped_density={density_val} is high; unit is "
                            "peds per m^2 of spawnable area (recommended "
                            "0.02-0.08). If this value was meant per "
                            "whole-map area or per route meter, it is likely "
                            "a unit confusion."
                        ),
                        "/simulation_config/ped_density",
                    )
            except (TypeError, ValueError):
                warn(
                    idx,
                    scenario_id,
                    "ped_density must be a number",
                    "/simulation_config/ped_density",
                )

        metadata = scenario.get("metadata")
        if not isinstance(metadata, Mapping):
            warn(idx, scenario_id, "metadata missing or not a mapping", "/metadata")
            metadata = {}
        archetype = metadata.get("archetype")
        density_tag = metadata.get("density")
        if not isinstance(archetype, str) or not archetype:
            warn(idx, scenario_id, "metadata.archetype missing", "/metadata/archetype")
        if not isinstance(density_tag, str) or not density_tag:
            warn(idx, scenario_id, "metadata.density missing", "/metadata/density")

        groups_val = sim_cfg.get("groups", 0.0)
        if archetype == "group_crossing" and groups_val != 0.5:
            warn(
                idx,
                scenario_id,
                f"group_crossing should set groups=0.5 (got {groups_val})",
                "/simulation_config/groups",
            )
        if archetype != "group_crossing" and groups_val not in (0.0, None):
            warn(
                idx,
                scenario_id,
                f"non-group archetype should not set groups (got {groups_val})",
                "/simulation_config/groups",
            )

        seeds = scenario.get("seeds")
        if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
            warn(idx, scenario_id, "seeds missing or not a list of ints", "/seeds")
        elif len(seeds) < 3:
            warn(idx, scenario_id, "seeds list should contain at least 3 entries", "/seeds")

    return warnings


def _resolve_warning_base_dir(matrix_path: str | Path | None) -> Path | None:
    """Resolve the best base directory for warning-only relative path checks.

    Returns:
        Path | None: Directory used for relative map checks, if known.
    """
    if matrix_path is None:
        return None
    if is_task_bundle_reference(matrix_path):
        source = describe_task_bundle_source(matrix_path)
        scenario_files = source.get("scenario_files")
        if isinstance(scenario_files, list) and scenario_files:
            first = scenario_files[0]
            if isinstance(first, str):
                return Path(first).parent
        return None
    return Path(matrix_path).parent


def _handle_validate_config(args) -> int:
    """Validate scenario matrix config against schema rules.

    Returns:
        Exit code (0 success, 2 failure).
    """
    try:
        scenarios = load_scenario_matrix(args.matrix)
        metadata_errors = validate_scenario_matrix_metadata(_load_matrix_metadata(args.matrix))
        errors = [*metadata_errors, *validate_scenario_list(scenarios)]
        warnings = _collect_scenario_warnings(scenarios, matrix_path=args.matrix)
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
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input/config boundary
        return 2


def _handle_preview_scenarios(args) -> int:
    """Preview scenario matrix with warn-only checks.

    Returns:
        Exit code (0 always).
    """
    try:
        scenarios = load_scenario_matrix(args.matrix)
        warnings = _collect_scenario_warnings(scenarios, matrix_path=args.matrix)
        summary = _summarize_scenarios(scenarios)
        source = _extract_matrix_source(args.matrix)
        report = {
            "num_scenarios": len(scenarios),
            "warnings": warnings,
            "summary": summary,
            "source": source,
            "policy": "warn-only",
        }
        print(json.dumps(report))
        return 0
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input/config boundary
        return 2


def _handle_planner_inclusion_check(args) -> int:
    """Run one planner through the mechanical inclusion-review gate.

    Returns:
        Exit code 0 for pass, 1 for gate revision, 2 for execution error.
    """
    try:
        report = run_planner_inclusion_check(
            algo=str(args.algo),
            matrix=Path(args.matrix),
            schema=Path(args.schema),
            output_dir=Path(args.output_dir),
            algo_config=Path(args.algo_config) if args.algo_config else None,
            benchmark_profile=str(args.benchmark_profile),
            base_seed=int(args.base_seed),
            repeats=args.repeats,
            horizon=int(args.horizon),
            dt=float(args.dt),
            workers=int(args.workers),
            record_forces=bool(args.record_forces),
            socnav_missing_prereq_policy=str(args.socnav_missing_prereq_policy),
            resume=bool(args.resume),
            criteria=InclusionCriteria(
                min_episodes=int(args.min_episodes),
                min_success_rate=float(args.min_success_rate),
                max_collision_rate=float(args.max_collision_rate),
                max_runtime_sec=float(args.max_runtime_sec),
            ),
        )
        print(json.dumps(to_jsonable_payload(report), indent=2, allow_nan=False))
        return 0 if report.get("decision") == "pass" else 1
    except _CLI_INPUT_ERRORS:  # pragma: no cover - input/runtime boundary
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
        help=(
            "Optional path to write intermediate episode JSONL "
            f"(default {DEFAULT_BASELINE_JSONL_PATH!s})"
        ),
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
        "--benchmark-profile",
        default="baseline-safe",
        choices=("baseline-safe", "experimental", "paper-baseline"),
        help="Algorithm readiness profile for map-based benchmark runs",
    )
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
        "--observation-noise",
        help="Path to an observation-noise YAML profile applied to planner inputs",
    )
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
        "--benchmark-profile",
        choices=["baseline-safe", "paper-baseline", "experimental"],
        default="baseline-safe",
        help=(
            "Algorithm readiness profile. "
            "'baseline-safe' blocks experimental/placeholder planners; "
            "'paper-baseline' additionally requires paper-grade PPO gating."
        ),
    )
    p.add_argument(
        "--socnav-missing-prereq-policy",
        choices=["fail-fast", "skip-with-warning", "fallback"],
        default="fail-fast",
        help=(
            "Behavior when SocNav dependencies/models are missing: "
            "raise, skip algorithm run, or force adapter fallback."
        ),
    )
    p.add_argument(
        "--adapter-impact-eval",
        action="store_true",
        help=(
            "Enable adapter-impact metadata probing. "
            "For mixed-command planners (e.g., PPO), records native vs adapted step usage."
        ),
    )
    p.add_argument(
        "--experimental-ped-impact",
        action="store_true",
        help="Emit schema-backed pedestrian-impact near-vs-far reductions.",
    )
    p.add_argument(
        "--ped-impact-radius-m",
        type=float,
        default=2.0,
        help="Near/far split radius in meters for --experimental-ped-impact.",
    )
    p.add_argument(
        "--ped-impact-window-steps",
        type=int,
        default=5,
        help="Trailing smoothing window length for --experimental-ped-impact.",
    )
    p.add_argument(
        "--observation-mode",
        default=None,
        help=(
            "Optional planner observation-mode override. Unsupported planner/mode "
            "combinations fail before episodes are written."
        ),
    )
    p.add_argument(
        "--observation-level",
        default=None,
        choices=OBSERVATION_LEVEL_KEYS,
        help=(
            "Optional graded benchmark observation-level override. Unsupported "
            "planner/level combinations fail before episodes are written."
        ),
    )
    p.add_argument(
        "--benchmark-track",
        default=None,
        help="Optional observation-track aggregation fence for track-aware benchmark rows.",
    )
    p.add_argument(
        "--track-schema-version",
        default=None,
        help="Optional version slug for the benchmark-track metadata contract.",
    )
    p.add_argument(
        "--record-simulation-step-trace",
        action="store_true",
        help="Embed analysis-only per-step trace frames in each aggregate episode row.",
    )
    p.add_argument(
        "--structured-output",
        choices=["none", "json", "jsonl"],
        default="none",
        help=(
            "Emit machine-readable run events to stdout. "
            "'json' emits a final summary object; 'jsonl' emits per-failure + summary events."
        ),
    )
    p.add_argument(
        "--external-log-noise",
        choices=["auto", "suppress", "verbose"],
        default="auto",
        help=(
            "Control suppression of noisy third-party logs. "
            "'auto' suppresses unless --log-level DEBUG."
        ),
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
        help="List scenario IDs from a scenario matrix YAML or bundle:<name>",
    )
    p2.add_argument("--matrix", required=True, help="Path to scenario matrix YAML or bundle:<name>")
    p2.set_defaults(cmd="list-scenarios")

    p3 = subparsers.add_parser(
        "validate-config",
        help="Validate a scenario matrix YAML or bundle:<name> for required fields and duplicates",
    )
    p3.add_argument("--matrix", required=True, help="Path to scenario matrix YAML or bundle:<name>")
    # optional: later we could add --verbose to print detailed schema errors
    p3.set_defaults(cmd="validate-config")

    p4 = subparsers.add_parser(
        "preview-scenarios",
        help="Preview scenarios with warn-only plausibility checks",
    )
    p4.add_argument("--matrix", required=True, help="Path to scenario matrix YAML or bundle:<name>")
    p4.set_defaults(cmd="preview-scenarios")


def _add_planner_inclusion_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the planner inclusion-check subcommand parser."""
    p = subparsers.add_parser(
        "planner-inclusion-check",
        help="Run one planner through the promotion inclusion-review gate",
    )
    p.add_argument("--algo", required=True, help="Planner algorithm key to evaluate")
    p.add_argument("--algo-config", default=None, help="Optional planner config YAML")
    p.add_argument(
        "--matrix",
        default=str(DEFAULT_INCLUSION_MATRIX),
        help="Reference scenario matrix YAML",
    )
    p.add_argument("--schema", default=DEFAULT_SCHEMA_PATH, help="Episode schema path")
    p.add_argument(
        "--output-dir",
        default="output/planner_inclusion",
        help="Directory for episodes JSONL and inclusion report",
    )
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--horizon", type=int, default=250)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--record-forces", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument(
        "--benchmark-profile",
        choices=["baseline-safe", "paper-baseline", "experimental"],
        default="experimental",
        help="Planner readiness profile used for the run",
    )
    p.add_argument(
        "--socnav-missing-prereq-policy",
        choices=["fail-fast", "skip-with-warning", "fallback"],
        default="fail-fast",
    )
    p.add_argument("--min-episodes", type=int, default=1)
    p.add_argument("--min-success-rate", type=float, default=0.5)
    p.add_argument("--max-collision-rate", type=float, default=0.0)
    p.add_argument("--max-runtime-sec", type=float, default=60.0)
    p.set_defaults(cmd="planner-inclusion-check")


def _add_doctor_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the local runtime doctor subcommand parser."""
    p = subparsers.add_parser(
        "doctor",
        help="Report local runtime diagnostics for setup and issue triage",
    )
    p.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("output"),
        help="Artifact root to probe for temporary write access (default: output)",
    )
    p.add_argument(
        "--skip-env-smoke",
        action="store_true",
        default=False,
        help="Skip the minimal reset/step environment smoke check",
    )
    p.set_defaults(cmd="doctor")


def _add_mapf_oracle_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the MAPF oracle diagnostics subcommand parser."""
    p = subparsers.add_parser(
        "mapf-oracle",
        help="Run MAPF oracle route-feasibility diagnostics on a scenario matrix",
        description=(
            "Run the MAPF oracle diagnostic (A* static-route feasibility) on each "
            "scenario in a matrix.  Diagnostic-only, not benchmark evidence."
        ),
    )
    p.add_argument("matrix", help="Scenario matrix YAML path")
    p.add_argument(
        "--grid-size",
        type=int,
        default=40,
        help="Grid resolution (grid_size x grid_size). Default: 40.",
    )
    p.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Substring filter on scenario names.",
    )
    p.set_defaults(cmd="mapf-oracle")


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
        default=DEFAULT_REPORT_GROUP_BY,
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
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
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default="strict",
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed; "
            "diagnostic-cross-track namespaces groups by track with caveats."
        ),
    )
    p.add_argument(
        "--snqi-weights",
        type=str,
        default=None,
        help="Optional SNQI weights JSON to compute metrics.snqi during aggregation "
        "(fills missing SNQI only; pass --recompute-snqi to override stored values)",
    )
    p.add_argument(
        "--snqi-baseline",
        type=str,
        default=None,
        help="Optional baseline stats JSON used for SNQI normalization",
    )
    p.add_argument(
        "--recompute-snqi",
        action="store_true",
        help="Recompute metrics.snqi from --snqi-weights even when episodes already "
        "contain a stored SNQI value (default: only fill missing SNQI).",
    )
    p.set_defaults(cmd="aggregate")


def _add_metric_layers_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register metric-layers subcommand parser."""
    p = subparsers.add_parser(
        "metric-layers",
        help="Build canonical metric-layer summary from episode JSONL records.",
    )
    p.add_argument("--episodes", required=True, help="Input episodes JSONL path")
    p.add_argument("--output", required=True, help="Output metric-layer JSON path")
    p.add_argument(
        "--group-by",
        default="scenario_params.algo",
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default="algo",
        help="Fallback grouping key when --group-by is missing. Default: algo",
    )
    p.set_defaults(cmd="metric-layers")


def _add_stress_coverage_report_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the stress-coverage-report subcommand parser."""
    p = subparsers.add_parser(
        "stress-coverage-report",
        help="Build or validate a stress_uncertainty_coverage.v1 report.",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--episodes-jsonl",
        nargs="+",
        help="Input episode JSONL path(s) used to build a v1 report.",
    )
    source.add_argument(
        "--summary-json",
        help="Existing v1 or legacy aggregate summary JSON to validate/normalize.",
    )
    p.add_argument("--out", required=True, help="Output report JSON path")
    p.add_argument("--report-id", default="stress-coverage-report")
    p.add_argument("--campaign-config-hash", default="unknown")
    p.add_argument("--scenario-matrix-hash", default="unknown")
    p.add_argument(
        "--schema-mode",
        choices=("required", "advisory", "diagnostic"),
        default="required",
    )
    p.add_argument(
        "--aggregate-mode",
        choices=("mean", "median", "descriptive_only"),
        default="mean",
    )
    p.add_argument(
        "--availability-status",
        choices=("available", "partial-failure", "failed", "not_available"),
        default="available",
    )
    p.add_argument("--bootstrap-samples", type=int, default=0)
    p.add_argument("--bootstrap-confidence", type=float, default=0.95)
    p.add_argument("--bootstrap-seed", type=int, default=None)
    p.set_defaults(cmd="stress-coverage-report")


def _add_classify_failure_mechanisms_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the classify-failure-mechanisms subcommand parser."""
    p = subparsers.add_parser(
        "classify-failure-mechanisms",
        help="Classify paired fixed/long-horizon failure mechanisms from episode JSONL.",
    )
    p.add_argument(
        "--episodes-jsonl",
        required=True,
        nargs="+",
        help="Input paired episode JSONL path(s).",
    )
    p.add_argument(
        "--scenario-certificates",
        default=None,
        help="Optional scenario_cert.v1 JSON or JSONL used to block planner attribution.",
    )
    p.add_argument("--out-json", required=True, help="Output classifier JSON path.")
    p.add_argument("--out-csv", required=True, help="Output classifier CSV path.")
    p.add_argument("--fixed-horizon", type=int, default=100)
    p.add_argument("--long-horizon", type=int, default=500)
    p.set_defaults(cmd="classify-failure-mechanisms")


def _add_collision_scenario_similarity_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register collision-scenario-similarity subcommand parser."""
    p = subparsers.add_parser(
        "collision-scenario-similarity",
        help="Build nearest-neighbor groups for collision and near-collision episodes.",
    )
    p.add_argument("--episodes-jsonl", required=True, help="Input benchmark episode JSONL path.")
    p.add_argument("--out-json", required=True, help="Output similarity report JSON path.")
    p.add_argument("--out-markdown", default=None, help="Optional Markdown inspection report path.")
    p.add_argument(
        "--nearest-k", type=int, default=3, help="Neighbors to list per selected record."
    )
    p.add_argument(
        "--group-threshold",
        type=float,
        default=0.35,
        help="Maximum normalized distance linking records into a group.",
    )
    p.add_argument(
        "--collision-threshold",
        type=float,
        default=1.0,
        help="Minimum collisions for selection.",
    )
    p.add_argument(
        "--near-miss-threshold",
        type=float,
        default=0.0,
        help="Strict lower bound for near_misses selection.",
    )
    p.add_argument(
        "--comfort-threshold",
        type=float,
        default=0.2,
        help="Minimum comfort_exposure for selection.",
    )
    p.add_argument(
        "--require-trajectory-comparison",
        action="store_true",
        help=(
            "Fail unless at least two selected records contain raw robot and pedestrian "
            "trajectory arrays."
        ),
    )
    p.set_defaults(cmd="collision-scenario-similarity")


def _add_validate_row_claims_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the validate-row-claims subcommand parser."""
    p = subparsers.add_parser(
        "validate-row-claims",
        help="Validate BenchmarkRowClaim.v1 records in leaderboard sidecar files",
    )
    p.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help="Path to a single leaderboard .rows.json sidecar file",
    )
    p.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Validate every docs/leaderboards/*.rows.json sidecar",
    )
    p.set_defaults(cmd="validate-row-claims")


def _add_claim_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the benchmark claim subcommand parser."""
    p = subparsers.add_parser(
        "claim",
        help="Generate a schema-checked BenchmarkClaim JSON artifact",
    )
    p.add_argument("--claim-id", required=True, help="Stable claim identifier")
    p.add_argument("--statement", required=True, help="Human-readable benchmark statement")
    p.add_argument(
        "--scenario-matrix",
        required=True,
        help="Frozen scenario matrix file used by the claim",
    )
    p.add_argument(
        "--scenario-matrix-sha256",
        required=True,
        help="Expected SHA-256 digest for --scenario-matrix",
    )
    p.add_argument(
        "--policy-metadata",
        required=True,
        help="JSON policy metadata with schema_version and policy SHA-256 values",
    )
    p.add_argument(
        "--training-episodes",
        nargs="*",
        default=[],
        help="Optional training episode JSONL artifacts, distinct from claim evidence",
    )
    p.add_argument(
        "--validation-episodes",
        nargs="*",
        default=[],
        help="Optional validation episode JSONL artifacts, distinct from final evidence",
    )
    p.add_argument(
        "--final-benchmark-episodes",
        nargs="+",
        required=True,
        help="Final benchmark episode JSONL artifacts that support the claim",
    )
    p.add_argument(
        "--aggregate-report",
        action="append",
        default=[],
        help="Optional schema/version-tagged aggregate or statistical report JSON",
    )
    p.add_argument(
        "--dependency-group",
        default="dev",
        help="Dependency group/profile used to create the benchmark environment",
    )
    p.add_argument(
        "--container-image-digest",
        default=None,
        help="Optional container image digest used for the benchmark environment",
    )
    p.add_argument("--output-json", required=True, help="Output claim JSON path")
    p.set_defaults(cmd="claim")


def _add_export_parquet_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the export-parquet subcommand parser."""
    p = subparsers.add_parser(
        "export-parquet",
        help="Convert benchmark episode JSONL into Parquet analytics tables",
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out-dir", required=True, help="Output directory for Parquet tables")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing export files in the output directory",
    )
    p.set_defaults(cmd="export-parquet")


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
        default=DEFAULT_REPORT_GROUP_BY,
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument("--format", choices=["md", "csv", "json"], default="md")
    p.add_argument("--top", type=int, default=None, help="Limit to top-N groups by base ranking")
    p.add_argument(
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default="strict",
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed; "
            "diagnostic-cross-track namespaces groups by track with caveats."
        ),
    )
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
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
        help="Grouping key (dotted path). Default: scenario_id",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument(
        "--metrics",
        default=None,
        help="Optional comma-separated list of metric names to include (default: all)",
    )
    p.add_argument(
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default="strict",
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed; "
            "diagnostic-cross-track namespaces groups by track with caveats."
        ),
    )
    p.set_defaults(cmd="seed-variance")


def _add_flakiness_audit_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the flakiness-audit subcommand parser."""
    p = subparsers.add_parser(
        "flakiness-audit",
        help=(
            "Audit scenario outcome flakiness: exact-repeat determinism and "
            "per-cell (scenario, planner) outcome-stability; writes a JSON report"
        ),
    )
    p.add_argument("--in", dest="in_path", required=True, help="Input episodes JSONL path")
    p.add_argument("--out", required=True, help="Output JSON report path")
    p.add_argument(
        "--outcome-metric",
        default="success",
        help="Binary outcome metric name (default: success)",
    )
    p.add_argument(
        "--group-by",
        default=DEFAULT_REPORT_GROUP_BY,
        help="Planner grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument(
        "--seed-field",
        default="seed",
        help="Dotted path to the seed field (default: seed)",
    )
    p.add_argument(
        "--stability-threshold",
        type=float,
        default=0.8,
        help=(
            "Majority-agreement fraction below which a cell is flagged knife-edge "
            "(range (0, 1]; default 0.8)"
        ),
    )
    p.add_argument(
        "--min-seeds",
        type=int,
        default=2,
        help="Minimum distinct seeds before a cell's stability is assessed (default: 2)",
    )
    p.add_argument(
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default=_DEFAULT_FLAKINESS_TRACK_MODE,
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed "
            "rather than pooling incompatible observation contracts; "
            "diagnostic-cross-track partitions cells per track with an explicit caveat."
        ),
    )
    p.set_defaults(cmd="flakiness-audit")


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
        default=DEFAULT_REPORT_GROUP_BY,
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument("--metric", default="collisions", help="Metric name under metrics.<name>")
    sort = p.add_mutually_exclusive_group()
    sort.add_argument("--ascending", action="store_true", default=True)
    sort.add_argument("--descending", dest="ascending", action="store_false")
    p.add_argument("--top", type=int, default=None, help="Limit to top N rows")
    p.add_argument(
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default="strict",
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed; "
            "diagnostic-cross-track namespaces groups by track with caveats."
        ),
    )
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
        default=DEFAULT_REPORT_GROUP_BY,
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
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
    p.add_argument(
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default="strict",
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed; "
            "diagnostic-cross-track namespaces rows by track with caveats."
        ),
    )
    p.set_defaults(cmd="table")


def _add_export_canonical_table_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the canonical table export subcommand parser."""
    p = subparsers.add_parser(
        "export-canonical-table",
        help="Export a named canonical benchmark table as csv/md/tex with metadata",
    )
    p.add_argument(
        "--table-id",
        required=True,
        choices=sorted(_canonical_table_specs),
        help="Canonical table contract to export",
    )
    p.add_argument("--rows", required=True, help="JSON list of row mappings")
    p.add_argument("--out-dir", required=True, help="Directory for table outputs and metadata")
    p.add_argument(
        "--formats",
        default="csv,md,tex",
        help="Comma-separated output formats. Supported: csv, md, tex",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal places for floating point cells",
    )
    p.add_argument(
        "--source",
        action="append",
        default=[],
        help="Source file to checksum in the metadata sidecar; may be repeated",
    )
    p.set_defaults(cmd="export-canonical-table")


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
        default=DEFAULT_REPORT_GROUP_BY,
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
        help="Fallback grouping key when group-by is missing. Default: scenario_id",
    )
    p.add_argument("--agg", choices=["mean", "median"], default="mean")
    p.add_argument("--x-higher-better", action="store_true", default=False)
    p.add_argument("--y-higher-better", action="store_true", default=False)
    p.add_argument("--title", default=None)
    p.add_argument("--out-pdf", default=None, help="Optional path to also export a vector PDF")
    p.add_argument(
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default="strict",
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed; "
            "diagnostic-cross-track namespaces groups by track with caveats."
        ),
    )
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
        default=DEFAULT_REPORT_GROUP_BY,
        help="Grouping key (dotted path). Default: scenario_params.algo",
    )
    p.add_argument(
        "--fallback-group-by",
        default=DEFAULT_REPORT_FALLBACK_GROUP_BY,
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
    p.add_argument(
        "--observation-track-mode",
        choices=["strict", "diagnostic-cross-track"],
        default="strict",
        help=(
            "How to handle mixed benchmark_track values. Default strict fails closed; "
            "diagnostic-cross-track namespaces groups by track with caveats."
        ),
    )
    p.set_defaults(cmd="plot-distributions")


def _add_plot_planner_tradeoff_subparser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the plot-planner-tradeoff subcommand parser."""
    p = subparsers.add_parser(
        "plot-planner-tradeoff",
        help="Plot planner collision-rate versus success-rate from a publication bundle",
    )
    p.add_argument(
        "--bundle-path",
        required=True,
        help="Publication bundle root containing payload/reports and payload/runs.",
    )
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--out-pdf", default=None, help="Optional path to also export vector PDF")
    p.add_argument(
        "--metadata-out",
        default=None,
        help="Optional JSON metadata path with plotted points and CI values",
    )
    p.add_argument(
        "--bootstrap-samples",
        type=int,
        default=400,
        help="Bootstrap resamples over seed-level means. Default: 400",
    )
    p.add_argument(
        "--ci-confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level. Default: 0.95",
    )
    p.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Seed for deterministic bootstrap resampling. Default: 42",
    )
    p.add_argument(
        "--title",
        default="Preferred region: lower collision, higher success",
        help="Optional plot title; pass an empty string for no title.",
    )
    p.set_defaults(cmd="plot-planner-tradeoff")


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


def _attach_core_subcommands(parser: argparse.ArgumentParser) -> None:  # noqa: PLR0915
    """Attach core benchmark CLI subcommands."""
    subparsers = parser.add_subparsers(dest="cmd")
    _add_baseline_subparser(subparsers)
    _add_run_subparser(subparsers)
    _add_summary_subparser(subparsers)
    _add_aggregate_subparser(subparsers)
    _add_metric_layers_subparser(subparsers)
    _add_stress_coverage_report_subparser(subparsers)
    _add_classify_failure_mechanisms_subparser(subparsers)
    _add_collision_scenario_similarity_subparser(subparsers)
    _add_claim_subparser(subparsers)
    _add_validate_row_claims_subparser(subparsers)
    _add_export_parquet_subparser(subparsers)
    _add_seed_variance_subparser(subparsers)
    _add_flakiness_audit_subparser(subparsers)
    _add_extract_failures_subparser(subparsers)
    _add_snqi_ablate_subparser(subparsers)
    _add_rank_subparser(subparsers)
    _add_table_subparser(subparsers)
    _add_export_canonical_table_subparser(subparsers)
    _add_debug_seeds_subparser(subparsers)
    _add_plot_pareto_subparser(subparsers)
    _add_plot_distributions_subparser(subparsers)
    _add_plot_planner_tradeoff_subparser(subparsers)
    _add_plot_scenarios_subparser(subparsers)
    _add_list_subparser(subparsers)
    _add_planner_inclusion_subparser(subparsers)
    _add_doctor_subparser(subparsers)
    _add_mapf_oracle_subparser(subparsers)
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
            "--decision-preflight",
            action="store_true",
            help="Enable fail-closed preflight for missing or invalid normalized inputs.",
        )
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
        p = sp.add_parser(
            "recompute",
            help="Recompute SNQI weights via predefined strategies",
            conflict_handler="resolve",
        )
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
            "--export-pareto-front",
            action="store_true",
            help="Export sampled Pareto frontier when strategy pareto is active.",
        )
        p.add_argument(
            "--pareto-front-samples",
            type=int,
            default=600,
            help="Number of Pareto frontier samples to draw when export is enabled.",
        )
        p.add_argument(
            "--missing-metric-max-list",
            type=int,
            default=5,
            help="Max example episode IDs per missing baseline metric",
        )
        p.add_argument("--fail-on-missing-metric", action="store_true")
        p.add_argument(
            "--decision-preflight",
            action="store_true",
            help="Enable fail-closed preflight for missing or invalid normalized inputs.",
        )
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
        p.add_argument(
            "--export-pareto-front",
            action="store_true",
            help="Export sampled Pareto frontier when strategy pareto is active.",
        )
        p.add_argument(
            "--pareto-front-samples",
            type=int,
            default=600,
            help="Number Pareto frontier samples draw when export enabled.",
        )
        p.add_argument(
            "--decision-preflight",
            action="store_true",
            help="Enable fail-closed preflight missing/invalid normalized inputs.",
        )
        p.add_argument(
            "--decision-reversal-threshold",
            type=float,
            default=0.0,
            help="If >0 compare-strategies, flag correlation pairs below value.",
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
            except (OSError, ImportError):  # pragma: no cover - load error
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
            except (OSError, ImportError):  # pragma: no cover - load error
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
                except (TypeError, ValueError):  # pragma: no cover - fallback safety
                    pass
            try:  # pragma: no cover (normal success path still covered elsewhere)
                return parser.parse_intermixed_args(args, namespace)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
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
            except (TypeError, ValueError):
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
    effective_argv = list(sys.argv if argv is None else ["robot_sf_bench", *argv])
    args._canonical_command = shlex.join(effective_argv)
    _configure_logging(getattr(args, "quiet", False), getattr(args, "log_level", "INFO"))
    # macOS safe start method for multiprocessing
    if getattr(args, "workers", 1) and int(getattr(args, "workers", 1)) > 1:
        try:
            multiprocessing_module = importlib.import_module("multiprocessing")
            multiprocessing_module.set_start_method("spawn", force=False)
        except (ImportError, RuntimeError):  # pragma: no cover - platform-specific boundary
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
        "preview-scenarios": _handle_preview_scenarios,
        "planner-inclusion-check": _handle_planner_inclusion_check,
        "run": _handle_run,
        "summary": _handle_summary,
        "aggregate": _handle_aggregate,
        "metric-layers": _handle_metric_layers,
        "stress-coverage-report": _handle_stress_coverage_report,
        "classify-failure-mechanisms": _handle_classify_failure_mechanisms,
        "collision-scenario-similarity": _handle_collision_scenario_similarity,
        "claim": _handle_claim,
        "validate-row-claims": _handle_validate_row_claims,
        "export-parquet": _handle_export_parquet,
        "seed-variance": _handle_seed_variance,
        "flakiness-audit": _handle_flakiness_audit,
        "extract-failures": _handle_extract_failures,
        "snqi-ablate": _handle_snqi_ablate,
        "rank": _handle_rank,
        "table": _handle_table,
        "export-canonical-table": _handle_export_canonical_table,
        "debug-seeds": _handle_debug_seeds,
        "plot-pareto": _handle_plot_pareto,
        "plot-distributions": _handle_plot_distributions,
        "plot-planner-tradeoff": _handle_plot_planner_tradeoff,
        "plot-scenarios": _handle_plot_scenarios,
        "doctor": _handle_doctor,
        "mapf-oracle": _handle_mapf_oracle,
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
