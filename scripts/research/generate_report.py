#!/usr/bin/env python3
"""
CLI script for generating research reports from imitation learning experiments.
Usage: scripts/research/generate_report.py --tracker-run <run_id> --experiment-name "My Experiment"
"""

import argparse
import sys
from pathlib import Path

from loguru import logger


def load_tracker_manifest(tracker_run_id: str) -> dict:
    """Load tracker manifest from run ID (supports jsonl + json)."""

    from robot_sf.research.exceptions import ValidationError
    from robot_sf.research.metadata import load_tracker_manifest_payload

    base_dir = Path("output/run-tracker") / tracker_run_id
    json_path = base_dir / "manifest.json"
    jsonl_path = base_dir / "manifest.jsonl"

    manifest_path = jsonl_path if jsonl_path.exists() else json_path
    if manifest_path.exists():
        try:
            return load_tracker_manifest_payload(manifest_path)
        except ValidationError as exc:
            logger.error(f"Tracker manifest validation failed: {exc}")
            raise SystemExit(1) from exc

    logger.error(f"Tracker manifest not found: {jsonl_path} or {json_path}")
    sys.exit(1)


def extract_metric_records_from_manifest(manifest: dict) -> tuple[list[dict], list[int]]:
    """Extract per-seed metric records from tracker manifest (best effort, no fabrication)."""

    from robot_sf.research.tracker_manifest import coerce_tracker_int

    summary = manifest.get("summary") or {}
    seeds: list[int] = []
    if isinstance(summary.get("seeds"), list):
        seeds = [coerce_tracker_int(seed, "summary.seeds") for seed in summary["seeds"]]

    metrics = summary.get("metrics") or {}
    records: list[dict] = []
    if metrics:
        # If aggregated metrics are present, keep them at the aggregated level to avoid inventing per-seed data.
        logger.warning(
            "Tracker manifest contains aggregated metrics only; per-seed records unavailable."
        )
    else:
        logger.warning(
            "No metrics found in tracker manifest; report will mark metrics as incomplete instead of fabricating data."
        )
    return records, seeds


def _coerce_summary_float_list(summary: dict, field: str) -> list[float]:
    """Coerce an optional summary sequence to finite floats."""
    from robot_sf.research.exceptions import ValidationError
    from robot_sf.research.tracker_manifest import coerce_tracker_float

    values = summary.get(field)
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValidationError(f"Tracker manifest summary {field} must be a list")
    return [
        coerce_tracker_float(value, f"summary.{field}") for value in values if value is not None
    ]


def _extract_report_inputs(
    manifest: dict,
) -> tuple[dict, list[dict], list[int], list[float], list[float]]:
    """Extract validated summary inputs used by report generation."""
    from robot_sf.research.exceptions import ValidationError

    summary = manifest.get("summary") or {}
    try:
        metric_records, seeds = extract_metric_records_from_manifest(manifest)
        seeds = sorted(seeds)
        baseline_timesteps = _coerce_summary_float_list(summary, "baseline_timesteps")
        pretrained_timesteps = _coerce_summary_float_list(summary, "pretrained_timesteps")
    except ValidationError as exc:
        logger.error(f"Tracker manifest numeric validation failed: {exc}")
        raise SystemExit(1) from exc
    return summary, metric_records, seeds, baseline_timesteps, pretrained_timesteps


def _comparison_metric_records(metrics_comp: object) -> tuple[list[dict], list[int]]:
    """Coerce comparison metrics into the minimal fallback record shape."""
    from robot_sf.research.exceptions import ValidationError
    from robot_sf.research.tracker_manifest import coerce_tracker_float

    if not isinstance(metrics_comp, dict):
        raise ValidationError("Tracker manifest comparison metrics must be an object")
    records: list[dict] = []
    for metric_name, values in metrics_comp.items():
        if not isinstance(values, dict):
            raise ValidationError("Tracker manifest comparison metric values must be objects")
        if "baseline" not in values or "pretrained" not in values:
            raise ValidationError(
                "Tracker manifest comparison metrics require baseline and pretrained values"
            )
        baseline_value = coerce_tracker_float(
            values["baseline"], f"comparison.metrics.{metric_name}.baseline"
        )
        pretrained_value = coerce_tracker_float(
            values["pretrained"], f"comparison.metrics.{metric_name}.pretrained"
        )
        records.extend(
            [
                {"seed": 0, "policy_type": "baseline", metric_name: baseline_value},
                {"seed": 1, "policy_type": "pretrained", metric_name: pretrained_value},
            ]
        )
    return records, [0, 1] if records else []


def _apply_comparison_fallback(
    summary: dict,
    metric_records: list[dict],
    seeds: list[int],
    baseline_timesteps: list[float],
    pretrained_timesteps: list[float],
) -> tuple[list[dict], list[int], list[float], list[float]]:
    """Apply validated comparison-summary fallback values without fabrication."""
    from robot_sf.research.exceptions import ValidationError
    from robot_sf.research.tracker_manifest import coerce_tracker_float

    if metric_records:
        return metric_records, seeds, baseline_timesteps, pretrained_timesteps

    try:
        comparison = summary.get("comparison") or {}
        if not isinstance(comparison, dict):
            raise ValidationError("Tracker manifest comparison must be an object")
        metrics_comp = comparison.get("metrics_comparison") or {}
        comparison_records, comparison_seeds = _comparison_metric_records(metrics_comp)
        if comparison_records:
            metric_records = comparison_records
            if not seeds:
                seeds = comparison_seeds
        if not baseline_timesteps and comparison.get("timesteps_to_convergence"):
            ts = comparison["timesteps_to_convergence"]
            if not isinstance(ts, dict):
                raise ValidationError("Tracker manifest comparison timesteps must be an object")
            baseline = ts.get("baseline")
            pretrained = ts.get("pretrained")
            baseline_timesteps = (
                [coerce_tracker_float(baseline, "comparison.timesteps.baseline")]
                if baseline is not None
                else []
            )
            pretrained_timesteps = (
                [coerce_tracker_float(pretrained, "comparison.timesteps.pretrained")]
                if pretrained is not None
                else []
            )
    except ValidationError as exc:
        logger.error(f"Tracker manifest comparison validation failed: {exc}")
        raise SystemExit(1) from exc

    return metric_records, seeds, baseline_timesteps, pretrained_timesteps


def main() -> None:
    """CLI entry point for report generation."""
    parser = argparse.ArgumentParser(
        description="Generate research report from imitation learning experiments."
    )

    parser.add_argument(
        "--tracker-run",
        type=str,
        required=True,
        help="Tracker run ID (directory name in output/run-tracker/)",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Human-readable experiment name for report title",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for report (default: output/research_reports/<timestamp>_<experiment_name>)",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="Hypothesis threshold percentage (default: 40.0)",
    )

    args = parser.parse_args()

    # Keep ``--help`` and argument validation usable in the core installation. The
    # report implementation imports optional analytics dependencies, so defer it
    # until after argparse has handled lightweight CLI paths.
    from robot_sf.research.orchestrator import ReportOrchestrator

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = args.experiment_name.lower().replace(" ", "_")
        output_dir = Path("output/research_reports") / f"{timestamp}_{safe_name}"

    logger.info(f"Generating report for tracker run: {args.tracker_run}")
    logger.info(f"Output directory: {output_dir}")

    # Load tracker manifest
    manifest = load_tracker_manifest(args.tracker_run)
    run_id = manifest.get("run_id", args.tracker_run)

    summary, metric_records, seeds, baseline_timesteps, pretrained_timesteps = (
        _extract_report_inputs(manifest)
    )

    metric_records, seeds, baseline_timesteps, pretrained_timesteps = _apply_comparison_fallback(
        summary, metric_records, seeds, baseline_timesteps, pretrained_timesteps
    )

    # No synthetic reward curves; skip figure generation unless metrics are present
    baseline_rewards: list[list[float]] | None = None
    pretrained_rewards: list[list[float]] | None = None

    # Generate report
    orchestrator = ReportOrchestrator(output_dir)
    report_path = orchestrator.generate_report(
        experiment_name=args.experiment_name,
        metric_records=metric_records,
        run_id=run_id,
        seeds=seeds,
        baseline_timesteps=baseline_timesteps,
        pretrained_timesteps=pretrained_timesteps,
        baseline_rewards=baseline_rewards,
        pretrained_rewards=pretrained_rewards,
        threshold=args.threshold,
    )

    logger.info(f"Report generated successfully: {report_path}")
    print(f"\nReport available at: {report_path}")


if __name__ == "__main__":
    main()
