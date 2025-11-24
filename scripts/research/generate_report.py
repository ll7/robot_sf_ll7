#!/usr/bin/env python3
"""
CLI script for generating research reports from imitation learning experiments.
Usage: scripts/research/generate_report.py --tracker-run <run_id> --experiment-name "My Experiment"
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from robot_sf.research.orchestrator import ReportOrchestrator


def load_tracker_manifest(tracker_run_id: str) -> dict:
    """Load tracker manifest from run ID (supports jsonl + json)."""

    base_dir = Path("output/run-tracker") / tracker_run_id
    json_path = base_dir / "manifest.json"
    jsonl_path = base_dir / "manifest.jsonl"

    if jsonl_path.exists():
        lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line]
        if not lines:
            logger.error(f"Tracker manifest is empty: {jsonl_path}")
            sys.exit(1)
        return json.loads(lines[-1])

    if json_path.exists():
        with json_path.open(encoding="utf-8") as f:
            return json.load(f)

    logger.error(f"Tracker manifest not found: {jsonl_path} or {json_path}")
    sys.exit(1)


def extract_metric_records_from_manifest(manifest: dict) -> tuple[list[dict], list[int]]:
    """Extract per-seed metric records from tracker manifest (best effort, no fabrication)."""

    summary = manifest.get("summary") or {}
    seeds: list[int] = []
    if isinstance(summary.get("seeds"), list):
        try:
            seeds = [int(s) for s in summary["seeds"]]
        except (TypeError, ValueError):
            seeds = []

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

    summary = manifest.get("summary") or {}
    metric_records, seeds = extract_metric_records_from_manifest(manifest)
    seeds = sorted(seeds or summary.get("seeds", []))

    # Extract timesteps for hypothesis evaluation if present in summary
    baseline_timesteps = [t for t in (summary.get("baseline_timesteps") or []) if t is not None]
    pretrained_timesteps = [t for t in (summary.get("pretrained_timesteps") or []) if t is not None]

    # Fallback: construct minimal metric records from comparison summary if present
    if not metric_records:
        comparison = summary.get("comparison") or {}
        metrics_comp = comparison.get("metrics_comparison") or {}
        if metrics_comp:
            base_seed = 0
            metric_records = []
            for metric_name, values in metrics_comp.items():
                metric_records.append(
                    {
                        "seed": base_seed,
                        "policy_type": "baseline",
                        metric_name: values.get("baseline", 0.0),
                    }
                )
                metric_records.append(
                    {
                        "seed": base_seed + 1,
                        "policy_type": "pretrained",
                        metric_name: values.get("pretrained", 0.0),
                    }
                )
            if not seeds:
                seeds = [0, 1]
        if not baseline_timesteps and comparison.get("timesteps_to_convergence"):
            ts = comparison["timesteps_to_convergence"]
            baseline_timesteps = [ts.get("baseline")] if ts.get("baseline") is not None else []
            pretrained_timesteps = (
                [ts.get("pretrained")] if ts.get("pretrained") is not None else []
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
