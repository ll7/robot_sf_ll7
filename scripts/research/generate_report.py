#!/usr/bin/env python3
"""
CLI script for generating research reports from imitation learning experiments.
Usage: scripts/research/generate_report.py --tracker-run <run_id> --experiment-name "My Experiment"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from robot_sf.research.orchestrator import ReportOrchestrator


def load_tracker_manifest(tracker_run_id: str) -> dict:
    """Load tracker manifest JSON from run ID."""
    manifest_path = Path("output/run-tracker") / tracker_run_id / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"Tracker manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def extract_metric_records_from_manifest(_manifest: dict) -> list[dict]:
    """Extract metric records from tracker manifest (placeholder implementation)."""
    # Placeholder: would parse telemetry.jsonl or steps data
    # For now, create synthetic data for demonstration
    logger.warning("Using synthetic metric records - integrate with actual tracker data")

    seeds = [42, 123, 456]
    metric_records = []

    for seed in seeds:
        # Baseline
        metric_records.append(
            {
                "seed": seed,
                "policy_type": "baseline",
                "success_rate": 0.70 + (seed % 10) * 0.01,
                "collision_rate": 0.15 + (seed % 10) * 0.01,
                "timesteps_to_convergence": 500000 + (seed % 10) * 10000,
                "final_reward_mean": 45.0 + (seed % 10) * 0.5,
                "run_duration_seconds": 3600.0,
            }
        )

        # Pretrained
        metric_records.append(
            {
                "seed": seed,
                "policy_type": "pretrained",
                "success_rate": 0.85 + (seed % 10) * 0.01,
                "collision_rate": 0.08 + (seed % 10) * 0.01,
                "timesteps_to_convergence": 280000 + (seed % 10) * 5000,
                "final_reward_mean": 52.0 + (seed % 10) * 0.5,
                "run_duration_seconds": 2100.0,
            }
        )

    return metric_records


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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = args.experiment_name.lower().replace(" ", "_")
        output_dir = Path("output/research_reports") / f"{timestamp}_{safe_name}"

    logger.info(f"Generating report for tracker run: {args.tracker_run}")
    logger.info(f"Output directory: {output_dir}")

    # Load tracker manifest
    manifest = load_tracker_manifest(args.tracker_run)
    run_id = manifest.get("run_id", args.tracker_run)

    # Extract metric records
    metric_records = extract_metric_records_from_manifest(manifest)

    # Extract seeds
    seeds = sorted({r["seed"] for r in metric_records})

    # Extract timesteps for hypothesis evaluation
    baseline_timesteps = [
        r["timesteps_to_convergence"] for r in metric_records if r["policy_type"] == "baseline"
    ]
    pretrained_timesteps = [
        r["timesteps_to_convergence"] for r in metric_records if r["policy_type"] == "pretrained"
    ]

    # Placeholder learning curves (would come from tracker telemetry)
    baseline_rewards = [[i * 0.1 for i in range(100)] for _ in seeds]
    pretrained_rewards = [[i * 0.15 for i in range(100)] for _ in seeds]

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
