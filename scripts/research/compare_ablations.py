"""CLI for ablation comparison (User Story 4) (T066).

Usage:
    uv run python scripts/research/compare_ablations.py \
        --config configs/research/example_ablation.yaml \
        --experiment-name BC_Ablation \
        --seeds 42 43 44 \
        --threshold 40.0 \
        --output output/research_reports/ablation_bc

Reads a YAML config with structure:
ablation_params:
  bc_epochs: [5,10,20]
  dataset_size: [100,200]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from robot_sf.research.orchestrator import AblationOrchestrator


def parse_args() -> argparse.Namespace:
    """Parse args.

    Returns:
        argparse.Namespace: Auto-generated placeholder description.
    """
    p = argparse.ArgumentParser(description="Run ablation matrix and generate report")
    p.add_argument("--config", required=True, type=Path, help="Path to ablation YAML config")
    p.add_argument("--experiment-name", required=True, help="Experiment name label")
    p.add_argument("--seeds", nargs="+", type=int, required=True, help="Random seeds list")
    # Escape % for argparse's internal %-formatting of help strings
    p.add_argument("--threshold", type=float, default=40.0, help="Improvement threshold (%%)")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output/research_reports/ablation"),
        help="Output directory for ablation report",
    )
    return p.parse_args()


def main() -> None:
    """Main.

    Returns:
        None: Auto-generated placeholder description.
    """
    args = parse_args()
    # Parse params first (use a temporary orchestrator for parser only)
    temp = AblationOrchestrator(
        experiment_name=args.experiment_name,
        seeds=args.seeds,
        ablation_params={},
        threshold=args.threshold,
        output_dir=args.output,
    )
    orch_params = temp.parse_ablation_config(args.config)
    orch = AblationOrchestrator(
        experiment_name=args.experiment_name,
        seeds=args.seeds,
        ablation_params=orch_params,
        threshold=args.threshold,
        output_dir=args.output,
    )
    logger.info(
        f"Running ablation for {args.experiment_name} with params: {orch_params} seeds={args.seeds} threshold={args.threshold}%"
    )
    variants = orch.run_ablation_matrix()
    variants = orch.handle_incomplete_variants(variants)
    report_path = orch.generate_ablation_report(variants)
    logger.info(f"Ablation report written to {report_path}")


if __name__ == "__main__":
    main()
