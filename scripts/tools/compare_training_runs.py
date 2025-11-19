"""Comparison tool for analyzing training runs and computing sample-efficiency metrics.

Loads training run manifests, compares baseline vs pre-trained PPO performance,
and generates comprehensive comparison reports including sample-efficiency ratios
and convergence timings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.benchmark.imitation_manifest import get_training_run_manifest_path
from robot_sf.common.artifact_paths import get_imitation_report_dir

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any


def _load_training_run(run_id: str) -> dict[str, Any]:
    """Load training run manifest from disk."""
    manifest_path = get_training_run_manifest_path(run_id)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training run manifest not found: {manifest_path}")

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _extract_convergence_timesteps(run_data: dict[str, Any]) -> int:
    """Extract convergence timesteps from training run notes."""
    # Look for convergence info in notes
    for note in run_data.get("notes", []):
        if "Converged at" in note:
            # Parse "Converged at X timesteps"
            parts = note.split()
            if len(parts) >= 3:
                try:
                    return int(parts[2])
                except (ValueError, IndexError):
                    pass

    # Fallback: assume full training timesteps
    return run_data.get("total_timesteps", 0)


def _compute_sample_efficiency_ratio(
    baseline_timesteps: int,
    pretrained_timesteps: int,
) -> float:
    """Compute sample-efficiency ratio (lower is better for pretrained)."""
    if baseline_timesteps == 0:
        return 1.0
    return pretrained_timesteps / baseline_timesteps


def _compare_metrics(
    baseline_metrics: dict[str, Any],
    pretrained_metrics: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compare metrics between baseline and pretrained runs."""
    comparison = {}

    all_keys = set(baseline_metrics.keys()) | set(pretrained_metrics.keys())

    for key in all_keys:
        baseline_val = baseline_metrics.get(key, {})
        pretrained_val = pretrained_metrics.get(key, {})

        # Extract mean values for comparison
        baseline_mean = baseline_val.get("mean", 0.0) if isinstance(baseline_val, dict) else 0.0
        pretrained_mean = (
            pretrained_val.get("mean", 0.0) if isinstance(pretrained_val, dict) else 0.0
        )

        comparison[key] = {
            "baseline": float(baseline_mean),
            "pretrained": float(pretrained_mean),
            "improvement": float(pretrained_mean - baseline_mean),
        }

    return comparison


def generate_comparison_report(
    run_group_id: str,
    baseline_run_id: str,
    pretrained_run_id: str,
) -> dict[str, Any]:
    """Generate comprehensive comparison report between training runs."""
    logger.info("Generating comparison report for group {}", run_group_id)

    # Load training runs
    baseline_data = _load_training_run(baseline_run_id)
    pretrained_data = _load_training_run(pretrained_run_id)

    # Extract convergence timesteps
    baseline_timesteps = _extract_convergence_timesteps(baseline_data)
    pretrained_timesteps = _extract_convergence_timesteps(pretrained_data)

    # Compute sample-efficiency ratio
    efficiency_ratio = _compute_sample_efficiency_ratio(
        baseline_timesteps,
        pretrained_timesteps,
    )

    # Compare metrics
    metrics_comparison = _compare_metrics(
        baseline_data.get("metrics", {}),
        pretrained_data.get("metrics", {}),
    )

    # Build report
    report = {
        "run_group_id": run_group_id,
        "baseline_run_id": baseline_run_id,
        "pretrained_run_id": pretrained_run_id,
        "sample_efficiency_ratio": efficiency_ratio,
        "timesteps_to_convergence": {
            "baseline": baseline_timesteps,
            "pretrained": pretrained_timesteps,
            "reduction_timesteps": baseline_timesteps - pretrained_timesteps,
            "reduction_percentage": (
                100.0 * (1.0 - efficiency_ratio) if baseline_timesteps > 0 else 0.0
            ),
        },
        "metrics_comparison": metrics_comparison,
        "meets_target": efficiency_ratio <= 0.70,  # Target from spec
    }

    return report


def save_comparison_report(report: dict[str, Any], output_path: Path) -> None:
    """Save comparison report to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    logger.info("Comparison report saved to {}", output_path)


def print_comparison_summary(report: dict[str, Any]) -> None:
    """Print human-readable summary of comparison report."""
    print("\n" + "=" * 70)
    print(f"Training Comparison Report: {report['run_group_id']}")
    print("=" * 70)

    print(f"\nBaseline Run: {report['baseline_run_id']}")
    print(f"Pretrained Run: {report['pretrained_run_id']}")

    conv = report["timesteps_to_convergence"]
    print("\nConvergence Timesteps:")
    print(f"  Baseline:   {conv['baseline']:,}")
    print(f"  Pretrained: {conv['pretrained']:,}")
    print(f"  Reduction:  {conv['reduction_timesteps']:,} ({conv['reduction_percentage']:.1f}%)")

    print(f"\nSample-Efficiency Ratio: {report['sample_efficiency_ratio']:.3f}")
    target_met = "✓ PASS" if report["meets_target"] else "✗ FAIL"
    print(f"Target (≤0.70): {target_met}")

    if report["metrics_comparison"]:
        print("\nMetrics Comparison:")
        for metric, values in report["metrics_comparison"].items():
            print(
                f"  {metric}: baseline={values['baseline']:.3f}, "
                f"pretrained={values['pretrained']:.3f}, "
                f"improvement={values['improvement']:+.3f}"
            )

    print("=" * 70 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for comparison tool."""
    parser = argparse.ArgumentParser(
        description="Compare baseline and pretrained PPO training runs."
    )
    parser.add_argument(
        "--group",
        required=True,
        help="Run group identifier",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline training run ID",
    )
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Pretrained training run ID",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for comparison report JSON (auto-generated if not specified)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress summary output (only save JSON)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for comparison tool."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Generate comparison report
    report = generate_comparison_report(
        run_group_id=args.group,
        baseline_run_id=args.baseline,
        pretrained_run_id=args.pretrained,
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        report_dir = get_imitation_report_dir() / "comparisons"
        output_path = report_dir / f"{args.group}_comparison.json"

    # Save report
    save_comparison_report(report, output_path)

    # Print summary unless quiet mode
    if not args.quiet:
        print_comparison_summary(report)

    # Exit with appropriate code
    return 0 if report["meets_target"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
