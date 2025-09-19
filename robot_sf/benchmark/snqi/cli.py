"""
SNQI CLI Tools

This module provides command-line interfaces for SNQI weight management:
- Weight recomputation from baseline statistics
- SNQI component ablation analysis
"""

import argparse
import json
import sys
from pathlib import Path

from robot_sf.benchmark.snqi.compute import compute_snqi_ablation, recompute_snqi_weights
from robot_sf.benchmark.snqi.types import SNQIWeights


def cmd_recompute_weights(args: argparse.Namespace) -> int:
    """Implement weight recompute CLI command."""
    try:
        # Load baseline statistics
        baseline_stats_path = Path(args.baseline_stats)
        if not baseline_stats_path.exists():
            print(f"ERROR: Baseline stats file not found: {baseline_stats_path}", file=sys.stderr)
            return 1

        with baseline_stats_path.open() as f:
            baseline_stats = json.load(f)

        # Recompute SNQI weights using baseline statistics
        weights = recompute_snqi_weights(
            baseline_stats=baseline_stats, method=args.method, seed=args.seed
        )

        # Save the recomputed weights
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        weights.save(output_path)

        print("✅ SNQI weights recomputed successfully")
        print(f"📊 Method: {args.method}")
        print(f"🌱 Seed: {args.seed}")
        print(f"💾 Saved to: {output_path}")

        # Print summary of computed weights
        print("\\n📋 Weight Summary:")
        for component, weight in weights.weights.items():
            print(f"  {component}: {weight:.3f}")

        return 0

    except Exception as e:
        print(f"ERROR: Failed to recompute SNQI weights: {e}", file=sys.stderr)
        return 1


def cmd_ablation_analysis(args: argparse.Namespace) -> int:
    """Implement SNQI ablation CLI command."""
    try:
        # Load episodes data
        episodes_path = Path(args.episodes)
        if not episodes_path.exists():
            print(f"ERROR: Episodes file not found: {episodes_path}", file=sys.stderr)
            return 1

        # Load SNQI weights if provided
        weights = None
        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.exists():
                print(f"ERROR: Weights file not found: {weights_path}", file=sys.stderr)
                return 1
            weights = SNQIWeights.from_file(weights_path)

        # Perform ablation analysis
        ablation_results = compute_snqi_ablation(
            episodes_path=episodes_path, weights=weights, components=args.components, seed=args.seed
        )

        # Save ablation results
        output_path = Path(args.summary_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            json.dump(ablation_results, f, indent=2)

        print("✅ SNQI ablation analysis completed")
        print(f"🧪 Components analyzed: {len(args.components) if args.components else 'all'}")
        print(f"🌱 Seed: {args.seed}")
        print(f"💾 Results saved to: {output_path}")

        # Print ranking sensitivity summary
        if "ranking_deltas" in ablation_results:
            print("\\n📊 Ranking Impact Summary:")
            for component, delta_info in ablation_results["ranking_deltas"].items():
                avg_delta = delta_info.get("avg_rank_change", 0)
                print(f"  {component}: {avg_delta:.2f} avg rank change")

        return 0

    except Exception as e:
        print(f"ERROR: Failed to perform SNQI ablation analysis: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for SNQI CLI tools."""
    parser = argparse.ArgumentParser(
        prog="robot_sf_snqi", description="SNQI weight management and analysis tools"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Recompute weights subcommand
    recompute_parser = subparsers.add_parser(
        "recompute", help="Recompute SNQI weights from baseline statistics"
    )
    recompute_parser.add_argument(
        "--baseline-stats", required=True, help="Path to baseline statistics JSON file"
    )
    recompute_parser.add_argument(
        "--out", required=True, help="Output path for recomputed weights JSON file"
    )
    recompute_parser.add_argument(
        "--method",
        default="pareto_optimization",
        choices=["pareto_optimization", "equal_weights", "safety_focused"],
        help="Weight computation method (default: pareto_optimization)",
    )
    recompute_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible weight computation (default: 42)",
    )

    # Ablation analysis subcommand
    ablation_parser = subparsers.add_parser(
        "ablation", help="Perform SNQI component ablation analysis"
    )
    ablation_parser.add_argument("--episodes", required=True, help="Path to episodes JSONL file")
    ablation_parser.add_argument(
        "--summary-out", required=True, help="Output path for ablation results JSON"
    )
    ablation_parser.add_argument(
        "--weights", help="Path to SNQI weights JSON file (optional, uses defaults if not provided)"
    )
    ablation_parser.add_argument(
        "--components",
        nargs="*",
        help="Specific SNQI components to analyze (default: all components)",
    )
    ablation_parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducible analysis (default: 123)"
    )

    return parser


def main() -> int:
    """Main entry point for SNQI CLI tools."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "recompute":
        return cmd_recompute_weights(args)
    elif args.command == "ablation":
        return cmd_ablation_analysis(args)
    else:
        print(f"ERROR: Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
