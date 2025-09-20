"""
SNQI CLI Tools

This module provides command-line interfaces for SNQI weight management:
- Weight recomputation from baseline statistics
- SNQI component ablation analysis
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from robot_sf.benchmark.snqi.compute import (
    WEIGHT_NAMES,
    compute_snqi_ablation,
    recompute_snqi_weights,
)
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


def _load_episodes_jsonl(path: Path) -> List[dict]:
    episodes: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed
    return episodes


def _extract_metric_values(episodes: Iterable[dict], metric: str) -> List[float]:
    vals = []
    for ep in episodes:
        metrics = ep.get("metrics", {})
        v = metrics.get(metric)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    return vals


def _compute_baseline_stats(episodes: List[dict]) -> Dict[str, Dict[str, float]]:
    """Compute median & p95 for metrics required in SNQI normalization.

    Metrics chosen align with those referenced in compute_snqi(): collisions, near_misses,
    force_exceed_events, jerk_mean. Missing metrics default to med=0, p95=1 for neutral scaling.
    """
    metric_names = [
        "collisions",
        "near_misses",
        "force_exceed_events",
        "jerk_mean",
    ]
    stats: Dict[str, Dict[str, float]] = {}
    for name in metric_names:
        values = _extract_metric_values(episodes, name)
        if not values:
            stats[name] = {"med": 0.0, "p95": 1.0}
            continue
        values_sorted = sorted(values)
        med = statistics.median(values_sorted)
        # p95 index calculation
        idx = min(len(values_sorted) - 1, max(0, int(round(0.95 * (len(values_sorted) - 1)))))
        p95 = float(values_sorted[idx])
        if abs(p95 - med) < 1e-12:  # ensure non-zero span for normalization downstream
            p95 = med + 1.0
        stats[name] = {"med": float(med), "p95": float(p95)}
    return stats


def cmd_ablation_analysis(args: argparse.Namespace) -> int:
    """Implement SNQI ablation CLI command.

    This implementation loads episodes from JSONL, derives baseline statistics on-the-fly
    (unless future extension supplies an external stats file), and computes the impact on
    mean SNQI when zeroing each component weight individually.
    """
    try:
        episodes_path = Path(args.episodes)
        if not episodes_path.exists():
            print(f"ERROR: Episodes file not found: {episodes_path}", file=sys.stderr)
            return 1

        episodes = _load_episodes_jsonl(episodes_path)
        if not episodes:
            print("ERROR: No valid episode records found (empty episodes file)", file=sys.stderr)
            return 1

        # Determine weights
        weight_map: Dict[str, float]
        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.exists():
                print(f"ERROR: Weights file not found: {weights_path}", file=sys.stderr)
                return 1
            weight_obj = SNQIWeights.from_file(weights_path)
            weight_map = dict(weight_obj.weights)
        else:
            # Fallback canonical defaults (mirrors compute.recompute_snqi_weights canonical branch)
            weight_map = {
                "w_success": 1.0,
                "w_time": 0.8,
                "w_collisions": 2.0,
                "w_near": 1.0,
                "w_comfort": 0.5,
                "w_force_exceed": 1.5,
                "w_jerk": 0.3,
            }

        # Filter components if user provided subset
        target_components = args.components if args.components else list(WEIGHT_NAMES)

        baseline_stats = _compute_baseline_stats(episodes)

        # Run ablation
        impacts = compute_snqi_ablation(
            episodes_data=episodes,
            weights=weight_map,
            baseline_stats=baseline_stats,
            components=target_components,
        )

        # Compose output structure (extensible for future ranking deltas)
        result_payload = {
            "impacts": impacts,
            "baseline_stats": baseline_stats,
            "weights_used": weight_map,
            "components": target_components,
            "episode_count": len(episodes),
            "seed": args.seed,
        }

        output_path = Path(args.summary_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result_payload, f, indent=2, sort_keys=True)

        print("✅ SNQI ablation analysis completed")
        print(f"🧪 Components analyzed: {len(target_components)}")
        print(f"🌱 Seed (reserved): {args.seed}")
        print(f"� Episodes: {episodes_path}")
        print(f"💾 Results saved to: {output_path}")

        print("\n📊 Component Impact Summary (mean SNQI drop when removed):")
        for comp in target_components:
            delta = impacts.get(comp, 0.0)
            print(f"  {comp}: {delta:.4f}")

        return 0
    except Exception as e:  # noqa: BLE001
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
