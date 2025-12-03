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
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

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
            return 1

        with baseline_stats_path.open("r", encoding="utf-8") as f:
            baseline_stats = json.load(f)

        # Map legacy/alias method names to canonical ones expected by the implementation
        # Canonical: {"canonical", "balanced", "optimized"}
        # Aliases maintained for backward compatibility with older docs/flags
        method_aliases = {
            "pareto_optimization": "optimized",
            "equal_weights": "balanced",
            "safety_focused": "optimized",  # closest existing behavior
        }
        method = args.method
        mapped_method = method_aliases.get(method, method)
        if mapped_method != method:
            logger.warning(
                "SNQI recompute: method '{method}' is deprecated; using '{mapped_method}'",
                method=method,
                mapped_method=mapped_method,
            )

        # Validate after mapping to provide a clearer error early
        if mapped_method not in {"canonical", "balanced", "optimized"}:
            logger.error(
                "Unknown SNQI recompute method '{method}'. Use one of: canonical|balanced|optimized",
                method=method,
            )
            return 2

        # Recompute SNQI weights using baseline statistics
        weights = recompute_snqi_weights(
            baseline_stats=baseline_stats,
            method=mapped_method,
            seed=args.seed,
        )

        # Save the recomputed weights
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        weights.save(output_path)

        # Print summary of computed weights
        for _component, _weight in weights.weights.items():
            pass

        return 0

    except Exception:
        return 1


def _load_episodes_jsonl(path: Path) -> list[dict]:
    """Load episodes jsonl.

    Args:
        path: Filesystem path to the resource.

    Returns:
        list[dict]: list of dict.
    """
    episodes: list[dict] = []
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


def _extract_metric_values(episodes: Iterable[dict], metric: str) -> list[float]:
    """Extract metric values.

    Args:
        episodes: Episode records consumed by the CLI.
        metric: Metric identifier.

    Returns:
        list[float]: list of float.
    """
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


def _compute_baseline_stats(episodes: list[dict]) -> dict[str, dict[str, float]]:
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
    stats: dict[str, dict[str, float]] = {}
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
            return 1

        episodes = _load_episodes_jsonl(episodes_path)
        if not episodes:
            return 1

        # Determine weights
        weight_map: dict[str, float]
        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.exists():
                return 1
            weight_obj = SNQIWeights.load(weights_path)
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

        for comp in target_components:
            impacts.get(comp, 0.0)

        return 0
    except Exception:
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for SNQI CLI tools."""
    parser = argparse.ArgumentParser(
        prog="robot_sf_snqi",
        description="SNQI weight management and analysis tools",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Recompute weights subcommand
    recompute_parser = subparsers.add_parser(
        "recompute",
        help="Recompute SNQI weights from baseline statistics",
    )
    recompute_parser.add_argument(
        "--baseline-stats",
        required=True,
        help="Path to baseline statistics JSON file",
    )
    recompute_parser.add_argument(
        "--out",
        required=True,
        help="Output path for recomputed weights JSON file",
    )
    recompute_parser.add_argument(
        "--method",
        default="canonical",
        choices=[
            # Canonical options (preferred)
            "canonical",
            "balanced",
            "optimized",
            # Backward-compatible aliases
            "pareto_optimization",
            "equal_weights",
            "safety_focused",
        ],
        help=(
            "Weight computation method. Canonical: canonical|balanced|optimized. "
            "Aliases (deprecated): pareto_optimization->optimized, equal_weights->balanced, "
            "safety_focused->optimized. Default: canonical"
        ),
    )
    recompute_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible weight computation (default: 42)",
    )

    # Ablation analysis subcommand
    ablation_parser = subparsers.add_parser(
        "ablation",
        help="Perform SNQI component ablation analysis",
    )
    ablation_parser.add_argument("--episodes", required=True, help="Path to episodes JSONL file")
    ablation_parser.add_argument(
        "--summary-out",
        required=True,
        help="Output path for ablation results JSON",
    )
    ablation_parser.add_argument(
        "--weights",
        help="Path to SNQI weights JSON file (optional, uses defaults if not provided)",
    )
    ablation_parser.add_argument(
        "--components",
        nargs="*",
        help="Specific SNQI components to analyze (default: all components)",
    )
    ablation_parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible analysis (default: 123)",
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
        return 1


if __name__ == "__main__":
    sys.exit(main())
