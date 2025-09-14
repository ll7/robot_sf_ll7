#!/usr/bin/env python3
"""SNQI Weight Recomputation Utilities

This script provides utilities to recompute SNQI weights using different strategies
and normalization approaches (median/p95). It's designed to work with the existing
SNQI implementation from the social navigation benchmark.

Usage:
    python scripts/recompute_snqi_weights.py --episodes episodes.jsonl --baseline baseline_stats.json --strategy pareto --output weights_optimized.json
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np

from robot_sf.benchmark.snqi import WEIGHT_NAMES, compute_snqi  # type: ignore
from robot_sf.benchmark.snqi.schema import assert_all_finite, validate_snqi

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SNQIWeightRecomputer:
    """Utilities for recomputing SNQI weights with different strategies."""

    def __init__(self, episodes_data: List[Dict], baseline_stats: Dict[str, Dict[str, float]]):
        self.episodes = episodes_data
        self.baseline_stats = baseline_stats
        self.weight_names = WEIGHT_NAMES

    def _episode_snqi(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """Wrapper calling shared compute_snqi keeping backward compatibility."""
        return compute_snqi(metrics, weights, self.baseline_stats)

    def default_weights(self) -> Dict[str, float]:
        """Return default weight configuration."""
        return {
            "w_success": 2.0,
            "w_time": 1.0,
            "w_collisions": 2.0,
            "w_near": 1.0,
            "w_comfort": 1.5,
            "w_force_exceed": 1.5,
            "w_jerk": 0.5,
        }

    def balanced_weights(self) -> Dict[str, float]:
        """Return balanced weight configuration."""
        return {name: 1.0 for name in self.weight_names}

    def safety_focused_weights(self) -> Dict[str, float]:
        """Return safety-focused weight configuration."""
        return {
            "w_success": 1.5,
            "w_time": 0.8,
            "w_collisions": 3.0,
            "w_near": 2.0,
            "w_comfort": 2.5,
            "w_force_exceed": 2.5,
            "w_jerk": 1.0,
        }

    def efficiency_focused_weights(self) -> Dict[str, float]:
        """Return efficiency-focused weight configuration."""
        return {
            "w_success": 2.5,
            "w_time": 2.0,
            "w_collisions": 1.5,
            "w_near": 1.0,
            "w_comfort": 1.0,
            "w_force_exceed": 1.0,
            "w_jerk": 1.5,
        }

    def compute_weight_statistics(self, weights: Dict[str, float]) -> Dict[str, Any]:
        """Compute statistics for given weights across all episodes."""
        scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]

        # Group by algorithm if available
        algo_scores = {}
        for ep in self.episodes:
            algo = ep.get("scenario_params", {}).get("algo", ep.get("scenario_id", "default"))
            if algo not in algo_scores:
                algo_scores[algo] = []
            algo_scores[algo].append(self._episode_snqi(ep.get("metrics", {}), weights))

        stats = {
            "overall": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "range": float(np.max(scores) - np.min(scores)),
            },
            "by_algorithm": {},
        }

        for algo, algo_score_list in algo_scores.items():
            if len(algo_score_list) > 0:
                stats["by_algorithm"][algo] = {
                    "mean": float(np.mean(algo_score_list)),
                    "std": float(np.std(algo_score_list)),
                    "count": len(algo_score_list),
                }

        return stats

    def rank_correlation_analysis(
        self, weights1: Dict[str, float], weights2: Dict[str, float]
    ) -> float:
        """Compute ranking correlation between two weight configurations."""
        scores1 = [self._episode_snqi(ep.get("metrics", {}), weights1) for ep in self.episodes]
        scores2 = [self._episode_snqi(ep.get("metrics", {}), weights2) for ep in self.episodes]

        ranking1 = np.argsort(np.argsort(scores1))
        ranking2 = np.argsort(np.argsort(scores2))

        from scipy.stats import spearmanr

        corr, _ = spearmanr(ranking1, ranking2)
        return corr if not np.isnan(corr) else 1.0

    def pareto_frontier_weights(self, n_samples: int = 1000) -> List[Dict[str, Any]]:
        """Generate Pareto-optimal weight configurations."""
        logger.info("Computing Pareto frontier with %d samples", n_samples)

        # Sample random weight configurations
        samples = []
        for _ in range(n_samples):
            # Sample weights uniformly from [0.1, 3.0]
            weights = {name: np.random.uniform(0.1, 3.0) for name in self.weight_names}

            # Compute objectives: discriminative power and ranking stability
            scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
            discriminative_power = np.std(scores)

            # Stability: based on score variance and range
            stability = 1.0 / (1.0 + abs(discriminative_power - 0.5))  # Prefer moderate variance

            samples.append(
                {
                    "weights": weights,
                    "discriminative_power": discriminative_power,
                    "stability": stability,
                    "mean_score": np.mean(scores),
                }
            )

        # Find Pareto frontier
        pareto_samples = []
        for i, sample in enumerate(samples):
            is_dominated = False
            for j, other in enumerate(samples):
                if i != j:
                    # Check if 'other' dominates 'sample'
                    if (
                        other["discriminative_power"] >= sample["discriminative_power"]
                        and other["stability"] >= sample["stability"]
                        and (
                            other["discriminative_power"] > sample["discriminative_power"]
                            or other["stability"] > sample["stability"]
                        )
                    ):
                        is_dominated = True
                        break
            if not is_dominated:
                pareto_samples.append(sample)

        # Sort by discriminative power
        pareto_samples.sort(key=lambda x: x["discriminative_power"], reverse=True)

        logger.info("Found %d Pareto-optimal configurations", len(pareto_samples))
        return pareto_samples[:10]  # Return top 10

    def recompute_with_strategy(self, strategy: str) -> Dict[str, Any]:
        """Recompute weights using specified strategy."""
        logger.info("Recomputing weights with strategy: %s", strategy)

        if strategy == "default":
            weights = self.default_weights()
        elif strategy == "balanced":
            weights = self.balanced_weights()
        elif strategy == "safety_focused":
            weights = self.safety_focused_weights()
        elif strategy == "efficiency_focused":
            weights = self.efficiency_focused_weights()
        elif strategy == "pareto":
            pareto_samples = self.pareto_frontier_weights()
            if pareto_samples:
                # Select the one with best balance of discriminative power and stability
                best_sample = max(
                    pareto_samples,
                    key=lambda x: 0.6 * x["discriminative_power"] + 0.4 * x["stability"],
                )
                weights = best_sample["weights"]
            else:
                weights = self.default_weights()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Compute statistics for the selected weights
        stats = self.compute_weight_statistics(weights)

        result = {
            "strategy": strategy,
            "weights": weights,
            "statistics": stats,
            "normalization_strategy": "median_p95",
        }

        if strategy == "pareto":
            result["pareto_alternatives"] = pareto_samples[:5]  # Include top 5 alternatives

        return result

    def compare_normalization_strategies(self, base_weights: Dict[str, float]) -> Dict[str, Any]:
        """Compare different normalization strategies with given weights."""
        logger.info("Comparing normalization strategies")

        # Create alternative baseline stats
        metric_values = {
            metric: []
            for metric in ["collisions", "near_misses", "force_exceed_events", "jerk_mean"]
        }

        for ep in self.episodes:
            metrics = ep.get("metrics", {})
            for metric_name in metric_values.keys():
                value = metrics.get(metric_name)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    metric_values[metric_name].append(float(value))

        strategies = {}

        # Original median/p95
        strategies["median_p95"] = self.baseline_stats

        # Alternative strategies
        for metric_name, values in metric_values.items():
            if len(values) > 0:
                arr = np.array(values)

                if "median_p90" not in strategies:
                    strategies["median_p90"] = {}
                strategies["median_p90"][metric_name] = {
                    "med": float(np.median(arr)),
                    "p95": float(np.percentile(arr, 90)),
                }

                if "p25_p75" not in strategies:
                    strategies["p25_p75"] = {}
                strategies["p25_p75"][metric_name] = {
                    "med": float(np.percentile(arr, 25)),
                    "p95": float(np.percentile(arr, 75)),
                }

        results = {}

        original_baseline = self.baseline_stats
        base_scores = None

        for strategy_name, strategy_baseline in strategies.items():
            self.baseline_stats = strategy_baseline

            scores = [
                self._episode_snqi(ep.get("metrics", {}), base_weights) for ep in self.episodes
            ]

            if base_scores is None:
                base_scores = scores

            results[strategy_name] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "score_range": float(np.max(scores) - np.min(scores)),
                "correlation_with_base": float(np.corrcoef(base_scores, scores)[0, 1])
                if len(scores) > 1
                else 1.0,
            }

        # Restore original baseline
        self.baseline_stats = original_baseline

        return results


def load_episodes_data(file_path: Path) -> List[Dict]:
    """Load episode data from JSONL file."""
    episodes = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skipping invalid JSON line: %s", e)
    logger.info("Loaded %d episodes from %s", len(episodes), file_path)
    return episodes


def load_baseline_stats(file_path: Path) -> Dict[str, Dict[str, float]]:
    """Load baseline statistics from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    logger.info("Loaded baseline statistics from %s", file_path)
    return stats


def main():  # noqa: C901
    _t_start_perf = perf_counter()
    _t_start_iso = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description="SNQI Weight Recomputation")
    parser.add_argument(
        "--episodes", type=Path, required=True, help="Path to episode data JSONL file"
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Path to baseline statistics JSON file"
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "balanced", "safety_focused", "efficiency_focused", "pareto"],
        default="default",
        help="Weight recomputation strategy",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path for recomputed weights JSON"
    )
    parser.add_argument(
        "--compare-normalization",
        action="store_true",
        help="Also compare different normalization strategies",
    )
    parser.add_argument(
        "--compare-strategies", action="store_true", help="Compare all available strategies"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (applies to stochastic sampling in strategies like pareto)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate JSON output structure and numeric finiteness before writing",
    )

    args = parser.parse_args()

    # Load data
    try:
        episodes = load_episodes_data(args.episodes)
        baseline_stats = load_baseline_stats(args.baseline)
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        return 1
    except Exception as e:
        logger.error("Error loading data: %s", e)
        return 1

    # Apply seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Initialize recomputer
    recomputer = SNQIWeightRecomputer(episodes, baseline_stats)

    results = {}

    if args.compare_strategies:
        # Compare all strategies
        logger.info("Comparing all weight strategies")
        all_strategies = ["default", "balanced", "safety_focused", "efficiency_focused", "pareto"]
        strategy_results = {}

        for strategy in all_strategies:
            try:
                strategy_result = recomputer.recompute_with_strategy(strategy)
                strategy_results[strategy] = strategy_result
            except Exception as e:
                logger.warning("Failed to compute strategy %s: %s", strategy, e)

        results["strategy_comparison"] = strategy_results

        # Compute correlations between strategies
        correlations = {}
        strategy_names = list(strategy_results.keys())
        for i, strategy1 in enumerate(strategy_names):
            for strategy2 in strategy_names[i + 1 :]:
                weights1 = strategy_results[strategy1]["weights"]
                weights2 = strategy_results[strategy2]["weights"]
                corr = recomputer.rank_correlation_analysis(weights1, weights2)
                correlations[f"{strategy1}_vs_{strategy2}"] = corr

        results["strategy_correlations"] = correlations

        # Select recommended strategy
        best_strategy = None
        best_score = -float("inf")

        for strategy_name, strategy_data in strategy_results.items():
            # Score based on discriminative power and stability
            stats = strategy_data["statistics"]["overall"]
            score = stats["range"] * 0.6 + (1.0 / (1.0 + stats["std"])) * 0.4

            if score > best_score:
                best_score = score
                best_strategy = strategy_name

        results["recommended_strategy"] = best_strategy
        results["recommended_weights"] = strategy_results[best_strategy]["weights"]

    else:
        # Single strategy
        result = recomputer.recompute_with_strategy(args.strategy)
        results["strategy_result"] = result
        results["recommended_weights"] = result["weights"]

    # Normalization comparison
    if args.compare_normalization:
        weights_for_norm = results["recommended_weights"]
        norm_results = recomputer.compare_normalization_strategies(weights_for_norm)
        results["normalization_comparison"] = norm_results

    # Augment results with metadata
    def _git_commit() -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:
            return "UNKNOWN"

    _t_end_perf = perf_counter()
    _t_end_iso = datetime.now(timezone.utc).isoformat()
    _runtime_seconds = _t_end_perf - _t_start_perf

    results_meta = {
        "schema_version": 1,
        "generated_at": _t_end_iso,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "start_time": _t_start_iso,
        "end_time": _t_end_iso,
        "runtime_seconds": _runtime_seconds,
        "provenance": {
            "episodes_file": str(args.episodes),
            "baseline_file": str(args.baseline),
            "invocation": "python " + " ".join(sys.argv),
            "strategy_requested": args.strategy,
            "compare_strategies": args.compare_strategies,
            "compare_normalization": args.compare_normalization,
        },
    }
    results["_metadata"] = results_meta

    # Validation & finiteness checks
    try:
        if args.validate:
            validate_snqi(results, "recompute", check_finite=True)
        else:
            assert_all_finite(results)
    except ValueError as e:
        logger.error("Validation failed: %s", e)
        return 1

    # Standardized summary block (after validation of structure, before save)
    if args.compare_strategies:
        method_descriptor = results.get("recommended_strategy")
    else:
        method_descriptor = results.get("strategy_result", {}).get("strategy", args.strategy)

    results["summary"] = {
        "method": method_descriptor,
        "weights": results.get("recommended_weights"),
        "compare_strategies": args.compare_strategies,
        "compare_normalization": args.compare_normalization,
        "seed": args.seed,
        "has_normalization_comparison": bool(results.get("normalization_comparison")),
        "runtime_seconds": _runtime_seconds,
        "start_time": _t_start_iso,
        "end_time": _t_end_iso,
    }

    # Save results after validation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", args.output)

    # Print summary
    print("\nWeight Recomputation Summary:")
    print(f"Episodes analyzed: {len(episodes)}")

    if args.compare_strategies:
        print(f"Recommended strategy: {results['recommended_strategy']}")
        print("Strategy correlations (top 3):")
        sorted_corrs = sorted(
            results["strategy_correlations"].items(), key=lambda x: x[1], reverse=True
        )
        for pair, corr in sorted_corrs[:3]:
            print(f"  {pair}: {corr:.4f}")
    else:
        print(f"Strategy used: {args.strategy}")

    print("\nRecommended Weights:")
    for weight_name, value in results["recommended_weights"].items():
        print(f"  {weight_name}: {value:.3f}")

    if args.compare_normalization:
        print("\nNormalization Strategy Impact:")
        norm_results = results["normalization_comparison"]
        for strategy, data in norm_results.items():
            if strategy != "median_p95":
                corr = data["correlation_with_base"]
                print(f"  {strategy}: correlation with base = {corr:.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
