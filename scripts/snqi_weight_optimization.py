#!/usr/bin/env python3
"""SNQI Weight Optimization Script

This script recomputes optimal weights for the Social Navigation Quality Index (SNQI)
using various optimization strategies including grid search, genetic algorithms, and
Pareto analysis. It supports the median/p95 normalization strategy implemented in the
social navigation benchmark.

Usage:
    python scripts/snqi_weight_optimization.py --episodes episodes.jsonl --baseline baseline_stats.json --output weights.json

Requirements:
    - Episode data in JSONL format (from benchmark runner)
    - Baseline statistics for normalization (median/p95 values)
    - Optional: multiple algorithm comparison data
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import differential_evolution

from robot_sf.benchmark.snqi import WEIGHT_NAMES, compute_snqi
from robot_sf.benchmark.snqi.schema import assert_all_finite, validate_snqi

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for weight optimization results."""

    weights: Dict[str, float]
    objective_value: float
    ranking_stability: float
    pareto_efficiency: float
    convergence_info: Dict[str, Any]


class SNQIWeightOptimizer:
    """SNQI weight optimization using multiple strategies."""

    def __init__(self, episodes_data: List[Dict], baseline_stats: Dict[str, Dict[str, float]]):
        """Initialize optimizer with episode data and baseline statistics.

        Args:
            episodes_data: List of episode records with metrics
            baseline_stats: Baseline statistics for normalization (median/p95)
        """
        self.episodes = episodes_data
        self.baseline_stats = baseline_stats
        # Maintain list locally for methods; keep reference to canonical constant
        self.weight_names = list(WEIGHT_NAMES)

    def _episode_snqi(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        """Wrapper using canonical compute_snqi with this instance's baseline stats."""
        return compute_snqi(metrics, weights, self.baseline_stats)

    def compute_ranking_stability(self, weights: Dict[str, float]) -> float:
        """Compute ranking stability metric across different algorithms/scenarios."""
        if len(self.episodes) < 2:
            return 1.0

        # Group episodes by algorithm if available
        algo_groups = {}
        for ep in self.episodes:
            algo = ep.get("scenario_params", {}).get("algo", ep.get("scenario_id", "default"))
            if algo not in algo_groups:
                algo_groups[algo] = []
            algo_groups[algo].append(ep)

        if len(algo_groups) < 2:
            # Not enough algorithms to compare, return based on variance within single group
            scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
            return float(1.0 / (1.0 + float(np.var(scores))))

        # Compute rankings within each scenario group and measure consistency
        group_rankings = {}

        for group_name, group_episodes in algo_groups.items():
            scores = [
                (self._episode_snqi(ep.get("metrics", {}), weights), i)
                for i, ep in enumerate(group_episodes)
            ]
            scores.sort(reverse=True)  # Higher SNQI is better
            group_rankings[group_name] = [item[1] for item in scores]

        # If multiple groups exist, compute rank correlation between them
        group_names = list(group_rankings.keys())
        if len(group_names) >= 2:
            from scipy.stats import spearmanr

            try:
                # Compare first two groups (could be extended to all pairs)
                corr, _ = spearmanr(group_rankings[group_names[0]], group_rankings[group_names[1]])
                return abs(corr) if not np.isnan(corr) else 0.5
            except Exception:
                return 0.5

        return 0.8  # Default for single group

    def objective_function(self, weight_vector: np.ndarray) -> float:
        """Objective function for optimization (to be minimized)."""
        weights = dict(zip(self.weight_names, weight_vector))

        # Multi-objective: maximize ranking stability and discriminative power
        stability = self.compute_ranking_stability(weights)

        # Discriminative power: SNQI should have good variance across episodes
        scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        discriminative_power = min(1.0, score_variance / 0.5)  # normalize to [0,1]

        # Combined objective (minimize negative of weighted sum)
        combined = -(0.6 * stability + 0.4 * discriminative_power)
        return combined

    def grid_search_optimization(self, grid_resolution: int = 5) -> OptimizationResult:
        """Grid search over weight space."""
        logger.info(f"Starting grid search optimization with resolution {grid_resolution}")

        best_objective = float("inf")
        best_weights = None
        best_stability = 0.0

        # Simple grid search (exponential in dimensions, so keep resolution small)
        grid_points = np.linspace(0.1, 3.0, grid_resolution)
        total_combinations = grid_resolution ** len(self.weight_names)

        logger.info(f"Evaluating {total_combinations} weight combinations")

        count = 0
        for weight_combo in np.ndindex(*[grid_resolution] * len(self.weight_names)):
            weight_values = [grid_points[i] for i in weight_combo]
            obj_value = self.objective_function(np.array(weight_values))

            if obj_value < best_objective:
                best_objective = obj_value
                best_weights = weight_values
                best_stability = self.compute_ranking_stability(
                    dict(zip(self.weight_names, weight_values))
                )

            count += 1
            if count % max(1, total_combinations // 10) == 0:
                logger.info(f"Grid search progress: {count}/{total_combinations}")

        if best_weights is None:  # Fallback safety (should not occur if grid_resolution>=1)
            best_weights = [1.0] * len(self.weight_names)
        return OptimizationResult(
            weights=dict(zip(self.weight_names, best_weights)),
            objective_value=-best_objective,  # Convert back to positive
            ranking_stability=best_stability,
            pareto_efficiency=0.8,  # Placeholder
            convergence_info={"method": "grid_search", "evaluated_points": total_combinations},
        )

    def differential_evolution_optimization(
        self, maxiter: int = 100, seed: int | None = None
    ) -> OptimizationResult:
        """Differential evolution optimization.

        Args:
            maxiter: Maximum number of iterations.
            seed: Optional random seed for reproducibility.
        """
        logger.info(
            "Starting differential evolution optimization with %d iterations (seed=%s)",
            maxiter,
            seed,
        )

        bounds = [(0.1, 3.0) for _ in self.weight_names]

        # Pass through seed if provided; SciPy accepts int | None
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=maxiter,
            popsize=15,
            seed=seed,
            polish=True,
        )

        final_weights = dict(zip(self.weight_names, result.x))
        stability = self.compute_ranking_stability(final_weights)

        return OptimizationResult(
            weights=final_weights,
            objective_value=-result.fun,  # Convert back to positive
            ranking_stability=stability,
            pareto_efficiency=0.9,  # Placeholder
            convergence_info={
                "method": "differential_evolution",
                "success": result.success,
                "iterations": result.nit,
                "function_evaluations": result.nfev,
                "seed": seed,
            },
        )

    def sensitivity_analysis(
        self, base_weights: Dict[str, float], perturbation_range: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """Perform sensitivity analysis by perturbing each weight."""
        logger.info("Performing sensitivity analysis")

        results = {}
        base_snqi_scores = [
            self._episode_snqi(ep.get("metrics", {}), base_weights) for ep in self.episodes
        ]
        base_stability = self.compute_ranking_stability(base_weights)

        for weight_name in self.weight_names:
            weight_results = {
                "stability_sensitivity": 0.0,
                "score_sensitivity": 0.0,
                "ranking_changes": 0.0,
            }

            # Test both increase and decrease
            for direction in [-1, 1]:
                perturbed_weights = base_weights.copy()
                original_value = base_weights.get(weight_name, 1.0)
                perturbed_weights[weight_name] = original_value * (
                    1 + direction * perturbation_range
                )

                # Compute metrics with perturbed weights
                perturbed_scores = [
                    self._episode_snqi(ep.get("metrics", {}), perturbed_weights)
                    for ep in self.episodes
                ]
                perturbed_stability = self.compute_ranking_stability(perturbed_weights)

                # Measure sensitivity
                score_change = np.mean(
                    np.abs(np.array(perturbed_scores) - np.array(base_snqi_scores))
                )
                stability_change = abs(perturbed_stability - base_stability)

                weight_results["score_sensitivity"] = max(
                    weight_results["score_sensitivity"], score_change
                )
                weight_results["stability_sensitivity"] = max(
                    weight_results["stability_sensitivity"], stability_change
                )

            results[weight_name] = weight_results

        return results


def load_episodes_data(file_path: Path) -> List[Dict]:
    """Load episode data from JSONL file."""
    episodes = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {e}")
    logger.info(f"Loaded {len(episodes)} episodes from {file_path}")
    return episodes


def load_baseline_stats(file_path: Path) -> Dict[str, Dict[str, float]]:
    """Load baseline statistics from JSON file."""
    with open(file_path, "r") as f:
        stats = json.load(f)
    logger.info(f"Loaded baseline statistics from {file_path}")
    return stats


def main():  # noqa: C901
    # Runtime start (high-resolution)
    _t_start_perf = perf_counter()
    _t_start_iso = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(description="SNQI Weight Optimization")
    parser.add_argument(
        "--episodes", type=Path, required=True, help="Path to episode data JSONL file"
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Path to baseline statistics JSON file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to output optimized weights JSON file"
    )
    parser.add_argument(
        "--method",
        choices=["grid", "evolution", "both"],
        default="both",
        help="Optimization method to use",
    )
    parser.add_argument(
        "--sensitivity", action="store_true", help="Also perform sensitivity analysis"
    )
    parser.add_argument(
        "--grid-resolution",
        type=int,
        default=5,
        help="Grid resolution for grid search (default: 5)",
    )
    parser.add_argument(
        "--maxiter", type=int, default=100, help="Maximum iterations for evolution (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (applies to differential evolution and any stochastic components)",
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
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    if len(episodes) == 0:
        logger.error("No valid episodes found in data file")
        sys.exit(1)

    # Apply seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)

    # Initialize optimizer
    optimizer = SNQIWeightOptimizer(episodes, baseline_stats)

    results = {}

    # Run optimization
    if args.method in ["grid", "both"]:
        grid_result = optimizer.grid_search_optimization(args.grid_resolution)
        results["grid_search"] = {
            "weights": grid_result.weights,
            "objective_value": grid_result.objective_value,
            "ranking_stability": grid_result.ranking_stability,
            "convergence_info": grid_result.convergence_info,
        }
        logger.info(f"Grid search completed. Best objective: {grid_result.objective_value:.4f}")

    if args.method in ["evolution", "both"]:
        evolution_result = optimizer.differential_evolution_optimization(
            args.maxiter, seed=args.seed
        )
        results["differential_evolution"] = {
            "weights": evolution_result.weights,
            "objective_value": evolution_result.objective_value,
            "ranking_stability": evolution_result.ranking_stability,
            "convergence_info": evolution_result.convergence_info,
        }
        logger.info(
            f"Differential evolution completed. Best objective: {evolution_result.objective_value:.4f}"
        )

    # Select best result
    if args.method == "both":
        best_method = "grid_search"
        if "differential_evolution" in results:
            if (
                results["differential_evolution"]["objective_value"]
                > results["grid_search"]["objective_value"]
            ):
                best_method = "differential_evolution"
        results["recommended"] = results[best_method].copy()
        results["recommended"]["method_used"] = best_method
    elif args.method == "grid":
        results["recommended"] = results["grid_search"].copy()
        results["recommended"]["method_used"] = "grid_search"
    else:
        results["recommended"] = results["differential_evolution"].copy()
        results["recommended"]["method_used"] = "differential_evolution"

    # Sensitivity analysis
    if args.sensitivity:
        logger.info("Running sensitivity analysis")
        recommended_weights = results["recommended"]["weights"]
        sensitivity_results = optimizer.sensitivity_analysis(recommended_weights)
        results["sensitivity_analysis"] = sensitivity_results

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
            "method_requested": args.method,
        },
    }
    results["_metadata"] = results_meta

    # Add standardized summary block
    recommended = results["recommended"]
    results["summary"] = {
        "method": recommended.get("method_used"),
        "objective_value": recommended.get("objective_value"),
        "ranking_stability": recommended.get("ranking_stability"),
        "weights": recommended.get("weights"),
        "available_methods": [
            k for k in results.keys() if k in ("grid_search", "differential_evolution")
        ],
        "seed": args.seed,
        "has_sensitivity": bool(results.get("sensitivity_analysis")),
        "runtime_seconds": _runtime_seconds,
        "start_time": _t_start_iso,
        "end_time": _t_end_iso,
    }

    # Validation & finiteness checks
    try:
        if args.validate:
            validate_snqi(results, "optimization", check_finite=True)
        else:
            # At least assert finiteness even if structural validation skipped
            assert_all_finite(results)
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # Save results (only after successful validation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")

    # Print summary
    recommended = results["recommended"]
    print("\nOptimization Summary:")
    print(f"Method: {recommended['method_used']}")
    print(f"Objective Value: {recommended['objective_value']:.4f}")
    print(f"Ranking Stability: {recommended['ranking_stability']:.4f}")
    print("\nRecommended Weights:")
    for weight_name, value in recommended["weights"].items():
        print(f"  {weight_name}: {value:.3f}")

    if args.sensitivity:
        print("\nSensitivity Analysis (top 3 most sensitive weights):")
        sensitivity = results["sensitivity_analysis"]
        sorted_weights = sorted(
            sensitivity.items(), key=lambda x: x[1]["score_sensitivity"], reverse=True
        )
        for weight_name, sens_data in sorted_weights[:3]:
            print(f"  {weight_name}: score_sensitivity={sens_data['score_sensitivity']:.4f}")


if __name__ == "__main__":
    main()
