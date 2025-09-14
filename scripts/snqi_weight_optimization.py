#!/usr/bin/env python3
"""SNQI Weight Optimization Script.

Refactored version with decomposed helpers for clarity and reduced complexity.
Supports grid search and differential evolution strategies plus optional sensitivity
analysis. Outputs metadata-rich JSON with schema + summary section.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
from scipy.optimize import differential_evolution

from robot_sf.benchmark.snqi import WEIGHT_NAMES, compute_snqi
from robot_sf.benchmark.snqi.schema import assert_all_finite, validate_snqi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for weight optimization results."""

    weights: Dict[str, float]
    objective_value: float
    ranking_stability: float
    convergence_info: Dict[str, Any]


class SNQIWeightOptimizer:
    """SNQI weight optimization using multiple strategies."""

    def __init__(self, episodes_data: List[Dict], baseline_stats: Dict[str, Dict[str, float]]):
        self.episodes = episodes_data
        self.baseline_stats = baseline_stats
        self.weight_names = list(WEIGHT_NAMES)

    def _episode_snqi(self, metrics: Dict[str, float], weights: Dict[str, float]) -> float:
        return compute_snqi(metrics, weights, self.baseline_stats)

    def compute_ranking_stability(self, weights: Dict[str, float]) -> float:
        if len(self.episodes) < 2:
            return 1.0
        algo_groups: Dict[str, list] = {}
        for ep in self.episodes:
            algo = ep.get("scenario_params", {}).get("algo", ep.get("scenario_id", "default"))
            algo_groups.setdefault(algo, []).append(ep)
        if len(algo_groups) < 2:
            scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
            return float(1.0 / (1.0 + float(np.var(scores))))
        group_rankings: Dict[str, List[int]] = {}
        for group_name, group_eps in algo_groups.items():
            scores = [
                (self._episode_snqi(ep.get("metrics", {}), weights), i)
                for i, ep in enumerate(group_eps)
            ]
            scores.sort(reverse=True)
            group_rankings[group_name] = [idx for _, idx in scores]
        group_names = list(group_rankings.keys())
        if len(group_names) >= 2:
            from scipy.stats import spearmanr

            try:
                corr, _ = spearmanr(group_rankings[group_names[0]], group_rankings[group_names[1]])
                return abs(corr) if not np.isnan(corr) else 0.5
            except Exception:  # noqa: BLE001
                return 0.5
        return 0.8

    def objective_function(self, weight_vector: np.ndarray) -> float:
        weights = dict(zip(self.weight_names, weight_vector))
        stability = self.compute_ranking_stability(weights)
        scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        discriminative_power = min(1.0, score_variance / 0.5)
        return -(0.6 * stability + 0.4 * discriminative_power)

    def grid_search_optimization(
        self, grid_resolution: int = 5, max_combinations: int | None = None
    ) -> OptimizationResult:
        logger.info("Starting grid search optimization with resolution %d", grid_resolution)
        n = len(self.weight_names)
        # Guard: shrink resolution if combinations explode
        if max_combinations is not None:
            while grid_resolution**n > max_combinations and grid_resolution > 2:
                grid_resolution -= 1
        grid_points = np.linspace(0.1, 3.0, grid_resolution)
        total_combinations = grid_resolution**n
        if max_combinations is not None and total_combinations > max_combinations:
            logger.warning(
                "Grid search still exceeds max combinations (%d > %d); sampling subset.",
                total_combinations,
                max_combinations,
            )
            rng = np.random.default_rng(42)
            sample_size = max_combinations
            sampled = [rng.choice(grid_points, size=n) for _ in range(sample_size)]
            combos_iter = sampled
        else:
            combos_iter = product(grid_points, repeat=n)
        best_obj = float("inf")
        best_weights: Dict[str, float] | None = None
        best_stability = 0.0
        evaluations = 0
        for combo in combos_iter:
            weights = {k: float(v) for k, v in zip(self.weight_names, combo)}
            obj = self.objective_function(np.array(list(weights.values())))
            if obj < best_obj:
                best_obj = obj
                best_weights = weights
                best_stability = self.compute_ranking_stability(weights)
            evaluations += 1
        if best_weights is None:  # Fallback (should not happen)
            best_weights = {k: 1.0 for k in self.weight_names}
        # Convert objective back to positive score (we minimized negative)
        positive_score = -best_obj
        return OptimizationResult(
            weights=best_weights,
            objective_value=positive_score,
            ranking_stability=best_stability,
            convergence_info={
                "evaluations": evaluations,
                "grid_resolution": grid_resolution,
                "total_combinations_considered": evaluations,
            },
        )

    def differential_evolution_optimization(
        self, maxiter: int = 30, seed: int | None = None
    ) -> OptimizationResult:
        bounds = [(0.1, 3.0)] * len(self.weight_names)
        result = differential_evolution(
            self.objective_function,
            bounds=bounds,
            maxiter=maxiter,
            seed=seed,
            polish=True,
            strategy="best1bin",
        )
        weights = dict(zip(self.weight_names, result.x))
        stability = self.compute_ranking_stability(weights)
        positive_score = -result.fun
        convergence = {
            "nit": result.nit,
            "nfev": result.nfev,
            "success": bool(result.success),
            "message": result.message,
        }
        return OptimizationResult(
            weights=weights,
            objective_value=positive_score,
            ranking_stability=stability,
            convergence_info=convergence,
        )

    def sensitivity_analysis(self, weights: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        base_scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
        base_mean = float(np.mean(base_scores)) if base_scores else 0.0
        results: Dict[str, Dict[str, float]] = {}
        for name, value in weights.items():
            delta = 0.1 * value if value != 0 else 0.1
            up = {**weights, name: min(3.0, value + delta)}
            down = {**weights, name: max(0.1, value - delta)}
            up_scores = [self._episode_snqi(ep.get("metrics", {}), up) for ep in self.episodes]
            down_scores = [self._episode_snqi(ep.get("metrics", {}), down) for ep in self.episodes]
            up_mean = float(np.mean(up_scores)) if up_scores else 0.0
            down_mean = float(np.mean(down_scores)) if down_scores else 0.0
            sensitivity = abs(up_mean - base_mean) + abs(down_mean - base_mean)
            results[name] = {
                "base_mean": base_mean,
                "up_mean": up_mean,
                "down_mean": down_mean,
                "score_sensitivity": sensitivity,
            }
        return results


def validate_weights_mapping(weights: Dict[str, float]) -> None:
    """Validate external weights mapping.

    Ensures keys match WEIGHT_NAMES and values are finite positive numbers in a reasonable range.
    """
    missing = [k for k in WEIGHT_NAMES if k not in weights]
    if missing:
        raise ValueError(f"Missing weight keys: {missing}")
    extraneous = [k for k in weights.keys() if k not in WEIGHT_NAMES]
    if extraneous:
        logger.warning("Extraneous weight keys will be ignored: %s", extraneous)
    for k, v in weights.items():
        try:
            fv = float(v)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Non-numeric weight for {k}: {v}") from e
        if not np.isfinite(fv) or fv <= 0:
            raise ValueError(f"Invalid weight value for {k}: {fv}")
        if fv > 10:
            logger.warning("Weight %s unusually large (%.3f) > 10", k, fv)


# ---------------------------- I/O helpers ---------------------------- #
def load_episodes_data(path: Path) -> List[Dict[str, Any]]:
    """Load episodes from JSONL file."""
    episodes: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    episodes.append(obj)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line in %s", path)
    return episodes


def load_baseline_stats(path: Path) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):  # basic validation
        raise ValueError("Baseline stats file must contain a JSON object")
    return data  # type: ignore[return-value]


# ------------------------- Orchestration helpers --------------------- #
def _load_inputs(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict[str, float]]]:
    episodes = load_episodes_data(args.episodes)
    baseline_stats = load_baseline_stats(args.baseline)
    return episodes, baseline_stats


def _select_best(results: Dict[str, Any], method: str) -> None:
    if method == "both":
        best_method = "grid_search"
        if "differential_evolution" in results and results["differential_evolution"][
            "objective_value"
        ] > results.get("grid_search", {}).get("objective_value", -float("inf")):
            best_method = "differential_evolution"
        results["recommended"] = results[best_method].copy()
        results["recommended"]["method_used"] = best_method
    elif method == "grid":
        results["recommended"] = results["grid_search"].copy()
        results["recommended"]["method_used"] = "grid_search"
    else:
        results["recommended"] = results["differential_evolution"].copy()
        results["recommended"]["method_used"] = "differential_evolution"


def _augment_metadata(
    results: Dict[str, Any], args: argparse.Namespace, start_iso: str, start_perf: float
) -> None:
    def _git_commit() -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
        except Exception:  # noqa: BLE001
            return "UNKNOWN"

    end_perf = perf_counter()
    end_iso = datetime.now(timezone.utc).isoformat()
    runtime_seconds = end_perf - start_perf
    results_meta = {
        "schema_version": 1,
        "generated_at": end_iso,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "start_time": start_iso,
        "end_time": end_iso,
        "runtime_seconds": runtime_seconds,
        "provenance": {
            "episodes_file": str(args.episodes),
            "baseline_file": str(args.baseline),
            "invocation": "python " + " ".join(sys.argv),
            "method_requested": args.method,
        },
    }
    results["_metadata"] = results_meta
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
        "runtime_seconds": runtime_seconds,
        "start_time": start_iso,
        "end_time": end_iso,
    }


def _print_summary(results: Dict[str, Any], args: argparse.Namespace) -> None:
    recommended = results["recommended"]
    print("\nOptimization Summary:")
    print(f"Method: {recommended['method_used']}")
    print(f"Objective Value: {recommended['objective_value']:.4f}")
    print(f"Ranking Stability: {recommended['ranking_stability']:.4f}")
    print("\nRecommended Weights:")
    for weight_name, value in recommended["weights"].items():
        print(f"  {weight_name}: {value:.3f}")
    if args.sensitivity and "sensitivity_analysis" in results:
        print("\nSensitivity Analysis (top 3 most sensitive weights):")
        sensitivity = results["sensitivity_analysis"]
        sorted_weights = sorted(
            sensitivity.items(), key=lambda x: x[1]["score_sensitivity"], reverse=True
        )
        for weight_name, sens_data in sorted_weights[:3]:
            print(f"  {weight_name}: score_sensitivity={sens_data['score_sensitivity']:.4f}")


# ----------------------------- Main runner --------------------------- #
def run(args: argparse.Namespace) -> int:  # noqa: C901 - acceptable after decomposition
    start_perf = perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    try:
        episodes, baseline_stats = _load_inputs(args)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed loading inputs: %s", e)
        return 1
    if not episodes:
        logger.error("No valid episodes found in data file")
        return 1
    if args.seed is not None:
        np.random.seed(args.seed)
    optimizer = SNQIWeightOptimizer(episodes, baseline_stats)
    results: Dict[str, Any] = {}
    if args.method in ["grid", "both"]:
        grid_result = optimizer.grid_search_optimization(
            args.grid_resolution, max_combinations=args.max_grid_combinations
        )
        results["grid_search"] = {
            "weights": grid_result.weights,
            "objective_value": grid_result.objective_value,
            "ranking_stability": grid_result.ranking_stability,
            "convergence_info": grid_result.convergence_info,
        }
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
    _select_best(results, args.method)
    if args.sensitivity:
        recommended_weights = results["recommended"]["weights"]
        results["sensitivity_analysis"] = optimizer.sensitivity_analysis(recommended_weights)
    _augment_metadata(results, args, start_iso, start_perf)
    try:
        if args.validate:
            validate_snqi(results, "optimization", check_finite=True)
        else:
            assert_all_finite(results)
    except ValueError as e:
        logger.error("Validation failed: %s", e)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)
    _print_summary(results, args)
    return 0


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize SNQI weights")
    parser.add_argument("--episodes", type=Path, required=True, help="Episodes JSONL file")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline stats JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument(
        "--method",
        choices=["grid", "evolution", "both"],
        default="both",
        help="Optimization method to use",
    )
    parser.add_argument("--grid-resolution", type=int, default=5, help="Grid resolution per weight")
    parser.add_argument(
        "--maxiter", type=int, default=30, help="Differential evolution max iterations"
    )
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--validate", action="store_true", help="Validate output schema")
    parser.add_argument(
        "--max-grid-combinations",
        type=int,
        default=20000,
        help="Guard threshold for total grid combinations (adaptive shrink)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
