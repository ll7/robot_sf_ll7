#!/usr/bin/env python3
"""SNQI Weight Recomputation Script.

Refactored for parity with the optimization script: decomposed into helpers,
provides consistent metadata + summary, optional strategy & normalization
comparisons, and schema/finite validation. Drops placeholder fields (no
pareto_efficiency) and keeps deterministic behavior via optional seed.
"""

from __future__ import annotations

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
    """Recompute SNQI weights using pre-defined selection strategies."""

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

    def pareto_frontier_weights(self, n_samples: int = 600) -> List[Dict[str, Any]]:
        """Sample candidate weights and return an approximate Pareto frontier.

        Keeps runtime modest (default 600 samples). Objective dimensions:
        (discriminative_power, stability). Returns top 10 non‑dominated sorted by
        discriminative_power.
        """
        logger.info("Computing Pareto frontier with %d samples", n_samples)
        samples: list[dict[str, Any]] = []
        for _ in range(n_samples):
            weights = {name: float(np.random.uniform(0.1, 3.0)) for name in self.weight_names}
            scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
            discriminative_power = float(np.std(scores))
            stability = float(1.0 / (1.0 + abs(discriminative_power - 0.5)))
            samples.append(
                {
                    "weights": weights,
                    "discriminative_power": discriminative_power,
                    "stability": stability,
                    "mean_score": float(np.mean(scores)),
                }
            )
        pareto: list[dict[str, Any]] = []
        for i, sample in enumerate(samples):
            dominated = False
            for j, other in enumerate(samples):
                if i == j:
                    continue
                if (
                    other["discriminative_power"] >= sample["discriminative_power"]
                    and other["stability"] >= sample["stability"]
                    and (
                        other["discriminative_power"] > sample["discriminative_power"]
                        or other["stability"] > sample["stability"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(sample)
        pareto.sort(key=lambda x: x["discriminative_power"], reverse=True)
        logger.info("Found %d Pareto-optimal configurations", len(pareto))
        return pareto[:10]

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


def load_episodes_data(path: Path) -> List[Dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
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
    if not isinstance(data, dict):
        raise ValueError("Baseline stats must be a JSON object")
    return data  # type: ignore[return-value]


def _compute_strategy_set(
    recomputer: SNQIWeightRecomputer, strategies: list[str]
) -> Dict[str, Any]:
    out: dict[str, Any] = {}
    for name in strategies:
        try:
            out[name] = recomputer.recompute_with_strategy(name)
        except Exception as e:  # noqa: BLE001
            logger.warning("Strategy %s failed: %s", name, e)
    return out


def _select_recommended(strategy_results: Dict[str, Any]) -> tuple[str, Dict[str, float]]:
    best_name: str | None = None
    best_score = -float("inf")
    for name, data in strategy_results.items():
        stats = data.get("statistics", {}).get("overall", {})
        rng = float(stats.get("range", 0.0) or 0.0)
        std = float(stats.get("std", 0.0) or 0.0)
        score = rng * 0.6 + (1.0 / (1.0 + std)) * 0.4
        if score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        return "", {k: 1.0 for k in WEIGHT_NAMES}
    return best_name, dict(strategy_results[best_name]["weights"])  # copy


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
    runtime = end_perf - start_perf
    results["_metadata"] = {
        "schema_version": 1,
        "generated_at": end_iso,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "start_time": start_iso,
        "end_time": end_iso,
        "runtime_seconds": runtime,
        "provenance": {
            "episodes_file": str(args.episodes),
            "baseline_file": str(args.baseline),
            "invocation": "python " + " ".join(sys.argv),
            "strategy_requested": args.strategy,
            "compare_strategies": args.compare_strategies,
            "compare_normalization": args.compare_normalization,
        },
    }
    # summary filled later after recommended weights known


def _finalize_summary(results: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.compare_strategies:
        method_descriptor = results.get("recommended_strategy")
    else:
        method_descriptor = results.get("strategy_result", {}).get("strategy", args.strategy)
    meta = results.get("_metadata", {})
    results["summary"] = {
        "method": method_descriptor,
        "weights": results.get("recommended_weights"),
        "compare_strategies": args.compare_strategies,
        "compare_normalization": args.compare_normalization,
        "seed": args.seed,
        "has_normalization_comparison": bool(results.get("normalization_comparison")),
        "runtime_seconds": meta.get("runtime_seconds"),
        "start_time": meta.get("start_time"),
        "end_time": meta.get("end_time"),
    }


def _print_summary(results: Dict[str, Any], args: argparse.Namespace, episodes_count: int) -> None:
    print("\nWeight Recomputation Summary:")
    print(f"Episodes analyzed: {episodes_count}")
    if args.compare_strategies:
        print(f"Recommended strategy: {results.get('recommended_strategy')}")
        if "strategy_correlations" in results:
            sorted_corrs = sorted(
                results["strategy_correlations"].items(), key=lambda x: x[1], reverse=True
            )
            print("Strategy correlations (top 3):")
            for pair, corr in sorted_corrs[:3]:
                print(f"  {pair}: {corr:.4f}")
    else:
        print(f"Strategy used: {args.strategy}")
    print("\nRecommended Weights:")
    for w, v in (results.get("recommended_weights") or {}).items():
        print(f"  {w}: {float(v):.3f}")
    if args.compare_normalization and "normalization_comparison" in results:
        print("\nNormalization Strategy Impact:")
        for name, data in results["normalization_comparison"].items():
            if name == "median_p95":
                continue
            corr = data.get("correlation_with_base")
            if corr is not None:
                print(f"  {name}: correlation with base = {corr:.4f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        "--compare-normalization", action="store_true", help="Compare normalization strategies"
    )
    parser.add_argument(
        "--compare-strategies", action="store_true", help="Compare all available strategies"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--validate", action="store_true", help="Validate JSON output structure")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:  # noqa: C901
    start_perf = perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    try:
        episodes = load_episodes_data(args.episodes)
        baseline = load_baseline_stats(args.baseline)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed loading inputs: %s", e)
        return 1
    if not episodes:
        logger.error("No episodes loaded from %s", args.episodes)
        return 1
    if args.seed is not None:
        np.random.seed(args.seed)
    recomputer = SNQIWeightRecomputer(episodes, baseline)
    results: Dict[str, Any] = {}
    if args.compare_strategies:
        all_strategies = ["default", "balanced", "safety_focused", "efficiency_focused", "pareto"]
        strategy_results = _compute_strategy_set(recomputer, all_strategies)
        results["strategy_comparison"] = strategy_results
        # correlations
        correlations: dict[str, float] = {}
        names = list(strategy_results.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1 :]:
                try:
                    correlations[f"{n1}_vs_{n2}"] = recomputer.rank_correlation_analysis(
                        strategy_results[n1]["weights"], strategy_results[n2]["weights"]
                    )
                except Exception as e:  # noqa: BLE001
                    logger.debug("Correlation %s vs %s failed: %s", n1, n2, e)
        results["strategy_correlations"] = correlations
        recommended_name, recommended_weights = _select_recommended(strategy_results)
        results["recommended_strategy"] = recommended_name
        results["recommended_weights"] = recommended_weights
    else:
        single = recomputer.recompute_with_strategy(args.strategy)
        results["strategy_result"] = single
        results["recommended_weights"] = single["weights"]
    # Normalization comparison
    if args.compare_normalization:
        results["normalization_comparison"] = recomputer.compare_normalization_strategies(
            results["recommended_weights"]
        )
    _augment_metadata(results, args, start_iso, start_perf)
    _finalize_summary(results, args)
    try:
        if args.validate:
            validate_snqi(results, "recompute", check_finite=True)
        else:
            assert_all_finite(results)
    except ValueError as e:
        logger.error("Validation failed: %s", e)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", args.output)
    _print_summary(results, args, len(episodes))
    return 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
