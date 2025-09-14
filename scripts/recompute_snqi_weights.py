#!/usr/bin/env python3
"""SNQI Weight Recomputation Script.

High-level responsibilities:
1. Load episodic benchmark data + baseline normalization statistics.
2. Generate candidate weight configurations via predefined strategies
     (default, balanced, safety_focused, efficiency_focused, pareto heuristic).
3. Optionally compare *all* strategies and compute pairwise rank correlations.
4. Optionally compare alternative normalization strategies (median/p95 vs variants).
5. Optionally evaluate an externally supplied weight JSON (sanity/what-if).
6. Emit a JSON document with `_metadata` (schema + provenance) and a compact `summary`.

Determinism:
Passing `--seed` seeds NumPy for Pareto sampling and any stochastic choices;
outputs embed seed + git commit for reproducibility.

Design Notes:
- Strategy correlation currently based on per-episode ranking Spearman correlations.
- Pareto frontier is a lightweight heuristic (sample + dominance filter) and
    may be replaced by a true multi-objective algorithm in future versions.
- Normalization comparison can highlight sensitivity to percentile choices.

See design doc section 8 / 8.1 for heuristic rationale.
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
from robot_sf.benchmark.snqi.exit_codes import (
    EXIT_INPUT_ERROR,
    EXIT_MISSING_METRIC_ERROR,
    EXIT_RUNTIME_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
)
from robot_sf.benchmark.snqi.schema import assert_all_finite, validate_snqi
from robot_sf.benchmark.snqi.weights_validation import (
    validate_weights_mapping as _validate_weights_mapping,
)

# Setup logging (single configuration point)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SNQIWeightRecomputer:
    """Recompute SNQI weights via predefined strategies and analyses.

    Provides helpers for deriving candidate weight sets and computing
    statistics under the canonical SNQI scoring function.

    Parameters
    ----------
    episodes_data:
        List of episode dictionaries (each must contain a `metrics` mapping).
    baseline_stats:
        Mapping of metric -> {"med": float, "p95": float} used for normalization.
    """

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
        """Compute descriptive statistics for a weight configuration.

        Returns overall distributional stats and per-algorithm aggregates
        (mean, std, count) if algorithm labels exist in episode `scenario_params`.
        """
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
        """Compute Spearman rank correlation between two weight configurations.

        Ranks are derived from per-episode SNQI scores for each weight set.
        Returns 1.0 when correlation is undefined (degenerate identical ranks).
        """
        scores1 = [self._episode_snqi(ep.get("metrics", {}), weights1) for ep in self.episodes]
        scores2 = [self._episode_snqi(ep.get("metrics", {}), weights2) for ep in self.episodes]

        ranking1 = np.argsort(np.argsort(scores1))
        ranking2 = np.argsort(np.argsort(scores2))

        from scipy.stats import spearmanr

        corr, _ = spearmanr(ranking1, ranking2)
        return corr if not np.isnan(corr) else 1.0

    def pareto_frontier_weights(self, n_samples: int = 600) -> List[Dict[str, Any]]:
        """Approximate a Pareto frontier by random sampling.

        Samples `n_samples` random weight vectors (uniform per weight), computes
        discriminative power (score std) and a heuristic stability proxy
        ``1 / (1 + |std - 0.5|)`` then applies dominance filtering.

        Returns
        -------
        list of dict
            Up to 10 non-dominated configurations sorted by discriminative power.
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
        """Run a single strategy pipeline returning weights + stats.

        Falls back to default weights if Pareto sampling produces no candidates.
        """
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
        """Compare alternative percentile-based normalization strategies.

        Generates median/p90 and interquartile (p25/p75) baselines for metrics
        present in the episodes and reports mean score + correlation versus
        canonical median/p95. Returns mapping of strategy name -> summary stats.
        """
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


def load_episodes_data(path: Path) -> tuple[List[Dict[str, Any]], int]:
    """Load episodes returning (episodes, skipped_malformed_lines)."""
    episodes: list[dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    episodes.append(obj)
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
                logger.debug("Malformed JSON (line %d) in %s", line_no, path)
    if skipped:
        logger.info("Skipped %d malformed/invalid lines in %s", skipped, path)
    return episodes, skipped


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
        "baseline_missing_metric_count": results.get("_metadata", {}).get(
            "baseline_missing_metric_count"
        ),
        "skipped_malformed_lines": 0,  # populated later
    }


def _print_summary(results: Dict[str, Any], args: argparse.Namespace, episodes_count: int) -> None:
    lines: list[str] = []
    lines.append("Weight Recomputation Summary:")
    lines.append(f"Episodes analyzed: {episodes_count}")
    if args.compare_strategies:
        lines.append(f"Recommended strategy: {results.get('recommended_strategy')}")
        if "strategy_correlations" in results:
            sorted_corrs = sorted(
                results["strategy_correlations"].items(), key=lambda x: x[1], reverse=True
            )
            lines.append("Strategy correlations (top 3):")
            for pair, corr in sorted_corrs[:3]:
                lines.append(f"  {pair}: {corr:.4f}")
    else:
        lines.append(f"Strategy used: {args.strategy}")
    lines.append("Recommended Weights:")
    for w, v in (results.get("recommended_weights") or {}).items():
        lines.append(f"  {w}: {float(v):.3f}")
    if args.compare_normalization and "normalization_comparison" in results:
        lines.append("Normalization Strategy Impact:")
        for name, data in results["normalization_comparison"].items():
            if name == "median_p95":
                continue
            corr = data.get("correlation_with_base")
            if corr is not None:
                lines.append(f"  {name}: correlation with base = {corr:.4f}")
    logger.info("\n%s", "\n".join(lines))


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
    parser.add_argument(
        "--external-weights-file",
        type=Path,
        default=None,
        help="Optional JSON file with externally supplied weights to evaluate (validated)",
    )
    parser.add_argument(
        "--missing-metric-max-list",
        type=int,
        default=5,
        help="Max example episode IDs to list per missing baseline metric",
    )
    parser.add_argument(
        "--fail-on-missing-metric",
        action="store_true",
        help="Treat presence of baseline-missing metrics as an error (non-zero exit)",
    )
    return parser.parse_args(argv)


def _detect_missing_baseline_metrics(
    episodes: list[dict[str, Any]],
    baseline_stats: dict[str, dict[str, float]],
    max_examples: int = 5,
) -> dict[str, Any]:
    """Detect metrics present in episodes but absent from baseline stats.

    Returns structure with total_missing and per-metric counts + example IDs.
    Metrics checked correspond to those normalized in `compute_snqi`.
    """
    normalized_metrics = [
        "collisions",
        "near_misses",
        "force_exceed_events",
        "jerk_mean",
    ]
    results: list[dict[str, Any]] = []
    for metric in normalized_metrics:
        if metric in baseline_stats:
            continue
        episodes_with_metric: list[str] = []
        count = 0
        for ep in episodes:
            metrics = ep.get("metrics", {}) or {}
            if metric in metrics:
                count += 1
                if len(episodes_with_metric) < max_examples:
                    ep_id = (
                        str(ep.get("scenario_id"))
                        or str(ep.get("id"))
                        or str(len(episodes_with_metric))
                    )
                    episodes_with_metric.append(ep_id)
        if count:
            results.append(
                {
                    "name": metric,
                    "episode_count_with_metric": count,
                    "example_episode_ids": episodes_with_metric,
                }
            )
    return {"total_missing": len(results), "metrics": results}


def run(args: argparse.Namespace) -> int:  # noqa: C901
    start_perf = perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    try:
        episodes, skipped_lines = load_episodes_data(args.episodes)
        baseline = load_baseline_stats(args.baseline)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed loading inputs: %s", e)
        return EXIT_INPUT_ERROR
    if not episodes:
        logger.error("No episodes loaded from %s", args.episodes)
        return EXIT_INPUT_ERROR
    if args.seed is not None:
        np.random.seed(args.seed)
    try:
        recomputer = SNQIWeightRecomputer(episodes, baseline)
        external_weights: Dict[str, float] | None = None
        if args.external_weights_file is not None:
            try:
                with open(args.external_weights_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if not isinstance(raw, dict):  # pragma: no cover - defensive
                    raise ValueError("External weights file must be a JSON object")
                external_weights = _validate_weights_mapping(raw)
                logger.info("Loaded external weights from %s", args.external_weights_file)
            except Exception as e:  # noqa: BLE001
                logger.error("Failed loading external weights: %s", e)
                return EXIT_INPUT_ERROR
        results: Dict[str, Any] = {}
        if args.compare_strategies:
            all_strategies = [
                "default",
                "balanced",
                "safety_focused",
                "efficiency_focused",
                "pareto",
            ]
            strategy_results = _compute_strategy_set(recomputer, all_strategies)
            results["strategy_comparison"] = strategy_results
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
        if args.compare_normalization:
            results["normalization_comparison"] = recomputer.compare_normalization_strategies(
                results["recommended_weights"]
            )
        # Baseline missing metric diagnostics
        missing_info = _detect_missing_baseline_metrics(
            episodes, baseline, args.missing_metric_max_list
        )
        results.setdefault("diagnostics", {})["baseline_missing_metrics"] = missing_info
        if missing_info["total_missing"]:
            logger.warning(
                "Baseline missing %d metric(s) present in episodes: %s",
                missing_info["total_missing"],
                ", ".join(m["name"] for m in missing_info["metrics"]),
            )
            if args.fail_on_missing_metric:
                logger.error("Failing due to --fail-on-missing-metric (missing baseline metrics).")
                return EXIT_MISSING_METRIC_ERROR
        if external_weights is not None:
            ext_stats = recomputer.compute_weight_statistics(external_weights)
            results["external_weights"] = {
                "weights": external_weights,
                "statistics": ext_stats,
                "correlation_with_recommended": recomputer.rank_correlation_analysis(
                    results["recommended_weights"], external_weights
                ),
            }
        _augment_metadata(results, args, start_iso, start_perf)
        # Record skipped malformed lines
        results.setdefault("_metadata", {})["skipped_malformed_lines"] = skipped_lines
        # Record missing metric count in metadata for summary consumption
        results.setdefault("_metadata", {})["baseline_missing_metric_count"] = missing_info[
            "total_missing"
        ]
        _finalize_summary(results, args)
        if "summary" in results:
            # Ensure key presence even if zero for contract stability
            results["summary"]["skipped_malformed_lines"] = skipped_lines
            results["summary"]["baseline_missing_metric_count"] = missing_info["total_missing"]
        try:
            if args.validate:
                validate_snqi(results, "recompute", check_finite=True)
            else:
                assert_all_finite(results)
        except ValueError as e:
            logger.error("Validation failed: %s", e)
            return EXIT_VALIDATION_ERROR
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", args.output)
        _print_summary(results, args, len(episodes))
        return EXIT_SUCCESS
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected runtime failure: %s", e)
        return EXIT_RUNTIME_ERROR


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
