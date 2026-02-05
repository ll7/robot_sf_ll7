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
from datetime import (
    UTC,  # type: ignore[attr-defined]
    datetime,
)
from pathlib import Path
from time import perf_counter
from typing import Any

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


def _apply_log_level(level_name: str | None) -> None:
    """Apply log level to root and module loggers from a string name.

    Accepts typical names (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.
    """
    if not level_name:
        level = logging.INFO
    else:
        level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


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

    def __init__(self, episodes_data: list[dict], baseline_stats: dict[str, dict[str, float]]):
        """TODO docstring. Document this function.

        Args:
            episodes_data: TODO docstring.
            baseline_stats: TODO docstring.
        """
        self.episodes = episodes_data
        self.baseline_stats = baseline_stats
        self.weight_names = WEIGHT_NAMES
        self.simplex = False  # toggled externally

    def _maybe_simplex(self, weights: dict[str, float], total: float = 10.0) -> dict[str, float]:
        """TODO docstring. Document this function.

        Args:
            weights: TODO docstring.
            total: TODO docstring.

        Returns:
            TODO docstring.
        """
        if not self.simplex:
            return weights
        s = sum(weights.values())
        if s <= 0:  # pragma: no cover - defensive
            return weights
        return {k: (v / s) * total for k, v in weights.items()}

    def _episode_snqi(self, metrics: dict[str, float], weights: dict[str, float]) -> float:
        """Wrapper calling shared compute_snqi keeping backward compatibility."""
        return compute_snqi(metrics, weights, self.baseline_stats)

    def default_weights(self) -> dict[str, float]:
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

    def balanced_weights(self) -> dict[str, float]:
        """Return balanced weight configuration."""
        return dict.fromkeys(self.weight_names, 1.0)

    def safety_focused_weights(self) -> dict[str, float]:
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

    def efficiency_focused_weights(self) -> dict[str, float]:
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

    def compute_weight_statistics(self, weights: dict[str, float]) -> dict[str, Any]:
        """Compute descriptive statistics for a weight configuration.

        Returns overall distributional stats and per-algorithm aggregates
        (mean, std, count) if algorithm labels exist in episode `scenario_params`.
        """
        weights = self._maybe_simplex(weights)
        scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]

        # Group by algorithm if available
        algo_scores: dict[str, list[float]] = {}
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
        self,
        weights1: dict[str, float],
        weights2: dict[str, float],
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

    def pareto_frontier_weights(self, n_samples: int = 600) -> list[dict[str, Any]]:
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
                },
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

    def recompute_with_strategy(self, strategy: str) -> dict[str, Any]:
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
        weights = self._maybe_simplex(weights)
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

    def compare_normalization_strategies(self, base_weights: dict[str, float]) -> dict[str, Any]:
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
            for metric_name in metric_values:
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


def load_episodes_data(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load episodes returning (episodes, skipped_malformed_lines)."""
    episodes: list[dict[str, Any]] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
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


def load_baseline_stats(path: Path) -> dict[str, dict[str, float]]:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.

    Returns:
        TODO docstring.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Baseline stats must be a JSON object")
    return data  # type: ignore[return-value]


def _compute_strategy_set(
    recomputer: SNQIWeightRecomputer,
    strategies: list[str],
) -> dict[str, Any]:
    """TODO docstring. Document this function.

    Args:
        recomputer: TODO docstring.
        strategies: TODO docstring.

    Returns:
        TODO docstring.
    """
    out: dict[str, Any] = {}
    for name in strategies:
        try:
            out[name] = recomputer.recompute_with_strategy(name)
        except Exception as e:
            logger.warning("Strategy %s failed: %s", name, e)
    return out


def _select_recommended(strategy_results: dict[str, Any]) -> tuple[str, dict[str, float]]:
    """TODO docstring. Document this function.

    Args:
        strategy_results: TODO docstring.

    Returns:
        TODO docstring.
    """
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
        return "", dict.fromkeys(WEIGHT_NAMES, 1.0)
    return best_name, dict(strategy_results[best_name]["weights"])  # copy


def _augment_metadata(
    results: dict[str, Any],
    args: argparse.Namespace,
    start_iso: str,
    start_perf: float,
    phase_timings: dict[str, float] | None = None,
    original_episode_count: int | None = None,
    used_episode_count: int | None = None,
) -> None:
    """TODO docstring. Document this function.

    Args:
        results: TODO docstring.
        args: TODO docstring.
        start_iso: TODO docstring.
        start_perf: TODO docstring.
        phase_timings: TODO docstring.
        original_episode_count: TODO docstring.
        used_episode_count: TODO docstring.
    """

    def _git_commit() -> str:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:
            return "UNKNOWN"

    end_perf = perf_counter()
    end_iso = datetime.now(UTC).isoformat()
    runtime = end_perf - start_perf
    meta: dict[str, Any] = {
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
    if original_episode_count is not None:
        meta["original_episode_count"] = original_episode_count
    if used_episode_count is not None:
        meta["used_episode_count"] = used_episode_count
    if phase_timings:
        meta["phase_timings"] = {k: phase_timings[k] for k in sorted(phase_timings)}
    results["_metadata"] = meta
    # summary filled later after recommended weights known


def _finalize_summary(results: dict[str, Any], args: argparse.Namespace) -> None:
    """TODO docstring. Document this function.

    Args:
        results: TODO docstring.
        args: TODO docstring.
    """
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
            "baseline_missing_metric_count",
        ),
        "skipped_malformed_lines": 0,  # populated later
    }


def _print_summary(results: dict[str, Any], args: argparse.Namespace, episodes_count: int) -> None:
    """TODO docstring. Document this function.

    Args:
        results: TODO docstring.
        args: TODO docstring.
        episodes_count: TODO docstring.
    """
    lines: list[str] = []
    lines.append("Weight Recomputation Summary:")
    lines.append(f"Episodes analyzed: {episodes_count}")
    if args.compare_strategies:
        lines.append(f"Recommended strategy: {results.get('recommended_strategy')}")
        if "strategy_correlations" in results:
            sorted_corrs = sorted(
                results["strategy_correlations"].items(),
                key=lambda x: x[1],
                reverse=True,
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
    """TODO docstring. Document this function.

    Args:
        argv: TODO docstring.

    Returns:
        TODO docstring.
    """
    parser = argparse.ArgumentParser(description="SNQI Weight Recomputation")
    parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
        help="Path to episode data JSONL file",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline statistics JSON file",
    )
    parser.add_argument(
        "--strategy",
        choices=["default", "balanced", "safety_focused", "efficiency_focused", "pareto"],
        default="default",
        help="Weight recomputation strategy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for recomputed weights JSON",
    )
    parser.add_argument(
        "--compare-normalization",
        action="store_true",
        help="Compare normalization strategies",
    )
    parser.add_argument(
        "--compare-strategies",
        action="store_true",
        help="Compare all available strategies",
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
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Deterministically sample N episodes prior to analysis (for speed)",
    )
    parser.add_argument(
        "--simplex",
        action="store_true",
        help="Project strategy and external weight vectors onto a simplex (sum constant) before evaluation",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Number of bootstrap resamples for stability/CI estimation (0 disables)",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap intervals (e.g., 0.95)",
    )
    parser.add_argument(
        "--small-dataset-threshold",
        type=int,
        default=20,
        help=(
            "Warn when the number of episodes used is below this threshold "
            "(stability and CIs may be unreliable)."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        default="INFO",
        help="Logging verbosity (default: INFO)",
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
                },
            )
    return {"total_missing": len(results), "metrics": results}


def run(args: argparse.Namespace) -> int:  # noqa: C901,PLR0912,PLR0915
    """Recompute SNQI weights from episode data and baseline stats.

    Args:
        args: Parsed CLI arguments controlling sampling, weighting, and output.

    Returns:
        Exit code (0 on success; non-zero on input/processing errors).
    """
    start_perf = perf_counter()
    start_iso = datetime.now(UTC).isoformat()
    phase_start = start_perf
    phase_timings: dict[str, float] = {}
    try:
        episodes, skipped_lines = load_episodes_data(args.episodes)
        baseline = load_baseline_stats(args.baseline)
    except Exception as e:
        logger.exception("Failed loading inputs: %s", e)
        return EXIT_INPUT_ERROR
    phase_timings["load_inputs"] = perf_counter() - phase_start
    if not episodes:
        logger.error("No episodes loaded from %s", args.episodes)
        return EXIT_INPUT_ERROR
    original_episode_count = len(episodes)
    if args.sample is not None and args.sample > 0 and args.sample < len(episodes):
        rng = np.random.default_rng(args.seed if args.seed is not None else 1337)
        idx = rng.choice(len(episodes), size=args.sample, replace=False)
        episodes = [episodes[i] for i in sorted(idx.tolist())]
        logger.info(
            "Sampled %d/%d episodes (--sample) for recomputation",
            len(episodes),
            original_episode_count,
        )
    used_episode_count = len(episodes)
    if args.seed is not None:
        np.random.seed(args.seed)
    # Warn on small dataset sizes which can make statistics unstable
    try:
        threshold = int(getattr(args, "small_dataset_threshold", 20))
    except Exception:
        threshold = 20
    if used_episode_count < threshold:
        logger.warning(
            "Small dataset: using %d episodes (< %d). Stability and bootstrap CIs may be unreliable.",
            used_episode_count,
            threshold,
        )
    try:
        recomputer = SNQIWeightRecomputer(episodes, baseline)
        recomputer.simplex = bool(args.simplex)
        external_weights: dict[str, float] | None = None
        if args.external_weights_file is not None:
            try:
                with open(args.external_weights_file, encoding="utf-8") as f:
                    raw = json.load(f)
                if not isinstance(raw, dict):  # pragma: no cover - defensive
                    raise ValueError("External weights file must be a JSON object")
                external_weights = _validate_weights_mapping(raw)
                if args.simplex:
                    s = sum(external_weights.values())
                    if s > 0:
                        external_weights = {k: (v / s) * 10.0 for k, v in external_weights.items()}
                logger.info("Loaded external weights from %s", args.external_weights_file)
            except Exception as e:
                logger.exception("Failed loading external weights: %s", e)
                return EXIT_INPUT_ERROR
        results: dict[str, Any] = {}
        if args.compare_strategies:
            phase_start = perf_counter()
            all_strategies = [
                "default",
                "balanced",
                "safety_focused",
                "efficiency_focused",
                "pareto",
            ]
            strategy_results = _compute_strategy_set(recomputer, all_strategies)
            phase_timings["strategy_comparison"] = perf_counter() - phase_start
            results["strategy_comparison"] = strategy_results
            correlations: dict[str, float] = {}
            names = list(strategy_results.keys())
            phase_start = perf_counter()
            for i, n1 in enumerate(names):
                for n2 in names[i + 1 :]:
                    try:
                        correlations[f"{n1}_vs_{n2}"] = recomputer.rank_correlation_analysis(
                            strategy_results[n1]["weights"],
                            strategy_results[n2]["weights"],
                        )
                    except Exception as e:
                        logger.debug("Correlation %s vs %s failed: %s", n1, n2, e)
            results["strategy_correlations"] = correlations
            phase_timings["strategy_correlations"] = perf_counter() - phase_start
            recommended_name, recommended_weights = _select_recommended(strategy_results)
            results["recommended_strategy"] = recommended_name
            results["recommended_weights"] = recommended_weights
        else:
            phase_start = perf_counter()
            single = recomputer.recompute_with_strategy(args.strategy)
            phase_timings["single_strategy"] = perf_counter() - phase_start
            results["strategy_result"] = single
            results["recommended_weights"] = single["weights"]
        if args.compare_normalization:
            phase_start = perf_counter()
            results["normalization_comparison"] = recomputer.compare_normalization_strategies(
                results["recommended_weights"],
            )
            phase_timings["normalization_comparison"] = perf_counter() - phase_start
        # Bootstrap confidence intervals (optional)
        if getattr(args, "bootstrap_samples", 0) and args.bootstrap_samples > 0:
            try:
                phase_start = perf_counter()
                bs_rng = np.random.default_rng(args.seed if args.seed is not None else 1234)
                rec_w = results.get("recommended_weights") or {}
                episode_scores = [
                    recomputer._episode_snqi(ep.get("metrics", {}), rec_w)
                    for ep in recomputer.episodes
                ]
                episode_scores = [s for s in episode_scores if np.isfinite(s)]
                n = len(episode_scores)
                if n:
                    reps = args.bootstrap_samples
                    means: list[float] = []
                    for _ in range(reps):
                        idx = bs_rng.integers(0, n, size=n)
                        sample = [episode_scores[i] for i in idx]
                        means.append(float(np.mean(sample)))
                    means_arr = np.array(means, dtype=float)
                    alpha = 1 - float(getattr(args, "bootstrap_confidence", 0.95))
                    lower = float(np.percentile(means_arr, 100 * (alpha / 2)))
                    upper = float(np.percentile(means_arr, 100 * (1 - alpha / 2)))
                    results.setdefault("bootstrap", {})["recommended_score"] = {
                        "samples": reps,
                        "mean_mean": float(np.mean(means_arr)),
                        "std_mean": float(np.std(means_arr, ddof=1)) if reps > 1 else 0.0,
                        "ci": [lower, upper],
                        "confidence_level": float(getattr(args, "bootstrap_confidence", 0.95)),
                    }
                phase_timings["bootstrap"] = perf_counter() - phase_start
            except Exception as e:
                logger.warning("Bootstrap computation failed: %s", e)
        # Baseline missing metric diagnostics
        phase_start = perf_counter()
        missing_info = _detect_missing_baseline_metrics(
            episodes,
            baseline,
            args.missing_metric_max_list,
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
        phase_timings["diagnostics"] = perf_counter() - phase_start
        if external_weights is not None:
            phase_start = perf_counter()
            ext_stats = recomputer.compute_weight_statistics(external_weights)
            results["external_weights"] = {
                "weights": external_weights,
                "statistics": ext_stats,
                "correlation_with_recommended": recomputer.rank_correlation_analysis(
                    results["recommended_weights"],
                    external_weights,
                ),
            }
            phase_timings["external_weights_eval"] = perf_counter() - phase_start
        _augment_metadata(
            results,
            args,
            start_iso,
            start_perf,
            phase_timings,
            original_episode_count=original_episode_count,
            used_episode_count=used_episode_count,
        )
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
            phase_start = perf_counter()
            if args.validate:
                validate_snqi(results, "recompute", check_finite=True)
            else:
                assert_all_finite(results)
        except ValueError as e:
            logger.exception("Validation failed: %s", e)
            return EXIT_VALIDATION_ERROR
        phase_timings["validation"] = perf_counter() - phase_start
        phase_start = perf_counter()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        phase_timings["write_output"] = perf_counter() - phase_start
        logger.info("Results saved to %s", args.output)
        if phase_timings:
            timing_lines = ["Phase timings (seconds):"] + [
                f"  {k}: {v:.4f}" for k, v in sorted(phase_timings.items())
            ]
            logger.info("\n%s", "\n".join(timing_lines))
        _print_summary(results, args, len(episodes))
        return EXIT_SUCCESS
    except Exception as e:
        logger.exception("Unexpected runtime failure: %s", e)
        return EXIT_RUNTIME_ERROR


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """TODO docstring. Document this function.

    Args:
        argv: TODO docstring.

    Returns:
        TODO docstring.
    """
    args = parse_args(argv)
    _apply_log_level(getattr(args, "log_level", None))
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
