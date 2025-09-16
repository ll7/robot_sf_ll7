#!/usr/bin/env python3
"""SNQI Weight Optimization Script.

Responsibilities:
1. Load episodes + baseline normalization statistics.
2. Optimize weight vector using grid search, differential evolution, or both.
3. Optionally run one-step local sensitivity analysis around recommended weights.
4. Emit reproducibility metadata (`_metadata`) and a concise `summary`.

Determinism:
Passing `--seed` seeds NumPy and SciPy differential evolution (forwarded via `seed`).

Heuristic Objective:
`0.6 * stability + 0.4 * discriminative_power` (see design doc §8.1). Stability falls
back to a variance-derived proxy when insufficient algorithm groups are present.
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
from typing import Any, Callable, Dict, Iterable, Iterator, List

import numpy as np
from scipy.optimize import differential_evolution

from robot_sf.benchmark.snqi import WEIGHT_NAMES, compute_snqi
from robot_sf.benchmark.snqi.exit_codes import (
    EXIT_INPUT_ERROR,
    EXIT_MISSING_METRIC_ERROR,
    EXIT_SUCCESS,
    EXIT_VALIDATION_ERROR,
)
from robot_sf.benchmark.snqi.schema import assert_all_finite, validate_snqi
from robot_sf.benchmark.snqi.weights_validation import (
    validate_weights_mapping as _validate_weights_mapping,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _apply_log_level(level_name: str | None) -> None:
    """Apply log level to root and module loggers from a string name (default INFO)."""
    if not level_name:
        level = logging.INFO
    else:
        level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


# ----------------------------- Progress helper ----------------------------- #
try:  # pragma: no cover - optional dependency
    from tqdm import tqdm

    def _progress(
        iterable: Iterable, desc: str | None = None, total: int | None = None
    ) -> Iterator:  # type: ignore[name-defined]
        return tqdm(iterable, desc=desc, total=total)  # type: ignore[no-any-return]

    _TQDM_AVAILABLE = True
except Exception:  # noqa: BLE001 - pragma: no cover

    def _progress(
        iterable: Iterable, desc: str | None = None, total: int | None = None
    ) -> Iterator:  # type: ignore[unused-argument]
        # Fallback: ignore progress parameters when tqdm unavailable
        return iter(iterable)

    _TQDM_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Container for weight optimization results."""

    weights: Dict[str, float]
    objective_value: float
    ranking_stability: float
    convergence_info: Dict[str, Any]
    objective_components: Dict[str, float] | None = None


class SNQIWeightOptimizer:
    """Optimize SNQI weights via grid search and/or differential evolution.

    Parameters
    ----------
    episodes_data:
        List of episode dicts each containing a `metrics` mapping.
    baseline_stats:
        Mapping metric -> {"med": float, "p95": float} for normalization.
    """

    def __init__(self, episodes_data: List[Dict], baseline_stats: Dict[str, Dict[str, float]]):
        self.episodes = episodes_data
        self.baseline_stats = baseline_stats
        self.weight_names = list(WEIGHT_NAMES)

    def _maybe_simplex(
        self, weights: Dict[str, float], simplex: bool, total: float = 10.0
    ) -> Dict[str, float]:
        if not simplex:
            return weights
        s = sum(weights.values())
        if s <= 0:  # pragma: no cover - defensive
            return weights
        return {k: (v / s) * total for k, v in weights.items()}

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

    def objective_function(self, weight_vector: np.ndarray, *, simplex: bool = False) -> float:
        """Heuristic objective (negative for minimizer)."""
        weights = dict(zip(self.weight_names, weight_vector))
        if simplex:
            weights = self._maybe_simplex(weights, True)
        stability = self.compute_ranking_stability(weights)
        scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        discriminative_power = min(1.0, score_variance / 0.5)
        return -(0.6 * stability + 0.4 * discriminative_power)

    def grid_search_optimization(
        self,
        grid_resolution: int = 5,
        max_combinations: int | None = None,
        show_progress: bool = False,
        simplex: bool = False,
    ) -> OptimizationResult:
        """Perform (possibly guarded) grid search over weight space.

        Shrinks resolution if raw combinations exceed `max_combinations`; if still
        excessive then samples a deterministic subset (fixed RNG seed) to bound runtime.
        Returns best weight configuration under heuristic objective.
        """
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
        iterator: Iterable = combos_iter
        if show_progress:
            materialized = list(combos_iter) if not isinstance(combos_iter, list) else combos_iter
            iterator = _progress(materialized, desc="grid", total=len(materialized))
        for combo in iterator:
            weights = {k: float(v) for k, v in zip(self.weight_names, combo)}
            obj = self.objective_function(np.array(list(weights.values())), simplex=simplex)
            if obj < best_obj:
                best_obj = obj
                best_weights = self._maybe_simplex(weights, simplex)
                best_stability = self.compute_ranking_stability(weights)
            evaluations += 1
        if best_weights is None:  # Fallback (should not happen)
            best_weights = {k: 1.0 for k in self.weight_names}
        # Convert objective back to positive score (we minimized negative)
        positive_score = -best_obj
        best_scores = [
            self._episode_snqi(ep.get("metrics", {}), best_weights) for ep in self.episodes
        ]
        best_var = np.var(best_scores) if len(best_scores) > 1 else 0.0
        best_discriminative = min(1.0, best_var / 0.5)
        return OptimizationResult(
            weights=best_weights,
            objective_value=positive_score,
            ranking_stability=best_stability,
            convergence_info={
                "evaluations": evaluations,
                "grid_resolution": grid_resolution,
                "total_combinations_considered": evaluations,
                "simplex": simplex,
            },
            objective_components={
                "stability_component": float(best_stability),
                "discriminative_component": float(best_discriminative),
            },
        )

    # --------------------- Differential Evolution helpers -------------------- #
    def _build_de_callback(
        self,
        *,
        maxiter: int,
        show_progress: bool,
        simplex: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,
    ) -> tuple[Callable | None, Any | None]:  # pragma: no cover - trivial glue
        """Create (callback, progress_bar) pair for differential evolution.

        The callback encapsulates both progress updates and early stopping logic.
        Returns (None, None) when neither feature is requested. Keeping this
        logic isolated reduces branching inside the main optimization method.
        """
        use_progress = show_progress and _TQDM_AVAILABLE
        wants_early_stop = early_stop_patience > 0
        if not use_progress and not wants_early_stop:
            return None, None

        pbar = tqdm(total=maxiter, desc="evolution") if use_progress else None
        best_positive_obj: float | None = None
        stagnant_iters = 0

        def _callback(xk: np.ndarray, _convergence: float) -> bool:  # noqa: D401
            nonlocal best_positive_obj, stagnant_iters
            current_positive = -self.objective_function(xk, simplex=simplex)
            improved = (
                best_positive_obj is None
                or current_positive - best_positive_obj >= early_stop_min_delta
            )
            if improved:
                best_positive_obj = current_positive
                stagnant_iters = 0
            else:
                stagnant_iters += 1
            if pbar is not None:
                pbar.update(1)
            if wants_early_stop and stagnant_iters >= early_stop_patience:
                if pbar is not None:
                    pbar.set_description("evolution (early-stop)")
                return True
            return False

        return _callback, pbar

    def _assemble_de_result(
        self,
        *,
        result: Any,
        simplex: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,
        maxiter: int,
    ) -> OptimizationResult:
        """Convert SciPy result object into `OptimizationResult` (pure helper)."""
        weights = dict(zip(self.weight_names, result.x))
        if simplex:
            weights = self._maybe_simplex(weights, True)
        stability = self.compute_ranking_stability(weights)
        positive_score = -result.fun
        convergence = {
            "nit": result.nit,
            "nfev": result.nfev,
            "success": bool(result.success),
            "message": result.message,
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta,
            "early_stopped": bool(early_stop_patience > 0 and result.nit < maxiter),
        }
        evo_scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
        evo_var = np.var(evo_scores) if len(evo_scores) > 1 else 0.0
        evo_discriminative = min(1.0, evo_var / 0.5)
        return OptimizationResult(
            weights=weights,
            objective_value=positive_score,
            ranking_stability=stability,
            convergence_info=convergence,
            objective_components={
                "stability_component": float(stability),
                "discriminative_component": float(evo_discriminative),
            },
        )

    def differential_evolution_optimization(
        self,
        maxiter: int = 30,
        seed: int | None = None,
        show_progress: bool = False,
        simplex: bool = False,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 1e-4,
    ) -> OptimizationResult:
        """Run differential evolution to optimize weights (thin wrapper).

        Full implementation lives in ``_differential_evolution_core`` to keep
        wrapper complexity below the C901 threshold while retaining a clear
        public method signature for external callers.
        """
        return self._differential_evolution_core(
            maxiter=maxiter,
            seed=seed,
            show_progress=show_progress,
            simplex=simplex,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
        )

    def _differential_evolution_core(
        self,
        *,
        maxiter: int,
        seed: int | None,
        show_progress: bool,
        simplex: bool,
        early_stop_patience: int,
        early_stop_min_delta: float,
    ) -> OptimizationResult:  # noqa: C901 - acceptable internal complexity
        bounds = [(0.1, 3.0)] * len(self.weight_names)
        callback, pbar = self._build_de_callback(
            maxiter=maxiter,
            show_progress=show_progress,
            simplex=simplex,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
        )
        result = self._run_de(bounds, maxiter, seed, simplex, callback, pbar)
        return self._assemble_de_result(
            result=result,
            simplex=simplex,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            maxiter=maxiter,
        )

    def _run_de(
        self,
        bounds: list[tuple[float, float]],
        maxiter: int,
        seed: int | None,
        simplex: bool,
        callback: Callable | None,
        pbar: Any | None,
    ) -> Any:  # pragma: no cover - thin wrapper
        """Execute SciPy differential_evolution with provided configuration."""

        def _wrapped(vec: np.ndarray) -> float:
            return self.objective_function(vec, simplex=simplex)

        result = differential_evolution(
            _wrapped,
            bounds=bounds,
            maxiter=maxiter,
            seed=seed,
            polish=True,
            strategy="best1bin",
            callback=callback,
        )
        if pbar is not None:
            pbar.close()
        return result

    def sensitivity_analysis(
        self, weights: Dict[str, float], show_progress: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """Compute simple local one-at-a-time sensitivity metrics.

        Perturbs each weight ±10% (clamped to domain) and measures change in
        mean SNQI score; returns per-weight sensitivity info.
        """
        base_scores = [self._episode_snqi(ep.get("metrics", {}), weights) for ep in self.episodes]
        base_mean = float(np.mean(base_scores)) if base_scores else 0.0
        results: Dict[str, Dict[str, float]] = {}
        iterator = weights.items()
        if show_progress:
            iterator = _progress(list(weights.items()), desc="sensitivity")  # type: ignore[assignment]
        for name, value in iterator:  # type: ignore[misc]
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


def _load_initial_weights(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):  # pragma: no cover - defensive
        raise ValueError("Initial weights file must be a JSON object")
    return _validate_weights_mapping(raw)


# ---------------------------- I/O helpers ---------------------------- #
def load_episodes_data(path: Path) -> tuple[List[Dict[str, Any]], int]:
    """Load episodes from JSONL file returning (episodes, skipped_malformed)."""
    episodes: List[Dict[str, Any]] = []
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
    if not isinstance(data, dict):  # basic validation
        raise ValueError("Baseline stats file must contain a JSON object")
    return data  # type: ignore[return-value]


# ------------------------- Orchestration helpers --------------------- #
def _load_inputs(
    args: argparse.Namespace,
) -> tuple[list[dict], dict[str, dict[str, float]], int]:
    episodes, skipped = load_episodes_data(args.episodes)
    baseline_stats = load_baseline_stats(args.baseline)
    return episodes, baseline_stats, skipped


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
    results: Dict[str, Any],
    args: argparse.Namespace,
    start_iso: str,
    start_perf: float,
    phase_timings: dict[str, float] | None = None,
    original_episode_count: int | None = None,
    used_episode_count: int | None = None,
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
            "simplex": bool(args.simplex),
        },
    }
    if original_episode_count is not None:
        results_meta["original_episode_count"] = original_episode_count
    if used_episode_count is not None:
        results_meta["used_episode_count"] = used_episode_count
    if phase_timings:
        # Sort timings for stable output ordering
        results_meta["phase_timings"] = {k: phase_timings[k] for k in sorted(phase_timings)}
    results["_metadata"] = results_meta
    recommended = results["recommended"]
    results["summary"] = {
        "method": recommended.get("method_used"),
        "objective_value": recommended.get("objective_value"),
        "ranking_stability": recommended.get("ranking_stability"),
        "weights": recommended.get("weights"),
        "objective_components": recommended.get("objective_components"),
        "available_methods": [
            k for k in results.keys() if k in ("grid_search", "differential_evolution")
        ],
        "seed": args.seed,
        "has_sensitivity": bool(results.get("sensitivity_analysis")),
        "runtime_seconds": runtime_seconds,
        "start_time": start_iso,
        "end_time": end_iso,
        "skipped_malformed_lines": 0,  # populated later
    }


def _detect_missing_baseline_metrics(
    episodes: list[dict[str, Any]],
    baseline_stats: dict[str, dict[str, float]],
    max_examples: int = 5,
) -> dict[str, Any]:
    """Detect metrics present in episodes but absent from baseline stats.

    Only checks metrics that participate in normalization (where a missing
    baseline leads to silent zero penalty). Returns structured info:

    {
        "total_missing": int,
        "metrics": [
            {
                "name": <metric>,
                "episode_count_with_metric": n,
                "example_episode_ids": [...]
            }, ...
        ]
    }
    """
    # Metrics whose normalization relies on baseline entries
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
    phase_start = start_perf
    phase_timings: dict[str, float] = {}
    try:
        episodes, baseline_stats, skipped_lines = _load_inputs(args)
    except Exception as e:  # noqa: BLE001
        logger.error("Failed loading inputs: %s", e)
        return EXIT_INPUT_ERROR
    phase_timings["load_inputs"] = perf_counter() - phase_start
    if not episodes:
        logger.error("No valid episodes found in data file")
        return EXIT_INPUT_ERROR
    original_episode_count = len(episodes)
    if args.sample is not None and args.sample > 0 and args.sample < len(episodes):
        # Deterministic sampling using provided seed else fixed fallback
        rng = np.random.default_rng(args.seed if args.seed is not None else 1337)
        indices = rng.choice(len(episodes), size=args.sample, replace=False)
        # Preserve order deterministically by sorting selected indices
        episodes = [episodes[i] for i in sorted(indices.tolist())]
        logger.info(
            "Sampled %d/%d episodes (--sample) for optimization",
            len(episodes),
            original_episode_count,
        )
    used_episode_count = len(episodes)
    if args.seed is not None:
        np.random.seed(args.seed)
    # Warn on small dataset sizes which can reduce reliability of statistics and CIs
    try:
        threshold = int(getattr(args, "small_dataset_threshold", 20))
    except Exception:  # noqa: BLE001 - defensive
        threshold = 20
    if used_episode_count < threshold:
        logger.warning(
            "Small dataset: using %d episodes (< %d). Stability and bootstrap CIs may be unreliable.",
            used_episode_count,
            threshold,
        )
    optimizer = SNQIWeightOptimizer(episodes, baseline_stats)
    initial_weights: Dict[str, float] | None = None
    if args.initial_weights_file is not None:
        try:
            initial_weights = _load_initial_weights(args.initial_weights_file)
            logger.info("Loaded initial weights from %s", args.initial_weights_file)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed loading initial weights: %s", e)
            return EXIT_INPUT_ERROR
    results: Dict[str, Any] = {}
    if args.method in ["grid", "both"]:
        phase_start = perf_counter()
        grid_result = optimizer.grid_search_optimization(
            args.grid_resolution,
            max_combinations=args.max_grid_combinations,
            show_progress=args.progress,
            simplex=args.simplex,
        )
        phase_timings["grid_search"] = perf_counter() - phase_start
        results["grid_search"] = {
            "weights": grid_result.weights,
            "objective_value": grid_result.objective_value,
            "ranking_stability": grid_result.ranking_stability,
            "convergence_info": grid_result.convergence_info,
            "objective_components": grid_result.objective_components,
        }
    if args.method in ["evolution", "both"]:
        phase_start = perf_counter()
        evolution_result = optimizer.differential_evolution_optimization(
            args.maxiter, seed=args.seed, show_progress=args.progress, simplex=args.simplex
        )
        phase_timings["differential_evolution"] = perf_counter() - phase_start
        results["differential_evolution"] = {
            "weights": evolution_result.weights,
            "objective_value": evolution_result.objective_value,
            "ranking_stability": evolution_result.ranking_stability,
            "convergence_info": evolution_result.convergence_info,
            "objective_components": evolution_result.objective_components,
        }
    _select_best(results, args.method)
    if initial_weights is not None:
        results["initial_weights"] = initial_weights
    if args.sensitivity:
        phase_start = perf_counter()
        recommended_weights = results["recommended"]["weights"]
        results["sensitivity_analysis"] = optimizer.sensitivity_analysis(
            recommended_weights, show_progress=args.progress
        )
        phase_timings["sensitivity_analysis"] = perf_counter() - phase_start
    # Detect baseline missing metrics (episodes contain metric but baseline lacks med/p95)
    phase_start = perf_counter()
    missing_info = _detect_missing_baseline_metrics(
        episodes, baseline_stats, args.missing_metric_max_list
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
    _augment_metadata(
        results,
        args,
        start_iso,
        start_perf,
        phase_timings,
        original_episode_count=original_episode_count,
        used_episode_count=used_episode_count,
    )
    # Bootstrap confidence intervals (optional)
    if getattr(args, "bootstrap_samples", 0) and args.bootstrap_samples > 0:
        try:
            phase_start = perf_counter()
            bs_rng = np.random.default_rng(args.seed if args.seed is not None else 1234)
            rec_weights = results.get("recommended", {}).get("weights")
            if rec_weights:
                episode_scores = [
                    optimizer._episode_snqi(ep.get("metrics", {}), rec_weights)  # noqa: SLF001
                    for ep in optimizer.episodes
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
        except Exception as e:  # noqa: BLE001
            logger.warning("Bootstrap computation failed: %s", e)
    if args.ci_placeholder:
        # Confidence interval scaffold referencing planned bootstrap integration
        results.setdefault("_metadata", {})["confidence_intervals_placeholder"] = {
            "status": "placeholder",
            "method": "bootstrap_future",
            "details": {
                "message": "CI computation not yet implemented; this is a forward-compatible scaffold.",
                "planned_bootstrap_function": "robot_sf.benchmark.snqi.bootstrap.bootstrap_stability",
            },
        }
    results.setdefault("_metadata", {})["skipped_malformed_lines"] = skipped_lines
    results.setdefault("_metadata", {})["baseline_missing_metric_count"] = missing_info[
        "total_missing"
    ]
    if "summary" in results:
        results["summary"]["baseline_missing_metric_count"] = missing_info["total_missing"]
        # Ensure presence of key even before file write (tests expect it)
        results["summary"]["skipped_malformed_lines"] = skipped_lines
    try:
        phase_start = perf_counter()
        if args.validate:
            validate_snqi(results, "optimization", check_finite=True)
        else:
            assert_all_finite(results)
    except ValueError as e:
        logger.error("Validation failed: %s", e)
        return EXIT_VALIDATION_ERROR
    phase_timings["validation"] = perf_counter() - phase_start
    phase_start = perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # Inject skipped count into summary after augmentation
    if "summary" in results:
        results["summary"]["skipped_malformed_lines"] = skipped_lines
    logger.info("Results saved to %s (skipped malformed lines: %d)", args.output, skipped_lines)
    phase_timings["write_output"] = perf_counter() - phase_start
    # Optionally log phase timings at INFO if user wants progress; else DEBUG
    if phase_timings:
        timing_lines = ["Phase timings (seconds):"] + [
            f"  {k}: {v:.4f}" for k, v in sorted(phase_timings.items())
        ]
        logger.info("\n%s", "\n".join(timing_lines))
    _print_summary(results, args)
    return EXIT_SUCCESS


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
    parser.add_argument(
        "--initial-weights-file",
        type=Path,
        default=None,
        help="Path to JSON file containing initial/seed weights mapping (validated)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars (requires tqdm; silently ignored if unavailable)",
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
        help="Deterministically sample N episodes before optimization (for faster experimentation)",
    )
    parser.add_argument(
        "--simplex",
        action="store_true",
        help="Project candidate weight vectors onto a simplex (sum constant) before evaluation",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Early stopping patience (iterations without improvement in positive objective). 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum positive objective improvement to reset early stopping patience.",
    )
    parser.add_argument(
        "--ci-placeholder",
        action="store_true",
        help="(DEPRECATED) Include placeholder CI scaffold; use --bootstrap-samples / --bootstrap-confidence instead",
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


def main(argv: List[str] | None = None) -> int:  # pragma: no cover
    args = parse_args(argv)
    _apply_log_level(getattr(args, "log_level", None))
    return run(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
