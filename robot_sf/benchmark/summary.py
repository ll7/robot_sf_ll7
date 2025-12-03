"""Quick diversity summary utilities.

Reads episode JSONL files and emits simple histograms for key metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np

# Use headless backend for CI/non-GUI
mpl.use("Agg", force=True)
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _iter_records(paths: Sequence[str | Path] | str | Path) -> Iterable[dict[str, Any]]:
    """Iterate over JSONL episode records stored at one or more paths.

    Args:
        paths: Single path or collection of JSONL file paths to scan.

    Yields:
        dict[str, Any]: Parsed JSON record for each non-empty line across the provided files.
    """
    if isinstance(paths, str | Path):
        path_list = [paths]
    else:
        path_list = list(paths)  # type: ignore[arg-type]
    for p in path_list:
        p = Path(p)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _get_nested(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Fetch a dotted-path value from a nested dict, returning a default when missing.

    Args:
        d: Arbitrary nested mapping object to inspect.
        path: Dotted key path (e.g. ``metrics.min_distance``).
        default: Value to return when the path cannot be resolved.

    Returns:
        Any: Resolved value when present; otherwise ``default``.
    """
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _safe_number(x: Any) -> float | None:
    """Convert input to ``float`` while filtering invalid or NaN values.

    Args:
        x: Value to coerce into a float.

    Returns:
        float | None: Parsed float when finite; otherwise ``None``.
    """
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def collect_values(
    records: Iterable[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Collect min_distance and avg_speed from episode records.

    Falls back to computing avg speed from robot_vel if present under
    record["trajectory"]["robot_vel"] as a list of [vx, vy]. If not available,
    the avg_speed entry may be missing.

    Returns
    -------
    tuple[list[float], list[float]]
        Tuple of (min_distances, avg_speeds) collected from records.
    """
    mins: list[float] = []
    speeds: list[float] = []
    for rec in records:
        md = _get_nested(rec, "metrics.min_distance")
        mdv = _safe_number(md)
        if mdv is not None:
            mins.append(mdv)
        av = _get_nested(rec, "metrics.avg_speed")
        avv = _safe_number(av)
        if avv is not None:
            speeds.append(avv)
        else:
            # Optional fallback: derive from trajectory if available
            traj = rec.get("trajectory") or {}
            rv = traj.get("robot_vel")
            if isinstance(rv, list) and rv and isinstance(rv[0], list | tuple):
                try:
                    arr = np.asarray(rv, dtype=float)
                    s = np.linalg.norm(arr, axis=1).mean()
                    if np.isfinite(s):
                        speeds.append(float(s))
                except Exception:
                    pass
    return mins, speeds


def plot_histograms(
    mins: Sequence[float],
    speeds: Sequence[float],
    out_dir: str | Path,
    *,
    bins: int = 30,
) -> list[str]:
    """TODO docstring. Document this function.

    Args:
        mins: TODO docstring.
        speeds: TODO docstring.
        out_dir: TODO docstring.
        bins: TODO docstring.

    Returns:
        TODO docstring.
    """
    out_paths: list[str] = []
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _save(fig, name: str) -> str:
        """TODO docstring. Document this function.

        Args:
            fig: TODO docstring.
            name: TODO docstring.

        Returns:
            TODO docstring.
        """
        path = str(Path(out_dir) / name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)
        return path

    if mins:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(mins, bins=bins, color="#4C78A8", edgecolor="white")
        ax.set_title("Min distance distribution")
        ax.set_xlabel("min_distance [m]")
        ax.set_ylabel("count")
        _save(fig, "hist_min_distance.png")

    if speeds:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(speeds, bins=bins, color="#F58518", edgecolor="white")
        ax.set_title("Average robot speed distribution")
        ax.set_xlabel("avg_speed [m/s]")
        ax.set_ylabel("count")
        _save(fig, "hist_avg_speed.png")

    return out_paths


def summarize_to_plots(paths: Sequence[str | Path] | str | Path, out_dir: str | Path) -> list[str]:
    """TODO docstring. Document this function.

    Args:
        paths: TODO docstring.
        out_dir: TODO docstring.

    Returns:
        TODO docstring.
    """
    mins, speeds = collect_values(_iter_records(paths))
    return plot_histograms(mins, speeds, out_dir)


def compute_sample_efficiency_delta(
    baseline_timesteps: int,
    pretrained_timesteps: int,
) -> dict[str, float]:
    """Compute sample-efficiency improvement metrics.

    Returns:
        Dictionary containing:
        - ratio: pretrained/baseline timesteps (lower is better)
        - reduction_timesteps: absolute timestep savings
        - reduction_percentage: percentage reduction
    """
    if baseline_timesteps == 0:
        return {
            "ratio": 1.0,
            "reduction_timesteps": 0,
            "reduction_percentage": 0.0,
        }

    ratio = pretrained_timesteps / baseline_timesteps
    reduction = baseline_timesteps - pretrained_timesteps
    reduction_pct = 100.0 * (1.0 - ratio)

    return {
        "ratio": ratio,
        "reduction_timesteps": reduction,
        "reduction_percentage": reduction_pct,
    }


def bootstrap_metric_confidence(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """Compute bootstrap confidence intervals for metric aggregates.

    Args:
        values: Sequence of metric values
        confidence: Confidence level (default 0.95 for 95% CI)
        n_samples: Number of bootstrap resamples
        seed: Optional random seed for reproducibility

    Returns:
        Dictionary with keys: mean, median, ci_low, ci_high
    """
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }

    arr = np.array(values)
    rng = np.random.default_rng(seed)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_samples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    # Compute percentiles for confidence interval
    alpha = 1.0 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)

    ci_low = float(np.percentile(bootstrap_means, lower_percentile))
    ci_high = float(np.percentile(bootstrap_means, upper_percentile))

    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def aggregate_training_metrics_with_bootstrap(
    records: Iterable[dict[str, Any]],
    metric_keys: Sequence[str],
    *,
    confidence: float = 0.95,
    n_samples: int = 1000,
    seed: int | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate metrics from training records with bootstrap CIs.

    Args:
        records: Iterable of episode/metric records
        metric_keys: List of metric paths to extract (e.g., "metrics.success_rate")
        confidence: Bootstrap confidence level
        n_samples: Number of bootstrap resamples
        seed: Optional seed for reproducibility

    Returns:
        Dictionary mapping metric key to bootstrap summary dict
    """
    # Collect values for each metric
    metric_values: dict[str, list[float]] = {key: [] for key in metric_keys}

    for rec in records:
        for key in metric_keys:
            value = _get_nested(rec, key)
            num_value = _safe_number(value)
            if num_value is not None:
                metric_values[key].append(num_value)

    # Compute bootstrap summaries
    results = {}
    for key, values in metric_values.items():
        results[key] = bootstrap_metric_confidence(
            values,
            confidence=confidence,
            n_samples=n_samples,
            seed=seed,
        )

    return results


__all__ = [
    "aggregate_training_metrics_with_bootstrap",
    "bootstrap_metric_confidence",
    "collect_values",
    "compute_sample_efficiency_delta",
    "plot_histograms",
    "summarize_to_plots",
]
