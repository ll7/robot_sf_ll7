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

from robot_sf.benchmark.errors import EpisodeRecordInputError

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _path_list(paths: Sequence[str | Path] | str | Path) -> list[Path]:
    """Normalize one or more path-like inputs into ``Path`` objects.

    Returns:
        List of paths to read.
    """
    if isinstance(paths, str | Path):
        return [Path(paths)]
    return [Path(path) for path in paths]  # type: ignore[arg-type]


def _raise_record_input_error(
    *,
    missing_paths: list[Path],
    malformed_lines: list[tuple[Path, int, str]],
    fallback_errors: list[str] | None = None,
) -> None:
    """Raise a compact error summarizing benchmark input data-loss risks."""
    fallback_errors = fallback_errors or []
    if not missing_paths and not malformed_lines and not fallback_errors:
        return
    details: list[str] = [
        "Episode JSONL input is not suitable for benchmark evidence",
        f"missing_paths={len(missing_paths)}",
        f"malformed_lines={len(malformed_lines)}",
        f"fallback_derivation_errors={len(fallback_errors)}",
    ]
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        details.append(f"missing=[{preview}]")
    if malformed_lines:
        preview = "; ".join(
            f"{path}:{line_number}: {message}" for path, line_number, message in malformed_lines[:5]
        )
        details.append(f"malformed=[{preview}]")
    if fallback_errors:
        preview = "; ".join(fallback_errors[:5])
        details.append(f"fallback_errors=[{preview}]")
    raise EpisodeRecordInputError("; ".join(details))


def load_episode_records(
    paths: Sequence[str | Path] | str | Path,
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Load episode records from JSONL paths.

    Args:
        paths: Single path or collection of JSONL file paths to scan.
        strict: When true, fail closed on missing paths and malformed JSONL lines. When false,
            skip malformed or missing inputs for explicitly exploratory/advisory use.

    Returns:
        Parsed episode records.
    """
    records: list[dict[str, Any]] = []
    missing_paths: list[Path] = []
    malformed_lines: list[tuple[Path, int, str]] = []

    for path in _path_list(paths):
        if not path.exists():
            if strict:
                missing_paths.append(path)
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    if strict:
                        malformed_lines.append((path, line_number, exc.msg))

    _raise_record_input_error(missing_paths=missing_paths, malformed_lines=malformed_lines)
    return records


def _iter_records(
    paths: Sequence[str | Path] | str | Path,
    *,
    strict: bool = True,
) -> Iterable[dict[str, Any]]:
    """Iterate over JSONL episode records stored at one or more paths.

    Args:
        paths: Single path or collection of JSONL file paths to scan.
        strict: When true, fail closed on malformed or missing input.

    Yields:
        dict[str, Any]: Parsed JSON record for each non-empty line across the provided files.
    """
    yield from load_episode_records(paths, strict=strict)


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
    except (ValueError, TypeError):
        return None


def collect_values(
    records: Iterable[dict[str, Any]],
    *,
    strict: bool = True,
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
    fallback_errors: list[str] = []
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
                except (ValueError, TypeError) as exc:
                    if strict:
                        record_id = rec.get("episode_id") or rec.get("scenario_id") or "<unknown>"
                        fallback_errors.append(f"{record_id}: {exc}")
    _raise_record_input_error(
        missing_paths=[],
        malformed_lines=[],
        fallback_errors=fallback_errors,
    )
    return mins, speeds


def plot_histograms(
    mins: Sequence[float],
    speeds: Sequence[float],
    out_dir: str | Path,
    *,
    bins: int = 30,
) -> list[str]:
    """Plot histograms for min distance and average speed.

    Returns:
        List of written image paths.
    """
    out_paths: list[str] = []
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _save(fig, name: str) -> str:
        """Save a histogram figure and return its path.

        Returns:
            Path string for the saved figure.
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


def summarize_to_plots(
    paths: Sequence[str | Path] | str | Path,
    out_dir: str | Path,
    *,
    strict: bool = True,
) -> list[str]:
    """Load episodes and write summary histogram plots.

    Returns:
        List of written image paths.
    """
    mins, speeds = collect_values(_iter_records(paths, strict=strict), strict=strict)
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
