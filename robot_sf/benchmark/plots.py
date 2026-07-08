"""Benchmark plotting helpers (Pareto fronts).

Create Pareto scatter plots to visualize trade-offs between two metrics.
"""

from __future__ import annotations

import gc
import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from robot_sf.benchmark.aggregate import (
    ensure_observation_track_policy,
    normalize_observation_track_mode,
    observation_track_group_label,
)
from robot_sf.benchmark.figures.style import metric_label, publication_style
from robot_sf.benchmark.grouping import resolve_report_group_key

if TYPE_CHECKING:
    from collections.abc import Iterable

try:
    from robot_sf.benchmark.plotting_style import apply_latex_style
except ImportError:  # pragma: no cover - optional styling helper
    apply_latex_style = None  # type: ignore[assignment]

Record = dict[str, object]


def _get_dotted(d: dict[str, object], path: str, default=None):
    """Get nested dict value via dotted path.

    Args:
        d: Dictionary to navigate.
        path: Dot-separated key path (e.g., "metrics.success").
        default: Value to return if path is not found.

    Returns:
        The value at the dotted path, or default if not found.
    """
    cur: object = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]  # type: ignore[index]
    return cur


def _group_values(
    records: Iterable[Record],
    group_by: str,
    fallback_group_by: str,
    metric: str,
    observation_track_mode: str = "strict",
    *,
    mode: str | None = None,
) -> dict[str, list[float]]:
    """Collect metric values grouped by a dotted key.

    Returns:
        Mapping of group id to list of metric values.
    """
    record_list = [dict(record) for record in records]
    if mode is None:
        track_meta = ensure_observation_track_policy(
            record_list,
            observation_track_mode=observation_track_mode,
        )
        mode = normalize_observation_track_mode(str(track_meta["mode"]))
    out: dict[str, list[float]] = {}
    for r in record_list:
        g = resolve_report_group_key(
            r,
            group_by=group_by,
            fallback_group_by=fallback_group_by,
            missing="skip",
        )
        val = _get_dotted(r, f"metrics.{metric}")
        if g is None or val is None:
            continue
        try:
            fv = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        key = observation_track_group_label(r, str(g), mode=mode)
        out.setdefault(key, []).append(fv)
    return out


def compute_pareto_points(
    records: Iterable[Record],
    x_metric: str,
    y_metric: str,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    agg: str = "mean",
    observation_track_mode: str = "strict",
) -> tuple[list[tuple[float, float]], list[str]]:
    """Compute per-group points (x, y) using agg over metric values.

    Returns
    - points: list of (x, y)
    - labels: matching list of group labels
    """
    record_list = [dict(record) for record in records]
    track_meta = ensure_observation_track_policy(
        record_list,
        observation_track_mode=observation_track_mode,
    )
    mode = normalize_observation_track_mode(str(track_meta["mode"]))
    gx = _group_values(
        record_list,
        group_by,
        fallback_group_by,
        x_metric,
        mode=mode,
    )
    gy = _group_values(
        record_list,
        group_by,
        fallback_group_by,
        y_metric,
        mode=mode,
    )
    labels: list[str] = []
    points: list[tuple[float, float]] = []

    def reducer(vals: list[float]) -> float:
        """Aggregate values using configured aggregation method.

        Args:
            vals: List of metric values to aggregate.

        Returns:
            Aggregated value (mean or median based on agg parameter).
        """
        if agg == "median":
            return float(np.median(vals))
        return float(np.mean(vals))

    for g, xs in gx.items():
        ys = gy.get(g)
        if not ys:
            continue
        labels.append(g)
        points.append((reducer(xs), reducer(ys)))
    return points, labels


def _maybe_apply_latex_style() -> None:
    """Attempt to import and apply LaTeX plotting style; no-op if unavailable.

    Kept as a separate helper to reduce complexity in plotting functions and to
    isolate optional dependency handling.
    """
    if apply_latex_style is None:
        return
    try:
        apply_latex_style()
    except (AttributeError, TypeError, ValueError, RuntimeError):
        # If the helper misbehaves, silently continue using defaults.
        return


def _dominates(
    a: tuple[float, float],
    b: tuple[float, float],
    x_higher_better: bool,
    y_higher_better: bool,
) -> bool:
    """Return True if point a Pareto-dominates point b."""
    ax, ay = a
    bx, by = b
    # Normalize to "lower is better" by flipping signs if higher is better
    axn = -ax if x_higher_better else ax
    ayn = -ay if y_higher_better else ay
    bxn = -bx if x_higher_better else bx
    byn = -by if y_higher_better else by
    return (axn <= bxn and ayn <= byn) and (axn < bxn or ayn < byn)


def pareto_front_indices(
    points: list[tuple[float, float]],
    x_higher_better: bool = False,
    y_higher_better: bool = False,
) -> list[int]:
    """Return indices of non-dominated points using simple O(n^2) check.

    Returns:
        List of indices corresponding to Pareto-optimal points.
    """
    n = len(points)
    idxs = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if _dominates(points[j], points[i], x_higher_better, y_higher_better):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs


def save_pareto_png(  # noqa: PLR0913
    records: Iterable[Record],
    out_path: str,
    x_metric: str,
    y_metric: str,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    agg: str = "mean",
    x_higher_better: bool = False,
    y_higher_better: bool = False,
    title: str | None = None,
    out_pdf: str | None = None,
    observation_track_mode: str = "strict",
) -> dict[str, object]:
    """Render and save a Pareto scatter with non-dominated points highlighted.

    When out_pdf is provided, also save a LaTeX-friendly vector PDF with consistent rcParams.

    Returns:
        Metadata dict with plot info, point counts, and output paths.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    points, labels = compute_pareto_points(
        records,
        x_metric,
        y_metric,
        group_by,
        fallback_group_by,
        agg,
        observation_track_mode,
    )
    if not points:
        raise ValueError("No points available for Pareto plot (check metrics and grouping).")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    front = pareto_front_indices(
        points,
        x_higher_better=x_higher_better,
        y_higher_better=y_higher_better,
    )

    # Use publication style context for consistent styling
    with publication_style(size="single"):
        fig, ax = plt.subplots()
        # All points - use neutral gray from colorblind-safe palette
        ax.scatter(xs, ys, c="#999999", label="Groups", s=24, alpha=0.7, linewidths=0.6)
        # Frontier - use vermilion from colorblind-safe palette for emphasis
        fxs = [xs[i] for i in front]
        fys = [ys[i] for i in front]
        ax.scatter(fxs, fys, c="#D55E00", label="Pareto front", s=36, marker="^", linewidths=0.8)

        # Use formatted metric labels with units
        ax.set_xlabel(metric_label(x_metric, aggregation=agg))
        ax.set_ylabel(metric_label(y_metric, aggregation=agg))
        if title:
            ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

        # Save PNG
        fig.savefig(out_path, dpi=150)

        if out_pdf is not None:
            # Save vector PDF for LaTeX inclusion
            pdf_dir = os.path.dirname(out_pdf)
            if pdf_dir:
                os.makedirs(pdf_dir, exist_ok=True)
            fig.savefig(out_pdf)

        plt.close(fig)

    # Force garbage collection to reduce memory footprint in long CI runs
    try:
        gc.collect()
    except Exception:
        pass

    front_labels = [labels[i] for i in front]
    payload: dict[str, object] = {
        "count": len(points),
        "front_size": len(front),
        "front_labels": front_labels,
    }
    if out_pdf is not None:
        payload["pdf"] = out_pdf
    return payload
