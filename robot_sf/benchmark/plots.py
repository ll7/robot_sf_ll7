"""Benchmark plotting helpers (Pareto fronts).

Create Pareto scatter plots to visualize trade-offs between two metrics.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

Record = dict[str, object]


def _get_dotted(d: dict[str, object], path: str, default=None):
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
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in records:
        g = _get_dotted(r, group_by)
        if g is None:
            g = _get_dotted(r, fallback_group_by)
        val = _get_dotted(r, f"metrics.{metric}")
        if g is None or val is None:
            continue
            try:
                fv = float(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            out.setdefault(str(g), []).append(fv)
    return out


def compute_pareto_points(
    records: Iterable[Record],
    x_metric: str,
    y_metric: str,
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
    agg: str = "mean",
) -> tuple[list[tuple[float, float]], list[str]]:
    """Compute per-group points (x, y) using agg over metric values.

    Returns
    - points: list of (x, y)
    - labels: matching list of group labels
    """
    gx = _group_values(records, group_by, fallback_group_by, x_metric)
    gy = _group_values(records, group_by, fallback_group_by, y_metric)
    labels: list[str] = []
    points: list[tuple[float, float]] = []

    def reducer(vals: list[float]) -> float:
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


def _dominates(
    a: tuple[float, float],
    b: tuple[float, float],
    x_higher_better: bool,
    y_higher_better: bool,
) -> bool:
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
    """Return indices of non-dominated points using simple O(n^2) check."""
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


def save_pareto_png(
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
) -> dict[str, object]:
    """Render and save a Pareto scatter with non-dominated points highlighted.

    When out_pdf is provided, also save a LaTeX-friendly vector PDF with consistent rcParams.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    # Style/rcParams for consistent figure exports
    try:
        from robot_sf.benchmark.plotting_style import apply_latex_style as _apply_latex_style
    except ImportError:
        _apply_latex_style = None
    if _apply_latex_style is not None:
        try:
            _apply_latex_style()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            # Fallback: keep defaults if helper misbehaves
            pass
    points, labels = compute_pareto_points(
        records,
        x_metric,
        y_metric,
        group_by,
        fallback_group_by,
        agg,
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

    plt.figure(figsize=(6, 4))
    # All points
    plt.scatter(xs, ys, c="#888", label="Groups", s=24, alpha=0.7, linewidths=0.6)
    # Frontier
    fxs = [xs[i] for i in front]
    fys = [ys[i] for i in front]
    plt.scatter(fxs, fys, c="#d62728", label="Pareto front", s=36, marker="^", linewidths=0.8)

    plt.xlabel(f"{x_metric} ({agg})")
    plt.ylabel(f"{y_metric} ({agg})")
    if title:
        plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if out_pdf is not None:
        # Save vector PDF for LaTeX inclusion
        pdf_dir = os.path.dirname(out_pdf)
        if pdf_dir:
            os.makedirs(pdf_dir, exist_ok=True)
        plt.savefig(out_pdf)
    plt.close()

    front_labels = [labels[i] for i in front]
    payload: dict[str, object] = {
        "count": len(points),
        "front_size": len(front),
        "front_labels": front_labels,
    }
    if out_pdf is not None:
        payload["pdf"] = out_pdf
    return payload
