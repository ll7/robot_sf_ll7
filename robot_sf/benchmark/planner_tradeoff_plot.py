"""Planner safety-efficiency tradeoff plotting for publication bundles.

The helpers in this module turn a camera-ready/publication benchmark bundle into
a scatter plot of collision rate versus success rate.  They intentionally read
the same bundle artifacts used by downstream paper repositories:

- ``payload/reports/campaign_table.csv`` for planner-level means
- ``payload/runs/<planner>*/episodes.jsonl`` for seed-level bootstrap intervals

This keeps paper-facing figures reproducible from Robot SF artifacts without
requiring a separate manuscript repository.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from robot_sf.benchmark.figures.style import planner_color, publication_style
from robot_sf.benchmark.utils import episode_collision_value, episode_success_value

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


DEFAULT_PLANNER_LABELS: dict[str, str] = {
    "goal": "goal",
    "orca": "orca",
    "ppo": "ppo",
    "social_force": "social_force",
    "prediction_planner": "prediction",
    "sacadrl": "sacadrl",
    "socnav_sampling": "socnav_samp.",
}

DEFAULT_HEADLINE_PLANNERS = ("orca", "ppo")
DEFAULT_CONTROL_PLANNERS = ("goal",)
DEFAULT_BOOTSTRAP_SAMPLES = 400
DEFAULT_CI_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SEED = 42

# Role-based color mapping using colorblind-safe palette
# Headline planners use their specific planner colors from the shared palette
# Control uses the goal planner color, experimental uses a neutral gray
_ROLE_COLORS: dict[str, str] = {
    "headline_orca": planner_color("orca"),  # Sky blue
    "headline_ppo": planner_color("ppo"),  # Blue
    "control": planner_color("goal"),  # Orange
    "experimental": "#999999",  # Neutral gray
}


@dataclass(frozen=True)
class PlannerTradeoffPoint:
    """One planner point plus optional bootstrap confidence intervals."""

    planner_key: str
    label: str
    role: str
    success_mean: float
    collision_mean: float
    success_ci: tuple[float, float] | None = None
    collision_ci: tuple[float, float] | None = None
    episodes_path: Path | None = None


def load_campaign_table(bundle_path: Path) -> list[dict[str, str]]:
    """Load planner rows from ``payload/reports/campaign_table.csv``.

    Older and newer bundles differ slightly in collision column naming.  This
    reader normalizes ``collisions_mean`` to ``collision_mean`` for the plotting
    layer while preserving the rest of the row values.

    Returns:
        Campaign table rows keyed by CSV header.
    """
    csv_path = bundle_path / "payload" / "reports" / "campaign_table.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"campaign_table.csv not found at {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        if "collision_mean" not in row and "collisions_mean" in row:
            row["collision_mean"] = row["collisions_mean"]
    return rows


def read_publication_run_id(bundle_path: Path) -> str:
    """Return the best available run identifier for a publication bundle."""
    manifest_path = bundle_path / "publication_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = payload.get("provenance", {}).get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    return bundle_path.name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL episode records from ``path``.

    Returns:
        Parsed JSON objects in file order.
    """
    if not path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    if not rows:
        raise ValueError(f"No episode records found at {path}")
    return rows


def find_planner_episodes_path(bundle_path: Path, planner_key: str) -> Path | None:
    """Find the episode JSONL for ``planner_key`` inside a bundle.

    Returns:
        Matching ``episodes.jsonl`` path, or ``None`` when absent.
    """
    runs_dir = bundle_path / "payload" / "runs"
    candidates = [*sorted(runs_dir.glob(f"{planner_key}__*")), runs_dir / planner_key]
    for candidate in candidates:
        episodes_path = candidate / "episodes.jsonl"
        if episodes_path.exists():
            return episodes_path
    return None


def _episode_metric_value(episode: dict[str, Any], metric: str) -> float | None:
    """Extract a supported logical metric from an episode record.

    Returns:
        Parsed metric value, or ``None`` when the metric is unavailable.
    """
    if metric == "success":
        value = episode_success_value(episode)
        return float(value) if value is not None else None
    if metric == "collisions":
        outcome = episode.get("outcome")
        if isinstance(outcome, Mapping) and "collision_event" in outcome:
            return 1.0 if bool(outcome.get("collision_event")) else 0.0
        value = episode_collision_value(episode)
        return float(value) if value is not None else None
    metrics = episode.get("metrics")
    if isinstance(metrics, Mapping):
        raw = metrics.get(metric)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def bootstrap_seed_mean_ci(
    episodes_jsonl: Path,
    metric: str,
    *,
    samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CI_CONFIDENCE,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Bootstrap a confidence interval over per-seed metric means.

    The sampling unit is the seed-level mean, not the individual episode.  This
    matches the AMV paper plotting convention and avoids overstating precision
    when scenarios share the same deterministic seed schedule.

    Returns:
        Lower and upper confidence interval bounds.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    by_seed: dict[int, list[float]] = {}
    for episode in _read_jsonl(episodes_jsonl):
        value = _episode_metric_value(episode, metric)
        if value is None:
            continue
        by_seed.setdefault(int(episode["seed"]), []).append(value)

    seed_means = np.array(
        [float(np.mean(values)) for values in by_seed.values() if values],
        dtype=float,
    )
    if seed_means.size == 0:
        raise ValueError(f"Metric {metric!r} could not be extracted from {episodes_jsonl}")

    rng = np.random.default_rng(seed)
    draws = np.array(
        [rng.choice(seed_means, size=seed_means.size, replace=True).mean() for _ in range(samples)],
        dtype=float,
    )
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.percentile(draws, 100.0 * alpha)),
        float(np.percentile(draws, 100.0 * (1.0 - alpha))),
    )


def build_tradeoff_points(
    bundle_path: Path,
    *,
    headline_planners: Sequence[str] = DEFAULT_HEADLINE_PLANNERS,
    control_planners: Sequence[str] = DEFAULT_CONTROL_PLANNERS,
    labels: Mapping[str, str] | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci_confidence: float = DEFAULT_CI_CONFIDENCE,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[PlannerTradeoffPoint]:
    """Build planner tradeoff points from a publication bundle.

    Returns:
        Planner points ready for plotting.
    """
    label_map = {**DEFAULT_PLANNER_LABELS, **(dict(labels) if labels else {})}
    headline = set(headline_planners)
    controls = set(control_planners)
    points: list[PlannerTradeoffPoint] = []

    for row in load_campaign_table(bundle_path):
        planner_key = str(row.get("planner_key", "")).strip()
        if not planner_key:
            continue
        if "success_mean" not in row or "collision_mean" not in row:
            raise ValueError(
                "campaign_table.csv must contain planner_key, success_mean, "
                "and collision_mean/collisions_mean columns"
            )

        episodes_path = find_planner_episodes_path(bundle_path, planner_key)
        success_ci = None
        collision_ci = None
        if episodes_path is not None:
            success_ci = bootstrap_seed_mean_ci(
                episodes_path,
                "success",
                samples=bootstrap_samples,
                confidence=ci_confidence,
                seed=bootstrap_seed,
            )
            collision_ci = bootstrap_seed_mean_ci(
                episodes_path,
                "collisions",
                samples=bootstrap_samples,
                confidence=ci_confidence,
                seed=bootstrap_seed,
            )

        if planner_key in headline:
            role = "headline"
        elif planner_key in controls:
            role = "control"
        else:
            role = "experimental"

        points.append(
            PlannerTradeoffPoint(
                planner_key=planner_key,
                label=label_map.get(planner_key, planner_key),
                role=role,
                success_mean=float(row["success_mean"]),
                collision_mean=float(row["collision_mean"]),
                success_ci=success_ci,
                collision_ci=collision_ci,
                episodes_path=episodes_path,
            )
        )

    if not points:
        raise ValueError(f"No planner rows found in bundle {bundle_path}")
    return points


def _finite_error(mean: float, ci: tuple[float, float] | None) -> tuple[float, float] | None:
    """Return asymmetric error-bar widths, or ``None`` for missing/non-finite CIs."""
    if ci is None:
        return None
    low, high = ci
    if not all(math.isfinite(v) for v in (mean, low, high)):
        return None
    return (max(0.0, mean - low), max(0.0, high - mean))


def plot_planner_tradeoff(
    points: Sequence[PlannerTradeoffPoint],
    *,
    title: str | None = "Preferred region: lower collision, higher success",
) -> plt.Figure:
    """Render a planner safety-efficiency tradeoff figure.

    Returns:
        Matplotlib figure with the plotted planner tradeoff.
    """
    plt.switch_backend("Agg")

    # Use publication style context for consistent styling
    # Create a custom size that matches the original figure dimensions
    with publication_style(size="double"):
        # Override figure size to match original (5.2, 3.6)
        mpl.rcParams["figure.figsize"] = (5.2, 3.6)
        mpl.rcParams["axes.grid"] = True
        mpl.rcParams["grid.alpha"] = 0.2
        mpl.rcParams["axes.titlesize"] = 9

        fig, ax = plt.subplots()

        for index, point in enumerate(points):
            x = point.collision_mean
            y = point.success_mean
            if point.role == "headline":
                color = _ROLE_COLORS.get(f"headline_{point.planner_key}", planner_color("orca"))
                xerr = _finite_error(x, point.collision_ci)
                yerr = _finite_error(y, point.success_ci)
                ax.errorbar(
                    x,
                    y,
                    xerr=([[xerr[0]], [xerr[1]]] if xerr else None),
                    yerr=([[yerr[0]], [yerr[1]]] if yerr else None),
                    fmt="o",
                    color=color,
                    markersize=7,
                    capsize=3,
                    linewidth=1.2,
                    zorder=4,
                )
            elif point.role == "control":
                color = _ROLE_COLORS["control"]
                ax.plot(
                    x,
                    y,
                    "o",
                    color="white",
                    markeredgecolor=color,
                    markeredgewidth=1.4,
                    markersize=7,
                    zorder=4,
                )
            else:
                color = _ROLE_COLORS["experimental"]
                ax.plot(x, y, "^", color=color, markersize=6, alpha=0.9, zorder=3)

            # Deterministic, dependency-free label staggering.  It is less elaborate
            # than adjustText, but keeps the core package free of another plotting
            # dependency while avoiding exact label overlap in common bundles.
            offset_y = 0.012 if index % 2 == 0 else -0.012
            ax.annotate(
                point.label,
                xy=(x, y),
                xytext=(5, 6 if offset_y > 0 else -8),
                textcoords="offset points",
                fontsize=8.0 if point.role == "experimental" else 7.5,
                color=color,
                alpha=0.95,
            )

        ax.set_xlabel("Collision rate (lower is safer)")
        ax.set_ylabel("Success rate (higher is better)")
        ax.set_xlim(left=-0.02)
        ax.set_ylim(bottom=-0.02)
        if title:
            ax.set_title(title, loc="left", color="#4a4a4a")

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_ROLE_COLORS["headline_orca"],
                markersize=6,
                label="orca (main)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=_ROLE_COLORS["headline_ppo"],
                markersize=6,
                label="ppo (main)",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor=_ROLE_COLORS["control"],
                markeredgewidth=1.2,
                markersize=6,
                label="goal (control)",
            ),
            Line2D(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor=_ROLE_COLORS["experimental"],
                markersize=6,
                label="experimental planners",
            ),
        ]
        ax.legend(handles=legend_elements, loc="best", frameon=True, framealpha=0.9)
        fig.tight_layout()

    return fig


def save_planner_tradeoff_figure(
    bundle_path: Path,
    *,
    out_png: Path,
    out_pdf: Path | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    ci_confidence: float = DEFAULT_CI_CONFIDENCE,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    title: str | None = "Preferred region: lower collision, higher success",
) -> dict[str, Any]:
    """Build and save a planner tradeoff figure from a publication bundle.

    Returns:
        Metadata for the saved figure and plotted planner points.
    """
    points = build_tradeoff_points(
        bundle_path,
        bootstrap_samples=bootstrap_samples,
        ci_confidence=ci_confidence,
        bootstrap_seed=bootstrap_seed,
    )
    fig = plot_planner_tradeoff(points, title=title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf)
    plt.close(fig)
    return {
        "run_id": read_publication_run_id(bundle_path),
        "bundle_path": str(bundle_path),
        "png": str(out_png),
        "pdf": str(out_pdf) if out_pdf is not None else None,
        "bootstrap_samples": bootstrap_samples,
        "ci_confidence": ci_confidence,
        "points": [
            {
                "planner_key": point.planner_key,
                "label": point.label,
                "role": point.role,
                "success_mean": point.success_mean,
                "collision_mean": point.collision_mean,
                "success_ci": point.success_ci,
                "collision_ci": point.collision_ci,
                "episodes_path": str(point.episodes_path) if point.episodes_path else None,
            }
            for point in points
        ],
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CI_CONFIDENCE",
    "PlannerTradeoffPoint",
    "bootstrap_seed_mean_ci",
    "build_tradeoff_points",
    "find_planner_episodes_path",
    "load_campaign_table",
    "plot_planner_tradeoff",
    "read_publication_run_id",
    "save_planner_tradeoff_figure",
]
