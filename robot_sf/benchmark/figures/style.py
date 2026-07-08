"""Publication-grade figure style context for benchmark visualization.

This module provides an opt-in style context for generating publication-ready
figures with consistent typography, colorblind-safe planner palette, and
standardized sizing presets.

Usage:
    from robot_sf.benchmark.figures.style import publication_style, planner_color

    with publication_style(size="single"):
        fig, ax = plt.subplots()
        # ... plot with publication styling ...
"""

from __future__ import annotations

import hashlib
import importlib
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator


# Colorblind-safe palette for planners (Wong 2011 + custom extensions)
# These colors are distinguishable under common color vision deficiencies
_PLANNER_COLORS: dict[str, str] = {
    "goal": "#E69F00",  # Orange
    "orca": "#56B4E9",  # Sky blue
    "social_force": "#009E73",  # Bluish green
    "socnav_sampling": "#F0E442",  # Yellow
    "ppo": "#0072B2",  # Blue
    "sacadrl": "#D55E00",  # Vermillion
    "prediction_planner": "#CC79A7",  # Reddish purple
    "prediction_mpc": "#999999",  # Gray
}

# Fallback palette for unknown planners (cycles through these)
_FALLBACK_PALETTE: list[str] = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
    "#66A61E",
    "#E7298A",
    "#7570B3",
    "#A6761D",
]

# Metric → (display_label, unit) mapping for consistent labeling across figures/tables
# Keys are common metric names as they appear in benchmark records
# Values are (human_readable_label, unit_string) tuples
_METRIC_LABELS: dict[str, tuple[str, str]] = {
    # Collision metrics
    "collision_rate": ("Collision rate", ""),
    "collision_mean": ("Collision rate", ""),
    "collisions": ("Collision rate", ""),
    "collisions_mean": ("Collision rate", ""),
    # Success metrics
    "success_rate": ("Success rate", ""),
    "success_mean": ("Success rate", ""),
    "success": ("Success rate", ""),
    # Time/distance metrics
    "time_to_goal": ("Time to goal", "s"),
    "avg_time_to_goal": ("Time to goal", "s"),
    "traveled_distance": ("Traveled distance", "m"),
    "avg_traveled_distance": ("Traveled distance", "m"),
    "displacement": ("Displacement", "m"),
    # Safety metrics
    "min_ttc": ("Minimum TTC", "s"),
    "min_ttc_mean": ("Minimum TTC", "s"),
    "near_miss_count": ("Near-miss count", ""),
    # Efficiency metrics
    "path_length": ("Path length", "m"),
    "avg_path_length": ("Path length", "m"),
    "efficiency": ("Efficiency", ""),
    # Quality metrics
    "snqi": ("SNQI score", ""),
    "comfort": ("Comfort", ""),
    "smoothness": ("Smoothness", ""),
    # Throughput metrics
    "throughput": ("Throughput", "agents/s"),
    "flow_rate": ("Flow rate", "agents/s"),
    # Episode count
    "episode_count": ("Episode count", ""),
    "total_episodes": ("Episode count", ""),
}


def planner_palette() -> dict[str, str]:
    """Return the canonical planner-to-color mapping.

    Returns:
        Dictionary mapping planner key names to hex color strings.
    """
    return dict(_PLANNER_COLORS)


def planner_color(planner_key: str) -> str:
    """Get the color for a specific planner.

    For known planners, returns the fixed color. For unknown planners,
    returns a deterministic fallback color based on a hash of the key.

    Args:
        planner_key: The planner identifier (e.g., "goal", "orca").

    Returns:
        Hex color string for the planner.
    """
    if planner_key in _PLANNER_COLORS:
        return _PLANNER_COLORS[planner_key]

    # Deterministic fallback: use a stable hash so an unknown planner keeps the
    # same color across processes/figures. Python's builtin hash() is salted per
    # process (PYTHONHASHSEED), so it must NOT be used here — it would give the
    # same planner a different color in each figure-generation run.
    digest = hashlib.sha256(planner_key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:8], "big") % len(_FALLBACK_PALETTE)
    return _FALLBACK_PALETTE[idx]


def metric_label(metric_key: str, *, aggregation: str | None = None) -> str:
    """Get the formatted label for a metric with optional unit and aggregation.

    Args:
        metric_key: The metric name (e.g., "collision_rate", "time_to_goal").
        aggregation: Optional aggregation suffix like "mean" or "median".

    Returns:
        Formatted label string like "Collision rate" or "Time to goal (s)".
        Includes aggregation in parentheses if provided.
    """
    label, unit = _METRIC_LABELS.get(metric_key, (metric_key.replace("_", " ").title(), ""))
    parts = [label]
    if unit:
        parts.append(f"({unit})")
    if aggregation:
        parts.append(f"({aggregation})")
    return " ".join(parts)


def figure_size(size: Literal["single", "double"]) -> tuple[float, float]:
    """Get figure dimensions for publication layout.

    Args:
        size: Either "single" (~3.4 inches) or "double" (~7 inches).

    Returns:
        Tuple of (width, height) in inches.

    Raises:
        ValueError: If size is not "single" or "double".
    """
    if size == "single":
        return (3.4, 2.5)
    elif size == "double":
        return (7.0, 4.0)
    else:
        raise ValueError(f"Invalid figure size: {size!r}. Must be 'single' or 'double'.")


@contextmanager
def publication_style(*, size: Literal["single", "double"] = "single") -> Iterator[None]:
    """Context manager for publication-grade matplotlib styling.

    Applies consistent typography, colorblind-safe colors, and sizing.
    Restores original rcParams on exit.

    Args:
        size: Figure size preset ("single" or "double").

    Yields:
        None (context manager).

    Example:
        with publication_style(size="single"):
            fig, ax = plt.subplots()
            # ... plot ...
    """
    plt = importlib.import_module("matplotlib.pyplot")
    original_params = dict(plt.rcParams)

    try:
        # Apply publication style
        plt.rcParams.update(
            {
                # Typography: serif fonts compatible with LaTeX
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
                "font.size": 10,
                "axes.labelsize": 11,
                "axes.titlesize": 12,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                # Figure sizing
                "figure.figsize": figure_size(size),
                "figure.dpi": 150,
                # Line and marker styles
                "lines.linewidth": 1.5,
                "lines.markersize": 4,
                # Grid and spines
                "axes.grid": False,
                "axes.spines.top": False,
                "axes.spines.right": False,
                # Legend
                "legend.frameon": False,
                "legend.loc": "best",
                # Save settings
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.pad_inches": 0.05,
            }
        )
        yield
    finally:
        # Restore original rcParams
        plt.rcParams.update(original_params)
