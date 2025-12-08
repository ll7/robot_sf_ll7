"""Off-screen chart rendering helpers for telemetry panels and headless exports.

Uses matplotlib with the Agg backend to produce RGBA pixel buffers that can be
blitted into Pygame surfaces or written to disk. Designed to stay dependency-
light and headless-friendly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import numpy as np

# Force headless-friendly backend (Agg) regardless of global defaults.
matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

try:  # Optional convenience when pygame is available
    import pygame
except ImportError:  # pragma: no cover - optional dependency
    pygame = None

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


DEFAULT_TELEMETRY_METRICS = ("fps", "reward", "collisions", "min_ped_distance", "action_norm")


def render_metric_panel(
    series: Mapping[str, Sequence[float]],
    metrics: Sequence[str] | None = None,
    *,
    width: int = 640,
    height: int = 360,
    dpi: int = 100,
) -> np.ndarray:
    """Render stacked line charts for the requested metrics and return an RGBA image array.

    Args:
        series: Mapping from metric name to numeric sequence (newest last).
        metrics: Metrics to render (defaults to common telemetry metrics).
        width: Target panel width in pixels.
        height: Target panel height in pixels.
        dpi: Matplotlib DPI; combined with width/height to size the figure.

    Returns:
        NumPy uint8 array shaped (H, W, 4) in RGBA order suitable for blitting via pygame.surfarray.
    """
    metrics = tuple(metrics or DEFAULT_TELEMETRY_METRICS)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")

    if len(metrics) == 0:
        return np.zeros((height, width, 4), dtype=np.uint8)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    axes = np.atleast_1d(axes)

    x_vals_cache: dict[int, np.ndarray] = {}
    for ax, metric in zip(axes, metrics, strict=False):
        values = np.asarray(series.get(metric, []), dtype=float)
        steps = values.shape[0]
        if steps not in x_vals_cache:
            x_vals_cache[steps] = np.arange(steps, dtype=float)
        ax.plot(x_vals_cache[steps], values, label=metric, linewidth=1.3)
        ax.set_title(metric)
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.tick_params(labelsize=8)
    fig.tight_layout()

    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buffer = buffer.reshape((h_px, w_px, 4))
    # Convert ARGB to RGBA
    rgba = buffer[:, :, [1, 2, 3, 0]]
    # Pad/crop to requested size if layout adjustments changed canvas size slightly
    rgba = _fit_to_size(rgba, target_height=height, target_width=width)
    plt.close(fig)
    return rgba


def make_surface_from_rgba(rgba: np.ndarray) -> pygame.Surface | None:
    """Convert an RGBA array into a Pygame surface if pygame is available.

    Args:
        rgba: Array shaped (H, W, 4) in uint8.

    Returns:
        pygame.Surface or None when pygame is unavailable.
    """
    if pygame is None:  # pragma: no cover - optional dependency
        return None
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8, copy=False)
    return pygame.image.frombuffer(rgba.copy(order="C"), (rgba.shape[1], rgba.shape[0]), "RGBA")


def _fit_to_size(rgba: np.ndarray, *, target_height: int, target_width: int) -> np.ndarray:
    """Pad or crop an RGBA image to an exact size.

    Returns:
        np.ndarray: RGBA array with shape (target_height, target_width, 4).
    """
    h, w, _ = rgba.shape
    if h == target_height and w == target_width:
        return rgba
    # Initialize transparent canvas
    canvas = np.zeros((target_height, target_width, 4), dtype=np.uint8)
    copy_h = min(h, target_height)
    copy_w = min(w, target_width)
    canvas[:copy_h, :copy_w] = rgba[:copy_h, :copy_w]
    return canvas


def export_combined_image(
    main_rgba: np.ndarray,
    pane_rgba: np.ndarray,
    out_path: str,
    *,
    layout: str = "vertical_split",
) -> str:
    """Export a combined image of the Pygame view and telemetry pane.

    Args:
        main_rgba: Main viewport image (H, W, 4) RGBA.
        pane_rgba: Telemetry pane image (h, w, 4) RGBA.
        out_path: Destination path for the PNG.
        layout: 'vertical_split' places pane on the right; 'horizontal_split' on bottom-left.

    Returns:
        Path to the written PNG file.
    """
    if main_rgba.ndim != 3 or pane_rgba.ndim != 3:
        raise ValueError("Images must be shaped (H, W, 4)")
    main = main_rgba.astype(np.uint8, copy=False)
    pane = pane_rgba.astype(np.uint8, copy=False)
    h, w, _ = main.shape
    ph, pw, _ = pane.shape
    if layout == "horizontal_split":
        total_h = h + ph
        total_w = max(w, pw)
        canvas = np.zeros((total_h, total_w, 4), dtype=np.uint8)
        canvas[:h, :w] = main
        canvas[h : h + ph, :pw] = pane
    else:
        total_h = max(h, ph)
        total_w = w + pw
        canvas = np.zeros((total_h, total_w, 4), dtype=np.uint8)
        canvas[:h, :w] = main
        canvas[:ph, w : w + pw] = pane
    plt.imsave(out_path, canvas)
    return out_path
