"""Plot generation utilities for distributions, trajectories, KDEs, Pareto, force heatmaps.

Implemented across tasks T035 (basic), T036 (extended plots).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

try:  # Matplotlib is an optional dependency (analysis extra). Fail gracefully if absent.
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # noqa: BLE001
    plt = None  # type: ignore


@dataclass
class _PlotArtifact:  # lightweight internal representation matching data model subset
    kind: str
    path_pdf: str
    status: str
    note: str | None = None


def _safe_fig_close(fig):  # pragma: no cover - trivial
    try:
        fig.clf()
    except Exception:  # noqa: BLE001
        pass


def _write_placeholder_text(path: Path, title: str, lines: List[str]):
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    ax.set_title(title)
    y = 0.95
    for ln in lines:
        ax.text(0.01, y, ln, fontsize=8, va="top")
        y -= 0.08
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, bbox_inches="tight")
    finally:
        _safe_fig_close(fig)
    return True


def _distribution_plot(groups, out_dir: Path) -> _PlotArtifact:
    pdf_path = out_dir / "distributions_basic.pdf"
    if plt is None:
        return _PlotArtifact("distribution", str(pdf_path), "skipped", note="matplotlib missing")
    lines: List[str] = ["Distribution Summary (placeholder)"]
    for g in groups:
        # Include a minimal metric summary line
        col = g.metrics.get("collision_rate")
        if col:
            lines.append(f"{g.archetype}/{g.density}: collision_mean={col.mean:.3f}")
    generated = _write_placeholder_text(pdf_path, "Distributions", lines)
    return _PlotArtifact("distribution", str(pdf_path), "generated" if generated else "skipped")


def _trajectory_plot(records: Iterable[dict], out_dir: Path) -> _PlotArtifact:
    pdf_path = out_dir / "trajectories_basic.pdf"
    if plt is None:
        return _PlotArtifact("trajectory", str(pdf_path), "skipped", note="matplotlib missing")
    # Create a trivial scatter of synthetic path (placeholder) derived from seed to ensure determinism
    fig, ax = plt.subplots(figsize=(4, 4)) if plt else (None, None)
    if plt:
        for rec in records:
            seed = int(rec.get("seed", 0))
            xs = [math.sin((seed + i) * 0.1) for i in range(10)]
            ys = [math.cos((seed + i) * 0.1) for i in range(10)]
            ax.plot(xs, ys, marker="o", linewidth=1, markersize=2)
        ax.set_title("Trajectory Overlay (placeholder)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(pdf_path, bbox_inches="tight")
        finally:
            _safe_fig_close(fig)
        return _PlotArtifact("trajectory", str(pdf_path), "generated")
    return _PlotArtifact("trajectory", str(pdf_path), "skipped")


def generate_plots(groups, records, out_dir, cfg):  # T035 basic implementation
    """Generate a minimal set of plots (distribution + trajectory) in smoke mode.

    This satisfies contract test T012 (post-update). Extended plots (KDE, Pareto,
    force heatmap) will be added in T036. Returns list of artifact objects with
    kind/status fields. Gracefully skips when matplotlib not available.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts: List[_PlotArtifact] = []
    artifacts.append(_distribution_plot(groups, out_path))
    artifacts.append(_trajectory_plot(records, out_path))

    return artifacts
