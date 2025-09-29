"""Plot generation utilities for distributions, trajectories, KDEs, Pareto, force heatmaps.

Implemented across tasks T035 (basic), T036 (extended plots).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

try:  # Matplotlib is an optional dependency (analysis extra). Fail gracefully if absent.
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:
    plt = None  # type: ignore


@dataclass
class _PlotArtifact:  # lightweight internal representation matching data model subset
    kind: str
    path_pdf: str
    status: str
    note: str | None = None


def _safe_fig_close(fig):  # pragma: no cover - trivial
    try:
        # Clear and fully close to avoid accumulating many open figures triggering warnings.
        fig.clf()
        import matplotlib.pyplot as _plt  # type: ignore

        # matplotlib close may raise on some backends; suppress non-fatal errors
        with contextlib.suppress(Exception):
            _plt.close(fig)
    except (RuntimeError, AttributeError, ValueError) as exc:  # pragma: no cover - defensive
        # Log at debug for visibility without changing behavior
        try:
            from loguru import logger

            logger.debug("_safe_fig_close failed: %s", exc)
        except ImportError:
            # If logger import fails we still want to silently ignore close failures
            pass


def _write_placeholder_text(path: Path, title: str, lines: list[str]):
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
    except OSError:
        # Filesystem/write errors -> report failure by closing and returning False
        _safe_fig_close(fig)
        return False
    finally:
        # Always attempt to close the figure; log on unexpected failures.
        try:
            _safe_fig_close(fig)
        except (RuntimeError, AttributeError, OSError) as exc:  # pragma: no cover - defensive
            try:
                from loguru import logger

                logger.debug("_write_placeholder_text close failed: %s", exc)
            except ImportError:
                pass
    return True


def _distribution_plot(groups, out_dir: Path) -> _PlotArtifact:
    pdf_path = out_dir / "distributions_basic.pdf"
    if plt is None:
        return _PlotArtifact("distribution", str(pdf_path), "skipped", note="matplotlib missing")
    lines: list[str] = ["Distribution Summary (placeholder)"]
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


def _kde_plot_placeholder(groups, out_dir: Path) -> _PlotArtifact:
    pdf_path = out_dir / "kde_placeholder.pdf"
    if plt is None:
        return _PlotArtifact("kde", str(pdf_path), "skipped", note="matplotlib missing")
    # Placeholder text figure; real implementation would compute spatial density KDE
    generated = _write_placeholder_text(
        pdf_path,
        "KDE Spatial Density (placeholder)",
        ["Not yet implemented: spatial sampling"],
    )
    return _PlotArtifact("kde", str(pdf_path), "generated" if generated else "skipped")


def _pareto_plot_placeholder(groups, out_dir: Path) -> _PlotArtifact:
    pdf_path = out_dir / "pareto_placeholder.pdf"
    if plt is None:
        return _PlotArtifact("pareto", str(pdf_path), "skipped", note="matplotlib missing")
    lines = ["Pareto Frontier (placeholder)"]
    for g in groups:
        suc = g.metrics.get("success_rate")
        col = g.metrics.get("collision_rate")
        if suc and col:
            lines.append(f"{g.archetype}/{g.density}: S={suc.mean:.2f} C={col.mean:.2f}")
    generated = _write_placeholder_text(pdf_path, "Pareto", lines)
    return _PlotArtifact("pareto", str(pdf_path), "generated" if generated else "skipped")


def _force_heatmap_placeholder(out_dir: Path) -> _PlotArtifact:
    pdf_path = out_dir / "force_heatmap_placeholder.pdf"
    if plt is None:
        return _PlotArtifact("force_heatmap", str(pdf_path), "skipped", note="matplotlib missing")
    generated = _write_placeholder_text(
        pdf_path,
        "Force Interaction Heatmap (placeholder)",
        ["No force data provided; skipped"],
    )
    return _PlotArtifact(
        "force_heatmap",
        str(pdf_path),
        "generated" if generated else "skipped",
        note="placeholder",
    )


def generate_plots(groups, records, out_dir, cfg):  # T035 basic + T036 extended placeholders
    """Generate plots returning artifact metadata list.

    Includes:
      - Distribution & trajectory (implemented minimal versions)
      - KDE, Pareto, force heatmap placeholders (T036) always produced as placeholder PDFs
        unless matplotlib missing (then skipped).
    In smoke mode all artifacts are still generated as lightweight placeholders.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    artifacts: list[_PlotArtifact] = []
    artifacts.append(_distribution_plot(groups, out_path))
    artifacts.append(_trajectory_plot(records, out_path))
    artifacts.append(_kde_plot_placeholder(groups, out_path))
    artifacts.append(_pareto_plot_placeholder(groups, out_path))
    artifacts.append(_force_heatmap_placeholder(out_path))

    return artifacts
