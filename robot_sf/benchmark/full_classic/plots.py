"""Plot generation utilities for distributions, trajectories, KDEs, Pareto, force heatmaps.

Implemented across tasks T035 (basic), T036 (extended plots).
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterable
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
    """TODO docstring. Document this class."""

    kind: str
    path_pdf: str
    status: str
    note: str | None = None


def _safe_fig_close(fig):  # pragma: no cover - trivial
    """TODO docstring. Document this function.

    Args:
        fig: TODO docstring.
    """
    try:
        # Clear and fully close to avoid accumulating many open figures triggering warnings.
        fig.clf()
        if plt is not None:
            # matplotlib close may raise on some backends; suppress non-fatal errors
            with contextlib.suppress(Exception):
                plt.close(fig)
    except (
        RuntimeError,
        AttributeError,
        ValueError,
    ) as exc:  # pragma: no cover - defensive
        # Log at debug for visibility without changing behavior
        try:
            logger = importlib.import_module("loguru").logger
            logger.debug("_safe_fig_close failed: %s", exc)
        except ImportError:
            # If logger import fails we still want to silently ignore close failures
            pass


def _write_placeholder_text(path: Path, title: str, lines: list[str]):
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
        title: TODO docstring.
        lines: TODO docstring.
    """
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
        except (
            RuntimeError,
            AttributeError,
            OSError,
        ) as exc:  # pragma: no cover - defensive
            try:
                logger = importlib.import_module("loguru").logger
                logger.debug("_write_placeholder_text close failed: %s", exc)
            except ImportError:
                pass
    return True


def _distribution_plot(groups, out_dir: Path) -> _PlotArtifact:
    """TODO docstring. Document this function.

    Args:
        groups: TODO docstring.
        out_dir: TODO docstring.

    Returns:
        TODO docstring.
    """
    pdf_path = out_dir / "distributions_basic.pdf"
    if plt is None:
        return _PlotArtifact("distribution", str(pdf_path), "skipped", note="matplotlib missing")
    status = "failed"
    note = None
    labels: list[str] = []
    success_vals: list[float] = []
    collision_vals: list[float] = []
    for g in groups:
        suc = g.metrics.get("success_rate")
        col = g.metrics.get("collision_rate")
        if suc is None and col is None:
            continue
        labels.append(f"{g.archetype}-{g.density}")
        success_vals.append(suc.mean if suc else 0.0)
        collision_vals.append(col.mean if col else 0.0)
    if not labels:
        return _PlotArtifact("distribution", str(pdf_path), "skipped", note="no-metrics")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = range(len(labels))
    ax.bar(x, success_vals, width=0.4, label="success_rate", color="tab:green")
    ax.bar(
        [v + 0.4 for v in x],
        collision_vals,
        width=0.4,
        label="collision_rate",
        color="tab:red",
    )
    ax.set_xticks([v + 0.2 for v in x])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("rate")
    ax.set_title("Success vs Collision Rates")
    ax.legend()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        status = "generated"
    except Exception as exc:  # pragma: no cover - defensive
        note = f"savefig-error:{exc}"
    finally:
        _safe_fig_close(fig)
    return _PlotArtifact("distribution", str(pdf_path), status, note=note)


def _trajectory_plot(records: Iterable[dict], out_dir: Path) -> _PlotArtifact:
    """TODO docstring. Document this function.

    Args:
        records: TODO docstring.
        out_dir: TODO docstring.

    Returns:
        TODO docstring.
    """
    pdf_path = out_dir / "trajectories_basic.pdf"
    if plt is None:
        return _PlotArtifact("trajectory", str(pdf_path), "skipped", note="matplotlib missing")
    status = "failed"
    note = None
    fig, ax = plt.subplots(figsize=(5, 5))
    plotted = False
    for rec in records:
        steps = rec.get("replay_steps")
        if not isinstance(steps, list) or len(steps) < 2:
            continue
        xs = [float(t[1]) for t in steps]
        ys = [float(t[2]) for t in steps]
        ax.plot(
            xs,
            ys,
            linewidth=1.2,
            marker=".",
            markersize=3,
            label=rec.get("scenario_id", "episode"),
        )
        plotted = True
    if not plotted:
        _safe_fig_close(fig)
        return _PlotArtifact("trajectory", str(pdf_path), "skipped", note="no-replay")
    ax.set_title("Robot Trajectories")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        status = "generated"
    except Exception as exc:  # pragma: no cover - defensive
        note = f"savefig-error:{exc}"
    finally:
        _safe_fig_close(fig)
    return _PlotArtifact("trajectory", str(pdf_path), status, note=note)


def _kde_plot_placeholder(groups, out_dir: Path) -> _PlotArtifact:
    """TODO docstring. Document this function.

    Args:
        groups: TODO docstring.
        out_dir: TODO docstring.

    Returns:
        TODO docstring.
    """
    pdf_path = out_dir / "path_efficiency.pdf"
    if plt is None:
        return _PlotArtifact("kde", str(pdf_path), "skipped", note="matplotlib missing")
    status = "failed"
    note = None
    vals: list[float] = []
    for g in groups:
        pe = g.metrics.get("path_efficiency")
        if pe:
            vals.append(float(pe.mean))
    if not vals:
        return _PlotArtifact("kde", str(pdf_path), "skipped", note="no-path-efficiency")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(vals, bins=min(10, len(vals)), color="tab:blue", alpha=0.8)
    ax.set_title("Path Efficiency Distribution")
    ax.set_xlabel("path_efficiency (mean)")
    ax.set_ylabel("count")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        status = "generated"
    except Exception as exc:  # pragma: no cover - defensive
        note = f"savefig-error:{exc}"
    finally:
        _safe_fig_close(fig)
    return _PlotArtifact("kde", str(pdf_path), status, note=note)


def _pareto_plot_placeholder(groups, out_dir: Path) -> _PlotArtifact:
    """TODO docstring. Document this function.

    Args:
        groups: TODO docstring.
        out_dir: TODO docstring.

    Returns:
        TODO docstring.
    """
    pdf_path = out_dir / "pareto_placeholder.pdf"
    if plt is None:
        return _PlotArtifact("pareto", str(pdf_path), "skipped", note="matplotlib missing")
    status = "failed"
    note = None
    fig, ax = plt.subplots(figsize=(5, 4))
    plotted = False
    for g in groups:
        suc = g.metrics.get("success_rate")
        col = g.metrics.get("collision_rate")
        if suc is None or col is None:
            continue
        ax.scatter(col.mean, suc.mean, label=f"{g.archetype}-{g.density}", s=40)
        plotted = True
    if not plotted:
        _safe_fig_close(fig)
        return _PlotArtifact("pareto", str(pdf_path), "skipped", note="no-metrics")
    ax.set_xlabel("collision_rate")
    ax.set_ylabel("success_rate")
    ax.set_title("Success vs Collision")
    ax.legend(fontsize=7)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        status = "generated"
    except Exception as exc:  # pragma: no cover - defensive
        note = f"savefig-error:{exc}"
    finally:
        _safe_fig_close(fig)
    return _PlotArtifact("pareto", str(pdf_path), status, note=note)


def _force_heatmap_placeholder(out_dir: Path, records: Iterable[dict]) -> _PlotArtifact:
    """TODO docstring. Document this function.

    Args:
        out_dir: TODO docstring.
        records: TODO docstring.

    Returns:
        TODO docstring.
    """
    pdf_path = out_dir / "episode_lengths.pdf"
    if plt is None:
        return _PlotArtifact("force_heatmap", str(pdf_path), "skipped", note="matplotlib missing")
    status = "failed"
    note = None
    step_counts = [int(r.get("steps", 0)) for r in records if r.get("steps") is not None]
    if not step_counts:
        return _PlotArtifact("force_heatmap", str(pdf_path), "skipped", note="no-steps")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(step_counts, bins=min(10, len(step_counts)), color="tab:purple", alpha=0.8)
    ax.set_xlabel("steps")
    ax.set_ylabel("episodes")
    ax.set_title("Episode Lengths")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(pdf_path, bbox_inches="tight")
        status = "generated"
    except Exception as exc:  # pragma: no cover - defensive
        note = f"savefig-error:{exc}"
    finally:
        _safe_fig_close(fig)
    return _PlotArtifact("force_heatmap", str(pdf_path), status, note=note)


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
    artifacts.append(_force_heatmap_placeholder(out_path, records))

    return artifacts
