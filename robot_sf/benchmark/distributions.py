"""Distribution plotting utilities for benchmark episode JSONL.

Features
- Collect per-metric values grouped by a dotted key
- Plot histogram and optional KDE per group
- Export PNG and optional LaTeX-friendly PDF (rcParams set as in dev_guide)

Programmatic entrypoint: save_distributions(...)
CLI wiring is done in robot_sf.benchmark.cli (plot-distributions).
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robot_sf.benchmark.plotting_style import apply_latex_style

Record = Mapping[str, object]


def _get_dotted(d: Mapping[str, object], path: str, default=None):
    """Return value at dotted path from a mapping."""
    cur: object = d
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:  # type: ignore[operator]
            return default
        cur = cur[part]  # type: ignore[index]
    return cur


def collect_grouped_values(
    records: Iterable[Record],
    *,
    metrics: Sequence[str],
    group_by: str = "scenario_params.algo",
    fallback_group_by: str = "scenario_id",
) -> dict[str, dict[str, list[float]]]:
    """Collect values per metric per group.

    Returns dict[group][metric] -> List[float]

    Returns:
        Nested dictionary mapping group names to metric names to value lists.
    """

    def _to_float(x: object | None) -> float | None:
        """Coerce to finite float if possible.

        Returns:
            Finite float value or None when invalid.
        """
        try:
            v = float(x)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return float(v) if np.isfinite(v) else None

    out: dict[str, dict[str, list[float]]] = {}
    for r in records:
        g = _get_dotted(r, group_by) or _get_dotted(r, fallback_group_by)
        if g is None:
            continue
        gm = out.setdefault(str(g), {m: [] for m in metrics})
        for m in metrics:
            fv = _to_float(_get_dotted(r, f"metrics.{m}"))
            if fv is None:
                continue
            gm.setdefault(m, []).append(fv)
    # Prune empties with a dict comprehension per group
    pruned: dict[str, dict[str, list[float]]] = {}
    for g, mvals in out.items():
        kept = {m: vals for m, vals in mvals.items() if vals}
        if kept:
            pruned[g] = kept
    return pruned


@dataclass
class DistPlotMeta:
    """Metadata for generated distribution plots."""

    wrote: list[str]
    pdfs: list[str]


def _apply_rcparams() -> None:
    """Apply LaTeX-friendly plotting defaults."""
    apply_latex_style()


def _maybe_kde(ax, data: np.ndarray, color: str) -> None:
    """Optionally overlay a KDE curve when scipy is available."""
    try:
        stats_module = importlib.import_module("scipy.stats")
        gaussian_kde = stats_module.gaussian_kde
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 200)
        ys = kde(xs)
        ax.plot(
            xs,
            ys * (len(data) * (xs[1] - xs[0])),
            color=color,
            alpha=0.8,
            linewidth=1.0,
        )
    except (ImportError, ModuleNotFoundError):
        # scipy optional; silently skip KDE if unavailable
        pass
    except (TypeError, ValueError):
        # Invalid data shape or values -> skip KDE
        pass


def _compute_hist_ci(
    vals: np.ndarray,
    *,
    bins: int,
    samples: int,
    confidence: float,
    rng: np.random.Generator | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Bootstrap histogram-count confidence band for given values.

    Returns (centers, low, high) or None if not enough/invalid data.

    Returns:
        Tuple of (bin_centers, lower_bound, upper_bound) or None if insufficient data.
    """
    if vals.size < 5:
        return None
    vmin = float(vals.min())
    vmax = float(vals.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    bin_edges = np.linspace(vmin, vmax, bins + 1)
    counts_samples: list[np.ndarray] = []
    local_rng = rng if rng is not None else np.random.default_rng()
    for _ in range(int(samples)):
        resample = local_rng.choice(vals, size=vals.size, replace=True)
        cts, _ = np.histogram(resample, bins=bin_edges)
        counts_samples.append(cts)
    cs = np.stack(counts_samples, axis=0)  # (S, B)
    alpha = (1.0 - confidence) / 2.0
    low = np.quantile(cs, alpha, axis=0)
    high = np.quantile(cs, 1.0 - alpha, axis=0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, low, high


def _render_metric(  # noqa: PLR0913
    ax,
    grouped: dict[str, dict[str, list[float]]],
    *,
    metric: str,
    bins: int,
    kde: bool,
    ci: bool,
    ci_samples: int,
    ci_confidence: float,
    ci_seed: int | None,
    palette: Sequence[str],
    legend_with_n: bool,
) -> None:
    """Render one metric across all groups onto the given axes."""
    rng = np.random.default_rng(ci_seed) if ci and ci_seed is not None else None
    for i, (group, mvals) in enumerate(sorted(grouped.items())):
        vals = np.asarray(mvals.get(metric, []), dtype=float)
        if vals.size == 0:
            continue
        color = palette[i % len(palette)]
        label = f"{group} (n={vals.size})" if legend_with_n else f"{group}"
        ax.hist(
            vals,
            bins=bins,
            alpha=0.35,
            label=label,
            color=color,
            edgecolor="white",
        )
        if kde and vals.size >= 5:
            _maybe_kde(ax, vals, color)
        if ci:
            ci_data = _compute_hist_ci(
                vals,
                bins=bins,
                samples=ci_samples,
                confidence=ci_confidence,
                rng=rng,
            )
            if ci_data is not None:
                centers, low, high = ci_data
                ax.fill_between(
                    centers,
                    low,
                    high,
                    color=color,
                    alpha=0.15,
                    step="mid",
                    linewidth=0.0,
                    label=None,
                )


def _metrics_in_grouped(grouped: dict[str, dict[str, list[float]]]) -> list[str]:
    """Return sorted unique metric names present in grouped dict.

    Returns:
        Sorted list of unique metric names found across all groups.
    """
    return sorted({m for gv in grouped.values() for m in gv})


def _save_one_metric(  # noqa: PLR0913
    out_dir: str,
    metric: str,
    grouped: dict[str, dict[str, list[float]]],
    *,
    bins: int,
    kde: bool,
    ci: bool,
    ci_samples: int,
    ci_confidence: float,
    ci_seed: int | None,
    palette: Sequence[str],
    out_pdf: bool,
) -> tuple[str, str | None]:
    """Render and save a single metric to PNG and optionally PDF; returns paths.

    Returns:
        Tuple of (png_path, pdf_path) where pdf_path is None if not generated.
    """
    # PNG
    fig, ax = plt.subplots(figsize=(6, 4))
    _render_metric(
        ax,
        grouped,
        metric=metric,
        bins=bins,
        kde=kde,
        ci=ci,
        ci_samples=ci_samples,
        ci_confidence=ci_confidence,
        ci_seed=ci_seed,
        palette=palette,
        legend_with_n=True,
    )
    ax.set_title(f"Distribution: {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize=8)
    png_path = str(Path(out_dir) / f"dist_{metric}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    pdf_path: str | None = None
    if out_pdf:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        _render_metric(
            ax2,
            grouped,
            metric=metric,
            bins=bins,
            kde=kde,
            ci=ci,
            ci_samples=ci_samples,
            ci_confidence=ci_confidence,
            ci_seed=ci_seed,
            palette=palette,
            legend_with_n=False,
        )
        ax2.set_xlabel(metric)
        ax2.set_ylabel("count")
        ax2.legend(loc="best", fontsize=8)
        pdf_path = str(Path(out_dir) / f"dist_{metric}.pdf")
        fig2.savefig(pdf_path)
        plt.close(fig2)

    return png_path, pdf_path


def save_distributions(  # noqa: PLR0913
    grouped: dict[str, dict[str, list[float]]],
    out_dir: str | Path,
    *,
    bins: int = 30,
    kde: bool = False,
    out_pdf: bool = False,
    ci: bool = False,
    ci_samples: int = 1000,
    ci_confidence: float = 0.95,
    ci_seed: int | None = 123,
) -> DistPlotMeta:
    """Save per-group per-metric histograms (and optional KDE overlays).

    If out_pdf is True, also export a vector PDF with LaTeX-friendly rcParams.

    Returns:
        DistPlotMeta containing lists of written PNG and PDF paths.
    """
    _apply_rcparams()
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    wrote: list[str] = []
    pdfs: list[str] = []
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B"]

    for metric in _metrics_in_grouped(grouped):
        png_path, pdf_path = _save_one_metric(
            out_dir,
            metric,
            grouped,
            bins=bins,
            kde=kde,
            ci=ci,
            ci_samples=ci_samples,
            ci_confidence=ci_confidence,
            ci_seed=ci_seed,
            palette=palette,
            out_pdf=out_pdf,
        )
        wrote.append(png_path)
        if pdf_path is not None:
            pdfs.append(pdf_path)

    return DistPlotMeta(wrote=wrote, pdfs=pdfs)


__all__ = [
    "DistPlotMeta",
    "collect_grouped_values",
    "save_distributions",
]
