"""Distribution plotting utilities for benchmark episode JSONL.

Features
- Collect per-metric values grouped by a dotted key
- Plot histogram and optional KDE per group
- Export PNG and optional LaTeX-friendly PDF (rcParams set as in dev_guide)

Programmatic entrypoint: save_distributions(...)
CLI wiring is done in robot_sf.benchmark.cli (plot-distributions).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from robot_sf.benchmark.plotting_style import apply_latex_style

Record = Mapping[str, object]


def _get_dotted(d: Mapping[str, object], path: str, default=None):
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
) -> Dict[str, Dict[str, List[float]]]:
    """Collect values per metric per group.

    Returns dict[group][metric] -> List[float]
    """

    def _to_float(x: object | None) -> float | None:
        try:
            v = float(x)  # type: ignore[arg-type]
        except Exception:
            return None
        return float(v) if np.isfinite(v) else None

    out: Dict[str, Dict[str, List[float]]] = {}
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
    pruned: Dict[str, Dict[str, List[float]]] = {}
    for g, mvals in out.items():
        kept = {m: vals for m, vals in mvals.items() if vals}
        if kept:
            pruned[g] = kept
    return pruned


@dataclass
class DistPlotMeta:
    wrote: List[str]
    pdfs: List[str]


def _apply_rcparams() -> None:
    apply_latex_style()


def _maybe_kde(ax, data: np.ndarray, color: str) -> None:
    try:
        from scipy.stats import gaussian_kde  # type: ignore

        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 200)
        ys = kde(xs)
        ax.plot(xs, ys * (len(data) * (xs[1] - xs[0])), color=color, alpha=0.8, linewidth=1.0)
    except Exception:
        # scipy optional; silently skip KDE if unavailable
        pass


def save_distributions(
    grouped: Dict[str, Dict[str, List[float]]],
    out_dir: str | Path,
    *,
    bins: int = 30,
    kde: bool = False,
    out_pdf: bool = False,
) -> DistPlotMeta:
    """Save per-group per-metric histograms (and optional KDE overlays).

    If out_pdf is True, also export a vector PDF with LaTeX-friendly rcParams.
    """
    _apply_rcparams()
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    wrote: List[str] = []
    pdfs: List[str] = []
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B"]

    for metric in sorted({m for gv in grouped.values() for m in gv.keys()}):
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, (group, mvals) in enumerate(sorted(grouped.items())):
            vals = np.asarray(mvals.get(metric, []), dtype=float)
            if vals.size == 0:
                continue
            color = palette[i % len(palette)]
            ax.hist(
                vals,
                bins=bins,
                alpha=0.35,
                label=f"{group} (n={vals.size})",
                color=color,
                edgecolor="white",
            )
            if kde and vals.size >= 5:
                _maybe_kde(ax, vals, color)

        ax.set_title(f"Distribution: {metric}")
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.legend(loc="best", fontsize=8)
        fname = f"dist_{metric}.png"
        fpath = str(Path(out_dir) / fname)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        wrote.append(fpath)
        if out_pdf:
            pdf_path = str(Path(out_dir) / f"dist_{metric}.pdf")
            # Re-render lightweightly for vector output
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            for i, (group, mvals) in enumerate(sorted(grouped.items())):
                vals = np.asarray(mvals.get(metric, []), dtype=float)
                if vals.size == 0:
                    continue
                color = palette[i % len(palette)]
                ax2.hist(vals, bins=bins, alpha=0.35, label=f"{group}", color=color)
                if kde and vals.size >= 5:
                    _maybe_kde(ax2, vals, color)
            ax2.set_xlabel(metric)
            ax2.set_ylabel("count")
            ax2.legend(loc="best", fontsize=8)
            fig2.savefig(pdf_path)
            plt.close(fig2)
            pdfs.append(pdf_path)

    return DistPlotMeta(wrote=wrote, pdfs=pdfs)


__all__ = [
    "collect_grouped_values",
    "save_distributions",
    "DistPlotMeta",
]
