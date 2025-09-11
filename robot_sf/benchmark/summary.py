"""Quick diversity summary utilities.

Reads episode JSONL files and emits simple histograms for key metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

# Use headless backend for CI/non-GUI
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def _iter_records(paths: Sequence[str | Path] | str | Path) -> Iterable[Dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _safe_number(x: Any) -> float | None:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def collect_values(records: Iterable[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Collect min_distance and avg_speed from episode records.

    Falls back to computing avg speed from robot_vel if present under
    record["trajectory"]["robot_vel"] as a list of [vx, vy]. If not available,
    the avg_speed entry may be missing.
    """
    mins: List[float] = []
    speeds: List[float] = []
    for rec in records:
        md = _get_nested(rec, "metrics.min_distance")
        mdv = _safe_number(md)
        if mdv is not None:
            mins.append(mdv)
        av = _get_nested(rec, "metrics.avg_speed")
        avv = _safe_number(av)
        if avv is not None:
            speeds.append(avv)
        else:
            # Optional fallback: derive from trajectory if available
            traj = rec.get("trajectory") or {}
            rv = traj.get("robot_vel")
            if isinstance(rv, list) and rv and isinstance(rv[0], (list, tuple)):
                try:
                    import numpy as np

                    arr = np.asarray(rv, dtype=float)
                    s = np.linalg.norm(arr, axis=1).mean()
                    if np.isfinite(s):
                        speeds.append(float(s))
                except Exception:
                    pass
    return mins, speeds


def plot_histograms(
    mins: Sequence[float],
    speeds: Sequence[float],
    out_dir: str | Path,
    *,
    bins: int = 30,
) -> List[str]:
    out_paths: List[str] = []
    out_dir = str(out_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _save(fig, name: str) -> str:
        path = str(Path(out_dir) / name)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(path)
        return path

    if mins:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(mins, bins=bins, color="#4C78A8", edgecolor="white")
        ax.set_title("Min distance distribution")
        ax.set_xlabel("min_distance [m]")
        ax.set_ylabel("count")
        _save(fig, "hist_min_distance.png")

    if speeds:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(speeds, bins=bins, color="#F58518", edgecolor="white")
        ax.set_title("Average robot speed distribution")
        ax.set_xlabel("avg_speed [m/s]")
        ax.set_ylabel("count")
        _save(fig, "hist_avg_speed.png")

    return out_paths


def summarize_to_plots(paths: Sequence[str | Path] | str | Path, out_dir: str | Path) -> List[str]:
    mins, speeds = collect_values(_iter_records(paths))
    return plot_histograms(mins, speeds, out_dir)


__all__ = [
    "collect_values",
    "plot_histograms",
    "summarize_to_plots",
]
