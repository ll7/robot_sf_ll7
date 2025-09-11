"""Baseline statistics collector for SNQI normalization.

Runs batches of episodes and computes per-metric baseline statistics
(median and p95), saved to a JSON file. These stats can be consumed by
`metrics.snqi` to normalize metrics consistently across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from robot_sf.benchmark.aggregate import read_jsonl
from robot_sf.benchmark.runner import run_batch

DEFAULT_METRICS: List[str] = [
    "time_to_goal_norm",
    "collisions",
    "near_misses",
    "min_distance",
    "path_efficiency",
    "force_exceed_events",
    "comfort_exposure",
    "jerk_mean",
    "curvature_mean",
    "energy",
]


def _extract_metric_values(records: List[Dict[str, Any]], key: str) -> List[float]:
    vals: List[float] = []
    for rec in records:
        m = rec.get("metrics") or {}
        v = m.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def compute_baseline_stats_from_records(
    records: List[Dict[str, Any]], metrics: Iterable[str] | None = None
) -> Dict[str, Dict[str, float]]:
    if metrics is None:
        metrics = tuple(DEFAULT_METRICS)
    stats: Dict[str, Dict[str, float]] = {}
    for key in metrics:
        vals = _extract_metric_values(records, key)
        if len(vals) == 0:
            stats[key] = {"med": float("nan"), "p95": float("nan")}
        else:
            arr = np.asarray(vals, dtype=float)
            stats[key] = {
                "med": float(np.nanmedian(arr)),
                "p95": float(np.nanpercentile(arr, 95)),
            }
    return stats


def run_and_compute_baseline(
    scenarios_or_path: List[Dict[str, Any]] | str | Path,
    *,
    out_json: str | Path,
    out_jsonl: str | Path | None = None,
    schema_path: str | Path,
    base_seed: int = 0,
    repeats_override: int | None = None,
    horizon: int = 100,
    dt: float = 0.1,
    record_forces: bool = False,
    metrics: Iterable[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    # Optionally run batch to collect JSONL
    tmp_jsonl: str | None = None
    if out_jsonl is not None:
        tmp_jsonl = str(out_jsonl)
    else:
        # default temp path under results/
        tmp_jsonl = str(Path("results") / "baseline_episodes.jsonl")

    run_batch(
        scenarios_or_path,
        out_path=tmp_jsonl,
        schema_path=schema_path,
        base_seed=base_seed,
        repeats_override=repeats_override,
        horizon=horizon,
        dt=dt,
        record_forces=record_forces,
        append=False,
        fail_fast=False,
    )

    records = read_jsonl(tmp_jsonl)
    stats = compute_baseline_stats_from_records(records, metrics=metrics)

    out_json = str(out_json)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    return stats


__all__ = [
    "compute_baseline_stats_from_records",
    "run_and_compute_baseline",
    "DEFAULT_METRICS",
]
