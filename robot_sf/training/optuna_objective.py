"""Utilities for robust Optuna objective extraction from eval histories."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import numpy as np

OBJECTIVE_MODES = ("best_checkpoint", "final_eval", "last_n_mean", "auc")

if TYPE_CHECKING:
    from pathlib import Path


def load_episode_records(path: Path) -> list[dict[str, object]]:
    """Load JSONL episode records produced by training scripts.

    Returns:
        List of parsed JSON-object records.
    """
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def eval_metric_series(
    records: Iterable[Mapping[str, object]],
    *,
    metric_name: str,
) -> list[tuple[int, float]]:
    """Aggregate per-episode eval records into a per-step metric series.

    Returns:
        Sorted ``(eval_step, mean_metric)`` tuples.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for record in records:
        eval_step_raw = record.get("eval_step")
        if eval_step_raw is None:
            continue
        try:
            eval_step = int(eval_step_raw)
        except (TypeError, ValueError):
            continue
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        metric_value = metrics.get(metric_name)
        if metric_value is None:
            continue
        try:
            value = float(metric_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        buckets[eval_step].append(value)

    series: list[tuple[int, float]] = []
    for eval_step in sorted(buckets):
        values = buckets[eval_step]
        if values:
            series.append((eval_step, float(np.mean(values))))
    return series


def objective_from_series(
    series: list[tuple[int, float]],
    *,
    mode: str,
    window: int,
) -> float | None:
    """Compute an objective scalar from a per-eval-step metric series.

    Returns:
        Reduced objective value, or ``None`` when mode delegates elsewhere.
    """
    if not series:
        return None
    if mode == "final_eval":
        return float(series[-1][1])
    if mode == "last_n_mean":
        tail = max(1, int(window))
        return float(np.mean([value for _, value in series[-tail:]]))
    if mode == "auc":
        if len(series) == 1:
            return float(series[0][1])
        x = np.asarray([step for step, _ in series], dtype=float)
        y = np.asarray([value for _, value in series], dtype=float)
        span = float(x[-1] - x[0])
        if span <= 0:
            return float(np.mean(y))
        trapz_compat = getattr(np, "trapezoid", np.trapz)
        area = float(trapz_compat(y, x))
        return area / span
    if mode == "best_checkpoint":
        return None
    raise ValueError(
        f"Unknown objective mode '{mode}'. Expected one of: {', '.join(OBJECTIVE_MODES)}"
    )
