"""Shared metric helpers for imitation learning analysis and reporting.

Provides utilities for extracting metric samples from manifests/records while
handling absent keys and type normalization.
"""

from __future__ import annotations

from typing import Any


def metric_samples(payload: dict[str, Any], key: str) -> list[float]:
    """Return a list of float samples for a metric, if available.

    Looks for both `<key>_samples` and `<key>` entries and filters to numeric values.
    """

    metrics = payload.get("metrics") or {}
    samples = metrics.get(f"{key}_samples") or metrics.get(key)
    if isinstance(samples, list):
        return [float(v) for v in samples if isinstance(v, (int, float))]
    return []
