"""Failure case extractor for Social Navigation Benchmark episodes.

Select episodes that meet failure-like criteria, e.g. collisions or low comfort.

Criteria
--------
- collisions >= collision_threshold (default 1)
- comfort_exposure >= comfort_threshold (default 0.2)
- near_misses > near_miss_threshold (default 0)
- optional: snqi < snqi_below (applied only when present in record metrics)

Return the matching records, optionally capped by max_count.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable


def _metric(rec: dict[str, Any], name: str, default: float = 0.0) -> float:
    """Metric.

    Args:
        rec: Single record dictionary.
        name: Human-friendly name.
        default: Default fallback value.

    Returns:
        float: Floating-point value.
    """
    m = rec.get("metrics") or {}
    v = m.get(name, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def is_failure(
    rec: dict[str, Any],
    *,
    collision_threshold: float = 1.0,
    comfort_threshold: float = 0.2,
    near_miss_threshold: float = 0.0,
    snqi_below: float | None = None,
) -> bool:
    """Return True when record matches any failure criteria.

    The criteria are OR'ed: any individual predicate being true marks failure.
    SNQI is applied only if `snqi_below` is not None and the metric is present.
    Missing metrics default to 0.0 and thus do not trigger failures.
    """
    if _metric(rec, "collisions", 0.0) >= float(collision_threshold):
        return True
    if _metric(rec, "comfort_exposure", 0.0) >= float(comfort_threshold):
        return True
    # Use strict greater-than for near misses so that a threshold of 0 does
    # not flag episodes with zero near misses as failures.
    if _metric(rec, "near_misses", 0.0) > float(near_miss_threshold):
        return True
    if snqi_below is not None:
        m = rec.get("metrics") or {}
        if "snqi" in m:
            try:
                if float(m["snqi"]) < float(snqi_below):
                    return True
            except Exception:
                pass
    return False


def extract_failures(
    records: Iterable[dict[str, Any]],
    *,
    collision_threshold: float = 1.0,
    comfort_threshold: float = 0.2,
    near_miss_threshold: float = 0.0,
    snqi_below: float | None = None,
    max_count: int | None = None,
) -> list[dict[str, Any]]:
    """Filter records and return those that match failure criteria.

    Parameters
    ----------
    records : iterable of dict
        Episode records loaded from JSONL.
    collision_threshold : float
        Minimum number of collisions to consider failure.
    comfort_threshold : float
        Minimum comfort exposure to consider failure.
    near_miss_threshold : float
        Minimum near-miss count to consider failure.
    snqi_below : float | None
        If provided and metrics.snqi is present, episodes with snqi < value
        are considered failures.
    max_count : int | None
        Limit the number of returned failures (None = no limit).
    """
    out: list[dict[str, Any]] = []
    for rec in records:
        if is_failure(
            rec,
            collision_threshold=collision_threshold,
            comfort_threshold=comfort_threshold,
            near_miss_threshold=near_miss_threshold,
            snqi_below=snqi_below,
        ):
            out.append(rec)
            if max_count is not None and len(out) >= int(max_count):
                break
    return out


__all__ = ["extract_failures", "is_failure"]
