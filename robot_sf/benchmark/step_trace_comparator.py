"""Canonical step-trace comparison for per-context benchmark determinism (issue #6126).

This module provides a fail-closed, actionable first-difference comparator and
canonical digest calculation for simulation step traces produced by benchmark
episodes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

SCHEMA_VERSION = "step_trace_comparator.v1"


def canonicalize_step_trace(
    trace: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract and normalize the step list from a raw simulation step trace payload.

    Args:
        trace: A dictionary containing a 'steps' list or a direct list of step dictionaries.

    Returns:
        List of step dictionaries.

    Raises:
        ValueError: If the input does not represent a valid step trace list.
    """
    if isinstance(trace, dict):
        steps = trace.get("steps")
        if not isinstance(steps, list):
            raise ValueError("Input trace dictionary must contain a 'steps' list.")
        return steps
    if isinstance(trace, list):
        return trace
    raise ValueError("Input trace must be a dictionary or a list.")


def canonical_step_trace_digest(trace: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Return a stable SHA-256 digest of the canonical step trace representation."""
    steps = canonicalize_step_trace(trace)
    encoded = json.dumps(steps, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _compare_dicts(val1: dict[str, Any], val2: dict[str, Any], path: str) -> str | None:
    """Compare two dictionaries recursively.

    Returns:
        First difference string if dictionaries differ, else None.
    """
    keys1 = set(val1.keys())
    keys2 = set(val2.keys())
    if keys1 != keys2:
        missing_in_2 = sorted(keys1 - keys2)
        missing_in_1 = sorted(keys2 - keys1)
        diffs = []
        if missing_in_2:
            diffs.append(f"missing in second: {missing_in_2}")
        if missing_in_1:
            diffs.append(f"missing in first: {missing_in_1}")
        return f"Key mismatch at '{path}': {', '.join(diffs)}"

    for key in sorted(keys1):
        sub_path = f"{path}.{key}" if path else key
        diff = find_first_trace_difference(val1[key], val2[key], path=sub_path)
        if diff is not None:
            return diff
    return None


def _compare_lists(val1: list[Any], val2: list[Any], path: str) -> str | None:
    """Compare two lists recursively.

    Returns:
        First difference string if lists differ, else None.
    """
    if len(val1) != len(val2):
        return f"Length mismatch at '{path}': {len(val1)} items vs {len(val2)} items"

    for idx, (item1, item2) in enumerate(zip(val1, val2, strict=True)):
        sub_path = f"{path}[{idx}]"
        diff = find_first_trace_difference(item1, item2, path=sub_path)
        if diff is not None:
            return diff
    return None


def _compare_floats(val1: float, val2: float, path: str) -> str | None:
    """Compare two floating-point numbers.

    Returns:
        First difference string if numbers differ, else None.
    """
    if math.isnan(val1) and math.isnan(val2):
        return None
    if math.isnan(val1) or math.isnan(val2):
        return f"NaN mismatch at '{path}': {val1!r} vs {val2!r}"
    if val1 != val2:
        delta = abs(val1 - val2)
        return f"Value mismatch at '{path}': {val1!r} != {val2!r} (abs delta = {delta:.6e})"
    return None


def find_first_trace_difference(
    val1: Any,
    val2: Any,
    path: str = "steps",
) -> str | None:
    """Recursively compare two trace data structures and return the first difference description.

    Args:
        val1: First value.
        val2: Second value.
        path: Current string key path for error formatting.

    Returns:
        A human-readable first-difference string if val1 != val2, else None.
    """
    if type(val1) is not type(val2):
        return (
            f"Type mismatch at '{path}': {type(val1).__name__} vs {type(val2).__name__} "
            f"({val1!r} vs {val2!r})"
        )

    if isinstance(val1, dict):
        return _compare_dicts(val1, val2, path)

    if isinstance(val1, list):
        return _compare_lists(val1, val2, path)

    if isinstance(val1, float):
        return _compare_floats(val1, val2, path)

    if val1 != val2:
        return f"Value mismatch at '{path}': {val1!r} != {val2!r}"

    return None


def compare_step_traces(
    trace1: dict[str, Any] | list[dict[str, Any]],
    trace2: dict[str, Any] | list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Compare two simulation step traces and report an actionable first difference.

    Args:
        trace1: First simulation step trace (dict or list of steps).
        trace2: Second simulation step trace (dict or list of steps).

    Returns:
        (True, None) if traces are identical.
        (False, difference_description) if traces differ.
    """
    steps1 = canonicalize_step_trace(trace1)
    steps2 = canonicalize_step_trace(trace2)

    diff = find_first_trace_difference(steps1, steps2, path="steps")
    if diff is None:
        return True, None
    return False, diff
