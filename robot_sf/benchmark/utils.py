"""Small utility helpers for benchmark reporting.

Phase A helpers are intentionally small and pure. The public function
``format_summary_table`` follows the contract in
``specs/139-extract-reusable-helpers/contracts/visualization_helper_contract.md``:
- accepts a mapping ``dict[str, float]`` of metrics
- returns a Markdown table string
- raises ValueError on empty input
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_optional_json(path: str | None) -> dict[str, Any] | None:
    """Load JSON from an optional file path.

    Args:
        path: File path to load JSON from, or None to return None.

    Returns:
        Parsed JSON dictionary, or None if path is None.

    Raises:
        FileNotFoundError: if path is provided but file doesn't exist.
        json.JSONDecodeError: if file contains invalid JSON.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a nested value from a dictionary using dot notation.

    Args:
        data: Dictionary to traverse.
        path: Dot-separated path (e.g., "metrics.collision_rate").
        default: Value to return if path is not found.

    Returns:
        The value at the specified path, or default if not found.

    Examples:
        >>> data = {"metrics": {"collision_rate": 0.1}}
        >>> get_nested_value(data, "metrics.collision_rate")
        0.1
        >>> get_nested_value(data, "missing.path", "fallback")
        'fallback'
    """
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def format_summary_table(metrics: dict[str, float]) -> str:
    """Format a metrics mapping as a simple Markdown table.

    Args:
        metrics: mapping of metric name to numeric value.

    Returns:
        A Markdown formatted table as a string.

    Raises:
        ValueError: if `metrics` is empty.
    """
    if not metrics:
        raise ValueError("metrics must not be empty")

    lines = ["| Metric | Value |", "|---|---:|"]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value} |")

    return "\n".join(lines)
