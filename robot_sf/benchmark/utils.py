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
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from robot_sf.common.artifact_paths import resolve_artifact_path

if TYPE_CHECKING:
    from collections.abc import Iterable


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


def ensure_directory(path: str | Path) -> Path:
    """Ensure a directory exists, creating parent directories as needed.

    Args:
        path: Directory path to create, or file path (will create parent directory).

    Returns:
        Path object representing the created directory.

    Examples:
        >>> ensure_directory("results/plots")  # Creates directory
        PosixPath('results/plots')
        >>> ensure_directory("results/data.json")  # Creates parent 'results' directory
        PosixPath('results')
    """
    path_obj = Path(path)
    # If the path has a suffix, assume it's a file and create parent directory
    if path_obj.suffix:
        directory = path_obj.parent
    else:
        directory = path_obj

    directory = resolve_artifact_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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


def determine_episode_outcome(info: dict[str, Any]) -> str:
    """Determine episode outcome from info dict.

    Args:
        info: Episode info dict from env.step().

    Returns:
        Outcome string: 'collision', 'success', 'timeout', or 'done'.

    Examples:
        >>> determine_episode_outcome({"collision": True})
        'collision'
        >>> determine_episode_outcome({"success": True})
        'success'
        >>> determine_episode_outcome({})
        'done'
    """
    if info.get("collision"):
        return "collision"
    if info.get("success"):
        return "success"
    if info.get("timeout"):
        return "timeout"
    return "done"


def format_overlay_text(scenario: str, seed: int, step: int, outcome: str | None = None) -> str:
    """Format overlay text for visualization.

    Args:
        scenario: Scenario name.
        seed: Random seed.
        step: Current step number.
        outcome: Optional outcome string.

    Returns:
        Formatted overlay text string.

    Examples:
        >>> format_overlay_text("test", 42, 100)
        'test | seed=42 | step=100'
        >>> format_overlay_text("test", 42, 100, "success")
        'test | seed=42 | step=100 | success'
    """
    base = f"{scenario} | seed={seed} | step={step}"
    if outcome:
        return base + f" | {outcome}"
    return base


def compute_fast_mode_and_cap(max_episodes: int) -> tuple[bool, int]:
    """Compute fast mode flag and episode cap for testing/demos.

    Args:
        max_episodes: Requested maximum episodes.

    Returns:
        Tuple of (fast_mode, effective_max_episodes).
        fast_mode is True if in pytest or ROBOT_SF_FAST_DEMO=1.
        effective_max_episodes is capped to 1 in fast mode.

    Examples:
        >>> # In normal mode
        >>> compute_fast_mode_and_cap(5)
        (False, 5)
        >>> # With ROBOT_SF_FAST_DEMO=1
        >>> import os
        ...
        ... os.environ["ROBOT_SF_FAST_DEMO"] = "1"
        >>> compute_fast_mode_and_cap(5)
        (True, 1)
    """
    in_pytest = "PYTEST_CURRENT_TEST" in os.environ
    fast_env_flag = bool(int(os.getenv("ROBOT_SF_FAST_DEMO", "0") or "0"))
    fast_mode = in_pytest or fast_env_flag
    if fast_mode and max_episodes > 1:
        max_episodes = 1
    return fast_mode, max_episodes


def format_episode_summary_table(rows: Iterable[dict[str, Any]]) -> str:
    """Format episode summaries as a readable table.

    Args:
        rows: Iterable of episode summary dicts with keys like 'scenario', 'seed', etc.

    Returns:
        Formatted table string with column alignment.

    Examples:
        >>> summaries = [
        ...     {
        ...         "scenario": "test",
        ...         "seed": 42,
        ...         "steps": 100,
        ...         "outcome": "success",
        ...         "recorded": True,
        ...     }
        ... ]
        >>> table = format_episode_summary_table(summaries)
        >>> "scenario" in table and "test" in table
        True
    """
    rows = list(rows)
    if not rows:
        return "(no episodes)"

    headers: list[str] = ["scenario", "seed", "steps", "outcome", "recorded"]
    col_widths = {h: max(len(h), *(len(str(cast("Any", r)[h])) for r in rows)) for h in headers}

    def fmt_row(r: dict[str, Any]) -> str:
        """TODO docstring. Document this function.

        Args:
            r: TODO docstring.

        Returns:
            TODO docstring.
        """
        return " | ".join(str(cast("Any", r)[h]).ljust(col_widths[h]) for h in headers)

    header_row: dict[str, Any] = {h: h for h in headers}
    lines = [fmt_row(header_row)]
    lines.append("-|-".join("-" * col_widths[h] for h in headers))
    lines.extend(fmt_row(r) for r in rows)
    return "\n" + "\n".join(lines)
