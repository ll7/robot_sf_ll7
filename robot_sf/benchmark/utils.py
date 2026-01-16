"""Small utility helpers for benchmark reporting.

Phase A helpers are intentionally small and pure. The public function
``format_summary_table`` follows the contract in
``specs/139-extract-reusable-helpers/contracts/visualization_helper_contract.md``:
- accepts a mapping ``dict[str, float]`` of metrics
- returns a Markdown table string
- raises ValueError on empty input
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

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
        >>> ensure_directory("output/results")  # Creates 'output' directory
        PosixPath('output')
        >>> ensure_directory("output/plots")  # Creates directory
        PosixPath('output/plots')
        >>> ensure_directory("output/data.json")  # Creates parent 'output' directory
        PosixPath('output')
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


def _git_hash_fallback() -> str:
    """Return the current git hash or 'unknown' if unavailable."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:  # pragma: no cover
        logger.debug("_git_hash_fallback failed: %s", exc)
        return "unknown"


def _config_hash(obj: Any) -> str:
    """Return a stable, deterministic 16-char hash from JSON serialization."""
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def compute_episode_id(scenario_params: dict[str, Any], seed: int) -> str:
    """Backward-compatible wrapper returning deterministic episode id.

    Uses a readable, stable id format suitable for resume semantics.

    Returns:
        Episode ID string in format <scenario_id>--<seed>.
    """
    scenario_id = (
        scenario_params.get("id")
        or scenario_params.get("name")
        or scenario_params.get("scenario_id")
        or "unknown"
    )
    return f"{scenario_id}--{seed}"


def episode_identity_hash() -> str:
    """Return a short hash that fingerprints episode identity definition."""
    try:
        src = inspect.getsource(compute_episode_id)
    except (OSError, TypeError):
        src = compute_episode_id.__name__
    return hashlib.sha256(src.encode()).hexdigest()[:12]


def index_existing(out_path: Path) -> set[str]:
    """Scan an existing JSONL file and return the set of episode_ids found.

    Returns:
        Set of episode_id values found in the JSONL file.
    """
    ids: set[str] = set()
    try:
        with out_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = rec.get("episode_id") if isinstance(rec, dict) else None
                if isinstance(eid, str):
                    ids.add(eid)
    except FileNotFoundError:
        return set()
    except OSError as exc:
        logger.debug("index_existing failed reading %s: %s", out_path, exc)
        return set()
    return ids


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
        """Format a row dict into a padded table row.

        Returns:
            Padded string row for the summary table.
        """
        return " | ".join(str(cast("Any", r)[h]).ljust(col_widths[h]) for h in headers)

    header_row: dict[str, Any] = {h: h for h in headers}
    lines = [fmt_row(header_row)]
    lines.append("-|-".join("-" * col_widths[h] for h in headers))
    lines.extend(fmt_row(r) for r in rows)
    return "\n" + "\n".join(lines)
