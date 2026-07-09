"""Exemplar episode auto-selection for campaign review.

Selects representative episodes per (planner x mechanism/outcome) cell from
campaign episodes.jsonl.  Emits a deterministic selection manifest that can be
fed into the replay bridge (#4776) or used for direct figure generation.

Part of issue #4778: exemplar selection + multi-planner trajectory overlays.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

SELECTION_MANIFEST_SCHEMA_VERSION = "exemplar-selection.v1"

SelectionMode = Literal["median", "best", "worst"]

VALID_MODES: frozenset[str] = frozenset({"median", "best", "worst"})

# Metric direction: "lower" means smaller is better (e.g. collisions),
# "higher" means larger is better (e.g. path_efficiency).
MetricDirection = Literal["lower", "higher"]

METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "collisions": "lower",
    "near_misses": "lower",
    "comfort_exposure": "lower",
    "clearing_distance_min": "higher",
    "clearing_distance_avg": "higher",
    "path_efficiency": "higher",
    "time_to_goal_norm": "lower",
    "energy": "lower",
    "jerk_mean": "lower",
    "socnavbench_path_irregularity": "lower",
    "success": "higher",
}


class ExemplarSelectionError(Exception):
    """Raised when exemplar selection inputs are malformed."""


@dataclass(frozen=True, slots=True)
class SelectedEpisode:
    """One selected exemplar episode within a cell."""

    episode_id: str
    planner_key: str
    scenario_id: str
    seed: int
    selection_mode: SelectionMode
    selection_rank: int
    metric_value: float
    reason: str


@dataclass(frozen=True, slots=True)
class SkippedCell:
    """A (planner x mechanism/outcome) cell that was not evaluable."""

    cell_key: str
    reason: str


@dataclass
class SelectionManifest:
    """The full exemplar selection manifest."""

    schema_version: str = SELECTION_MANIFEST_SCHEMA_VERSION
    source_episodes: str = ""
    source_sha256: str = ""
    group_by: list[str] = field(default_factory=list)
    metric: str = ""
    metric_direction: MetricDirection = "lower"
    selected: list[SelectedEpisode] = field(default_factory=list)
    skipped_cells: list[SkippedCell] = field(default_factory=list)
    git_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict.

        Returns:
            JSON-serializable dict representation.
        """
        return asdict(self)


def _git_sha_short(length: int = 7) -> str:
    """Return short git SHA for the current HEAD.

    Returns:
        Short git SHA or "unknown" if unavailable.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", f"--short={length}", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode("utf-8")
            .strip()
        )
        return sha or "unknown"
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ):
        return "unknown"


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of a file for provenance.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_nested(record: dict[str, Any], path: str, default: Any = None) -> Any:
    """Resolve a dotted-path value from a dict.

    Returns:
        Value at the dotted path, or *default* when any segment is missing.
    """
    cur: Any = record
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _normalize_planner_key(record: dict[str, Any]) -> str:
    """Extract planner key from an episode record.

    Returns:
        Planner identifier string.
    """
    for path in ("algo", "scenario_params.algo", "algorithm_metadata.algorithm"):
        val = _get_nested(record, path)
        if val is not None:
            return str(val).strip()
    return "unknown_planner"


def _normalize_mechanism_key(record: dict[str, Any]) -> str:
    """Extract mechanism label from an episode record.

    Returns:
        Mechanism label string.
    """
    val = _get_nested(record, "mechanism_label")
    if val is not None:
        return str(val).strip()
    return "unknown_mechanism"


def _normalize_outcome_key(record: dict[str, Any]) -> str:
    """Extract outcome classification from an episode record.

    Returns:
        Outcome classification string.
    """
    outcome = record.get("outcome", {})
    if isinstance(outcome, dict):
        if outcome.get("collision_event"):
            return "collision"
        if outcome.get("route_complete"):
            return "success"
        if outcome.get("timeout_event"):
            return "timeout"
    status = record.get("status")
    if status is not None:
        return str(status)
    return "unknown_outcome"


def _normalize_grouping_key(record: dict[str, Any], key: str) -> str:
    """Extract and normalize a grouping key from an episode record.

    Returns:
        Normalized grouping key string.
    """
    if key == "planner_key":
        return _normalize_planner_key(record)

    if key == "mechanism_label":
        return _normalize_mechanism_key(record)

    if key == "outcome":
        return _normalize_outcome_key(record)

    # Generic dotted path
    val = _get_nested(record, key)
    if val is not None:
        return str(val)
    return f"unknown_{key}"


def _extract_metric(record: dict[str, Any], metric: str) -> float | None:
    """Extract a metric value from an episode record.

    Returns:
        Metric value or ``None`` when missing or non-finite.
    """
    metrics = record.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    val = metrics.get(metric)
    if val is None:
        return None

    if isinstance(val, bool):
        return 1.0 if val else 0.0

    if isinstance(val, (int, float)):
        fval = float(val)
        if math.isfinite(fval):
            return fval
        return None

    return None


def _extract_step_count(record: dict[str, Any]) -> int | None:
    """Extract step count from an episode record for tie-breaking.

    Returns:
        Step count or ``None`` when missing.
    """
    # Direct step_count field
    val = record.get("step_count")
    if isinstance(val, (int, float)) and val > 0:
        return int(val)

    # From summary or metrics
    for path in ("summary.step_count", "metrics.step_count"):
        val = _get_nested(record, path)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)

    # From trace data
    trace = record.get("algorithm_metadata", {}).get("simulation_step_trace")
    if trace and "steps" in trace:
        return len(trace["steps"])

    return None


def _build_cell_key(group_values: dict[str, str]) -> str:
    """Build a deterministic cell key from group values.

    Returns:
        Deterministic cell key string.
    """
    parts = [f"{k}={v}" for k, v in sorted(group_values.items())]
    return "|".join(parts)


def _filter_by_step_count(
    cell_episodes: list[dict[str, Any]],
    min_step_count: int,
    cell_key: str,
) -> list[dict[str, Any]] | SkippedCell:
    """Filter episodes by minimum step count.

    Returns:
        Filtered episodes list, or SkippedCell if all episodes are filtered out.
    """
    filtered = []
    for ep in cell_episodes:
        steps = _extract_step_count(ep)
        if steps is not None and steps >= min_step_count:
            filtered.append(ep)
    if not filtered:
        return SkippedCell(
            cell_key=cell_key,
            reason=f"all {len(cell_episodes)} episodes have step_count < {min_step_count}",
        )
    return filtered


def _build_sort_key(
    effective_tie_breaker: str | None,
) -> Any:
    """Build sort key function for scored episodes.

    Returns:
        Sort key function.
    """

    def _sort_key(item: tuple[float, dict[str, Any]]) -> tuple:
        primary_val, ep = item
        if effective_tie_breaker:
            tb_val = _extract_metric(ep, effective_tie_breaker)
            if tb_val is None:
                return (primary_val, float("-inf"))
            return (primary_val, tb_val)
        return (primary_val,)

    return _sort_key


def _select_in_cell(
    cell_episodes: list[dict[str, Any]],
    metric: str,
    metric_direction: MetricDirection,
    modes: Sequence[SelectionMode],
    group_values: dict[str, str],
    *,
    tie_breaker: str | None = None,
    min_step_count: int | None = None,
) -> tuple[list[SelectedEpisode], SkippedCell | None]:
    """Select exemplars within one cell.

    Args:
        cell_episodes: Episodes in this cell.
        metric: Primary ranking metric.
        metric_direction: ``"lower"`` or ``"higher"``.
        modes: Selection modes to apply.
        group_values: Grouping key values for this cell.
        tie_breaker: Optional secondary metric for tie-breaking when
            primary metric values are equal.
        min_step_count: Optional minimum step count filter. Episodes with
            fewer steps are excluded before selection.

    Returns:
        Tuple of (selected episodes, optional skipped cell).
    """
    cell_key = _build_cell_key(group_values)

    # Filter by minimum step count if specified
    filtered_episodes = cell_episodes
    if min_step_count is not None:
        result = _filter_by_step_count(cell_episodes, min_step_count, cell_key)
        if isinstance(result, SkippedCell):
            return [], result
        filtered_episodes = result

    # Extract metric values
    scored: list[tuple[float, dict[str, Any]]] = []
    for ep in filtered_episodes:
        val = _extract_metric(ep, metric)
        if val is not None:
            scored.append((val, ep))

    if not scored:
        return [], SkippedCell(
            cell_key=cell_key,
            reason=f"metric '{metric}' missing or non-finite for all {len(filtered_episodes)} episodes",
        )

    # Determine effective tie-breaker
    effective_tie_breaker = tie_breaker
    if effective_tie_breaker is None and min_step_count is not None:
        effective_tie_breaker = "step_count"

    # Sort with optional tie-breaker
    scored.sort(key=_build_sort_key(effective_tie_breaker))

    return _build_selections(scored, modes, metric_direction, group_values, effective_tie_breaker)


def _build_selections(
    scored: list[tuple[float, dict[str, Any]]],
    modes: Sequence[SelectionMode],
    metric_direction: MetricDirection,
    group_values: dict[str, str],
    effective_tie_breaker: str | None,
) -> tuple[list[SelectedEpisode], None]:
    """Build SelectedEpisode list from scored episodes.

    Returns:
        Tuple of (selected episodes, None).
    """
    cell_key = _build_cell_key(group_values)
    planner_key = group_values.get("planner_key", "unknown_planner")
    results: list[SelectedEpisode] = []

    for mode in modes:
        if mode == "best":
            idx = len(scored) - 1 if metric_direction == "higher" else 0
        elif mode == "worst":
            idx = 0 if metric_direction == "higher" else len(scored) - 1
        else:  # median
            idx = len(scored) // 2

        val, ep = scored[idx]
        episode_id = ep.get("episode_id", "")
        scenario_id = ep.get("scenario_id", "")
        seed = ep.get("seed", 0)

        reason_parts = [f"{mode} within {cell_key}"]
        if effective_tie_breaker and len(scored) > 1:
            tb_val = _extract_metric(ep, effective_tie_breaker)
            if tb_val is not None:
                reason_parts.append(f"tie_breaker={effective_tie_breaker}={tb_val}")

        results.append(
            SelectedEpisode(
                episode_id=str(episode_id),
                planner_key=planner_key,
                scenario_id=str(scenario_id),
                seed=int(seed),
                selection_mode=mode,
                selection_rank=idx,
                metric_value=val,
                reason=", ".join(reason_parts),
            )
        )

    return results, None


def select_exemplars(
    episodes: list[dict[str, Any]],
    *,
    group_by: Sequence[str],
    metric: str,
    metric_direction: MetricDirection | None = None,
    modes: Sequence[SelectionMode] = ("median", "best", "worst"),
    tie_breaker: str | None = None,
    min_step_count: int | None = None,
) -> tuple[list[SelectedEpisode], list[SkippedCell]]:
    """Select exemplar episodes grouped by cell keys.

    Args:
        episodes: Parsed episode records from JSONL.
        group_by: Grouping keys (e.g., ``["planner_key", "outcome"]``).
        metric: Metric name to rank by.
        metric_direction: ``"lower"`` or ``"higher"``.  Auto-detected from
            ``METRIC_DIRECTIONS`` when ``None``.
        modes: Selection modes to apply per cell.
        tie_breaker: Optional secondary metric for tie-breaking when
            primary metric values are equal (e.g., ``"step_count"``).
        min_step_count: Optional minimum step count filter. Episodes with
            fewer steps are excluded before selection.

    Returns:
        Tuple of (selected episodes, skipped cells).
    """
    if metric_direction is None:
        metric_direction = METRIC_DIRECTIONS.get(metric, "lower")

    # Validate modes
    for m in modes:
        if m not in VALID_MODES:
            raise ExemplarSelectionError(f"Invalid selection mode: {m!r}")

    # Group episodes by cell
    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cell_group_values: dict[str, dict[str, str]] = {}

    for ep in episodes:
        group_values = {}
        for key in group_by:
            group_values[key] = _normalize_grouping_key(ep, key)
        cell_key = _build_cell_key(group_values)
        cells[cell_key].append(ep)
        cell_group_values[cell_key] = group_values

    # Select within each cell
    all_selected: list[SelectedEpisode] = []
    all_skipped: list[SkippedCell] = []

    for cell_key in sorted(cells.keys()):
        selected, skipped = _select_in_cell(
            cells[cell_key],
            metric,
            metric_direction,
            modes,
            cell_group_values[cell_key],
            tie_breaker=tie_breaker,
            min_step_count=min_step_count,
        )
        all_selected.extend(selected)
        if skipped is not None:
            all_skipped.append(skipped)

    return all_selected, all_skipped


def build_manifest(
    *,
    source_episodes_path: Path | str,
    group_by: Sequence[str],
    metric: str,
    metric_direction: MetricDirection | None = None,
    selected: list[SelectedEpisode],
    skipped_cells: list[SkippedCell],
) -> SelectionManifest:
    """Build a complete selection manifest from selection results.

    Returns:
        Populated ``SelectionManifest``.
    """
    source_path = Path(source_episodes_path)
    source_sha256 = _compute_file_sha256(source_path) if source_path.is_file() else "unknown"

    if metric_direction is None:
        metric_direction = METRIC_DIRECTIONS.get(metric, "lower")

    return SelectionManifest(
        source_episodes=str(source_path),
        source_sha256=source_sha256,
        group_by=list(group_by),
        metric=metric,
        metric_direction=metric_direction,
        selected=selected,
        skipped_cells=skipped_cells,
        git_hash=_git_sha_short(),
    )


def save_manifest(manifest: SelectionManifest, output_path: Path | str) -> Path:
    """Write the selection manifest to a JSON file.

    Returns:
        Path to the written manifest file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=False)
    return output


__all__ = [
    "METRIC_DIRECTIONS",
    "SELECTION_MANIFEST_SCHEMA_VERSION",
    "ExemplarSelectionError",
    "MetricDirection",
    "SelectedEpisode",
    "SelectionManifest",
    "SelectionMode",
    "SkippedCell",
    "build_manifest",
    "save_manifest",
    "select_exemplars",
]
