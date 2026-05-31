"""Shared grouping helpers for benchmark report surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from robot_sf.benchmark.errors import AggregationMetadataError

DEFAULT_REPORT_GROUP_BY = "scenario_params.algo"
DEFAULT_REPORT_FALLBACK_GROUP_BY = "scenario_id"
EFFECTIVE_REPORT_GROUP_KEY = (
    "scenario_params.algo | algo | algorithm_metadata.algorithm | scenario_id"
)

MissingGroupPolicy = Literal["skip", "unknown", "error"]


def get_nested(record: Mapping[str, Any], path: str, default: Any | None = None) -> Any:
    """Read a dotted-path value from a mapping.

    Returns:
        The nested value when present, otherwise ``default``.
    """
    cur: Any = record
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _normalize_algo(value: Any) -> str | None:
    """Normalize algorithm identifiers to a non-empty string.

    Returns:
        Normalized string or None if empty/invalid.
    """
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
        return None
    if value is None:
        return None
    return str(value)


def resolve_report_group_key(
    record: Mapping[str, Any],
    *,
    group_by: str,
    fallback_group_by: str,
    missing: MissingGroupPolicy = "skip",
) -> str | None:
    """Resolve a report grouping key with the shared legacy-row fallback contract.

    Aggregate computation keeps a stricter fail-closed path because missing algorithm metadata can
    invalidate benchmark claims. Human-facing report surfaces historically accepted legacy rows by
    falling back to ``fallback_group_by`` or an explicit ``unknown`` bucket; this helper centralizes
    that compatibility behavior for tables, rankings, plots, distributions, SNQI ablations, and
    seed-variance summaries.

    Args:
        record: Episode record.
        group_by: Primary dotted path.
        fallback_group_by: Fallback dotted path.
        missing: ``"skip"`` for ``None``, ``"unknown"`` for an explicit bucket, or ``"error"`` to
            raise when neither primary nor fallback exists.

    Returns:
        Group key string, ``"unknown"``, or ``None`` depending on ``missing``.
    """
    if missing not in {"skip", "unknown", "error"}:
        raise ValueError("missing must be one of {'skip', 'unknown', 'error'}")

    primary_value = get_nested(record, group_by)
    if primary_value is not None:
        return str(primary_value)

    if group_by == DEFAULT_REPORT_GROUP_BY:
        top_level_algo = _normalize_algo(record.get("algo"))
        if top_level_algo is not None:
            return top_level_algo
        metadata_algo = _normalize_algo(get_nested(record, "algorithm_metadata.algorithm"))
        if metadata_algo is not None:
            return metadata_algo

    fallback_value = get_nested(record, fallback_group_by)
    if fallback_value is not None:
        return str(fallback_value)

    if missing == "unknown":
        return "unknown"
    if missing == "error":
        episode_id = record.get("episode_id")
        raise AggregationMetadataError(
            "Unable to determine report group key for episode.",
            episode_id=str(episode_id) if episode_id is not None else None,
            missing_fields=(group_by, fallback_group_by),
            advice="Verify that episode records include the requested report grouping metadata.",
        )
    return None
