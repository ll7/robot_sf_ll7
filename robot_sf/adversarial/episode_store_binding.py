"""Lightweight validation of canonical episode-store bytes and row identity."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.termination_reason import (
    TERMINATION_REASONS,
    outcome_contradictions,
    status_from_termination_reason,
)

_EPISODE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark/schemas/episode.schema.v1.json"
)


def validate_episode_store_row_binding(
    episode_store_bytes: bytes,
    *,
    source: Mapping[str, Any],
    role: str,
    expected_episode_id: Any = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one normalized execution against its schema-valid source episode row.

    Replay rows carry the source execution outcome; the separate replay-result artifact binds the
    resimulated outcome.
    """
    if role not in {"reference", "target", "replay"}:
        raise ValueError("role must be 'reference', 'target', or 'replay'")
    episode, parse_reason = _episode_row_for_binding(
        episode_store_bytes,
        episode_id=(
            source.get("episode_id") if expected_episode_id is None else expected_episode_id
        ),
    )
    if episode is None:
        return None, parse_reason
    problem = _episode_row_binding_problem(
        episode,
        source,
        expected_episode_id=expected_episode_id,
        compare_route_outcome=role != "replay",
    )
    return (episode, None) if problem is None else (None, problem)


def _episode_row_for_binding(
    episode_store_bytes: bytes, *, episode_id: Any
) -> tuple[dict[str, Any] | None, str]:
    """Parse a JSONL store and select exactly one schema-valid source episode row."""
    if not isinstance(episode_id, str) or not episode_id.strip():
        return None, "episode_identity_mismatch"
    matches: list[dict[str, Any]] = []
    try:
        text = episode_store_bytes.decode("utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                return None, "episode_store_malformed"
            if value.get("episode_id") == episode_id:
                matches.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "episode_store_malformed"
    if len(matches) != 1:
        return None, "episode_identity_missing_or_ambiguous"
    episode = matches[0]
    if list(_episode_schema_validator().iter_errors(episode)):
        return None, "episode_store_episode_schema_invalid"
    return episode, ""


@lru_cache(maxsize=1)
def _episode_schema_validator() -> Draft202012Validator:
    """Reuse the canonical v1 episode validator across candidate classifications."""
    schema = json.loads(_EPISODE_SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def _episode_row_binding_problem(
    episode: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    expected_episode_id: Any = None,
    compare_route_outcome: bool = True,
) -> str | None:
    """Check source episode identity, clean runtime status, and route outcome against its row."""
    outcome = episode.get("outcome")
    integrity = episode.get("integrity")
    metadata = episode.get("algorithm_metadata")
    termination = episode.get("termination_reason")
    expected_id = source.get("episode_id") if expected_episode_id is None else expected_episode_id
    if (
        episode.get("episode_id") != expected_id
        or episode.get("scenario_id") != source.get("scenario_id")
        or episode.get("seed") != source.get("seed")
        or episode.get("algo") != source.get("planner_id")
        or episode.get("git_hash") != source.get("source_commit")
    ):
        return "episode_identity_mismatch"
    episode_horizon = episode.get("horizon")
    if (
        not isinstance(episode_horizon, int)
        or isinstance(episode_horizon, bool)
        or episode_horizon != source.get("horizon_steps")
    ):
        return "episode_horizon_mismatch"
    if (
        not isinstance(outcome, Mapping)
        or type(outcome.get("route_complete")) is not bool
        or (compare_route_outcome and outcome["route_complete"] is not source.get("route_complete"))
        or not isinstance(termination, str)
        or termination not in TERMINATION_REASONS
        or episode.get("status") != status_from_termination_reason(termination)
        or outcome_contradictions(
            termination_reason=termination,
            outcome=outcome,
            metrics=episode.get("metrics") if isinstance(episode.get("metrics"), Mapping) else None,
        )
    ):
        return "episode_outcome_mismatch"
    contradictions = integrity.get("contradictions") if isinstance(integrity, Mapping) else None
    if not isinstance(contradictions, list) or contradictions:
        return "episode_integrity_invalid_or_contradictory"
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("status") != "ok"
        or runtime_fallback_or_degraded_marker(dict(episode)) is not None
    ):
        return "episode_runtime_unavailable_or_degraded"
    return None
