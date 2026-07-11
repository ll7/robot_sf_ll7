"""Validate generated-scenario catalog entries before they can be persisted.

The contract intentionally marks every entry as a review-pending hypothesis.  It
does not make generated scenarios part of the benchmark matrix or evidence base.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.errors import RobotSfError

CATALOG_ENTRY_SCHEMA_VERSION = "generated-scenario-catalog-entry.v1"
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "generated_scenario_catalog_entry.v1.json"
)


class GeneratedScenarioCatalogValidationError(RobotSfError, ValueError):
    """Raised when a catalog entry cannot safely be stored as a hypothesis."""


def load_catalog_entry_schema() -> dict[str, Any]:
    """Return the versioned JSON Schema for generated catalog entries."""

    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return json.load(schema_file)


def validate_catalog_entry(entry: Mapping[str, Any]) -> None:
    """Fail closed unless *entry* is a non-benchmark, replay-pending catalog entry.

    Raises:
        GeneratedScenarioCatalogValidationError: If JSON Schema or temporal
            invariants reject the entry.
    """

    if not isinstance(entry, Mapping):
        raise GeneratedScenarioCatalogValidationError("catalog entry must be a mapping")

    _validate_json_schema(entry)
    _validate_temporal_contract(entry)
    _validate_replay_contract(entry)
    _validate_trace_frames(entry)


def _validate_json_schema(entry: Mapping[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(load_catalog_entry_schema()).iter_errors(dict(entry)),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        formatted = "; ".join(
            f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
            for error in errors
        )
        raise GeneratedScenarioCatalogValidationError(formatted)


def _validate_temporal_contract(entry: Mapping[str, Any]) -> None:
    segment = entry["segment"]
    start_s = float(segment["window_start_s"])
    end_s = float(segment["window_end_s"])
    observed_at_s = float(entry["criticality"]["observed_at_s"])
    if not all(math.isfinite(value) for value in (start_s, end_s, observed_at_s)):
        raise GeneratedScenarioCatalogValidationError("segment times must be finite")
    if end_s < start_s:
        raise GeneratedScenarioCatalogValidationError(
            "segment.window_end_s must be >= window_start_s"
        )
    if not start_s <= observed_at_s <= end_s:
        raise GeneratedScenarioCatalogValidationError(
            "criticality.observed_at_s must lie inside the extracted segment"
        )


def _validate_replay_contract(entry: Mapping[str, Any]) -> None:
    if entry["replay"]["source_seed"] != entry["source_episode"]["source_seed"]:
        raise GeneratedScenarioCatalogValidationError(
            "replay.source_seed must equal source_episode.source_seed"
        )
    if entry["replay"]["status"] == "not_representable_yet" and not any(
        warning.startswith("replay_gap:") for warning in entry["replay"]["warnings"]
    ):
        raise GeneratedScenarioCatalogValidationError(
            "not_representable_yet entries must include a replay_gap warning"
        )


def _validate_trace_frames(entry: Mapping[str, Any]) -> None:
    segment = entry["segment"]
    start_s = float(segment["window_start_s"])
    end_s = float(segment["window_end_s"])
    frames = segment["trace_frames"]
    frame_times = [float(frame["time_s"]) for frame in frames]
    if frame_times != sorted(frame_times) or len(set(frame_times)) != len(frame_times):
        raise GeneratedScenarioCatalogValidationError(
            "segment.trace_frames must have increasing time_s"
        )
    if frame_times[0] != start_s or frame_times[-1] != end_s:
        raise GeneratedScenarioCatalogValidationError(
            "segment bounds must equal the first and last trace-frame timestamps"
        )
    if segment["initial_robot_state"] != frames[0]["robot"]:
        raise GeneratedScenarioCatalogValidationError(
            "segment.initial_robot_state must equal the first trace-frame robot state"
        )
    for frame in frames:
        _require_finite_position(frame["robot"]["position"])
        for pedestrian in frame["pedestrians"]:
            _require_finite_position(pedestrian["position"])


def _require_finite_position(position: list[object]) -> None:
    if not all(isinstance(value, int | float) and math.isfinite(value) for value in position):
        raise GeneratedScenarioCatalogValidationError("trace positions must be finite numbers")


__all__ = [
    "CATALOG_ENTRY_SCHEMA_VERSION",
    "GeneratedScenarioCatalogValidationError",
    "load_catalog_entry_schema",
    "validate_catalog_entry",
]
