"""Build renderer-neutral ``simulation_timeline.v1`` artifacts from trace exports."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExport,
    SimulationTraceFrame,
    load_simulation_trace_export,
)
from robot_sf.errors import RobotSfError

SIMULATION_TIMELINE_SCHEMA_VERSION = "simulation_timeline.v1"
SIMULATION_TIMELINE_SCHEMA_FILE = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "analysis_workbench"
    / "schemas"
    / "simulation_timeline.v1.json"
)
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "planner_sanity_open_episode_0000.json"
)


class SimulationTimelineValidationError(RobotSfError, ValueError):
    """Raised when a ``simulation_timeline.v1`` payload fails schema validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_simulation_timeline_schema() -> dict[str, Any]:
    """Load the public ``simulation_timeline.v1`` JSON schema.

    Returns:
        Parsed JSON Schema document.
    """

    return json.loads(SIMULATION_TIMELINE_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_simulation_timeline(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> None:
    """Validate a timeline payload against the public JSON schema."""

    validator = Draft202012Validator(load_simulation_timeline_schema())
    errors = [
        f"{_json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda err: list(err.absolute_path),
        )
    ]
    if errors:
        raise SimulationTimelineValidationError(errors, source=source)


def build_simulation_timeline(input_path: Path) -> dict[str, Any]:
    """Build a renderer-neutral timeline artifact from a trace export JSON file.

    Returns:
        Schema-valid ``simulation_timeline.v1`` payload.
    """

    trace = load_simulation_trace_export(input_path)
    payload: dict[str, Any] = {
        "schema_version": SIMULATION_TIMELINE_SCHEMA_VERSION,
        "timeline_id": f"{trace.trace_id}-timeline",
        "source_trace": _source_trace_metadata(trace),
        "evidence_boundary": trace.evidence_boundary,
        "coordinate_frame": trace.coordinate_frame,
        "units": trace.units,
        "frames": [
            _timeline_frame(frame, frame_index=index) for index, frame in enumerate(trace.frames)
        ],
        "events": [
            event
            for index, frame in enumerate(trace.frames)
            if (event := _timeline_event(frame, frame_index=index)) is not None
        ],
    }
    validate_simulation_timeline(payload, source=input_path)
    return payload


def write_simulation_timeline(input_path: Path, output_path: Path) -> Path:
    """Write a schema-valid timeline artifact.

    Returns:
        Output path that received the timeline JSON.
    """

    payload = build_simulation_timeline(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _source_trace_metadata(trace: SimulationTraceExport) -> dict[str, Any]:
    """Return source-trace metadata copied from the validated trace export."""

    return {
        "schema_version": SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
        "trace_id": trace.trace_id,
        "source": {
            "scenario_id": trace.source.scenario_id,
            "seed": trace.source.seed,
            "planner_id": trace.source.planner_id,
            "episode_id": trace.source.episode_id,
            "generated_by": trace.source.generated_by,
        },
    }


def _timeline_frame(frame: SimulationTraceFrame, *, frame_index: int) -> dict[str, Any]:
    """Convert one trace frame into renderer-neutral timeline state.

    Returns:
        Frame payload ready for ``simulation_timeline.v1`` schema validation.
    """

    return {
        "frame_index": frame_index,
        "step": frame.step,
        "time_s": frame.time_s,
        "state": {
            "planner_decision": _planner_decision(frame.planner),
            "policy": _optional_mapping_slot(frame.planner, "policy", "policy_state"),
            "sensors": _mapping_slot(frame.planner, "sensors", "sensor_state"),
            "reward": _optional_mapping_slot(frame.planner, "reward", "reward_state"),
            "collision": _collision_slot(frame.planner),
            "route": _optional_mapping_slot(frame.planner, "route", "route_state"),
            "robot": dict(frame.robot),
            "pedestrians": [dict(pedestrian) for pedestrian in frame.pedestrians],
        },
    }


def _planner_decision(planner: Mapping[str, Any]) -> dict[str, Any]:
    """Return planner-decision fields without duplicating known optional state slots."""

    excluded = {
        "policy",
        "policy_state",
        "sensors",
        "sensor_state",
        "reward",
        "reward_state",
        "collision",
        "collision_state",
        "route",
        "route_state",
    }
    return {str(key): value for key, value in planner.items() if str(key) not in excluded}


def _mapping_slot(planner: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    """Return the first mapping-valued slot from ``keys``, or an empty slot."""

    for key in keys:
        value = planner.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _optional_mapping_slot(planner: Mapping[str, Any], *keys: str) -> dict[str, Any] | None:
    """Return the first mapping-valued slot from ``keys``, or ``None`` when unavailable."""

    for key in keys:
        value = planner.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return None


def _collision_slot(planner: Mapping[str, Any]) -> dict[str, Any]:
    """Return collision state while making unknown evidence explicit."""

    for key in ("collision", "collision_state"):
        value = planner.get(key)
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, bool):
            return {"status": "observed" if value else "none", "value": value}
    return {"status": "unknown"}


def _timeline_event(frame: SimulationTraceFrame, *, frame_index: int) -> dict[str, Any] | None:
    """Return a stable event record when a planner frame declares an event."""

    raw_event = frame.planner.get("event")
    if not isinstance(raw_event, str) or not raw_event:
        return None
    event_type = raw_event.strip()
    if not event_type:
        return None
    event_id = frame.planner.get("event_id")
    if isinstance(event_id, str):
        event_id = event_id.strip()
    if not event_id:
        event_id = f"frame-{frame_index:04d}-{_slug(event_type)}"
    return {
        "event_id": event_id,
        "event_type": event_type,
        "frame_index": frame_index,
        "step": frame.step,
        "time_s": frame.time_s,
    }


def _slug(value: str) -> str:
    """Return a compact identifier fragment."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "event"


def _json_pointer(path: Any) -> str:
    """Render a JSON Schema error path as an RFC6901-style pointer.

    Returns:
        JSON pointer string for the validation error path.
    """

    parts = [str(part).replace("~", "~0").replace("/", "~1") for part in path]
    return "/" + "/".join(parts) if parts else "/"
