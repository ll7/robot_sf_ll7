"""Typed loader for ``simulation_trace_export.v1`` analysis-workbench traces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

SIMULATION_TRACE_EXPORT_SCHEMA_VERSION = "simulation_trace_export.v1"
SIMULATION_TRACE_EXPORT_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "simulation_trace_export.v1.json"
)


@dataclass(frozen=True, slots=True)
class SimulationTraceSource:
    """Source metadata for an exported simulation trace."""

    scenario_id: str
    seed: int
    planner_id: str
    episode_id: str
    generated_by: str


@dataclass(frozen=True, slots=True)
class SimulationTraceFrame:
    """One playback frame in an analysis-workbench trace."""

    step: int
    time_s: float
    robot: dict[str, Any]
    pedestrians: list[dict[str, Any]]
    planner: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SimulationTraceExport:
    """Typed ``simulation_trace_export.v1`` payload."""

    schema_version: str
    trace_id: str
    source: SimulationTraceSource
    evidence_boundary: str
    coordinate_frame: str
    units: dict[str, str]
    frames: list[SimulationTraceFrame]

    def to_dict(self) -> dict[str, Any]:
        """Convert the export to JSON-safe primitives.

        Returns:
            Dictionary representation suitable for JSON Schema validation.
        """

        return asdict(self)


class SimulationTraceExportValidationError(RobotSfError, ValueError):
    """Raised when a simulation trace export fails validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_simulation_trace_export_schema() -> dict[str, Any]:
    """Load the public ``simulation_trace_export.v1`` JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(SIMULATION_TRACE_EXPORT_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_simulation_trace_export(path: Path) -> SimulationTraceExport:
    """Load one simulation trace export from JSON.

    Returns:
        Typed simulation trace export.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise SimulationTraceExportValidationError(["expected a mapping payload"], source=path)
    return simulation_trace_export_from_dict(raw, source=path)


def simulation_trace_export_from_dict(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> SimulationTraceExport:
    """Validate and convert a mapping into a typed simulation trace export.

    Returns:
        Typed simulation trace export.
    """

    errors = _schema_validation_errors(payload)
    errors.extend(_semantic_validation_errors(payload))
    if errors:
        raise SimulationTraceExportValidationError(errors, source=source)
    return _export_from_payload(payload)


def _schema_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return sorted JSON Schema validation errors."""

    validator = Draft202012Validator(load_simulation_trace_export_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda err: list(err.absolute_path),
        )
    ]


def _semantic_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return trace-order validation errors not expressible in the JSON Schema."""

    frames = payload.get("frames")
    if not isinstance(frames, list):
        return []
    errors: list[str] = []
    previous_step: int | None = None
    previous_time: float | None = None
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            continue
        step = frame.get("step")
        time_s = frame.get("time_s")
        if isinstance(step, int):
            if previous_step is not None and step <= previous_step:
                errors.append(f"/frames/{index}/step: expected strictly increasing step")
            previous_step = step
        if isinstance(time_s, int | float):
            time_value = float(time_s)
            if previous_time is not None and time_value <= previous_time:
                errors.append(f"/frames/{index}/time_s: expected strictly increasing time_s")
            previous_time = time_value
    return errors


def _export_from_payload(payload: Mapping[str, Any]) -> SimulationTraceExport:
    """Build a typed export from a schema-valid payload.

    Returns:
        Typed simulation trace export.
    """

    source = payload["source"]
    return SimulationTraceExport(
        schema_version=str(payload["schema_version"]),
        trace_id=str(payload["trace_id"]),
        source=SimulationTraceSource(
            scenario_id=str(source["scenario_id"]),
            seed=int(source["seed"]),
            planner_id=str(source["planner_id"]),
            episode_id=str(source["episode_id"]),
            generated_by=str(source["generated_by"]),
        ),
        evidence_boundary=str(payload["evidence_boundary"]),
        coordinate_frame=str(payload["coordinate_frame"]),
        units=dict(payload["units"]),
        frames=[
            SimulationTraceFrame(
                step=int(frame["step"]),
                time_s=float(frame["time_s"]),
                robot=dict(frame["robot"]),
                pedestrians=[dict(pedestrian) for pedestrian in frame["pedestrians"]],
                planner=dict(frame["planner"]),
            )
            for frame in payload["frames"]
        ],
    )
