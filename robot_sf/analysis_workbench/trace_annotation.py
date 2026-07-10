"""Typed loader for ``trace_annotation_set.v1`` analysis-workbench annotations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExport,
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
)
from robot_sf.common.json_pointer import json_pointer
from robot_sf.errors import RobotSfError

TRACE_ANNOTATION_SET_SCHEMA_VERSION = "trace_annotation_set.v1"
TRACE_ANNOTATION_SET_SCHEMA_FILE = (
    Path(__file__).with_name("schemas") / "trace_annotation_set.v1.json"
)


@dataclass(frozen=True, slots=True)
class TraceAnnotationTimelineRef:
    """Reference to the trace export annotated by an annotation set."""

    path: str
    schema_version: str
    trace_id: str


@dataclass(frozen=True, slots=True)
class TraceAnnotationProvenance:
    """Review provenance and evidence boundary for annotation payloads."""

    kind: str
    source_issue: str
    author: str
    created_at: str
    evidence_boundary: str


@dataclass(frozen=True, slots=True)
class TraceAnnotationEntityRef:
    """Entity named by a frame-range annotation."""

    type: str
    id: str


@dataclass(frozen=True, slots=True)
class TraceAnnotationAnchor:
    """Frame-range anchor for a qualitative annotation."""

    frame_start: int
    frame_end: int
    event_ids: list[str]
    entities: list[TraceAnnotationEntityRef]


@dataclass(frozen=True, slots=True)
class TraceAnnotation:
    """One qualitative annotation on a simulation trace export."""

    annotation_id: str
    category: str
    evidence_type: str
    anchor: TraceAnnotationAnchor
    summary: str
    details: str | None = None


@dataclass(frozen=True, slots=True)
class TraceAnnotationSet:
    """Typed ``trace_annotation_set.v1`` payload."""

    schema_version: str
    annotation_set_id: str
    timeline: TraceAnnotationTimelineRef
    provenance: TraceAnnotationProvenance
    annotations: list[TraceAnnotation]

    def to_dict(self) -> dict[str, Any]:
        """Convert the annotation set to JSON-safe primitives.

        Returns:
            Dictionary representation suitable for JSON Schema validation.
        """

        payload = asdict(self)
        for annotation in payload["annotations"]:
            if annotation.get("details") is None:
                annotation.pop("details", None)
        return payload


class TraceAnnotationSetValidationError(RobotSfError, ValueError):
    """Raised when a trace annotation set fails validation."""

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = tuple(errors)
        self.source = str(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(errors))


@lru_cache(maxsize=1)
def load_trace_annotation_set_schema() -> dict[str, Any]:
    """Load the public ``trace_annotation_set.v1`` JSON schema.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(TRACE_ANNOTATION_SET_SCHEMA_FILE.read_text(encoding="utf-8"))


def load_trace_annotation_set(path: Path) -> TraceAnnotationSet:
    """Load one trace annotation set from JSON.

    Returns:
        Typed trace annotation set.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TraceAnnotationSetValidationError(["expected a mapping payload"], source=path)
    return trace_annotation_set_from_dict(raw, source=path)


def trace_annotation_set_from_dict(
    payload: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> TraceAnnotationSet:
    """Validate and convert a mapping into a typed trace annotation set.

    Returns:
        Typed trace annotation set.
    """

    source_path = Path(source) if source is not None else None
    errors = _schema_validation_errors(payload)
    trace = _load_referenced_trace(payload, source_path=source_path, errors=errors)
    if trace is not None:
        errors.extend(_semantic_validation_errors(payload, trace))
    if errors:
        raise TraceAnnotationSetValidationError(errors, source=source)
    return _annotation_set_from_payload(payload)


def _schema_validation_errors(payload: Mapping[str, Any]) -> list[str]:
    """Return sorted JSON Schema validation errors."""

    validator = Draft202012Validator(load_trace_annotation_set_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(
            validator.iter_errors(payload),
            key=lambda err: list(err.absolute_path),
        )
    ]


def _load_referenced_trace(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None,
    errors: list[str],
) -> SimulationTraceExport | None:
    """Load the referenced timeline when enough payload fields are schema-shaped.

    Returns:
        Referenced trace export, or ``None`` when the reference is invalid.
    """

    timeline = payload.get("timeline")
    if not isinstance(timeline, Mapping):
        return None
    raw_path = timeline.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    timeline_path = Path(raw_path)
    if timeline_path.is_absolute():
        errors.append("/timeline/path: expected repository-relative or fixture-relative path")
        return None
    if any(part in {"output", "results"} for part in timeline_path.parts):
        errors.append("/timeline/path: expected tracked fixture path, not generated output")
        return None
    base_dir = source_path.parent if source_path is not None and source_path.parent else Path.cwd()
    resolved_path = (base_dir / timeline_path).resolve()
    if not resolved_path.exists():
        errors.append(f"/timeline/path: referenced trace fixture does not exist: {raw_path}")
        return None
    try:
        return load_simulation_trace_export(resolved_path)
    except (
        OSError,
        json.JSONDecodeError,
        SimulationTraceExportValidationError,
    ) as exc:  # pragma: no cover - exercised by trace-export tests.
        errors.append(f"/timeline/path: referenced trace fixture is invalid: {exc}")
        return None


def _semantic_validation_errors(
    payload: Mapping[str, Any],
    trace: SimulationTraceExport,
) -> list[str]:
    """Return cross-file validation errors for a trace annotation payload."""

    errors: list[str] = []
    errors.extend(_timeline_validation_errors(payload, trace))

    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        return errors

    frame_steps = {frame.step for frame in trace.frames}
    event_ids = {
        event_id
        for frame in trace.frames
        if isinstance(event_id := frame.planner.get("event_id"), str)
    }
    pedestrian_ids = {
        str(pedestrian["id"])
        for frame in trace.frames
        for pedestrian in frame.pedestrians
        if "id" in pedestrian
    }

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            continue
        anchor = annotation.get("anchor")
        if isinstance(anchor, Mapping):
            errors.extend(
                _anchor_validation_errors(
                    anchor,
                    index=index,
                    frame_steps=frame_steps,
                    event_ids=event_ids,
                    pedestrian_ids=pedestrian_ids,
                )
            )
    return errors


def _timeline_validation_errors(
    payload: Mapping[str, Any],
    trace: SimulationTraceExport,
) -> list[str]:
    """Return errors for the annotation-to-trace reference."""

    errors: list[str] = []
    timeline = payload.get("timeline")
    if isinstance(timeline, Mapping):
        if timeline.get("schema_version") != SIMULATION_TRACE_EXPORT_SCHEMA_VERSION:
            errors.append("/timeline/schema_version: unsupported trace schema version")
        if timeline.get("trace_id") != trace.trace_id:
            errors.append(f"/timeline/trace_id: expected referenced trace_id {trace.trace_id!r}")
    return errors


def _anchor_validation_errors(
    anchor: Mapping[str, Any],
    *,
    index: int,
    frame_steps: set[int],
    event_ids: set[str],
    pedestrian_ids: set[str],
) -> list[str]:
    """Return errors for one annotation anchor against a loaded trace."""

    errors = _frame_range_validation_errors(
        anchor,
        index=index,
        frame_steps=frame_steps,
    )
    errors.extend(_event_id_validation_errors(anchor, index=index, event_ids=event_ids))
    errors.extend(
        _entity_anchor_validation_errors(
            anchor,
            index=index,
            pedestrian_ids=pedestrian_ids,
        )
    )
    return errors


def _frame_range_validation_errors(
    anchor: Mapping[str, Any],
    *,
    index: int,
    frame_steps: set[int],
) -> list[str]:
    """Return errors for one annotation frame range."""

    errors: list[str] = []
    min_step = min(frame_steps)
    max_step = max(frame_steps)
    frame_start = anchor.get("frame_start")
    frame_end = anchor.get("frame_end")
    if isinstance(frame_start, int) and isinstance(frame_end, int):
        if frame_start > frame_end:
            errors.append(f"/annotations/{index}/anchor/frame_start: expected <= frame_end")
        if frame_start < min_step or frame_end > max_step:
            errors.append(
                f"/annotations/{index}/anchor: frame range {frame_start}-{frame_end} "
                f"outside referenced trace steps {min_step}-{max_step}"
            )
        if frame_start not in frame_steps:
            errors.append(
                f"/annotations/{index}/anchor/frame_start: "
                f"unknown referenced trace step {frame_start}"
            )
        if frame_end not in frame_steps:
            errors.append(
                f"/annotations/{index}/anchor/frame_end: unknown referenced trace step {frame_end}"
            )
    return errors


def _event_id_validation_errors(
    anchor: Mapping[str, Any],
    *,
    index: int,
    event_ids: set[str],
) -> list[str]:
    """Return errors for event IDs referenced by one annotation anchor."""

    errors: list[str] = []
    event_id_values = anchor.get("event_ids", [])
    if isinstance(event_id_values, list):
        for event_index, event_id in enumerate(event_id_values):
            if not isinstance(event_id, str) or event_id in event_ids:
                continue
            errors.append(
                f"/annotations/{index}/anchor/event_ids/{event_index}: "
                f"unknown referenced trace event_id {event_id!r}"
            )
    return errors


def _entity_anchor_validation_errors(
    anchor: Mapping[str, Any],
    *,
    index: int,
    pedestrian_ids: set[str],
) -> list[str]:
    """Return errors for entities referenced by one annotation anchor."""

    errors: list[str] = []
    entity_values = anchor.get("entities", [])
    if isinstance(entity_values, list):
        for entity_index, entity in enumerate(entity_values):
            if isinstance(entity, Mapping):
                errors.extend(
                    _entity_validation_errors(
                        entity,
                        index=index,
                        entity_index=entity_index,
                        pedestrian_ids=pedestrian_ids,
                    )
                )
    return errors


def _entity_validation_errors(
    entity: Mapping[str, Any],
    *,
    index: int,
    entity_index: int,
    pedestrian_ids: set[str],
) -> list[str]:
    """Return errors for one entity reference."""

    entity_type = entity.get("type")
    entity_id = entity.get("id")
    if entity_type == "robot" and entity_id != "robot":
        return [
            f"/annotations/{index}/anchor/entities/{entity_index}/id: "
            "robot entity id must be 'robot'"
        ]
    if entity_type == "pedestrian":
        if not isinstance(entity_id, str) or entity_id in pedestrian_ids:
            return []
        return [
            f"/annotations/{index}/anchor/entities/{entity_index}/id: "
            f"unknown pedestrian id {entity_id!r}"
        ]
    return []


def _annotation_set_from_payload(payload: Mapping[str, Any]) -> TraceAnnotationSet:
    """Build a typed annotation set from a schema-valid payload.

    Returns:
        Typed trace annotation set.
    """

    timeline = payload["timeline"]
    provenance = payload["provenance"]
    return TraceAnnotationSet(
        schema_version=str(payload["schema_version"]),
        annotation_set_id=str(payload["annotation_set_id"]),
        timeline=TraceAnnotationTimelineRef(
            path=str(timeline["path"]),
            schema_version=str(timeline["schema_version"]),
            trace_id=str(timeline["trace_id"]),
        ),
        provenance=TraceAnnotationProvenance(
            kind=str(provenance["kind"]),
            source_issue=str(provenance["source_issue"]),
            author=str(provenance["author"]),
            created_at=str(provenance["created_at"]),
            evidence_boundary=str(provenance["evidence_boundary"]),
        ),
        annotations=[
            TraceAnnotation(
                annotation_id=str(annotation["annotation_id"]),
                category=str(annotation["category"]),
                evidence_type=str(annotation["evidence_type"]),
                anchor=TraceAnnotationAnchor(
                    frame_start=int(annotation["anchor"]["frame_start"]),
                    frame_end=int(annotation["anchor"]["frame_end"]),
                    event_ids=[
                        str(event_id) for event_id in annotation["anchor"].get("event_ids", [])
                    ],
                    entities=[
                        TraceAnnotationEntityRef(
                            type=str(entity["type"]),
                            id=str(entity["id"]),
                        )
                        for entity in annotation["anchor"].get("entities", [])
                    ],
                ),
                summary=str(annotation["summary"]),
                details=(str(annotation["details"]) if "details" in annotation else None),
            )
            for annotation in payload["annotations"]
        ],
    )
