"""Synchronized inspection recordings over review contracts (SREV-19, issue #9289).

This module is the review-rerun contract consumer: it turns explicitly
selected simulation trace exports into an offline inspection timeline plus a
measured prototype report, and additionally writes a Rerun recording when the
optional ``rerun-sdk`` package is installed. It reuses the SREV-01
shared-contract surface
(:mod:`robot_sf.analysis_workbench.review_contracts`) for envelopes,
validation, and digests, and the canonical trace owner
(:mod:`robot_sf.analysis_workbench.simulation_trace_export`) to validate every
consumed trace. Core inspection always works offline without Rerun: a missing
SDK only marks the optional recording stream unavailable, never blocks the
supported timeline and report outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    SourceRef,
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.analysis_workbench.simulation_timeline import (
    build_simulation_timeline,
    validate_simulation_timeline,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
)

COMPONENT_ID = "srev19-review-rerun"
COMPONENT_VERSION = "1.0.0"

FORMAT_TRACE_EXPORT = "simulation_trace_export.v1"

TIMELINE_SCHEMA_VERSION = "review-rerun-timeline.v1"
REPORT_SCHEMA_VERSION = "review-rerun-report.v1"

TIMELINE_SCHEMA_FILE = (
    Path(__file__).resolve().parents[1]
    / "analysis_workbench"
    / "schemas"
    / "review_rerun_timeline.v1.json"
)
REPORT_SCHEMA_FILE = (
    Path(__file__).resolve().parents[1]
    / "analysis_workbench"
    / "schemas"
    / "review_rerun_report.v1.json"
)

REPORT_FILENAME = "prototype-report.json"
DESCRIPTOR_FILENAME = "component-descriptor.json"

_DESCRIBE_OUTPUT_TYPES = ("review-rerun.v1",)

_DESCRIPTOR_DOC: dict[str, Any] = {
    "schema_version": "component-descriptor.v1",
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "required_capabilities": [],
    "optional_capabilities": [],
    "output_types": list(_DESCRIBE_OUTPUT_TYPES),
}

# Fail fast on descriptor drift: the shipped descriptor must stay schema-valid.
DESCRIPTOR = component_descriptor_from_dict(_DESCRIPTOR_DOC)

RECORDING_MODES = ("auto", "json", "rerun")
_SAFE_ENTITY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class RerunRecordingError(RuntimeError):
    """Raised when the optional Rerun stream cannot satisfy its contract."""


def descriptor() -> dict[str, Any]:
    """Return the versioned capability descriptor for this component.

    Returns:
        Descriptor document declaring exact required/optional capabilities
        (none) and versioned result artifact types.
    """
    return json.loads(json.dumps(_DESCRIPTOR_DOC))


def _canonical_sha256(value: Any) -> str:
    """Hash logical content deterministically (location-independent).

    Returns:
        Hex SHA-256 digest of the canonical encoding.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validate_json_schema(payload: dict[str, Any], schema_path: Path) -> None:
    """Validate one published JSON artifact against its checked-in schema."""

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = [
        "/" + "/".join(str(token) for token in error.absolute_path) + f": {error.message}"
        for error in sorted(
            validator.iter_errors(payload), key=lambda item: list(item.absolute_path)
        )
    ]
    if errors:
        raise ValueError(f"{schema_path.name}: {'; '.join(errors)}")


def _safe_staging_path(staging_dir: Path, filename: str) -> Path:
    """Resolve one output filename and enforce containment below staging.

    Returns:
        Resolved path below ``staging_dir``.
    """

    pure = PureWindowsPath(filename)
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or pure.is_absolute()
        or bool(pure.drive)
        or ".." in Path(filename).parts
    ):
        raise ValueError(f"unsafe-output-artifact: {filename!r}")
    root = staging_dir.resolve(strict=False)
    candidate = (root / filename).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"unsafe-output-artifact: {filename!r}") from error
    return candidate


def _request_integrity_errors(request: ComponentRequest) -> list[str]:
    """Return direct-request safety violations for callers bypassing the parser."""

    errors: list[str] = []
    seen: set[str] = set()
    for index, ref in enumerate(request.sources):
        artifact_id = ref.artifact_id
        pure = PureWindowsPath(artifact_id)
        if (
            not artifact_id
            or "/" in artifact_id
            or "\\" in artifact_id
            or pure.is_absolute()
            or bool(pure.drive)
            or ".." in Path(artifact_id).parts
            or any(ord(character) < 32 or ord(character) == 127 for character in artifact_id)
        ):
            errors.append(f"/sources/{index}/artifact_id: unsafe artifact id: {artifact_id!r}")
        if artifact_id in seen:
            errors.append(f"/sources: duplicate scoped artifact id: {artifact_id}")
        seen.add(artifact_id)
    return errors


def _result_payload(result: ComponentResult) -> dict[str, Any]:
    """Serialize a result with the version field required by its envelope schema.

    Returns:
        JSON-safe, schema-validated component-result payload.
    """

    payload = json.loads(
        json.dumps({"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)})
    )
    component_result_from_dict(payload)
    return payload


def _write_json(path: Path, payload: Any) -> str:
    """Write JSON atomically and return the hex SHA-256 of the file bytes.

    Returns:
        Hex SHA-256 digest of the written file bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    """List requested capabilities this component does not provide.

    Returns:
        Requested capability names absent from the descriptor.
    """
    supported = set(DESCRIPTOR.required_capabilities) | set(DESCRIPTOR.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _major(version: str) -> str | None:
    """Return the ``v<major>`` marker of a dotted version string, if parseable."""
    head = version.strip().split(".", 1)[0].lstrip("vV")
    return f"v{head}" if head.isdigit() else None


def _recording_mode(config: dict[str, Any]) -> tuple[str | None, str | None]:
    """Validate the requested recording mode.

    Returns:
        Tuple of (mode, error); exactly one side is meaningful.
    """
    recording = config.get("recording", {"mode": "auto"})
    if not isinstance(recording, dict):
        return None, "corrupt-recording: config must carry a 'recording' mapping"
    mode = recording.get("mode", "auto")
    if mode not in RECORDING_MODES:
        return None, f"corrupt-recording: 'mode' must be one of {RECORDING_MODES}"
    return str(mode), None


def _xy(position: Any) -> list[float] | None:
    """Extract a finite 2D point from a trace position payload.

    Returns:
        The ``[x, y]`` point, or ``None`` when the payload is not finite 2D.
    """
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    x, y = position[0], position[1]
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return [float(x), float(y)]


def _timeline_frame(frame: Any, episode_id: str) -> dict[str, Any]:
    """Map one validated trace frame onto an inspection-timeline frame.

    Returns:
        Frame mapping with step, time, geometry, and event identity.
    """
    robot_xy = _xy(frame.robot.get("position")) if isinstance(frame.robot, dict) else None
    pedestrians: list[dict[str, Any]] = []
    if isinstance(frame.pedestrians, list):
        for pedestrian in frame.pedestrians:
            if not isinstance(pedestrian, dict):
                continue
            point = _xy(pedestrian.get("position"))
            if point is None:
                continue
            pedestrians.append(
                {
                    "id": str(pedestrian["id"]),
                    "xy": point,
                    "state": dict(pedestrian),
                }
            )
    planner = frame.planner if isinstance(frame.planner, dict) else {}
    selected_action = planner.get("selected_action")
    return {
        "episode_id": episode_id,
        "step": frame.step,
        "time_s": frame.time_s,
        "robot_xy": robot_xy,
        "pedestrians": pedestrians,
        "event_id": planner.get("event_id"),
        "robot": dict(frame.robot),
        "planner": dict(planner),
        "selected_action": (
            dict(selected_action) if isinstance(selected_action, dict) else selected_action
        ),
    }


def _build_timeline(
    trace: SimulationTraceExport,
    artifact_id: str,
    *,
    source_ref: SourceRef,
    source_sha256: str,
    config_sha256: str,
    canonical_timeline: dict[str, Any],
) -> dict[str, Any]:
    """Build the deterministic offline inspection timeline for one trace.

    Returns:
        Timeline document with per-frame geometry and counts.
    """
    episode_id = trace.source.episode_id if trace.source is not None else trace.trace_id
    frames = [_timeline_frame(frame, episode_id) for frame in trace.frames]
    source_trace = {
        "schema_version": trace.schema_version,
        "trace_id": trace.trace_id,
        "source": asdict(trace.source),
    }
    source = {
        "artifact_id": artifact_id,
        "uri": source_ref.uri,
        "format": source_ref.format,
        "sha256": source_sha256,
        "declared_sha256": source_ref.sha256 or None,
        "source_commit": source_ref.source_commit or None,
        "config_identity": source_ref.config_identity or None,
    }
    timeline = {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "artifact_id": artifact_id,
        "trace_id": trace.trace_id,
        "source": source,
        "source_trace": source_trace,
        "canonical_timeline": canonical_timeline,
        "evidence_boundary": trace.evidence_boundary,
        "diagnostic_only": True,
        "admission": "not_evaluated",
        "coordinate_frame": trace.coordinate_frame,
        "units": dict(trace.units),
        "frames": frames,
        "events": list(canonical_timeline["events"]),
        "counts": {
            "frames": len(frames),
            "pedestrian_points": sum(len(frame["pedestrians"]) for frame in frames),
        },
        "provenance": {
            "source_sha256": source_sha256,
            "source_identity": source_trace["source"],
            "source_identity_sha256": _canonical_sha256(source_trace),
            "source_commit": source_ref.source_commit or None,
            "source_commit_sha256": (
                _canonical_sha256(source_ref.source_commit) if source_ref.source_commit else None
            ),
            "config_identity": source_ref.config_identity or None,
            "config_sha256": config_sha256,
            "claim_boundary": "diagnostic_only; not benchmark evidence",
        },
    }
    validate_simulation_timeline(canonical_timeline)
    _validate_json_schema(timeline, TIMELINE_SCHEMA_FILE)
    return timeline


def _load_trace_source(
    artifact_id: str, path: Path
) -> tuple[SimulationTraceExport | None, str | None]:
    """Validate one trace export through its canonical owner.

    Returns:
        Tuple of (trace, error); exactly one side is meaningful.
    """
    try:
        return load_simulation_trace_export(path), None
    except (SimulationTraceExportValidationError, OSError, ValueError) as error:
        return None, f"corrupt-source: {artifact_id}: {error}"


def _try_import_rerun() -> tuple[Any | None, str | None]:
    """Import the optional Rerun SDK without installing anything.

    Returns:
        Tuple of (module, error); exactly one side is meaningful.
    """
    try:
        # Deferred optional import: a top-level import would make this module
        # unimportable without the optional rerun-sdk package.
        import rerun as rr  # noqa: PLC0415
    except ImportError as error:
        return None, f"rerun-sdk-missing: {error}"
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        # Import hooks and optional SDK module initialization are environment-specific.
        return None, f"rerun-sdk-import-failed: {type(error).__name__}: {error}"
    return rr, None


def _entity_component(value: str, *, prefix: str) -> str:
    """Return a stable, path-safe Rerun entity component for untrusted IDs."""

    if _SAFE_ENTITY_RE.fullmatch(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _pedestrian_entity(episode_id: str, actor_id: str) -> str:
    """Build a stable per-actor entity path without using actor text as a path.

    Returns:
        Stable Rerun entity path.
    """

    episode = _entity_component(episode_id, prefix="episode")
    actor = _entity_component(actor_id, prefix="actor")
    return f"{episode}/pedestrians/{actor}"


def _rerun_any_values(rr: Any, values: dict[str, Any]) -> Any:
    """Build the current Rerun arbitrary-value metadata archetype.

    Returns:
        SDK metadata archetype instance.
    """

    constructor = getattr(rr, "AnyValues", None)
    if not callable(constructor):
        raise RerunRecordingError("rerun SDK lacks AnyValues metadata support")
    return constructor(**values)


def _log_rerun_metadata(rr: Any, path: str, values: dict[str, Any]) -> None:
    """Log JSON-safe frame/source metadata alongside geometry."""

    rr.log(path, _rerun_any_values(rr, values))


def _clear_rerun_entity(rr: Any, path: str) -> None:
    """Clear one disappeared actor, using the SDK's explicit clear archetype."""

    constructor = getattr(rr, "Clear", None)
    if not callable(constructor):
        raise RerunRecordingError("rerun SDK lacks Clear support for empty actor frames")
    rr.log(path, constructor(recursive=True))


def _write_rerun_recording(rr: Any, timeline: dict[str, Any], path: Path, episode_id: str) -> None:
    """Log source-faithful timeline frames with simulation-time authority."""

    rr.init("robot_sf_review_rerun", spawn=False)
    source_path = f"{_entity_component(episode_id, prefix='episode')}/source"
    source_trace = timeline["source_trace"]
    _log_rerun_metadata(
        rr,
        source_path,
        {
            "artifact_id": timeline["artifact_id"],
            "source_trace_json": json.dumps(source_trace, sort_keys=True),
            "source_sha256": timeline["source"]["sha256"],
            "evidence_boundary": timeline["evidence_boundary"],
            "diagnostic_only": timeline["diagnostic_only"],
            "coordinate_frame": timeline["coordinate_frame"],
            "units_json": json.dumps(timeline["units"], sort_keys=True),
        },
    )
    previous_entities: dict[str, str] = {}
    for frame in timeline["frames"]:
        rr.set_time_seconds("time", float(frame["time_s"]))
        rr.set_time_sequence("step", int(frame["step"]))
        if frame["robot_xy"] is not None:
            rr.log(
                f"{_entity_component(episode_id, prefix='episode')}/robot",
                rr.Points2D([frame["robot_xy"]], radii=0.2),
            )
        current_entities: dict[str, str] = {}
        for pedestrian in frame["pedestrians"]:
            actor_id = str(pedestrian["id"])
            entity = _pedestrian_entity(episode_id, actor_id)
            current_entities[actor_id] = entity
            rr.log(entity, rr.Points2D([pedestrian["xy"]], radii=0.15))
            _log_rerun_metadata(
                rr,
                f"{entity}/metadata",
                {
                    "actor_id": actor_id,
                    "state_json": json.dumps(pedestrian["state"], sort_keys=True),
                },
            )
        for actor_id, entity in previous_entities.items():
            if actor_id not in current_entities:
                _clear_rerun_entity(rr, entity)
        _log_rerun_metadata(
            rr,
            f"{_entity_component(episode_id, prefix='episode')}/frames/{frame['step']}",
            {
                "event_id": str(frame["event_id"] or ""),
                "selected_action_json": json.dumps(frame["selected_action"], sort_keys=True),
                "coordinate_frame": timeline["coordinate_frame"],
                "units_json": json.dumps(timeline["units"], sort_keys=True),
                "evidence_boundary": timeline["evidence_boundary"],
                "diagnostic_only": timeline["diagnostic_only"],
            },
        )
        previous_entities = current_entities
    rr.save(str(path))


def _record_episode(
    timeline: dict[str, Any], staging_dir: Path, mode: str, output_directory: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Write the timeline and optionally record it with the Rerun SDK.

    Files land directly in the caller's staging directory under per-trace
    names so multi-source requests never collide.

    Returns:
        Tuple of (artifacts, diagnostics, measurements) for one trace.
    """
    started = time.perf_counter()
    timeline_name = f"{timeline['artifact_id']}.inspection-timeline.json"
    timeline_path = _safe_staging_path(staging_dir, timeline_name)
    timeline_digest = _write_json(timeline_path, timeline)
    artifacts = [
        {
            "artifact_id": timeline_name,
            "uri": str(Path(output_directory) / timeline_name),
            "sha256": timeline_digest,
        }
    ]
    diagnostics: list[dict[str, Any]] = []
    measurements: dict[str, Any] = {
        "frames": timeline["counts"]["frames"],
        "pedestrian_points": timeline["counts"]["pedestrian_points"],
        "timeline_bytes": timeline_path.stat().st_size,
        "timeline_sha256": timeline_digest,
    }
    if mode == "json":
        measurements["recording"] = {"mode": "json", "format": None}
    else:
        rr, error = _try_import_rerun()
        if rr is None or error is not None:
            diagnostics.append(
                {"artifact_id": timeline["artifact_id"], "reason": error or "unknown"}
            )
            measurements["recording"] = {"mode": mode, "format": None}
        else:
            recording_name = f"{timeline['artifact_id']}.inspection-recording.rrd"
            recording_path = _safe_staging_path(staging_dir, recording_name)
            try:
                _write_rerun_recording(
                    rr, timeline, recording_path, timeline["frames"][0]["episode_id"]
                )
                recording_digest = hashlib.sha256(recording_path.read_bytes()).hexdigest()
            except Exception as error:
                raise RerunRecordingError(f"{type(error).__name__}: {error}") from error
            measurements["recording"] = {
                "mode": mode,
                "format": "rerun",
                "bytes": recording_path.stat().st_size,
                "status": "complete",
            }
            artifacts.append(
                {
                    "artifact_id": recording_name,
                    "uri": str(Path(output_directory) / recording_name),
                    "sha256": recording_digest,
                }
            )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    measurements["encode_elapsed_ms"] = elapsed_ms
    measurements["encode_elapsed_note"] = "environment-specific wall time, not a claim"
    return artifacts, diagnostics, measurements


def _build_report(
    request: ComponentRequest,
    mode: str,
    timelines: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compose the measured prototype report over all recorded traces.

    Returns:
        Report payload ready for staging and digesting.
    """
    logical = {
        "mode": mode,
        "traces": [
            {
                "artifact_id": timeline["artifact_id"],
                "trace_id": timeline["trace_id"],
                "frames": timeline["counts"]["frames"],
                "pedestrian_points": timeline["counts"]["pedestrian_points"],
                "timeline_sha256": item["timeline_sha256"],
                "source_sha256": timeline["source"]["sha256"],
            }
            for timeline, item in zip(timelines, measurements, strict=True)
        ],
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "prototype": True,
        "mode": mode,
        "evidence_boundary": "analysis_workbench_only",
        "diagnostic_only": True,
        "admission": "not_evaluated",
        "timelines": logical["traces"],
        "measurements": measurements,
        "report_sha256": _canonical_sha256(logical),
        "provenance": {
            "source_artifact_ids": sorted(timeline["artifact_id"] for timeline in timelines),
            "sources": [
                {
                    "request_source": dict(timeline["source"]),
                    "trace_identity": dict(timeline["source_trace"]["source"]),
                    "source_identity_sha256": timeline["provenance"]["source_identity_sha256"],
                    "source_commit_sha256": timeline["provenance"]["source_commit_sha256"],
                    "config_identity": timeline["provenance"]["config_identity"],
                    "config_sha256": timeline["provenance"]["config_sha256"],
                }
                for timeline in timelines
            ],
            "config_sha256": _canonical_sha256(request.config),
            "claim_boundary": "diagnostic_only; not benchmark evidence",
            "note": "Source bytes are hashed before canonical trace validation. "
            "Wall-time measurements are environment-specific.",
        },
    }
    _validate_json_schema(report, REPORT_SCHEMA_FILE)
    return report


def _resolve_output_directory(root: Path, output_directory: str) -> tuple[Path | None, str | None]:
    """Resolve an output path and reject symlink escapes from the base directory.

    Returns:
        A resolved output path and no error, or ``(None, reason)`` for an escape.
    """
    try:
        resolved_root = root.resolve(strict=False)
        output_dir = (resolved_root / output_directory).resolve(strict=False)
        output_dir.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None, "unsafe-output-path: output directory must resolve within base"
    return output_dir, None


def _unavailable(request: ComponentRequest, reason: str) -> ComponentResult:
    """Build an unavailable result without artifacts.

    Returns:
        Unavailable component result carrying only the reason.
    """
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="unavailable",
        reason=reason,
    )


def _failed(
    request: ComponentRequest, reason: str, diagnostics: tuple[dict[str, Any], ...] = ()
) -> ComponentResult:
    """Build a failed result without artifacts.

    Returns:
        Failed component result carrying the reason and diagnostics.
    """
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="failed",
        diagnostics=diagnostics,
        reason=reason,
    )


def _availability_gate(request: ComponentRequest) -> ComponentResult | None:
    """Reject unsupported components, capabilities, and versions.

    Returns:
        An unavailable result, or ``None`` when invocation may proceed.
    """
    if request.component_id != COMPONENT_ID:
        return _unavailable(request, f"unsupported component: {request.component_id}")
    missing = _unsupported_capabilities(request)
    if missing:
        return _unavailable(request, f"missing capabilities: {', '.join(sorted(missing))}")
    required_version = request.config.get("required_component_version")
    if required_version is not None and _major(str(required_version)) != _major(COMPONENT_VERSION):
        return _unavailable(
            request,
            f"incompatible-required-version: {required_version!r}; "
            f"this component implements {COMPONENT_VERSION}",
        )
    return None


def _resolve_source_file(source_base: Path, uri: str) -> Path | None:
    """Resolve a source URI under the base without allowing escapes.

    Returns:
        The resolved path, or ``None`` when it escapes the base.
    """
    try:
        resolved_base = source_base.resolve(strict=False)
        resolved = (resolved_base / uri).resolve(strict=False)
        resolved.relative_to(resolved_base)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _load_source_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read one source JSON document.

    Returns:
        Tuple of (payload, error); exactly one side is meaningful.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return None, f"unreadable-source: {error}"
    except json.JSONDecodeError as error:
        return None, f"corrupt-source: invalid JSON: {error}"
    if not isinstance(payload, dict):
        return None, "corrupt-source: top-level JSON value must be an object"
    return payload, None


def _collect_traces(
    request: ComponentRequest, sources_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], ComponentResult | None]:
    """Load and validate every trace source, diagnosing skipped ones.

    Returns:
        Tuple of (timelines, diagnostics, terminal result). The terminal
        result is set for missing or wholly unusable evidence.
    """
    timelines: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for ref in request.sources:
        source_file = _resolve_source_file(sources_root, ref.uri)
        if source_file is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": "unsafe-source-uri"})
            continue
        payload, load_error = _load_source_payload(source_file)
        if load_error is not None or payload is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": load_error})
            continue
        if ref.format != FORMAT_TRACE_EXPORT:
            diagnostics.append(
                {
                    "artifact_id": ref.artifact_id,
                    "reason": f"unsupported-evidence-format: {ref.format}",
                }
            )
            continue
        trace, error = _load_trace_source(ref.artifact_id, source_file)
        if error is not None or trace is None:
            diagnostics.append({"artifact_id": ref.artifact_id, "reason": error})
            continue
        try:
            source_sha256 = hashlib.sha256(source_file.read_bytes()).hexdigest()
            if ref.sha256 and source_sha256.lower() != ref.sha256.lower():
                diagnostics.append(
                    {
                        "artifact_id": ref.artifact_id,
                        "reason": "source-digest-mismatch",
                        "declared_sha256": ref.sha256,
                        "observed_sha256": source_sha256,
                    }
                )
                continue
            canonical_timeline = build_simulation_timeline(source_file)
        except (OSError, ValueError) as error:
            diagnostics.append(
                {"artifact_id": ref.artifact_id, "reason": f"corrupt-source: {error}"}
            )
            continue
        timelines.append(
            _build_timeline(
                trace,
                ref.artifact_id,
                source_ref=ref,
                source_sha256=source_sha256,
                config_sha256=_canonical_sha256(request.config),
                canonical_timeline=canonical_timeline,
            )
        )
    if not request.sources:
        return (
            timelines,
            diagnostics,
            _failed(request, "missing-evidence: request carries no sources"),
        )
    if not timelines:
        return (
            timelines,
            diagnostics,
            ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="unavailable",
                diagnostics=tuple(diagnostics),
                reason="unsupported-evidence: no source could be recorded",
            ),
        )
    return timelines, diagnostics, None


def run(
    request: ComponentRequest, *, base: Path | None = None, source_base: Path | None = None
) -> ComponentResult:
    """Record one synchronized inspection over validated trace exports.

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.
        source_base: Base directory source URIs resolve under (CLI: the request
            file's directory). Defaults to the current working directory.

    Returns:
        Component result: ``complete`` with timeline/report/recording and
        descriptor artifacts, ``unavailable`` for unsupported components,
        capabilities, versions, evidence, or recording modes, or ``failed``
        for corrupt inputs and output collisions.
    """
    root = base if base is not None else Path.cwd()
    sources_root = source_base if source_base is not None else Path.cwd()
    integrity_errors = _request_integrity_errors(request)
    if integrity_errors:
        return _failed(
            request,
            "invalid-source-identities",
            tuple({"reason": error} for error in integrity_errors),
        )
    if not isinstance(request.config, dict):
        return _failed(request, "corrupt-config: request config must be a mapping")
    gate = _availability_gate(request)
    if gate is not None:
        return gate
    mode, mode_error = _recording_mode(request.config)
    if mode_error is not None or mode is None:
        return _failed(request, mode_error or "corrupt-recording")
    if mode == "rerun":
        _rr, rr_error = _try_import_rerun()
        if _rr is None:
            return _unavailable(
                request,
                f"rerun-mode-unavailable: {rr_error or 'unknown'}; "
                "explicit rerun recordings require the optional rerun-sdk package",
            )
    timelines, diagnostics, terminal = _collect_traces(request, sources_root)
    if terminal is not None:
        return terminal

    output_dir, path_error = _resolve_output_directory(root, request.output_directory)
    if path_error is not None or output_dir is None:
        return _failed(
            request, path_error or "unsafe-output-path: output directory must resolve within base"
        )
    if output_dir.exists():
        return _failed(
            request,
            f"output-collision: output directory already exists: {request.output_directory}",
        )
    return _publish_outputs(output_dir, request, timelines, mode, diagnostics)


def _publish_outputs(
    output_dir: Path,
    request: ComponentRequest,
    timelines: list[dict[str, Any]],
    mode: str,
    diagnostics: list[dict[str, Any]],
) -> ComponentResult:
    """Record every timeline and publish the output directory atomically.

    Returns:
        Complete component result, or a failed result on write errors.
    """
    staging_dir: Path | None = None
    try:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
        artifacts: list[dict[str, Any]] = []
        measurements: list[dict[str, Any]] = []
        for timeline in timelines:
            trace_artifacts, trace_diagnostics, item = _record_episode(
                timeline, staging_dir, mode, request.output_directory
            )
            artifacts.extend(trace_artifacts)
            diagnostics.extend(trace_diagnostics)
            measurements.append(item)
        if mode == "rerun" and any(
            item.get("recording", {}).get("status") == "failed" for item in measurements
        ):
            failed_items = [
                item["recording"].get("reason", "unknown")
                for item in measurements
                if item.get("recording", {}).get("status") == "failed"
            ]
            raise RerunRecordingError("; ".join(failed_items))
        report = _build_report(request, mode, timelines, measurements)
        report_path = _safe_staging_path(staging_dir, REPORT_FILENAME)
        descriptor_path = _safe_staging_path(staging_dir, DESCRIPTOR_FILENAME)
        report_digest = _write_json(report_path, report)
        descriptor_digest = _write_json(descriptor_path, _DESCRIPTOR_DOC)
        artifacts.append(
            {
                "artifact_id": REPORT_FILENAME,
                "uri": str(Path(request.output_directory) / REPORT_FILENAME),
                "sha256": report_digest,
            }
        )
        artifacts.append(
            {
                "artifact_id": DESCRIPTOR_FILENAME,
                "uri": str(Path(request.output_directory) / DESCRIPTOR_FILENAME),
                "sha256": descriptor_digest,
            }
        )
        for artifact in artifacts:
            artifact_path = _safe_staging_path(staging_dir, artifact["artifact_id"])
            observed_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            if observed_digest != artifact["sha256"]:
                raise ValueError(
                    f"artifact-digest-mismatch: {artifact['artifact_id']}: "
                    f"expected {artifact['sha256']}, observed {observed_digest}"
                )
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {request.output_directory}")
        staging_dir.replace(output_dir)
        staging_dir = None
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="complete",
            artifacts=tuple(artifacts),
            diagnostics=tuple(diagnostics),
            provenance={
                "output_directory": request.output_directory,
                "report_sha256": report["report_sha256"],
                "component_version": COMPONENT_VERSION,
                "recording_mode": mode,
                "evidence_boundary": "analysis_workbench_only",
                "diagnostic_only": True,
                "admission": "not_evaluated",
                "config_sha256": _canonical_sha256(request.config),
            },
        )
    except RerunRecordingError as error:
        return _failed(request, f"rerun-recording-failed: {error}")
    except (OSError, ValueError) as error:
        return _failed(request, f"output-write-failed: {error}")
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the review-rerun component.

    Returns:
        Argument parser with input/config/output/base options.
    """
    parser = argparse.ArgumentParser(description="Record SREV-19 synchronized inspections.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-rerun component.

    Returns:
        Process exit code (0 when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error
    if args.config is not None:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if not isinstance(config, dict):
            raise ReviewContractsValidationError(
                ["config must be a JSON object"], source=args.config
            )
        request_config = payload.get("config", {})
        if not isinstance(request_config, dict):
            raise ReviewContractsValidationError(["request config must be a JSON object"])
        payload = {**payload, "config": {**request_config, **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(
        request,
        base=Path(args.base) if args.base is not None else None,
        source_base=input_path.parent,
    )
    print(json.dumps(_result_payload(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
