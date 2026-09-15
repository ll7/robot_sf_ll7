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
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
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
    pedestrians = []
    if isinstance(frame.pedestrians, list):
        for pedestrian in frame.pedestrians:
            if not isinstance(pedestrian, dict):
                continue
            point = _xy(pedestrian.get("position"))
            if point is None:
                continue
            pedestrians.append({"id": str(pedestrian.get("id", "")), "xy": point})
    planner = frame.planner if isinstance(frame.planner, dict) else {}
    return {
        "episode_id": episode_id,
        "step": frame.step,
        "time_s": frame.time_s,
        "robot_xy": robot_xy,
        "pedestrians": pedestrians,
        "event_id": planner.get("event_id"),
    }


def _build_timeline(trace: SimulationTraceExport, artifact_id: str) -> dict[str, Any]:
    """Build the deterministic offline inspection timeline for one trace.

    Returns:
        Timeline document with per-frame geometry and counts.
    """
    episode_id = trace.source.episode_id if trace.source is not None else trace.trace_id
    frames = [_timeline_frame(frame, episode_id) for frame in trace.frames]
    return {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "artifact_id": artifact_id,
        "trace_id": trace.trace_id,
        "coordinate_frame": trace.coordinate_frame,
        "units": dict(trace.units),
        "frames": frames,
        "counts": {
            "frames": len(frames),
            "pedestrian_points": sum(len(frame["pedestrians"]) for frame in frames),
        },
    }


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
    return rr, None


def _write_rerun_recording(rr: Any, timeline: dict[str, Any], path: Path, episode_id: str) -> None:
    """Log timeline frames to a Rerun recording with simulation-time authority."""
    rr.init("robot_sf_review_rerun", spawn=False)
    for frame in timeline["frames"]:
        rr.set_time_seconds("time", float(frame["time_s"]))
        rr.set_time_sequence("step", int(frame["step"]))
        if frame["robot_xy"] is not None:
            rr.log(f"{episode_id}/robot", rr.Points2D([frame["robot_xy"]], radii=0.2))
        points = [item["xy"] for item in frame["pedestrians"]]
        if points:
            rr.log(f"{episode_id}/pedestrians", rr.Points2D(points, radii=0.15))
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
    timeline_digest = _write_json(staging_dir / timeline_name, timeline)
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
        "timeline_bytes": (staging_dir / timeline_name).stat().st_size,
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
            recording_path = staging_dir / recording_name
            _write_rerun_recording(
                rr, timeline, recording_path, timeline["frames"][0]["episode_id"]
            )
            measurements["recording"] = {
                "mode": mode,
                "format": "rerun",
                "bytes": recording_path.stat().st_size,
            }
            artifacts.append(
                {
                    "artifact_id": recording_name,
                    "uri": str(Path(output_directory) / recording_name),
                    "sha256": hashlib.sha256(recording_path.read_bytes()).hexdigest(),
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
            }
            for timeline, item in zip(timelines, measurements, strict=True)
        ],
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "component": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "request_id": request.request_id,
        "prototype": True,
        "mode": mode,
        "timelines": logical["traces"],
        "measurements": measurements,
        "report_sha256": _canonical_sha256(logical),
        "provenance": {
            "source_artifact_ids": sorted(timeline["artifact_id"] for timeline in timelines),
            "note": "Source identities are copied from the request, not verified "
            "against source bytes. Wall-time measurements are environment-specific.",
        },
    }


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
        timelines.append(_build_timeline(trace, ref.artifact_id))
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
    gate = _availability_gate(request)
    if gate is not None:
        return gate
    if not isinstance(request.config, dict):
        return _failed(request, "corrupt-config: request config must be a mapping")
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
        report = _build_report(request, mode, timelines, measurements)
        report_digest = _write_json(staging_dir / REPORT_FILENAME, report)
        descriptor_digest = _write_json(staging_dir / DESCRIPTOR_FILENAME, _DESCRIPTOR_DOC)
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
            },
        )
    except (OSError, ValueError) as error:
        return _failed(request, f"output-write-failed: {error}")
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir)


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
        if isinstance(config, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **config}}
    payload = {**payload, "output_directory": args.output}
    request = component_request_from_dict(payload, source=args.input)
    result = run(
        request,
        base=Path(args.base) if args.base is not None else None,
        source_base=input_path.parent,
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
