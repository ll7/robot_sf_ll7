"""SREV-03 video-sync component: map capture frames to simulation time.

This module owns the SREV-03 leaf surface only: a ``run(request)`` adapter plus a
standalone CLI that consumes the SREV-01 shared contracts and builds an explicit
frame-to-simulation presentation map from fixture-provided capture metadata. It
never advances or mutates simulation state, never guesses frame alignment, and
never generates media: all timestamps come from explicit inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
)

COMPONENT_ID = "srev03-video-sync"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = (
    "capture-frames",
    "sim-stamps",
)
OPTIONAL_CAPABILITIES = ("camera-calibration",)

OUTPUT_MAPPING_FILENAME = "media-mapping.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("media-mapping.v1", "missing-capability-report.v1"),
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)

# Presentation preset for tests (mirrors the SREV-01 test preset shape).
_TEST_PRESET = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}


@dataclass
class _ImportOutcome:
    """Per-source load result before mapping."""

    payload: dict[str, Any] | None = None
    file_sha256: str = ""
    diagnostics: list[str] = field(default_factory=list)


def descriptor() -> dict[str, Any]:
    """Return this component's self-contained capability descriptor."""
    return asdict(_DESCRIPTOR)


def _sha256_bytes(payload: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Atomically write strict-JSON.

    Returns:
        Hex digest of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return _sha256_bytes(text.encode("utf-8"))


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a failure reason when the request demands a newer component.

    Returns:
        Failure reason string, or None when the component version satisfies it.
    """
    minimum = config.get("min_component_version")
    if minimum is None:
        return None
    try:
        wanted = int(str(minimum).split(".", maxsplit=1)[0])
        ours = int(COMPONENT_VERSION.split(".", maxsplit=1)[0])
    except ValueError:
        return f"incompatible_component_version: malformed min_component_version: {minimum!r}"
    if wanted > ours:
        return f"incompatible_component_version: request needs v{wanted}, component is v{ours}"
    return None


def _reject_not_applicable(request: ComponentRequest) -> ComponentResult | None:
    """Reject requests this component cannot serve.

    Returns:
        An unavailable/failed result, or None when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {request.component_id}",
        )
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(sorted(missing))}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=version_error,
        )
    return None


def _resolve_source(path_value: str, base: Path) -> Path:
    """Resolve a source URI under the base directory.

    Returns:
        Resolved path; absolute URIs and traversal are rejected.
    """
    candidate = Path(path_value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReviewContractsValidationError(
            [f"source uri rejected (absolute or traversal): {path_value}"]
        )
    return base / candidate


def _load_document(artifact_id: str, uri: str, root: Path) -> _ImportOutcome:
    """Read and parse one JSON source without modifying it.

    Returns:
        Import outcome with the parsed payload or a diagnostic.
    """
    try:
        raw = _resolve_source(uri, root).read_bytes()
    except OSError:
        return _ImportOutcome(diagnostics=[f"{artifact_id}: source_unreadable"])
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _ImportOutcome(diagnostics=[f"{artifact_id}: source_not_json"])
    if not isinstance(payload, dict):
        return _ImportOutcome(diagnostics=[f"{artifact_id}: source_not_json_object"])
    return _ImportOutcome(payload=payload, file_sha256=_sha256_bytes(raw))


def _finite_number(value: Any) -> float | None:
    """Return a finite float, or None for malformed input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _parse_frame_rows(raw_frames: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse and validate capture frame rows.

    Returns:
        Tuple of (valid frames, diagnostic codes).
    """
    frames: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    if not isinstance(raw_frames, list) or not raw_frames:
        return [], ["capture_frames_missing_or_empty"]
    for index, item in enumerate(raw_frames):
        if not isinstance(item, dict):
            diagnostics.append(f"frame_row_{index}_malformed")
            continue
        frame_index = item.get("frame_index")
        pts = _finite_number(item.get("pts_s"))
        if not isinstance(frame_index, int) or pts is None:
            diagnostics.append(f"frame_row_{index}_malformed")
            continue
        frames.append({"frame_index": frame_index, "pts_s": pts})
    frames.sort(key=lambda row: row["frame_index"])
    return frames, diagnostics


def _parse_step_rows(raw_steps: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse and validate simulation stamp rows.

    Returns:
        Tuple of (valid steps sorted by time, diagnostic codes).
    """
    steps: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    if not isinstance(raw_steps, list) or not raw_steps:
        return [], ["sim_stamps_missing_or_empty"]
    for index, item in enumerate(raw_steps):
        if not isinstance(item, dict):
            diagnostics.append(f"step_row_{index}_malformed")
            continue
        step = item.get("step")
        when = _finite_number(item.get("time_s"))
        if not isinstance(step, int) or when is None:
            diagnostics.append(f"step_row_{index}_malformed")
            continue
        steps.append(
            {
                "step": step,
                "time_s": when,
                "episode_id": item.get("episode_id", "unknown"),
                "reset_id": item.get("reset_id", "unknown"),
            }
        )
    steps.sort(key=lambda row: row["time_s"])
    return steps, diagnostics


def _detect_sampling_gaps(
    frames: list[dict[str, Any]], fps_nominal: float
) -> tuple[list[int], list[str]]:
    """Detect skipped indexes and nonuniform presentation deltas.

    Returns:
        Tuple of (skipped frame indexes, diagnostic codes).
    """
    seen = [row["frame_index"] for row in frames]
    skipped = [i for i in range(seen[0], seen[-1] + 1) if i not in set(seen)]
    diagnostics = []
    if skipped:
        diagnostics.append(f"skipped_frames:{','.join(str(i) for i in skipped)}")
    expected_delta = 1.0 / fps_nominal
    for first, second in itertools.pairwise(frames):
        if abs(second["pts_s"] - first["pts_s"] - expected_delta) > 0.5 * expected_delta:
            diagnostics.append(
                f"nonuniform_sampling:frames_{first['frame_index']}_{second['frame_index']}"
            )
            break
    return skipped, diagnostics


def _map_frames(
    frames: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    time_origin: float,
    speed: float,
    camera: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Anchor each frame to the latest stamp at or before its sim time.

    Returns:
        Tuple of (mapping entries, diagnostic codes).
    """
    diagnostics: list[str] = []
    first_time = time_origin + frames[0]["pts_s"] / speed
    last_time = time_origin + frames[-1]["pts_s"] / speed
    if first_time < steps[0]["time_s"] or last_time > steps[-1]["time_s"]:
        diagnostics.append("frames_outside_stamp_range")
    mapping: list[dict[str, Any]] = []
    for frame in frames:
        sim_time = time_origin + frame["pts_s"] / speed
        earlier = [step for step in steps if step["time_s"] <= sim_time]
        if not earlier:
            diagnostics.append(f"frame_{frame['frame_index']}_before_first_stamp")
            continue
        anchor = earlier[-1]
        mapping.append(
            {
                "frame_index": frame["frame_index"],
                "pts_s": frame["pts_s"],
                "sim_time_s": sim_time,
                "sim_step": anchor["step"],
                "episode_id": anchor["episode_id"],
                "reset_id": anchor["reset_id"],
                "camera": camera,
            }
        )
    return mapping, diagnostics


def _build_mapping(
    frames_doc: dict[str, Any],
    stamps_doc: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Build the explicit frame-to-simulation presentation map.

    Returns:
        Tuple of (mapping document, diagnostic codes). Skipped frames,
        nonuniform sampling, and reset boundaries are recorded explicitly;
        alignment is never guessed.
    """
    diagnostics: list[str] = []
    raw_frames = frames_doc.get("frames")
    raw_steps = stamps_doc.get("steps")
    fps_nominal = _finite_number(frames_doc.get("fps_nominal"))
    if fps_nominal is None or fps_nominal <= 0:
        return {}, ["fps_nominal_missing_or_invalid"]
    presentation = config.get("presentation", {})
    if not isinstance(presentation, dict):
        presentation = {}
    speed = _finite_number(presentation.get("speed", 1.0)) or 1.0
    if speed <= 0:
        return {}, ["presentation_speed_invalid"]
    time_origin = _finite_number(config.get("time_origin_s", 0.0))
    if time_origin is None:
        return {}, ["time_origin_invalid"]
    frames, diagnostics = _parse_frame_rows(raw_frames)
    steps, step_diagnostics = _parse_step_rows(raw_steps)
    diagnostics.extend(step_diagnostics)
    if not frames or not steps:
        return {}, diagnostics or ["no_mappable_rows"]
    skipped, gap_diagnostics = _detect_sampling_gaps(frames, fps_nominal)
    diagnostics.extend(gap_diagnostics)
    camera = frames_doc.get("camera", {})
    if not isinstance(camera, dict):
        camera = {}
    mapping, map_diagnostics = _map_frames(frames, steps, time_origin, speed, camera)
    diagnostics.extend(map_diagnostics)
    resets = sorted({row["reset_id"] for row in mapping})
    document = {
        "schema_version": "media-mapping.v1",
        "fps_nominal": fps_nominal,
        "speed": speed,
        "time_origin_s": time_origin,
        "skipped_frame_indexes": skipped,
        "reset_ids": resets,
        "first_frame": mapping[0] if mapping else None,
        "last_frame": mapping[-1] if mapping else None,
        "entries": mapping,
    }
    return document, sorted(set(diagnostics))


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Map capture frames to simulation time with an explicit presentation map.

    Args:
        request: Validated component request with capture-frames and sim-stamps sources.
        base: Base directory source URIs and the output directory resolve under.

    Returns:
        Component result: ``complete`` only when every frame maps cleanly,
        ``partial`` on skipped/out-of-range/malformed rows, ``unavailable`` when
        the component or a required capability does not apply, ``failed`` on
        validation, version, collision, or internal errors.
    """
    root = base if base is not None else Path.cwd()
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    output_dir = root / request.output_directory
    if output_dir.exists():
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason=f"output_collision: already exists: {request.output_directory}",
        )
    by_format: dict[str, _ImportOutcome] = {}
    diagnostics: list[str] = []
    try:
        for ref in request.sources:
            if ref.format not in REQUIRED_CAPABILITIES + OPTIONAL_CAPABILITIES:
                diagnostics.append(f"{ref.artifact_id}: unknown_source_format:{ref.format}")
                continue
            if (
                ref.format in OPTIONAL_CAPABILITIES
                and ref.format not in request.required_capabilities
            ):
                diagnostics.append(f"{ref.artifact_id}: optional_stream_skipped:{ref.format}")
                continue
            outcome = _load_document(ref.artifact_id, ref.uri, root)
            diagnostics.extend(outcome.diagnostics)
            if outcome.payload is not None and ref.format not in by_format:
                by_format[ref.format] = outcome
        if "capture-frames" not in by_format or "sim-stamps" not in by_format:
            diagnostics.append("required_source_family_missing")
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="required_source_family_missing: " + "; ".join(sorted(set(diagnostics))[:5]),
            )
        frames_doc = by_format["capture-frames"].payload or {}
        stamps_doc = by_format["sim-stamps"].payload or {}
        document, mapping_diagnostics = _build_mapping(frames_doc, stamps_doc, request.config)
        diagnostics.extend(mapping_diagnostics)
        if not document.get("entries"):
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status=STATUS_FAILED,
                reason="no_entries_mapped: " + "; ".join(sorted(set(diagnostics))[:5]),
            )
        capability_payload = {
            "missing_capabilities": [],
            "diagnostics": sorted(set(diagnostics)),
        }
        mapping_digest = _write_json(output_dir / OUTPUT_MAPPING_FILENAME, document)
        _write_json(output_dir / OUTPUT_CAPABILITY_FILENAME, capability_payload)
        informational = {"optional_stream_skipped", "nonuniform_sampling"}
        blocking = [item for item in diagnostics if not any(tag in item for tag in informational)]
        partial = bool(blocking)
        status = STATUS_PARTIAL if partial else STATUS_COMPLETE
        reason = "" if status == STATUS_COMPLETE else "; ".join(sorted(set(diagnostics))[:8])
        artifacts: tuple[dict[str, Any], ...] = (
            (
                {
                    "artifact_id": OUTPUT_MAPPING_FILENAME,
                    "uri": str(Path(request.output_directory) / OUTPUT_MAPPING_FILENAME),
                    "sha256": mapping_digest,
                },
            )
            if status == STATUS_COMPLETE
            else ()
        )
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=status,
            artifacts=artifacts,
            diagnostics=tuple({"code": item} for item in sorted(set(diagnostics))),
            provenance={
                "output_directory": request.output_directory,
                "entries": len(document["entries"]),
            },
            reason=reason,
        )
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status=STATUS_FAILED,
            reason="; ".join(error.errors),
        )


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the video-sync component."""
    parser = argparse.ArgumentParser(description="Map capture frames to simulation time.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the video-sync component.

    Returns:
        Process exit code (0 only when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
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
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
