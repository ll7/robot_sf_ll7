"""SREV-10 review-encode component: reproducible video edits from frame sequences.

This module owns the SREV-10 leaf surface only: a ``run(request, base)`` adapter
plus a standalone CLI that consumes the SREV-01 shared contracts, applies a
declared edit plan (cuts, pauses, speed changes, crop) to a source frame
sequence or clip, and emits a playable MP4 with an encoder receipt and a
piecewise source-time/presentation-time map. Frame pixels are rendered
deterministically in-component; MP4 byte identity is promised only within the
declared encoder environment recorded in the receipt. Audio output is always
silent (v1); any other audio policy fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
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

COMPONENT_ID = "srev10-review-encode"
COMPONENT_VERSION = "1.0.0"

REQUIRED_CAPABILITIES = ("frame-sequence",)
OPTIONAL_CAPABILITIES = ("source-clip", "storyboard")

OUTPUT_VIDEO_FILENAME = "edit.mp4"
OUTPUT_RECEIPT_FILENAME = "encode-receipt.json"
OUTPUT_TIMEMAP_FILENAME = "time-map.json"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

SOURCE_FPS = "source_fps"
SOURCE_FRAMES = "source_frames"

ENCODER_NAME = "imageio-ffmpeg"
ENCODER_CODEC = "libx264"
AUDIO_POLICY_SILENT = "silent"

TINY_PRESET = {"width": 160, "height": 120, "fps": 10}

_Descriptor = ComponentDescriptor
_DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("review-edit-mp4.v1", "encode-receipt.v1", "presentation-time-map.v1"),
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


@dataclass
class _MapSegment:
    """One piecewise presentation/source interval with its producing operation."""

    presentation_start_s: float
    presentation_end_s: float
    source_start_s: float
    source_end_s: float
    operation: str


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


def _canonical_digest(payload: Any) -> str:
    """Return the stable logical digest of a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256_bytes(encoded.encode("utf-8"))


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


def _check_capabilities(request: ComponentRequest) -> str | None:
    """Return an unavailability reason when a required capability is unsupported.

    Returns:
        Unavailability reason string, or None when every required capability is
        either required or optional (supported) by this component.
    """
    supported = set(_DESCRIPTOR.required_capabilities) | set(_DESCRIPTOR.optional_capabilities)
    missing = [name for name in request.required_capabilities if name not in supported]
    if missing:
        return "unsupported_required_capability: " + ",".join(sorted(missing))
    return None


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    """Read and parse a JSON document.

    Returns:
        Tuple of (payload or None, problem code or None).
    """
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, f"{path.name}: source_missing"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        return None, f"{path.name}: source_unreadable:{type(error).__name__}"


def _resolve_under(path: Path, root: Path) -> Path | None:
    """Resolve a request URI against the fixture root without escaping it.

    Returns:
        Resolved path, or None when the URI escapes the root.
    """
    text = str(path)
    if Path(text).is_absolute():
        return None
    resolved = (root / text).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return resolved


def _encoder_probe() -> tuple[dict[str, Any] | None, str | None]:
    """Probe the declared MP4 encoder backend without importing simulation code.

    Returns:
        Tuple of (environment record or None, unavailability reason or None).
    """
    try:
        import imageio_ffmpeg  # noqa: PLC0415
        import numpy  # noqa: PLC0415
        import PIL  # noqa: PLC0415
    except ImportError as error:
        return None, f"missing_encoder_backend: {error.name or error}"
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as error:
        return None, f"missing_encoder_backend: ffmpeg-binary:{error}"
    import imageio  # noqa: PLC0415

    return (
        {
            "encoder": ENCODER_NAME,
            "codec": ENCODER_CODEC,
            "ffmpeg_binary": Path(exe).name,
            "imageio_version": getattr(imageio, "__version__", "unknown"),
            "pillow_version": getattr(PIL, "__version__", "unknown"),
            "numpy_version": getattr(numpy, "__version__", "unknown"),
            "platform": platform.system(),
            "audio_policy": AUDIO_POLICY_SILENT,
        },
        None,
    )


def _validate_manifest(doc: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a frame-sequence manifest document.

    Returns:
        Tuple of (manifest mapping or None, failure reason or None).
    """
    if not isinstance(doc, dict):
        return None, "manifest_invalid: document must be an object"
    fps = doc.get(SOURCE_FPS)
    frames = doc.get(SOURCE_FRAMES)
    if not isinstance(fps, (int, float)) or fps <= 0:
        return None, "manifest_invalid: source_fps must be a positive number"
    if not isinstance(frames, int) or frames < 1:
        return None, "manifest_invalid: source_frames must be a positive integer"
    return {"source_fps": float(fps), "source_frames": int(frames)}, None


def _validate_cut(item: dict[str, Any], position: int, duration_s: float) -> dict[str, Any] | str:
    """Validate one cut operation.

    Returns:
        Normalized edit, or a failure reason string.
    """
    start, end = float(item.get("start_s", -1)), float(item.get("end_s", -1))
    if not 0 <= start < end:
        return f"edit_plan_invalid: cut[{position}] needs 0 <= start < end"
    if end > duration_s:
        return f"edit_plan_invalid: cut[{position}] exceeds duration"
    return {"op": "cut", "start_s": start, "end_s": end}


def _validate_pause(item: dict[str, Any], position: int, duration_s: float) -> dict[str, Any] | str:
    """Validate one pause operation.

    Returns:
        Normalized edit, or a failure reason string.
    """
    at_s, held = float(item.get("at_s", -1)), float(item.get("duration_s", 0))
    if at_s < 0 or at_s > duration_s or held <= 0:
        return f"edit_plan_invalid: pause[{position}] out of range"
    return {"op": "pause", "at_s": at_s, "duration_s": held}


def _validate_speed(item: dict[str, Any], position: int, duration_s: float) -> dict[str, Any] | str:
    """Validate one speed operation.

    Returns:
        Normalized edit, or a failure reason string.
    """
    start = float(item.get("start_s", -1))
    end = float(item.get("end_s", -1))
    factor = float(item.get("factor", 0))
    if not 0 <= start < end <= duration_s or factor <= 0:
        return f"edit_plan_invalid: speed[{position}] out of range"
    return {"op": "speed", "start_s": start, "end_s": end, "factor": factor}


def _validate_crop(item: dict[str, Any], position: int) -> dict[str, Any] | str:
    """Validate one crop operation.

    Returns:
        Normalized edit, or a failure reason string.
    """
    box = {key: item.get(key) for key in ("x", "y", "width", "height")}
    if any(not isinstance(value, (int, float)) for value in box.values()):
        return f"edit_plan_invalid: crop[{position}] needs numbers"
    return {"op": "crop", **{k: float(v) for k, v in box.items()}}


def _validate_one(
    item: dict[str, Any], position: int, duration_s: float
) -> dict[str, Any] | str | None:
    """Dispatch one edit item to its per-operation validator.

    Returns:
        Normalized edit, a failure reason string, or None for unknown ops.
    """
    kind = str(item.get("op"))
    if kind == "cut":
        return _validate_cut(item, position, duration_s)
    if kind == "pause":
        return _validate_pause(item, position, duration_s)
    if kind == "speed":
        return _validate_speed(item, position, duration_s)
    if kind == "crop":
        return _validate_crop(item, position)
    return None


def _validate_edits(
    raw: Any, duration_s: float
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    """Validate the edit plan against the source duration.

    Returns:
        Tuple of (normalized edits, diagnostics, failure reason or None).
        Conflicting time maps fail; clamped or dropped edits diagnose partial.
    """
    diagnostics: list[str] = []
    if raw is None:
        return [], diagnostics, None
    if not isinstance(raw, list):
        return [], diagnostics, "edit_plan_invalid: edits must be a list"
    edits: list[dict[str, Any]] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            return [], diagnostics, f"edit_plan_invalid: edits[{position}] must be an object"
        checked = _validate_one(item, position, duration_s)
        if checked is None:
            return [], diagnostics, f"edit_plan_invalid: unknown op {item.get('op')!r}"
        if isinstance(checked, str):
            return [], diagnostics, checked
        edits.append(checked)
    conflict = _find_time_conflict(edits)
    if conflict is not None:
        return [], diagnostics, "conflicting_time_map: " + conflict
    return edits, diagnostics, None


def _find_time_conflict(edits: list[dict[str, Any]]) -> str | None:
    """Detect contradictory overlapping time mappings.

    Returns:
        Conflict description, or None when the plan is consistent.
    """
    speeds = [item for item in edits if item["op"] == "speed"]
    for left_pos, left in enumerate(speeds):
        for right in speeds[left_pos + 1 :]:
            overlap = min(left["end_s"], right["end_s"]) - max(left["start_s"], right["start_s"])
            if overlap > 0 and left["factor"] != right["factor"]:
                return (
                    f"overlapping speed {left['factor']}x"
                    f" [{left['start_s']},{left['end_s']}] vs"
                    f" {right['factor']}x [{right['start_s']},{right['end_s']}]"
                )
    return None


def _apply_cuts(
    kept: list[tuple[float, float]], cuts: list[dict[str, Any]]
) -> tuple[list[tuple[float, float]], list[str], str | None]:
    """Subtract cut intervals from kept source spans (overlapping cuts merge).

    Returns:
        Tuple of (kept spans, diagnostics, failure reason or None).
    """
    spans = sorted(kept)
    for cut in sorted(cuts, key=lambda item: item["start_s"]):
        merged: list[tuple[float, float]] = []
        for start, end in spans:
            if end <= cut["start_s"] or start >= cut["end_s"]:
                merged.append((start, end))
                continue
            if start < cut["start_s"]:
                merged.append((start, cut["start_s"]))
            if end > cut["end_s"]:
                merged.append((cut["end_s"], end))
        spans = merged
    if not spans:
        return [], [], "empty_timeline: cuts remove the whole source"
    return spans, [], None


def _apply_speeds(
    spans: list[tuple[float, float]], speeds: list[dict[str, Any]], fps: float
) -> tuple[list[tuple[int, list[int]]], list[str]]:
    """Expand kept spans into per-segment source frame index lists.

    Returns:
        Tuple of (segments as (segment_id, frame indices), diagnostics).
    """
    segments: list[tuple[int, list[int]]] = []
    for segment_id, (start, end) in enumerate(spans):
        first = round(start * fps)
        last = round(end * fps) - 1
        indices = list(range(max(first, 0), max(last + 1, 0)))
        factor = 1.0
        for item in speeds:
            if item["start_s"] < end and item["end_s"] > start and item["factor"] != factor:
                overlap_a = max(start, item["start_s"])
                overlap_b = min(end, item["end_s"])
                if overlap_a <= start and overlap_b >= end:
                    factor = item["factor"]
        if factor != 1.0:
            step = factor
            picked = [indices[round(k * step)] for k in range(int(len(indices) / step))]
            indices = picked or indices[:1]
        segments.append((segment_id, indices))
    return segments, []


def _apply_pauses(
    segments: list[tuple[int, list[int]]],
    pauses: list[dict[str, Any]],
    fps: float,
    out_fps: float,
) -> tuple[list[int], list[str], list[tuple[int, int, float]]]:
    """Flatten segments into the presentation frame order, inserting pauses.

    Returns:
        Tuple of (source frame index per presentation frame, diagnostics, pause
        spans as (first frame offset, frame count, source time)).
        Pauses landing in cut-away regions are dropped with a diagnostic.
    """
    order: list[int] = [index for _, indices in segments for index in indices]
    if not order:
        return order, [], []
    diagnostics: list[str] = []
    inserts: list[tuple[int, int]] = []
    for pause in pauses:
        position = round(pause["at_s"] * fps)
        hold = round(pause["duration_s"] * out_fps)
        if position in order:
            inserts.append((order.index(position), hold))
        else:
            diagnostics.append(f"dropped_pause: at_s {pause['at_s']} lies in a cut-away region")
    pause_spans: list[tuple[int, int, float]] = []
    offset = 0
    for slot, hold in sorted(inserts):
        start = slot + offset + 1
        pause_spans.append((start, hold, order[slot] / fps))
        offset += hold
    for slot, hold in sorted(inserts, reverse=True):
        order[slot + 1 : slot + 1] = [order[slot]] * hold
    return order, diagnostics, pause_spans


def _render_frame(source_index: int, width: int, height: int) -> Any:
    """Render one deterministic numbered RGB frame for a source index.

    Returns:
        HxWx3 uint8 RGB array with the source index drawn as text.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415

    shade = (source_index * 37) % 200 + 20
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (shade, (shade * 2) % 256, (shade * 3) % 256)
    canvas[0:4, :] = 255
    canvas[-4:, :] = 255
    canvas[:, 0:4] = 255
    canvas[:, -4:] = 255
    image = Image.fromarray(canvas)
    ImageDraw.Draw(image).text((8, 8), f"src-{source_index:05d}", fill=(255, 255, 255))
    return np.asarray(image)


def _encode_mp4(
    frames: list[Any], path: Path, fps: float, env: dict[str, Any]
) -> tuple[str | None, dict[str, Any]]:
    """Encode presentation frames to MP4 with the declared backend.

    Returns:
        Tuple of (failure reason or None, encoder record for the receipt).
    """
    import imageio.v2 as imageio  # noqa: PLC0415

    record = dict(env)
    try:
        imageio.mimsave(str(path), frames, fps=fps, codec=ENCODER_CODEC, macro_block_size=None)
    except Exception as error:  # noqa: BLE001 - encoder crash must fail closed
        return f"encoder_failed: {type(error).__name__}:{str(error)[:120]}", record
    record["output_fps"] = fps
    record["output_frames"] = len(frames)
    return None, record


def _decode_frame_count(path: Path) -> int | None:
    """Count decodable frames in an encoded MP4.

    Returns:
        Frame count, or None when the file cannot be decoded.
    """
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        with imageio.get_reader(str(path)) as reader:
            return sum(1 for _ in reader)
    except Exception:  # noqa: BLE001 - undecodable output is reported, not raised
        return None


def _build_result(
    request: ComponentRequest,
    status: str,
    reason: str,
    out_dir: Path,
    artifacts: list[dict[str, Any]],
    extra_provenance: dict[str, Any] | None = None,
) -> ComponentResult:
    """Assemble the result envelope for one run.

    Returns:
        ComponentResult with request identity, status, and provenance.
    """
    provenance = {
        "component_id": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "output_directory": str(out_dir),
    }
    if extra_provenance:
        provenance.update(extra_provenance)
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status=status,
        reason=reason,
        artifacts=tuple(artifacts),
        provenance=provenance,
    )


def _failure(request: ComponentRequest, out_dir: Path, reason: str) -> ComponentResult:
    """Return a failed envelope; failed outputs carry no complete artifacts.

    Returns:
        Failed ComponentResult with the given reason.
    """
    return _build_result(request, STATUS_FAILED, reason, out_dir, [])


@dataclass
class _Prepared:
    """Validated inputs shared by the timeline and emit phases."""

    out_dir: Path
    config: dict[str, Any]
    env: dict[str, Any]
    by_format: dict[str, list[Any]]
    manifest: dict[str, Any]
    manifest_doc: Any
    width: int
    height: int
    out_fps: float


def _load_manifest_phase(
    request: ComponentRequest,
    by_format: dict[str, list[Any]],
    root: Path,
    out_dir: Path,
) -> tuple[dict[str, Any] | None, Any, ComponentResult | None]:
    """Load and validate the frame-sequence manifest document.

    Returns:
        Tuple of (manifest or None, raw document or None, early result or None).
    """
    manifest_ref = by_format["frame-sequence-manifest.v1"][0]
    manifest_path = _resolve_under(Path(manifest_ref.uri), root)
    if manifest_path is None:
        return (
            None,
            None,
            _failure(request, out_dir, f"{manifest_ref.artifact_id}: source_uri_escapes_root"),
        )
    manifest_doc, manifest_problem = _read_json(manifest_path)
    if manifest_doc is None:
        return None, None, _failure(request, out_dir, manifest_problem or "manifest_unreadable")
    manifest, manifest_error = _validate_manifest(manifest_doc)
    if manifest is None or manifest_error is not None:
        return None, None, _failure(request, out_dir, manifest_error or "manifest_invalid")
    return manifest, manifest_doc, None


def _prepare(
    request: ComponentRequest, root: Path
) -> tuple[_Prepared | None, ComponentResult | None]:
    """Validate the request envelope, backend, manifest, and preset.

    Returns:
        Tuple of (prepared inputs or None, early result envelope or None).
        Exactly one element is not None.
    """
    out_dir = root / request.output_directory
    config = dict(request.config or {})

    version_problem = _check_version_compatible(config)
    if version_problem is not None:
        return None, _failure(request, out_dir, version_problem)
    capability_problem = _check_capabilities(request)
    if capability_problem is not None:
        return None, _build_result(request, STATUS_UNAVAILABLE, capability_problem, out_dir, [])

    audio = config.get("audio_policy", AUDIO_POLICY_SILENT)
    if audio != AUDIO_POLICY_SILENT:
        return None, _failure(
            request, out_dir, f"unsupported_audio_policy: v1 output is silent, got {audio!r}"
        )

    env, encoder_problem = _encoder_probe()
    if env is None or encoder_problem is not None:
        report = out_dir / OUTPUT_CAPABILITY_FILENAME
        _write_json(report, {"missing": ["mp4-encode"], "reason": encoder_problem})
        return None, _build_result(request, STATUS_UNAVAILABLE, encoder_problem or "", out_dir, [])

    by_format: dict[str, list[Any]] = {}
    for ref in request.sources:
        by_format.setdefault(ref.format, []).append(ref)
    if "frame-sequence-manifest.v1" not in by_format:
        report = out_dir / OUTPUT_CAPABILITY_FILENAME
        _write_json(
            report, {"missing": ["frame-sequence"], "reason": "required source family absent"}
        )
        return None, _build_result(
            request,
            STATUS_UNAVAILABLE,
            "required_source_family_missing: frame-sequence",
            out_dir,
            [],
        )

    manifest, manifest_doc, manifest_early = _load_manifest_phase(request, by_format, root, out_dir)
    if manifest is None or manifest_early is not None:
        return None, manifest_early

    preset = config.get("preset", dict(TINY_PRESET))
    try:
        width, height = int(preset["width"]), int(preset["height"])
        out_fps = float(preset["fps"])
    except (KeyError, TypeError, ValueError):
        return None, _failure(request, out_dir, "preset_invalid: need width/height/fps numbers")
    if width < 16 or height < 16 or out_fps <= 0:
        return None, _failure(request, out_dir, "preset_invalid: degenerate output geometry")
    prepared = _Prepared(
        out_dir=out_dir,
        config=config,
        env=env,
        by_format=by_format,
        manifest=manifest,
        manifest_doc=manifest_doc,
        width=width,
        height=height,
        out_fps=out_fps,
    )
    return prepared, None


@dataclass
class _Planned:
    """Timeline plan shared by the segment builder and emit phase."""

    order: list[int]
    segments: list[tuple[int, list[int]]]
    speeds: list[dict[str, Any]]
    cuts: list[dict[str, Any]]
    pause_spans: list[tuple[int, int, float]]
    diagnostics: list[str]
    crop_box: tuple[int, int, int, int] | None


def _resolve_crop(
    crops: list[dict[str, Any]], width: int, height: int
) -> tuple[tuple[int, int, int, int] | None, str | None, str | None]:
    """Resolve the crop box against the output geometry.

    Returns:
        Tuple of (box or None, diagnostic or None, failure reason or None).
    """
    if not crops:
        return None, None, None
    box = crops[-1]
    left = max(0, int(box["x"]))
    top = max(0, int(box["y"]))
    right = min(width, left + max(1, int(box["width"])))
    bottom = min(height, top + max(1, int(box["height"])))
    if right <= left or bottom <= top:
        return None, None, "crop_invalid: crop lies outside the frame"
    diagnostic = None
    if left != int(box["x"]) or top != int(box["y"]) or right - left != int(box["width"]):
        diagnostic = "crop_clamped: box clipped to frame bounds"
    return (left, top, right, bottom), diagnostic, None


def _plan_order(prepared: _Prepared) -> tuple[_Planned | None, str | None]:
    """Build the presentation frame order from cuts, speeds, pauses, and crop.

    Returns:
        Tuple of (plan or None, failure reason or None).
    """
    manifest = prepared.manifest
    duration_s = manifest["source_frames"] / manifest["source_fps"]
    edits, edit_diagnostics, edit_error = _validate_edits(prepared.config.get("edits"), duration_s)
    if edit_error is not None:
        return None, edit_error

    cuts = [item for item in edits if item["op"] == "cut"]
    speeds = [item for item in edits if item["op"] == "speed"]
    pauses = [item for item in edits if item["op"] == "pause"]
    crops = [item for item in edits if item["op"] == "crop"]

    kept, _, cut_error = _apply_cuts([(0.0, duration_s)], cuts)
    if cut_error is not None:
        return None, cut_error
    segments, _ = _apply_speeds(kept, speeds, manifest["source_fps"])
    order, pause_diagnostics, pause_spans = _apply_pauses(
        segments, pauses, manifest["source_fps"], prepared.out_fps
    )
    if not order:
        return None, "empty_timeline: no presentation frames remain"
    crop_box, crop_diagnostic, crop_error = _resolve_crop(crops, prepared.width, prepared.height)
    if crop_error is not None:
        return None, crop_error
    diagnostics = list(edit_diagnostics) + list(pause_diagnostics)
    if crop_diagnostic is not None:
        diagnostics.append(crop_diagnostic)
    plan = _Planned(
        order=order,
        segments=segments,
        speeds=speeds,
        cuts=cuts,
        pause_spans=pause_spans,
        diagnostics=diagnostics,
        crop_box=crop_box,
    )
    return plan, None


def _flush_segment(block: dict[str, Any] | None, segments_out: list[dict[str, Any]]) -> None:
    """Round one accumulated map segment and append it to the output list."""
    if block is not None:
        block["presentation_start_s"] = round(block["presentation_start_s"], 6)
        block["presentation_end_s"] = round(block["presentation_end_s"], 6)
        block["source_start_s"] = round(block["source_start_s"], 6)
        block["source_end_s"] = round(block["source_end_s"], 6)
        segments_out.append(block)


def _build_segments(
    order: list[int],
    segments: list[tuple[int, list[int]]],
    speeds: list[dict[str, Any]],
    cuts: list[dict[str, Any]],
    pause_spans: list[tuple[int, int, float]],
    source_fps: float,
    out_fps: float,
) -> list[dict[str, Any]]:
    """Group the final frame order into piecewise map segments.

    Returns:
        Ordered segment mappings with presentation/source intervals.
    """
    segments_out: list[dict[str, Any]] = []
    pause_positions = {
        position for first, count, _ in pause_spans for position in range(first, first + count)
    }
    segment_frames = [(seg_id, index) for seg_id, indices in segments for index in indices]
    pointer = 0
    current: dict[str, Any] | None = None
    for position, source_index in enumerate(order):
        if position in pause_positions:
            operation, seg_id = "pause", -1
        else:
            seg_id, _ = segment_frames[pointer]
            pointer += 1
            operation = "speed" if speeds else "cut" if cuts else "passthrough"
        source_time = source_index / source_fps
        if current is None or current["operation"] != operation or current["segment"] != seg_id:
            _flush_segment(current, segments_out)
            current = {
                "presentation_start_s": position / out_fps,
                "presentation_end_s": (position + 1) / out_fps,
                "source_start_s": source_time,
                "source_end_s": source_time,
                "operation": operation,
                "segment": seg_id if seg_id >= 0 else len(segments_out),
            }
        else:
            current["presentation_end_s"] = (position + 1) / out_fps
            current["source_end_s"] = source_time
    _flush_segment(current, segments_out)
    return segments_out


def run(request: ComponentRequest, base: Path | str = Path(".")) -> ComponentResult:
    """Execute the review-encode component for one validated request.

    Returns:
        Complete, partial, unavailable, or failed result envelope.
    """
    root = Path(base)
    prepared, early = _prepare(request, root)
    if prepared is None or early is not None:
        return early if early is not None else _failure(request, root, "prepare_failed")
    plan, plan_error = _plan_order(prepared)
    if plan is None or plan_error is not None:
        return _failure(request, prepared.out_dir, plan_error or "plan_failed")

    out_dir = prepared.out_dir
    if out_dir.exists() and any(out_dir.iterdir()) and not prepared.config.get("overwrite", False):
        return _failure(request, out_dir, "output_collision: output directory is not empty")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [_render_frame(index, prepared.width, prepared.height) for index in plan.order]
    if plan.crop_box is not None:
        left, top, right, bottom = plan.crop_box
        frames = [frame[top:bottom, left:right] for frame in frames]

    segments_out = _build_segments(
        plan.order,
        plan.segments,
        plan.speeds,
        plan.cuts,
        plan.pause_spans,
        prepared.manifest["source_fps"],
        prepared.out_fps,
    )

    video_path = out_dir / OUTPUT_VIDEO_FILENAME
    encode_error, encoder_record = _encode_mp4(frames, video_path, prepared.out_fps, prepared.env)
    if encode_error is not None:
        return _failure(request, out_dir, encode_error)
    decoded = _decode_frame_count(video_path)
    if decoded != len(frames):
        return _failure(
            request, out_dir, f"encoder_verification_failed: decoded {decoded}, wrote {len(frames)}"
        )
    return _emit_outputs(request, prepared, plan, frames, segments_out, encoder_record, root)


def _probe_clip(by_format: dict[str, list[Any]], root: Path) -> tuple[str | None, str | None]:
    """Digest the optional source clip without decoding it.

    Returns:
        Tuple of (clip digest or None, diagnostic or None). An unreadable
        optional clip is ignored with a diagnostic, never a failure.
    """
    if "source-clip" not in by_format:
        return None, None
    clip_path = _resolve_under(Path(by_format["source-clip"][0].uri), root)
    if clip_path is None or not clip_path.is_file():
        return None, "source_clip_unreadable: optional clip ignored"
    try:
        return _sha256_bytes(clip_path.read_bytes()), None
    except OSError:
        return None, "source_clip_unreadable: optional clip ignored"


def _emit_outputs(
    request: ComponentRequest,
    prepared: _Prepared,
    plan: _Planned,
    frames: list[Any],
    segments_out: list[dict[str, Any]],
    encoder_record: dict[str, Any],
    root: Path,
) -> ComponentResult:
    """Write the video, receipt, and time map, then assemble the envelope.

    Returns:
        Complete envelope, or partial when diagnostics were recorded.
    """
    out_dir = prepared.out_dir
    manifest = prepared.manifest
    clip_digest, clip_diagnostic = _probe_clip(prepared.by_format, root)
    receipt = {
        "component": COMPONENT_ID,
        "component_version": COMPONENT_VERSION,
        "encoder": encoder_record,
        "source": {
            "manifest_digest": _canonical_digest(prepared.manifest_doc),
            "source_frames": manifest["source_frames"],
            "source_fps": manifest["source_fps"],
            "clip_digest": clip_digest,
        },
        "edit_plan_digest": _canonical_digest(prepared.config.get("edits", [])),
        "output": {
            "video": OUTPUT_VIDEO_FILENAME,
            "frames": len(frames),
            "width": frames[0].shape[1],
            "height": frames[0].shape[0],
            "fps": prepared.out_fps,
        },
        "presentation_duration_s": round(len(frames) / prepared.out_fps, 6),
    }
    time_map = {
        "component": COMPONENT_ID,
        "segments": segments_out,
        "frame_order_source_indices": plan.order,
        "first_source_frame": plan.order[0],
        "terminal_source_frame": plan.order[-1],
    }
    _write_json(out_dir / OUTPUT_RECEIPT_FILENAME, receipt)
    _write_json(out_dir / OUTPUT_TIMEMAP_FILENAME, time_map)

    diagnostics = list(plan.diagnostics)
    if clip_diagnostic is not None:
        diagnostics.append(clip_diagnostic)
    if diagnostics:
        return _build_result(
            request,
            STATUS_PARTIAL,
            "; ".join(sorted(set(diagnostics))[:5]),
            out_dir,
            [],
            {"presentation_frames": len(frames)},
        )
    artifacts = [
        {
            "artifact_id": "review-edit",
            "uri": OUTPUT_VIDEO_FILENAME,
            "format": "review-edit-mp4.v1",
        },
        {
            "artifact_id": "encode-receipt",
            "uri": OUTPUT_RECEIPT_FILENAME,
            "format": "encode-receipt.v1",
        },
        {
            "artifact_id": "presentation-time-map",
            "uri": OUTPUT_TIMEMAP_FILENAME,
            "format": "presentation-time-map.v1",
        },
    ]
    return _build_result(
        request,
        STATUS_COMPLETE,
        "encoded",
        out_dir,
        artifacts,
        {"presentation_frames": len(frames)},
    )


def _load_request(input_path: Path) -> ComponentRequest:
    """Load and validate the component request envelope.

    Returns:
        Validated ComponentRequest.
    """
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return component_request_from_dict(payload, source=str(input_path))


def _load_config(config_path: Path | None) -> dict[str, Any]:
    """Load request-level config overrides merged over the envelope config.

    Returns:
        Override mapping (empty when no config path is given).
    """
    if config_path is None:
        return {}
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReviewContractsValidationError(["config must be an object"], source=str(config_path))
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m robot_sf.render.review_encode``.

    Returns:
        Process exit code (0 on handled completion, 2 on invalid input).
    """
    parser = argparse.ArgumentParser(
        description="Encode reproducible review edits from frame sequences."
    )
    parser.add_argument("--input", required=True, help="Component request JSON path.")
    parser.add_argument("--config", help="Config override JSON path (merged over envelope config).")
    parser.add_argument("--output", required=True, help="Output directory for the edit artifacts.")
    parser.add_argument("--base", default=".", help="Fixture root for relative source URIs.")
    args = parser.parse_args(argv)

    try:
        request = _load_request(Path(args.input))
    except (OSError, ValueError, ReviewContractsValidationError) as error:
        print(json.dumps({"status": "failed", "reason": f"request_invalid: {error}"}))  # noqa: T201 - CLI output
        return 2
    try:
        overrides = _load_config(Path(args.config) if args.config else None)
    except (OSError, ValueError, ReviewContractsValidationError) as error:
        print(json.dumps({"status": "failed", "reason": f"config_invalid: {error}"}))  # noqa: T201 - CLI output
        return 2
    merged = ComponentRequest(
        request_id=request.request_id,
        component_id=request.component_id,
        sources=request.sources,
        output_directory=args.output,
        config={**request.config, **overrides},
        required_capabilities=request.required_capabilities,
    )
    result = run(merged, base=Path(args.base))
    print(  # noqa: T201 - CLI output
        json.dumps(
            {
                "status": result.status,
                "reason": result.reason,
                "artifacts": list(result.artifacts),
                "provenance": result.provenance,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
