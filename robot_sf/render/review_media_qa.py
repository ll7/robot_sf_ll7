"""Standalone video and visual-quality checks for scenario review (SREV-13, issue #9282).

Consumes the v1 scenario-review contracts fixed by #9270 (`component-request.v1`,
`component-descriptor.v1`) and delivers a JSON quality report, a contact sheet,
and explicit pass/fail/unavailable checks for fixture media.

Boundaries enforced by this module:
- Fixture media are synthetic frame-sequence documents (`media-qa-frames.v1`);
  real video containers are never decoded without an explicit decoder capability
  and are reported as unsupported rather than silently passed.
- Simulation time is telemetry authority: synchronized checks require an
  explicit presentation-timestamp map (`presentation-timestamp-map.v1`).
  Guessed frame/fps alignment is forbidden; a missing map disables
  synchronized layers with a reason.
- Storyboard-declared pauses are intentional holds; unexplained repeated
  timestamps are freezes and fail the monotonicity check.
- Label findings (unreadable text, pairwise overlap, frame clipping) are
  reported per frame with stable reason codes.
- No metric is derived from screen pixels; geometry stays in world units.
- Core inspection works offline without AI, hosted, or renderer dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
)

COMPONENT_ID = "review-media-qa"
COMPONENT_VERSION = "0.1.0"
DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"
MEDIA_QUALITY_REPORT_SCHEMA_VERSION = "media-quality-report.v1"
CONTACT_SHEET_SCHEMA_VERSION = "contact-sheet.v1"
FRAMES_SCHEMA_VERSION = "media-qa-frames.v1"
TIMESTAMP_MAP_SCHEMA_VERSION = "presentation-timestamp-map.v1"
STORYBOARD_SPEC_SCHEMA_VERSION = "storyboard-spec.v1"

DEFAULT_PRESENTATION = {"width": 1920, "height": 1080, "fps": 30.0, "speed": 1.0}
TEST_PRESENTATION = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}
DEFAULT_DURATION_TOLERANCE_S = 0.05
DEFAULT_MIN_LABEL_CHARS = 1
CONTACT_SHEET_COLUMNS = 3
CONTACT_SHEET_CELL = (160, 90)

REQUIRED_CAPABILITIES = ("video-quality-checks",)
OPTIONAL_CAPABILITIES = (
    "contact-sheet-render",
    "container-decode",
    "timestamp-sync-checks",
    "storyboard-pause-analysis",
)
OUTPUT_TYPES = (MEDIA_QUALITY_REPORT_SCHEMA_VERSION, CONTACT_SHEET_SCHEMA_VERSION)

DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=OUTPUT_TYPES,
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)

_CONTAINER_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
_MAX_FINDING_DETAILS = 25


def descriptor_document() -> dict[str, Any]:
    """Return the schema-shaped capability descriptor for this component.

    Returns:
        ``component-descriptor.v1`` document for review-media-qa.
    """
    payload = asdict(DESCRIPTOR)
    for key in (
        "supported_input_versions",
        "output_types",
        "required_capabilities",
        "optional_capabilities",
    ):
        payload[key] = list(payload[key])
    return {"schema_version": DESCRIPTOR_SCHEMA_VERSION, **payload}


def _resolve_output_dir(request: ComponentRequest, base: Path) -> Path:
    """Fail closed on an existing output directory so sources and prior runs survive.

    Returns:
        Resolved output directory path that does not exist yet.

    Raises:
        ReviewContractsValidationError: If the output directory already exists.
    """
    output_dir = base / request.output_directory
    if output_dir.exists():
        raise ReviewContractsValidationError(
            [f"/output_directory: output collision, already exists: {request.output_directory}"]
        )
    return output_dir


def _write_bytes(path: Path, raw: bytes) -> str:
    """Write one artifact atomically inside the requested output directory.

    Returns:
        SHA-256 digest of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_bytes(raw)
    tmp_path.replace(path)
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Serialize and write JSON atomically with sorted keys and indentation.

    Returns:
        SHA-256 digest of the written JSON string.
    """
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_bytes(path, text.encode("utf-8"))


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    """List requested capabilities the descriptor does not declare.

    Returns:
        List of unsupported capability names.
    """
    supported = set(DESCRIPTOR.required_capabilities) | set(DESCRIPTOR.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _is_finite_number(value: Any) -> bool:
    """Return whether a value is a finite int/float (excluding booleans)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_document(
    check_id: str, verdict: str, *, reason_code: str, detail: str = ""
) -> dict[str, Any]:
    """Build one stable check record for the quality report.

    Returns:
        Check record with check_id, verdict, reason_code, and optional detail.
    """
    record: dict[str, Any] = {"check_id": check_id, "verdict": verdict, "reason_code": reason_code}
    if detail:
        record["detail"] = detail
    return record


def _read_source_bytes(root: Path, uri: str) -> tuple[bytes | None, dict[str, Any] | None]:
    """Read one declared source file, reporting launcher-level failures as findings.

    Returns:
        Tuple of (raw bytes or None, finding or None).
    """
    source_path = root / uri
    if not source_path.exists():
        return None, {
            "artifact_id": uri,
            "reason_code": "source_missing",
            "detail": f"source path does not exist: {uri}",
        }
    try:
        return source_path.read_bytes(), None
    except OSError as exc:
        return None, {
            "artifact_id": uri,
            "reason_code": "source_unreadable",
            "detail": f"cannot read source: {exc}",
        }


def _parse_frames_document(raw: bytes, artifact_id: str) -> tuple[dict[str, Any] | None, str]:
    """Parse a synthetic frame-sequence document, rejecting corrupt payloads.

    Returns:
        Tuple of (parsed mapping or None, error detail or empty string).
    """
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"clip is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "clip document must be a JSON object"
    if parsed.get("schema_version") != FRAMES_SCHEMA_VERSION:
        return None, (
            f"unsupported clip schema: {parsed.get('schema_version')!r}; "
            f"expected {FRAMES_SCHEMA_VERSION!r}"
        )
    frames = parsed.get("frames")
    if not isinstance(frames, list):
        return None, "clip document must declare a frames list"
    _ = artifact_id
    return parsed, ""


def _validate_presentation(cfg: Any) -> dict[str, Any]:
    """Validate the presentation preset and return the effective mapping.

    Returns:
        Effective presentation mapping with width, height, fps, and speed.

    Raises:
        ReviewContractsValidationError: For non-finite or non-positive dimensions.
    """
    presentation = dict(DEFAULT_PRESENTATION)
    if isinstance(cfg, dict):
        presentation.update(cfg)
    errors: list[str] = []
    for key in ("width", "height", "fps", "speed"):
        value = presentation.get(key)
        if not _is_finite_number(value) or float(value) <= 0:
            errors.append(f"/config/presentation/{key}: must be a finite positive number")
    if errors:
        raise ReviewContractsValidationError(errors)
    return {
        "width": int(presentation["width"]),
        "height": int(presentation["height"]),
        "fps": float(presentation["fps"]),
        "speed": float(presentation["speed"]),
    }


def _validate_thresholds(cfg: Any) -> dict[str, float]:
    """Validate QA thresholds and return the effective mapping.

    Returns:
        Effective thresholds mapping with duration tolerance and min label chars.

    Raises:
        ReviewContractsValidationError: For non-finite or negative thresholds.
    """
    raw = cfg if isinstance(cfg, dict) else {}
    tolerance = raw.get("duration_tolerance_s", DEFAULT_DURATION_TOLERANCE_S)
    min_chars = raw.get("min_label_chars", DEFAULT_MIN_LABEL_CHARS)
    errors: list[str] = []
    if not _is_finite_number(tolerance) or float(tolerance) < 0:
        errors.append("/config/thresholds/duration_tolerance_s: must be finite and >= 0")
    if isinstance(min_chars, bool) or not isinstance(min_chars, int) or min_chars < 1:
        errors.append("/config/thresholds/min_label_chars: must be an integer >= 1")
    if errors:
        raise ReviewContractsValidationError(errors)
    return {"duration_tolerance_s": float(tolerance), "min_label_chars": float(min_chars)}


def _clip_integrity_check(
    clip: dict[str, Any] | None, error: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate clip integrity (corrupt, truncated, blank).

    Returns:
        Tuple of (check record, frames list, which may be empty on failure).
    """
    if clip is None:
        return _check_document(
            "clip_integrity", "fail", reason_code="corrupt_clip", detail=error
        ), []
    frames = [entry for entry in clip.get("frames", []) if isinstance(entry, dict)]
    declared = clip.get("declared_frame_count")
    if isinstance(declared, int) and declared != len(frames):
        return _check_document(
            "clip_integrity",
            "fail",
            reason_code="truncated_clip",
            detail=f"declared_frame_count={declared} but parsed {len(frames)} frames",
        ), frames
    if not frames:
        return _check_document(
            "clip_integrity", "fail", reason_code="blank_clip", detail="clip declares no frames"
        ), []
    if all(entry.get("blank", False) is True for entry in frames):
        return _check_document(
            "clip_integrity", "fail", reason_code="blank_clip", detail="all frames are blank"
        ), frames
    return _check_document(
        "clip_integrity", "pass", reason_code="ok", detail=f"{len(frames)} frames parsed"
    ), frames


def _frame_times(frames: list[dict[str, Any]]) -> list[float] | None:
    """Extract frame timestamps, returning None when any timestamp is not finite.

    Returns:
        List of finite timestamps, or None when any timestamp is not finite.
    """
    times: list[float] = []
    for entry in frames:
        stamp = entry.get("t_s")
        if not _is_finite_number(stamp):
            return None
        times.append(float(stamp))
    return times


def _duration_check(
    clip: dict[str, Any], times: list[float] | None, tolerance_s: float
) -> dict[str, Any]:
    """Compare the declared clip duration against measured frame timestamps.

    Returns:
        Duration check record with pass, fail, or unavailable verdict.
    """
    if times is None:
        return _check_document(
            "duration_match",
            "fail",
            reason_code="non_finite_timestamp",
            detail="frame timestamps must be finite numbers",
        )
    declared = clip.get("duration_s")
    if not _is_finite_number(declared):
        return _check_document(
            "duration_match",
            "unavailable",
            reason_code="missing_expected_duration",
            detail="clip declares no finite duration_s to compare against",
        )
    measured = (max(times) - min(times)) if times else 0.0
    if abs(float(declared) - measured) <= tolerance_s:
        return _check_document(
            "duration_match",
            "pass",
            reason_code="ok",
            detail=f"declared={float(declared):g}s measured={measured:g}s",
        )
    return _check_document(
        "duration_match",
        "fail",
        reason_code="duration_mismatch",
        detail=f"declared={float(declared):g}s measured={measured:g}s tolerance={tolerance_s:g}s",
    )


def _pause_ranges(storyboard: dict[str, Any] | None) -> list[tuple[int, int]]:
    """Extract declared intentional pause ranges as (from, to) frame-index pairs.

    Returns:
        List of validated (from, to) frame-index pause ranges.
    """
    if not isinstance(storyboard, dict):
        return []
    ranges: list[tuple[int, int]] = []
    for pause in storyboard.get("pauses", []):
        if not isinstance(pause, dict):
            continue
        start = pause.get("from_frame_index")
        end = pause.get("to_frame_index")
        if isinstance(start, int) and isinstance(end, int) and start <= end and start >= 0:
            ranges.append((start, end))
    return ranges


def _in_pause(index: int, ranges: list[tuple[int, int]]) -> bool:
    """Return whether a frame-index step falls inside a declared pause range."""
    return any(start <= index <= end for start, end in ranges)


def _monotonicity_check(
    times: list[float], storyboard: dict[str, Any] | None, analyze_pauses: bool
) -> dict[str, Any]:
    """Distinguish intentional storyboard pauses from unexplained freezes and regressions.

    Returns:
        Monotonicity check record with pass, fail, or unavailable verdict.
    """
    if not times:
        return _check_document(
            "timestamp_monotonicity",
            "unavailable",
            reason_code="no_frames",
            detail="no frame timestamps to evaluate",
        )
    ranges = _pause_ranges(storyboard) if analyze_pauses else []
    freezes: list[str] = []
    for position in range(1, len(times)):
        previous, current = times[position - 1], times[position]
        if current < previous:
            return _check_document(
                "timestamp_monotonicity",
                "fail",
                reason_code="nonmonotonic_timestamps",
                detail=f"t_s decreases at frame {position}: {previous:g} -> {current:g}",
            )
        if current == previous and not _in_pause(position, ranges):
            freezes.append(f"frame {position} repeats t_s={current:g}s without a declared pause")
    if freezes:
        shown = "; ".join(freezes[:_MAX_FINDING_DETAILS])
        suffix = "" if len(freezes) <= _MAX_FINDING_DETAILS else f" (+{len(freezes) - 25} more)"
        return _check_document(
            "timestamp_monotonicity",
            "fail",
            reason_code="unexplained_freeze",
            detail=shown + suffix,
        )
    if ranges:
        return _check_document(
            "timestamp_monotonicity",
            "pass",
            reason_code="ok_with_declared_pauses",
            detail=f"{len(ranges)} declared pause range(s) honored",
        )
    return _check_document("timestamp_monotonicity", "pass", reason_code="ok")


def _parse_timestamp_map(raw: bytes) -> tuple[dict[str, Any] | None, str]:
    """Parse an explicit presentation-timestamp map, rejecting guessed alignment inputs.

    Returns:
        Tuple of (parsed mapping or None, error detail or empty string).
    """
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"timestamp map is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "timestamp map must be a JSON object"
    if parsed.get("schema_version") != TIMESTAMP_MAP_SCHEMA_VERSION:
        return None, (
            f"unsupported timestamp map schema: {parsed.get('schema_version')!r}; "
            f"expected {TIMESTAMP_MAP_SCHEMA_VERSION!r}"
        )
    if not isinstance(parsed.get("entries"), list):
        return None, "timestamp map must declare an entries list"
    return parsed, ""


def _timestamp_sync_check(
    frames: list[dict[str, Any]],
    timestamp_map: dict[str, Any] | None,
    map_error: str,
    supported: bool,
) -> dict[str, Any]:
    """Verify explicit frame-to-source time mapping without guessing alignment.

    Returns:
        Timestamp sync check record with pass, fail, or unavailable verdict.
    """
    if not supported:
        return _check_document(
            "timestamp_sync",
            "unavailable",
            reason_code="capability_not_requested",
            detail="timestamp-sync-checks capability was not requested",
        )
    if timestamp_map is None:
        reason = "missing_timestamp_map" if not map_error else "timestamp_map_unreadable"
        detail = map_error or (
            "synchronized checks require an explicit presentation-timestamp map; "
            "frame/fps guessing is forbidden"
        )
        return _check_document("timestamp_sync", "unavailable", reason_code=reason, detail=detail)
    mapped: dict[int, tuple[float, float]] = {}
    for entry in timestamp_map.get("entries", []):
        if not isinstance(entry, dict):
            continue
        index = entry.get("frame_index")
        media = entry.get("media_t_s")
        source = entry.get("source_t_s")
        if isinstance(index, int) and _is_finite_number(media) and _is_finite_number(source):
            mapped[index] = (float(media), float(source))
    missing = [
        entry.get("frame_index")
        for entry in frames
        if isinstance(entry, dict)
        and isinstance(entry.get("frame_index"), int)
        and entry["frame_index"] not in mapped
    ]
    if missing:
        shown = ", ".join(str(value) for value in sorted(set(missing))[:_MAX_FINDING_DETAILS])
        return _check_document(
            "timestamp_sync",
            "fail",
            reason_code="unmapped_frames",
            detail=f"frames without an explicit map entry: {shown}",
        )
    return _check_document(
        "timestamp_sync",
        "pass",
        reason_code="ok",
        detail=f"{len(mapped)} explicit frame mapping(s) verified",
    )


def _boxes_overlap(first: list[float], second: list[float]) -> bool:
    """Return whether two [x0, y0, x1, y1] boxes share a positive area."""
    return min(first[2], second[2]) > max(first[0], second[0]) and min(first[3], second[3]) > max(
        first[1], second[1]
    )


def _label_box(label: Any, width: int, height: int) -> tuple[list[float] | None, str]:
    """Normalize one label box, returning (box or None, problem or empty string).

    Returns:
        Tuple of (normalized box or None, problem code or empty string).
    """
    if not isinstance(label, dict):
        return None, "malformed_label"
    text = label.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return None, "unreadable_label"
    box = label.get("box")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not all(_is_finite_number(value) for value in box)
    ):
        return None, "malformed_label_box"
    coords = [float(value) for value in box]
    if coords[2] <= coords[0] or coords[3] <= coords[1]:
        return None, "malformed_label_box"
    if coords[0] < 0 or coords[1] < 0 or coords[2] > width or coords[3] > height:
        return list(coords), "label_clipping"
    return list(coords), ""


def _frame_label_findings(
    entry: dict[str, Any], width: int, height: int, min_chars: int
) -> list[str]:
    """Evaluate one frame's labels for legibility, overlap, and clipping.

    Returns:
        List of per-frame finding strings, empty when the frame is clean.
    """
    index = entry.get("frame_index", "?")
    labels = entry.get("labels", [])
    if not isinstance(labels, list):
        return [f"frame {index}: labels must be a list"]
    findings: list[str] = []
    boxes: list[list[float]] = []
    for label in labels:
        box, problem = _label_box(label, width, height)
        text = label.get("text", "") if isinstance(label, dict) else ""
        if problem in ("unreadable_label", "malformed_label", "malformed_label_box"):
            findings.append(f"frame {index}: {problem}")
            continue
        if isinstance(text, str) and len(text.strip()) < min_chars:
            findings.append(f"frame {index}: unreadable_label")
            continue
        if problem == "label_clipping":
            findings.append(f"frame {index}: label_clipping")
            continue
        if box is None:
            findings.append(f"frame {index}: malformed_label_box")
            continue
        if any(_boxes_overlap(box, other) for other in boxes):
            findings.append(f"frame {index}: label_overlap")
        boxes.append(box)
    return findings


def _label_quality_check(
    frames: list[dict[str, Any]], width: int, height: int, min_chars: int
) -> dict[str, Any]:
    """Evaluate label legibility, pairwise overlap, and frame clipping.

    Returns:
        Label quality check record with pass or fail verdict.
    """
    findings: list[str] = []
    for entry in frames:
        findings.extend(_frame_label_findings(entry, width, height, min_chars))
    if findings:
        shown = "; ".join(findings[:_MAX_FINDING_DETAILS])
        suffix = "" if len(findings) <= _MAX_FINDING_DETAILS else f" (+{len(findings) - 25} more)"
        return _check_document(
            "label_quality", "fail", reason_code="label_findings", detail=shown + suffix
        )
    return _check_document("label_quality", "pass", reason_code="ok")


def _container_decode_check(
    container_present: bool, decoder_available: bool
) -> dict[str, Any] | None:
    """Report container decode support without ever silently passing real media.

    Returns:
        Container check record, or None when no container source was declared.
    """
    if not container_present:
        return None
    if not decoder_available:
        return _check_document(
            "container_decode",
            "unavailable",
            reason_code="decoder_unavailable",
            detail="real video containers require an explicit decoder; none is available offline",
        )
    return _check_document(
        "container_decode",
        "unavailable",
        reason_code="not_exercised",
        detail="container decode is capability-gated and not part of fixture proof",
    )


def _decoder_available() -> bool:
    """Return whether an explicit video decoder is importable (optional dependency)."""
    try:
        import cv2  # noqa: F401, PLC0415 - optional availability probe, never fixture proof

        return True
    except ImportError:
        return False


def _render_contact_sheet(
    frames: list[dict[str, Any]], presentation: dict[str, int], output_dir: Path
) -> tuple[str | None, str]:
    """Render a deterministic PIL contact-sheet grid, returning (filename, error).

    Returns:
        Tuple of (written PNG filename or None, error detail or empty string).
    """
    columns = CONTACT_SHEET_COLUMNS
    cell_w, cell_h = CONTACT_SHEET_CELL
    rows = max(1, math.ceil(len(frames) / columns))
    sheet = Image.new("RGB", (columns * cell_w, rows * (cell_h + 14)), color=(24, 24, 24))
    canvas = ImageDraw.Draw(sheet)
    for position, entry in enumerate(frames):
        row, column = divmod(position, columns)
        origin_x, origin_y = column * cell_w, row * (cell_h + 14)
        canvas.rectangle(
            [origin_x, origin_y, origin_x + cell_w - 1, origin_y + cell_h - 1],
            outline=(200, 200, 200),
        )
        label_count = len(entry.get("labels", [])) if isinstance(entry.get("labels"), list) else 0
        caption = f"f{entry.get('frame_index', '?')} t={entry.get('t_s', '?')}s n={label_count}"
        canvas.text((origin_x + 4, origin_y + cell_h + 1), str(caption), fill=(230, 230, 230))
    filename = f"{CONTACT_SHEET_SCHEMA_VERSION}.png"
    target = output_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG")
    return filename, ""


@dataclass
class _IngestedSources:
    """Mutable holder for classified request sources during ingestion."""

    clip: dict[str, Any] | None = None
    clip_error: str = ""
    timestamp_map: dict[str, Any] | None = None
    map_error: str = ""
    storyboard: dict[str, Any] | None = None
    container_present: bool = False
    clip_declared: bool = False


def _fetch_source_bytes(
    root: Path, uri: str, artifact_id: str, expected_sha256: str
) -> tuple[bytes | None, dict[str, Any] | None, str | None]:
    """Read one source and verify integrity, mapping failures to findings.

    Returns:
        Tuple of (raw bytes or None, finding or None, integrity state or None).
    """
    raw, failure = _read_source_bytes(root, uri)
    if failure is not None:
        return None, {"artifact_id": artifact_id, **failure}, None
    assert raw is not None
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and expected_sha256.lower() != digest.lower():
        return (
            None,
            {
                "artifact_id": artifact_id,
                "reason_code": "source_integrity_mismatch",
                "detail": f"source SHA-256 mismatch for {artifact_id}",
            },
            "mismatch",
        )
    return raw, None, "match"


def _register_parsed_source(
    artifact_id: str,
    parsed: dict[str, Any],
    raw: bytes,
    ingested: _IngestedSources,
    diagnostics: list[dict[str, Any]],
) -> None:
    """Classify one parsed source document into the ingestion holder."""
    schema = parsed.get("schema_version")
    if schema == FRAMES_SCHEMA_VERSION and ingested.clip is None:
        clip, error = _parse_frames_document(raw, artifact_id)
        if clip is None:
            diagnostics.append(
                {"artifact_id": artifact_id, "reason_code": "corrupt_clip", "detail": error}
            )
            ingested.clip_error = error
        else:
            ingested.clip = clip
    elif schema == TIMESTAMP_MAP_SCHEMA_VERSION and ingested.timestamp_map is None:
        timestamp_map, error = _parse_timestamp_map(raw)
        if timestamp_map is None:
            diagnostics.append(
                {
                    "artifact_id": artifact_id,
                    "reason_code": "timestamp_map_unreadable",
                    "detail": error,
                }
            )
            ingested.map_error = error
        else:
            ingested.timestamp_map = timestamp_map
    elif schema == STORYBOARD_SPEC_SCHEMA_VERSION and ingested.storyboard is None:
        ingested.storyboard = parsed
    else:
        diagnostics.append(
            {
                "artifact_id": artifact_id,
                "reason_code": "unsupported_source_schema",
                "detail": f"unsupported source schema: {schema!r}",
            }
        )


def _load_inputs(
    request: ComponentRequest, root: Path
) -> tuple[_IngestedSources, list[dict[str, Any]], dict[str, str]]:
    """Read and classify declared sources without guessing media semantics.

    Returns:
        Tuple of (ingested sources, diagnostics, integrity states).
    """
    ingested = _IngestedSources()
    diagnostics: list[dict[str, Any]] = []
    integrity: dict[str, str] = {}

    for source_ref in request.sources:
        if source_ref.uri.lower().endswith(tuple(_CONTAINER_SUFFIXES)):
            ingested.container_present = True
            diagnostics.append(
                {
                    "artifact_id": source_ref.artifact_id,
                    "reason_code": "container_source_noted",
                    "detail": "real video containers are capability-gated, never fixture proof",
                }
            )
            continue
        if source_ref.format == FRAMES_SCHEMA_VERSION:
            ingested.clip_declared = True
        raw, failure, state = _fetch_source_bytes(
            root, source_ref.uri, source_ref.artifact_id, source_ref.sha256
        )
        if failure is not None:
            diagnostics.append(failure)
            continue
        assert raw is not None and state is not None
        integrity[source_ref.artifact_id] = state
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            diagnostics.append(
                {
                    "artifact_id": source_ref.artifact_id,
                    "reason_code": "source_malformed_json",
                    "detail": f"source {source_ref.artifact_id} is not valid JSON",
                }
            )
            continue
        if not isinstance(parsed, dict):
            diagnostics.append(
                {
                    "artifact_id": source_ref.artifact_id,
                    "reason_code": "source_malformed_json",
                    "detail": f"source {source_ref.artifact_id} must be a JSON object",
                }
            )
            continue
        _register_parsed_source(source_ref.artifact_id, parsed, raw, ingested, diagnostics)
    return ingested, diagnostics, integrity


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Execute one review-media-qa request to check fixture video and visual quality.

    Args:
        request: Validated component request specifying media sources and QA configuration.
        base: Base directory under which request paths resolve (defaults to current
            working directory).

    Returns:
        ComponentResult with complete, partial, unavailable, or failed status.
    """
    root = base if base is not None else Path.cwd()

    if request.component_id != COMPONENT_ID:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"unsupported component: {request.component_id}",
        )

    unsupported = _unsupported_capabilities(request)
    if unsupported:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"missing capabilities: {', '.join(sorted(unsupported))}",
        )

    try:
        output_dir = _resolve_output_dir(request, root)
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="; ".join(error.errors),
        )

    try:
        presentation = _validate_presentation(request.config.get("presentation"))
        thresholds = _validate_thresholds(request.config.get("thresholds"))
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="; ".join(error.errors),
        )

    ingested, diagnostics, integrity = _load_inputs(request, root)

    if ingested.clip is None and not ingested.clip_declared and not ingested.container_present:
        diagnostics.append(
            {
                "artifact_id": "",
                "reason_code": "no_clip_source",
                "detail": "no media-qa-frames.v1 source was declared",
            }
        )
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason="no clip source declared",
            diagnostics=tuple(diagnostics),
        )

    checks: list[dict[str, Any]] = []
    integrity_check, frames = _clip_integrity_check(
        ingested.clip,
        ingested.clip_error or "clip source declared but not readable as media-qa-frames.v1",
    )
    checks.append(integrity_check)
    times = _frame_times(frames) if frames else []
    checks.append(_duration_check(ingested.clip or {}, times, thresholds["duration_tolerance_s"]))
    analyze_pauses = "storyboard-pause-analysis" in request.required_capabilities
    checks.append(_monotonicity_check(times or [], ingested.storyboard, analyze_pauses))
    sync_supported = "timestamp-sync-checks" in request.required_capabilities
    checks.append(
        _timestamp_sync_check(frames, ingested.timestamp_map, ingested.map_error, sync_supported)
    )
    checks.append(
        _label_quality_check(
            frames,
            presentation["width"],
            presentation["height"],
            int(thresholds["min_label_chars"]),
        )
    )
    container_check = _container_decode_check(ingested.container_present, _decoder_available())
    if container_check is not None:
        checks.append(container_check)

    return _finalize_outputs(
        request, output_dir, frames, checks, diagnostics, integrity, presentation, thresholds
    )


def _finalize_outputs(
    request: ComponentRequest,
    output_dir: Path,
    frames: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    integrity: dict[str, str],
    presentation: dict[str, int],
    thresholds: dict[str, float],
) -> ComponentResult:
    """Write report artifacts and map check verdicts to the result envelope.

    Returns:
        ComponentResult with complete or partial status and written artifacts.
    """
    failed = [check for check in checks if check["verdict"] == "fail"]
    overall = "fail" if failed else "pass"

    report = {
        "schema_version": MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
        "report_id": f"qa-{request.request_id}",
        "request_id": request.request_id,
        "overall_verdict": overall,
        "checks": checks,
        "thresholds": {
            "duration_tolerance_s": thresholds["duration_tolerance_s"],
            "min_label_chars": int(thresholds["min_label_chars"]),
        },
        "qa_tool_version": COMPONENT_VERSION,
        "presentation": presentation,
        "evidence_boundary": "diagnostic_only",
    }

    cells = [
        {
            "frame_index": entry.get("frame_index"),
            "t_s": entry.get("t_s"),
            "label_count": len(entry.get("labels", []))
            if isinstance(entry.get("labels"), list)
            else 0,
        }
        for entry in frames
    ]
    manifest: dict[str, Any] = {
        "schema_version": CONTACT_SHEET_SCHEMA_VERSION,
        "sheet_id": f"sheet-{request.request_id}",
        "request_id": request.request_id,
        "grid": {"columns": CONTACT_SHEET_COLUMNS, "cell": list(CONTACT_SHEET_CELL)},
        "cells": cells,
    }

    artifacts: list[dict[str, Any]] = []
    render_requested = "contact-sheet-render" in request.required_capabilities
    image_filename: str | None = None
    if render_requested and frames:
        image_filename, render_error = _render_contact_sheet(frames, presentation, output_dir)
        if image_filename is None:
            diagnostics.append(
                {
                    "artifact_id": "",
                    "reason_code": "contact_sheet_unavailable",
                    "detail": render_error,
                }
            )
    if image_filename is not None:
        manifest["image"] = image_filename

    output_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"{MEDIA_QUALITY_REPORT_SCHEMA_VERSION}.json"
    report_sha = _write_json(output_dir / report_filename, report)
    artifacts.append(
        {
            "artifact_id": report_filename,
            "uri": str(Path(request.output_directory) / report_filename),
            "sha256": report_sha,
        }
    )
    manifest_filename = f"{CONTACT_SHEET_SCHEMA_VERSION}.json"
    manifest_sha = _write_json(output_dir / manifest_filename, manifest)
    artifacts.append(
        {
            "artifact_id": manifest_filename,
            "uri": str(Path(request.output_directory) / manifest_filename),
            "sha256": manifest_sha,
        }
    )
    if image_filename is not None:
        image_bytes = (output_dir / image_filename).read_bytes()
        artifacts.append(
            {
                "artifact_id": image_filename,
                "uri": str(Path(request.output_directory) / image_filename),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            }
        )

    if failed or diagnostics:
        status = "partial"
        reason = "; ".join(
            [f"{check['check_id']}: {check['reason_code']}" for check in failed]
            + [str(item.get("reason_code", "diagnostic")) for item in diagnostics]
        )
    else:
        status = "complete"
        reason = ""
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status=status,
        reason=reason,
        artifacts=tuple(artifacts),
        diagnostics=tuple(diagnostics),
        provenance={
            "output_directory": request.output_directory,
            "component_version": COMPONENT_VERSION,
            "source_integrity": integrity,
            "overall_verdict": overall,
            "thresholds": report["thresholds"],
            "qa_tool_version": COMPONENT_VERSION,
            "admission": "diagnostic_only",
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for review_media_qa.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Run standalone video and visual-quality checks on fixture media."
    )
    parser.add_argument("--input", required=False, default=None, help="Component request JSON.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=False, default=None, help="Output directory.")
    parser.add_argument("--base", required=False, default=None, help="Base directory for paths.")
    parser.add_argument(
        "--descriptor",
        action="store_true",
        help="Print the component descriptor instead of running a request.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the review-media-qa component.

    Returns:
        0 for complete, 2 for partial/unavailable, 1 for failed.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.descriptor:
        print(json.dumps(descriptor_document(), sort_keys=True, indent=2))  # noqa: T201 - CLI output
        return 0

    if args.input is None or args.output is None:
        parser.error("--input and --output are required unless --descriptor is used")

    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"cannot read request: {error}"]) from error

    if args.config is not None:
        try:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReviewContractsValidationError([f"cannot read config: {error}"]) from error
        if isinstance(cfg, dict):
            payload = {**payload, "config": {**payload.get("config", {}), **cfg}}

    payload["output_directory"] = args.output
    request = component_request_from_dict(payload, source=args.input)
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(json.dumps(asdict(result), sort_keys=True, indent=2))  # noqa: T201 - CLI output

    if result.status == "complete":
        return 0
    if result.status in ("partial", "unavailable"):
        return 2
    return 1


__all__ = [
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "CONTACT_SHEET_CELL",
    "CONTACT_SHEET_COLUMNS",
    "CONTACT_SHEET_SCHEMA_VERSION",
    "DEFAULT_DURATION_TOLERANCE_S",
    "DEFAULT_MIN_LABEL_CHARS",
    "DEFAULT_PRESENTATION",
    "DESCRIPTOR",
    "FRAMES_SCHEMA_VERSION",
    "MEDIA_QUALITY_REPORT_SCHEMA_VERSION",
    "OPTIONAL_CAPABILITIES",
    "OUTPUT_TYPES",
    "REQUIRED_CAPABILITIES",
    "STORYBOARD_SPEC_SCHEMA_VERSION",
    "TEST_PRESENTATION",
    "TIMESTAMP_MAP_SCHEMA_VERSION",
    "descriptor_document",
    "main",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
