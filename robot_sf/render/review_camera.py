"""Generate camera and layout specifications for scenario presentations (SREV-11, issue #9280).

Consumes the v1 scenario-review contracts fixed by #9270 (`component-request.v1`,
`visualization-spec.v1`, `component-descriptor.v1`) and produces renderer-neutral
camera tracks (full-scene static, follow, event) and overview layout specifications.

Boundaries enforced by this module:
- Aspect ratio, finite/in-bounds camera coordinates, stable deterministic interpolation,
  and context retention are validated.
- Full-scene static camera is always available whenever geometry is provided.
- Follow and event camera tracks require stated trajectory data; if absent, they fail closed
  with a stable diagnostic reason.
- Cropped content is disclosed with full-scene reference geometry and an optional inset specification.
- Simulation and presentation measurements remain in world units (meters, seconds, radians).
  No metrics are derived from screen pixels.
- Core inspection works offline without AI or external renderer dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    component_request_from_dict,
)

COMPONENT_ID = "review-camera"
COMPONENT_VERSION = "0.1.0"
DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"
CAMERA_LAYOUT_SPEC_SCHEMA_VERSION = "camera-layout-spec.v1"
VISUALIZATION_SPEC_SCHEMA_VERSION = "visualization-spec.v1"

DEFAULT_PRESENTATION = {"width": 1920, "height": 1080, "fps": 30.0, "speed": 1.0}
TEST_PRESENTATION = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}
DEFAULT_MARGIN_M = 1.0
DEFAULT_CONTEXT_RETENTION_M = 5.0

REQUIRED_CAPABILITIES = ("full-scene-camera", "camera-interpolation")
OPTIONAL_CAPABILITIES = ("follow-camera", "event-camera", "context-retention", "inset-view")
OUTPUT_TYPES = (CAMERA_LAYOUT_SPEC_SCHEMA_VERSION, VISUALIZATION_SPEC_SCHEMA_VERSION)

DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=OUTPUT_TYPES,
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


def descriptor_document() -> dict[str, Any]:
    """Return the schema-shaped capability descriptor for this component.

    Returns:
        ``component-descriptor.v1`` document for review-camera.
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


def _write_text(path: Path, text: str) -> str:
    """Write one artifact atomically inside the requested output directory.

    Returns:
        SHA-256 digest of the written text.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    """Serialize and write JSON atomically with sorted keys and indentation.

    Returns:
        SHA-256 digest of the written JSON string.
    """
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_text(path, text)


def _unsupported_capabilities(request: ComponentRequest) -> list[str]:
    """List requested capabilities the descriptor does not declare.

    Returns:
        List of unsupported capability names.
    """
    supported = set(DESCRIPTOR.required_capabilities) | set(DESCRIPTOR.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _validate_aspect_ratio(width: int | float, height: int | float) -> float:
    """Validate and compute the aspect ratio (width / height).

    Returns:
        Computed aspect ratio as a positive float.

    Raises:
        ReviewContractsValidationError: If dimensions are non-positive or non-finite.
    """
    try:
        w = float(width)
        h = float(height)
    except (TypeError, ValueError) as exc:
        raise ReviewContractsValidationError(
            [f"presentation dimensions must be numeric: width={width!r}, height={height!r}"]
        ) from exc
    if not math.isfinite(w) or not math.isfinite(h) or w <= 0.0 or h <= 0.0:
        raise ReviewContractsValidationError(
            [f"presentation dimensions must be finite and > 0: width={width!r}, height={height!r}"]
        )
    return w / h


def _bounds_from_map(map_data: Mapping[str, Any]) -> list[float] | None:
    """Derive bounding box from map bounds, width/height, or obstacles.

    Returns:
        [min_x, max_x, min_y, max_y] if derivable, else None.
    """
    if "bounds" in map_data and isinstance(map_data["bounds"], list) and map_data["bounds"]:
        xs: list[float] = []
        ys: list[float] = []
        for segment in map_data["bounds"]:
            if isinstance(segment, (list, tuple)) and len(segment) >= 4:
                xs.extend([float(segment[0]), float(segment[1])])
                ys.extend([float(segment[2]), float(segment[3])])
        if xs and ys:
            return [min(xs), max(xs), min(ys), max(ys)]

    if "width" in map_data and "height" in map_data:
        return [0.0, float(map_data["width"]), 0.0, float(map_data["height"])]

    if "obstacles" in map_data and isinstance(map_data["obstacles"], list):
        xs = []
        ys = []
        for obs in map_data["obstacles"]:
            for pt in obs.get("vertices", []):
                xs.append(float(pt[0]))
                ys.append(float(pt[1]))
        if xs and ys:
            return [min(xs), max(xs), min(ys), max(ys)]

    return None


def _parse_bounds_value(bounds: Any) -> tuple[float, float, float, float]:
    """Parse a dictionary or 4-element sequence into 4 boundary floats.

    Returns:
        Tuple of (min_x, max_x, min_y, max_y).

    Raises:
        ReviewContractsValidationError: If bounds are malformed, non-finite, or inverted.
    """
    if isinstance(bounds, dict):
        try:
            min_x = float(bounds["min_x"])
            max_x = float(bounds["max_x"])
            min_y = float(bounds["min_y"])
            max_y = float(bounds["max_y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewContractsValidationError(
                [f"invalid geometry_bounds dict: {bounds!r}"]
            ) from exc
    elif isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        try:
            min_x = float(bounds[0])
            max_x = float(bounds[1])
            min_y = float(bounds[2])
            max_y = float(bounds[3])
        except (TypeError, ValueError) as exc:
            raise ReviewContractsValidationError(
                [f"invalid geometry_bounds sequence: {bounds!r}"]
            ) from exc
    else:
        raise ReviewContractsValidationError([f"unrecognized geometry_bounds format: {bounds!r}"])

    for val, name in ((min_x, "min_x"), (max_x, "max_x"), (min_y, "min_y"), (max_y, "max_y")):
        if not math.isfinite(val):
            raise ReviewContractsValidationError([f"geometry bound {name} must be finite: {val!r}"])

    if max_x < min_x or max_y < min_y:
        raise ReviewContractsValidationError(
            [f"inverted geometry bounds: x=({min_x}, {max_x}), y=({min_y}, {max_y})"]
        )

    return min_x, max_x, min_y, max_y


def _extract_geometry_bounds(
    source: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    """Extract and validate scenario geometry bounds (min_x, max_x, min_y, max_y).

    Returns:
        Tuple of (min_x, max_x, min_y, max_y).

    Raises:
        ReviewContractsValidationError: If bounds are missing or invalid.
    """
    bounds = config.get("geometry_bounds")
    if bounds is None and source is not None:
        bounds = source.get("geometry_bounds")
        if bounds is None and "map" in source and isinstance(source["map"], dict):
            bounds = _bounds_from_map(source["map"])

    if bounds is None:
        raise ReviewContractsValidationError(
            ["scenario geometry bounds are required to generate camera specifications"]
        )

    return _parse_bounds_value(bounds)


def _trajectory_from_frames(frames: list[Any]) -> list[dict[str, Any]]:
    """Extract robot/actor positions from frame records.

    Returns:
        List of parsed frame dictionaries.
    """
    parsed_frames: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        t_s = float(frame.get("t_s", idx * 0.1))
        timestep = int(frame.get("timestep", idx))
        pos = None
        if "robot" in frame and isinstance(frame["robot"], dict):
            pos = frame["robot"].get("position")
        elif "robot_pos" in frame:
            pos = frame["robot_pos"]
        elif "position" in frame:
            pos = frame["position"]
        if pos is not None and isinstance(pos, (list, tuple)) and len(pos) >= 2:
            parsed_frames.append(
                {
                    "timestep": timestep,
                    "t_s": t_s,
                    "position": [float(pos[0]), float(pos[1])],
                }
            )
    return parsed_frames


def _parse_raw_trajectory(raw_traj: list[Any]) -> list[dict[str, Any]]:
    """Parse and validate raw trajectory items.

    Returns:
        List of trajectory dictionaries.

    Raises:
        ReviewContractsValidationError: On non-finite values.
    """
    parsed: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_traj):
        if isinstance(item, dict):
            t_s = float(item.get("t_s", idx * 0.1))
            timestep = int(item.get("timestep", idx))
            pos = item.get("position", item.get("pos"))
            if pos is None or not isinstance(pos, (list, tuple)) or len(pos) < 2:
                continue
            px, py = float(pos[0]), float(pos[1])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            t_s = float(idx * 0.1)
            timestep = idx
            px, py = float(item[0]), float(item[1])
        else:
            continue

        if not math.isfinite(t_s) or not math.isfinite(px) or not math.isfinite(py):
            raise ReviewContractsValidationError(
                [f"non-finite trajectory data at index {idx}: t={t_s}, pos=({px}, {py})"]
            )
        parsed.append({"timestep": timestep, "t_s": t_s, "position": [px, py]})
    return parsed


def _extract_trajectory(
    source: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Extract and validate timestamped actor trajectory positions.

    Returns:
        List of dicts: ``{"timestep": int, "t_s": float, "position": [float, float]}``.
    """
    raw_traj = config.get("trajectory")
    if raw_traj is None and source is not None:
        raw_traj = source.get("trajectory")
        if raw_traj is None and "frames" in source and isinstance(source["frames"], list):
            raw_traj = _trajectory_from_frames(source["frames"])

    if not isinstance(raw_traj, list):
        return []
    return _parse_raw_trajectory(raw_traj)


def _extract_events(
    source: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Extract scenario events from config or source.

    Returns:
        List of event dictionaries.

    Raises:
        ReviewContractsValidationError: On non-finite event timestamps or coordinates.
    """
    raw_events = config.get("events")
    if raw_events is None and source is not None:
        raw_events = source.get("events")
    if not isinstance(raw_events, list):
        return []

    events: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_events):
        if not isinstance(item, dict):
            continue
        event_id = str(item.get("event_id", f"evt-{idx:03d}"))
        event_type = str(item.get("type", "event"))
        t_s = float(item.get("t_s", 0.0))
        pos = item.get("position", item.get("pos"))
        position = [float(pos[0]), float(pos[1])] if pos and len(pos) >= 2 else None
        if not math.isfinite(t_s) or (
            position and not all(math.isfinite(coord) for coord in position)
        ):
            raise ReviewContractsValidationError(
                [f"non-finite event values in event {event_id}: t={t_s}, pos={position}"]
            )
        events.append(
            {
                "event_id": event_id,
                "type": event_type,
                "t_s": t_s,
                "position": position,
            }
        )
    return events


def compute_full_scene_camera(
    bounds: tuple[float, float, float, float],
    aspect_ratio: float,
    *,
    margin_m: float = DEFAULT_MARGIN_M,
) -> dict[str, Any]:
    """Compute the full-scene static camera parameters given geometry and aspect ratio.

    Returns:
        Dictionary containing static camera viewport, center, and extents.
    """
    min_x, max_x, min_y, max_y = bounds
    geo_w = max_x - min_x
    geo_h = max_y - min_y

    content_w = max(geo_w + 2.0 * margin_m, 1.0)
    content_h = max(geo_h + 2.0 * margin_m, 1.0)

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    if content_w / content_h >= aspect_ratio:
        view_w = content_w
        view_h = content_w / aspect_ratio
    else:
        view_h = content_h
        view_w = content_h * aspect_ratio

    return {
        "kind": "orthographic",
        "elevation_deg": 90.0,
        "azimuth_deg": 0.0,
        "center": [center_x, center_y],
        "extents_m": [view_w, view_h],
        "view_bounds": {
            "min_x": center_x - view_w / 2.0,
            "max_x": center_x + view_w / 2.0,
            "min_y": center_y - view_h / 2.0,
            "max_y": center_y + view_h / 2.0,
        },
        "margin_m": margin_m,
        "is_cropped": False,
    }


def compute_follow_camera_track(
    trajectory: list[dict[str, Any]],
    geometry_bounds: tuple[float, float, float, float],
    aspect_ratio: float,
    *,
    context_retention_m: float = DEFAULT_CONTEXT_RETENTION_M,
    margin_m: float = DEFAULT_MARGIN_M,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Compute deterministic follow-camera parameters and per-step track samples.

    Returns:
        Tuple of (camera_spec, track_samples, is_cropped).
    """
    min_x, max_x, min_y, max_y = geometry_bounds
    geo_w = max_x - min_x
    geo_h = max_y - min_y

    req_w = 2.0 * context_retention_m
    req_h = req_w / aspect_ratio
    if req_h < 2.0 * context_retention_m:
        req_h = 2.0 * context_retention_m
        req_w = req_h * aspect_ratio

    view_w = min(req_w, max(geo_w + 2.0 * margin_m, req_w))
    view_h = min(req_h, max(geo_h + 2.0 * margin_m, req_h))

    is_cropped = (view_w < geo_w) or (view_h < geo_h)

    track_samples: list[dict[str, Any]] = []
    for step in trajectory:
        px, py = step["position"]
        if geo_w > view_w:
            cx = min(max(px, min_x + view_w / 2.0), max_x - view_w / 2.0)
        else:
            cx = (min_x + max_x) / 2.0

        if geo_h > view_h:
            cy = min(max(py, min_y + view_h / 2.0), max_y - view_h / 2.0)
        else:
            cy = (min_y + max_y) / 2.0

        sample_cropped = (
            (cx - view_w / 2.0 > min_x)
            or (cx + view_w / 2.0 < max_x)
            or (cy - view_h / 2.0 > min_y)
            or (cy + view_h / 2.0 < max_y)
        )

        track_samples.append(
            {
                "timestep": step["timestep"],
                "t_s": step["t_s"],
                "center": [cx, cy],
                "extents_m": [view_w, view_h],
                "actor_position": [px, py],
                "is_cropped": sample_cropped,
            }
        )

    spec = {
        "kind": "orthographic",
        "mode": "follow",
        "elevation_deg": 90.0,
        "azimuth_deg": 0.0,
        "context_retention_m": context_retention_m,
        "extents_m": [view_w, view_h],
        "sample_count": len(track_samples),
        "is_cropped": is_cropped,
    }
    return spec, track_samples, is_cropped


def compute_event_camera(
    events: list[dict[str, Any]],
    geometry_bounds: tuple[float, float, float, float],
    aspect_ratio: float,
    *,
    context_retention_m: float = DEFAULT_CONTEXT_RETENTION_M,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Compute event-focused camera specifications for scenario keyframes.

    Returns:
        Tuple of (camera_spec, event_keyframes, is_cropped).
    """
    min_x, max_x, min_y, max_y = geometry_bounds
    geo_w = max_x - min_x
    geo_h = max_y - min_y

    view_w = 2.0 * context_retention_m
    view_h = view_w / aspect_ratio
    is_cropped = (view_w < geo_w) or (view_h < geo_h)

    event_keyframes: list[dict[str, Any]] = []
    for evt in events:
        pos = evt.get("position")
        if pos:
            px, py = pos
            cx = (
                min(max(px, min_x + view_w / 2.0), max_x - view_w / 2.0)
                if geo_w > view_w
                else (min_x + max_x) / 2.0
            )
            cy = (
                min(max(py, min_y + view_h / 2.0), max_y - view_h / 2.0)
                if geo_h > view_h
                else (min_y + max_y) / 2.0
            )
        else:
            cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

        event_keyframes.append(
            {
                "event_id": evt["event_id"],
                "type": evt["type"],
                "t_s": evt["t_s"],
                "center": [cx, cy],
                "extents_m": [view_w, view_h],
                "is_cropped": is_cropped,
            }
        )

    spec = {
        "kind": "orthographic",
        "mode": "event",
        "elevation_deg": 90.0,
        "azimuth_deg": 0.0,
        "context_retention_m": context_retention_m,
        "extents_m": [view_w, view_h],
        "event_count": len(event_keyframes),
        "is_cropped": is_cropped,
    }
    return spec, event_keyframes, is_cropped


def _load_sources(
    request: ComponentRequest, root: Path
) -> tuple[dict[str, Any] | None, dict[str, str], list[dict[str, Any]]]:
    """Read, verify, and parse declared source payloads.

    Returns:
        Tuple of (source_payload, source_integrity, diagnostics).
    """
    source_payload: dict[str, Any] | None = None
    source_integrity: dict[str, str] = {}
    diagnostics: list[dict[str, Any]] = []

    for source_ref in request.sources:
        source_path = root / source_ref.uri
        if not source_path.exists():
            diagnostics.append(
                {
                    "artifact_id": source_ref.artifact_id,
                    "reason_code": "source_missing",
                    "detail": f"source path does not exist: {source_ref.uri}",
                }
            )
            continue
        try:
            raw_bytes = source_path.read_bytes()
        except OSError as exc:
            diagnostics.append(
                {
                    "artifact_id": source_ref.artifact_id,
                    "reason_code": "source_unreadable",
                    "detail": f"cannot read source: {exc}",
                }
            )
            continue

        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if source_ref.sha256 and source_ref.sha256.lower() != sha256.lower():
            source_integrity[source_ref.artifact_id] = "mismatch"
            diagnostics.append(
                {
                    "artifact_id": source_ref.artifact_id,
                    "reason_code": "source_integrity_mismatch",
                    "detail": f"source SHA-256 mismatch for {source_ref.artifact_id}",
                }
            )
            continue
        source_integrity[source_ref.artifact_id] = "match"

        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
            if isinstance(parsed, dict) and source_payload is None:
                source_payload = parsed
        except (json.JSONDecodeError, UnicodeDecodeError):
            diagnostics.append(
                {
                    "artifact_id": source_ref.artifact_id,
                    "reason_code": "source_malformed_json",
                    "detail": f"source {source_ref.artifact_id} is not valid JSON",
                }
            )

    return source_payload, source_integrity, diagnostics


def _build_camera_and_layout(
    geometry_bounds: tuple[float, float, float, float],
    trajectory: list[dict[str, Any]],
    events: list[dict[str, Any]],
    presentation_cfg: dict[str, Any],
    camera_mode: str,
    margin_m: float,
    context_m: float,
    aspect_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    """Assemble cameras, tracks, layout, and disclosures.

    Returns:
        Tuple of (cameras, tracks, layout, disclosures).
    """
    full_scene_cam = compute_full_scene_camera(geometry_bounds, aspect_ratio, margin_m=margin_m)
    cameras: dict[str, Any] = {"full_scene": full_scene_cam}
    tracks: dict[str, Any] = {}
    is_any_cropped = False

    if trajectory:
        follow_spec, follow_track, follow_cropped = compute_follow_camera_track(
            trajectory,
            geometry_bounds,
            aspect_ratio,
            context_retention_m=context_m,
            margin_m=margin_m,
        )
        cameras["follow"] = follow_spec
        tracks["follow"] = follow_track
        if camera_mode == "follow" and follow_cropped:
            is_any_cropped = True

    if events:
        event_spec, event_keyframes, event_cropped = compute_event_camera(
            events, geometry_bounds, aspect_ratio, context_retention_m=context_m
        )
        cameras["event"] = event_spec
        tracks["event"] = event_keyframes
        if camera_mode == "event" and event_cropped:
            is_any_cropped = True

    layout: dict[str, Any] = {
        "viewport": {
            "width": int(presentation_cfg["width"]),
            "height": int(presentation_cfg["height"]),
            "aspect_ratio": aspect_ratio,
        },
        "inset": {
            "enabled": is_any_cropped,
            "position": "top-right",
            "width_fraction": 0.25,
            "height_fraction": 0.25,
            "aspect_ratio": aspect_ratio,
            "reference_bounds": {
                "min_x": geometry_bounds[0],
                "max_x": geometry_bounds[1],
                "min_y": geometry_bounds[2],
                "max_y": geometry_bounds[3],
            },
            "disclosed": True,
        },
    }

    disclosures = [
        "Simulation and presentation measurements remain in world units (meters, seconds, radians). No metrics are derived from screen pixels.",
        "Camera coordinates are finite and bounded to the specified scenario geometry.",
    ]
    if is_any_cropped:
        disclosures.append(
            "Cropped content is disclosed with full-scene reference geometry and an overview inset."
        )
    if "follow" in cameras:
        disclosures.append(
            f"Follow camera track uses deterministic interpolation with {context_m:g} m context retention."
        )

    return cameras, tracks, layout, disclosures


def _write_artifacts(
    output_dir: Path,
    request: ComponentRequest,
    camera_layout_spec: dict[str, Any],
    visualization_spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Persist generated camera and visualization spec JSON artifacts.

    Returns:
        List of generated artifact records.
    """
    artifacts: list[dict[str, Any]] = []
    spec_filename = f"{CAMERA_LAYOUT_SPEC_SCHEMA_VERSION}.json"
    spec_sha256 = _write_json(output_dir / spec_filename, camera_layout_spec)
    artifacts.append(
        {
            "artifact_id": spec_filename,
            "uri": str(Path(request.output_directory) / spec_filename),
            "sha256": spec_sha256,
        }
    )

    vis_filename = f"{VISUALIZATION_SPEC_SCHEMA_VERSION}.json"
    vis_sha256 = _write_json(output_dir / vis_filename, visualization_spec)
    artifacts.append(
        {
            "artifact_id": vis_filename,
            "uri": str(Path(request.output_directory) / vis_filename),
            "sha256": vis_sha256,
        }
    )
    return artifacts


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Execute one review-camera request to generate camera and layout specifications.

    Args:
        request: Validated component request specifying scenario sources and camera configuration.
        base: Base directory under which request paths resolve (defaults to current working directory).

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

    source_payload, source_integrity, diagnostics = _load_sources(request, root)

    presentation_cfg = dict(DEFAULT_PRESENTATION)
    if "presentation" in request.config and isinstance(request.config["presentation"], dict):
        presentation_cfg.update(request.config["presentation"])

    try:
        aspect_ratio = _validate_aspect_ratio(
            presentation_cfg.get("width", 1920), presentation_cfg.get("height", 1080)
        )
        geometry_bounds = _extract_geometry_bounds(source_payload, request.config)
        trajectory = _extract_trajectory(source_payload, request.config)
        events = _extract_events(source_payload, request.config)
    except ReviewContractsValidationError as error:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="failed",
            reason="; ".join(error.errors),
            diagnostics=tuple(diagnostics),
        )

    camera_mode = str(request.config.get("camera_mode", "full_scene")).lower().replace("-", "_")
    requires_follow = "follow-camera" in request.required_capabilities or camera_mode == "follow"
    requires_event = "event-camera" in request.required_capabilities or camera_mode == "event"

    if requires_follow and not trajectory:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason="follow mode requires stated trajectory data",
            diagnostics=tuple(diagnostics),
        )

    if requires_event and not events and not trajectory:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason="event mode requires stated trajectory or event data",
            diagnostics=tuple(diagnostics),
        )

    margin_m = float(request.config.get("margin_m", DEFAULT_MARGIN_M))
    context_m = float(request.config.get("context_retention_m", DEFAULT_CONTEXT_RETENTION_M))

    cameras, tracks, layout, disclosures = _build_camera_and_layout(
        geometry_bounds,
        trajectory,
        events,
        presentation_cfg,
        camera_mode,
        margin_m,
        context_m,
        aspect_ratio,
    )

    spec_id = f"cam-spec-{request.request_id}"
    camera_layout_spec = {
        "schema_version": CAMERA_LAYOUT_SPEC_SCHEMA_VERSION,
        "spec_id": spec_id,
        "request_id": request.request_id,
        "active_camera_mode": camera_mode,
        "presentation": presentation_cfg,
        "geometry_bounds": {
            "min_x": geometry_bounds[0],
            "max_x": geometry_bounds[1],
            "min_y": geometry_bounds[2],
            "max_y": geometry_bounds[3],
            "width": geometry_bounds[1] - geometry_bounds[0],
            "height": geometry_bounds[3] - geometry_bounds[2],
        },
        "cameras": cameras,
        "layout": layout,
        "tracks": tracks,
        "disclosures": disclosures,
    }

    vis_sources = [{"artifact_id": s.artifact_id} for s in request.sources] or [
        {"artifact_id": "inline-geometry"}
    ]
    visualization_spec = {
        "schema_version": VISUALIZATION_SPEC_SCHEMA_VERSION,
        "spec_id": f"vis-{spec_id}",
        "sources": vis_sources,
        "camera": cameras.get(camera_mode, cameras["full_scene"]),
        "layout": layout,
        "presentation": {
            "width": int(presentation_cfg["width"]),
            "height": int(presentation_cfg["height"]),
            "fps": float(presentation_cfg.get("fps", 30.0)),
            "speed": float(presentation_cfg.get("speed", 1.0)),
        },
        "units": "meters, seconds, radians (no metrics derived from screen pixels)",
    }

    artifacts = _write_artifacts(output_dir, request, camera_layout_spec, visualization_spec)

    status = "partial" if diagnostics else "complete"
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status=status,
        artifacts=tuple(artifacts),
        diagnostics=tuple(diagnostics),
        provenance={
            "output_directory": request.output_directory,
            "component_version": COMPONENT_VERSION,
            "source_integrity": source_integrity,
            "geometry_bounds": {
                "min_x": geometry_bounds[0],
                "max_x": geometry_bounds[1],
                "min_y": geometry_bounds[2],
                "max_y": geometry_bounds[3],
            },
            "active_mode": camera_mode,
            "pixel_metric_free": True,
            "admission": "diagnostic_only",
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for review_camera.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Generate camera and layout specifications for scenario presentations."
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
    """CLI entry point for the review-camera component.

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
    "CAMERA_LAYOUT_SPEC_SCHEMA_VERSION",
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "DEFAULT_CONTEXT_RETENTION_M",
    "DEFAULT_MARGIN_M",
    "DEFAULT_PRESENTATION",
    "DESCRIPTOR",
    "OPTIONAL_CAPABILITIES",
    "OUTPUT_TYPES",
    "REQUIRED_CAPABILITIES",
    "TEST_PRESENTATION",
    "VISUALIZATION_SPEC_SCHEMA_VERSION",
    "compute_event_camera",
    "compute_follow_camera_track",
    "compute_full_scene_camera",
    "descriptor_document",
    "main",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
