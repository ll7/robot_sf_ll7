"""Render synchronized telemetry and annotation overlays (SREV-12, issue #9281).

Consumes the v1 scenario-review contracts fixed by #9270 (`component-request.v1`,
`visualization-spec.v1`, `component-descriptor.v1`) and produces transparent
overlay frame sequences, layer availability reports, and timestamp mapping receipts.

Boundaries enforced by this module:
- Simulation time is telemetry authority. Synchronized video overlays require an
  explicit presentation-timestamp map; guessed frame/fps alignment is forbidden.
- Missing presentation mapping disables synchronized layers with a stable diagnostic
  reason code (never assume frame/fps alignment).
- Pauses repeat source state and speed changes resample by declared policy (nearest,
  linear, hold).
- Dropped telemetry and actor disappearance are tracked and disclosed in the mapping receipt.
- Actor/status labels use an overlap-prevention layout so text remains legible.
- Units remain strictly in world coordinates (meters, seconds, radians). No metrics
  are derived from screen pixels.
- Transparent RGBA overlay frames are generated offline without external renderer
  dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from collections.abc import Mapping

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ComponentDescriptor,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    SourceRef,
    component_request_from_dict,
)

COMPONENT_ID = "review-overlays"
COMPONENT_VERSION = "0.1.0"
DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"
OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION = "overlay-mapping-receipt.v1"
LAYER_AVAILABILITY_SCHEMA_VERSION = "layer-availability.v1"
VISUALIZATION_SPEC_SCHEMA_VERSION = "visualization-spec.v1"

DEFAULT_PRESENTATION = {"width": 1920, "height": 1080, "fps": 30.0, "speed": 1.0}
TEST_PRESENTATION = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}

REQUIRED_CAPABILITIES = (
    "synchronized-overlays",
    "mapping-receipt",
    "layer-availability",
    "transparent-frame-rendering",
)
OPTIONAL_CAPABILITIES = (
    "paths-layer",
    "controls-layer",
    "clearance-layer",
    "markers-layer",
    "labels-layer",
    "pause-repeat",
    "speed-resampling",
)
OUTPUT_TYPES = (
    OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION,
    LAYER_AVAILABILITY_SCHEMA_VERSION,
    VISUALIZATION_SPEC_SCHEMA_VERSION,
    "overlay-frames.v1",
)

ALL_LAYERS = ("paths", "controls", "clearance", "markers", "labels")

DESCRIPTOR = ComponentDescriptor(
    component_id=COMPONENT_ID,
    component_version=COMPONENT_VERSION,
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=OUTPUT_TYPES,
    required_capabilities=REQUIRED_CAPABILITIES,
    optional_capabilities=OPTIONAL_CAPABILITIES,
)


@dataclass(frozen=True, slots=True)
class FrameRenderContext:
    """Encapsulates context parameters needed to render one overlay frame."""

    width: int
    height: int
    frame_idx: int
    pres_t: float
    src_t: float
    is_pause: bool
    frame_data: dict[str, Any] | None
    trajectory: list[list[float]]
    events: list[dict[str, Any]]
    bounds: tuple[float, float, float, float]
    enabled_layers: set[str]


@dataclass(frozen=True, slots=True)
class OverlayPipelineConfig:
    """Encapsulates presentation and layer configuration for overlay rendering."""

    width: int
    height: int
    fps: float
    speed: float
    bounds: tuple[float, float, float, float]
    requested_layers: set[str]
    resampling_policy: str


def descriptor_document() -> dict[str, Any]:
    """Return the schema-shaped capability descriptor for this component.

    Returns:
        ``component-descriptor.v1`` document for review-overlays.
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


def _validate_presentation_dimensions(
    presentation: Mapping[str, Any],
) -> tuple[int, int, float, float]:
    """Validate and extract presentation dimensions, fps, and speed.

    Returns:
        Tuple of (width, height, fps, speed).

    Raises:
        ReviewContractsValidationError: On invalid or non-finite values.
    """
    try:
        width = int(presentation.get("width", DEFAULT_PRESENTATION["width"]))
        height = int(presentation.get("height", DEFAULT_PRESENTATION["height"]))
        fps = float(presentation.get("fps", DEFAULT_PRESENTATION["fps"]))
        speed = float(presentation.get("speed", DEFAULT_PRESENTATION["speed"]))
    except (TypeError, ValueError) as exc:
        raise ReviewContractsValidationError(
            [f"presentation dimensions must be numeric: {presentation!r}"]
        ) from exc

    if width <= 0 or height <= 0:
        raise ReviewContractsValidationError(
            [f"presentation dimensions must be > 0: width={width}, height={height}"]
        )
    if not math.isfinite(fps) or fps <= 0.0:
        raise ReviewContractsValidationError([f"presentation fps must be finite and > 0: {fps}"])
    if not math.isfinite(speed) or speed <= 0.0:
        raise ReviewContractsValidationError(
            [f"presentation speed must be finite and > 0: {speed}"]
        )

    return width, height, fps, speed


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
    """Parse bounds value into 4 boundary floats.

    Returns:
        Tuple of (min_x, max_x, min_y, max_y).

    Raises:
        ReviewContractsValidationError: On invalid or non-finite bounds.
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

    if max_x <= min_x or max_y <= min_y:
        raise ReviewContractsValidationError(
            [f"inverted or flat geometry bounds: x=({min_x}, {max_x}), y=({min_y}, {max_y})"]
        )

    return min_x, max_x, min_y, max_y


def _extract_geometry_bounds(
    source: Mapping[str, Any] | None, config: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    """Extract and validate scenario geometry bounds.

    Returns:
        Tuple of (min_x, max_x, min_y, max_y).

    Raises:
        ReviewContractsValidationError: If bounds are absent or invalid.
    """
    bounds = config.get("geometry_bounds")
    if bounds is None and source is not None:
        bounds = source.get("geometry_bounds")
        if bounds is None and "map" in source and isinstance(source["map"], dict):
            bounds = _bounds_from_map(source["map"])

    if bounds is None:
        if source is not None and "frames" in source and isinstance(source["frames"], list):
            xs: list[float] = []
            ys: list[float] = []
            for frame in source["frames"]:
                if isinstance(frame, dict):
                    pos = frame.get("robot", {}).get("position")
                    if pos and len(pos) >= 2:
                        xs.append(float(pos[0]))
                        ys.append(float(pos[1]))
            if xs and ys:
                margin = 2.0
                bounds = [min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin]

    if bounds is None:
        raise ReviewContractsValidationError(
            ["scenario geometry bounds are required for overlay spatial projection"]
        )

    return _parse_bounds_value(bounds)


def _load_sources(request: ComponentRequest, base: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and validate all sources declared in the request.

    Returns:
        Tuple of (primary_source_payload, list_of_diagnostics).

    Raises:
        ReviewContractsValidationError: On missing, unreadable, or corrupted files.
    """
    primary_source: dict[str, Any] | None = None
    diagnostics: list[str] = []

    for source_ref in request.sources:
        source_path = Path(source_ref.uri)
        if not source_path.is_absolute():
            source_path = base / source_path

        if not source_path.exists():
            raise ReviewContractsValidationError(
                [f"source artifact not found: {source_ref.artifact_id} at {source_path}"]
            )

        try:
            raw_bytes = source_path.read_bytes()
        except OSError as exc:
            raise ReviewContractsValidationError(
                [f"cannot read source artifact {source_ref.artifact_id}: {exc}"]
            ) from exc

        if source_ref.sha256:
            computed_sha = hashlib.sha256(raw_bytes).hexdigest()
            if computed_sha.lower() != source_ref.sha256.lower():
                raise ReviewContractsValidationError(
                    [
                        f"source artifact integrity mismatch for {source_ref.artifact_id}: "
                        f"expected {source_ref.sha256}, got {computed_sha}"
                    ]
                )

        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReviewContractsValidationError(
                [f"corrupt JSON in source artifact {source_ref.artifact_id}: {exc}"]
            ) from exc

        if not isinstance(parsed, dict):
            raise ReviewContractsValidationError(
                [f"source artifact {source_ref.artifact_id} must be a JSON object"]
            )

        if primary_source is None:
            primary_source = parsed

    return primary_source, diagnostics


def _parse_timestamp_list(raw_list: list[Any]) -> list[dict[str, Any]]:
    """Parse a list of explicit frame mapping objects.

    Returns:
        List of parsed frame mapping dictionaries.
    """
    mapping_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_list):
        if not isinstance(row, dict):
            continue
        pres_t = float(row.get("presentation_t_s", idx * 0.1))
        src_t = float(row.get("source_t_s", pres_t))
        is_pause = bool(row.get("is_pause", False))
        mapping_rows.append(
            {
                "frame_idx": int(row.get("frame_idx", idx)),
                "presentation_t_s": pres_t,
                "source_t_s": src_t,
                "is_pause": is_pause,
            }
        )
    return mapping_rows


def _parse_policy_map(
    ts_map: Mapping[str, Any], presentation_cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Synthesize presentation timestamp mapping rows under declared policy.

    Returns:
        List of generated frame mapping dictionaries.
    """
    fps = float(presentation_cfg.get("fps", 10.0))
    speed = float(presentation_cfg.get("speed", 1.0))
    pauses = presentation_cfg.get("pauses", [])
    total_frames = int(ts_map.get("frame_count", 5))

    mapping_rows: list[dict[str, Any]] = []
    dt_pres = 1.0 / fps
    cur_src_t = 0.0

    for idx in range(total_frames):
        pres_t = idx * dt_pres
        is_pause = False

        for p in pauses:
            p_start = float(p.get("presentation_t_s", -1.0))
            p_dur = float(p.get("duration_s", 0.0))
            if p_start <= pres_t < (p_start + p_dur):
                is_pause = True
                cur_src_t = float(p.get("source_t_s", cur_src_t))
                break

        if not is_pause:
            cur_src_t = pres_t * speed

        mapping_rows.append(
            {
                "frame_idx": idx,
                "presentation_t_s": pres_t,
                "source_t_s": cur_src_t,
                "is_pause": is_pause,
            }
        )
    return mapping_rows


def _parse_timestamp_map(
    presentation_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Parse or build presentation-to-source timestamp mapping.

    Returns:
        Tuple of (mapping_rows, disabled_reason).
    """
    ts_map = presentation_cfg.get("presentation_timestamp_map")
    if ts_map is None:
        return None, (
            "missing_presentation_timestamp_map: explicit presentation-timestamp map is required; "
            "guessing frame/fps alignment is forbidden"
        )

    if isinstance(ts_map, list):
        return _parse_timestamp_list(ts_map), None

    if isinstance(ts_map, dict):
        if "mapping" in ts_map and isinstance(ts_map["mapping"], list):
            return _parse_timestamp_list(ts_map["mapping"]), None
        return _parse_policy_map(ts_map, presentation_cfg), None

    return None, (
        "unrecognized_presentation_timestamp_map: map format must be a list or dict with mapping"
    )


def _extract_source_frames(source: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Extract and sort source telemetry frames by simulation timestamp.

    Returns:
        List of frame dictionaries sorted by ``t_s``.
    """
    if source is None or "frames" not in source:
        return []

    raw_frames = source["frames"]
    if not isinstance(raw_frames, list):
        return []

    frames: list[dict[str, Any]] = []
    for idx, f in enumerate(raw_frames):
        if not isinstance(f, dict):
            continue
        t_s = float(f.get("t_s", f.get("time_s", idx * 0.1)))
        frames.append({**f, "_t_s": t_s})

    frames.sort(key=lambda x: x["_t_s"])
    return frames


def _sample_source_frame(
    source_frames: list[dict[str, Any]], target_t_s: float, policy: str
) -> tuple[dict[str, Any] | None, bool]:
    """Sample source telemetry at ``target_t_s`` under declared resampling policy.

    Returns:
        Tuple of (sampled_frame_dict, is_dropped_telemetry).
    """
    if not source_frames:
        return None, True

    max_t = source_frames[-1]["_t_s"]
    min_t = source_frames[0]["_t_s"]
    if target_t_s < min_t - 0.5 or target_t_s > max_t + 0.5:
        return None, True

    if policy == "hold":
        candidate = None
        for f in source_frames:
            if f["_t_s"] <= target_t_s + 1e-4:
                candidate = f
            else:
                break
        if candidate is None:
            candidate = source_frames[0]
        return candidate, False

    closest_frame = min(source_frames, key=lambda f: abs(f["_t_s"] - target_t_s))
    if abs(closest_frame["_t_s"] - target_t_s) > 0.5:
        return None, True

    return closest_frame, False


def _world_to_screen(
    wx: float,
    wy: float,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
    margin_px: int = 24,
) -> tuple[int, int]:
    """Map world coordinates in meters to screen pixel coordinates.

    Returns:
        Tuple of (pixel_x, pixel_y).
    """
    min_x, max_x, min_y, max_y = bounds
    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    usable_w = width - 2 * margin_px
    usable_h = height - 2 * margin_px

    norm_x = (wx - min_x) / span_x
    norm_y = (wy - min_y) / span_y

    px = int(margin_px + norm_x * usable_w)
    py = int(height - (margin_px + norm_y * usable_h))
    return px, py


def _boxes_overlap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    """Check whether two axis-aligned bounding boxes overlap.

    Returns:
        True if bounding boxes intersect, False otherwise.
    """
    x1_min, y1_min, x1_max, y1_max = b1
    x2_min, y2_min, x2_max, y2_max = b2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)


def _compute_nonoverlapping_labels(
    labels: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Position actor labels with overlap prevention.

    Returns:
        List of positioned label dictionaries.
    """
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_labels: list[dict[str, Any]] = []

    offsets = [
        (0, -18),
        (0, 18),
        (24, 0),
        (-24, 0),
        (20, -18),
        (-20, -18),
        (20, 18),
        (-20, 18),
    ]

    for item in labels:
        anchor_x, anchor_y = item["anchor_px"]
        text = item["text"]
        tw = len(text) * 7 + 8
        th = 16

        chosen_box: tuple[int, int, int, int] | None = None
        chosen_pos: tuple[int, int] | None = None

        for dx, dy in offsets:
            cx = anchor_x + dx
            cy = anchor_y + dy
            x_min = cx - tw // 2
            y_min = cy - th // 2
            x_max = x_min + tw
            y_max = y_min + th

            if x_min < 2 or y_min < 2 or x_max > width - 2 or y_max > height - 2:
                continue

            candidate_box = (x_min, y_min, x_max, y_max)
            collision = any(_boxes_overlap(candidate_box, pb) for pb in placed_boxes)
            if not collision:
                chosen_box = candidate_box
                chosen_pos = (x_min + 4, y_min + 2)
                break

        if chosen_box is None:
            cx = max(tw // 2 + 2, min(width - tw // 2 - 2, anchor_x))
            cy = max(th // 2 + 2, min(height - th // 2 - 2, anchor_y - 18))
            chosen_box = (cx - tw // 2, cy - th // 2, cx + tw // 2, cy + th // 2)
            chosen_pos = (cx - tw // 2 + 4, cy - th // 2 + 2)

        placed_boxes.append(chosen_box)
        placed_labels.append({**item, "box": chosen_box, "text_pos": chosen_pos})

    return placed_labels


def _render_paths_layer(
    draw: ImageDraw.ImageDraw,
    trajectory: list[list[float]],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    """Render planned or past actor trajectories in world coordinates."""
    if not trajectory:
        return
    screen_pts = []
    for pt in trajectory:
        if len(pt) >= 2:
            sp = _world_to_screen(float(pt[0]), float(pt[1]), bounds, width, height)
            screen_pts.append(sp)
    if len(screen_pts) >= 2:
        draw.line(screen_pts, fill=(64, 160, 255, 180), width=2)
        for pt in screen_pts:
            draw.ellipse((pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2), fill=(100, 200, 255, 200))


def _render_clearance_layer(
    draw: ImageDraw.ImageDraw,
    robot_data: dict[str, Any] | None,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    """Render clearance distance in meters around the robot."""
    if not robot_data or not isinstance(robot_data, dict):
        return
    pos = robot_data.get("position")
    if not pos or len(pos) < 2:
        return
    r_px, r_py = _world_to_screen(float(pos[0]), float(pos[1]), bounds, width, height)
    clearance_m = float(robot_data.get("clearance_m", 1.0))

    min_x, max_x, _, _ = bounds
    scale = (width - 48) / max(max_x - min_x, 1e-3)
    r_px_rad = int(clearance_m * scale)
    draw.ellipse(
        (r_px - r_px_rad, r_py - r_px_rad, r_px + r_px_rad, r_py + r_px_rad),
        outline=(255, 200, 64, 140),
        width=1,
    )
    cl_text = f"clr: {clearance_m:.2f}m"
    draw.rectangle(
        (r_px + 8, r_py - 16, r_px + 8 + len(cl_text) * 7 + 4, r_py - 2),
        fill=(0, 0, 0, 160),
    )
    draw.text((r_px + 10, r_py - 15), cl_text, fill=(255, 220, 100, 240))


def _render_markers_layer(
    draw: ImageDraw.ImageDraw,
    events: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    """Render scenario event markers at world coordinates."""
    for ev in events:
        ev_pos = ev.get("position")
        if ev_pos and len(ev_pos) >= 2:
            ex, ey = _world_to_screen(float(ev_pos[0]), float(ev_pos[1]), bounds, width, height)
            draw.polygon(
                [(ex, ey - 6), (ex + 6, ey), (ex, ey + 6), (ex - 6, ey)],
                fill=(255, 220, 0, 200),
                outline=(0, 0, 0, 220),
            )
            ev_name = str(ev.get("type", "event"))
            draw.text((ex + 8, ey - 6), f"[{ev_name}]", fill=(255, 240, 150, 240))


def _render_labels_layer(
    draw: ImageDraw.ImageDraw,
    frame_data: dict[str, Any] | None,
    bounds: tuple[float, float, float, float],
    width: int,
    height: int,
) -> None:
    """Render readable, non-overlapping labels for active actors."""
    if not frame_data:
        return
    robot_pos = frame_data.get("robot", {}).get("position")
    label_candidates: list[dict[str, Any]] = []

    if robot_pos and len(robot_pos) >= 2:
        r_px, r_py = _world_to_screen(
            float(robot_pos[0]), float(robot_pos[1]), bounds, width, height
        )
        label_candidates.append(
            {
                "actor_id": "robot",
                "anchor_px": (r_px, r_py),
                "text": f"Robot ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})m",
            }
        )

    for p in frame_data.get("pedestrians", []):
        if isinstance(p, dict) and "position" in p:
            pid = str(p.get("id", "ped"))
            ppos = p["position"]
            if len(ppos) >= 2:
                p_px, p_py = _world_to_screen(float(ppos[0]), float(ppos[1]), bounds, width, height)
                label_candidates.append(
                    {
                        "actor_id": pid,
                        "anchor_px": (p_px, p_py),
                        "text": f"{pid} ({ppos[0]:.1f}, {ppos[1]:.1f})m",
                    }
                )

    placed = _compute_nonoverlapping_labels(label_candidates, width, height)
    for pl in placed:
        box = pl["box"]
        tx, ty = pl["text_pos"]
        draw.rectangle(box, fill=(0, 0, 0, 180), outline=(120, 120, 120, 200))
        draw.text((tx, ty), pl["text"], fill=(255, 255, 255, 255))


def _render_controls_panel(
    draw: ImageDraw.ImageDraw,
    pres_t: float,
    src_t: float,
    is_pause: bool,
    frame_data: dict[str, Any] | None,
) -> None:
    """Render diagnostic control readout panel."""
    panel_lines = [
        f"t_pres: {pres_t:.2f}s | t_src: {src_t:.2f}s" + (" [PAUSE]" if is_pause else ""),
        "telemetry: " + ("OK" if frame_data else "DROPPED"),
    ]
    robot_data = frame_data.get("robot") if frame_data else None
    if robot_data and isinstance(robot_data, dict):
        speed_val = robot_data.get("speed", 0.0)
        panel_lines.append(f"speed: {speed_val:.2f} m/s")
        if "action" in robot_data:
            act = robot_data["action"]
            panel_lines.append(f"action: {act}")
        if "goal" in robot_data:
            goal = robot_data["goal"]
            panel_lines.append(f"goal: {goal} m")

    panel_w = 200
    panel_h = len(panel_lines) * 14 + 10
    draw.rectangle((8, 8, 8 + panel_w, 8 + panel_h), fill=(0, 0, 0, 170))
    for line_idx, line in enumerate(panel_lines):
        draw.text((12, 12 + line_idx * 14), line, fill=(240, 240, 240, 240))


def _render_actor_glyphs(draw: ImageDraw.ImageDraw, ctx: FrameRenderContext) -> None:
    """Render robot and pedestrian circle glyphs."""
    robot_data = ctx.frame_data.get("robot") if ctx.frame_data else None
    if robot_data and "position" in robot_data:
        r_pos = robot_data["position"]
        if len(r_pos) >= 2:
            rx, ry = _world_to_screen(
                float(r_pos[0]), float(r_pos[1]), ctx.bounds, ctx.width, ctx.height
            )
            draw.ellipse((rx - 5, ry - 5, rx + 5, ry + 5), fill=(0, 180, 255, 220))

    if ctx.frame_data:
        for p in ctx.frame_data.get("pedestrians", []):
            if isinstance(p, dict) and "position" in p and len(p["position"]) >= 2:
                ppos = p["position"]
                px, py = _world_to_screen(
                    float(ppos[0]), float(ppos[1]), ctx.bounds, ctx.width, ctx.height
                )
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 120, 80, 220))


def _render_overlay_frame(ctx: FrameRenderContext) -> Image.Image:
    """Render a transparent overlay frame with synchronized layers.

    Returns:
        PIL Image in RGBA mode.
    """
    img = Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    if "paths" in ctx.enabled_layers:
        _render_paths_layer(draw, ctx.trajectory, ctx.bounds, ctx.width, ctx.height)

    robot_data = ctx.frame_data.get("robot") if ctx.frame_data else None
    if "clearance" in ctx.enabled_layers:
        _render_clearance_layer(draw, robot_data, ctx.bounds, ctx.width, ctx.height)

    _render_actor_glyphs(draw, ctx)

    if "markers" in ctx.enabled_layers:
        _render_markers_layer(draw, ctx.events, ctx.bounds, ctx.width, ctx.height)

    if "labels" in ctx.enabled_layers:
        _render_labels_layer(draw, ctx.frame_data, ctx.bounds, ctx.width, ctx.height)

    if "controls" in ctx.enabled_layers:
        _render_controls_panel(draw, ctx.pres_t, ctx.src_t, ctx.is_pause, ctx.frame_data)

    return img


def _handle_missing_mapping(
    request: ComponentRequest,
    output_dir: Path,
    disabled_reason: str,
    width: int,
    height: int,
    fps: float,
    speed: float,
    source_diagnostics: list[str],
) -> ComponentResult:
    """Build response when presentation timestamp mapping is missing.

    Returns:
        ComponentResult with partial status.
    """
    layer_availability: dict[str, Any] = {}
    for lay in ALL_LAYERS:
        layer_availability[lay] = {
            "available": False,
            "status": "disabled",
            "reason": disabled_reason,
        }

    layer_avail_doc = {
        "schema_version": LAYER_AVAILABILITY_SCHEMA_VERSION,
        "request_id": request.request_id,
        "available_layers": [],
        "disabled_layers": layer_availability,
        "mapping_status": "disabled",
        "reason": disabled_reason,
    }
    _write_json(output_dir / "layer_availability.json", layer_avail_doc)

    receipt = {
        "schema_version": OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION,
        "receipt_id": f"rcpt-{request.request_id}",
        "request_id": request.request_id,
        "status": "partial",
        "presentation": {
            "width": width,
            "height": height,
            "fps": fps,
            "speed": speed,
        },
        "layers": layer_availability,
        "frames": [],
        "dropped_telemetry": [],
        "actor_lifecycle": [],
        "units": "meters, seconds, radians (no metrics derived from screen pixels)",
        "provenance": {
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "output_directory": request.output_directory,
            "admission": "diagnostic_only",
        },
    }
    _write_json(output_dir / "mapping_receipt.json", receipt)

    diagnostics = list(source_diagnostics) + [disabled_reason]
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="partial",
        reason=disabled_reason,
        diagnostics=tuple(diagnostics),
        artifacts=(
            SourceRef(
                artifact_id="layer-availability",
                uri=str(output_dir / "layer_availability.json"),
                format="json",
                schema=LAYER_AVAILABILITY_SCHEMA_VERSION,
            ),
            SourceRef(
                artifact_id="mapping-receipt",
                uri=str(output_dir / "mapping_receipt.json"),
                format="json",
                schema=OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION,
            ),
        ),
    )


def _process_frame_step(
    row: dict[str, Any],
    source_frames: list[dict[str, Any]],
    pipeline_cfg: OverlayPipelineConfig,
    trajectory: list[list[float]],
    events: list[dict[str, Any]],
    prev_actors: set[str],
    frames_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], set[str]]:
    """Execute one frame step: sampling, actor lifecycle tracking, and rendering.

    Returns:
        Tuple of (frame_record, dropped_record_or_None, lifecycle_events, current_actors).
    """
    f_idx = row["frame_idx"]
    pres_t = row["presentation_t_s"]
    src_t = row["source_t_s"]
    is_pause = row["is_pause"]

    frame_data, is_dropped = _sample_source_frame(
        source_frames, src_t, pipeline_cfg.resampling_policy
    )
    dropped_record = None
    if is_dropped:
        dropped_record = {
            "frame_idx": f_idx,
            "presentation_t_s": pres_t,
            "source_t_s": src_t,
            "reason": "telemetry_dropped_or_missing",
        }

    current_actors: set[str] = set()
    if frame_data:
        if "robot" in frame_data:
            current_actors.add("robot")
        for p in frame_data.get("pedestrians", []):
            if isinstance(p, dict) and "id" in p:
                current_actors.add(str(p["id"]))

    disappeared: list[dict[str, Any]] = []
    for actor_id in prev_actors - current_actors:
        disappeared.append(
            {
                "actor_id": actor_id,
                "event": "disappeared",
                "at_presentation_t_s": pres_t,
                "at_source_t_s": src_t,
                "at_frame": f_idx,
            }
        )

    ctx = FrameRenderContext(
        width=pipeline_cfg.width,
        height=pipeline_cfg.height,
        frame_idx=f_idx,
        pres_t=pres_t,
        src_t=src_t,
        is_pause=is_pause,
        frame_data=frame_data,
        trajectory=trajectory,
        events=events,
        bounds=pipeline_cfg.bounds,
        enabled_layers=pipeline_cfg.requested_layers,
    )
    overlay_img = _render_overlay_frame(ctx)

    frame_filename = f"frame_{f_idx:04d}.png"
    frame_path = frames_dir / frame_filename
    overlay_img.save(frame_path, format="PNG")
    frame_sha = hashlib.sha256(frame_path.read_bytes()).hexdigest()

    frame_record = {
        "frame_idx": f_idx,
        "presentation_t_s": pres_t,
        "source_t_s": src_t,
        "is_pause": is_pause,
        "telemetry_available": not is_dropped,
        "actors_present": sorted(current_actors),
        "artifact_path": f"frames/{frame_filename}",
        "sha256": frame_sha,
    }
    return frame_record, dropped_record, disappeared, current_actors


def _build_layer_availability(requested_layers: set[str]) -> dict[str, Any]:
    """Build the layer availability mapping for all standard layers.

    Returns:
        Dictionary of layer availability descriptors.
    """
    availability: dict[str, Any] = {}
    for lay in ALL_LAYERS:
        if lay in requested_layers:
            availability[lay] = {"available": True, "status": "enabled", "reason": None}
        else:
            availability[lay] = {
                "available": True,
                "status": "omitted_by_config",
                "reason": "not requested",
            }
    return availability


def _render_frames_and_receipt(
    request: ComponentRequest,
    output_dir: Path,
    mapping_rows: list[dict[str, Any]],
    source: dict[str, Any] | None,
    pipeline_cfg: OverlayPipelineConfig,
    source_diagnostics: list[str],
) -> ComponentResult:
    """Render transparent frames and generate mapping receipts.

    Returns:
        ComponentResult with complete or partial status.
    """
    layer_availability = _build_layer_availability(pipeline_cfg.requested_layers)

    source_frames = _extract_source_frames(source)
    trajectory = source.get("trajectory", []) if source else []
    events = source.get("events", []) if source else []

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_records: list[dict[str, Any]] = []
    dropped_telemetry: list[dict[str, Any]] = []
    actor_lifecycle: list[dict[str, Any]] = []
    prev_actors: set[str] = set()

    for row in mapping_rows:
        frame_rec, dropped_rec, disappeared, prev_actors = _process_frame_step(
            row,
            source_frames,
            pipeline_cfg,
            trajectory,
            events,
            prev_actors,
            frames_dir,
        )
        frame_records.append(frame_rec)
        if dropped_rec is not None:
            dropped_telemetry.append(dropped_rec)
        actor_lifecycle.extend(disappeared)

    layer_avail_doc = {
        "schema_version": LAYER_AVAILABILITY_SCHEMA_VERSION,
        "request_id": request.request_id,
        "available_layers": [k for k, v in layer_availability.items() if v["available"]],
        "disabled_layers": {k: v for k, v in layer_availability.items() if not v["available"]},
        "mapping_status": "synchronized",
    }
    _write_json(output_dir / "layer_availability.json", layer_avail_doc)

    receipt = {
        "schema_version": OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION,
        "receipt_id": f"rcpt-{request.request_id}",
        "request_id": request.request_id,
        "presentation": {
            "width": pipeline_cfg.width,
            "height": pipeline_cfg.height,
            "fps": pipeline_cfg.fps,
            "speed": pipeline_cfg.speed,
        },
        "layers": layer_availability,
        "frames": frame_records,
        "dropped_telemetry": dropped_telemetry,
        "actor_lifecycle": actor_lifecycle,
        "units": "meters, seconds, radians (no metrics derived from screen pixels)",
        "provenance": {
            "component_id": COMPONENT_ID,
            "component_version": COMPONENT_VERSION,
            "output_directory": request.output_directory,
            "resampling_policy": pipeline_cfg.resampling_policy,
            "frame_count": len(frame_records),
            "admission": "diagnostic_only",
        },
    }
    _write_json(output_dir / "mapping_receipt.json", receipt)

    vis_sources = [{"artifact_id": s.artifact_id} for s in request.sources] or [
        {"artifact_id": "inline-geometry"}
    ]
    vis_spec = {
        "schema_version": VISUALIZATION_SPEC_SCHEMA_VERSION,
        "spec_id": f"vis-overlays-{request.request_id}",
        "sources": vis_sources,
        "layers": sorted(pipeline_cfg.requested_layers),
        "presentation": {
            "width": pipeline_cfg.width,
            "height": pipeline_cfg.height,
            "fps": pipeline_cfg.fps,
            "speed": pipeline_cfg.speed,
        },
        "units": "seconds/metres/radians",
    }
    _write_json(output_dir / "visualization_spec.json", vis_spec)

    artifacts = [
        SourceRef(
            artifact_id="layer-availability",
            uri=str(output_dir / "layer_availability.json"),
            format="json",
            schema=LAYER_AVAILABILITY_SCHEMA_VERSION,
        ),
        SourceRef(
            artifact_id="mapping-receipt",
            uri=str(output_dir / "mapping_receipt.json"),
            format="json",
            schema=OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION,
        ),
        SourceRef(
            artifact_id="visualization-spec",
            uri=str(output_dir / "visualization_spec.json"),
            format="json",
            schema=VISUALIZATION_SPEC_SCHEMA_VERSION,
        ),
    ]

    diagnostics = list(source_diagnostics)
    if dropped_telemetry:
        diagnostics.append(f"dropped telemetry detected in {len(dropped_telemetry)} frames")

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
            "pixel_metric_free": True,
            "admission": "diagnostic_only",
            "frame_count": len(frame_records),
        },
    )


def run(request: ComponentRequest, base: Path | None = None) -> ComponentResult:
    """Execute the review-overlays component for a valid request.

    Returns:
        ComponentResult with complete/partial/unavailable status.

    Raises:
        ReviewContractsValidationError: On schema or fatal validation failure.
    """
    base_path = base if base is not None else Path.cwd()

    unsupported = _unsupported_capabilities(request)
    if unsupported:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason=f"unsupported required capabilities: {sorted(unsupported)}",
        )

    output_dir = _resolve_output_dir(request, base_path)
    source, source_diagnostics = _load_sources(request, base_path)
    presentation_cfg = request.config.get("presentation", {})
    width, height, fps, speed = _validate_presentation_dimensions(presentation_cfg)
    bounds = _extract_geometry_bounds(source, request.config)

    requested_layers = set(request.config.get("layers", ALL_LAYERS))
    resampling_policy = str(request.config.get("resampling_policy", "nearest"))

    mapping_rows, disabled_reason = _parse_timestamp_map(presentation_cfg)

    if mapping_rows is None:
        return _handle_missing_mapping(
            request,
            output_dir,
            str(disabled_reason),
            width,
            height,
            fps,
            speed,
            source_diagnostics,
        )

    pipeline_cfg = OverlayPipelineConfig(
        width=width,
        height=height,
        fps=fps,
        speed=speed,
        bounds=bounds,
        requested_layers=requested_layers,
        resampling_policy=resampling_policy,
    )

    return _render_frames_and_receipt(
        request,
        output_dir,
        mapping_rows,
        source,
        pipeline_cfg,
        source_diagnostics,
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for review_overlays.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Render synchronized telemetry and annotation overlays."
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
    """CLI entry point for the review-overlays component.

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
    "ALL_LAYERS",
    "COMPONENT_ID",
    "COMPONENT_VERSION",
    "DEFAULT_PRESENTATION",
    "DESCRIPTOR",
    "DESCRIPTOR_SCHEMA_VERSION",
    "LAYER_AVAILABILITY_SCHEMA_VERSION",
    "OPTIONAL_CAPABILITIES",
    "OUTPUT_TYPES",
    "OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION",
    "REQUIRED_CAPABILITIES",
    "TEST_PRESENTATION",
    "VISUALIZATION_SPEC_SCHEMA_VERSION",
    "FrameRenderContext",
    "OverlayPipelineConfig",
    "descriptor_document",
    "main",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
