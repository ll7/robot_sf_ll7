"""Render print-ready top-down figures from pinned exemplar trace episodes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.transforms import Bbox, IdentityTransform

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

INK = "#2B2B2B"
GRAY = "#5D6D7E"
BLUE = "#2C6FB5"
TEAL = "#1F8A8A"
GREEN = "#2E8B57"
ORANGE = "#D98326"
PURPLE = "#7D5BA6"
RED = "#C0392B"

PEDESTRIAN_COLORS = (BLUE, TEAL, ORANGE, PURPLE, GREEN, RED)
FOCAL_PEDESTRIAN_COLOR = ORANGE
CONTEXT_PEDESTRIAN_COLOR = GRAY
CONTEXT_PEDESTRIAN_ALPHA = 0.35
LABEL_MIN_GAP_M = 1.5
DEFAULT_COLLISION_ENVELOPE_M = 1.4
DEFAULT_COMFORT_DISTANCE_M = 1.2
_MATRIX_PATH = Path(__file__).resolve().parents[2] / "configs/scenarios/classic_interactions.yaml"
_SUCCESS_STATUSES = {"success", "goal", "goal_reached", "completed"}
_DEFAULT_MARKER_INTERVAL_S = 2.0
_LONG_EPISODE_MARKER_INTERVAL_S = 4.0
_LONG_EPISODE_THRESHOLD_S = 25.0
_STATIONARY_MARKER_DISPLACEMENT_M = 0.5
_ZONE_LABEL_SUPPRESSION_RADIUS_M = 2.0
_LABEL_AXES_MARGIN_PX = 4.0
_LABEL_COLLISION_PADDING_PX = 1.5
_LABEL_LEADER_THRESHOLD_PX = 18.0
_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

PedestrianSample = tuple[float, float, float]
AxisLimits = tuple[tuple[float, float], tuple[float, float]]


class TraceSchemaError(ValueError):
    """Raised when an exemplar bundle does not satisfy the trace figure schema."""


@dataclass(frozen=True)
class EpisodeTrace:
    """Normalized trajectory data needed by the trace-scene renderer."""

    metadata: Mapping[str, Any]
    steps: tuple[int, ...]
    time_s: tuple[float, ...]
    robot_xy: tuple[tuple[float, float], ...]
    robot_heading_rad: tuple[float, ...]
    executed_speed_m_s: tuple[float, ...]
    min_robot_ped_distance_m: tuple[float, ...]
    nearest_pedestrian_id: tuple[str | None, ...]
    pedestrian_tracks: Mapping[str, tuple[PedestrianSample, ...]]
    episode_dir: Path | None = None


@dataclass(frozen=True)
class _RobotSeries:
    steps: tuple[int, ...]
    times: tuple[float, ...]
    xy: tuple[tuple[float, float], ...]
    headings: tuple[float, ...]
    speeds: tuple[float, ...]
    min_distances: tuple[float, ...]
    nearest_ids: tuple[str | None, ...]


@dataclass(frozen=True)
class _PedestrianSelection:
    tracks: Mapping[str, tuple[PedestrianSample, ...]]
    filtered_count: int
    radius_m: float


@dataclass(frozen=True)
class _MarkerLabelSpec:
    marker_position: int
    text: str


@dataclass(frozen=True)
class _PedestrianStyle:
    color: str
    linewidth: float
    alpha: float
    draw_markers: bool
    open_markers: bool
    show_label: bool


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TraceSchemaError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TraceSchemaError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TraceSchemaError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TraceSchemaError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise TraceSchemaError(f"{context} must be an array")
    return value


def _number(row: Mapping[str, Any], key: str, context: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceSchemaError(f"{context}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TraceSchemaError(f"{context}.{key} must be finite")
    return result


def _integer(row: Mapping[str, Any], key: str, context: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceSchemaError(f"{context}.{key} must be an integer")
    return value


def _episode_status(metadata: Mapping[str, Any]) -> str:
    summary = _mapping(metadata.get("summary"), "metadata.summary")
    status = metadata.get("episode_status", metadata.get("status", summary.get("episode_status")))
    if not isinstance(status, str) or not status.strip():
        raise TraceSchemaError(
            "metadata must define episode_status, status, or summary.episode_status"
        )
    return status.strip()


def _require_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    for key in ("planner", "scenario_id", "seed"):
        if key not in normalized:
            raise TraceSchemaError(f"metadata missing required key: {key}")
    if not isinstance(normalized["planner"], str) or not normalized["planner"]:
        raise TraceSchemaError("metadata.planner must be a non-empty string")
    if not isinstance(normalized["scenario_id"], str) or not normalized["scenario_id"]:
        raise TraceSchemaError("metadata.scenario_id must be a non-empty string")
    if isinstance(normalized["seed"], bool) or not isinstance(normalized["seed"], int):
        raise TraceSchemaError("metadata.seed must be an integer")

    summary = _mapping(normalized.get("summary"), "metadata.summary")
    for key in (
        "global_min_robot_ped_distance_m",
        "global_min_distance_step",
        "step_count",
        "termination_reason",
    ):
        if key not in summary:
            raise TraceSchemaError(f"metadata.summary missing required key: {key}")
    _number(summary, "global_min_robot_ped_distance_m", "metadata.summary")
    _integer(summary, "global_min_distance_step", "metadata.summary")
    _integer(summary, "step_count", "metadata.summary")
    if not isinstance(summary["termination_reason"], str):
        raise TraceSchemaError("metadata.summary.termination_reason must be a string")

    normalized["episode_status"] = _episode_status(normalized)
    return normalized


def _parse_derived_rows(raw_rows: Sequence[Any]) -> _RobotSeries:
    steps: list[int] = []
    times: list[float] = []
    robot_xy: list[tuple[float, float]] = []
    headings: list[float] = []
    speeds: list[float] = []
    min_distances: list[float] = []
    nearest_ids: list[str | None] = []
    for index, raw_row in enumerate(raw_rows):
        context = f"trace_series.derived_rows[{index}]"
        row = _mapping(raw_row, context)
        steps.append(_integer(row, "step", context))
        times.append(_number(row, "time_s", context))
        robot_xy.append((_number(row, "robot_x_m", context), _number(row, "robot_y_m", context)))
        headings.append(_number(row, "robot_heading_rad", context))
        speeds.append(_number(row, "executed_speed_m_s", context))
        min_distances.append(_number(row, "min_robot_ped_distance_m", context))
        nearest_id = row.get("nearest_pedestrian_id")
        nearest_ids.append(None if nearest_id is None else str(nearest_id))

    if len(set(steps)) != len(steps):
        raise TraceSchemaError("trace_series.derived_rows contains duplicate step values")
    if any(next_time <= time for time, next_time in pairwise(times)):
        raise TraceSchemaError(
            "trace_series.derived_rows time_s values must be strictly increasing"
        )
    return _RobotSeries(
        steps=tuple(steps),
        times=tuple(times),
        xy=tuple(robot_xy),
        headings=tuple(headings),
        speeds=tuple(speeds),
        min_distances=tuple(min_distances),
        nearest_ids=tuple(nearest_ids),
    )


def _parse_frames(
    raw_frames: Sequence[Any], time_by_step: Mapping[int, float]
) -> dict[str, tuple[PedestrianSample, ...]]:
    tracks: dict[str, list[PedestrianSample]] = {}
    frame_steps: list[int] = []
    for index, raw_frame in enumerate(raw_frames):
        context = f"trace_series.frames[{index}]"
        frame = _mapping(raw_frame, context)
        step = _integer(frame, "step", context)
        frame_steps.append(step)
        if step not in time_by_step:
            raise TraceSchemaError(f"{context}.step has no matching derived row: {step}")
        time = _number(frame, "time_s", context)
        if not math.isclose(time, time_by_step[step], abs_tol=1e-9):
            raise TraceSchemaError(f"{context}.time_s does not match its derived row")
        pedestrians = _sequence(frame.get("pedestrians"), f"{context}.pedestrians")
        for ped_index, raw_pedestrian in enumerate(pedestrians):
            ped_context = f"{context}.pedestrians[{ped_index}]"
            pedestrian = _mapping(raw_pedestrian, ped_context)
            if "id" not in pedestrian:
                raise TraceSchemaError(f"{ped_context} missing required key: id")
            position = _sequence(pedestrian.get("position"), f"{ped_context}.position")
            if len(position) != 2:
                raise TraceSchemaError(f"{ped_context}.position must contain [x, y]")
            position_row = {"x": position[0], "y": position[1]}
            ped_id = str(pedestrian["id"])
            tracks.setdefault(ped_id, []).append(
                (
                    time,
                    _number(position_row, "x", f"{ped_context}.position"),
                    _number(position_row, "y", f"{ped_context}.position"),
                )
            )
    if len(frame_steps) != len(time_by_step) or set(frame_steps) != set(time_by_step):
        raise TraceSchemaError("trace_series.frames and derived_rows must cover the same steps")
    return {ped_id: tuple(track) for ped_id, track in tracks.items()}


def load_episode(episode_dir: Path) -> EpisodeTrace:
    """Load and validate one pinned exemplar trace episode directory.

    Returns:
        EpisodeTrace: Normalized per-step robot data and pedestrian tracks.
    """

    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        raise TraceSchemaError(f"episode directory does not exist: {episode_dir}")
    metadata = _require_metadata(_read_json(episode_dir / "metadata.json"))
    trace = _read_json(episode_dir / "trace_series.json")
    raw_rows = _sequence(trace.get("derived_rows"), "trace_series.derived_rows")
    raw_frames = _sequence(trace.get("frames"), "trace_series.frames")
    if not raw_rows:
        raise TraceSchemaError("trace_series.derived_rows must not be empty")
    if not raw_frames:
        raise TraceSchemaError("trace_series.frames must not be empty")

    robot = _parse_derived_rows(raw_rows)
    time_by_step = dict(zip(robot.steps, robot.times, strict=True))
    tracks = _parse_frames(raw_frames, time_by_step)
    summary = _mapping(metadata["summary"], "metadata.summary")
    min_step = _integer(summary, "global_min_distance_step", "metadata.summary")
    if min_step not in time_by_step:
        raise TraceSchemaError("metadata.summary.global_min_distance_step is not in the trace")
    if _integer(summary, "step_count", "metadata.summary") != len(robot.steps):
        raise TraceSchemaError("metadata.summary.step_count does not match the trace length")

    return EpisodeTrace(
        metadata=metadata,
        steps=robot.steps,
        time_s=robot.times,
        robot_xy=robot.xy,
        robot_heading_rad=robot.headings,
        executed_speed_m_s=robot.speeds,
        min_robot_ped_distance_m=robot.min_distances,
        nearest_pedestrian_id=robot.nearest_ids,
        pedestrian_tracks=tracks,
        episode_dir=episode_dir.resolve(),
    )


def _pedestrian_sort_key(ped_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(ped_id))
    except ValueError:
        return (1, ped_id)


def _focused_pedestrian_tracks(
    episode: EpisodeTrace, radius_m: float
) -> tuple[dict[str, tuple[PedestrianSample, ...]], int]:
    if radius_m <= 0:
        raise ValueError("ped_focus_radius_m must be greater than zero")
    robot_t = np.asarray(episode.time_s)
    robot_x = np.asarray([point[0] for point in episode.robot_xy])
    robot_y = np.asarray([point[1] for point in episode.robot_xy])
    focused: dict[str, tuple[PedestrianSample, ...]] = {}
    for ped_id in sorted(episode.pedestrian_tracks, key=_pedestrian_sort_key):
        track = episode.pedestrian_tracks[ped_id]
        ped_t = np.asarray([sample[0] for sample in track])
        in_range = (ped_t >= robot_t[0]) & (ped_t <= robot_t[-1])
        if not np.any(in_range):
            continue
        ped_x = np.asarray([sample[1] for sample in track])[in_range]
        ped_y = np.asarray([sample[2] for sample in track])[in_range]
        sample_t = ped_t[in_range]
        sync_robot_x = np.interp(sample_t, robot_t, robot_x)
        sync_robot_y = np.interp(sample_t, robot_t, robot_y)
        min_distance = float(np.min(np.hypot(ped_x - sync_robot_x, ped_y - sync_robot_y)))
        if min_distance < radius_m:
            focused[ped_id] = track
    return focused, len(episode.pedestrian_tracks) - len(focused)


def _snap_marker_indices(times: Sequence[float], interval_s: float) -> tuple[int, ...]:
    if interval_s <= 0:
        raise ValueError("marker_interval_s must be greater than zero")
    if not times:
        return ()
    samples = np.asarray(times, dtype=float)
    target_count = math.floor((samples[-1] + interval_s / 2 + 1e-9) / interval_s)
    indices: list[int] = []
    for multiple in range(1, target_count + 1):
        index = int(np.argmin(np.abs(samples - multiple * interval_s)))
        if not indices or index != indices[-1]:
            indices.append(index)
    return tuple(indices)


def _effective_marker_interval(
    episodes: Sequence[EpisodeTrace], requested_interval_s: float | None
) -> float:
    if requested_interval_s is not None:
        if requested_interval_s <= 0:
            raise ValueError("marker_interval_s must be greater than zero")
        return requested_interval_s
    longest_duration = max(episode.time_s[-1] - episode.time_s[0] for episode in episodes)
    if longest_duration > _LONG_EPISODE_THRESHOLD_S:
        return _LONG_EPISODE_MARKER_INTERVAL_S
    return _DEFAULT_MARKER_INTERVAL_S


def _marker_label_specs(
    marker_points: Sequence[tuple[float, float]],
    marker_interval_s: float,
    *,
    label_min_gap_m: float = LABEL_MIN_GAP_M,
) -> tuple[_MarkerLabelSpec, ...]:
    """Plan thinned time labels while consolidating stationary skipped runs.

    Returns:
        tuple[_MarkerLabelSpec, ...]: Labels keyed by positions in the marker sequence.
    """

    if label_min_gap_m <= 0:
        raise ValueError("label_min_gap_m must be greater than zero")
    if not marker_points:
        return ()

    regular_label_positions: list[int] = []
    skipped_positions: set[int] = set()
    for marker_position, point in enumerate(marker_points):
        if not regular_label_positions or all(
            math.dist(point, marker_points[labeled_position]) >= label_min_gap_m
            for labeled_position in regular_label_positions
        ):
            regular_label_positions.append(marker_position)
        else:
            skipped_positions.add(marker_position)

    specs = [
        _MarkerLabelSpec(
            marker_position=marker_position,
            text=f"t={(marker_position + 1) * marker_interval_s:g}s",
        )
        for marker_position in regular_label_positions
    ]
    marker_position = 1
    while marker_position < len(marker_points):
        if (
            marker_position not in skipped_positions
            or math.dist(marker_points[marker_position - 1], marker_points[marker_position])
            >= _STATIONARY_MARKER_DISPLACEMENT_M
        ):
            marker_position += 1
            continue
        run_start = marker_position
        run_end = run_start
        while (
            run_end + 1 < len(marker_points)
            and run_end + 1 in skipped_positions
            and math.dist(marker_points[run_end], marker_points[run_end + 1])
            < _STATIONARY_MARKER_DISPLACEMENT_M
        ):
            run_end += 1
        if run_end > run_start:
            first_time = (run_start + 1) * marker_interval_s
            last_time = (run_end + 1) * marker_interval_s
            specs.append(
                _MarkerLabelSpec(
                    marker_position=run_start,
                    text=f"t={first_time:g}-{last_time:g} s (stopped)",
                )
            )
        marker_position = run_end + 1
    return tuple(sorted(specs, key=lambda spec: spec.marker_position))


def _obstacle_vertices(obstacle: Any) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in obstacle.vertices]


def _bbox(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _boxes_intersect(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> bool:
    return not (
        first[1] < second[0] or second[1] < first[0] or first[3] < second[2] or second[3] < first[2]
    )


def _compute_scene_extent(
    episodes: Sequence[EpisodeTrace],
    focused_tracks: Sequence[Mapping[str, tuple[PedestrianSample, ...]]],
    map_definition: Any,
    *,
    padding_m: float = 2.0,
) -> AxisLimits:
    """Compute a map-clipped action extent shared by one or more episodes.

    Returns:
        AxisLimits: Shared x and y limits in metres.
    """

    points = [point for episode in episodes for point in episode.robot_xy]
    points.extend(
        (sample[1], sample[2])
        for tracks in focused_tracks
        for track in tracks.values()
        for sample in track
    )
    if not points:
        raise ValueError("cannot compute an extent without trajectory points")
    action_bbox = _bbox(points)
    included_obstacle_points: list[tuple[float, float]] = []
    for obstacle in map_definition.obstacles:
        vertices = _obstacle_vertices(obstacle)
        if vertices and _boxes_intersect(action_bbox, _bbox(vertices)):
            included_obstacle_points.extend(vertices)
    if included_obstacle_points:
        points.extend(included_obstacle_points)
    min_x, max_x, min_y, max_y = _bbox(points)
    return (
        (max(0.0, min_x - padding_m), min(float(map_definition.width), max_x + padding_m)),
        (
            max(0.0, min_y - padding_m),
            min(float(map_definition.height), max_y + padding_m),
        ),
    )


@lru_cache(maxsize=16)
def _load_map_definition(scenario_id: str) -> Any:
    from robot_sf.benchmark.classic_interactions_loader import (  # noqa: PLC0415
        load_classic_matrix,
        select_scenario,
    )
    from robot_sf.nav.svg_map_parser import SvgMapConverter  # noqa: PLC0415

    matrix = load_classic_matrix(str(_MATRIX_PATH))
    scenario = select_scenario(matrix, scenario_id)
    map_file = scenario.get("map_file")
    if not isinstance(map_file, str) or not map_file:
        raise TraceSchemaError(f"scenario '{scenario_id}' has no usable map_file")
    map_path = Path(map_file)
    if not map_path.is_absolute():
        map_path = (_MATRIX_PATH.parent / map_path).resolve()
    if not map_path.is_file():
        raise TraceSchemaError(f"scenario '{scenario_id}' map file does not exist: {map_path}")
    return SvgMapConverter(str(map_path)).map_definition


def _zone_vertices(zone: Any) -> list[tuple[float, float]]:
    first, second, third = zone
    fourth = (first[0] + third[0] - second[0], first[1] + third[1] - second[1])
    return [first, second, third, fourth]


def _position_at_time(track: Sequence[PedestrianSample], time_s: float) -> tuple[float, float]:
    index = min(range(len(track)), key=lambda candidate: abs(track[candidate][0] - time_s))
    return track[index][1], track[index][2]


def _marker_label_layout(
    points: Sequence[tuple[float, float]], index: int
) -> tuple[tuple[float, float], str, str]:
    """Place dense time labels radially, falling back to alternating trajectory normals.

    Returns:
        tuple: Offset in points plus horizontal and vertical text alignment.
    """

    x, y = points[index]
    neighbors = [point for point in points if math.dist((x, y), point) <= 3.0]
    center_x = sum(point[0] for point in neighbors) / len(neighbors)
    center_y = sum(point[1] for point in neighbors) / len(neighbors)
    direction_x = x - center_x
    direction_y = y - center_y
    if math.hypot(direction_x, direction_y) < 0.15:
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        tangent_x = after[0] - before[0]
        tangent_y = after[1] - before[1]
        sign = -1.0 if index % 2 else 1.0
        direction_x = -tangent_y * sign
        direction_y = tangent_x * sign
    magnitude = max(math.hypot(direction_x, direction_y), 1e-9)
    recurring_position_count = sum(
        math.dist((x, y), points[prior_index]) <= 1.2 for prior_index in range(index)
    )
    offset_distance = 9.0 + 6.0 * min(recurring_position_count, 3)
    offset_x = offset_distance * direction_x / magnitude
    offset_y = offset_distance * direction_y / magnitude
    horizontal_alignment = "left" if offset_x >= 0 else "right"
    vertical_alignment = "bottom" if offset_y >= 0 else "top"
    return (offset_x, offset_y), horizontal_alignment, vertical_alignment


def _scene_label_priority(text: str) -> int:
    """Rank annotations so time and pedestrian labels yield to semantic key frames.

    Returns:
        int: Lower values indicate labels that should retain their preferred position.
    """

    if text.startswith("t="):
        return 2 if "(stopped)" in text else 4
    if text.startswith("ped ") or "distant pedestrians" in text:
        return 3
    if text.endswith(" m") and not text.startswith("d_min"):
        return 3
    return 0


def _shifted_bbox(bbox: Bbox, shift_x: float, shift_y: float) -> Bbox:
    return Bbox.from_extents(
        bbox.x0 + shift_x,
        bbox.y0 + shift_y,
        bbox.x1 + shift_x,
        bbox.y1 + shift_y,
    )


def _clamp_bbox_shift(bbox: Bbox, axes_bbox: Bbox) -> tuple[float, float]:
    """Return the display-space shift needed to keep a label inside its panel."""

    inner = Bbox.from_extents(
        axes_bbox.x0 + _LABEL_AXES_MARGIN_PX,
        axes_bbox.y0 + _LABEL_AXES_MARGIN_PX,
        axes_bbox.x1 - _LABEL_AXES_MARGIN_PX,
        axes_bbox.y1 - _LABEL_AXES_MARGIN_PX,
    )
    shift_x = max(inner.x0 - bbox.x0, 0.0) + min(inner.x1 - bbox.x1, 0.0)
    shift_y = max(inner.y0 - bbox.y0, 0.0) + min(inner.y1 - bbox.y1, 0.0)
    return shift_x, shift_y


def _bboxes_collide(first: Bbox, second: Bbox) -> bool:
    first_padded = first.padded(_LABEL_COLLISION_PADDING_PX)
    second_padded = second.padded(_LABEL_COLLISION_PADDING_PX)
    intersection = Bbox.intersection(first_padded, second_padded)
    return intersection is not None and intersection.width > 0 and intersection.height > 0


def _bbox_overlap_area(first: Bbox, second: Bbox) -> float:
    intersection = Bbox.intersection(first, second)
    if intersection is None:
        return 0.0
    return max(intersection.width, 0.0) * max(intersection.height, 0.0)


def _move_text_in_display_space(text: Any, shift_x: float, shift_y: float) -> None:
    transform = text.get_transform()
    display_position = transform.transform(text.get_position())
    text.set_position(transform.inverted().transform(display_position + (shift_x, shift_y)))


def _place_scene_annotations(ax: Axes) -> None:
    """Place scene annotations without label collisions or panel-boundary overflow.

    Matplotlib resolves text extents only after a canvas draw. This pass works in display pixels,
    prioritizes key-frame labels, and tries compact north/east/south/west offsets before larger
    displacements. Labels that require a substantial move retain their anchor context through a
    thin leader line.
    """

    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes_bbox = ax.get_window_extent(renderer)
    labels = [text for text in ax.texts if text.get_visible() and text.get_text().strip()]
    labels.sort(key=lambda text: (_scene_label_priority(text.get_text()), ax.texts.index(text)))
    placed_bboxes: list[Bbox] = []
    candidate_offsets = (
        (0.0, 0.0),
        (0.0, 12.0),
        (12.0, 0.0),
        (0.0, -12.0),
        (-12.0, 0.0),
        (12.0, 12.0),
        (12.0, -12.0),
        (-12.0, -12.0),
        (-12.0, 12.0),
        (0.0, 24.0),
        (24.0, 0.0),
        (0.0, -24.0),
        (-24.0, 0.0),
        (24.0, 24.0),
        (24.0, -24.0),
        (-24.0, -24.0),
        (-24.0, 24.0),
        (0.0, 40.0),
        (40.0, 0.0),
        (0.0, -40.0),
        (-40.0, 0.0),
    )

    for text in labels:
        original_bbox = text.get_window_extent(renderer)
        candidates: list[tuple[float, float, Bbox]] = []
        for offset_x, offset_y in candidate_offsets:
            candidate_bbox = _shifted_bbox(original_bbox, offset_x, offset_y)
            clamp_x, clamp_y = _clamp_bbox_shift(candidate_bbox, axes_bbox)
            shift_x = offset_x + clamp_x
            shift_y = offset_y + clamp_y
            candidates.append((shift_x, shift_y, _shifted_bbox(original_bbox, shift_x, shift_y)))

        collision_free = next(
            (
                candidate
                for candidate in candidates
                if not any(
                    _bboxes_collide(candidate[2], placed_bbox) for placed_bbox in placed_bboxes
                )
            ),
            None,
        )
        if collision_free is None:
            collision_free = min(
                candidates,
                key=lambda candidate: (
                    sum(
                        _bbox_overlap_area(candidate[2], placed_bbox)
                        for placed_bbox in placed_bboxes
                    ),
                    math.hypot(candidate[0], candidate[1]),
                ),
            )
        shift_x, shift_y, final_bbox = collision_free
        _move_text_in_display_space(text, shift_x, shift_y)
        placed_bboxes.append(final_bbox)
        if math.hypot(shift_x, shift_y) >= _LABEL_LEADER_THRESHOLD_PX:
            original_center = (
                original_bbox.x0 + original_bbox.width / 2,
                original_bbox.y0 + original_bbox.height / 2,
            )
            final_center = (
                final_bbox.x0 + final_bbox.width / 2,
                final_bbox.y0 + final_bbox.height / 2,
            )
            ax.add_line(
                Line2D(
                    [original_center[0], final_center[0]],
                    [original_center[1], final_center[1]],
                    transform=IdentityTransform(),
                    color=GRAY,
                    linewidth=0.45,
                    alpha=0.65,
                    zorder=6,
                    clip_on=True,
                )
            )


def _draw_robot_time_markers(
    ax: Axes,
    episode: EpisodeTrace,
    marker_indices: Sequence[int],
    marker_interval_s: float,
) -> tuple[_MarkerLabelSpec, ...]:
    marker_points = [episode.robot_xy[index] for index in marker_indices]
    label_specs = _marker_label_specs(marker_points, marker_interval_s)
    for x, y in marker_points:
        ax.plot(x, y, marker="o", markersize=3.5, color=INK, linestyle="none", zorder=8)
    for label_spec in label_specs:
        x, y = marker_points[label_spec.marker_position]
        offset, horizontal_alignment, vertical_alignment = _marker_label_layout(
            marker_points, label_spec.marker_position
        )
        ax.annotate(
            label_spec.text,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=GRAY,
            ha=horizontal_alignment,
            va=vertical_alignment,
        )
    return label_specs


def _global_min_index(episode: EpisodeTrace) -> int:
    summary = _mapping(episode.metadata.get("summary"), "metadata.summary")
    min_step = _integer(summary, "global_min_distance_step", "metadata.summary")
    try:
        return episode.steps.index(min_step)
    except ValueError:
        return int(np.argmin(episode.min_robot_ped_distance_m))


def _nearest_pedestrian_position(
    episode: EpisodeTrace, index: int
) -> tuple[str, tuple[float, float]] | None:
    time = episode.time_s[index]
    nearest_id = episode.nearest_pedestrian_id[index]
    if nearest_id is not None and nearest_id in episode.pedestrian_tracks:
        return nearest_id, _position_at_time(episode.pedestrian_tracks[nearest_id], time)
    robot_x, robot_y = episode.robot_xy[index]
    candidates = [
        (ped_id, _position_at_time(track, time))
        for ped_id, track in episode.pedestrian_tracks.items()
        if track
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: math.dist((robot_x, robot_y), item[1]))


def _focal_pedestrian_id(episode: EpisodeTrace) -> str | None:
    nearest = _nearest_pedestrian_position(episode, _global_min_index(episode))
    return nearest[0] if nearest is not None else None


def _pedestrian_styles(
    episode: EpisodeTrace,
    focused_tracks: Mapping[str, tuple[PedestrianSample, ...]],
    *,
    highlight_focal: bool,
) -> dict[str, _PedestrianStyle]:
    if not highlight_focal:
        return {
            ped_id: _PedestrianStyle(
                color=PEDESTRIAN_COLORS[index % len(PEDESTRIAN_COLORS)],
                linewidth=0.9,
                alpha=0.75,
                draw_markers=True,
                open_markers=False,
                show_label=True,
            )
            for index, ped_id in enumerate(focused_tracks)
        }

    focal_id = _focal_pedestrian_id(episode)
    return {
        ped_id: (
            _PedestrianStyle(
                color=FOCAL_PEDESTRIAN_COLOR,
                linewidth=1.35,
                alpha=1.0,
                draw_markers=True,
                open_markers=True,
                show_label=True,
            )
            if ped_id == focal_id
            else _PedestrianStyle(
                color=CONTEXT_PEDESTRIAN_COLOR,
                linewidth=0.8,
                alpha=CONTEXT_PEDESTRIAN_ALPHA,
                draw_markers=False,
                open_markers=False,
                show_label=False,
            )
        )
        for ped_id in focused_tracks
    }


def _draw_obstacles(ax: Axes, map_definition: Any, limits: AxisLimits) -> None:
    viewport = (limits[0][0], limits[0][1], limits[1][0], limits[1][1])
    for obstacle in map_definition.obstacles:
        vertices = _obstacle_vertices(obstacle)
        if vertices and _boxes_intersect(viewport, _bbox(vertices)):
            ax.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    facecolor=GRAY,
                    edgecolor=GRAY,
                    alpha=0.30,
                    linewidth=0.8,
                    zorder=1,
                )
            )


def _draw_zones(ax: Axes, map_definition: Any, episode: EpisodeTrace) -> list[tuple[float, float]]:
    drawn_label_centers: list[tuple[float, float]] = []
    for zones, linestyle, label, marker_point in (
        (map_definition.robot_spawn_zones, "--", "start", episode.robot_xy[0]),
        (map_definition.robot_goal_zones, "-", "goal", episode.robot_xy[-1]),
    ):
        for zone_index, zone in enumerate(zones):
            vertices = _zone_vertices(zone)
            ax.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    fill=False,
                    edgecolor=GREEN,
                    linestyle=linestyle,
                    linewidth=1.0,
                    zorder=2,
                )
            )
            if zone_index == 0:
                center_x = sum(point[0] for point in vertices) / len(vertices)
                center_y = sum(point[1] for point in vertices) / len(vertices)
                center = (center_x, center_y)
                if math.dist(center, marker_point) > _ZONE_LABEL_SUPPRESSION_RADIUS_M:
                    ax.text(
                        center_x,
                        center_y,
                        label,
                        color=GRAY,
                        fontsize=7,
                        ha="center",
                        va="center",
                    )
                    drawn_label_centers.append(center)
    return drawn_label_centers


def _scale_bar_geometry(limits: AxisLimits, length: float, corner: str) -> tuple[float, float]:
    width = limits[0][1] - limits[0][0]
    height = limits[1][1] - limits[1][0]
    inset_x = 0.05 * width
    inset_y = 0.06 * height
    start_x = (
        limits[0][1] - inset_x - length if corner.endswith("right") else limits[0][0] + inset_x
    )
    y = limits[1][1] - inset_y if corner.startswith("top") else limits[1][0] + inset_y
    return start_x, y


def _choose_scale_bar_corner(
    limits: AxisLimits,
    length: float,
    trajectory_points: Sequence[tuple[float, float]],
    zone_label_centers: Sequence[tuple[float, float]],
) -> str:
    width = limits[0][1] - limits[0][0]
    height = limits[1][1] - limits[1][0]
    horizontal_padding = max(1.0, 0.08 * width)
    vertical_padding = max(1.0, 0.08 * height)
    occupancy_points = [*trajectory_points, *zone_label_centers]
    corners = ("bottom-right", "bottom-left", "top-right", "top-left")
    counts: list[int] = []
    for corner in corners:
        start_x, y = _scale_bar_geometry(limits, length, corner)
        counts.append(
            sum(
                start_x - horizontal_padding <= point[0] <= start_x + length + horizontal_padding
                and y - vertical_padding <= point[1] <= y + vertical_padding
                for point in occupancy_points
            )
        )
    return corners[min(range(len(corners)), key=counts.__getitem__)]


def _draw_scale_bar(
    ax: Axes,
    limits: AxisLimits,
    trajectory_points: Sequence[tuple[float, float]],
    zone_label_centers: Sequence[tuple[float, float]],
) -> str:
    width = limits[0][1] - limits[0][0]
    height = limits[1][1] - limits[1][0]
    length = 5.0 if width >= 12.0 else 2.0
    corner = _choose_scale_bar_corner(limits, length, trajectory_points, zone_label_centers)
    start_x, y = _scale_bar_geometry(limits, length, corner)
    ax.plot([start_x, start_x + length], [y, y], color=INK, linewidth=2.0, zorder=8)
    ax.plot(
        [start_x, start_x],
        [y - 0.012 * height, y + 0.012 * height],
        color=INK,
        linewidth=1.0,
        zorder=8,
    )
    ax.plot(
        [start_x + length, start_x + length],
        [y - 0.012 * height, y + 0.012 * height],
        color=INK,
        linewidth=1.0,
        zorder=8,
    )
    label_offset = 0.025 * height
    label_y = y - label_offset if corner.startswith("top") else y + label_offset
    ax.text(
        start_x + length / 2,
        label_y,
        f"{length:g} m",
        ha="center",
        va="top" if corner.startswith("top") else "bottom",
        fontsize=8,
    )
    return corner


def _draw_key_frames(ax: Axes, episode: EpisodeTrace) -> None:
    start_x, start_y = episode.robot_xy[0]
    ax.scatter(
        [start_x], [start_y], s=28, color=GREEN, edgecolors="white", linewidths=0.5, zorder=9
    )
    ax.annotate(
        "start",
        (start_x, start_y),
        xytext=(5, -10),
        textcoords="offset points",
        fontsize=8,
        color=GREEN,
    )

    min_index = _global_min_index(episode)
    robot_x, robot_y = episode.robot_xy[min_index]
    nearest = _nearest_pedestrian_position(episode, min_index)
    if nearest is not None:
        _, (ped_x, ped_y) = nearest
        ax.plot(
            [robot_x, ped_x],
            [robot_y, ped_y],
            color=RED,
            linestyle="--",
            linewidth=1.0,
            zorder=7,
        )
        ax.scatter(
            [robot_x, ped_x],
            [robot_y, ped_y],
            s=48,
            facecolors="none",
            edgecolors=RED,
            linewidths=1.2,
            zorder=9,
        )
        summary = _mapping(episode.metadata.get("summary"), "metadata.summary")
        distance = float(summary["global_min_robot_ped_distance_m"])
        ax.annotate(
            f"d_min = {distance:.2f} m",
            ((robot_x + ped_x) / 2, (robot_y + ped_y) / 2),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=8,
            color=RED,
            ha="center",
        )

    terminal_x, terminal_y = episode.robot_xy[-1]
    status = str(episode.metadata["episode_status"])
    if status.lower() in _SUCCESS_STATUSES:
        ax.scatter(
            [terminal_x],
            [terminal_y],
            s=42,
            facecolors="none",
            edgecolors=GREEN,
            linewidths=1.4,
            zorder=9,
        )
        terminal_label = "goal"
        terminal_color = GREEN
    else:
        ax.scatter(
            [terminal_x],
            [terminal_y],
            s=28,
            color=RED,
            edgecolors="white",
            linewidths=0.5,
            zorder=9,
        )
        summary = _mapping(episode.metadata.get("summary"), "metadata.summary")
        terminal_label = str(summary["termination_reason"])
        terminal_color = RED
    terminal_offset = (8, 8) if status.lower() in _SUCCESS_STATUSES else (8, -10)
    ax.annotate(
        terminal_label,
        (terminal_x, terminal_y),
        xytext=terminal_offset,
        textcoords="offset points",
        fontsize=8,
        color=terminal_color,
        va="bottom" if status.lower() in _SUCCESS_STATUSES else "top",
    )


def _draw_scene_panel(
    ax: Axes,
    episode: EpisodeTrace,
    map_definition: Any,
    pedestrian_selection: _PedestrianSelection,
    marker_interval_s: float,
    limits: AxisLimits,
    title: str,
    *,
    highlight_focal: bool,
) -> tuple[int, ...]:
    focused_tracks = pedestrian_selection.tracks
    marker_indices = _snap_marker_indices(episode.time_s, marker_interval_s)
    marker_times = [episode.time_s[index] for index in marker_indices]
    _draw_obstacles(ax, map_definition, limits)
    zone_label_centers = _draw_zones(ax, map_definition, episode)

    robot_x = [point[0] for point in episode.robot_xy]
    robot_y = [point[1] for point in episode.robot_xy]
    ax.plot(robot_x, robot_y, color=INK, linewidth=1.8, zorder=5)
    marker_points = [episode.robot_xy[index] for index in marker_indices]
    marker_label_specs = _draw_robot_time_markers(ax, episode, marker_indices, marker_interval_s)
    labeled_robot_points = [
        marker_points[label_spec.marker_position] for label_spec in marker_label_specs
    ]

    pedestrian_styles = _pedestrian_styles(episode, focused_tracks, highlight_focal=highlight_focal)
    for ped_index, (ped_id, track) in enumerate(focused_tracks.items()):
        style = pedestrian_styles[ped_id]
        ax.plot(
            [sample[1] for sample in track],
            [sample[2] for sample in track],
            color=style.color,
            linewidth=style.linewidth,
            alpha=style.alpha,
            zorder=4,
        )
        if style.draw_markers:
            for marker_time in marker_times:
                x, y = _position_at_time(track, marker_time)
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=4.0 if style.open_markers else 2.5,
                    markerfacecolor="white" if style.open_markers else style.color,
                    markeredgecolor=style.color,
                    markeredgewidth=0.9 if style.open_markers else 0.0,
                    linestyle="none",
                    alpha=style.alpha,
                    zorder=7,
                )
        if not style.show_label:
            continue
        label_anchor = (track[0][1], track[0][2])
        offset_x = 4 if ped_index % 2 == 0 else -4
        offset_y = 4 if ped_index % 4 < 2 else -8
        if labeled_robot_points:
            nearest_robot_label = min(
                labeled_robot_points, key=lambda point: math.dist(label_anchor, point)
            )
            if math.dist(label_anchor, nearest_robot_label) < LABEL_MIN_GAP_M:
                direction_x = label_anchor[0] - nearest_robot_label[0]
                direction_y = label_anchor[1] - nearest_robot_label[1]
                offset_x = 12 if direction_x >= 0 else -12
                offset_y = 12 if direction_y >= 0 else -12
        ax.annotate(
            f"ped {ped_id}",
            label_anchor,
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=7,
            color=style.color,
            ha="left" if offset_x >= 0 else "right",
            va="bottom" if offset_y >= 0 else "top",
        )

    trajectory_points = [*episode.robot_xy]
    trajectory_points.extend(
        (sample[1], sample[2]) for track in focused_tracks.values() for sample in track
    )
    scale_bar_corner = _draw_scale_bar(ax, limits, trajectory_points, zone_label_centers)
    if pedestrian_selection.filtered_count:
        annotation_y = 0.98 if scale_bar_corner.startswith("bottom") else 0.02
        ax.text(
            0.98,
            annotation_y,
            f"{pedestrian_selection.filtered_count} distant pedestrians not drawn "
            f"(>{pedestrian_selection.radius_m:g} m)",
            transform=ax.transAxes,
            ha="right",
            va="top" if annotation_y > 0.5 else "bottom",
            fontsize=7,
            color=GRAY,
        )
    _draw_key_frames(ax, episode)

    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, pad=8)
    ax.tick_params(colors=GRAY, width=0.6, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.6)
    return marker_indices


def _draw_timeline(
    ax: Axes,
    episode: EpisodeTrace,
    marker_indices: Sequence[int],
    *,
    collision_envelope_m: float,
    comfort_distance_m: float,
) -> None:
    time = episode.time_s
    ax.plot(time, episode.min_robot_ped_distance_m, color=INK, linewidth=1.3)
    ax.axhline(
        collision_envelope_m,
        color=RED,
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
        zorder=1,
    )
    distance_values = [
        *episode.min_robot_ped_distance_m,
        collision_envelope_m,
        comfort_distance_m,
    ]
    lower_data = min(distance_values)
    upper_data = max(distance_values)
    span = max(upper_data - lower_data, 1.0)
    padding = max(0.5, 0.08 * span)
    ax.set_ylim(max(0.0, lower_data - padding), upper_data + padding)
    ax.annotate(
        f"collision envelope ({collision_envelope_m:g} m)",
        (0.99, collision_envelope_m),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 3),
        textcoords="offset points",
        color=RED,
        alpha=0.7,
        fontsize=7,
        ha="right",
        va="bottom",
    )
    lower_y, upper_y = ax.get_ylim()
    if (
        not math.isclose(comfort_distance_m, collision_envelope_m)
        and lower_y <= comfort_distance_m <= upper_y
        and lower_y <= collision_envelope_m <= upper_y
    ):
        ax.axhline(
            comfort_distance_m,
            color=GRAY,
            linestyle="--",
            linewidth=0.7,
            alpha=0.5,
            zorder=1,
        )
        ax.annotate(
            "personal space",
            (0.62, comfort_distance_m),
            xycoords=ax.get_yaxis_transform(),
            xytext=(0, 3),
            textcoords="offset points",
            color=GRAY,
            alpha=0.7,
            fontsize=7,
            ha="right",
            va="bottom",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("min robot-ped distance (m)", color=INK, labelpad=7)
    ax.tick_params(axis="both", labelsize=8, width=0.6, pad=2)
    for marker_index in marker_indices:
        ax.axvline(time[marker_index], color=GRAY, linestyle="--", linewidth=0.6, alpha=0.7)
    min_index = _global_min_index(episode)
    ax.axvline(time[min_index], color=RED, linewidth=0.9, alpha=0.8)
    ax.plot(
        time[min_index],
        episode.min_robot_ped_distance_m[min_index],
        marker="o",
        markersize=4,
        color=RED,
    )
    speed_ax = ax.twinx()
    speed_ax.plot(time, episode.executed_speed_m_s, color=BLUE, linewidth=1.1)
    speed_ax.set_ylabel("speed (m/s)", color=BLUE)
    speed_ax.tick_params(axis="y", colors=BLUE, labelsize=8, width=0.6)
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.6)
    for spine in speed_ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.6)


def _prepare_output(out: Path) -> Path:
    out = Path(out)
    if not out.suffix:
        out = out.with_suffix(".pdf")
    if out.suffix.lower() not in {".pdf", ".png"}:
        raise ValueError("output path must end in .pdf or .png")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _save_figure(figure: Figure, out: Path, dpi: int, *, close: bool = True) -> Path:
    out = _prepare_output(out)
    save_options: dict[str, Any] = {"bbox_inches": "tight"}
    if out.suffix.lower() == ".png":
        save_options["dpi"] = dpi
    figure.savefig(out, **save_options)
    if close:
        plt.close(figure)
    return out


def render_scene(  # noqa: PLR0913 - public rendering controls are explicit keyword arguments
    episode: EpisodeTrace,
    out: Path,
    *,
    marker_interval_s: float | None = None,
    ped_focus_radius_m: float = 8.0,
    highlight_focal: bool = True,
    collision_envelope_m: float = DEFAULT_COLLISION_ENVELOPE_M,
    comfort_distance_m: float = DEFAULT_COMFORT_DISTANCE_M,
    timeline: bool = True,
    dpi: int = 300,
    return_figure: bool = False,
) -> Path | tuple[Path, Figure]:
    """Render one top-down exemplar trace figure and return the output path.

    Returns:
        Path | tuple[Path, Figure]: Output path, plus the open Figure when requested.
    """

    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")
    if collision_envelope_m <= 0 or comfort_distance_m <= 0:
        raise ValueError("timeline distance references must be greater than zero")
    effective_marker_interval_s = _effective_marker_interval([episode], marker_interval_s)
    map_definition = _load_map_definition(str(episode.metadata["scenario_id"]))
    focused, filtered_count = _focused_pedestrian_tracks(episode, ped_focus_radius_m)
    pedestrian_selection = _PedestrianSelection(
        tracks=focused,
        filtered_count=filtered_count,
        radius_m=ped_focus_radius_m,
    )
    limits = _compute_scene_extent([episode], [focused], map_definition)
    status = str(episode.metadata["episode_status"])
    title = (
        f"{episode.metadata['planner']} -- {episode.metadata['scenario_id']} "
        f"(seed {episode.metadata['seed']}) -- {status}"
    )

    width = 7.2
    scene_ratio = (limits[1][1] - limits[1][0]) / max(limits[0][1] - limits[0][0], 1.0)
    scene_height = min(8.6, max(4.8, width * scene_ratio))
    with plt.rc_context(_RC_PARAMS):
        if timeline:
            figure = plt.figure(figsize=(width, scene_height + 2.0), constrained_layout=True)
            grid = figure.add_gridspec(2, 1, height_ratios=(3.6, 1.0))
            scene_ax = figure.add_subplot(grid[0])
            timeline_ax = figure.add_subplot(grid[1])
        else:
            figure, scene_ax = plt.subplots(figsize=(width, scene_height))
            timeline_ax = None
        marker_indices = _draw_scene_panel(
            scene_ax,
            episode,
            map_definition,
            pedestrian_selection,
            effective_marker_interval_s,
            limits,
            title,
            highlight_focal=highlight_focal,
        )
        if timeline_ax is not None:
            _draw_timeline(
                timeline_ax,
                episode,
                marker_indices,
                collision_envelope_m=collision_envelope_m,
                comfort_distance_m=comfort_distance_m,
            )
        else:
            figure.subplots_adjust(left=0.12, right=0.96, bottom=0.09, top=0.93)
        _place_scene_annotations(scene_ax)
        figure.set_layout_engine("none")
        output = _save_figure(figure, out, dpi, close=not return_figure)
        return (output, figure) if return_figure else output


def render_comparison(  # noqa: PLR0913 - mirrors the single-scene rendering controls
    episodes: Sequence[EpisodeTrace],
    out: Path,
    *,
    marker_interval_s: float | None = None,
    ped_focus_radius_m: float = 8.0,
    highlight_focal: bool = True,
    collision_envelope_m: float = DEFAULT_COLLISION_ENVELOPE_M,
    comfort_distance_m: float = DEFAULT_COMFORT_DISTANCE_M,
    timeline: bool = True,
    dpi: int = 300,
    return_figure: bool = False,
) -> Path | tuple[Path, Figure]:
    """Render two same-scenario exemplar traces with shared scene limits.

    Returns:
        Path | tuple[Path, Figure]: Output path, plus the open Figure when requested.
    """

    if len(episodes) != 2:
        raise ValueError("comparison rendering requires exactly two episodes")
    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")
    if collision_envelope_m <= 0 or comfort_distance_m <= 0:
        raise ValueError("timeline distance references must be greater than zero")
    effective_marker_interval_s = _effective_marker_interval(episodes, marker_interval_s)
    scenario_ids = {str(episode.metadata["scenario_id"]) for episode in episodes}
    if len(scenario_ids) != 1:
        raise ValueError("comparison episodes must use the same scenario_id")
    map_definition = _load_map_definition(scenario_ids.pop())
    focused_results = []
    for episode in episodes:
        focused, filtered_count = _focused_pedestrian_tracks(episode, ped_focus_radius_m)
        focused_results.append(
            _PedestrianSelection(
                tracks=focused,
                filtered_count=filtered_count,
                radius_m=ped_focus_radius_m,
            )
        )
    limits = _compute_scene_extent(
        episodes, [selection.tracks for selection in focused_results], map_definition
    )

    with plt.rc_context(_RC_PARAMS):
        figure = plt.figure(
            figsize=(10.8, 8.0 if timeline else 6.2),
            constrained_layout=True,
        )
        if timeline:
            grid = figure.add_gridspec(2, 2, height_ratios=(3.6, 1.0))
            scene_axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
            timeline_axes = [figure.add_subplot(grid[1, column]) for column in range(2)]
        else:
            grid = figure.add_gridspec(1, 2)
            scene_axes = [figure.add_subplot(grid[0, column]) for column in range(2)]
            timeline_axes = [None, None]

        for episode, pedestrian_selection, scene_ax, timeline_ax in zip(
            episodes, focused_results, scene_axes, timeline_axes, strict=True
        ):
            summary = _mapping(episode.metadata.get("summary"), "metadata.summary")
            title = (
                f"{episode.metadata['planner']} -- {episode.metadata['episode_status']} "
                f"(min d = {float(summary['global_min_robot_ped_distance_m']):.2f} m)"
            )
            marker_indices = _draw_scene_panel(
                scene_ax,
                episode,
                map_definition,
                pedestrian_selection,
                effective_marker_interval_s,
                limits,
                title,
                highlight_focal=highlight_focal,
            )
            if timeline_ax is not None:
                _draw_timeline(
                    timeline_ax,
                    episode,
                    marker_indices,
                    collision_envelope_m=collision_envelope_m,
                    comfort_distance_m=comfort_distance_m,
                )

        legend_handle = Line2D(
            [],
            [],
            color=INK,
            marker="o",
            markersize=3.5,
            linewidth=0.8,
            label=f"time-synchronized markers every {effective_marker_interval_s:g} s",
        )
        figure.legend(
            handles=[legend_handle],
            loc="outside lower center",
            frameon=False,
            fontsize=8,
        )
        figure.canvas.draw()
        for scene_ax in scene_axes:
            _place_scene_annotations(scene_ax)
        figure.set_layout_engine("none")
        output = _save_figure(figure, out, dpi, close=not return_figure)
        return (output, figure) if return_figure else output


__all__ = [
    "BLUE",
    "CONTEXT_PEDESTRIAN_ALPHA",
    "CONTEXT_PEDESTRIAN_COLOR",
    "DEFAULT_COLLISION_ENVELOPE_M",
    "DEFAULT_COMFORT_DISTANCE_M",
    "FOCAL_PEDESTRIAN_COLOR",
    "GRAY",
    "GREEN",
    "INK",
    "LABEL_MIN_GAP_M",
    "ORANGE",
    "PURPLE",
    "RED",
    "TEAL",
    "EpisodeTrace",
    "TraceSchemaError",
    "load_episode",
    "render_comparison",
    "render_scene",
]
