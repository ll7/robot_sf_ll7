"""Export ``simulation_trace_export.v1`` traces for a static Three.js viewer."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from importlib import resources
from numbers import Real
from pathlib import Path
from typing import Any

from loguru import logger

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceExport,
    load_simulation_trace_export,
)
from robot_sf.analysis_workbench.trace_annotation import load_trace_annotation_set
from robot_sf.render.threejs_viewer import SCENE_SCHEMA_VERSION

TRACE_VIEWER_SCENE_VERSION = "trace-viewer.v1"


@dataclass(frozen=True)
class TraceViewerResult:
    """Files written for a browser-viewable trace export."""

    output_dir: Path
    html_path: Path
    scene_path: Path


_MAP_GEOMETRY_ZONE_KEYS = (
    "robot_spawn_zones",
    "robot_goal_zones",
    "ped_spawn_zones",
    "ped_goal_zones",
    "ped_crowded_zones",
)
_MAP_GEOMETRY_OBSTACLE_KEY = "obstacles"
_MAP_GEOMETRY_KNOWN_KEYS = {_MAP_GEOMETRY_OBSTACLE_KEY} | set(_MAP_GEOMETRY_ZONE_KEYS)
_MAP_GEOMETRY_OBSTACLE_KEYS = {"vertices", "lines"}


def build_trace_scene(
    trace: SimulationTraceExport,
    *,
    source: str | None = None,
    annotations: list[dict[str, Any]] | None = None,
    map_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the renderer-neutral scene payload from a simulation trace export.

    Args:
        trace: A loaded simulation trace export.
        source: Optional source identifier for the scene payload.
        annotations: Optional list of annotation dicts to embed in the scene.
        map_geometry: Optional map geometry payload with ``obstacles`` and zone
            lists.  When supplied the geometry is merged into the auto-computed
            map; when ``None`` the auto-computed empty-layout map is used.
            Raises ``ValueError`` for unrecognised keys or malformed entries.

    Returns:
        dict[str, Any]: JSON-safe scene payload with auto-computed map bounds,
        animation frames, trace metadata, and optional annotations.
    """
    frames = [_trace_frame_to_scene_frame(frame) for frame in trace.frames]
    if not frames:
        raise ValueError("Trace viewer export requires at least one trace frame")

    validated_geometry = _validate_map_geometry(map_geometry) if map_geometry is not None else None
    map_payload = _compute_trace_map(trace, map_geometry=validated_geometry)
    if validated_geometry is not None:
        _merge_map_geometry(map_payload, validated_geometry)

    trajectory = [
        frame["robot"]["position"]
        for frame in frames
        if frame.get("robot") and frame["robot"].get("position")
    ]

    diagnostic_only = True

    limitations = [
        "Trace viewer for qualitative review of simulation_trace_export.v1 fixtures.",
    ]
    if map_geometry is not None:
        limitations.append(
            "Map geometry overlaid from supplied metadata; not ground-truth map validation."
        )
    else:
        limitations.append(
            "Map bounds are auto-computed from trace positions; no SVG map geometry is available."
        )
    limitations.append(
        "This viewer is diagnostic-only; not benchmark evidence.",
    )

    scene: dict[str, Any] = {
        "schema_version": SCENE_SCHEMA_VERSION,
        "trace_viewer_version": TRACE_VIEWER_SCENE_VERSION,
        "source": source,
        "trace_id": trace.trace_id,
        "episode_id": trace.source.episode_id,
        "metadata": {
            "schema_version": trace.schema_version,
            "coordinate_frame": trace.coordinate_frame,
            "units": {key: str(val) for key, val in (trace.units or {}).items()},
            "evidence_boundary": trace.evidence_boundary,
            "diagnostic_only": diagnostic_only,
            "source": {
                "scenario_id": trace.source.scenario_id,
                "seed": trace.source.seed,
                "planner_id": trace.source.planner_id,
                "episode_id": trace.source.episode_id,
                "generated_by": trace.source.generated_by,
            },
        },
        "map": map_payload,
        "frames": frames,
        "trajectory": trajectory,
        "reset_points": [],
        "limitations": limitations,
    }
    if annotations:
        scene["annotations"] = list(annotations)
    return scene


def _compute_trace_map(
    trace: SimulationTraceExport,
    *,
    map_geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Auto-compute map bounds and a minimal empty layout from scene coordinates.

    Returns:
        dict[str, Any]: Map payload with computed bounds and empty obstacle/zone lists.
    """
    all_x: list[float] = []
    all_y: list[float] = []
    for frame in trace.frames:
        robot_pos = frame.robot.get("position")
        if isinstance(robot_pos, (list, tuple)) and len(robot_pos) >= 2:
            all_x.append(float(robot_pos[0]))
            all_y.append(float(robot_pos[1]))
        for ped in frame.pedestrians:
            ped_pos = ped.get("position")
            if isinstance(ped_pos, (list, tuple)) and len(ped_pos) >= 2:
                all_x.append(float(ped_pos[0]))
                all_y.append(float(ped_pos[1]))

    if map_geometry is not None:
        for x, y in _map_geometry_points(map_geometry):
            all_x.append(x)
            all_y.append(y)

    if not all_x or not all_y:
        origin_x, origin_y = 0.0, 0.0
        width, height = 10.0, 10.0
    else:
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        padding = max((max_x - min_x) * 0.2, (max_y - min_y) * 0.2, 1.0)
        origin_x = min_x - padding
        origin_y = min_y - padding
        width = max_x - min_x + padding * 2
        height = max_y - min_y + padding * 2

    return {
        "origin": [origin_x, origin_y],
        "width": width,
        "height": height,
        "bounds": [
            [origin_x, origin_x + width, origin_y, origin_y],
            [origin_x, origin_x + width, origin_y + height, origin_y + height],
            [origin_x, origin_x, origin_y, origin_y + height],
            [origin_x + width, origin_x + width, origin_y, origin_y + height],
        ],
        "obstacles": [],
        "robot_spawn_zones": [],
        "robot_goal_zones": [],
        "ped_spawn_zones": [],
        "ped_goal_zones": [],
        "ped_crowded_zones": [],
        "_padding": max(padding, 1.0) if all_x else 1.0,
    }


def _validate_coordinate(value: Any, *, path: str) -> float:
    """Return one finite renderer-safe coordinate.

    Raises:
        ValueError: If the value is not a finite real number.
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{path}: expected a finite number")
    return float(value)


def _validate_point(value: Any, *, path: str) -> list[float]:
    """Return an exact two-coordinate point suitable for the Three.js renderer.

    Raises:
        ValueError: If the point is not an exact finite ``[x, y]`` pair.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path}: expected [x, y] point pair")
    return [
        _validate_coordinate(value[0], path=f"{path}[0]"),
        _validate_coordinate(value[1], path=f"{path}[1]"),
    ]


def _validate_line(value: Any, *, path: str) -> list[float]:
    """Return one flat ``[x_start, x_end, y_start, y_end]`` line segment.

    Raises:
        ValueError: If the segment cannot be consumed by ``makeLine`` in the browser viewer.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{path}: expected [x_start, x_end, y_start, y_end] line segment")
    return [
        _validate_coordinate(coordinate, path=f"{path}[{index}]")
        for index, coordinate in enumerate(value)
    ]


def _validate_obstacles(obstacles: list[Any]) -> list[dict[str, list[list[float]]]]:
    """Validate and normalize obstacle list entries.

    Returns:
        list[dict[str, list[list[float]]]]: JSON-safe obstacle payloads.

    Raises:
        ValueError: If any obstacle is malformed.
    """
    normalized: list[dict[str, list[list[float]]]] = []
    for i, obs in enumerate(obstacles):
        if not isinstance(obs, dict):
            raise ValueError(f"map_geometry.obstacles[{i}]: expected a dict")
        unknown_keys = set(obs) - _MAP_GEOMETRY_OBSTACLE_KEYS
        if unknown_keys:
            raise ValueError(
                f"map_geometry.obstacles[{i}]: unrecognised key {sorted(unknown_keys)[0]!r}"
            )
        for field in ("vertices", "lines"):
            if field not in obs:
                raise ValueError(f"map_geometry.obstacles[{i}]: missing '{field}'")
            if not isinstance(obs[field], list):
                raise ValueError(
                    f"map_geometry.obstacles[{i}].{field}:"
                    f" expected a list, got {type(obs[field]).__name__}"
                )
        normalized.append(
            {
                "vertices": [
                    _validate_point(vertex, path=f"map_geometry.obstacles[{i}].vertices[{j}]")
                    for j, vertex in enumerate(obs["vertices"])
                ],
                "lines": [
                    _validate_line(line, path=f"map_geometry.obstacles[{i}].lines[{j}]")
                    for j, line in enumerate(obs["lines"])
                ],
            }
        )
    return normalized


def _validate_zone_list(key: str, zones: list[Any]) -> list[list[list[float]]]:
    """Validate and normalize zone polygons for the Three.js renderer.

    Returns:
        list[list[list[float]]]: JSON-safe zone polygons.

    Raises:
        ValueError: If any zone or point is malformed.
    """
    normalized: list[list[list[float]]] = []
    for i, zone in enumerate(zones):
        if not isinstance(zone, list):
            raise ValueError(f"map_geometry.{key}[{i}]: expected a list, got {type(zone).__name__}")
        if len(zone) < 3:
            raise ValueError(
                f"map_geometry.{key}[{i}]: expected a polygon with at least three points"
            )
        normalized.append(
            [
                _validate_point(point, path=f"map_geometry.{key}[{i}][{j}]")
                for j, point in enumerate(zone)
            ]
        )
    return normalized


def _validate_map_geometry(geometry: Any) -> dict[str, Any]:
    """Validate and normalize optional geometry before it reaches the renderer.

    Returns:
        dict[str, Any]: Renderer-safe JSON payload with only supported geometry keys.

    Raises:
        ValueError: If geometry is not a supported JSON-object payload or contains invalid data.
    """
    if not isinstance(geometry, dict):
        raise ValueError(f"map_geometry: expected an object, got {type(geometry).__name__}")

    normalized: dict[str, Any] = {}
    for key, value in geometry.items():
        if key not in _MAP_GEOMETRY_KNOWN_KEYS:
            raise ValueError(f"map_geometry: unrecognised key '{key}'")
        if not isinstance(value, list):
            raise ValueError(f"map_geometry.{key}: expected a list, got {type(value).__name__}")
        if key == _MAP_GEOMETRY_OBSTACLE_KEY:
            normalized[key] = _validate_obstacles(value)
        else:
            normalized[key] = _validate_zone_list(key, value)
    return normalized


def _merge_map_geometry(map_payload: dict[str, Any], geometry: dict[str, Any]) -> None:
    """Merge validated optional map geometry into a computed map payload.

    Args:
        map_payload: Auto-computed map payload (mutated in place).
        geometry: Renderer-safe, validated geometry dict.
    """
    map_payload.update(geometry)


def _map_geometry_points(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    """Collect all validated geometry endpoints for camera-bound calculation.

    Returns:
        list[tuple[float, float]]: All obstacle and zone vertices in world coordinates.
    """
    points: list[tuple[float, float]] = []
    for obstacle in geometry.get(_MAP_GEOMETRY_OBSTACLE_KEY, []):
        points.extend((vertex[0], vertex[1]) for vertex in obstacle["vertices"])
        for x_start, x_end, y_start, y_end in obstacle["lines"]:
            points.extend(((x_start, y_start), (x_end, y_end)))
    for key in _MAP_GEOMETRY_ZONE_KEYS:
        for zone in geometry.get(key, []):
            points.extend((point[0], point[1]) for point in zone)
    return points


def _trace_frame_to_scene_frame(frame: Any) -> dict[str, Any]:
    """Convert one simulation trace frame to a Three.js-compatible animation frame.

    Returns:
        dict[str, Any]: Frame dict with robot pose, pedestrians, and planner metadata.
    """
    robot_pose = frame.robot
    robot_payload: dict[str, Any] | None = None
    if robot_pose.get("position") is not None:
        pos = robot_pose["position"]
        robot_payload = {
            "position": [float(pos[0]), float(pos[1])],
            "heading": float(robot_pose.get("heading", 0.0)),
            "velocity": (
                [float(v) for v in robot_pose["velocity"]] if robot_pose.get("velocity") else None
            ),
        }

    pedestrians: list[dict[str, Any]] = []
    for ped in frame.pedestrians:
        ped_pos = ped.get("position")
        if ped_pos is not None:
            ped_entry: dict[str, Any] = {
                "id": str(ped.get("id", "")),
                "position": [float(ped_pos[0]), float(ped_pos[1])],
            }
            if ped.get("velocity"):
                ped_entry["velocity"] = [float(v) for v in ped["velocity"]]
            pedestrians.append(ped_entry)

    planner = frame.planner
    scene_frame: dict[str, Any] = {
        "frame_idx": frame.step,
        "timestep": frame.step,
        "time_s": frame.time_s,
        "robot": robot_payload,
        "pedestrians": pedestrians,
        "rays": [],
        "planned_path": [],
    }

    event = planner.get("event")
    if event is not None:
        scene_frame["event"] = str(event)
    event_id = planner.get("event_id")
    if event_id is not None:
        scene_frame["event_id"] = str(event_id)

    selected_action = planner.get("selected_action")
    if selected_action is not None:
        scene_frame["planner_action"] = {
            "linear_velocity": float(selected_action.get("linear_velocity", 0.0)),
            "angular_velocity": float(selected_action.get("angular_velocity", 0.0)),
        }

    return scene_frame


def export_trace_viewer(
    trace: SimulationTraceExport,
    output_dir: str | Path,
    *,
    source: str | None = None,
    annotations: list[dict[str, Any]] | None = None,
    map_geometry: dict[str, Any] | None = None,
) -> TraceViewerResult:
    """Export a simulation trace export into static browser viewer files.

    Args:
        trace: A loaded simulation trace export.
        output_dir: Directory for the viewer files.
        source: Optional source identifier.
        annotations: Optional annotations to embed.
        map_geometry: Optional map geometry payload forwarded to
            :func:`build_trace_scene`.

    Returns:
        TraceViewerResult: Paths to the generated viewer directory, HTML file, and scene JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene = build_trace_scene(
        trace,
        source=source,
        annotations=annotations,
        map_geometry=map_geometry,
    )

    scene_path = output_path / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    html_path = output_path / "index.html"
    _copy_web_asset("index.html", html_path)
    _copy_web_asset("viewer.js", output_path / "viewer.js")

    return TraceViewerResult(output_dir=output_path, html_path=html_path, scene_path=scene_path)


def _copy_web_asset(asset_name: str, destination: Path) -> None:
    """Copy a packaged static web asset into an export directory."""
    asset = resources.files("robot_sf.render.web_assets").joinpath(asset_name)
    with resources.as_file(asset) as asset_path:
        shutil.copyfile(asset_path, destination)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for exporting a static Three.js trace viewer.

    Returns:
        int: Process exit status code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Path to a simulation_trace_export.v1 JSON fixture")
    parser.add_argument(
        "--output-dir",
        default="output/trace_viewer",
        help="Directory for index.html, viewer.js, and scene.json",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Optional path to a trace_annotation_set.v1 JSON fixture",
    )
    parser.add_argument(
        "--map-geometry",
        default=None,
        help="Optional path to a JSON object with renderer-safe obstacle and zone geometry",
    )
    args = parser.parse_args(argv)

    try:
        trace_path = Path(args.trace)
        trace = load_simulation_trace_export(trace_path)
        annotations_payload = _load_annotation_payload(
            Path(args.annotations) if args.annotations else None,
            trace_id=trace.trace_id,
        )
        map_geometry = _load_map_geometry(Path(args.map_geometry) if args.map_geometry else None)
        result = export_trace_viewer(
            trace,
            args.output_dir,
            source=str(trace_path),
            annotations=annotations_payload,
            map_geometry=map_geometry,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    logger.info("Wrote trace viewer: {}", result.html_path)
    logger.info("Wrote scene payload: {}", result.scene_path)
    return 0


def _load_annotation_payload(
    annotation_path: Path | None,
    *,
    trace_id: str,
) -> list[dict[str, Any]] | None:
    """Load optional annotation payload and ensure it matches the selected trace.

    Returns:
        Annotation dictionaries for the scene payload, or ``None`` when no annotation path is
        supplied.
    """
    if annotation_path is None:
        return None
    annotation_set = load_trace_annotation_set(annotation_path)
    if annotation_set.timeline.trace_id != trace_id:
        raise ValueError(
            "annotation trace_id does not match supplied trace: "
            f"{annotation_set.timeline.trace_id!r} != {trace_id!r}"
        )
    return [
        {
            "annotation_id": a.annotation_id,
            "category": a.category,
            "evidence_type": a.evidence_type,
            "anchor": {
                "frame_start": a.anchor.frame_start,
                "frame_end": a.anchor.frame_end,
                "event_ids": list(a.anchor.event_ids),
                "entities": [{"type": e.type, "id": e.id} for e in a.anchor.entities],
            },
            "summary": a.summary,
            "details": a.details,
        }
        for a in annotation_set.annotations
    ]


def _load_map_geometry(path: Path | None) -> dict[str, Any] | None:
    """Load the optional renderer-safe map geometry JSON object.

    Returns:
        dict[str, Any] | None: Parsed geometry object, or ``None`` without a path.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"map geometry file {path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
