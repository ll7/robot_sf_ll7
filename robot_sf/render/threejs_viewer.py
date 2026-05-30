"""Export JSONL simulation recordings for an optional Three.js viewer."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.render.jsonl_playback import JSONLPlaybackLoader, PlaybackEpisode

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition
    from robot_sf.nav.obstacle import Obstacle
    from robot_sf.render.sim_view import VisualizableSimState


SCENE_SCHEMA_VERSION = "threejs-viewer.v1"


@dataclass(frozen=True)
class ThreeJSExportResult:
    """Files written for a browser-viewable recording export."""

    output_dir: Path
    html_path: Path
    scene_path: Path


def build_threejs_scene(
    episode: PlaybackEpisode,
    map_def: MapDefinition,
    *,
    source: str | None = None,
) -> dict[str, Any]:
    """Build the renderer-neutral scene payload consumed by the web viewer.

    Returns:
        dict[str, Any]: JSON-safe scene payload with map geometry, zones, and animation frames.
    """
    frames = [_state_to_frame(state, frame_idx) for frame_idx, state in enumerate(episode.states)]
    if not frames:
        raise ValueError("Three.js viewer export requires at least one playback frame")

    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "source": source,
        "episode_id": episode.episode_id,
        "metadata": episode.metadata or {},
        "map": _map_to_payload(map_def),
        "frames": frames,
        "trajectory": [frame["robot"]["position"] for frame in frames if frame.get("robot")],
        "reset_points": episode.reset_points or [],
        "limitations": [
            "Initial Three.js viewer slice for qualitative playback review.",
            "Pygame remains the reference renderer for full overlay parity.",
        ],
    }


def export_threejs_viewer(
    recording_path: str | Path, output_dir: str | Path
) -> ThreeJSExportResult:
    """Export a JSONL or pickle recording into static browser viewer files.

    Returns:
        ThreeJSExportResult: Paths to the generated viewer directory, HTML file, and scene JSON.
    """
    recording_path = Path(recording_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode, map_def = JSONLPlaybackLoader().load_single_episode(recording_path)
    scene = build_threejs_scene(episode, map_def, source=str(recording_path))

    scene_path = output_dir / "scene.json"
    scene_path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    html_path = output_dir / "index.html"
    _copy_web_asset("index.html", html_path)
    _copy_web_asset("viewer.js", output_dir / "viewer.js")

    return ThreeJSExportResult(output_dir=output_dir, html_path=html_path, scene_path=scene_path)


def _copy_web_asset(asset_name: str, destination: Path) -> None:
    """Copy a packaged static web asset into an export directory."""
    asset = resources.files("robot_sf.render.web_assets").joinpath(asset_name)
    with resources.as_file(asset) as asset_path:
        shutil.copyfile(asset_path, destination)


def _map_to_payload(map_def: MapDefinition) -> dict[str, Any]:
    """Convert a map definition into JSON-safe scalar geometry lists.

    Returns:
        dict[str, Any]: Map dimensions, bounds, obstacles, and zones as JSON-safe lists.
    """
    return {
        "width": float(map_def.width),
        "height": float(map_def.height),
        "bounds": [_line_to_list(bound) for bound in getattr(map_def, "bounds", [])],
        "obstacles": [
            _obstacle_to_payload(obstacle) for obstacle in getattr(map_def, "obstacles", [])
        ],
        "robot_spawn_zones": [
            _zone_to_payload(zone) for zone in getattr(map_def, "robot_spawn_zones", [])
        ],
        "robot_goal_zones": [
            _zone_to_payload(zone) for zone in getattr(map_def, "robot_goal_zones", [])
        ],
        "ped_spawn_zones": [
            _zone_to_payload(zone) for zone in getattr(map_def, "ped_spawn_zones", [])
        ],
        "ped_goal_zones": [
            _zone_to_payload(zone) for zone in getattr(map_def, "ped_goal_zones", [])
        ],
        "ped_crowded_zones": [
            _zone_to_payload(zone) for zone in getattr(map_def, "ped_crowded_zones", [])
        ],
    }


def _obstacle_to_payload(obstacle: Obstacle) -> dict[str, Any]:
    """Convert an obstacle object into JSON-safe vertices and line segments.

    Returns:
        dict[str, Any]: Obstacle vertices and lines as float lists.
    """
    return {
        "vertices": [_point_to_list(vertex) for vertex in obstacle.vertices],
        "lines": [_line_to_list(line) for line in obstacle.lines],
    }


def _state_to_frame(state: VisualizableSimState, frame_idx: int) -> dict[str, Any]:
    """Convert one visualizable state into a JSON-safe animation frame.

    Returns:
        dict[str, Any]: Frame metadata, poses, pedestrians, rays, and planned path.
    """
    frame = {
        "frame_idx": frame_idx,
        "timestep": int(getattr(state, "timestep", frame_idx)),
        "time_s": float(getattr(state, "timestep", frame_idx))
        * float(getattr(state, "time_per_step_in_secs", 0.1) or 0.1),
        "robot": _pose_to_payload(getattr(state, "robot_pose", None)),
        "pedestrians": _pedestrians_to_payload(state),
        "rays": _vectors_to_payload(getattr(state, "ray_vecs", None)),
        "planned_path": [
            _point_to_list(point) for point in (getattr(state, "planned_path", None) or [])
        ],
    }
    ego_ped_pose = getattr(state, "ego_ped_pose", None)
    if ego_ped_pose is not None:
        frame["ego_pedestrian"] = _pose_to_payload(ego_ped_pose)
    return frame


def _pedestrians_to_payload(state: VisualizableSimState) -> list[dict[str, Any]]:
    """Convert optional pedestrian positions into JSON-safe id/position entries.

    Returns:
        list[dict[str, Any]]: Pedestrian ids and two-dimensional positions, or an empty list.
    """
    positions = getattr(state, "pedestrian_positions", None)
    if positions is None:
        return []
    return [
        {"id": idx, "position": _point_to_list(position)}
        for idx, position in enumerate(positions)
        if len(position) >= 2
    ]


def _pose_to_payload(pose: Any) -> dict[str, Any] | None:
    """Convert an optional ``(position, heading)`` pose into a JSON-safe dict.

    Returns:
        dict[str, Any] | None: Position and heading, or ``None`` for absent poses.
    """
    if pose is None:
        return None
    position, heading = pose
    return {"position": _point_to_list(position), "heading": float(heading)}


def _vectors_to_payload(vectors: Any) -> list[list[list[float]]]:
    """Convert optional ray vectors into nested JSON-safe float lists.

    Returns:
        list[list[list[float]]]: Ray vectors, or an empty list when absent.
    """
    if vectors is None:
        return []
    if hasattr(vectors, "tolist"):
        vectors = vectors.tolist()
    return vectors


def _zone_to_payload(zone: Any) -> list[list[float]]:
    """Convert a polygon-like zone into JSON-safe point lists.

    Returns:
        list[list[float]]: Zone vertices as ``[x, y]`` lists.
    """
    return [_point_to_list(point) for point in zone]


def _line_to_list(line: Any) -> list[float]:
    """Convert a four-value line-like object into JSON-safe floats.

    Returns:
        list[float]: Four line coordinates.
    """
    return [float(value) for value in line]


def _point_to_list(point: Any) -> list[float]:
    """Convert a two-value point-like object into a JSON-safe ``[x, y]`` list.

    Returns:
        list[float]: Two point coordinates.
    """
    return [float(point[0]), float(point[1])]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for exporting a static Three.js recording viewer.

    Returns:
        int: Process exit status code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", help="JSONL or legacy pickle recording to export")
    parser.add_argument(
        "--output-dir",
        default="output/threejs_viewer",
        help="Directory for index.html, viewer.js, and scene.json",
    )
    args = parser.parse_args(argv)

    result = export_threejs_viewer(args.recording, args.output_dir)
    logger.info("Wrote Three.js viewer: {}", result.html_path)
    logger.info("Wrote scene payload: {}", result.scene_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
