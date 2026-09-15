"""Export JSONL simulation recordings for an optional Three.js viewer."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.render.jsonl_playback import JSONLPlaybackLoader, PlaybackEpisode
from robot_sf.render.presentation_scene import describe_view

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition
    from robot_sf.nav.obstacle import Obstacle
    from robot_sf.render.sim_view import VisualizableSimState


SCENE_SCHEMA_VERSION = "threejs-viewer.v1"

#: Symbolic pedestrian radius (metres) used only when the source supplies no
#: recorded radius. Renderers must disclose symbolic geometry, never present
#: it as measured (issue #9368).
SYMBOLIC_PED_RADIUS_M = 0.24


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
        "fidelity": _fidelity_block(frames),
        "limitations": [
            "Initial Three.js viewer slice for qualitative playback review.",
            "Pygame remains the reference renderer for full overlay parity.",
            "Pedestrian geometry without a recorded radius renders symbolic; "
            "see fidelity.geometry for the per-export mode.",
            "Pedestrian identity without recorded ids is slot-index based; "
            "see fidelity.identity. Trails must not assume persistence.",
        ],
    }


def export_threejs_viewer(
    recording_path: str | Path,
    output_dir: str | Path,
    *,
    view_preset: str | None = None,
) -> ThreeJSExportResult:
    """Export a JSONL or pickle recording into static browser viewer files.

    When ``view_preset`` is ``"top-down"`` or ``"isometric"``, the scene gains
    a presentation-view descriptor and the export additionally stages the
    ``components/presentation_scene`` web module. ``None`` keeps the legacy
    export byte-identical.

    Returns:
        ThreeJSExportResult: Paths to the generated viewer directory, HTML file, and scene JSON.
    """
    recording_path = Path(recording_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode, map_def = JSONLPlaybackLoader().load_single_episode(recording_path)
    scene = build_threejs_scene(episode, map_def, source=str(recording_path))
    if view_preset is not None:
        scene["view"] = describe_view(scene, view_preset)  # type: ignore[arg-type]
        _copy_web_asset_tree("components", output_dir / "components")

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


def _copy_web_asset_tree(asset_name: str, destination: Path) -> None:
    """Copy a packaged static web asset directory into an export directory."""
    asset = resources.files("robot_sf.render.web_assets").joinpath(asset_name)
    with resources.as_file(asset) as asset_path:
        shutil.copytree(asset_path, destination, dirs_exist_ok=True)


def _map_to_payload(map_def: MapDefinition) -> dict[str, Any]:
    """Convert a map definition into JSON-safe scalar geometry lists.

    Returns:
        dict[str, Any]: Map dimensions, bounds, obstacles, and zones as JSON-safe lists.
    """
    return {
        "width": float(map_def.width),
        "height": float(map_def.height),
        "origin": _map_origin_to_list(map_def),
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


def _fidelity_block(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize source-fidelity modes for one exported scene (issue #9368).

    Timing is ``monotonic`` only when every frame carries finite, non-decreasing
    ``time_s``; otherwise the offending frame indexes are named so playback can
    refuse timestamp-driven advance. Identity is ``stable`` only when every
    pedestrian entry in every frame carries a recorded identity; geometry is
    ``recorded`` only when every pedestrian radius is recorded.

    Returns:
        dict[str, Any]: JSON-safe fidelity summary.
    """
    times = [frame.get("time_s") for frame in frames]
    bad_time_frames = [
        idx
        for idx, value in enumerate(times)
        if not isinstance(value, (int, float)) or not math.isfinite(value)
    ]
    nonmonotonic_frames = [
        idx
        for idx in range(1, len(times))
        if idx not in bad_time_frames
        and idx - 1 not in bad_time_frames
        and times[idx] < times[idx - 1]
    ]
    identity_stable = all(
        ped.get("identity_source") == "stable"
        for frame in frames
        for ped in frame.get("pedestrians", [])
    )
    has_peds = any(frame.get("pedestrians") for frame in frames)
    geometry_recorded = has_peds and all(
        ped.get("radius_source") == "recorded"
        for frame in frames
        for ped in frame.get("pedestrians", [])
    )
    explicit_timing = all(frame.get("timing_source") == "explicit_time_s" for frame in frames)
    return {
        "timing": {
            "mode": "monotonic"
            if not bad_time_frames and not nonmonotonic_frames
            else "nonmonotonic_time_s",
            "explicit_time_s": explicit_timing,
            "bad_time_frames": bad_time_frames,
            "nonmonotonic_frames": nonmonotonic_frames,
        },
        "identity": {"mode": "stable" if identity_stable and has_peds else "slot_index"},
        "geometry": {
            "mode": "recorded" if geometry_recorded else "symbolic",
            "symbolic_radius_m": SYMBOLIC_PED_RADIUS_M,
        },
    }


def _map_origin_to_list(map_def: MapDefinition) -> list[float]:
    """Return the map origin as JSON-safe floats, defaulting to ``[0.0, 0.0]``.

    The viewer previously assumed a zero origin; an explicit field keeps
    asymmetric and negative-origin maps aligned (issue #9368).

    Returns:
        list[float]: Two origin coordinates.
    """
    origin = getattr(map_def, "origin", None)
    try:
        values = [float(origin[0]), float(origin[1])]
    except (TypeError, ValueError, IndexError):
        return [0.0, 0.0]
    if not all(math.isfinite(value) for value in values):
        return [0.0, 0.0]
    return values


def _obstacle_to_payload(obstacle: Obstacle) -> dict[str, Any]:
    """Convert an obstacle object into JSON-safe vertices and line segments.

    Returns:
        dict[str, Any]: Obstacle vertices and lines as float lists.
    """
    return {
        "vertices": [_point_to_list(vertex) for vertex in obstacle.vertices],
        "lines": [_line_to_list(line) for line in obstacle.lines],
    }


def _finite_float(value: Any, default: float) -> tuple[float, bool]:
    """Return ``(finite float, was_recorded)`` for an optional numeric field."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default, False
    if not math.isfinite(number):
        return default, False
    return number, True


def _json_scalar_id(value: Any) -> str | int | float | None:
    """Return a JSON-safe stable identity, or ``None`` when not representable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value if value else None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_scalar_id(item())
        except (TypeError, ValueError):
            return None
    return None


def _finite_float(value: Any, default: float) -> tuple[float, bool]:
    """Return ``(finite float, was_recorded)`` for an optional numeric field."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default, False
    if not math.isfinite(number):
        return default, False
    return number, True


def _state_to_frame(state: VisualizableSimState, frame_idx: int) -> dict[str, Any]:
    """Convert one visualizable state into a JSON-safe animation frame.

    Source time prefers an explicitly recorded ``time_s`` attribute and falls
    back to ``timestep * dt``; the provenance travels in ``timing_source`` so
    playback can refuse to present guessed seconds as authoritative (#9368).

    Returns:
        dict[str, Any]: Frame metadata, poses, pedestrians, rays, and planned path.
    """
    dt, _ = _finite_float(getattr(state, "time_per_step_in_secs", None), 0.1)
    if dt <= 0.0:
        dt = 0.1
    recorded_time, has_recorded_time = _finite_float(getattr(state, "time_s", None), 0.0)
    timestep = int(getattr(state, "timestep", frame_idx))
    if has_recorded_time:
        time_s = recorded_time
        timing_source = "explicit_time_s"
    else:
        time_s = float(timestep) * dt
        timing_source = "timestep_times_dt"
    frame = {
        "frame_idx": frame_idx,
        "timestep": timestep,
        "time_s": time_s,
        "timing_source": timing_source,
        "dt_s": dt,
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
    """Convert optional pedestrian positions into JSON-safe frame entries.

    Stable ``pedestrian_ids``/``pedestrian_headings``/``pedestrian_radii``
    attributes are recorded verbatim when present; otherwise the payload
    carries slot-index identity, unknown heading, and symbolic radius with
    explicit per-field sources so renderers cannot present guesses as
    measured (issue #9368).

    Returns:
        list[dict[str, Any]]: Pedestrian entries, or an empty list.
    """
    positions = getattr(state, "pedestrian_positions", None)
    if positions is None:
        return []
    raw_ids = getattr(state, "pedestrian_ids", None)
    raw_headings = getattr(state, "pedestrian_headings", None)
    raw_radii = getattr(state, "pedestrian_radii", None)
    entries: list[dict[str, Any]] = []
    for idx, position in enumerate(positions):
        if len(position) < 2:
            continue
        ped_id: Any = idx
        identity_source = "slot_index"
        if raw_ids is not None and idx < len(raw_ids):
            stable_id = _json_scalar_id(raw_ids[idx])
            if stable_id is not None:
                ped_id = stable_id
                identity_source = "stable"
        heading: float | None = None
        heading_source = "unknown"
        if raw_headings is not None and idx < len(raw_headings):
            value, recorded = _finite_float(raw_headings[idx], 0.0)
            if recorded:
                heading = value
                heading_source = "recorded"
        radius_m: float | None = None
        radius_source = "unknown"
        if raw_radii is not None and idx < len(raw_radii):
            value, recorded = _finite_float(raw_radii[idx], 0.0)
            if recorded and value > 0.0:
                radius_m = value
                radius_source = "recorded"
        entries.append(
            {
                "id": ped_id,
                "identity_source": identity_source,
                "position": _point_to_list(position),
                "heading": heading,
                "heading_source": heading_source,
                "radius_m": radius_m,
                "radius_source": radius_source,
            }
        )
    return entries


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
