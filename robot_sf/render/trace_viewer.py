"""Export ``simulation_trace_export.v1`` traces for a static Three.js viewer."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from importlib import resources
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


def build_trace_scene(
    trace: SimulationTraceExport,
    *,
    source: str | None = None,
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the renderer-neutral scene payload from a simulation trace export.

    Args:
        trace: A loaded simulation trace export.
        source: Optional source identifier for the scene payload.
        annotations: Optional list of annotation dicts to embed in the scene.

    Returns:
        dict[str, Any]: JSON-safe scene payload with auto-computed map bounds,
        animation frames, trace metadata, and optional annotations.
    """
    frames = [_trace_frame_to_scene_frame(frame) for frame in trace.frames]
    if not frames:
        raise ValueError("Trace viewer export requires at least one trace frame")

    map_payload = _compute_trace_map(trace)
    trajectory = [
        frame["robot"]["position"]
        for frame in frames
        if frame.get("robot") and frame["robot"].get("position")
    ]

    diagnostic_only = True

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
        "limitations": [
            "Trace viewer for qualitative review of simulation_trace_export.v1 fixtures.",
            "Map bounds are auto-computed from trace positions; no SVG map geometry is available.",
            "This viewer is diagnostic-only; not benchmark evidence.",
        ],
    }
    if annotations:
        scene["annotations"] = list(annotations)
    return scene


def _compute_trace_map(trace: SimulationTraceExport) -> dict[str, Any]:
    """Auto-compute map bounds and a minimal empty layout from trace positions.

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
) -> TraceViewerResult:
    """Export a simulation trace export into static browser viewer files.

    Args:
        trace: A loaded simulation trace export.
        output_dir: Directory for the viewer files.
        source: Optional source identifier.
        annotations: Optional annotations to embed.

    Returns:
        TraceViewerResult: Paths to the generated viewer directory, HTML file, and scene JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene = build_trace_scene(trace, source=source, annotations=annotations)

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
    args = parser.parse_args(argv)

    try:
        trace_path = Path(args.trace)
        trace = load_simulation_trace_export(trace_path)
        annotations_payload = _load_annotation_payload(
            Path(args.annotations) if args.annotations else None,
            trace_id=trace.trace_id,
        )
        result = export_trace_viewer(
            trace,
            args.output_dir,
            source=str(trace_path),
            annotations=annotations_payload,
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


if __name__ == "__main__":
    raise SystemExit(main())
