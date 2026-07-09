"""Export episode trajectories to a lightweight debug timeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import read_jsonl as _load_jsonl

SCHEMA_VERSION = "robot-sf-debug-timeline.v1"
ANNOTATION_KEYS = ("clearance", "min_clearance", "min_ttc", "pet", "ttc")


def _pose(value: Any) -> dict[str, float | None] | None:
    """Normalize common pose shapes into x/y/theta fields."""
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "theta": float(value["theta"]) if value.get("theta") is not None else None,
            }
        if value.get("position") is not None:
            return _pose(value["position"])
    if isinstance(value, list | tuple) and len(value) >= 2:
        return {
            "x": float(value[0]),
            "y": float(value[1]),
            "theta": float(value[2]) if len(value) >= 3 and value[2] is not None else None,
        }
    return None


def _robot_pose(frame: dict[str, Any]) -> dict[str, float | None] | None:
    """Extract a robot pose from one trajectory frame."""
    for key in ("robot", "robot_pose", "robot_position", "ego", "pose"):
        pose = _pose(frame.get(key))
        if pose is not None:
            return pose
    return _pose(frame)


def _pedestrian_items(value: Any) -> list[tuple[str, Any]]:
    """Return pedestrian id/payload pairs from mapping or list containers."""
    if isinstance(value, dict):
        return [(str(key), item) for key, item in value.items()]
    if isinstance(value, list | tuple):
        items: list[tuple[str, Any]] = []
        for index, item in enumerate(value):
            if isinstance(item, dict) and item.get("id") is not None:
                items.append((str(item["id"]), item))
            else:
                items.append((str(index), item))
        return items
    return []


def _pedestrians(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract pedestrian pose entries from one trajectory frame."""
    payload = frame.get("pedestrians", frame.get("pedestrian_positions"))
    pedestrians: list[dict[str, Any]] = []
    for entity_id, pedestrian in _pedestrian_items(payload):
        pose = _pose(pedestrian)
        if pose is not None:
            pedestrians.append({"entity_id": entity_id, **pose})
    return pedestrians


def _annotations(record: dict[str, Any], frame: dict[str, Any]) -> dict[str, float]:
    """Extract metric annotations for a frame, falling back to episode metrics."""
    source = (
        frame.get("metrics") if isinstance(frame.get("metrics"), dict) else record.get("metrics")
    )
    if not isinstance(source, dict):
        return {}
    annotations: dict[str, float] = {}
    for key in ANNOTATION_KEYS:
        value = source.get(key)
        if isinstance(value, int | float):
            annotations[key] = float(value)
    return annotations


def _action(frame: dict[str, Any]) -> Any:
    """Extract the selected or proposed action from a frame when present."""
    for key in ("action", "selected_action", "planner_action", "proposed_action"):
        if key in frame:
            return frame[key]
    return None


def _trajectory_frames(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert common trajectory_data frames into debug-timeline frames."""
    trajectory = record.get("trajectory_data")
    if not isinstance(trajectory, list):
        return []
    frames: list[dict[str, Any]] = []
    for index, item in enumerate(trajectory):
        if not isinstance(item, dict):
            continue
        frame = {
            "step": int(item.get("step", index)),
            "time_s": float(item["time_s"]) if item.get("time_s") is not None else None,
            "robot": _robot_pose(item),
            "pedestrians": _pedestrians(item),
            "action": _action(item),
            "annotations": _annotations(record, item),
        }
        frames.append(frame)
    return frames


def build_debug_timeline(records: list[dict[str, Any]], *, source_path: Path) -> dict[str, Any]:
    """Build the debug timeline payload for episode records."""
    episodes: list[dict[str, Any]] = []
    frame_count = 0
    for record in records:
        frames = _trajectory_frames(record)
        frame_count += len(frames)
        terminal_event = record.get("termination_reason") or record.get("terminal_event")
        episodes.append(
            {
                "episode_id": str(record.get("episode_id", "")),
                "scenario_id": str(record.get("scenario_id", "")),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "terminal_event": terminal_event,
                "frames": frames,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_path": source_path.as_posix(),
        "summary": {"episodes": len(episodes), "frames": frame_count},
        "episodes": episodes,
    }


def write_json_debug_export(*, source: Path, output: Path) -> Path:
    """Write a JSON debug timeline for an episode JSONL source."""
    payload = build_debug_timeline(_load_jsonl(source), source_path=source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_rerun_debug_export(*, source: Path, output: Path) -> Path:
    """Write a Rerun recording when the optional SDK is installed."""
    try:
        import rerun as rr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Rerun export requires the optional 'rerun-sdk' package. "
            "Install it in a disposable environment or use --format json."
        ) from exc

    payload = build_debug_timeline(_load_jsonl(source), source_path=source)
    output.parent.mkdir(parents=True, exist_ok=True)
    rr.init("robot_sf_debug_export", spawn=False)
    for index, episode in enumerate(payload["episodes"]):
        episode_id = episode["episode_id"] or f"episode_{index}"
        for frame in episode["frames"]:
            step = frame["step"]
            if frame["time_s"] is not None:
                rr.set_time_seconds("time", float(frame["time_s"]))
            else:
                rr.set_time_sequence("step", int(step))
            robot = frame["robot"]
            if robot is not None:
                rr.log(f"{episode_id}/robot", rr.Points2D([[robot["x"], robot["y"]]], radii=0.2))
            pedestrian_points = [
                [item["x"], item["y"]]
                for item in frame["pedestrians"]
                if item.get("x") is not None and item.get("y") is not None
            ]
            if pedestrian_points:
                rr.log(
                    f"{episode_id}/pedestrians",
                    rr.Points2D(pedestrian_points, radii=0.15),
                )
    rr.save(str(output))
    return output


def _build_parser() -> argparse.ArgumentParser:
    """Build the debug-export CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Episode JSONL source path.")
    parser.add_argument("--output", required=True, type=Path, help="Output debug artifact path.")
    parser.add_argument(
        "--format",
        choices=("json", "rerun"),
        default="json",
        help="Debug export format. Rerun requires optional rerun-sdk.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the trajectory debug export CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.format == "json":
            write_json_debug_export(source=args.source, output=args.output)
        else:
            write_rerun_debug_export(source=args.source, output=args.output)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {args.format} debug timeline to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
