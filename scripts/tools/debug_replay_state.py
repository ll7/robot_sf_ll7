"""CLI for inspecting replay state snapshots from JSONL episode recordings.

The utility loads a replay using the existing :mod:`robot_sf.render.jsonl_playback`
loader APIs and prints a compact, structured diagnostic view for one selected frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.render.jsonl_playback import PlaybackEpisode, create_playback_loader


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_path", type=Path, help="Path to replay JSONL file or directory")
    parser.add_argument(
        "--episode-index",
        type=int,
        help="Episode index in the loaded replay batch (0-based)",
    )
    parser.add_argument(
        "--episode-id",
        type=int,
        help="Episode id saved in the replay payload (0-based)",
    )
    parser.add_argument(
        "--step",
        "--frame",
        dest="frame_index",
        type=int,
        default=None,
        help="State frame index within the selected episode (0-based). Defaults to the last frame",
    )
    parser.add_argument(
        "--agent-id",
        help="Optional agent selector: robot (default), ego, or pedestrian index/label (0, 1, ped_0)",
    )
    parser.add_argument(
        "--output-mode",
        choices=("text", "json"),
        default="text",
        help="Output mode for diagnostics",
    )
    return parser


def _coerce_replay_records(
    replay_file: Path,
    *,
    episode_id: int,
) -> list[dict[str, Any]]:
    """Load JSONL replay records with state payloads for one episode."""

    if not replay_file.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_file}")
    if replay_file.suffix != ".jsonl":
        return []

    records: list[dict[str, Any]] = []
    with replay_file.open(encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue

            event = record.get("event", "step")
            if event in {"episode_start", "episode_end"}:
                continue

            state = record.get("state")
            if not isinstance(state, dict):
                continue

            record_episode_id = record.get("episode_id")
            if record_episode_id != episode_id:
                continue

            records.append(record)
    return records


def _safe_json_shape(value: Any) -> list[int] | None:
    """Return JSON-friendly dimensions for nested structures where possible."""

    try:
        return [int(dim) for dim in np.asarray(value, dtype=object).shape]
    except (TypeError, ValueError):
        return None


def _coerce_float_array(values: Any) -> np.ndarray | None:
    """Convert arbitrary values to a float numpy array when possible."""

    if values is None:
        return None
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None


def _summarize_vector_array(values: Any) -> dict[str, Any] | None:
    """Summarize vector-like arrays for readable diagnostics."""

    if values is None:
        return None

    array = _coerce_float_array(values)
    if array is None:
        shape = _safe_json_shape(values)
        return {"present": True, "shape": shape, "dtype": str(type(values).__name__)}

    if array.size == 0:
        return {"present": True, "count": 0, "shape": [int(dim) for dim in array.shape]}

    flat = array
    if array.ndim >= 2:
        flat = array.reshape(array.shape[0], -1)

    norms = np.linalg.norm(flat, axis=1) if flat.ndim > 1 else np.abs(flat)
    return {
        "present": True,
        "count": int(array.shape[0]) if array.ndim >= 1 else int(array.size),
        "shape": [int(dim) for dim in array.shape],
        "norm_min": float(np.min(norms)) if norms.size else 0.0,
        "norm_max": float(np.max(norms)) if norms.size else 0.0,
    }


def _pose_to_dict(pose: Any) -> dict[str, float] | None:
    """Serialize a renderer pose tuple into a small JSON object."""

    if pose is None:
        return None
    try:
        return {
            "x": float(pose[0][0]),
            "y": float(pose[0][1]),
            "theta": float(pose[1]),
        }
    except (TypeError, ValueError, IndexError):
        return None


def _xy_to_dict(position: Any) -> dict[str, float] | None:
    """Serialize a two-dimensional position into a small JSON object."""

    if position is None:
        return None
    try:
        return {
            "x": float(position[0]),
            "y": float(position[1]),
        }
    except (TypeError, ValueError, IndexError):
        return None


def _build_pedestrian_summary(ped_positions: Any) -> dict[str, Any]:
    """Build compact pedestrian position/count summary."""

    array = _coerce_float_array(ped_positions)
    if array is None or array.size == 0:
        return {"count": 0, "positions": []}

    if array.ndim == 1 and array.size >= 2:
        positions = array.reshape(1, -1)
    else:
        positions = array

    sample = []
    for idx in range(min(len(positions), 3)):
        pos = _xy_to_dict(positions[idx])
        if pos is not None:
            sample.append({"index": idx, **pos})

    return {
        "count": int(positions.shape[0]) if positions.ndim > 1 else 1,
        "shape": [int(dim) for dim in array.shape],
        "positions": sample,
    }


def _resolve_agent_position(
    episode_state: Any,
    agent_id: str | None,
) -> dict[str, Any]:
    """Resolve selected agent pose for output diagnostics."""

    if agent_id is None or agent_id.lower() == "robot":
        return {
            "kind": "robot",
            "index": None,
            "pose": _pose_to_dict(episode_state.robot_pose),
        }

    if agent_id.lower() in {"ego", "ego_ped", "ego-ped"}:
        return {
            "kind": "ego_pedestrian",
            "index": None,
            "pose": _pose_to_dict(episode_state.ego_ped_pose),
        }

    if agent_id.startswith("ped_"):
        agent_id = agent_id.split("_", 1)[1]
    try:
        ped_index = int(agent_id)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported --agent-id {agent_id!r}; use robot, ego, or pedestrian index"
        ) from exc

    ped_positions = _coerce_float_array(episode_state.pedestrian_positions)
    if ped_positions is None:
        return {"kind": "pedestrian", "index": ped_index, "pose": None}

    if ped_positions.ndim == 1 and ped_positions.size >= 2:
        ped_positions = ped_positions.reshape(1, -1)

    if ped_index < 0 or ped_index >= len(ped_positions):
        raise ValueError(f"Pedestrian index out of range: {ped_index}")

    return {
        "kind": "pedestrian",
        "index": ped_index,
        "pose": _xy_to_dict(ped_positions[ped_index]),
    }


def _ensure_json_payload(value: Any, *, _depth: int = 0) -> Any:
    """Fallback serialization helper for non-JSON-safe values."""

    if _depth > 20:
        return str(value)

    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _ensure_json_payload(v, _depth=_depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_ensure_json_payload(v, _depth=_depth + 1) for v in value]
        return str(value)


def _load_manifest_episodes(
    *,
    manifest_path: Path,
    loader: Any,
) -> list[tuple[Path, PlaybackEpisode]]:
    """Load episodes listed by a manifest while preserving source paths."""

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        raise ValueError(f"Invalid manifest JSON (expected a dictionary): {manifest_path}")
    episode_paths = manifest_data.get("episodes")
    if not isinstance(episode_paths, list) or not episode_paths:
        raise ValueError(f"Manifest has no episodes: {manifest_path}")

    result = []
    for entry in episode_paths:
        if not isinstance(entry, str):
            raise ValueError(f"Invalid manifest entry (expected string path): {entry!r}")
        full_path = manifest_path.parent / entry
        episode, _ = loader.load_single_episode(full_path)
        result.append((full_path, episode))
    return result


def _load_episode_entries(replay_path: Path) -> list[tuple[Path, PlaybackEpisode]]:
    """Load replay episodes from a single file, directory, or manifest."""

    loader = create_playback_loader()

    if not replay_path.exists():
        raise FileNotFoundError(f"Replay path not found: {replay_path}")

    if replay_path.is_dir():
        file_patterns = ["*.jsonl", "*.pkl"]
        files = sorted(path for pattern in file_patterns for path in replay_path.glob(pattern))
        if not files:
            raise ValueError(f"No replay files found in: {replay_path}")

        episodes: list[tuple[Path, PlaybackEpisode]] = []
        for file_path in files:
            episode, _ = loader.load_single_episode(file_path)
            episodes.append((file_path, episode))
        return episodes

    if replay_path.suffix == ".json":
        return _load_manifest_episodes(manifest_path=replay_path, loader=loader)

    episode, _ = loader.load_single_episode(replay_path)
    return [(replay_path, episode)]


def _select_episode_entry(
    episode_entries: list[tuple[Path, PlaybackEpisode]],
    *,
    episode_index: int | None,
    episode_id: int | None,
) -> tuple[int, Path, PlaybackEpisode]:
    """Select one episode by optional index/id."""

    if episode_index is not None:
        if episode_index < 0:
            raise ValueError(f"Episode index must be >= 0: {episode_index}")
        if episode_index >= len(episode_entries):
            raise ValueError(f"Episode index out of range: {episode_index}")
        selected_index, (file_path, episode) = episode_index, episode_entries[episode_index]
        if episode_id is not None and episode.episode_id != episode_id:
            raise ValueError(
                f"Selected episode index {episode_index} has id {episode.episode_id}, "
                f"not requested id {episode_id}"
            )
        return selected_index, file_path, episode

    if episode_id is not None:
        for idx, (file_path, episode) in enumerate(episode_entries):
            if episode.episode_id == episode_id:
                return idx, file_path, episode
        raise ValueError(f"Episode id not found in replay: {episode_id}")

    if len(episode_entries) != 1:
        raise ValueError(
            "Multiple episodes found. Provide --episode-index or --episode-id to select one."
        )

    file_path, episode = episode_entries[0]
    return 0, file_path, episode


def _select_frame_index(episode: PlaybackEpisode, frame_index: int | None) -> int:
    """Resolve frame index defaulting to the last state."""

    if frame_index is None:
        if not episode.states:
            raise ValueError("Selected episode has no states")
        return len(episode.states) - 1

    if frame_index < 0:
        raise ValueError(f"Frame index must be >= 0: {frame_index}")
    if frame_index >= len(episode.states):
        raise ValueError(f"Frame index out of range: {frame_index}")
    return frame_index


def _build_sensor_summary(record_state: dict[str, Any] | None) -> dict[str, Any]:
    """Build lightweight sensor/ray summary from the raw JSONL state record."""

    if not isinstance(record_state, dict):
        return {}

    summary: dict[str, Any] = {}
    for key in ("observation_image", "ego_ped_ray_vecs", "planned_path"):
        if key in record_state:
            summary[key] = _summarize_vector_array(record_state[key])

    if "robot_action" in record_state:
        robot_action = record_state.get("robot_action")
        if isinstance(robot_action, dict):
            summary["robot_action"] = {
                "fields": sorted(robot_action.keys()),
            }
        else:
            summary["robot_action"] = {"present": bool(robot_action is not None)}

    return summary


def _build_payload(
    replay_path: Path,
    episode_index: int,
    episode_file: Path,
    episode: PlaybackEpisode,
    frame_index: int,
    state_record: dict[str, Any],
    state: Any,
    agent_id: str | None,
) -> dict[str, Any]:
    """Build final diagnostics payload for CLI output."""

    state_dict = state_record.get("state", {}) if isinstance(state_record, dict) else {}
    return {
        "replay_path": str(replay_path),
        "episode_file": str(episode_file),
        "episode": {
            "index": episode_index,
            "id": episode.episode_id,
            "metadata": _ensure_json_payload(episode.metadata),
        },
        "frame": {
            "index": frame_index,
            "timestep": float(state.timestep),
            "step_idx": int(state_record.get("step_idx", frame_index))
            if isinstance(state_record, dict)
            else frame_index,
        },
        "robot": {
            "pose": _pose_to_dict(state.robot_pose),
            "position": _xy_to_dict(state.robot_pose[0]) if state.robot_pose else None,
        },
        "pedestrians": _build_pedestrian_summary(state.pedestrian_positions),
        "ray_summary": _summarize_vector_array(state.ray_vecs),
        "sensor_summary": _build_sensor_summary(state_dict),
        "selected_agent": _resolve_agent_position(state, agent_id),
    }


def _format_text(payload: dict[str, Any]) -> str:
    """Format payload as human-readable diagnostics."""

    lines = [
        f"Replay: {payload['replay_path']}",
        f"Episode index/id: {payload['episode']['index']}/{payload['episode']['id']}",
        f"Frame: {payload['frame']['index']} (timestep={payload['frame']['timestep']})",
        f"Robot: {payload['robot']['pose']}",
        f"Pedestrians: count={payload['pedestrians']['count']}",
        f"Selected agent: {payload['selected_agent']}",
    ]

    if payload["ray_summary"] is not None:
        lines.append(f"Ray summary: {payload['ray_summary']}")

    if payload["sensor_summary"]:
        lines.append(f"Sensor summary: {payload['sensor_summary']}")

    if payload["episode"].get("metadata") is not None:
        metadata = payload["episode"]["metadata"]
        for key in ("algorithm", "scenario", "seed", "config_hash"):
            if key in metadata:
                lines.append(f"{key}: {metadata[key]}")

    return "\n".join(lines)


def _resolve_replay_record(
    replay_file: Path,
    episode: PlaybackEpisode,
    frame_index: int,
) -> dict[str, Any] | None:
    """Return the matching JSONL record for the selected episode/frame."""

    records = _coerce_replay_records(replay_file, episode_id=episode.episode_id)
    if len(records) <= frame_index:
        return None
    return records[frame_index]


def run_debug_replay_state(args: argparse.Namespace) -> dict[str, Any]:
    """Run state extraction and return structured diagnostics payload."""

    entries = _load_episode_entries(args.replay_path)
    selected_episode_index, selected_file, selected_episode = _select_episode_entry(
        entries,
        episode_index=args.episode_index,
        episode_id=args.episode_id,
    )
    frame_index = _select_frame_index(
        selected_episode,
        frame_index=args.frame_index,
    )

    state = selected_episode.states[frame_index]
    state_record = _resolve_replay_record(selected_file, selected_episode, frame_index)

    return _build_payload(
        replay_path=args.replay_path,
        episode_index=selected_episode_index,
        episode_file=selected_file,
        episode=selected_episode,
        frame_index=frame_index,
        state_record=state_record or {},
        state=state,
        agent_id=args.agent_id,
    )


def main() -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        payload = run_debug_replay_state(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output_mode == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_format_text(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
