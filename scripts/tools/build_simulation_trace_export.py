"""Build renderer-neutral trace-export artifacts from simulation JSONL source."""
# ruff: noqa: C901, PLR0912, PLR0915

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExportValidationError,
    simulation_trace_export_from_dict,
)
from robot_sf.benchmark.identity.hash_utils import read_jsonl as _load_jsonl
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256

SCHEMA_VERSION = SIMULATION_TRACE_EXPORT_SCHEMA_VERSION


def _source_metadata_path(source: Path) -> Path | None:
    """Resolve companion metadata file for a JSON source.

    The trace source naming in this repo uses both legacy and new conventions,
    so check both.
    """

    candidates = [
        source.with_name(f"{source.stem}.meta.json"),
        source.with_name(f"{source.name}.meta.json"),
        source.with_suffix(source.suffix + ".meta.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _load_source_metadata(source_path: Path) -> dict[str, Any]:
    """Load optional companion metadata for a recording source."""

    metadata_path = _source_metadata_path(source_path)
    if metadata_path is None:
        return {}

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot read metadata file {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata file {metadata_path} must contain a JSON object")
    return metadata


def _pose(record_state: dict[str, Any]) -> tuple[float, float, float]:
    """Extract robot pose from a single ``robot_pose``-like tuple."""

    robot_pose = record_state.get("robot_pose")
    if not isinstance(robot_pose, (list, tuple)) or len(robot_pose) < 2:
        raise ValueError("missing robot_pose for step frame")

    position = robot_pose[0]
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        raise ValueError("robot_pose position must be [x, y]")

    orientation = robot_pose[1]
    if not isinstance(orientation, (int, float)):
        raise ValueError("robot_pose orientation must be numeric")

    return float(position[0]), float(position[1]), float(orientation)


def _pedestrians(record_state: dict[str, Any]) -> list[dict[str, float]]:
    """Convert source pedestrian list to renderer-neutral tuple list."""

    pedestrians = record_state.get("pedestrian_positions")
    if not isinstance(pedestrians, list):
        return []

    packed: list[dict[str, float]] = []
    for index, item in enumerate(pedestrians):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        packed.append({"id": f"ped-{index}", "position": [float(item[0]), float(item[1])]})
    return packed


def _normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi) for diff continuity."""

    return math.atan2(math.sin(angle), math.cos(angle))


def _derive_velocity(
    current_pose: tuple[float, float, float],
    previous_pose: tuple[float, float, float] | None,
    dt: float,
) -> tuple[float, float]:
    """Compute linear and angular velocity from pose deltas."""

    if previous_pose is None or dt <= 0:
        return 0.0, 0.0

    dx = current_pose[0] - previous_pose[0]
    dy = current_pose[1] - previous_pose[1]
    linear = (dx * dx + dy * dy) ** 0.5 / dt
    angular = _normalize_angle(current_pose[2] - previous_pose[2]) / dt
    return float(linear), float(angular)


def _select_action(
    state: dict[str, Any],
    previous_pose: tuple[float, float, float] | None,
    dt: float,
) -> dict[str, float]:
    """Prefer explicit action fields, else derive from pose deltas."""

    selected_action = state.get("robot_action")
    if isinstance(selected_action, dict):
        linear = selected_action.get("linear_velocity")
        angular = selected_action.get("angular_velocity")
        if isinstance(linear, (int, float)) and isinstance(angular, (int, float)):
            return {"linear_velocity": float(linear), "angular_velocity": float(angular)}

    if isinstance(selected_action, (list, tuple)) and len(selected_action) >= 2:
        if all(isinstance(item, (int, float)) for item in selected_action[:2]):
            return {
                "linear_velocity": float(selected_action[0]),
                "angular_velocity": float(selected_action[1]),
            }

    action_pose = state.get("robot_pose")
    if isinstance(action_pose, (list, tuple)) and len(action_pose) >= 2:
        try:
            current_pose = _pose(state)
        except ValueError:
            return {"linear_velocity": 0.0, "angular_velocity": 0.0}

        linear, angular = _derive_velocity(current_pose, previous_pose, dt)
        return {"linear_velocity": linear, "angular_velocity": angular}

    return {"linear_velocity": 0.0, "angular_velocity": 0.0}


def _aggregate_algorithm_metadata(record: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate-row algorithm metadata when present."""

    metadata = record.get("algorithm_metadata")
    return metadata if isinstance(metadata, dict) else {}


def _aggregate_scenario_params(record: dict[str, Any]) -> dict[str, Any]:
    """Return aggregate-row scenario params when present."""

    params = record.get("scenario_params")
    return params if isinstance(params, dict) else {}


def _copy_trace_frame(frame: dict[str, Any], *, source: Path, index: int) -> dict[str, Any]:
    """Copy a schema-shaped aggregate trace frame for export validation."""

    required = {"step", "time_s", "robot", "pedestrians", "planner"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"{source}: aggregate trace frame {index} missing {missing}")

    pedestrians = frame["pedestrians"]
    if isinstance(pedestrians, list):
        pedestrians = [
            {**pedestrian, "id": str(pedestrian["id"])}
            if isinstance(pedestrian, dict) and "id" in pedestrian
            else pedestrian
            for pedestrian in pedestrians
        ]

    copied = {
        "step": frame["step"],
        "time_s": frame["time_s"],
        "robot": frame["robot"],
        "pedestrians": pedestrians,
        "planner": frame["planner"],
    }
    if not isinstance(copied["robot"], dict):
        raise ValueError(f"{source}: aggregate trace frame {index} robot must be an object")
    if not isinstance(copied["pedestrians"], list):
        raise ValueError(f"{source}: aggregate trace frame {index} pedestrians must be a list")
    if not isinstance(copied["planner"], dict):
        raise ValueError(f"{source}: aggregate trace frame {index} planner must be an object")
    return copied


def _frames_from_aggregate_records(
    source: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract opt-in simulation step traces from aggregate benchmark JSONL records."""

    frames: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        trace = _aggregate_algorithm_metadata(record).get("simulation_step_trace")
        if not isinstance(trace, dict):
            continue
        steps = trace.get("steps")
        if not isinstance(steps, list):
            raise ValueError(
                f"{source}: aggregate record {record_index} trace steps must be a list"
            )
        for step_index, frame in enumerate(steps):
            if not isinstance(frame, dict):
                raise ValueError(
                    f"{source}: aggregate record {record_index} trace step {step_index} "
                    "must be an object"
                )
            frames.append(_copy_trace_frame(frame, source=source, index=step_index))
    return frames


def build_simulation_trace_export(
    source: Path,
    *,
    planner_id: str | None = None,
    scenario_id: str | None = None,
    source_signature: str | None = None,
) -> dict[str, Any]:
    """Build a validated analysis-workbench timeline payload."""

    records = _load_jsonl(source)
    metadata = _load_source_metadata(source)
    first_record = records[0]
    first_algo_metadata = _aggregate_algorithm_metadata(first_record)
    first_scenario_params = _aggregate_scenario_params(first_record)

    planner_id = planner_id or str(
        metadata.get("algorithm")
        or metadata.get("planner")
        or metadata.get("planner_id")
        or first_algo_metadata.get("algorithm")
        or first_scenario_params.get("algo")
        or "unknown_planner"
    )
    scenario = scenario_id or str(
        metadata.get("scenario")
        or metadata.get("scenario_id")
        or first_record.get("scenario_id")
        or first_scenario_params.get("scenario_id")
        or first_scenario_params.get("id")
        or "unknown_scenario"
    )
    seed = metadata.get("seed", first_record.get("seed", 0))
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        seed_int = 0

    episode_id = metadata.get("episode_id", first_record.get("episode_id", 0))
    if episode_id is None:
        episode_id = 0

    digest = source_signature or _sha256(source)
    trace_id = f"{scenario}-ep{episode_id}-seed{seed_int}-source-{digest[:8]}"
    generated_by = (
        f"scripts.tools.build_simulation_trace_export from {source.name}"
        f" source_sha256:{digest[:12]}"
    )

    frames: list[dict[str, Any]] = []
    previous_pose: tuple[float, float, float] | None = None
    previous_timestamp: float | None = None
    included = 0

    for index, record in enumerate(records):
        event = str(record.get("event", "step"))
        state = record.get("state")

        if event != "step":
            continue

        if not isinstance(state, dict) or not state:
            continue

        step_idx = record.get("step_idx")
        if not isinstance(step_idx, int):
            if isinstance(step_idx, float) and step_idx.is_integer():
                step_idx = int(step_idx)
            else:
                raise ValueError(f"{source}: non-integer step_idx for record {index}")

        timestamp = record.get("timestamp")
        if timestamp is None:
            timestamp = state.get("timestep", step_idx)
        try:
            time_s = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: invalid timestamp for step {step_idx}") from exc

        if previous_timestamp is None:
            previous_timestamp = time_s

        current_pose = _pose(state)
        dt = time_s - previous_timestamp
        linear_velocity, angular_velocity = _derive_velocity(
            current_pose,
            previous_pose,
            dt,
        )

        frame = {
            "step": step_idx,
            "time_s": time_s,
            "robot": {
                "position": [current_pose[0], current_pose[1]],
                "heading": current_pose[2],
                "velocity": [0.0, 0.0],
            },
            "pedestrians": [
                {"id": ped["id"], "position": ped["position"], "velocity": [0.0, 0.0]}
                for ped in _pedestrians(state)
            ],
            "planner": {
                "selected_action": {
                    "linear_velocity": linear_velocity,
                    "angular_velocity": angular_velocity,
                },
                "event": event,
                "event_id": f"{source.stem}-frame-{included:04d}",
            },
        }

        # Prefer explicit robot velocity fields if present.
        robot_velocity = state.get("robot_velocity")
        if isinstance(robot_velocity, (list, tuple)) and len(robot_velocity) >= 2:
            vx, vy = robot_velocity[0], robot_velocity[1]
            if isinstance(vx, (int, float)) and isinstance(vy, (int, float)):
                frame["robot"]["velocity"] = [float(vx), float(vy)]
        else:
            frame["robot"]["velocity"] = [
                (current_pose[0] - previous_pose[0]) / dt if previous_pose and dt > 0 else 0.0,
                (current_pose[1] - previous_pose[1]) / dt if previous_pose and dt > 0 else 0.0,
            ]

        frame["planner"]["selected_action"] = _select_action(state, previous_pose, dt)
        frames.append(frame)

        previous_pose = current_pose
        previous_timestamp = time_s
        included += 1

    if not frames:
        frames = _frames_from_aggregate_records(source, records)
    if not frames:
        raise ValueError(f"{source} has no step frames for conversion")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "trace_id": trace_id,
        "source": {
            "scenario_id": scenario,
            "seed": seed_int,
            "planner_id": planner_id,
            "episode_id": str(episode_id),
            "generated_by": generated_by,
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {
            "position": "m",
            "heading": "rad",
            "time": "s",
            "velocity": "m/s",
        },
        "frames": frames,
    }

    simulation_trace_export_from_dict(payload, source=source)
    return payload


def write_simulation_trace_export(
    *,
    source: Path,
    output: Path,
    planner_id: str | None = None,
    scenario_id: str | None = None,
) -> Path:
    """Write a validated analysis-workbench timeline payload."""

    payload = build_simulation_trace_export(
        source,
        planner_id=planner_id,
        scenario_id=scenario_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for trace-export generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Episode JSONL recording source."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output simulation export path.")
    parser.add_argument("--planner-id", default=None, help="Planner contract identifier override.")
    parser.add_argument("--scenario-id", default=None, help="Scenario identifier override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the conversion command."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        output = write_simulation_trace_export(
            source=args.source,
            output=args.output,
            planner_id=args.planner_id,
            scenario_id=args.scenario_id,
        )
    except (OSError, ValueError, SimulationTraceExportValidationError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    print(f"wrote simulation trace export to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
