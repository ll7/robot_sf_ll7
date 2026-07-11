"""Materialize trace-distilled entries as generated-only replay scenarios.

The adapter deliberately preserves the generated-hypothesis boundary.  It only
uses positions present in the catalog entry, and reports a concrete blocker
when the production scenario contract cannot represent the trace shape.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from typing import Any

import yaml

from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition

_RUNTIME_SCHEMA_VERSION = "robot_sf.generated_replay_runtime.v1"


@dataclass(frozen=True)
class GeneratedScenarioMaterialization:
    """Result of attempting to adapt one catalog entry for production loading."""

    status: str
    warnings: tuple[str, ...]
    scenario_document: dict[str, Any] | None


def materialize_generated_scenario(
    entry: dict[str, Any], *, max_episode_steps: int = 100
) -> GeneratedScenarioMaterialization:
    """Return a standalone scenario document or an entry-specific blocker.

    Robot start and goal are the first and final sampled robot positions.
    Pedestrian routes are the ordered sampled positions for each stable trace
    index.  This is a directional replay setup, not a criticality-reproduction
    claim.
    """

    validate_catalog_entry(entry)
    if isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int):
        raise ValueError("max_episode_steps must be an integer")
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be >= 1")

    frames = entry["segment"]["trace_frames"]
    if len(frames) < 2:
        return _not_representable("trace has fewer than two frames for a pinned robot goal")

    pedestrian_count = len(frames[0]["pedestrians"])
    if pedestrian_count < 1:
        return _not_representable("trace has no pedestrians to materialize")
    for frame_index, frame in enumerate(frames[1:], start=1):
        if len(frame["pedestrians"]) != pedestrian_count:
            return _not_representable(
                "pedestrian count changes at trace frame "
                f"{frame_index} ({pedestrian_count} -> {len(frame['pedestrians'])})"
            )

    robot_trajectory = [list(frame["robot"]["position"]) for frame in frames]
    if robot_trajectory[0] == robot_trajectory[-1]:
        return _not_representable("robot start and final sampled position are identical")

    pedestrians = []
    for pedestrian_index in range(pedestrian_count):
        trajectory = [list(frame["pedestrians"][pedestrian_index]["position"]) for frame in frames]
        pedestrians.append(
            {
                "id": f"generated-pedestrian-{pedestrian_index:03d}",
                "start": trajectory[0],
                "trajectory": trajectory,
            }
        )

    source_map = entry["source_episode"]["source_map"]
    scenario = {
        "schema_version": "robot_sf.scenario_matrix.v1",
        "scenarios": [
            {
                "name": entry["scenario_id"],
                "map_file": source_map,
                "seeds": [entry["replay"]["source_seed"]],
                "simulation_config": {
                    "max_episode_steps": max_episode_steps,
                    "ped_density": 0.0,
                },
                "robot_config": {"type": "differential_drive"},
                "generated_replay": {
                    "schema_version": _RUNTIME_SCHEMA_VERSION,
                    "robot": {
                        "start": robot_trajectory[0],
                        "goal": robot_trajectory[-1],
                        "trajectory": robot_trajectory,
                    },
                    "pedestrians": pedestrians,
                },
                "metadata": {
                    "source": "auto_generated",
                    "required_manual_review": True,
                    "benchmark_evidence": False,
                    "generated_replay": {
                        "source_catalog_schema": entry["schema_version"],
                        "replay_status": "loads_only",
                        "claim_boundary": "generated scenario hypotheses only",
                    },
                },
            }
        ],
    }
    return GeneratedScenarioMaterialization("loads_only", (), scenario)


def dump_generated_scenario_yaml(result: GeneratedScenarioMaterialization) -> str:
    """Serialize a materialized scenario deterministically.

    Raises:
        ValueError: If the entry was not representable as a standalone scenario.

    Returns:
        Deterministic YAML containing the generated-only scenario matrix.
    """

    if result.scenario_document is None:
        detail = "; ".join(result.warnings) or "unknown materialization failure"
        raise ValueError(f"generated scenario is not representable yet: {detail}")
    return yaml.safe_dump(result.scenario_document, sort_keys=True)


def generated_replay_status_entry(
    entry: dict[str, Any], result: GeneratedScenarioMaterialization
) -> dict[str, Any]:
    """Return a catalog-entry copy with the adapter's explicit replay status."""

    updated = deepcopy(entry)
    updated["replay"]["status"] = result.status
    updated["replay"]["warnings"] = list(result.warnings)
    validate_catalog_entry(updated)
    return updated


def apply_generated_replay_runtime(  # noqa: C901
    map_def: MapDefinition, runtime: object
) -> MapDefinition:
    """Return a source-map clone pinned to a generated replay runtime block.

    The source map's obstacles and bounds are retained.  Only actor placement
    and routes are replaced, so materialization cannot accidentally promote the
    generated hypothesis into a hand-authored map or benchmark scenario.
    """

    if not isinstance(runtime, dict):
        raise ValueError("generated_replay must be a mapping")
    expected = {"schema_version", "robot", "pedestrians"}
    unknown = sorted(set(runtime) - expected)
    if unknown:
        raise ValueError(f"generated_replay has unknown keys: {', '.join(unknown)}")
    if runtime.get("schema_version") != _RUNTIME_SCHEMA_VERSION:
        raise ValueError(f"generated_replay.schema_version must be '{_RUNTIME_SCHEMA_VERSION}'")
    robot = runtime.get("robot")
    if not isinstance(robot, dict) or set(robot) != {"start", "goal", "trajectory"}:
        raise ValueError("generated_replay.robot must contain only start, goal, and trajectory")
    start = _point(robot["start"], "generated_replay.robot.start")
    goal = _point(robot["goal"], "generated_replay.robot.goal")
    trajectory = _trajectory(robot["trajectory"], "generated_replay.robot.trajectory")
    if trajectory[0] != start or trajectory[-1] != goal:
        raise ValueError("generated_replay.robot trajectory endpoints must match start and goal")
    if start == goal:
        raise ValueError("generated_replay.robot start and goal must differ")

    pedestrians = runtime.get("pedestrians")
    if not isinstance(pedestrians, list) or not pedestrians:
        raise ValueError("generated_replay.pedestrians must be a non-empty list")
    single_pedestrians: list[SinglePedestrianDefinition] = []
    seen_ids: set[str] = set()
    for index, pedestrian in enumerate(pedestrians):
        if not isinstance(pedestrian, dict) or set(pedestrian) != {"id", "start", "trajectory"}:
            raise ValueError(
                f"generated_replay.pedestrians[{index}] must contain only id, start, and trajectory"
            )
        pedestrian_id = pedestrian["id"]
        if not isinstance(pedestrian_id, str) or not pedestrian_id.strip():
            raise ValueError(f"generated_replay.pedestrians[{index}].id must be non-empty")
        if pedestrian_id in seen_ids:
            raise ValueError(f"generated_replay contains duplicate pedestrian id '{pedestrian_id}'")
        seen_ids.add(pedestrian_id)
        pedestrian_start = _point(
            pedestrian["start"], f"generated_replay.pedestrians[{index}].start"
        )
        pedestrian_trajectory = _trajectory(
            pedestrian["trajectory"], f"generated_replay.pedestrians[{index}].trajectory"
        )
        if pedestrian_trajectory[0] != pedestrian_start:
            raise ValueError(
                "generated_replay pedestrian trajectory must begin at its declared start"
            )
        single_pedestrians.append(
            SinglePedestrianDefinition(
                id=pedestrian_id,
                start=pedestrian_start,
                trajectory=pedestrian_trajectory,
                metadata={"source": "generated_replay"},
            )
        )

    spawn_zone = _point_zone(start)
    goal_zone = _point_zone(goal)
    robot_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=trajectory,
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
        source_label="generated_replay",
    )
    return MapDefinition(
        width=map_def.width,
        height=map_def.height,
        obstacles=deepcopy(map_def.obstacles),
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[],
        robot_goal_zones=[goal_zone],
        bounds=deepcopy(map_def.bounds),
        robot_routes=[robot_route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=single_pedestrians,
        infrastructure_zones=deepcopy(map_def.infrastructure_zones),
    )


def _point(value: object, field_name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item list")
    coordinates = tuple(float(coordinate) for coordinate in value)
    if not all(isfinite(coordinate) for coordinate in coordinates):
        raise ValueError(f"{field_name} must contain finite coordinates")
    return coordinates  # type: ignore[return-value]


def _trajectory(value: object, field_name: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{field_name} must contain at least two positions")
    return [_point(point, f"{field_name}[{index}]") for index, point in enumerate(value)]


def _point_zone(point: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    """Return a degenerate triangular zone so runtime placement is exactly pinned."""

    return (point, point, point)


def _not_representable(detail: str) -> GeneratedScenarioMaterialization:
    return GeneratedScenarioMaterialization(
        "not_representable_yet",
        (f"replay_gap: {detail}",),
        None,
    )


__all__ = [
    "GeneratedScenarioMaterialization",
    "apply_generated_replay_runtime",
    "dump_generated_scenario_yaml",
    "generated_replay_status_entry",
    "materialize_generated_scenario",
]
