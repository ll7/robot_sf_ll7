"""Runtime adapters for scripted multi-pedestrian adversarial configs."""

from __future__ import annotations

import re
from copy import deepcopy
from itertools import combinations
from math import dist

from shapely.geometry import Point

from robot_sf.adversarial.config import MultiPedAdversarialConfig
from robot_sf.adversarial.materialize import materialize_multi_ped_single_pedestrian_overrides
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.sim.sim_config import SimulationSettings

DEFAULT_MIN_START_SEPARATION_M = 0.8
DEFAULT_MIN_OBSTACLE_CLEARANCE_M = 0.05
DEFAULT_MAX_SCRIPTED_SPEED_MPS = 2.5


def multi_ped_config_to_single_pedestrian_definitions(
    config: MultiPedAdversarialConfig,
) -> list[SinglePedestrianDefinition]:
    """Convert a multi-ped adversarial config into runtime single-pedestrian definitions.

    Raises:
        ValueError: If the config-level candidate contract is invalid.
    """
    errors = config.validate()
    if errors:
        raise ValueError("; ".join(errors))

    definitions: list[SinglePedestrianDefinition] = []
    for entry in materialize_multi_ped_single_pedestrian_overrides(config):
        definitions.append(
            SinglePedestrianDefinition(
                id=str(entry["id"]),
                start=_tuple_point(entry["start"]),
                goal=_tuple_point(entry["goal"]),
                speed_m_s=float(entry["speed_m_s"]),
                start_delay_s=float(entry["start_delay_s"]),
                note=str(entry["note"]) if entry.get("note") is not None else None,
                metadata=dict(entry.get("metadata") or {}),
            )
        )
    return definitions


def validate_multi_ped_runtime_plausibility(
    config: MultiPedAdversarialConfig,
    base_map: MapDefinition,
    *,
    min_start_separation_m: float = DEFAULT_MIN_START_SEPARATION_M,
    min_obstacle_clearance_m: float = DEFAULT_MIN_OBSTACLE_CLEARANCE_M,
    max_speed_mps: float = DEFAULT_MAX_SCRIPTED_SPEED_MPS,
) -> list[str]:
    """Return fail-closed runtime plausibility errors for a scripted multi-ped config."""
    errors = config.validate()
    if min_start_separation_m < 0.0:
        errors.append("min_start_separation_m must be >= 0")
    if min_obstacle_clearance_m < 0.0:
        errors.append("min_obstacle_clearance_m must be >= 0")
    if max_speed_mps <= 0.0:
        errors.append("max_speed_mps must be > 0")
    if errors:
        return errors

    for index, pedestrian in enumerate(config.pedestrians):
        if pedestrian.speed_mps > max_speed_mps:
            errors.append(
                f"pedestrians[{index}] speed_mps ({pedestrian.speed_mps:.3f}) exceeds "
                f"max_speed_mps ({max_speed_mps:.3f})"
            )
        start = pedestrian.start.x, pedestrian.start.y
        goal = pedestrian.goal.x, pedestrian.goal.y
        errors.extend(
            _point_runtime_errors(
                start,
                label=f"pedestrians[{index}].start",
                base_map=base_map,
                min_obstacle_clearance_m=min_obstacle_clearance_m,
            )
        )
        errors.extend(
            _point_runtime_errors(
                goal,
                label=f"pedestrians[{index}].goal",
                base_map=base_map,
                min_obstacle_clearance_m=min_obstacle_clearance_m,
            )
        )

    for left, right in combinations(config.pedestrians, 2):
        separation = dist((left.start.x, left.start.y), (right.start.x, right.start.y))
        if separation < min_start_separation_m:
            errors.append(
                f"pedestrians '{left.id}' and '{right.id}' start separation "
                f"({separation:.3f}m) is less than min_start_separation_m "
                f"({min_start_separation_m:.3f}m)"
            )
    return errors


def build_multi_ped_adversarial_robot_config(
    config: MultiPedAdversarialConfig,
    base_map: MapDefinition,
    *,
    map_id: str | None = None,
    min_start_separation_m: float = DEFAULT_MIN_START_SEPARATION_M,
    min_obstacle_clearance_m: float = DEFAULT_MIN_OBSTACLE_CLEARANCE_M,
    max_speed_mps: float = DEFAULT_MAX_SCRIPTED_SPEED_MPS,
    sim_time_in_secs: float = 30.0,
    time_per_step_in_secs: float = 0.1,
) -> RobotSimulationConfig:
    """Build a robot environment config for a scripted multi-ped adversarial scenario.

    The returned config uses a deep-copied map with the generated adversarial
    ``single_pedestrians`` and zero background pedestrian density. This helper proves the
    environment reset/step path; it does not certify the scenario as benchmark-frozen.

    Raises:
        ValueError: If config or runtime plausibility checks fail.
    """
    errors = validate_multi_ped_runtime_plausibility(
        config,
        base_map,
        min_start_separation_m=min_start_separation_m,
        min_obstacle_clearance_m=min_obstacle_clearance_m,
        max_speed_mps=max_speed_mps,
    )
    if errors:
        raise ValueError("; ".join(errors))

    runtime_map_id = map_id or _default_map_id(config)
    runtime_map = _copy_map_with_single_pedestrians(
        base_map,
        multi_ped_config_to_single_pedestrian_definitions(config),
    )
    sim_config = SimulationSettings(
        sim_time_in_secs=sim_time_in_secs,
        time_per_step_in_secs=time_per_step_in_secs,
        ped_density_by_difficulty=[0.0],
        difficulty=0,
        route_spawn_seed=int(config.scenario_seed),
        max_total_pedestrians=len(config.pedestrians),
    )
    return RobotSimulationConfig(
        sim_config=sim_config,
        map_pool=MapDefinitionPool(map_defs={runtime_map_id: runtime_map}),
        map_id=runtime_map_id,
    )


def _copy_map_with_single_pedestrians(
    base_map: MapDefinition,
    single_pedestrians: list[SinglePedestrianDefinition],
) -> MapDefinition:
    """Return a validated deep copy of a map with explicit single pedestrians replaced."""
    return MapDefinition(
        width=base_map.width,
        height=base_map.height,
        obstacles=deepcopy(base_map.obstacles),
        robot_spawn_zones=deepcopy(base_map.robot_spawn_zones),
        ped_spawn_zones=deepcopy(base_map.ped_spawn_zones),
        robot_goal_zones=deepcopy(base_map.robot_goal_zones),
        bounds=deepcopy(base_map.bounds),
        robot_routes=deepcopy(base_map.robot_routes),
        ped_goal_zones=deepcopy(base_map.ped_goal_zones),
        ped_crowded_zones=deepcopy(base_map.ped_crowded_zones),
        ped_routes=deepcopy(base_map.ped_routes),
        single_pedestrians=single_pedestrians,
        poi_positions=deepcopy(base_map.poi_positions),
        poi_labels=deepcopy(base_map.poi_labels),
        allowed_areas=deepcopy(base_map.allowed_areas),
    )


def _point_runtime_errors(
    point: tuple[float, float],
    *,
    label: str,
    base_map: MapDefinition,
    min_obstacle_clearance_m: float,
) -> list[str]:
    """Return runtime errors for one candidate point."""
    errors: list[str] = []
    x, y = point
    if not (0.0 <= x <= base_map.width and 0.0 <= y <= base_map.height):
        errors.append(
            f"{label} ({x:.3f}, {y:.3f}) is outside map bounds "
            f"(0, 0) to ({base_map.width:.3f}, {base_map.height:.3f})"
        )
        return errors

    shapely_point = Point(point)
    for obstacle_index, obstacle in enumerate(base_map.obstacles):
        if obstacle.contains_point(point):
            errors.append(f"{label} ({x:.3f}, {y:.3f}) is inside obstacle {obstacle_index}")
            continue
        geometry = getattr(obstacle, "geometry", None)
        if geometry is None or min_obstacle_clearance_m <= 0.0:
            continue
        clearance = float(geometry.distance(shapely_point))
        if clearance < min_obstacle_clearance_m:
            errors.append(
                f"{label} ({x:.3f}, {y:.3f}) obstacle clearance ({clearance:.3f}m) is less "
                f"than min_obstacle_clearance_m ({min_obstacle_clearance_m:.3f}m)"
            )
    return errors


def _tuple_point(value: object) -> tuple[float, float]:
    """Coerce a materialized point into a runtime tuple."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"point must be a 2-item list or tuple, got: {value!r}")
    return (float(value[0]), float(value[1]))


def _default_map_id(config: MultiPedAdversarialConfig) -> str:
    """Return a stable map id for an adversarial runtime map."""
    family = re.sub(r"[^A-Za-z0-9_]+", "_", config.family).strip("_") or "multi_ped"
    return f"{family}_multi_ped_adversarial_{int(config.scenario_seed):04d}"


__all__ = [
    "DEFAULT_MAX_SCRIPTED_SPEED_MPS",
    "DEFAULT_MIN_OBSTACLE_CLEARANCE_M",
    "DEFAULT_MIN_START_SEPARATION_M",
    "build_multi_ped_adversarial_robot_config",
    "multi_ped_config_to_single_pedestrian_definitions",
    "validate_multi_ped_runtime_plausibility",
]
