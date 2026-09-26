"""Release-arm episode harness for planner metamorphic relations.

The release ``social_force``, ``orca`` and hybrid v3 arms are built with the same
map-runner policy builder, structured observation, occupancy grid, and command
conversion used by benchmark runs. Scenes are small synthetic maps so a relation
runs in seconds; the harness is a test fixture, not benchmark evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.map_runner.map_runner import build_map_policy
from robot_sf.benchmark.map_runner_policies.map_runner_actions import (
    policy_command_to_env_action,
)
from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.observation_mode import ObservationMode
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.occupancy_grid import GridChannel, GridConfig
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sim.sim_config import SimulationSettings

ROOT = Path(__file__).resolve().parents[2]
RELEASE_MANIFEST = ROOT / "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml"
# The template is the successor campaign (0.0.7 onward); it adds the terminal-goal
# social-force input. Both planner lists are read; the successor entry wins for arms.
RELEASE_TEMPLATE_CAMPAIGN = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
)
HYBRID_V3_ARM = "hybrid_rule_v3_fast_progress_static_escape"
RELEASE_ARMS = ("social_force", "orca", HYBRID_V3_ARM)

MAP_SIZE = 20.0
DT = 0.1

Point = tuple[float, float]
PointTransform = Callable[[Point], Point]


def load_yaml(relative: str | Path) -> dict[str, Any]:
    """Load one repository YAML mapping.

    Returns:
        The parsed mapping.
    """
    payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8")) or {}
    assert isinstance(payload, dict), relative
    return payload


def release_campaign_planners() -> tuple[dict[str, Any], ...]:
    """Return release campaign planner entries, canonical campaign first.

    Returns:
        Unique ``(key, algo, algo_config)`` entries in campaign order.
    """
    manifest = load_yaml(RELEASE_MANIFEST.relative_to(ROOT))
    canonical = (RELEASE_MANIFEST.parent / manifest["canonical_campaign_config"]).resolve()
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for campaign in (canonical, RELEASE_TEMPLATE_CAMPAIGN):
        for entry in load_yaml(campaign.relative_to(ROOT))["planners"]:
            identity = (entry["key"], entry["algo"], entry.get("algo_config"))
            if identity not in seen:
                seen.add(identity)
                entries.append(dict(entry))
    return tuple(entries)


def _load_base_config(config_path: object) -> dict[str, Any]:
    """Resolve a manifest ``base_config_path`` from the repository root.

    Returns:
        The base mapping, or an empty mapping when none is declared.
    """
    if not isinstance(config_path, str) or not config_path.strip():
        return {}
    return load_yaml(config_path)


def resolve_release_algo_config(
    algo: str, algo_config: str | None, scenario: str = "__default__"
) -> tuple[str, dict[str, Any]]:
    """Resolve one release ``algo_config`` for a scenario as the map runner does.

    Returns:
        The effective algorithm key and flattened runtime config.
    """
    if not algo_config:
        return algo, {}
    return resolve_candidate_manifest_runtime(
        default_algo=algo,
        manifest=load_yaml(algo_config),
        scenario={"name": scenario},
        load_config=_load_base_config,
    )


def release_arm(key: str) -> tuple[str, dict[str, Any]]:
    """Return the runtime ``(algo, config)`` of one release arm (successor entry wins).

    Returns:
        Effective algorithm key and config for the synthetic test scenario.
    """
    matches = [entry for entry in release_campaign_planners() if entry["key"] == key]
    assert matches, f"release roster has no arm {key!r}"
    entry = matches[-1]
    return resolve_release_algo_config(entry["algo"], entry.get("algo_config"), "metamorphic")


def identity(point: Point) -> Point:
    """Return the point unchanged."""
    return (float(point[0]), float(point[1]))


def mirror_y(point: Point) -> Point:
    """Reflect across the horizontal centre line ``y = MAP_SIZE / 2``."""
    return (float(point[0]), MAP_SIZE - float(point[1]))


def mirror_x(point: Point) -> Point:
    """Reflect across the vertical centre line ``x = MAP_SIZE / 2``."""
    return (MAP_SIZE - float(point[0]), float(point[1]))


def rotate_90(point: Point) -> Point:
    """Rotate 90 degrees counter-clockwise about the map centre."""
    return (MAP_SIZE - float(point[1]), float(point[0]))


def _bounds(transform: PointTransform) -> list[tuple[Point, Point]]:
    corners = ((0.0, 0.0), (MAP_SIZE, 0.0), (MAP_SIZE, MAP_SIZE), (0.0, MAP_SIZE))
    return [(transform(corners[index]), transform(corners[(index + 1) % 4])) for index in range(4)]


def interaction_scene(
    transform: PointTransform = identity,
    *,
    pedestrian: bool = True,
    obstacle: bool = True,
) -> MapDefinition:
    """Build a fixed-start scene with a diagonal crossing pedestrian and a side obstacle.

    Coordinates avoid multiples of the 0.2 m grid cell relative to the robot, so a
    reflected scene rasterizes to the reflected grid rather than to a cell-boundary
    tie. The pedestrian moves with both velocity components, crossing both mirror axes.

    Returns:
        The (optionally transformed) map definition.
    """
    start, goal = (3.0, 10.0), (12.0, 10.0)
    spawn = tuple(transform(start) for _ in range(3))
    goal_zone = tuple(transform(goal) for _ in range(3))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[transform(start), transform(goal)],
        spawn_zone=spawn,  # type: ignore[arg-type]
        goal_zone=goal_zone,  # type: ignore[arg-type]
    )
    obstacle_vertices = [(6.03, 10.67), (7.03, 10.67), (7.03, 11.47), (6.03, 11.47)]
    pedestrians = (
        [
            SinglePedestrianDefinition(
                id="diagonal",
                start=transform((9.03, 7.07)),
                goal=transform((6.03, 13.07)),
                speed_m_s=0.8,
            )
        ]
        if pedestrian
        else []
    )
    return MapDefinition(
        width=MAP_SIZE,
        height=MAP_SIZE,
        obstacles=(
            [Obstacle([transform(point) for point in obstacle_vertices])] if obstacle else []
        ),
        robot_spawn_zones=[spawn],  # type: ignore[list-item]
        ped_spawn_zones=[],
        robot_goal_zones=[goal_zone],  # type: ignore[list-item]
        bounds=_bounds(transform),
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=pedestrians,
    )


def crossing_scene(*, pedestrian: bool, pedestrian_start_y: float = 4.07) -> MapDefinition:
    """Build a straight 12 m route crossed at x = 9.03 m by one 1 m/s pedestrian.

    Returns:
        The map definition with or without the crossing pedestrian.
    """
    start, goal = (3.0, 10.0), (15.0, 10.0)
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[start, goal],
        spawn_zone=(start, start, start),
        goal_zone=(goal, goal, goal),
    )
    pedestrians = (
        [
            SinglePedestrianDefinition(
                id="crossing",
                start=(9.03, pedestrian_start_y),
                goal=(9.03, pedestrian_start_y + 14.0),
                speed_m_s=1.0,
            )
        ]
        if pedestrian
        else []
    )
    return MapDefinition(
        width=MAP_SIZE,
        height=MAP_SIZE,
        obstacles=[],
        robot_spawn_zones=[(start, start, start)],
        ped_spawn_zones=[],
        robot_goal_zones=[(goal, goal, goal)],
        bounds=_bounds(identity),
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=pedestrians,
    )


def sampled_scene() -> MapDefinition:
    """Build a scene whose robot start, goal, and crowd are all sampled from areas.

    Returns:
        A map definition with real-area zones and a route plus crowded-zone crowd.
    """
    robot_spawn = ((2.0, 8.0), (4.0, 8.0), (4.0, 12.0))
    robot_goal = ((15.0, 8.0), (17.0, 8.0), (17.0, 12.0))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(3.0, 10.0), (16.0, 10.0)],
        spawn_zone=robot_spawn,
        goal_zone=robot_goal,
    )
    ped_spawn = ((8.0, 3.0), (12.0, 3.0), (12.0, 5.0))
    ped_goal = ((8.0, 15.0), (12.0, 15.0), (12.0, 17.0))
    ped_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(10.0, 4.0), (10.0, 16.0)],
        spawn_zone=ped_spawn,
        goal_zone=ped_goal,
    )
    return MapDefinition(
        width=MAP_SIZE,
        height=MAP_SIZE,
        obstacles=[],
        robot_spawn_zones=[robot_spawn],
        ped_spawn_zones=[ped_spawn],
        robot_goal_zones=[robot_goal],
        bounds=_bounds(identity),
        robot_routes=[route],
        ped_goal_zones=[ped_goal],
        ped_crowded_zones=[((6.0, 6.0), (9.0, 6.0), (9.0, 14.0))],
        ped_routes=[ped_route],
        single_pedestrians=[],
    )


SAMPLED_PED_DENSITY = 0.06


def robot_env_config(
    map_def: MapDefinition,
    *,
    max_steps: int,
    ped_density: float = 0.0,
    observation_mode: ObservationMode = ObservationMode.SOCNAV_STRUCT,
) -> RobotSimulationConfig:
    """Return the benchmark map-runner env config for one synthetic map.

    The observation mode and ego-frame occupancy grid match
    ``map_runner_env.build_env_config``.

    Returns:
        A differential-drive robot config over the synthetic map.
    """
    config = RobotSimulationConfig()
    config.sim_config = SimulationSettings(
        sim_time_in_secs=max_steps * DT,
        time_per_step_in_secs=DT,
        ped_density_by_difficulty=[ped_density],
        difficulty=0,
        max_total_pedestrians=12,
    )
    config.map_pool = MapDefinitionPool(map_defs={"metamorphic": map_def})
    config.map_id = "metamorphic"
    config.robot_config = DifferentialDriveSettings()
    config.observation_mode = observation_mode
    if observation_mode == ObservationMode.SOCNAV_STRUCT:
        config.use_occupancy_grid = True
        config.include_grid_in_observation = True
        config.grid_config = GridConfig(
            resolution=0.2,
            width=32.0,
            height=32.0,
            channels=[GridChannel.OBSTACLES, GridChannel.PEDESTRIANS, GridChannel.COMBINED],
            use_ego_frame=True,
            center_on_robot=True,
        )
    return config


@dataclass(frozen=True, slots=True)
class ArmEpisode:
    """Immutable closed-loop trace of one release-arm episode."""

    poses: tuple[tuple[float, float, float], ...]
    commands: tuple[tuple[float, float], ...]
    actions: tuple[tuple[float, float], ...]
    pedestrian_positions: tuple[tuple[tuple[float, float], ...], ...]
    success: bool
    collision: bool
    step_limit_reached: bool
    status: str


def run_arm_episode(
    arm: str,
    map_def: MapDefinition,
    *,
    seed: int,
    max_steps: int,
    ped_density: float = 0.0,
) -> ArmEpisode:
    """Drive one release arm through a seeded map-runner-style episode.

    ``step_limit_reached`` means the loop used ``max_steps`` without the
    environment terminating; it is not the environment's time-limit flag.

    Returns:
        The robot poses (including the reset pose), commands, env actions, and outcome.
    """
    algo, algo_config = release_arm(arm)
    config = robot_env_config(map_def, max_steps=max_steps + 1, ped_density=ped_density)
    env = make_robot_env(config=config, seed=seed)
    policy, meta = build_map_policy(algo, dict(algo_config), robot_kinematics="differential_drive")
    poses: list[tuple[float, float, float]] = []
    commands: list[tuple[float, float]] = []
    actions: list[tuple[float, float]] = []
    pedestrians: list[tuple[tuple[float, float], ...]] = []
    info: dict[str, Any] = {}
    terminated = truncated = False

    def record_state() -> None:
        pose = env.simulator.robot_poses[0]
        poses.append((float(pose[0][0]), float(pose[0][1]), float(pose[1])))
        pedestrians.append(
            tuple((float(x), float(y)) for x, y in np.asarray(env.simulator.ped_pos, dtype=float))
        )

    try:
        observation, info = env.reset(seed=seed)
        reset = getattr(policy, "reset", None)
        if callable(reset):
            try:
                reset(seed=seed)
            except TypeError:
                reset()
        record_state()
        for _ in range(max_steps):
            command = policy(observation)
            action = policy_command_to_env_action(env=env, config=config, command=command)
            observation, _reward, terminated, truncated, info = env.step(action)
            values = np.asarray(command, dtype=float).reshape(-1)
            action_values = np.asarray(action, dtype=float).reshape(-1)
            commands.append((float(values[0]), float(values[1])))
            actions.append((float(action_values[0]), float(action_values[1])))
            record_state()
            if terminated or truncated:
                break
    finally:
        env.close()
    return ArmEpisode(
        poses=tuple(poses),
        commands=tuple(commands),
        actions=tuple(actions),
        pedestrian_positions=tuple(pedestrians),
        success=bool(info.get("success", False)),
        collision=bool(info.get("collision", False)),
        step_limit_reached=not (terminated or truncated),
        status=str(meta.get("status")),
    )
