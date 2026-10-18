"""Bounded real-simulator contact and blocked-route diagnostics, not release evidence."""

import faulthandler
import hashlib
import json
import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from pysocialforce.forces import (
    GroupCoherenceForceAlt,
    GroupGazeForceAlt,
    GroupRepulsiveForce,
    ObstacleForce,
    SocialForce,
    closest_point_on_segment,
    obstacle_force_for_law,
)
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

import robot_sf
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import Simulator, init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios


class _TimedBehavior(Protocol):
    """Writable timestep supplied by the time-based pedestrian behaviours."""

    time_step_s: float


class _ForceComputer(Protocol):
    """Per-instance force callback replaced by the diagnostic trace closure."""

    compute_forces: Callable[[], np.ndarray]


faulthandler.register(signal.SIGUSR1)
root = Path(os.environ["WALL_CONTACT_ARTIFACT_ROOT"])
manifest = json.loads((root / "measurement-manifest.json").read_text())
version, index, output = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
task = manifest["tasks"][index]
source = Path(os.environ["SOURCE_ROOT"])
assert Path(robot_sf.__file__).resolve().parents[1] == source.resolve()
seed = task["seed"]
assert 1001 <= seed <= 1200
np.random.seed(seed)
law = manifest["laws"][version]
controls = manifest.get("controls", {}).get(version, {})
dt = float(controls.get("dt_s", 0.1))


def aperture_map(width):
    """Build the original synthetic aperture without changing its body or endpoints."""
    walls = [
        Obstacle([(4.9, 0), (5.1, 0), (5.1, 4 - width / 2), (4.9, 4 - width / 2)]),
        Obstacle([(4.9, 4 + width / 2), (5.1, 4 + width / 2), (5.1, 8), (4.9, 8)]),
    ]
    spawn, goal = ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8)), ((9.2, 7.2), (9.8, 7.2), (9.2, 7.8))
    jitter = float(np.random.default_rng(seed).uniform(-0.04, 0.04))
    return MapDefinition(
        width=10,
        height=8,
        obstacles=walls,
        robot_spawn_zones=[spawn],
        robot_goal_zones=[goal],
        ped_spawn_zones=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        bounds=[((0, 0), (10, 0)), ((10, 0), (10, 8)), ((10, 8), (0, 8)), ((0, 8), (0, 0))],
        robot_routes=[
            GlobalRoute(
                spawn_id=0,
                goal_id=0,
                waypoints=[(0.5, 0.5), (9.5, 7.5)],
                spawn_zone=spawn,
                goal_zone=goal,
            )
        ],
        single_pedestrians=[
            SinglePedestrianDefinition(
                id="lone-aperture",
                start=(2, 4 + jitter),
                trajectory=[(8, 4 + jitter)],
                speed_m_s=0.65,
            )
        ],
    )


if task["kind"] == "aperture":
    map_def = aperture_map(task["width_m"])
    settings = SimulationSettings(
        sim_time_in_secs=70,
        time_per_step_in_secs=0.1,
        difficulty=0,
        ped_density_by_difficulty=[0.0],
        pedestrian_seed=seed,
        route_spawn_seed=seed,
        max_total_pedestrians=1,
        obstacle_force_law=law,
    )
    sim = Simulator(
        settings,
        map_def,
        robots=[],
        goal_proximity_threshold=0.0,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )
else:
    path = source / task["manifest"]
    scenario = next(
        row for row in load_scenarios(path) if row.get("name", row.get("id")) == task["scenario"]
    )
    cfg = build_robot_config_from_scenario(scenario, scenario_path=path)
    settings = cfg.sim_config
    settings.obstacle_force_law = law
    settings.pedestrian_seed = settings.route_spawn_seed = seed
    settings.sim_time_in_secs = 70
    settings.time_per_step_in_secs = 0.1
    map_def = next(iter(cfg.map_pool.map_defs.values()))
    sim = init_simulators(cfg, map_def, num_robots=1, random_start_pos=False)[0]

assert sim.config.obstacle_force_law == law
for behavior in sim.peds_behaviors:
    if hasattr(behavior, "time_step_s"):
        cast("_TimedBehavior", behavior).time_step_s = dt
sim.config.time_per_step_in_secs = dt
sim.pysf_sim.peds.d_t = dt
sim.pysf_sim.config.obstacle_force_config.factor *= float(controls.get("wall_scale", 1.0))
for component in sim.pysf_sim.forces:
    if type(component).__name__ == "SocialForce" and controls.get("disable_social"):
        cast("SocialForce", component).config.factor = 0.0
    if type(component).__name__.startswith("Group") and controls.get("disable_groups"):
        # The simulator registers these three group forces; keep the diagnostic's name guard.
        cast(
            "GroupCoherenceForceAlt | GroupRepulsiveForce | GroupGazeForceAlt", component
        ).config.factor = 0.0
state = np.asarray(sim.pysf_state.pysf_states(), dtype=float).copy()
assert len(state) > 0, "MISSING-PEDESTRIANS"
radius = float(settings.ped_radius)
goals = state[:, 4:6].copy()
for behavior in sim.peds_behaviors:
    for gid, route in getattr(behavior, "route_assignments", {}).items():
        for ped_id in sim.groups.groups[gid]:
            goals[ped_id] = route.waypoints[-1]
    for runtime in getattr(behavior, "_runtimes", []):
        if runtime.trajectory:
            goals[runtime.ped_id] = runtime.trajectory[-1]
        elif runtime.definition.goal is not None:
            goals[runtime.ped_id] = runtime.definition.goal

polygons = [polygon for obstacle in map_def.obstacles for polygon in obstacle.iter_polygons()]
segments = [LineString(edge[:4].reshape(2, 2)) for edge in np.asarray(sim.get_obstacle_lines())]
geometry = unary_union([*polygons, *segments])
positions, velocities, current_goals = [], [], []
wall_clearances, pair_clearances = [], []
respawns = np.zeros(len(state), dtype=int)
distance_travelled = np.zeros(len(state))
minimum_goal_distance = np.full(len(state), np.inf)
stopping_run = stopping_max = 0
trace_enabled = task.get("scenario") == "classic_realworld_double_bottleneck_high"
force_states, force_components, segment_forces, closest_points, integration = [], [], [], [], []
raw_edges = np.asarray(sim.pysf_sim.get_raw_obstacles(), dtype=float)
wall_component = next(f for f in sim.pysf_sim.forces if isinstance(f, ObstacleForce))
component_names = []
if trace_enabled:

    def traced_compute():
        """Evaluate each native component once and retain its integration inputs."""
        current = np.asarray(sim.pysf_state.pysf_states(), dtype=float).copy()
        values = [np.asarray(f(), dtype=float) for f in sim.pysf_sim.forces]
        total = np.zeros_like(current[:, :2])
        for value in values:
            total += value
        force_states.append(current)
        force_components.append(np.asarray(values))
        component_names[:] = [type(f).__name__ for f in sim.pysf_sim.forces]
        offset = (
            (
                wall_component.config.threshold
                + sim.pysf_sim.peds.agent_radius * wall_component.config.sigma
            )
            if law == "legacy_shifted_gradient_v1"
            else sim.pysf_sim.peds.agent_radius
        )
        segment_forces.append(
            np.asarray(
                [
                    [
                        obstacle_force_for_law(
                            tuple(edge[:4]), tuple(edge[4:]), tuple(point), offset, law
                        )
                        for edge in raw_edges
                    ]
                    for point in current[:, :2]
                ]
            )
            * wall_component.config.factor
        )
        closest_points.append(
            np.asarray(
                [
                    [closest_point_on_segment(tuple(edge[:4]), tuple(point)) for edge in raw_edges]
                    for point in current[:, :2]
                ]
            )
        )
        diagnostic = sim.pysf_sim.peds.compute_step_diagnostics(total)
        integration.append(
            np.asarray(
                [
                    diagnostic.previous_velocity,
                    diagnostic.uncapped_velocity,
                    diagnostic.applied_velocity,
                    diagnostic.position_velocity,
                ]
            )
        )
        return total

    # A per-instance closure is intentionally installed without binding a self argument.
    cast("_ForceComputer", sim.pysf_sim).compute_forces = traced_compute

steps = round(70 / dt)
for tick in range(steps + 1):
    now = np.asarray(sim.pysf_state.pysf_states(), dtype=float).copy()
    assert now.shape == state.shape
    p = now[:, :2]
    wall = np.array([Point(point).distance(geometry) - radius for point in p])
    pair = np.linalg.norm(p[:, None] - p[None, :], axis=-1) - 2 * radius
    np.fill_diagonal(pair, np.inf)
    if tick:
        previous = positions[-1]
        delta = p - previous
        jumps = np.linalg.norm(delta, axis=1) > 0.5
        respawns += jumps
        distance_travelled += np.where(jumps, 0, np.linalg.norm(delta, axis=1))
        for ped_id in np.flatnonzero(~jumps):
            wall[ped_id] = min(
                wall[ped_id], LineString([previous[ped_id], p[ped_id]]).distance(geometry) - radius
            )
        relative = previous[:, None] - previous[None, :]
        motion = delta[:, None] - delta[None, :]
        denominator = np.sum(motion * motion, axis=-1)
        t = np.clip(
            np.divide(
                -np.sum(relative * motion, axis=-1),
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            ),
            0,
            1,
        )
        swept = np.linalg.norm(relative + t[..., None] * motion, axis=-1) - 2 * radius
        swept[jumps, :] = np.inf
        swept[:, jumps] = np.inf
        np.fill_diagonal(swept, np.inf)
        pair = np.minimum(pair, swept)
    positions.append(p.copy())
    velocities.append(now[:, 2:4].copy())
    current_goals.append(now[:, 4:6].copy())
    wall_clearances.append(wall)
    pair_clearances.append(float(np.min(pair)) if len(p) > 1 else None)
    minimum_goal_distance = np.minimum(minimum_goal_distance, np.linalg.norm(p - goals, axis=1))
    if task["kind"] == "aperture" and tick:
        stopped = 2.5 < p[0, 0] < 5.5 and np.linalg.norm(now[0, 2:4]) < 0.05
        stopping_run = stopping_run + 1 if stopped else 0
        stopping_max = max(stopping_max, stopping_run)
    if tick < steps:
        sim.step_once([(0.0, 0.0)] * len(sim.robots))

positions = np.asarray(positions)
velocities = np.asarray(velocities)
wall_clearances = np.asarray(wall_clearances)
final_goal_distance = np.linalg.norm(positions[-1] - goals, axis=1)
recent_displacement = np.linalg.norm(positions[-1] - positions[-round(10 / dt) - 1], axis=1)
goal_reached = minimum_goal_distance <= 1.0
blocked = (
    (~goal_reached) & (respawns == 0) & (recent_displacement < 0.25) & (wall_clearances[-1] < 1.5)
)
np.savez_compressed(
    output / "trajectory.npz",
    positions=positions,
    velocities=velocities,
    current_goals=np.asarray(current_goals),
    assigned_goals=goals,
    wall_clearances=wall_clearances,
)
trace_digest = hashlib.sha256((output / "trajectory.npz").read_bytes()).hexdigest()
if trace_enabled:
    np.savez_compressed(
        output / "force-trace.npz",
        force_states=np.asarray(force_states),
        force_components=np.asarray(force_components),
        segment_forces=np.asarray(segment_forces),
        closest_points=np.asarray(closest_points),
        integration=np.asarray(integration),
        raw_edges=raw_edges,
    )
    (output / "geometry.json").write_text(
        json.dumps(
            {
                "polygons_wkt": [p.wkt for p in polygons],
                "component_names": component_names,
                "wall_factor": wall_component.config.factor,
                "integration_scheme": sim.pysf_sim.peds.integration_scheme,
                "geometry_sha256": hashlib.sha256(geometry.wkb).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
valid_pairs = [value for value in pair_clearances if value is not None]
metrics = {
    "task": task,
    "version": version,
    "source_sha": manifest["sources"][version],
    "law": law,
    "dt_s": dt,
    "duration_s": 70,
    "steps": steps,
    "pedestrians": len(state),
    "geometry_sha256": hashlib.sha256(geometry.wkb).hexdigest(),
    "controls": controls,
    "physical_radius_m": radius,
    "force_radius_m": float(sim.pysf_sim.peds.agent_radius),
    "min_wall_clearance_m": float(np.min(wall_clearances)),
    "initial_min_wall_clearance_m": float(np.min(wall_clearances[0])),
    "wall_penetration_pedestrian_steps": int(np.count_nonzero(wall_clearances < -1e-9)),
    "min_pair_clearance_m": min(valid_pairs) if valid_pairs else None,
    "initial_min_pair_clearance_m": pair_clearances[0],
    "pair_overlap_steps": sum(value is not None and value < -1e-9 for value in pair_clearances),
    "assigned_goal_reached": int(np.count_nonzero(goal_reached)),
    "blocked_pedestrians": int(np.count_nonzero(blocked)),
    "trajectory_sha256": trace_digest,
    "per_pedestrian": [
        {
            "id": i,
            "assigned_goal": goals[i].tolist(),
            "start_goal_distance_m": float(np.linalg.norm(positions[0, i] - goals[i])),
            "min_goal_distance_m": float(minimum_goal_distance[i]),
            "final_goal_distance_m": float(final_goal_distance[i]),
            "goal_reached_within_1m": bool(goal_reached[i]),
            "path_length_m_excluding_respawns": float(distance_travelled[i]),
            "mean_speed_mps": float(np.mean(np.linalg.norm(velocities[:, i], axis=-1))),
            "last_10s_displacement_m": float(recent_displacement[i]),
            "respawn_or_relocation_events": int(respawns[i]),
            "obstacle_blocked": bool(blocked[i]),
        }
        for i in range(len(state))
    ],
}
if task["kind"] == "aperture":
    metrics.update(
        crossed_gap=bool(np.any(positions[:, 0, 0] >= 5.6)), max_pre_gap_stop_s=stopping_max * dt
    )
(output / "measurement.json").write_text(json.dumps(metrics, indent=2) + "\n")
print(
    json.dumps(
        {
            key: metrics[key]
            for key in (
                "task",
                "version",
                "source_sha",
                "pedestrians",
                "min_wall_clearance_m",
                "min_pair_clearance_m",
                "assigned_goal_reached",
                "blocked_pedestrians",
            )
        }
    )
)
