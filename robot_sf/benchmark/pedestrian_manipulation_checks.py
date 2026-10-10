"""Small native-substrate manipulation probes for #10190 / #10074.

These simplified rigid-disc probes are diagnostics, not source-study replications.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import Point

from robot_sf.benchmark.pedestrian_sensitivity import profile_scenario, require_dev_seeds
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sim.simulator import _build_pysf_simulation
from robot_sf.training.scenario_loader import build_robot_config_from_scenario

if TYPE_CHECKING:
    from pysocialforce import Simulator as PySFSimulator

    from robot_sf.ped_npc.ped_behavior import PedestrianBehavior


def build_probe(
    profile: dict,
    seed: int,
    actors: list,
    obstacles: list,
    *,
    width: float = 120,
    height: float = 140,
) -> tuple[PySFSimulator, list[PedestrianBehavior], float]:
    """Build the same production pedestrian substrate as benchmark episodes.

    Returns:
        PySF simulator, behavior list and configured collision radius.
    """
    require_dev_seeds([seed])
    scenario = profile_scenario({"simulation_config": {"ped_density": 0}}, profile, seed)
    config = build_robot_config_from_scenario(scenario, scenario_path=Path(__file__)).sim_config
    bounds = [
        ((0, 0), (width, 0)),
        ((width, 0), (width, height)),
        ((width, height), (0, height)),
        ((0, height), (0, 0)),
    ]
    remote_zone = ((width - 2, height - 2), (width - 1, height - 2), (width - 2, height - 1))
    map_def = MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles,
        robot_spawn_zones=[remote_zone],
        ped_spawn_zones=[],
        robot_goal_zones=[remote_zone],
        bounds=bounds,
        robot_routes=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=actors,
    )
    sim, _, _, behaviors, _ = _build_pysf_simulation(
        config=config,
        map_def=map_def,
        robots=[],
        robot_pose_provider=lambda: [],
        peds_have_obstacle_forces=True,
    )
    return sim, behaviors, config.ped_radius


def advance(sim, behaviors) -> None:
    """Advance native behavioral controls before one physics step."""
    for behavior in behaviors:
        behavior.step()
    sim.step()
    if not np.all(np.isfinite(sim.peds.state)):
        raise ValueError("nonfinite manipulation trajectory")


def manipulation_checks(profile: dict, seeds: list[int]) -> dict:
    """Measure desired/realized speed, aperture passage/contact, and head-on clearance.

    Returns:
        Raw seed-level observations and explicit source-comparison limitations.
    """
    require_dev_seeds(seeds)
    free, doors, passing = [], [], []
    for seed in seeds:
        actors = [
            SinglePedestrianDefinition(id=f"free-{i}", start=(5, 4 * i + 4), goal=(110, 4 * i + 4))
            for i in range(32)
        ]
        sim, behaviors, radius = build_probe(profile, seed, actors, [])
        desired = sim.peds.max_speeds.copy()
        speeds = []
        for step in range(100):
            advance(sim, behaviors)
            if step >= 50:
                speeds.append(np.linalg.norm(sim.peds.vel(), axis=1))
        free.append(
            {
                "seed": seed,
                "desired_m_s": desired.tolist(),
                "realized_5_to_10_s_m_s": np.mean(speeds, axis=0).tolist(),
                "force_radius_m": float(sim.peds.agent_radius),
                "collision_radius_m": radius,
                "wall_law": str(sim.config.obstacle_force_config.law_version),
                "wall_factor": float(sim.config.obstacle_force_config.factor),
                "wall_threshold_m": float(sim.config.obstacle_force_config.threshold),
            }
        )
        for aperture in (0.8, 1.0, 1.2, 1.4):
            lower, upper = 3 - aperture / 2, 3 + aperture / 2
            obstacles = [
                Obstacle([(7.9, 0), (8.1, 0), (8.1, lower), (7.9, lower)]),
                Obstacle([(7.9, upper), (8.1, upper), (8.1, 6), (7.9, 6)]),
            ]
            sim, behaviors, radius = build_probe(
                profile,
                seed,
                [SinglePedestrianDefinition(id="door", start=(4, 3), goal=(12, 3))],
                obstacles,
                width=16,
                height=6,
            )
            trace, clearance, passage_s, contact_steps = [], float("inf"), None, 0
            for step in range(400):
                advance(sim, behaviors)
                xy = sim.peds.pos()[0]
                gap = min(Point(xy).distance(obstacle.geometry) for obstacle in obstacles) - radius
                clearance = min(clearance, gap)
                contact_steps += int(gap < 0)
                trace.append([float(xy[0]), float(xy[1]), float(np.linalg.norm(sim.peds.vel()[0]))])
                if xy[0] > 8.1 + radius:
                    passage_s = (step + 1) * 0.1
                    break
            doors.append(
                {
                    "seed": seed,
                    "width_m": aperture,
                    "passage_s": passage_s,
                    "contact_steps": contact_steps,
                    "min_surface_clearance_m": clearance,
                    "trace_xy_speed": trace,
                }
            )
        sim, behaviors, radius = build_probe(
            profile,
            seed,
            [
                SinglePedestrianDefinition(id="a", start=(4, 2.98), goal=(12, 2.98)),
                SinglePedestrianDefinition(id="b", start=(12, 3.02), goal=(4, 3.02)),
            ],
            [],
            width=16,
            height=6,
        )
        minimum, swapped, trace = float("inf"), False, []
        for _ in range(200):
            advance(sim, behaviors)
            xy = sim.peds.pos()
            minimum = min(minimum, float(np.linalg.norm(xy[0] - xy[1])))
            trace.append(xy.tolist())
            if xy[0, 0] > xy[1, 0]:
                swapped = True
                break
        passing.append(
            {
                "seed": seed,
                "passed": swapped,
                "min_centre_distance_m": minimum,
                "min_surface_clearance_m": minimum - 2 * radius,
                "trace_xy": trace,
            }
        )
    desired_all = [x for row in free for x in row["desired_m_s"]]
    realized_all = [x for row in free for x in row["realized_5_to_10_s_m_s"]]
    return {
        "evidence_status": "diagnostic-only",
        "free_speed": {
            "raw": free,
            "desired_mean_m_s": float(np.mean(desired_all)),
            "desired_sd_m_s": float(np.std(desired_all)),
            "realized_mean_m_s": float(np.mean(realized_all)),
            "realized_sd_m_s": float(np.std(realized_all)),
        },
        "doorway": doors,
        "head_on": passing,
        "targets": {
            "free_speed_mean_sd_m_s": [1.29, 0.19],
            "doorway": "#10074 V2: passage without contact; human shoulder rotation absent",
            "head_on": "#10074 V6 measures onset, not minimum passing distance",
        },
        "limitations": "32 separated walkers; dt=0.1 s; 40 s doorway cap; 20 s head-on cap; "
        "head-on lateral asymmetry +/-0.02 m. No body rotation, no empirical passing-distance "
        "acceptance threshold; V6 onset is not inferred from minimum distance. "
        "Door widths 0.8..1.4 m are study probes, not Wilmut's source protocol.",
    }
