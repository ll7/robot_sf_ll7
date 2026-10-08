"""Diagnostic ON/OFF production doorway probes, never an acceptance gate.

Geometry follows tests/sim/test_obstacle_force_profile.py::_doorway_force and
uses three intact pedestrians, including two near the corridor walls. The narrow
opening is a bottleneck stress case. Only development seeds 1001-1005 are used.
Raw traces go to the explicitly supplied external output directory.
"""

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import _build_pysf_simulation

FLAGS = {
    "10073": {"obstacle_force_profile": "gradient_v3"},
    "10075": {"obstacle_force_profile": "gradient_v3", "pedestrian_radius_m": 0.28},
    "10094": {"obstacle_force_profile": "gradient_v3", "pedestrian_radius_m": 0.28},
    "10104": {
        "obstacle_force_profile": "calibrated_v2",
        "pedestrian_radius_m": 0.28,
        "pedestrian_contact_rule": "projection_v1",
        "pedestrian_wall_rule": "bounded_edge_v1",
    },
}


def build(seed, width, flags):
    """Construct the real substrate with pedestrians intact and effective flags."""
    np.random.seed(seed)
    settings = SimulationSettings(difficulty=0, ped_density_by_difficulty=[0.0])
    for key, value in flags.items():
        setattr(settings, key, value)
    lo, hi = 2 - width / 2, 2 + width / 2
    polygons = [
        Polygon([(8, hi), (8.1, hi), (8.1, 4), (8, 4)]),
        Polygon([(8, 0), (8.1, 0), (8.1, lo), (8, lo)]),
    ]
    room = Polygon([(0, 0), (16, 0), (16, 4), (0, 4)])
    map_def = MapDefinition(
        width=16.0,
        height=4.0,
        obstacles=[Obstacle(list(p.exterior.coords)[:-1]) for p in polygons],
        robot_spawn_zones=[],
        ped_spawn_zones=[],
        robot_goal_zones=[],
        bounds=[((0, 0), (16, 0)), ((16, 0), (16, 4)), ((16, 4), (0, 4)), ((0, 4), (0, 0))],
        robot_routes=[],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[
            SinglePedestrianDefinition(id="walker", start=(6, 2), goal=(15, 2)),
            SinglePedestrianDefinition(id="near_low_wall", start=(5, 0.45), goal=(15, 1.9)),
            SinglePedestrianDefinition(id="near_high_wall", start=(5, 3.55), goal=(15, 2.1)),
        ],
    )
    sim, *_ = _build_pysf_simulation(
        config=settings,
        map_def=map_def,
        robots=[],
        robot_pose_provider=lambda: [],
        peds_have_obstacle_forces=True,
    )
    return sim, settings, polygons, room


def measure(seed, width, flags, output):
    """Record paths, contacts, signed wall clearance, speeds and swept crossings."""
    sim, settings, walls, room = build(seed, width, flags)
    radius = float(settings.ped_radius)
    states = [sim.peds.state.copy()]
    for _ in range(300):
        sim.step()
        states.append(sim.peds.state.copy())
    states = np.asarray(states)
    if not np.isfinite(states).all():
        raise ValueError("Non-finite pedestrian state")
    positions = states[:, :, :2]
    assert positions.shape == (301, 3, 2), "Pedestrian census changed"
    distances = np.linalg.norm(positions[:, :, None, :] - positions[:, None, :, :], axis=-1)
    pairs = distances[:, np.triu_indices(3, 1)[0], np.triu_indices(3, 1)[1]]
    clearances = []
    inside = 0
    tunnels = 0
    for frame in positions:
        for xy in frame:
            point = Point(xy)
            room_clearance = point.distance(room.boundary) * (1 if room.covers(point) else -1)
            obstacle_clearance = min(
                -point.distance(wall.boundary) if wall.contains(point) else point.distance(wall)
                for wall in walls
            )
            clearances.append(min(room_clearance, obstacle_clearance) - radius)
            inside += int(any(wall.contains(point) for wall in walls) or not room.covers(point))
    for before, after in pairwise(positions):
        for a, b in zip(before, after, strict=True):
            if np.array_equal(a, b):
                continue
            segment = LineString([a, b])
            for wall in walls:
                interior = wall.buffer(-1e-9)
                tunnels += int(
                    not wall.covers(Point(a))
                    and not wall.covers(Point(b))
                    and segment.intersects(interior)
                )
    encoded = json.dumps(states.tolist(), separators=(",", ":")).encode()
    output.write_bytes(encoded + b"\n")
    force = sim.config.obstacle_force_config
    return {
        "seed": seed,
        "opening_m": width,
        "flags": flags,
        "pedestrians": 3,
        "steps": 300,
        "dt_s": float(sim.config.scene_config.dt_secs),
        "physical_radius_m": radius,
        "force_radius_m": float(sim.config.scene_config.agent_radius),
        "force": {
            "factor": force.factor,
            "threshold": force.threshold,
            "sigma": force.sigma,
            "law": force.law_version,
        },
        "trajectory_sha256": hashlib.sha256(encoded).hexdigest(),
        "raw_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "path_length_m": np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum(axis=0).tolist(),
        "door_crossings": int(((positions[:-1, :, 0] < 8.1) & (positions[1:, :, 0] >= 8.1)).sum()),
        "pair_contact_steps": int((pairs[1:] < 2 * radius - 1e-9).sum()),
        "initial_pair_contacts": int((pairs[0] < 2 * radius - 1e-9).sum()),
        "max_wall_penetration_m": max(0.0, -min(clearances)),
        "wall_penetration_samples": int((np.asarray(clearances) < -1e-9).sum()),
        "centre_inside_wall_samples": inside,
        "solid_wall_tunnelling_steps": tunnels,
        "mean_speed_m_s": float(np.linalg.norm(states[1:, :, 2:4], axis=-1).mean()),
        "max_speed_m_s": float(np.linalg.norm(states[1:, :, 2:4], axis=-1).max()),
        "max_displacement_per_step_m": float(
            np.linalg.norm(np.diff(positions, axis=0), axis=-1).max()
        ),
        "all_states_finite": True,
        "sample_positions": positions[[0, 1, 300]].tolist(),
    }


def main():
    """Write small diagnostic summaries separately from raw episode traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", choices=["main", *FLAGS], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for width in (1.2, 0.8):
        for seed in range(1001, 1006):
            for arm in ("off",) if args.pr == "main" else ("off", "on"):
                flags = {} if arm == "off" else FLAGS[args.pr]
                raw = args.output / f"{width}-{seed}-{arm}.json"
                rows.append({"arm": arm, **measure(seed, width, flags, raw)})
    summary = {"status": "diagnostic-only; not model acceptance", "pr": args.pr, "rows": rows}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pr": args.pr,
                "rows": len(rows),
                "finite": all(r["all_states_finite"] for r in rows),
                "tunnels": sum(r["solid_wall_tunnelling_steps"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
