"""Measure successor authoring on development seeds without running a release."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from loguru import logger
from shapely.geometry import Point, Polygon

from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

BASE = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
SUCCESSOR = Path("configs/scenarios/classic_interactions_francis2023_authoring_0_1_0_v1.yaml")
NAMES = ("classic_realworld_double_bottleneck_high", "classic_station_platform_medium")


def measure(matrix: Path, name: str, seed: int, tier: str | None = None) -> dict:
    """Run one fixed-robot diagnostic and return pedestrian motion summaries."""
    if seed not in range(1001, 1031):
        raise ValueError("This diagnostic only permits development seeds 1001-1030.")
    random.seed(seed)
    np.random.seed(seed)
    scenario = dict(next(s for s in load_scenarios(matrix) if s["name"] == name))
    if tier is not None:
        scenario["simulation_config"] = {**scenario["simulation_config"], "ped_speed_tier": tier}
    inputs = []
    config = build_robot_config_from_scenario(
        scenario, scenario_path=matrix, runtime_input_records=inputs
    )
    config.sim_config.pedestrian_seed = seed
    config.sim_config.route_spawn_seed = seed
    config.sim_config.desired_speed_seed = seed
    map_def = next(iter(config.map_pool.map_defs.values()))
    sim = init_simulators(config, map_def)[0]
    caps = sim.pysf_sim.peds.max_speeds.copy()
    positions = [sim.pysf_state.ped_positions.copy()]
    runtime = next(
        (
            r
            for b in sim.peds_behaviors
            for r in getattr(b, "_runtimes", [])
            if r.definition.id == "p3"
        ),
        None,
    )
    pause_steps = 0
    for _ in range(config.sim_config.max_sim_steps):
        sim.step_once([(0.0, 0.0)])
        positions.append(sim.pysf_state.ped_positions.copy())
        if runtime is not None and runtime.wait_remaining_s > 0:
            pause_steps += 1
    trace = np.asarray(positions, dtype=np.float64)
    polygons = [Polygon(o.vertices) for o in map_def.obstacles]
    inside = sum(any(p.covers(Point(xy)) for p in polygons) for frame in trace[1:] for xy in frame)
    result = {
        "scenario": name,
        "matrix": str(matrix),
        "seed": seed,
        "tier": tier or "native",
        "matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "runtime_inputs": [
            {
                "role": r["role"],
                "path": str(Path(r["path"]).relative_to(Path.cwd())),
                "sha256": r["sha256"],
            }
            for r in inputs
        ],
        "steps": len(trace) - 1,
        "pedestrians": trace.shape[1],
        "route_pedestrians": trace.shape[1] - len(map_def.single_pedestrians),
        "caps_mean_mps": float(caps.mean()),
        "caps_min_mps": float(caps.min()),
        "caps_max_mps": float(caps.max()),
        "trajectory_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
        "inside_obstacle_ped_steps": inside,
        "last_10s_stalls": int((np.linalg.norm(trace[-1] - trace[-101], axis=1) < 0.2).sum()),
    }
    if "bottleneck" in name:
        # Exclude route-end respawn jumps from opening-crossing counts.
        continuous = np.linalg.norm(np.diff(trace, axis=0), axis=2) < 1.0
        result["crossings"] = {
            str(x): int((((trace[:-1, :, 0] < x) != (trace[1:, :, 0] < x)) & continuous).sum())
            for x in (20, 40)
        }
    if runtime is not None:
        result["p3_waypoint_index"] = runtime.waypoint_index
        result["p3_pause_steps"] = pause_steps
        result["p3_final_xy"] = trace[-1, runtime.ped_id].tolist()
    return result


def main() -> None:
    """Write a small reproducible before/after artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    rows = [
        measure(matrix, name, seed)
        for seed in range(1001, 1006)
        for name in NAMES
        for matrix in (BASE, SUCCESSOR)
    ]
    rows.extend(measure(SUCCESSOR, NAMES[0], 1001, tier) for tier in ("typical",))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "classification": "diagnostic-only",
                "robot_action": [0, 0],
                "baseline_revision": "66df3de19",
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
