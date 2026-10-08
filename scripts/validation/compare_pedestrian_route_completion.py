"""Paired, diagnostic-only live-crowd comparison of pedestrian completion rules.

Run from the repository root with numerical threads set to one. The old arm
disables only ordered pedestrian completion; all other runtime code is identical.
Robots remain stationary, and native pedestrian physics runs for the full
authored horizon, independent of robot success/collision termination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.ped_npc.ped_behavior import FollowRouteBehavior
from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
SCENARIOS = (
    "francis2023_circular_crossing",
    "francis2023_parallel_traffic",
    "classic_head_on_corridor_medium",
)


def _frame(sim) -> bytes:
    """Canonical pedestrian row order, little-endian float64 x/y/vx/vy."""
    state = np.asarray(sim.pysf_state.pysf_states()[:, :4], dtype="<f8")
    if not np.isfinite(state).all():
        raise ValueError("Non-finite pedestrian trajectory")
    return np.asarray(state.shape, dtype="<i8").tobytes() + state.tobytes(order="C")


def _arm(scenario_name: str, seed: int, ordered: bool) -> dict:
    logger.remove()
    random.seed(seed)
    np.random.seed(seed)
    scenario = next(s for s in load_scenarios(MATRIX) if s["name"] == scenario_name)
    scenario["seeds"] = [seed]
    config = build_robot_config_from_scenario(scenario, scenario_path=MATRIX)
    config.sim_config.pedestrian_seed = seed
    definition = next(iter(config.map_pool.map_defs.values()))
    sim = init_simulators(config, definition)[0]
    counts = {"group_respawns": 0, "pedestrian_respawns": 0, "premature_group_respawns": 0}
    route_groups = 0
    for behavior in sim.peds_behaviors:
        if not isinstance(behavior, FollowRouteBehavior):
            continue
        route_groups += len(behavior.navigators)
        for nav in behavior.navigators.values():
            if not hasattr(nav, "require_final_waypoint"):
                raise RuntimeError("Requires the ordered-completion implementation from #10220")
            nav.require_final_waypoint = ordered
        original = behavior.respawn_group_at_start

        def count_respawn(gid, *, guard_robot=True, original=original, behavior=behavior):
            counts["group_respawns"] += 1
            counts["pedestrian_respawns"] += len(behavior.groups.groups[gid])
            nav = behavior.navigators[gid]
            counts["premature_group_respawns"] += nav.waypoint_id < len(nav.waypoints) - 1
            return original(gid, guard_robot=guard_robot)

        behavior.respawn_group_at_start = count_respawn
    initial = _frame(sim)
    digest = hashlib.sha256(initial)
    steps = round(config.sim_config.sim_time_in_secs / config.sim_config.time_per_step_in_secs)
    population = len(sim.pysf_state.pysf_states())
    if population == 0 or route_groups == 0:
        raise ValueError("Comparison requires live route-following pedestrians")
    for _ in range(steps):
        sim.step_once([(0.0, 0.0)] * len(sim.robots))
        digest.update(_frame(sim))
    return {
        **counts,
        "population": population,
        "route_groups": route_groups,
        "steps": steps,
        "dt_s": config.sim_config.time_per_step_in_secs,
        "require_final_waypoint": ordered,
        "initial_sha256": hashlib.sha256(initial).hexdigest(),
        "trajectory_sha256": digest.hexdigest(),
    }


def compare_pair(task: tuple[str, int]) -> dict:
    """Run the same seed twice, changing only the completion predicate."""
    scenario, seed = task
    old = _arm(scenario, seed, False)
    new = _arm(scenario, seed, True)
    if old["initial_sha256"] != new["initial_sha256"]:
        raise ValueError("Paired initial populations differ")
    return {
        "scenario": scenario,
        "seed": seed,
        "old": old,
        "new": new,
        "trajectory_equal": old["trajectory_sha256"] == new["trajectory_sha256"],
        "respawns_equal": old["group_respawns"] == new["group_respawns"],
    }


def main() -> None:
    """Write compact per-pair evidence, refusing unbounded workers or seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1001, 1011)))
    args = parser.parse_args()
    if any(seed not in range(1001, 1031) for seed in args.seeds):
        parser.error("Only development seeds 1001–1030 are allowed")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            parser.error(f"Set {name}=1")
    tasks = [(name, seed) for name in SCENARIOS for seed in args.seeds]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        pairs = list(pool.map(compare_pair, tasks))
    result = {
        "schema": "pedestrian_route_completion_comparison.v1",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "matrix": MATRIX.as_posix(),
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "workers": args.workers,
        "numerical_threads": 1,
        "robot_action": [0.0, 0.0],
        "execution": "native full authored horizon; diagnostic only",
        "digest_format": "initial and post-step frames: little-endian int64 shape, float64 x/y/vx/vy",
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
