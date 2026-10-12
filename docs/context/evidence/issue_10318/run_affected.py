"""Reproduce the six #10318 cells through the canonical runner on development seeds."""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from loguru import logger

from robot_sf.benchmark.classic_interactions_loader import load_classic_matrix
from robot_sf.benchmark.map_runner.map_runner import _run_map_episode

MATRIX = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
IDS = {
    "francis2023_" + name
    for name in (
        "following_human",
        "entering_elevator",
        "exiting_elevator",
        "narrow_hallway",
        "robot_crowding",
        "join_group",
    )
}


def run(task):
    """Run one native-predictor, declared-adapter episode with complete step traces."""
    import torch

    torch.set_num_threads(1)
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    scenario, seed = task
    return _run_map_episode(
        scenario,
        seed,
        horizon=400,
        dt=0.1,
        record_forces=False,
        snqi_weights=None,
        snqi_baseline=None,
        algo="prediction_planner",
        algo_config_path="configs/algos/prediction_planner_release_v0_0_8.yaml",
        scenario_path=MATRIX,
        record_simulation_step_trace=True,
    )


def main():
    """Validate development seeds before starting any worker or episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not all(1001 <= seed <= 1200 for seed in args.seeds):
        parser.error("only development seeds 1001–1200 are allowed")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    scenes = [
        s for s in load_classic_matrix(str(MATRIX)) if s.get("name") in IDS or s.get("id") in IDS
    ]
    if len(scenes) != 6:
        raise ValueError("The released matrix must resolve all six affected cells")
    tasks = [(scenario, seed) for scenario in scenes for seed in args.seeds]
    with args.out.open("w") as output, ProcessPoolExecutor(max_workers=6) as pool:
        for row in pool.map(run, tasks):
            output.write(json.dumps(row, default=str) + "\n")
            output.flush()
            print(
                row["scenario_id"], row["seed"], row["steps"], row["metrics"]["success"], flush=True
            )


if __name__ == "__main__":
    main()
