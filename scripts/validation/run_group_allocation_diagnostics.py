"""Development-only reset diagnostics through the benchmark map-runner environment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from loguru import logger

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios
from scripts.validation.run_empty_world_sweep import assert_dev_seeds


def main() -> None:
    """Compare legacy and exact allocation at the production map-runner reset seam."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = assert_dev_seeds(range(1001, 1031))
    logger.remove()
    matrix = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
    scenarios = [
        dict(s) for s in load_scenarios(matrix) if s["name"].startswith("classic_group_crossing_")
    ]
    rows = []
    for scenario in scenarios:
        for mode in ("legacy", "exact_small_crowd_v1"):
            item = copy.deepcopy(scenario)
            item["seeds"] = seeds
            if mode != "legacy":
                item["simulation_config"]["group_allocation_mode"] = mode
            for seed in seeds:
                env = make_robot_env(build_env_config(item, scenario_path=matrix), seed=seed)
                try:
                    env.reset(seed=seed)
                    sim = env.unwrapped.simulator
                    sizes = sorted(len(g) for g in sim.groups.groups_as_lists if g)
                    rows.append(
                        {
                            "scenario": item["name"],
                            "mode": mode,
                            "seed": seed,
                            "population": len(sim.ped_pos),
                            "sizes": sizes,
                            "grouped": sum(s for s in sizes if s > 1),
                        }
                    )
                finally:
                    env.close()
    payload = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "matrix": matrix.as_posix(),
        "matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
        "classification": "development reset diagnostics; no planner or held-out evaluation",
        "seeds": seeds,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
