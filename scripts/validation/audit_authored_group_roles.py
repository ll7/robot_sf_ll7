#!/usr/bin/env python3
"""Dev-only runtime group inventory and stationary-robot join/leave diagnostic (#10028)."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from loguru import logger

from robot_sf.sim.simulator import init_simulators
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios


def group_stats(sim) -> dict:
    """Count live members, excluding empty retired group containers."""
    sizes = [len(group) for group in sim.groups.groups.values() if group]
    population = sum(sizes)
    grouped = sum(size for size in sizes if size > 1)
    return {
        "population": population,
        "size_histogram": dict(sorted(Counter(sizes).items())),
        "multi_member_groups": sum(size > 1 for size in sizes),
        "grouped_pedestrians": grouped,
        "grouped_fraction": grouped / population if population else 0.0,
    }


def main() -> None:
    """Audit every scenario at reset and the two group transitions over 40 seconds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(seed < 1001 or seed > 1030 for seed in args.seeds):
        parser.error("only development seeds 1001-1030 are allowed")
    logger.remove()
    scenarios = load_scenarios(args.matrix)
    rows = []
    for scenario in scenarios:
        for seed in args.seeds:
            config = build_robot_config_from_scenario(scenario, scenario_path=args.matrix)
            config.sim_config.pedestrian_seed = seed
            sim = init_simulators(
                config, next(iter(config.map_pool.map_defs.values())), random_start_pos=False
            )[0]
            row = {
                "scenario": scenario["name"],
                "seed": seed,
                "declared_crowd_groups": scenario.get("simulation_config", {}).get("groups"),
                "authored_group_labels": sorted(
                    {
                        ped.initial_group_id
                        for ped in sim.map_def.single_pedestrians
                        if getattr(ped, "initial_group_id", None) is not None
                    }
                ),
                "initial": group_stats(sim),
            }
            if scenario["name"] in {"francis2023_join_group", "francis2023_leave_group"}:
                completed_at = None
                for step in range(400):
                    sim.step_once([(0.0, 0.0)])
                    if (
                        scenario["name"] == "francis2023_join_group"
                        and group_stats(sim)["grouped_pedestrians"] == 3
                        and completed_at is None
                    ):
                        completed_at = (step + 1) * config.sim_config.time_per_step_in_secs
                row["after_40s"] = group_stats(sim)
                row["join_completed_at_s"] = completed_at
            rows.append(row)
    result = {
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "matrix": args.matrix.as_posix(),
        "seeds": args.seeds,
        "scenario_count": len(scenarios),
        "scenario_count_with_runtime_groups": len(
            {row["scenario"] for row in rows if row["initial"]["multi_member_groups"] > 0}
        ),
        "rows": rows,
        "scope": "dev diagnostic; stationary robot, not planner-ranking or release evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
