#!/usr/bin/env python3
"""Run the non-reactive response multiplier sweep for issue #4850.

Runs simulations for multiplier values 0.0, 0.1, and 0.3, writes episode records,
and outputs a complete JSONL dataset for each value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.map_runner import _build_policy
from robot_sf.benchmark.map_runner_episode import run_map_episode

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--multiplier",
        type=float,
        required=True,
        choices=[0.0, 0.1, 0.3],
        help="Non-reactive response multiplier value (issue #4850).",
    )
    parser.add_argument(
        "--output-dir",
        default="output/issue_4850_multiplier_sweep",
        help="Base output directory for multiplier sweep results.",
    )
    return parser.parse_args()


def main() -> int:
    """Run all scenarios for the given multiplier value."""
    args = parse_args()
    multiplier = args.multiplier
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"multiplier_{multiplier}_episode_records.jsonl"

    # Define the scenario parameters (smoke-scale: 2 planners x 2 seeds)
    planners = ["goal", "social_force"]
    seeds = [101, 102]
    scenario_id = "issue_4850_classic_crossing_multiplier_sweep"

    # Map file
    map_path = REPO_ROOT / "maps/svg_maps/classic_crossing.svg"
    if not map_path.exists():
        print(f"Error: Map file not found at {map_path}")
        return 1

    records: list[dict[str, Any]] = []
    run_idx = 0
    total_runs = len(planners) * len(seeds)

    for planner in planners:
        for seed in seeds:
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] Running multiplier={multiplier} planner={planner} seed={seed}"
            )

            # Build the scenario dict with the non_reactive_response_multiplier
            scenario_dict: dict[str, Any] = {
                "name": scenario_id,
                "map_file": str(map_path.resolve()),
                "simulation_config": {
                    "max_episode_steps": 600,
                    "ped_density": 0.02,
                    "population_size": 12,
                    # Issue #4850: non-reactive response multiplier sweep
                    "non_reactive_response_multiplier": multiplier,
                    # Mean-matched heterogeneity (from issue #3574)
                    "response_law_composition": {
                        "standard": 0.75,  # 75% standard reactive pedestrians
                        "non_yielding": 0.25,  # 25% non-yielding (affected by multiplier)
                    },
                    "response_law_seed": 4850,
                },
                "robot_config": {},
            }

            try:
                # Run the episode
                rec = run_map_episode(
                    scenario=scenario_dict,
                    seed=seed,
                    horizon=600,
                    dt=0.1,
                    record_forces=True,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo=planner,
                    scenario_path=Path(__file__),  # dummy path
                    record_planner_decision_trace=False,
                    record_simulation_step_trace=True,
                    policy_builder=_build_policy,
                )
                # Annotate with run metadata for ranking/analysis scripts
                rec["population_arm"] = f"multiplier_{multiplier}"  # Use population_arm for compatibility with rank-sensitivity
                rec["non_reactive_response_multiplier"] = multiplier
                rec["planner"] = planner
                rec["seed"] = seed
                rec["scenario_id"] = scenario_id

                # Make sure scenario params matches too
                if "scenario_params" not in rec:
                    rec["scenario_params"] = {}
                rec["scenario_params"]["population_arm"] = f"multiplier_{multiplier}"
                rec["scenario_params"]["non_reactive_response_multiplier"] = multiplier
                rec["scenario_params"]["planner"] = planner
                rec["scenario_params"]["seed"] = seed
                rec["scenario_params"]["scenario_id"] = scenario_id

                records.append(rec)
            except (RuntimeError, ValueError, KeyError, TypeError, OSError) as e:
                print(f"Exception during simulation (multiplier={multiplier}, planner={planner}, seed={seed}): {e}")
                raise

    # Write out JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully wrote {len(records)} episode records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
