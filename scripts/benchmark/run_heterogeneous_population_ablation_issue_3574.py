#!/usr/bin/env python3
"""Run the issue #3574 mean-matched heterogeneity ablation campaign.

Runs simulations for each row of the dry-run manifest, writes episode records,
and outputs a complete JSONL dataset.
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
        "--manifest",
        default="output/issue_3574_mean_matched_harness/manifest.json",
        help="Path to the pre-run manifest JSON.",
    )
    parser.add_argument(
        "--output",
        default="output/issue_3574_mean_matched_harness/episode_records.jsonl",
        help="Path to write the resulting episode records JSONL.",
    )
    return parser.parse_args()


def main() -> int:
    """Run all ablation scenarios."""
    args = parse_args()
    manifest_path = REPO_ROOT / args.manifest
    output_path = REPO_ROOT / args.output

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Build it first.")
        return 1

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_rows = manifest.get("manifest_rows", [])
    print(f"Loaded {len(manifest_rows)} manifest rows from {manifest_path}")

    # Build scenarios dynamically matching the manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    for idx, row in enumerate(manifest_rows):
        scenario_id = row["scenario_id"]
        planner = row["planner"]
        seed = int(row["seed"])
        arm = row["population_arm"]
        density = float(row["density"])
        response_law_fraction_value = row.get("response_law_fraction")
        response_law_fraction = float(
            0.0 if response_law_fraction_value is None else response_law_fraction_value
        )

        print(
            f"[{idx + 1}/{len(manifest_rows)}] Running scenario={scenario_id} arm={arm} planner={planner} seed={seed}"
        )

        # Assemble the runtime scenario dict
        # Map file is relative to repo root
        map_path = REPO_ROOT / "maps/svg_maps/classic_crossing.svg"
        if not map_path.exists():
            print(f"Error: Map file not found at {map_path}")
            return 1

        pop_size = sum(row["arm_population"]["counts"].values())

        scenario_dict: dict[str, Any] = {
            "name": scenario_id,
            "map_file": str(map_path.resolve()),
            "simulation_config": {
                "max_episode_steps": 600,
                "ped_density": density,
                "population_size": pop_size,
                "response_law_composition": row["arm_population"].get("response_law_composition"),
                "response_law_seed": row["arm_population"].get("response_law_seed"),
                "archetype_composition": row["arm_population"].get("composition"),
                # We can determine the speed factors from the labels or composition
                "archetype_speed_factors": {
                    "cautious": 0.7,
                    "standard": 1.0,
                    "hurried": 1.4,
                    "mean_matched_homogeneous": 1.025,  # Weighted mean speed factor
                },
                "archetype_seed": 3574,
            },
            "pedestrian_control_trace_labels": row["arm_population"][
                "pedestrian_control_trace_labels"
            ],
            "robot_config": {},
        }

        try:
            # Run the episode
            # We record simulation trace to retrieve pedestrian positions, speeds, etc.
            rec = run_map_episode(
                scenario=scenario_dict,
                seed=seed,
                horizon=600,
                dt=0.1,
                record_forces=True,
                snqi_weights=None,
                snqi_baseline=None,
                algo=planner,
                scenario_path=manifest_path,
                record_planner_decision_trace=False,
                record_simulation_step_trace=True,
                policy_builder=_build_policy,
            )
            # Annotate with run metadata for ranking/analysis scripts
            rec["population_arm"] = arm
            rec["planner"] = planner
            rec["seed"] = seed
            rec["scenario_id"] = scenario_id
            rec["response_law_fraction"] = response_law_fraction

            # Make sure scenario params matches too
            if "scenario_params" not in rec:
                rec["scenario_params"] = {}
            rec["scenario_params"]["population_arm"] = arm
            rec["scenario_params"]["planner"] = planner
            rec["scenario_params"]["seed"] = seed
            rec["scenario_params"]["scenario_id"] = scenario_id
            rec["scenario_params"]["response_law_fraction"] = response_law_fraction

            records.append(rec)
        except (RuntimeError, ValueError, KeyError, TypeError, OSError) as e:
            # Narrow the handler (issue #4618 CI-B) to the runtime/config failures a
            # simulation run can plausibly raise; annotate with the failing arm/planner/seed
            # before re-raising so the traceback identifies the offending row.
            print(f"Exception during simulation (arm={arm}, planner={planner}, seed={seed}): {e}")
            raise

    # Write out JSONL
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully wrote {len(records)} episode records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
