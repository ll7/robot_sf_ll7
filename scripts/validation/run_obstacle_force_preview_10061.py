"""Paired development preview of the 48-row 0.0.8 matrix, never release evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from robot_sf.benchmark.map_runner.map_runner import run_map_batch
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner.map_runner_identity import _scenario_with_episode_seed_defaults
from robot_sf.evidence.writers import write_json
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sim.spawn_validation import reset_spawn_clearance
from robot_sf.training.scenario_loader import load_scenarios

SEEDS = tuple(range(1001, 1006))
PROFILES = ("legacy_v1", "calibrated_v2")
ARMS = ("goal", "orca", "social_force")


def preview_scenario(scenario: dict, profile: str) -> dict:
    """Apply the production selector and overwrite all episode seeds explicitly.

    Returns:
        Scenario carrying exactly the preview dev seeds and requested profile.
    """
    result = deepcopy(scenario)
    result["seeds"] = list(SEEDS)
    result.pop("seed", None)
    result.setdefault("simulation_config", {})["obstacle_force_profile"] = profile
    return result


def run_cell(task: tuple) -> dict:
    """Run one arm/profile/scenario through the canonical production map runner.

    Returns:
        Coverage and output identity for five paired dev-seed episodes.
    """
    root, matrix, output, scenario, horizon, arm, profile = task
    scenario = preview_scenario(scenario, profile)
    path = output / f"{arm}__{scenario['name']}__{profile}.jsonl"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite prior preview: {path}")
    algo_config = root / "configs/algos/social_force_release_v0_0_8.yaml"
    summary = run_map_batch(
        [scenario],
        path,
        root / "robot_sf/benchmark/schemas/episode.schema.v1.json",
        scenario_path=matrix,
        horizon=horizon,
        dt=0.1,
        record_forces=True,
        algo=arm,
        algo_config_path=str(algo_config) if arm == "social_force" else None,
        benchmark_profile="baseline-safe",
        socnav_missing_prereq_policy="fail-fast",
        workers=1,
        resume=False,
    )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(SEEDS) or {int(row["seed"]) for row in rows} != set(SEEDS):
        raise RuntimeError(f"incomplete or unsafe preview coverage: {path}")
    if summary.get("failed_jobs", 0):
        raise RuntimeError(f"failed preview jobs: {summary}")
    for row in rows:
        sites = row["algorithm_metadata"]["obstacle_force_law"]["sites"]
        expected = (10.0, -0.57) if profile == "legacy_v1" else (0.003, 0.375)
        parameters = sites["fast_pysf"]["parameters"]
        if (parameters["factor"], parameters["threshold"]) != expected:
            raise RuntimeError(f"profile did not take effect: {parameters}")
        if row["algorithm_metadata"].get("status") in {"fallback", "degraded"}:
            raise RuntimeError("fallback/degraded row is not preview evidence")
    return {
        "arm": arm,
        "profile": profile,
        "scenario": scenario["name"],
        "seeds": list(SEEDS),
        "written": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def reset_pair(task: tuple) -> dict:
    """Compare both profiles' physical reset state without stepping any environment.

    Returns:
        Per-profile clearances and exact equality of pedestrian/robot start states.
    """
    matrix, scenario, seed = task
    states = []
    clearances = []
    for profile in PROFILES:
        seeded = _scenario_with_episode_seed_defaults(
            preview_scenario(scenario, profile), seed=seed
        )
        config = build_env_config(seeded, scenario_path=matrix)
        env = make_robot_env(config=config, seed=seed, debug=False)
        try:
            env.reset(seed=seed)
            sim = env.simulator
            states.append((np.asarray(sim.pysf_sim.peds.state).copy(), deepcopy(sim.robot_poses)))
            clearances.append(reset_spawn_clearance(sim))
        finally:
            env.close()
    same = np.array_equal(states[0][0], states[1][0]) and states[0][1] == states[1][1]
    new_overlap = not clearances[0]["overlap"] and clearances[1]["overlap"]
    return {
        "scenario": scenario["name"],
        "seed": seed,
        "identical_reset_state": same,
        "new_overlap": new_overlap,
        "clearances": dict(zip(PROFILES, clearances, strict=True)),
    }


def main() -> None:
    """Run a bounded, fail-closed preview or reset audit and preserve its manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("episodes", "reset"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must be in 1..16")
    root = Path(__file__).resolve().parents[2]
    matrix = root / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    horizons_path = (
        root / "configs/benchmarks/obstacle_force_calibration_0_0_9/preview_horizons.yaml"
    )
    horizons = yaml.safe_load(horizons_path.read_text())["scenarios"]
    scenarios = [dict(row) for row in load_scenarios(matrix)]
    if len(scenarios) != 48 or len({row["name"] for row in scenarios}) != 48:
        raise RuntimeError("preview must resolve exactly 48 unique scenarios")
    if set(horizons) != {row["name"] for row in scenarios}:
        raise RuntimeError("authored horizon schedule does not cover the 48-row matrix exactly")
    print(f"RESOLVED SEEDS {list(SEEDS)}; profiles={PROFILES}; scenarios=48", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "episodes":
        tasks = [
            (
                root,
                matrix,
                args.output,
                scenario,
                int(horizons[scenario["name"]]["recommended_horizon_steps"]),
                arm,
                profile,
            )
            for profile in PROFILES
            for arm in ARMS
            for scenario in scenarios
        ]
        worker = run_cell
    else:
        tasks = [(matrix, scenario, seed) for scenario in scenarios for seed in SEEDS]
        worker = reset_pair
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            write_json(args.output / f"progress_{args.mode}.json", {"rows": rows})
            print(f"completed {len(rows)}/{len(tasks)}", flush=True)
    write_json(
        args.output / f"manifest_{args.mode}.json",
        {
            "evidence_status": "diagnostic-only",
            "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "matrix_sha256": hashlib.sha256(matrix.read_bytes()).hexdigest(),
            "horizons_sha256": hashlib.sha256(horizons_path.read_bytes()).hexdigest(),
            "seeds": list(SEEDS),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
