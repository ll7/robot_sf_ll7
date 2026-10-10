"""Check the literal goal-only contract on every development cell before stepping.

Run from the repository root. This records configuration errors separately from
native success, collision and timeout outcomes; it never advances an invalid env.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))


def probe(task):
    """Reset one development cell and require the real policy to reject missing validity.

    Returns:
        The verified pre-step error and effective switch values.
    """
    from loguru import logger

    from robot_sf.common.hybrid_defaults import active_default_policy, defaults_for_source
    from scripts.validation import run_hybrid_feasibility_diagnostics as native

    name, seed = task
    native.assert_dev_seeds([seed])
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    with defaults_for_source(None):
        scenario, matrix = native.load_cells([name])[name]
        scenario = native._scenario_with_episode_seed_defaults(
            dict(scenario, seeds=[seed]), seed=seed
        )
        cfg = native._build_env_config(scenario, scenario_path=matrix)
        cfg.include_goal_next_valid = False
        mapping = native.hybrid_config(scenario, enabled=False, goal_validity=True)
        policy, _ = native._build_policy(
            "hybrid_rule_local_planner", mapping, robot_kinematics="differential_drive"
        )
        planner = policy._planner_adapter
        flags = {
            "physical_static_exclusion_enabled": planner.config.physical_static_exclusion_enabled,
            "goal_next_validity_enabled": planner.config.goal_next_validity_enabled,
            "include_goal_next_valid": cfg.include_goal_next_valid,
        }
        assert tuple(flags.values()) == (False, True, False)
        env = native.make_robot_env(config=cfg, seed=seed, debug=False)
        try:
            obs, _ = env.reset(seed=seed)
            policy._planner_bind_env(env)
            policy._planner_reset(seed=seed)
            assert "next_valid" not in planner._socnav_fields(obs)[1]
            try:
                policy(obs)
            except ValueError as error:
                message = str(error)
                assert "goal_next_validity_enabled requires observation next_valid" in message
            else:
                raise AssertionError("Literal goal-only unexpectedly accepted a missing sensor")
            return {
                "scenario": name,
                "seed": seed,
                "arm": "goal_only_invalid",
                "classification": "invalid_observation_contract",
                "environment_steps": 0,
                "effective_switches": flags,
                "next_valid_field_observed": False,
                "default_policy": active_default_policy(),
                "error": message,
            }
        finally:
            env.close()


def main():
    """Publish complete first-call contract accounting for the standard development matrix."""
    from scripts.validation.run_hybrid_feasibility_diagnostics import MAIN_MATRIX, load_scenarios

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "diagnostic-only",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenarios": [row["name"] for row in load_scenarios(MAIN_MATRIX)],
        "seeds": list(range(1001, 1031)),
        "workers": args.workers,
        "scope": "native reset and first policy call; no environment step",
    }
    assert len(manifest["scenarios"]) == 48
    tasks = [(name, seed) for name in manifest["scenarios"] for seed in manifest["seeds"]]
    records = []
    pending = iter(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(probe, next(pending)) for _ in range(args.workers)}
        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                futures.remove(future)
                records.append(future.result())
                if len(records) % 30 == 0:
                    print(f"verified {len(records)}/{len(tasks)} configuration errors", flush=True)
                task = next(pending, None)
                if task is not None:
                    futures.add(executor.submit(probe, task))
    records.sort(key=lambda row: (row["scenario"], row["seed"]))
    assert len(records) == 1440
    result = {
        "manifest": manifest,
        "complete": True,
        "cells": len(records),
        "configuration_errors": len(records),
        "environment_steps": 0,
        "records": records,
    }
    encoded = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    (args.output / "literal_goal_only.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    (args.output / "literal_goal_only_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("complete: 1440/1440 configuration errors; zero environment steps", flush=True)


if __name__ == "__main__":
    main()
