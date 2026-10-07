"""Diagnostic reflection gate: five hybrid switch arms, dev1001, 60 plant steps."""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
from loguru import logger

from scripts.validation.run_hybrid_default_comparison import PER_SWITCH_ARMS
from scripts.validation.run_hybrid_feasibility_diagnostics import ARM_SWITCHES, assert_dev_seeds
from tests.metamorphic import planner_arms as harness


def run_probe():
    """Run all fifteen real episodes and verify their typed switch values.

    Returns:
        Source-bound complete trajectories and both reflection errors for each arm.
    """
    seed = assert_dev_seeds([1001])[0]
    algo, raw = harness.resolve_release_algo_config(
        "hybrid_rule_local_planner", harness.HYBRID_V4_DIAGNOSTIC_CONFIG, "metamorphic"
    )
    original_env, original_policy = harness.robot_env_config, harness.build_map_policy
    rows, traces = [], {}
    checks = 0
    for arm in PER_SWITCH_ARMS:
        bits = ARM_SWITCHES[arm]
        mapping = dict(raw)
        mapping.update(
            physical_static_exclusion_enabled=bits[0], goal_next_validity_enabled=bits[1]
        )
        if arm == "current_defaults":
            mapping.pop("physical_static_exclusion_enabled")
            mapping.pop("goal_next_validity_enabled")

        def config(*args, **kwargs):
            result = original_env(*args, **kwargs)
            result.include_goal_next_valid = bits[2]
            return result

        def policy(*args, **kwargs):
            nonlocal checks
            fn, meta = original_policy(*args, **kwargs)
            cfg = fn._planner_adapter.config
            assert (cfg.physical_static_exclusion_enabled, cfg.goal_next_validity_enabled) == bits[
                :2
            ]
            checks += 1
            return fn, meta

        episodes = {}
        with (
            patch.object(harness, "release_arm", lambda _: (algo, dict(mapping))),
            patch.object(harness, "robot_env_config", config),
            patch.object(harness, "build_map_policy", policy),
        ):
            for name, transform in (
                ("identity", harness.identity),
                ("mirror_y", harness.mirror_y),
                ("mirror_x", harness.mirror_x),
            ):
                episode = harness.run_arm_episode(
                    "reflection-control",
                    harness.interaction_scene(transform),
                    seed=seed,
                    max_steps=60,
                )
                episodes[name] = episode
                traces[f"{arm}/{name}"] = dataclasses.asdict(episode)
        for name, transform in (("mirror_y", harness.mirror_y), ("mirror_x", harness.mirror_x)):
            expected = np.asarray([transform(p[:2]) for p in episodes["identity"].poses])
            actual = np.asarray(episodes[name].poses)[:, :2]
            error = float(np.max(np.abs(actual - expected)))
            rows.append(
                {
                    "arm": arm,
                    "switches": bits,
                    "seed": seed,
                    "transform": name,
                    "steps": 60,
                    "position_tolerance_m": 0.0001,
                    "max_position_error_m": error,
                    "position_relation_passed": error <= 0.0001,
                    "same_outcome": (
                        episodes["identity"].success,
                        episodes["identity"].collision,
                        episodes["identity"].step_limit_reached,
                    )
                    == (
                        episodes[name].success,
                        episodes[name].collision,
                        episodes[name].step_limit_reached,
                    ),
                }
            )
    sources = (
        Path(__file__),
        harness.ROOT / "tests/metamorphic/planner_arms.py",
        harness.ROOT / "robot_sf/planner/hybrid_rule_local_planner.py",
        harness.ROOT / harness.HYBRID_V4_DIAGNOSTIC_CONFIG,
    )
    return {
        "evidence_status": "diagnostic-only",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": seed,
        "workers": 1,
        "episodes": 15,
        "max_steps": 60,
        "native_config_switch_checks": checks,
        "sha256": {
            str(p.relative_to(harness.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "rows": rows,
        "traces": traces,
    }


def main():
    """Write complete diagnostic evidence and reject any failed reflection relation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    result = run_probe()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(result, allow_nan=False).encode(), mtime=0))
    print(json.dumps(result["rows"], indent=2))
    if not all(r["position_relation_passed"] and r["same_outcome"] for r in result["rows"]):
        raise SystemExit("Hybrid reflection relation exceeds 0.1 mm or changes the outcome")


if __name__ == "__main__":
    main()
