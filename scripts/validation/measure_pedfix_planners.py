"""Measure global NumPy consumption and paired PEDFIX episode outcomes on dev seeds."""

import argparse
import hashlib
import json
import pickle
import subprocess
import traceback
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from robot_sf.benchmark.map_runner.map_runner import build_map_policy
from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
from robot_sf.training.scenario_loader import load_scenarios

logger.remove()
p = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
a = argparse.ArgumentParser()
a.add_argument("--mode", choices=["roster", "gate"], required=True)
args = a.parse_args()
sc = {s["name"]: s for s in load_scenarios(p)}
roster = yaml.safe_load(
    Path(
        "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
    ).read_text()
)["planners"]
names = (
    ["classic_head_on_corridor_medium", "francis2023_circular_crossing"]
    if args.mode == "roster"
    else [
        "classic_head_on_corridor_medium",
        "francis2023_circular_crossing",
        "classic_group_crossing_low",
        "classic_group_crossing_medium",
        "classic_group_crossing_high",
    ]
)
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
errors = 0
for planner in roster:
    if args.mode == "gate" and planner["key"] not in [
        "goal",
        "orca",
        "hybrid_rule_v4_fast_progress_static_escape",
    ]:
        continue
    for name in names:
        for seed in [1001, 1002]:
            assert 1001 <= seed <= 1030
            cfg = (
                yaml.safe_load(Path(planner["algo_config"]).read_text())
                if planner.get("algo_config")
                else {}
            )
            cfg.update(
                benchmark_profile=planner["benchmark_profile"],
                socnav_missing_prereq_policy="fail-fast",
            )
            calls = []
            builds = []

            def state():
                """Fingerprint the complete process-global NumPy state.

                Returns:
                    A digest including the index and cached Gaussian state.
                """
                return hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()

            def builder(*bargs, **kwargs):
                """Wrap the canonical builder without changing policy inputs.

                Returns:
                    A measured policy callable and its original metadata.
                """
                before = state()
                policy, meta = build_map_policy(*bargs, **kwargs)
                builds.append(before != state())

                def wrapped(obs):
                    before = state()
                    out = policy(obs)
                    calls.append(before != state())
                    return out

                # Preserve runtime admission/telemetry attributes on actual policy callable.
                wrapped.__dict__.update(getattr(policy, "__dict__", {}))
                return wrapped, meta

            try:
                rec = run_map_episode(
                    dict(sc[name]),
                    seed,
                    horizon=600,
                    dt=0.1,
                    record_forces=False,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo=planner["algo"],
                    scenario_path=p,
                    algo_config=cfg,
                    algo_config_path=planner.get("algo_config"),
                    policy_builder=builder,
                    adapter_impact_eval=planner.get("adapter_impact_eval", False),
                )
                row = {
                    "planner": planner["key"],
                    "name": name,
                    "seed": seed,
                    "source_sha": sha,
                    "status": "ok",
                    "calls": len(calls),
                    "changed_calls": sum(calls),
                    "first_changed_call": next((i for i, c in enumerate(calls) if c), None),
                    "build_changed": any(builds),
                    "record": rec,
                }
            except Exception as e:  # noqa: BLE001 - retain each failed matrix cell explicitly
                errors += 1
                traceback.print_exc()
                row = {
                    "planner": planner["key"],
                    "name": name,
                    "seed": seed,
                    "source_sha": sha,
                    "status": "error",
                    "error": repr(e),
                    "calls": len(calls),
                    "changed_calls": sum(calls),
                    "build_changed": any(builds),
                }
            print(json.dumps(row, default=str), flush=True)

raise SystemExit(1 if errors else 0)
