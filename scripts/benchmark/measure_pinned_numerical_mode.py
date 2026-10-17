#!/usr/bin/env python3
"""Measure inference on retained scenario reset inputs in a fresh numerical context.

Run capture first, then default and pinned in separate interpreters. Both timings
use identical adapted observations. This measures learned inference, excluding
simulation, guard decisions, checkpoint loading and observation adaptation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from robot_sf._numerical_mode import bootstrap_numerical_mode
from robot_sf._numerical_thread_env import pin_thread_env_for_determinism

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("capture", "default", "pinned"), required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
pin_thread_env_for_determinism()
if args.mode == "pinned":
    bootstrap_numerical_mode("pinned_float64_v1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from robot_sf._numerical_mode import effective_numerical_mode, initialize_pinned_torch  # noqa: E402
from robot_sf.baselines.ppo import PPOPlanner  # noqa: E402
from robot_sf.benchmark.camera_ready._config import (  # noqa: E402
    _load_campaign_scenarios,
    load_campaign_config,
)
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config  # noqa: E402
from robot_sf.gym_env.environment_factory import make_robot_env  # noqa: E402

initialize_pinned_torch()
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0.yaml"
SCENARIOS = (
    "classic_cross_trap_medium",
    "francis2023_narrow_doorway",
    "francis2023_circular_crossing",
)
ARMS = {
    "ppo": "configs/baselines/ppo_release_robot_0_1_0_cpu.yaml",
    "ppo_expert": "configs/baselines/ppo_expert_0_1_0_cpu.yaml",
    "guarded_ppo": "configs/algos/guarded_ppo_release_v0_1_0.yaml",
}


def planner_for(path):
    """Load the same checkpoint and adapter with the selected numerical policy."""
    from dataclasses import fields

    from robot_sf.baselines.ppo import PPOPlannerConfig

    payload = yaml.safe_load((ROOT / path).read_text())
    if args.mode != "pinned":
        payload.pop("numerical_mode", None)
    allowed = {f.name for f in fields(PPOPlannerConfig)}
    return PPOPlanner({k: v for k, v in payload.items() if k in allowed})


if args.mode == "capture":
    cfg = load_campaign_config(CONFIG)
    scenarios = {s["name"]: s for s in _load_campaign_scenarios(cfg)}
    planners = {arm: planner_for(path) for arm, path in ARMS.items()}
    data = {}
    for name in SCENARIOS:
        for seed in range(1001, 1004):
            scenario = dict(scenarios[name])
            scenario["map_file"] = str((ROOT / scenario["map_file"]).resolve())
            config = build_env_config(scenario, scenario_path=cfg.scenario_matrix_path)
            env = make_robot_env(config=config, seed=seed)
            obs, _ = env.reset(seed=seed)
            for arm, planner in planners.items():
                planner.bind_env(env)
                adapted = planner._build_model_obs_dict(obs)
                for key, value in adapted.items():
                    data[f"{arm}/{name}/{seed}/{key}"] = value
            env.close()
    np.savez_compressed(args.inputs, **data)
    args.output.write_text(
        json.dumps(
            {"coordinates": 9, "scenario_ids": SCENARIOS, "seeds": [1001, 1002, 1003]}, indent=2
        )
    )
else:
    inputs = np.load(args.inputs)
    rows = []
    for arm, path in ARMS.items():
        planner = planner_for(path)
        reference = None
        if args.mode == "pinned":
            import copy

            reference = copy.deepcopy(planner._model.policy).double().eval()
        for name in SCENARIOS:
            for seed in range(1001, 1004):
                prefix = f"{arm}/{name}/{seed}/"
                obs = {k[len(prefix) :]: inputs[k] for k in inputs.files if k.startswith(prefix)}
                input_sha256 = hashlib.sha256(
                    b"".join(key.encode() + obs[key].tobytes() for key in sorted(obs))
                ).hexdigest()
                action = planner._predict_action(obs)
                if action is None or not np.isfinite(action).all():
                    raise RuntimeError("Inference probe produced an unavailable/nonfinite action")
                error = None
                if reference is not None:
                    tensors = {
                        k: torch.as_tensor(v, dtype=torch.float64).unsqueeze(0)
                        for k, v in obs.items()
                    }
                    with torch.no_grad():
                        features = reference.pi_features_extractor(tensors)
                        expected = reference.action_net(
                            reference.mlp_extractor.forward_actor(features)
                        ).numpy()
                    error = float(np.max(np.abs(expected - planner._pinned_actor.mean(obs))))
                    if error > 2e-12:
                        raise RuntimeError(f"Float64 reference disagreement: {error}")
                for _ in range(20):
                    planner._predict_action(obs)
                blocks = []
                for _ in range(5):
                    started = perf_counter()
                    for _ in range(100):
                        planner._predict_action(obs)
                    blocks.append((perf_counter() - started) * 10)
                rows.append(
                    {
                        "planner": arm,
                        "scenario": name,
                        "seed": seed,
                        "ms_per_call": float(np.mean(blocks)),
                        "blocks_ms": blocks,
                        "float64_reference_max_abs_error": error,
                        "input_sha256": input_sha256,
                    }
                )
        planner.close()
    args.output.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "kernel_context": effective_numerical_mode(),
                "warmup_calls": 20,
                "blocks": 5,
                "calls_per_block": 100,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
