"""Run actual PPO learning for one rollout on CPU without replacing the full recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

# Set before importing Torch/SB3, including on GPU-equipped development machines.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from loguru import logger

from robot_sf.training.scenario_loader import load_scenarios
from scripts.training.train_ppo import (
    _init_training_model,
    _prepare_seed_state,
    load_expert_training_config,
)


def main() -> int:
    """Learn 2048 CPU transitions using the leaf's network, reward and scenario sampler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOT_SF_ARTIFACT_ROOT"] = str(args.output.resolve())
    logger.remove()
    logger.add(args.output / "training.log", level="INFO")
    torch.set_num_threads(1)
    source = load_expert_training_config(args.config.resolve())
    config = replace(
        source,
        num_envs=1,
        worker_mode="dummy",
        env_overrides=dict(source.env_overrides, predictive_foresight_device="cpu"),
    )
    _prepare_seed_state(config)
    scenarios = load_scenarios(config.scenario_config)
    model, env, _, _, _ = _init_training_model(
        config,
        scenario=None,
        scenario_definitions=scenarios,
        exclude_scenarios=(),
        run_id=f"{config.policy_id}-cpu-smoke",
        tensorboard_log=args.output / "tensorboard",
        resume_from=None,
    )
    try:
        model.learn(total_timesteps=2048)
        assert model.num_timesteps == 2048
        model.save(str(args.output / "smoke_model.zip"))
        receipt = {
            "status": "startup-proof-only",
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "runtime_sha256": {
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in (
                    "robot_sf/gym_env/robot_env.py",
                    "robot_sf/robot/action_adapters.py",
                    "robot_sf/baselines/ppo.py",
                    "scripts/training/train_ppo.py",
                )
            },
            "policy_id": config.policy_id,
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "training_seeds": list(config.seeds),
            "steps": model.num_timesteps,
            "device": str(model.device),
            "source_config": asdict(source),
            "smoke_overrides": {
                "num_envs": 1,
                "worker_mode": "dummy",
                "predictive_foresight_device": "cpu",
                "steps": 2048,
            },
            "action_space": {
                "low": env.action_space.low.tolist(),
                "high": env.action_space.high.tolist(),
            },
        }
        (args.output / "receipt.json").write_text(json.dumps(receipt, default=str, indent=2) + "\n")
        print(f"{config.policy_id}: learned {model.num_timesteps} CPU transitions")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
