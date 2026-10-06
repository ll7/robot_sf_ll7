"""Diagnostic-only real-checkpoint delta probe and dev-seed release episode slice.

Run from the repository root through run_worktree_shared_venv.sh --profile training.
Use --mode episodes for two scenarios x seeds 1001/1002 x H200 per arm.
The probe records actual commands; the regression test owns the independent reference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger

from robot_sf.baselines.ppo import PPOPlanner
from robot_sf.benchmark.map_runner import map_runner
from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.benchmark.map_runner_policies.map_runner_actions import policy_command_to_env_action
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.models import get_registry_entry, resolve_model_path, sha256_of_file
from robot_sf.training.scenario_loader import load_scenarios

CONFIGS = {
    "ppo": "configs/baselines/ppo_issue_791_eval_aligned_large_capacity_cpu.yaml",
    "guarded_ppo": "configs/algos/guarded_ppo_camera_ready_cpu.yaml",
}
SCENARIOS = [
    (
        "configs/scenarios/archetypes/issue_596_frame_consistency.yaml",
        "empty_map_8_directions_east",
    ),
    (
        "configs/scenarios/archetypes/classic_head_on_corridor.yaml",
        "classic_head_on_corridor_medium",
    ),
]


def release_config(arm: str) -> dict[str, Any]:
    """Read a shipped CPU release config without replacing the robot contract."""
    return yaml.safe_load(Path(CONFIGS[arm]).read_text())


def scenario_input(path: str, name: str) -> dict[str, Any]:
    """Select an existing scenario; seeds are supplied explicitly at execution time."""
    return next(row for row in load_scenarios(path) if row["name"] == name)


def real_checkpoint_probe(arm: str, initial_speed: tuple[float, float]) -> dict[str, Any]:
    """Freeze real model inputs while applying five steps through the release drive.

    Guarded PPO uses its actual guard; its proposal and decision are recorded at each step.
    """
    torch.set_num_threads(1)
    config = release_config(arm)
    planner = PPOPlanner(map_runner._ppo_planner_config(config))
    assert planner.get_metadata()["status"] == "ok", "fallback is not probe evidence"
    checkpoint = resolve_model_path(config["model_id"])
    digest = sha256_of_file(checkpoint)
    assert digest == get_registry_entry(config["model_id"])["github_release"]["sha256"]
    with zipfile.ZipFile(checkpoint) as archive:
        saved = torch.load(
            io.BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True
        )
    loaded = planner._model.policy.state_dict()
    assert saved.keys() == loaded.keys()
    assert all(torch.equal(value, loaded[key].cpu()) for key, value in saved.items())
    scenario_path, name = SCENARIOS[0]
    env_config = build_env_config(
        scenario_input(scenario_path, name), scenario_path=Path(scenario_path)
    )
    env_config.sim_config.time_per_step_in_secs = 0.1
    env = make_robot_env(config=env_config, seed=1001, debug=False)
    # Use the actual planner object in the standard map policy builder, preserving all adapters.
    original_constructor = map_runner.PPOPlanner
    map_runner.PPOPlanner = lambda *args, **kwargs: planner
    try:
        policy, metadata = map_runner._build_policy(
            arm, config, robot_kinematics="differential_drive"
        )
    finally:
        map_runner.PPOPlanner = original_constructor
    try:
        obs, _ = env.reset(seed=1001)
        policy._planner_bind_env(env)
        robot = env.simulator.robots[0]
        robot.state.velocity = initial_speed
        obs["robot_speed"] = np.asarray(initial_speed, dtype=np.float32)
        obs["robot_angular_velocity"] = np.asarray([initial_speed[1]], dtype=np.float32)
        # Same immutable model observation for all predictions, as in PPOS.
        model_obs = planner._build_model_obs_dict(copy.deepcopy(obs))
        observation_digest = hashlib.sha256()
        for key, value in sorted(model_obs.items()):
            value = np.asarray(value)
            observation_digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
            observation_digest.update(value.tobytes())
        raw, _ = planner._model.predict(model_obs, deterministic=True)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        planner._build_model_obs_dict = lambda _obs: model_obs
        rows = []
        for step in range(1, 6):
            obs["robot_speed"] = np.asarray(robot.current_speed, dtype=float)
            obs["robot_angular_velocity"] = np.asarray([robot.current_speed[1]], dtype=float)
            before = list(robot.current_speed)
            command = policy(obs)
            acceleration = policy_command_to_env_action(env=env, config=env_config, command=command)
            robot.apply_action(robot.parse_action(acceleration), 0.1)
            rows.append(
                {
                    "step": step,
                    "pre_speed": before,
                    "command": list(command),
                    "acceleration_request": acceleration.tolist(),
                    "applied": list(robot.current_speed),
                    "shield": copy.deepcopy(metadata.get("shield_stats", {}).get("last_decision")),
                }
            )
        return {
            "arm": arm,
            "model_id": config["model_id"],
            "checkpoint_sha256": digest,
            "saved_policy_tensors_equal": True,
            "policy_tensor_count": len(saved),
            "initial_speed": list(initial_speed),
            "raw": raw.tolist(),
            "steps": rows,
            "model_observation_sha256": observation_digest.hexdigest(),
            "action_semantics": metadata.get("action_semantics", "absolute_velocity"),
            "config_path": CONFIGS[arm],
            "config_sha256": sha256_of_file(CONFIGS[arm]),
            "scenario_path": scenario_path,
            "scenario": name,
            "seed": 1001,
            "dt": 0.1,
            "robot_contract": {
                "max_linear_speed": robot.config.max_linear_speed,
                "max_angular_speed": robot.config.max_angular_speed,
                "allow_backwards": robot.config.allow_backwards,
                "max_linear_accel": robot.config.max_linear_accel,
                "max_linear_decel": robot.config.max_linear_decel,
                "max_angular_accel": robot.config.max_angular_accel,
            },
        }
    finally:
        env.close()
        policy._planner_close()


def dev_episodes() -> list[dict[str, Any]]:
    """Run only the explicitly authorized 1001/1002 seeds through the episode executor."""
    rows = []
    for arm in CONFIGS:
        config = release_config(arm)
        for path, name in SCENARIOS:
            scenario = scenario_input(path, name)
            for seed in (1001, 1002):
                assert 1001 <= seed <= 1030
                record = map_runner._run_map_episode(
                    scenario,
                    seed,
                    horizon=200,
                    dt=0.1,
                    record_forces=False,
                    snqi_weights=None,
                    snqi_baseline=None,
                    algo=arm,
                    scenario_path=Path(path),
                    algo_config=config,
                    algo_config_path=CONFIGS[arm],
                    record_planner_decision_trace=True,
                    record_simulation_step_trace=True,
                )
                rows.append(record)
    return rows


def main() -> None:
    """Write reconstructable diagnostic results without publishing benchmark claims."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["probe", "episodes"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO")
    rows = (
        dev_episodes()
        if args.mode == "episodes"
        else [
            real_checkpoint_probe(arm, speed)
            for arm in CONFIGS
            for speed in [(0.6, 0.2), (0.6, 0.95)]
        ]
    )
    result = {
        "evidence_status": "diagnostic-only",
        "mode": args.mode,
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch_version": torch.__version__,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, default=lambda value: np.asarray(value).tolist()) + "\n"
    )


if __name__ == "__main__":
    main()
