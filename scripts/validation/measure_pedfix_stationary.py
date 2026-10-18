"""Measure continuous pedestrian stalls with a stationary robot on development seeds."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from loguru import logger

from robot_sf.benchmark.map_runner.map_runner_env import build_env_config
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.scenario_loader import load_scenarios

logger.remove()
p = Path("configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml")
a = argparse.ArgumentParser()
a.add_argument("--part", type=int, default=0)
a.add_argument("--parts", type=int, default=1)
args = a.parse_args()
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
for i, s in enumerate(load_scenarios(p)):
    if i % args.parts != args.part:
        continue
    for seed in range(1001, 1006):
        assert 1001 <= seed <= 1030
        cfg = build_env_config(dict(s), scenario_path=p)
        env = make_robot_env(config=cfg, seed=seed, debug=False)
        env.reset(seed=seed)
        sim = env.simulator
        peds = sim.pysf_sim.peds
        n = peds.size()
        dt = cfg.sim_config.time_per_step_in_secs
        count = np.zeros(n, int)
        longest = np.zeros(n, int)
        ever = np.zeros(n, bool)
        nearwall = np.zeros(n, bool)
        walls = np.asarray(sim.map_def.obstacles_pysf, dtype=float).reshape(-1, 4)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        steps = round(cfg.sim_config.sim_time_in_secs / dt)
        for t in range(steps):
            env.step(action)
            pos = peds.pos()
            speed = np.linalg.norm(peds.vel(), axis=1)
            dg = np.linalg.norm(peds.goal() - pos, axis=1)
            dr = np.linalg.norm(pos - np.asarray(sim.robot_pos[0]), axis=1)
            eligible = (
                (speed < 0.1)
                & (dg > float(sim.pysf_sim.config.desired_force_config.goal_threshold))
                & (dr > float(sim.robots[0].config.radius) + cfg.sim_config.ped_radius + 2.0)
            )
            count = np.where(eligible, count + 1, 0)
            longest = np.maximum(longest, count)
            hit = count * dt > 5.0 + 1e-9
            ever |= hit
            if n and len(walls):
                start = walls[:, [0, 2]]
                end = walls[:, [1, 3]]
                d = end - start
                delta = pos[:, None, :] - start
                frac = np.clip(
                    np.sum(delta * d, axis=2) / np.maximum(np.sum(d * d, axis=1), 1e-12), 0, 1
                )
                wd = np.linalg.norm(delta - frac[:, :, None] * d, axis=2).min(axis=1)
                nearwall |= hit & (wd < 3.0)
        print(
            json.dumps(
                {
                    "name": s["name"],
                    "seed": seed,
                    "source_sha": sha,
                    "n": n,
                    "steps": steps,
                    "dt": dt,
                    "blocked_slots": np.where(ever)[0].tolist(),
                    "blocked": int(ever.sum()),
                    "fraction": float(ever.mean()) if n else None,
                    "wall_near_blocked": int(nearwall.sum()),
                    "longest_s": (longest * dt).tolist(),
                }
            ),
            flush=True,
        )
        env.close()
