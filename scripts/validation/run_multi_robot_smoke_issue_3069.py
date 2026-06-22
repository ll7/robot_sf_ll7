#!/usr/bin/env python3
"""Run a minimal multi-robot research smoke and report per-agent telemetry (issue #3069).

This is a **diagnostic-only / smoke** foothold for the multi-robot research lane
(parent #3057). It builds the smallest runnable multi-robot environment via
``robot_sf.gym_env.environment_factory.make_multi_robot_env``, rolls out a fixed
number of steps with a simple goal-seeking controller, and reports per-agent
collision and progress fields straight from the environment's native
``info["agents"]`` metadata.

Claim boundary: this proves only that the multi-robot environment steps
end-to-end and surfaces inter-robot collision/progress telemetry. It is **not** a
benchmark and makes no fleet-scale, MAPPO, or paper-facing multi-robot claim.

Usage::

    uv run python scripts/validation/run_multi_robot_smoke_issue_3069.py --help
    uv run python scripts/validation/run_multi_robot_smoke_issue_3069.py \\
        --json-output output/multi_robot_smoke_issue_3069.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.gym_env.environment_factory import make_multi_robot_env
from robot_sf.gym_env.unified_config import MultiRobotConfig
from robot_sf.sensor.range_sensor import LidarScannerSettings

SCHEMA_VERSION = "multi_robot_smoke.v1"
ISSUE = 3069
DEFAULT_CONFIG = Path("configs/multi_robot/issue_3069_smoke.yaml")

# Per-agent flags surfaced verbatim from RobotState.meta_dict().
_COLLISION_FLAGS = (
    "is_robot_collision",
    "is_pedestrian_collision",
    "is_obstacle_collision",
)
_PROGRESS_FLAGS = (
    "is_route_complete",
    "is_waypoint_complete",
    "is_timesteps_exceeded",
)


def _load_config(path: Path) -> dict[str, Any]:
    """Load the smoke config mapping (empty dict when absent)."""
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _goal_seeking_actions(env: Any) -> np.ndarray:
    """Return simple unicycle goal-seeking actions for every robot.

    Falls back to a hold-position (zero) action for any robot without a
    resolvable goal so the smoke never crashes on incomplete scenarios.
    """
    actions: list[list[float]] = []
    for sim in env.simulators:
        goals = list(getattr(sim, "goal_pos", []) or [])
        for index, robot in enumerate(sim.robots):
            goal = goals[index] if index < len(goals) else None
            if goal is None:
                actions.append([0.0, 0.0])
                continue
            pos = np.asarray(robot.pose[0], dtype=float)
            heading = float(robot.pose[1])
            goal_xy = np.asarray(goal, dtype=float)
            delta = goal_xy - pos
            distance = float(np.linalg.norm(delta))
            target_heading = float(np.arctan2(delta[1], delta[0])) if distance > 1e-9 else heading
            heading_error = ((target_heading - heading + np.pi) % (2.0 * np.pi)) - np.pi
            linear = min(0.6, distance)
            angular = float(np.clip(2.0 * heading_error, -1.0, 1.0))
            actions.append([linear, angular])
    return np.asarray(actions, dtype=np.float32)


def _actions(env: Any, policy: str) -> np.ndarray:
    """Return the action array for the requested smoke policy."""
    if policy == "zero":
        return np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    return _goal_seeking_actions(env)


def _summarize_agent(index: int, first: dict[str, Any], last: dict[str, Any]) -> dict[str, Any]:
    """Summarize one agent's collision and progress telemetry."""
    initial = first.get("prev_distance_to_goal", first.get("distance_to_goal"))
    final = last.get("distance_to_goal")
    initial_f = float(initial) if isinstance(initial, (int, float)) else None
    final_f = float(final) if isinstance(final, (int, float)) else None
    progress: float | None = None
    if initial_f is not None and final_f is not None and math.isfinite(initial_f - final_f):
        progress = initial_f - final_f
    summary: dict[str, Any] = {
        "agent_index": index,
        "step_of_episode": int(last.get("step_of_episode", 0)),
        "initial_distance_to_goal": initial_f,
        "final_distance_to_goal": final_f,
        "distance_progress": progress,
    }
    for flag in (*_COLLISION_FLAGS, *_PROGRESS_FLAGS):
        summary[flag] = bool(last.get(flag, False))
    return summary


def run_smoke(
    *,
    num_robots: int = 2,
    steps: int = 40,
    seed: int = 0,
    deterministic_lidar: bool = True,
    policy: str = "goal_seeking",
) -> dict[str, Any]:
    """Roll out a small multi-robot episode and return a diagnostic smoke report."""
    if num_robots < 1:
        raise ValueError("num_robots must be >= 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    config_kwargs: dict[str, Any] = {"num_robots": num_robots}
    if deterministic_lidar:
        config_kwargs["lidar_config"] = LidarScannerSettings(scan_noise=[0.0, 0.0])
    config = MultiRobotConfig(**config_kwargs)

    env = make_multi_robot_env(config=config, seed=seed)
    started = time.time()
    steps_executed = 0
    total_reward = 0.0
    terminated = False
    truncated = False
    first_metas: list[dict[str, Any]] | None = None
    last_metas: list[dict[str, Any]] | None = None
    try:
        env.reset(seed=seed)
        for _ in range(steps):
            _obs, reward, terminated, truncated, info = env.step(_actions(env, policy))
            steps_executed += 1
            total_reward += float(reward)
            metas = [dict(agent["meta"]) for agent in info["agents"]]
            if first_metas is None:
                first_metas = metas
            last_metas = metas
            if terminated or truncated:
                break
    finally:
        env.close()
    wall_time_sec = max(1e-9, time.time() - started)

    agents: list[dict[str, Any]] = []
    if first_metas is not None and last_metas is not None:
        for index, last in enumerate(last_metas):
            first = first_metas[index] if index < len(first_metas) else last
            agents.append(_summarize_agent(index, first, last))

    robot_collisions = [a for a in agents if a["is_robot_collision"]]
    inter_robot = {
        "any_robot_collision": bool(robot_collisions),
        "robot_collision_count": len(robot_collisions),
        "robot_collision_agents": [a["agent_index"] for a in robot_collisions],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "evidence_tier": "smoke",
        "claim_boundary": "diagnostic_only",
        "multi_robot_benchmark_claim": False,
        "config": {
            "num_robots": num_robots,
            "steps": steps,
            "seed": seed,
            "deterministic_lidar": deterministic_lidar,
            "policy": policy,
        },
        "rollout": {
            "steps_executed": steps_executed,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_reward": total_reward,
            "wall_time_sec": wall_time_sec,
        },
        "agents": agents,
        "inter_robot": inter_robot,
        "notes": (
            "Diagnostic-only multi-robot foothold (issue #3069). Proves the "
            "multi-robot environment steps end-to-end and surfaces per-agent "
            "collision/progress telemetry. Not a benchmark; no fleet-scale, "
            "MAPPO, or paper-facing claim."
        ),
    }


def _format_report(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary."""
    cfg = report["config"]
    rollout = report["rollout"]
    lines = [
        "multi-robot research smoke (issue #3069) — diagnostic_only",
        f"  num_robots={cfg['num_robots']} steps_executed={rollout['steps_executed']} "
        f"seed={cfg['seed']} policy={cfg['policy']}",
        f"  terminated={rollout['terminated']} total_reward={rollout['total_reward']:.4f} "
        f"wall_time_sec={rollout['wall_time_sec']:.3f}",
        f"  inter_robot_collision={report['inter_robot']['any_robot_collision']} "
        f"(count={report['inter_robot']['robot_collision_count']})",
    ]
    for agent in report["agents"]:
        progress = agent["distance_progress"]
        progress_str = f"{progress:.3f}" if isinstance(progress, float) else "n/a"
        lines.append(
            f"  agent[{agent['agent_index']}] progress={progress_str} "
            f"route_complete={agent['is_route_complete']} "
            f"robot_collision={agent['is_robot_collision']} "
            f"timeout={agent['is_timesteps_exceeded']}"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run a diagnostic-only multi-robot research smoke (issue #3069).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Smoke config providing defaults (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument("--num-robots", type=int, default=None, help="Override robot count.")
    parser.add_argument("--steps", type=int, default=None, help="Override rollout step count.")
    parser.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    parser.add_argument(
        "--policy",
        choices=("goal_seeking", "zero"),
        default=None,
        help="Override smoke control policy.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Write the full smoke report as JSON to this path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the multi-robot smoke CLI. Returns the process exit code."""
    args = _parse_args(argv)
    config = _load_config(args.config)

    num_robots = (
        args.num_robots if args.num_robots is not None else int(config.get("num_robots", 2))
    )
    steps = args.steps if args.steps is not None else int(config.get("steps", 40))
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    policy = args.policy if args.policy is not None else str(config.get("policy", "goal_seeking"))
    deterministic_lidar = bool(config.get("deterministic_lidar", True))

    report = run_smoke(
        num_robots=num_robots,
        steps=steps,
        seed=seed,
        deterministic_lidar=deterministic_lidar,
        policy=policy,
    )

    print(_format_report(report))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
