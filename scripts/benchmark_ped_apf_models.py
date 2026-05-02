"""Benchmark robot policies with and without adversarial pedestrian force.

This script runs two robot policies (default: run_023 and run_043) for a fixed
number of episodes under two conditions:
1) APF disabled
2) APF enabled

It prints a concise summary and writes full results to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.common.artifact_paths import ensure_canonical_tree, get_artifact_category_path
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.ped_npc.adversial_ped_force import AdversialPedForceConfig
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveSettings
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.sim.sim_config import SimulationSettings


@dataclass
class ConditionMetrics:
    """Aggregated episode metrics for one model-condition pair."""

    model: str
    apf_enabled: bool
    episodes: int
    wall_time_seconds: float
    total_sim_steps: int
    steps_per_second: float
    avg_sim_steps: float
    std_sim_steps: float
    avg_episode_reward: float
    std_episode_reward: float
    success_rate: float
    timeout_rate: float
    collision_rate: float
    pedestrian_collision_rate: float
    obstacle_collision_rate: float
    robot_collision_rate: float
    avg_distance_to_goal_end: float | None


def get_model_profile(model_name: str) -> dict[str, Any]:
    """Return environment/runtime profile settings for a given model."""
    if model_name == "run_023":
        return {
            "stack_steps": 1,
            "difficulty": 0,
            "ped_density_by_difficulty": [0.04],
            "robot_config": DifferentialDriveSettings(radius=1.0, max_angular_speed=0.5),
        }

    if model_name == "run_043":
        return {
            "stack_steps": 3,
            "difficulty": 0,
            "ped_density_by_difficulty": [0.04],
            "robot_config": BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
        }

    raise ValueError(
        f"Unsupported model name: {model_name!r}. "
        "Use one of: run_023, run_043 or extend get_model_profile()."
    )


def resolve_model_path(model_name: str) -> Path:
    """Resolve built-in model names to canonical model files."""
    return Path(__file__).resolve().parents[1] / "model" / f"{model_name}.zip"


def load_model(model_name: str):
    """Load a trained robot policy model by built-in name."""
    model_path = resolve_model_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_trained_policy(str(model_path))


def get_obs_adapter(model_name: str):
    """Return optional observation adapter for model compatibility."""
    if model_name == "run_023":

        def adapt_obs_run_023(obs_dict):
            if isinstance(obs_dict, dict):
                drive_state = np.asarray(obs_dict[OBS_DRIVE_STATE])
                ray_state = np.asarray(obs_dict[OBS_RAYS])

                drive_state = drive_state[:, :-1]
                drive_state[:, 2] *= 10

                drive_state = np.squeeze(drive_state).reshape(-1)
                ray_state = np.squeeze(ray_state).reshape(-1)
                return np.concatenate((ray_state, drive_state), axis=0)
            return obs_dict

        return adapt_obs_run_023

    return None


def make_env(svg_map_path: str, model_name: str, apf_enabled: bool, apf_offset: float):
    """Create robot env configured for one model and APF condition."""
    profile = get_model_profile(model_name)
    map_definition = convert_map(svg_map_path)

    apf_config = AdversialPedForceConfig(is_active=apf_enabled, offset=apf_offset)
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"benchmark_map": map_definition}),
        sim_config=SimulationSettings(
            stack_steps=profile["stack_steps"],
            difficulty=profile["difficulty"],
            ped_density_by_difficulty=profile["ped_density_by_difficulty"],
            peds_reset_follow_route_at_start=True,
            apf_config=apf_config,
            debug_without_robot_movement=False,
        ),
        robot_config=profile["robot_config"],
    )
    return make_robot_env(config=config, debug=False, recording_enabled=False)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total > 0 else 0.0


def _safe_avg_distance(values: list[float]) -> float | None:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return None
    return float(np.mean(clean))


def run_condition(
    svg_map_path: str,
    model_name: str,
    apf_enabled: bool,
    num_episodes: int,
    base_seed: int,
    apf_offset: float,
) -> ConditionMetrics:
    """Run one benchmark condition and aggregate per-episode metrics."""
    env = make_env(
        svg_map_path=svg_map_path,
        model_name=model_name,
        apf_enabled=apf_enabled,
        apf_offset=apf_offset,
    )
    model = load_model(model_name)
    obs_adapter = get_obs_adapter(model_name)

    episode_steps: list[float] = []
    episode_rewards: list[float] = []
    end_distances: list[float] = []

    success_count = 0
    timeout_count = 0
    ped_collision_count = 0
    obst_collision_count = 0
    robot_collision_count = 0
    any_collision_count = 0

    t0 = time.perf_counter()
    try:
        for episode_idx in range(num_episodes):
            episode_seed = base_seed + episode_idx
            obs, _ = env.reset(seed=episode_seed)
            done = False
            reward_sum = 0.0
            steps = 0
            meta: dict[str, Any] = {}

            while not done:
                adapted_obs = obs_adapter(obs) if obs_adapter is not None else obs
                action, _ = model.predict(adapted_obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                steps += 1
                done = bool(terminated or truncated)
                if done:
                    meta = info.get("meta", {}) if isinstance(info, dict) else {}

            step_of_episode = int(meta.get("step_of_episode", steps))
            is_success = bool(meta.get("is_route_complete", False))
            is_timeout = bool(meta.get("is_timesteps_exceeded", False))
            is_ped_collision = bool(meta.get("is_pedestrian_collision", False))
            is_obst_collision = bool(meta.get("is_obstacle_collision", False))
            is_robot_collision = bool(meta.get("is_robot_collision", False))

            any_collision = is_ped_collision or is_obst_collision or is_robot_collision

            episode_steps.append(float(step_of_episode))
            episode_rewards.append(float(reward_sum))
            end_distances.append(float(meta.get("distance_to_goal", math.nan)))

            success_count += int(is_success)
            timeout_count += int(is_timeout)
            ped_collision_count += int(is_ped_collision)
            obst_collision_count += int(is_obst_collision)
            robot_collision_count += int(is_robot_collision)
            any_collision_count += int(any_collision)

            if (episode_idx + 1) % 25 == 0 or (episode_idx + 1) == num_episodes:
                logger.info(
                    "{} | APF={} | completed {}/{} episodes",
                    model_name,
                    apf_enabled,
                    episode_idx + 1,
                    num_episodes,
                )
    finally:
        if hasattr(env, "exit"):
            env.exit()
        else:
            env.close()

    wall_time = float(time.perf_counter() - t0)
    total_steps = int(sum(int(v) for v in episode_steps))

    return ConditionMetrics(
        model=model_name,
        apf_enabled=apf_enabled,
        episodes=num_episodes,
        wall_time_seconds=wall_time,
        total_sim_steps=total_steps,
        steps_per_second=float(total_steps / max(wall_time, 1e-9)),
        avg_sim_steps=_safe_mean(episode_steps),
        std_sim_steps=_safe_std(episode_steps),
        avg_episode_reward=_safe_mean(episode_rewards),
        std_episode_reward=_safe_std(episode_rewards),
        success_rate=_safe_rate(success_count, num_episodes),
        timeout_rate=_safe_rate(timeout_count, num_episodes),
        collision_rate=_safe_rate(any_collision_count, num_episodes),
        pedestrian_collision_rate=_safe_rate(ped_collision_count, num_episodes),
        obstacle_collision_rate=_safe_rate(obst_collision_count, num_episodes),
        robot_collision_rate=_safe_rate(robot_collision_count, num_episodes),
        avg_distance_to_goal_end=_safe_avg_distance(end_distances),
    )


def _print_results_table(results: list[ConditionMetrics]) -> None:
    """Print compact benchmark table to stdout/log."""
    headers = (
        "model",
        "apf",
        "episodes",
        "avg_steps",
        "coll_rate",
        "ped_coll",
        "obst_coll",
        "robot_coll",
        "success",
        "timeout",
        "avg_reward",
    )
    logger.info(
        "{:<8} {:<5} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12}",
        *headers,
    )
    for m in results:
        logger.info(
            "{:<8} {:<5} {:>8} {:>10.2f} {:>10.2%} {:>10.2%} {:>10.2%} {:>10.2%} {:>10.2%} {:>10.2%} {:>12.3f}",
            m.model,
            "on" if m.apf_enabled else "off",
            m.episodes,
            m.avg_sim_steps,
            m.collision_rate,
            m.pedestrian_collision_rate,
            m.obstacle_collision_rate,
            m.robot_collision_rate,
            m.success_rate,
            m.timeout_rate,
            m.avg_episode_reward,
        )


def build_comparison(results: list[ConditionMetrics]) -> dict[str, dict[str, float | None]]:
    """Build APF on/off deltas per model."""
    by_model: dict[str, dict[bool, ConditionMetrics]] = {}
    for metric in results:
        by_model.setdefault(metric.model, {})[metric.apf_enabled] = metric

    comparisons: dict[str, dict[str, float | None]] = {}
    for model_name, variants in by_model.items():
        off = variants.get(False)
        on = variants.get(True)
        if off is None or on is None:
            continue

        dist_delta: float | None = None
        if off.avg_distance_to_goal_end is not None and on.avg_distance_to_goal_end is not None:
            dist_delta = float(on.avg_distance_to_goal_end - off.avg_distance_to_goal_end)

        comparisons[model_name] = {
            "delta_avg_sim_steps_on_minus_off": float(on.avg_sim_steps - off.avg_sim_steps),
            "delta_collision_rate_on_minus_off": float(on.collision_rate - off.collision_rate),
            "delta_success_rate_on_minus_off": float(on.success_rate - off.success_rate),
            "delta_avg_reward_on_minus_off": float(on.avg_episode_reward - off.avg_episode_reward),
            "delta_avg_distance_to_goal_end_on_minus_off": dist_delta,
        }
    return comparisons


def _default_output_path() -> Path:
    ensure_canonical_tree(categories=("benchmarks",))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_artifact_category_path("benchmarks") / f"ped_apf_models_benchmark_{stamp}.json"


def parse_args() -> argparse.Namespace:
    """Parse command line args."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark robot models with APF disabled/enabled and report aggregated metrics."
        ),
    )
    parser.add_argument(
        "--map",
        type=str,
        default="maps/svg_maps/masterthesis/headon.svg",
        help="Path to SVG map.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["run_023", "run_043"],
        help="Model names to benchmark.",
    )
    parser.add_argument(
        "--episodes-per-condition",
        type=int,
        default=100,
        help="Number of episodes for each model and APF condition.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed used to create deterministic per-episode seeds.",
    )
    parser.add_argument(
        "--apf-offset",
        type=float,
        default=10.0,
        help="APF offset used when APF is enabled.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path (default: output/benchmarks/...).",
    )
    return parser.parse_args()


def main() -> int:
    """Run full benchmark matrix and write summary JSON."""
    args = parse_args()

    results: list[ConditionMetrics] = []
    for model_name in args.models:
        for apf_enabled in (False, True):
            logger.info(
                "Running model={} with APF={} for {} episodes...",
                model_name,
                apf_enabled,
                args.episodes_per_condition,
            )
            metrics = run_condition(
                svg_map_path=args.map,
                model_name=model_name,
                apf_enabled=apf_enabled,
                num_episodes=args.episodes_per_condition,
                base_seed=args.seed,
                apf_offset=args.apf_offset,
            )
            results.append(metrics)

    _print_results_table(results)
    comparisons = build_comparison(results)

    output_path = Path(args.out) if args.out is not None else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "map": args.map,
        "models": args.models,
        "episodes_per_condition": int(args.episodes_per_condition),
        "seed": int(args.seed),
        "apf_offset": float(args.apf_offset),
        "results": [asdict(x) for x in results],
        "comparisons": comparisons,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote benchmark results to {}", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
