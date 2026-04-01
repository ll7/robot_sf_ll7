"""Benchmark a pedestrian policy with collision-speed and impact-angle metrics.

The benchmark runs a pedestrian policy for a fixed number of episodes and
collects aggregate episode metrics plus collision-specific kinematics:
- robot speed at robot-pedestrian collision
- pedestrian speed at robot-pedestrian collision
- collision impact angle (deg)

Results are printed as a compact table and exported as JSON under output/benchmarks/.
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
from robot_sf.gym_env.environment_factory import make_pedestrian_env
from robot_sf.gym_env.reward import stationary_collision_ped_reward
from robot_sf.gym_env.unified_config import PedestrianSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.robot.bicycle_drive import BicycleDriveSettings
from robot_sf.sensor.range_sensor import LidarScannerSettings
from robot_sf.sim.sim_config import SimulationSettings


@dataclass
class PedPolicyMetrics:
    """Aggregated benchmark metrics for one pedestrian-policy run."""

    ped_model: str
    robot_model: str
    episodes: int
    wall_time_seconds: float
    total_sim_steps: int
    steps_per_second: float
    avg_sim_steps: float
    std_sim_steps: float
    avg_episode_reward: float
    std_episode_reward: float
    timeout_count: int
    robot_reached_goal_count: int
    timeout_rate: float
    pedestrian_collision_rate: float
    obstacle_collision_rate: float
    robot_collision_rate: float
    robot_at_goal_rate: float
    robot_obstacle_collision_rate: float
    robot_pedestrian_collision_rate: float
    avg_distance_to_robot_end: float | None
    collision_events: int
    front_hit_percentage: float
    back_hit_percentage: float
    side_hit_percentage: float
    avg_robot_speed_at_collision: float | None
    std_robot_speed_at_collision: float | None
    avg_ped_speed_at_collision: float | None
    std_ped_speed_at_collision: float | None
    avg_collision_impact_angle_deg: float | None
    std_collision_impact_angle_deg: float | None


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total > 0 else 0.0


def _safe_avg_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    clean = [v for v in values if math.isfinite(v)]
    return float(np.mean(clean)) if clean else None


def _safe_std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    clean = [v for v in values if math.isfinite(v)]
    return float(np.std(clean)) if clean else None


def _resolve_model_path(path_or_name: str, base_dir_name: str) -> Path:
    candidate = Path(path_or_name)
    if candidate.suffix == ".zip" and candidate.exists():
        return candidate

    root = Path(__file__).resolve().parents[1]
    named_candidate = root / base_dir_name / f"{path_or_name}.zip"
    if named_candidate.exists():
        return named_candidate

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve model from {path_or_name!r}. "
        f"Expected existing path or name under '{base_dir_name}/<name>.zip'."
    )


def _latest_ped_model_path() -> Path:
    model_dir = Path(__file__).resolve().parents[1] / "model_ped"
    if not model_dir.exists():
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    candidates = sorted(model_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No pedestrian model checkpoints found in model_ped/")
    return candidates[-1]


def _make_env(svg_map_path: str, robot_model_name_or_path: str):
    map_definition = convert_map(svg_map_path)
    robot_model_path = _resolve_model_path(robot_model_name_or_path, base_dir_name="model")
    robot_model = load_trained_policy(str(robot_model_path))
    if robot_model_path.stem != "run_043":
        raise ValueError(f"Unsupported robot model for this benchmark: {robot_model_path.stem!r}")

    config = PedestrianSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"benchmark_map": map_definition}),
        sim_config=SimulationSettings(
            difficulty=0,
            ped_density_by_difficulty=[0.04],
            debug_without_robot_movement=False,
            peds_reset_follow_route_at_start=True,
        ),
        robot_config=BicycleDriveSettings(radius=0.5, max_accel=3.0, allow_backwards=True),
        spawn_near_robot=False,
        ego_ped_lidar_config=LidarScannerSettings.ego_pedestrian_lidar(),
    )
    env = make_pedestrian_env(
        config=config,
        robot_model=robot_model,
        debug=False,
        recording_enabled=False,
        reward_func=stationary_collision_ped_reward,
    )
    return env, str(robot_model_path)


def _get_meta(info: Any) -> dict[str, Any]:
    if isinstance(info, dict):
        meta = info.get("meta", {})
        if isinstance(meta, dict):
            return meta
    return {}


def _extract_robot_speed(meta: dict[str, Any], env: Any) -> float:
    del meta
    return _extract_linear_speed(getattr(env.simulator.robots[0], "current_speed", math.nan))


def _extract_linear_speed(speed_like: Any) -> float:
    """Extract linear speed from scalar/tuple/array-like speed containers."""
    if isinstance(speed_like, np.ndarray):
        if speed_like.size == 0:
            return math.nan
        return float(speed_like.reshape(-1)[0])

    if isinstance(speed_like, (tuple, list)):
        if not speed_like:
            return math.nan
        return float(speed_like[0])

    try:
        return float(speed_like)
    except Exception:
        return math.nan


def _extract_ped_speed(env: Any) -> float:
    return _extract_linear_speed(getattr(env.simulator.ego_ped, "current_speed", math.nan))


def _run_single_episode(env: Any, ped_model: Any, seed: int) -> tuple[dict[str, Any], float, int]:
    """Run one episode and return final metadata, cumulative reward, and steps."""
    reset_out = env.reset(seed=seed)
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    done = False
    reward_sum = 0.0
    steps = 0
    meta: dict[str, Any] = {}

    while not done:
        action, _ = ped_model.predict(obs, deterministic=True)
        step_out = env.step(action)

        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        elif len(step_out) == 4:
            obs, reward, done, info = step_out
            done = bool(done)
        else:
            raise RuntimeError(f"Unexpected env.step() output length: {len(step_out)}")

        reward_sum += float(reward)
        steps += 1

        if isinstance(obs, tuple):
            obs = obs[0]

        if done:
            meta = _get_meta(info)

    return meta, reward_sum, steps


def _is_robot_collision(meta: dict[str, Any]) -> bool:
    return bool(meta.get("is_robot_collision", False))


def _is_robot_pedestrian_collision(meta: dict[str, Any]) -> bool:
    """Return whether the terminal event was specifically a robot-pedestrian collision."""
    if "is_robot_pedestrian_collision" in meta:
        return bool(meta.get("is_robot_pedestrian_collision", False))
    if "is_robot_ped_collision" in meta:
        return bool(meta.get("is_robot_ped_collision", False))
    return bool(meta.get("robot_ped_collision_zone"))


def _collect_collision_samples(
    *,
    meta: dict[str, Any],
    env: Any,
    robot_speed_at_collision: list[float],
    ped_speed_at_collision: list[float],
    impact_angle_deg_at_collision: list[float],
    zone_counts: dict[str, int],
) -> None:
    if not _is_robot_pedestrian_collision(meta):
        return
    robot_speed_at_collision.append(_extract_robot_speed(meta, env))
    ped_speed_at_collision.append(_extract_ped_speed(env))
    impact_angle_deg_at_collision.append(float(meta.get("collision_impact_angle_deg", math.nan)))

    zone = str(meta.get("robot_ped_collision_zone", "")).lower()
    if zone in zone_counts:
        zone_counts[zone] += 1


def _update_outcome_counts(meta: dict[str, Any], counts: dict[str, int]) -> None:
    counts["timeout"] += int(bool(meta.get("is_timesteps_exceeded", False)))
    counts["ped_collision"] += int(bool(meta.get("is_pedestrian_collision", False)))
    counts["obst_collision"] += int(bool(meta.get("is_obstacle_collision", False)))
    counts["robot_collision"] += int(_is_robot_collision(meta))
    counts["robot_at_goal"] += int(bool(meta.get("is_robot_at_goal", False)))
    counts["robot_obstacle_collision"] += int(bool(meta.get("is_robot_obstacle_collision", False)))
    counts["robot_ped_collision"] += int(_is_robot_pedestrian_collision(meta))


def run_benchmark(
    *,
    svg_map_path: str,
    ped_model_name_or_path: str,
    robot_model_name_or_path: str,
    num_episodes: int,
    base_seed: int,
) -> PedPolicyMetrics:
    """Run benchmark episodes for a pedestrian policy and aggregate metrics."""
    env, robot_model_path = _make_env(
        svg_map_path=svg_map_path,
        robot_model_name_or_path=robot_model_name_or_path,
    )

    ped_model_path = (
        _latest_ped_model_path()
        if ped_model_name_or_path == "latest"
        else _resolve_model_path(ped_model_name_or_path, base_dir_name="model_ped")
    )
    ped_model = load_trained_policy(str(ped_model_path))

    episode_steps: list[float] = []
    episode_rewards: list[float] = []
    end_distances: list[float] = []

    robot_speed_at_collision: list[float] = []
    ped_speed_at_collision: list[float] = []
    impact_angle_deg_at_collision: list[float] = []

    outcome_counts = {
        "timeout": 0,
        "ped_collision": 0,
        "obst_collision": 0,
        "robot_collision": 0,
        "robot_at_goal": 0,
        "robot_obstacle_collision": 0,
        "robot_ped_collision": 0,
    }
    zone_counts = {"front": 0, "back": 0, "side": 0}

    t0 = time.perf_counter()
    try:
        for episode_idx in range(num_episodes):
            episode_seed = base_seed + episode_idx
            meta, reward_sum, steps = _run_single_episode(env, ped_model, episode_seed)

            _collect_collision_samples(
                meta=meta,
                env=env,
                robot_speed_at_collision=robot_speed_at_collision,
                ped_speed_at_collision=ped_speed_at_collision,
                impact_angle_deg_at_collision=impact_angle_deg_at_collision,
                zone_counts=zone_counts,
            )

            step_of_episode = int(meta.get("step_of_episode", steps))
            episode_steps.append(float(step_of_episode))
            episode_rewards.append(float(reward_sum))
            end_distances.append(float(meta.get("distance_to_robot", math.nan)))
            _update_outcome_counts(meta, outcome_counts)

            if (episode_idx + 1) % 25 == 0 or (episode_idx + 1) == num_episodes:
                logger.info("Completed {}/{} episodes", episode_idx + 1, num_episodes)
    finally:
        if hasattr(env, "exit"):
            env.exit()
        else:
            env.close()

    wall_time = float(time.perf_counter() - t0)
    total_steps = int(sum(int(v) for v in episode_steps))
    collision_events = len(robot_speed_at_collision)

    return PedPolicyMetrics(
        ped_model=str(ped_model_path),
        robot_model=str(robot_model_path),
        episodes=num_episodes,
        wall_time_seconds=wall_time,
        total_sim_steps=total_steps,
        steps_per_second=float(total_steps / max(wall_time, 1e-9)),
        avg_sim_steps=_safe_mean(episode_steps),
        std_sim_steps=_safe_std(episode_steps),
        avg_episode_reward=_safe_mean(episode_rewards),
        std_episode_reward=_safe_std(episode_rewards),
        timeout_count=int(outcome_counts["timeout"]),
        robot_reached_goal_count=int(outcome_counts["robot_at_goal"]),
        timeout_rate=_safe_rate(outcome_counts["timeout"], num_episodes),
        pedestrian_collision_rate=_safe_rate(outcome_counts["ped_collision"], num_episodes),
        obstacle_collision_rate=_safe_rate(outcome_counts["obst_collision"], num_episodes),
        robot_collision_rate=_safe_rate(outcome_counts["robot_collision"], num_episodes),
        robot_at_goal_rate=_safe_rate(outcome_counts["robot_at_goal"], num_episodes),
        robot_obstacle_collision_rate=_safe_rate(
            outcome_counts["robot_obstacle_collision"],
            num_episodes,
        ),
        robot_pedestrian_collision_rate=_safe_rate(
            outcome_counts["robot_ped_collision"],
            num_episodes,
        ),
        avg_distance_to_robot_end=_safe_avg_or_none(end_distances),
        collision_events=collision_events,
        front_hit_percentage=_safe_rate(zone_counts["front"], collision_events),
        back_hit_percentage=_safe_rate(zone_counts["back"], collision_events),
        side_hit_percentage=_safe_rate(zone_counts["side"], collision_events),
        avg_robot_speed_at_collision=_safe_avg_or_none(robot_speed_at_collision),
        std_robot_speed_at_collision=_safe_std_or_none(robot_speed_at_collision),
        avg_ped_speed_at_collision=_safe_avg_or_none(ped_speed_at_collision),
        std_ped_speed_at_collision=_safe_std_or_none(ped_speed_at_collision),
        avg_collision_impact_angle_deg=_safe_avg_or_none(impact_angle_deg_at_collision),
        std_collision_impact_angle_deg=_safe_std_or_none(impact_angle_deg_at_collision),
    )


def _print_results(metrics: PedPolicyMetrics) -> None:
    """Print compact summary table for benchmark metrics."""
    logger.info("Pedestrian policy benchmark summary")
    logger.info("ped_model: {}", metrics.ped_model)
    logger.info("robot_model: {}", metrics.robot_model)
    logger.info(
        "episodes={} | avg_steps={:.2f} | avg_reward={:.3f} | robot_collision_rate={:.2%}",
        metrics.episodes,
        metrics.avg_sim_steps,
        metrics.avg_episode_reward,
        metrics.robot_collision_rate,
    )
    logger.info(
        "timeouts={} ({:.2%}) | robot_reached_goal={} ({:.2%})",
        metrics.timeout_count,
        metrics.timeout_rate,
        metrics.robot_reached_goal_count,
        metrics.robot_at_goal_rate,
    )
    logger.info(
        "outcome_breakdown robot_collision={:.2%} | pedestrian_collision={:.2%} | obstacle_collision={:.2%} | timeout={:.2%} | robot_goal={:.2%}",
        metrics.robot_collision_rate,
        metrics.pedestrian_collision_rate,
        metrics.obstacle_collision_rate,
        metrics.timeout_rate,
        metrics.robot_at_goal_rate,
    )
    logger.info(
        "collision_events={} | avg_robot_speed={:.3f} | avg_ped_speed={:.3f} | avg_angle_deg={:.2f}",
        metrics.collision_events,
        metrics.avg_robot_speed_at_collision or 0.0,
        metrics.avg_ped_speed_at_collision or 0.0,
        metrics.avg_collision_impact_angle_deg or 0.0,
    )
    logger.info(
        "hit_zones(front/back/side)={:.1%}/{:.1%}/{:.1%}",
        metrics.front_hit_percentage,
        metrics.back_hit_percentage,
        metrics.side_hit_percentage,
    )


def _default_output_path() -> Path:
    ensure_canonical_tree(categories=("benchmarks",))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_artifact_category_path("benchmarks") / f"ped_policy_collision_benchmark_{stamp}.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark a pedestrian policy and report collision kinematics metrics.",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="maps/svg_maps/masterthesis/headon.svg",
        help="Path to SVG map.",
    )
    parser.add_argument(
        "--ped-model",
        type=str,
        required=True,
        help=(
            "Pedestrian model path or pinned model name. "
            "Pass 'latest' explicitly to opt into mtime-based selection."
        ),
    )
    parser.add_argument(
        "--robot-model",
        type=str,
        default="run_043",
        help="Robot model path or model name (default: run_043).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of benchmark episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base seed used to create deterministic per-episode seeds.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output JSON path (default: output/benchmarks/...).",
    )
    return parser.parse_args()


def main() -> int:
    """Run pedestrian benchmark and write JSON results."""
    args = parse_args()

    metrics = run_benchmark(
        svg_map_path=args.map,
        ped_model_name_or_path=args.ped_model,
        robot_model_name_or_path=args.robot_model,
        num_episodes=int(args.episodes),
        base_seed=int(args.seed),
    )

    _print_results(metrics)

    output_path = Path(args.out) if args.out is not None else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "map": args.map,
        "ped_model": args.ped_model,
        "robot_model": args.robot_model,
        "episodes": int(args.episodes),
        "seed": int(args.seed),
        "metrics": asdict(metrics),
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote benchmark results to {}", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
