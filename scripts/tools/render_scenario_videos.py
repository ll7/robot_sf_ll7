#!/usr/bin/env python3
"""Render MP4 videos for scenarios in a scenario matrix.

Usage examples:
  uv run python scripts/tools/render_scenario_videos.py \
    --scenario configs/scenarios/francis2023.yaml \
    --all

  uv run python scripts/tools/render_scenario_videos.py \
    --scenario configs/scenarios/francis2023.yaml \
    --all --policy ppo --model-path model/run_023.zip

  uv run python scripts/tools/render_scenario_videos.py \
    --scenario configs/scenarios/francis2023.yaml \
    --scenario-id francis2023_parallel_traffic

Headless runs:
  SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python scripts/tools/render_scenario_videos.py \
    --scenario configs/scenarios/francis2023.yaml --all
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger

from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.common.artifact_paths import (
    ensure_canonical_tree,
    get_artifact_category_path,
    resolve_artifact_path,
)
from robot_sf.common.logging import configure_logging
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.sensor.sensor_fusion import OBS_DRIVE_STATE, OBS_RAYS
from robot_sf.training.scenario_loader import (
    build_robot_config_from_scenario,
    load_scenarios,
    select_scenario,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass
class RenderResult:
    """Outcome metadata for a single rendered scenario video."""

    scenario_id: str
    seed: int
    policy: str
    video_path: str
    steps: int
    status: str
    note: str | None = None
    error: str | None = None


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        type=Path,
        default=Path("configs/scenarios/francis2023.yaml"),
        help="Path to the scenario YAML file.",
    )
    parser.add_argument(
        "--scenario-id",
        type=str,
        help="Scenario name or id to render (defaults to all scenarios).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render videos for every scenario in the file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for rendered videos (defaults to timestamped folder).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for recorded videos.",
    )
    parser.add_argument(
        "--policy",
        choices=["goal", "ppo"],
        default="goal",
        help="Robot controller policy: 'goal' or PPO model (default: goal).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/run_023.zip"),
        help="Path to the PPO model when --policy ppo is selected.",
    )
    parser.add_argument(
        "--robot-speed",
        type=float,
        default=1.0,
        help="Nominal robot linear speed in m/s for the goal-seeking controller.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override per-scenario max_episode_steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        default=[],
        help="Override scenario seeds (repeatable).",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Limit the number of seeds per scenario.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def _slugify(value: str) -> str:
    """Convert a string into a filesystem-friendly slug."""
    cleaned = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "scenario"


def _resolve_output_root(
    scenario_path: Path,
    output: Path | None,
    *,
    policy: str,
) -> Path:
    """Resolve the output root directory for rendered videos."""
    if output is not None:
        resolved = resolve_artifact_path(output)
        if resolved.suffix:
            raise ValueError("--output must be a directory, not a file path")
        return resolved

    ensure_canonical_tree(categories=("recordings",))
    base = get_artifact_category_path("recordings")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"scenario_videos_{scenario_path.stem}_{policy}_{timestamp}"


def _resolve_scenario_name(scenario: Mapping[str, Any]) -> str:
    """Resolve a stable display name for a scenario entry."""
    name = scenario.get("name") or scenario.get("scenario_id")
    if name:
        return str(name)
    map_file = str(scenario.get("map_file") or "scenario")
    return Path(map_file).stem


def _resolve_seeds(
    scenario: Mapping[str, Any],
    *,
    seed_override: list[int],
    max_seeds: int | None,
) -> list[int]:
    """Determine the seed list for a scenario."""
    if seed_override:
        seeds = list(seed_override)
    else:
        seeds = list(scenario.get("seeds") or [0])
    if max_seeds is not None:
        seeds = seeds[: max(1, int(max_seeds))]
    return [int(seed) for seed in seeds]


def _wrap_angle(angle: float) -> float:
    """Normalize an angle to the [-pi, pi] range."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _goal_action(env, *, speed: float) -> np.ndarray:
    """Return a simple goal-seeking action for differential-drive robots."""
    sim = getattr(env, "simulator", None)
    if sim is None:
        return np.zeros(2, dtype=float)

    robot_pos = np.asarray(sim.robot_pos[0], dtype=float)
    robot_theta = float(sim.robot_poses[0][1])
    goal = np.asarray(sim.goal_pos[0], dtype=float)
    vec = goal - robot_pos
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return np.zeros(2, dtype=float)

    desired = math.atan2(vec[1], vec[0])
    heading_error = _wrap_angle(desired - robot_theta)
    linear = min(speed, dist)
    angular = heading_error
    action = np.array([linear, angular], dtype=float)

    if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
        low = np.asarray(env.action_space.low, dtype=float)
        high = np.asarray(env.action_space.high, dtype=float)
        if low.shape == action.shape and high.shape == action.shape:
            action = np.clip(action, low, high)
    return action


def _defensive_obs_adapter(orig_obs: Mapping[str, Any]) -> np.ndarray:
    """Adapt observations for the defensive PPO policy."""
    drive_state = orig_obs[OBS_DRIVE_STATE]
    ray_state = orig_obs[OBS_RAYS]
    drive_state = drive_state[:, :-1]
    drive_state[:, 2] *= 10
    drive_state = np.squeeze(drive_state)
    ray_state = np.squeeze(ray_state)
    return np.concatenate((ray_state, drive_state), axis=0)


def _sync_policy_spaces(env, policy_model: Any | None) -> None:
    """Sync env spaces to the policy model when available."""
    if policy_model is None:
        return
    obs_space = getattr(policy_model, "observation_space", None)
    if obs_space is not None:
        env.observation_space = obs_space
    action_space = getattr(policy_model, "action_space", None)
    if action_space is not None:
        env.action_space = action_space


def _resolve_goal_action(
    *,
    action_space: Any | None,
    policy_model: Any | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
    scenario_name: str,
    seed: int,
) -> bool:
    """Determine whether to use the goal-seeking controller."""
    use_goal_action = policy_model is None
    if use_goal_action:
        if action_space is None or getattr(action_space, "shape", None) != (2,):
            use_goal_action = False
        else:
            low = np.asarray(getattr(action_space, "low", [0.0, 0.0]), dtype=float)
            if not (low.shape == (2,) and float(low[0]) >= 0.0):
                use_goal_action = False
    if use_goal_action:
        logger.info("Scenario {} seed {} uses goal-seeking policy.", scenario_name, seed)
    elif policy_model is None or policy_obs_adapter is None:
        logger.warning(
            "Scenario {} seed {} policy fallback: random actions.",
            scenario_name,
            seed,
        )
    return use_goal_action


def _adapt_obs_for_policy(
    obs: Mapping[str, Any],
    *,
    policy_model: Any | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
) -> Mapping[str, Any] | np.ndarray:
    """Apply the policy observation adapter when needed."""
    if policy_model is not None and policy_obs_adapter is not None:
        return policy_obs_adapter(obs)
    return obs


def _select_action(
    *,
    env,
    obs: Any,
    action_space: Any | None,
    use_goal_action: bool,
    policy_model: Any | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
    robot_speed: float,
) -> np.ndarray:
    """Choose an action based on the selected policy mode."""
    if use_goal_action:
        return _goal_action(env, speed=robot_speed)
    if policy_model is not None and policy_obs_adapter is not None:
        action, _ = policy_model.predict(obs, deterministic=True)
        return action
    if action_space is not None:
        return action_space.sample()
    return np.zeros(2)


def _run_steps(
    *,
    env,
    obs: Any,
    action_space: Any | None,
    use_goal_action: bool,
    policy_model: Any | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
    robot_speed: float,
    max_steps: int,
) -> int:
    """Advance the environment until termination or the step budget."""
    steps = 0
    for _ in range(max_steps):
        action = _select_action(
            env=env,
            obs=obs,
            action_space=action_space,
            use_goal_action=use_goal_action,
            policy_model=policy_model,
            policy_obs_adapter=policy_obs_adapter,
            robot_speed=robot_speed,
        )
        obs, _, terminated, truncated, _ = env.step(action)
        obs = _adapt_obs_for_policy(
            obs,
            policy_model=policy_model,
            policy_obs_adapter=policy_obs_adapter,
        )
        env.render()
        steps += 1
        if terminated or truncated:
            break
    return steps


def _close_env(env) -> None:
    """Best-effort shutdown for the environment."""
    for method in ("exit", "close"):
        try:
            getattr(env, method)()
        except Exception:
            pass


def _validate_video(
    video_path: Path,
    *,
    status: str,
    note: str | None,
) -> tuple[str, str | None]:
    """Verify the output video and update the status if missing."""
    if status != "success":
        return status, note
    try:
        if not video_path.exists() or video_path.stat().st_size == 0:
            return "missing", "video missing or empty"
    except OSError as exc:
        return "missing", f"video stat failed: {exc}"
    return status, note


def _run_episode(
    scenario: Mapping[str, Any],
    *,
    scenario_path: Path,
    seed: int,
    output_dir: Path,
    fps: int,
    max_steps: int,
    robot_speed: float,
    policy_name: str,
    policy_model: Any | None,
    policy_obs_adapter: Callable[[Mapping[str, Any]], np.ndarray] | None,
) -> RenderResult:
    """Run a single scenario/seed episode and render a video."""
    scenario_name = _resolve_scenario_name(scenario)
    slug = _slugify(scenario_name)
    policy_slug = _slugify(policy_name)
    video_path = output_dir / f"{slug}_seed{seed}_{policy_slug}.mp4"

    config = build_robot_config_from_scenario(scenario, scenario_path=scenario_path)
    if policy_name == "ppo":
        config.sim_config.stack_steps = 1
    env = make_robot_env(
        config=config,
        seed=int(seed),
        debug=True,
        record_video=True,
        video_path=str(video_path),
        video_fps=float(fps),
    )

    steps = 0
    status = "success"
    note = None
    error = None
    try:
        _sync_policy_spaces(env, policy_model)
        obs, _ = env.reset()
        action_space = getattr(env, "action_space", None)
        use_goal_action = _resolve_goal_action(
            action_space=action_space,
            policy_model=policy_model,
            policy_obs_adapter=policy_obs_adapter,
            scenario_name=scenario_name,
            seed=seed,
        )
        obs = _adapt_obs_for_policy(
            obs,
            policy_model=policy_model,
            policy_obs_adapter=policy_obs_adapter,
        )
        steps = _run_steps(
            env=env,
            obs=obs,
            action_space=action_space,
            use_goal_action=use_goal_action,
            policy_model=policy_model,
            policy_obs_adapter=policy_obs_adapter,
            robot_speed=robot_speed,
            max_steps=max_steps,
        )
    except Exception as exc:
        status = "error"
        error = repr(exc)
        logger.exception("Scenario run failed: {} seed={}", scenario_name, seed)
    finally:
        _close_env(env)

    status, note = _validate_video(video_path, status=status, note=note)

    return RenderResult(
        scenario_id=scenario_name,
        seed=seed,
        policy=policy_name,
        video_path=str(video_path),
        steps=steps,
        status=status,
        note=note,
        error=error,
    )


def _scenario_max_steps(scenario: Mapping[str, Any], override: int | None) -> int:
    """Resolve the max step budget for a scenario."""
    if override is not None:
        return max(1, int(override))
    sim_cfg = scenario.get("simulation_config") or {}
    return max(1, int(sim_cfg.get("max_episode_steps", 400)))


def _write_manifest(
    output_dir: Path,
    *,
    scenario_path: Path,
    fps: int,
    robot_speed: float,
    policy: str,
    model_path: Path | None,
    max_steps: int | None,
    seed_override: list[int],
    max_seeds: int | None,
    results: list[RenderResult],
) -> Path:
    """Write a JSON manifest summarizing rendered videos."""
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenario_path": str(scenario_path),
        "output_dir": str(output_dir),
        "video_fps": fps,
        "robot_speed_m_s": robot_speed,
        "policy": policy,
        "model_path": str(model_path) if model_path is not None else None,
        "max_steps_override": max_steps,
        "seed_override": seed_override or None,
        "max_seeds": max_seeds,
        "results": [result.__dict__ for result in results],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for scenario video rendering."""
    args = _build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)

    scenario_path = args.scenario.resolve()
    scenarios = load_scenarios(scenario_path)

    if args.scenario_id and args.all:
        raise ValueError("Use --all or --scenario-id, not both.")

    if args.scenario_id:
        scenarios = [select_scenario(scenarios, args.scenario_id)]
    elif not args.all:
        logger.info("No scenario-id provided; defaulting to --all.")

    output_root = _resolve_output_root(scenario_path, args.output, policy=args.policy)
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Rendering videos to {}", output_root)

    policy_model = None
    policy_obs_adapter = None
    if args.policy == "ppo":
        policy_model = load_trained_policy(str(args.model_path))
        policy_obs_adapter = _defensive_obs_adapter

    results: list[RenderResult] = []
    for scenario in scenarios:
        scenario_name = _resolve_scenario_name(scenario)
        seeds = _resolve_seeds(
            scenario,
            seed_override=args.seed,
            max_seeds=args.max_seeds,
        )
        step_budget = _scenario_max_steps(scenario, args.max_steps)
        for seed in seeds:
            logger.info(
                "Rendering scenario={} seed={} steps={} fps={}",
                scenario_name,
                seed,
                step_budget,
                args.video_fps,
            )
            result = _run_episode(
                scenario,
                scenario_path=scenario_path,
                seed=seed,
                output_dir=output_root,
                fps=args.video_fps,
                max_steps=step_budget,
                robot_speed=args.robot_speed,
                policy_name=args.policy,
                policy_model=policy_model,
                policy_obs_adapter=policy_obs_adapter,
            )
            results.append(result)

    manifest_path = _write_manifest(
        output_root,
        scenario_path=scenario_path,
        fps=args.video_fps,
        robot_speed=args.robot_speed,
        policy=args.policy,
        model_path=args.model_path if args.policy == "ppo" else None,
        max_steps=args.max_steps,
        seed_override=args.seed,
        max_seeds=args.max_seeds,
        results=results,
    )

    missing = [r for r in results if r.status != "success"]
    logger.info("Rendered {} videos ({} issues).", len(results), len(missing))
    logger.info("Manifest written to {}", manifest_path)
    if missing:
        logger.warning("Some videos were missing or failed; check manifest for details.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
