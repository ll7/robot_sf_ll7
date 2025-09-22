"""Example: Visualize Classic Interaction Scenarios with a PPO Policy.

Run a selected scenario from `configs/scenarios/classic_interactions.yaml` using the
pre-trained PPO model and render with `SimulationView`.

Usage (interactive window if display available):
  uv run python examples/classic_interactions_pygame.py \
      --model model/ppo_model_retrained_10m_2025-02-01.zip \
      --scenario classic_crossing_low \
      --matrix configs/scenarios/classic_interactions.yaml \
      --episodes 2 --record

Headless (record only) example:
  DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python examples/classic_interactions_pygame.py --episodes 1 --record

This script is intentionally self-contained and lightweight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore

try:
    from stable_baselines3 import PPO  # type: ignore
except Exception as exc:  # noqa: BLE001
    PPO = None  # type: ignore
    _PPO_IMPORT_ERROR = exc
else:
    _PPO_IMPORT_ERROR = None

from robot_sf.benchmark.classic_interactions_loader import (
    iter_episode_seeds,
    load_classic_matrix,
    select_scenario,
)
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.render.sim_view import MOVIEPY_AVAILABLE


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classic Interaction PPO Visualization")
    p.add_argument("--model", default="model/ppo_model_retrained_10m_2025-02-01.zip")
    p.add_argument("--matrix", default="configs/scenarios/classic_interactions.yaml")
    p.add_argument("--scenario", default=None, help="Scenario name to run (default: first in file)")
    p.add_argument("--episodes", type=int, default=1, help="Max episodes to run")
    p.add_argument("--record", action="store_true", help="Enable MP4 recording per episode")
    p.add_argument("--outdir", default="results/vis_runs", help="Output directory for recordings")
    p.add_argument("--no-overlay", action="store_true", help="Disable overlay text for perf")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs then exit")
    return p.parse_args(argv)


def _load_policy(path: str):
    if PPO is None:
        raise RuntimeError(
            f"stable_baselines3 PPO import failed: {_PPO_IMPORT_ERROR}. Install dependency to use this example."
        )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return PPO.load(path)


def run_episode(
    env, policy, record: bool, out_dir: Path, scenario_name: str, seed: int, episode_index: int
) -> dict[str, Any]:  # noqa: D401
    obs, _ = env.reset(seed=seed)
    done = False
    frames = []  # collected only when recording
    step = 0
    outcome = None
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # The environment may already perform rendering; we only increment step count here.
        step += 1
    # Determine outcome heuristically
    if info.get("collision"):
        outcome = "collision"
    elif info.get("success"):
        outcome = "success"
    elif info.get("timeout"):
        outcome = "timeout"
    else:
        outcome = "done"

    if record and MOVIEPY_AVAILABLE:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            # If env has its own recorded frames use them; else create a simple placeholder frame summary.
            if hasattr(env, "sim_ui") and getattr(
                env.sim_ui, "frames", None
            ):  # pragma: no branch - simple attribute check
                frames = env.sim_ui.frames  # type: ignore[attr-defined]
            else:
                frames = [np.zeros((360, 640, 3), dtype=np.uint8)]
            if frames:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore

                clip = ImageSequenceClip(frames, fps=10)
                video_name = f"{scenario_name}_seed{seed}_ep{episode_index}.mp4"
                clip.write_videofile(str(out_dir / video_name), codec="libx264", fps=10)
        except Exception as exc:  # noqa: BLE001
            print(f"[recording skipped] error: {exc}")
    elif record and not MOVIEPY_AVAILABLE:
        print("[recording skipped] moviepy/ffmpeg not available")

    return {
        "scenario": scenario_name,
        "seed": seed,
        "steps": step,
        "outcome": outcome,
        "recorded": bool(frames),
    }


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    scenarios = load_classic_matrix(args.matrix)
    scenario = select_scenario(scenarios, args.scenario)
    seeds = iter_episode_seeds(scenario)
    model = None
    if args.dry_run:
        # Validate model path existence only
        if not Path(args.model).exists():
            print(f"Model file missing: {args.model}")
            return 1
        print(f"Dry run OK. First scenario: {scenario.get('name')} (seeds: {seeds})")
        return 0
    model = _load_policy(args.model)

    # Configure environment
    sim_cfg = RobotSimulationConfig()
    # Optionally adjust config here (e.g., shorter max steps for interactive demo)
    env = make_robot_env(config=sim_cfg, debug=False)

    # Note: Environment internally manages its SimulationView when configured; external view not required here.

    results = []
    for ep_index, seed in enumerate(seeds):
        if ep_index >= args.episodes:
            break
        print(
            f"Running scenario={scenario.get('name')} seed={seed} ({ep_index + 1}/{args.episodes})"
        )
        result = run_episode(
            env=env,
            policy=model,
            record=args.record,
            out_dir=Path(args.outdir),
            scenario_name=scenario.get("name", "unknown"),
            seed=seed,
            episode_index=ep_index,
        )
        results.append(result)
    print("Summary:")
    for r in results:
        print(r)
    env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual usage
    raise SystemExit(main(sys.argv[1:]))
