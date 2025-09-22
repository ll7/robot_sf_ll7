"""Example: Classic Interaction Scenario Visualization with PPO (No CLI).

This out-of-the-box demo replays a classic interaction scenario using a
pre-trained PPO model. Configuration is controlled by constants below – no
argument parsing or CLI flags are used (per feature requirement).

Quick start:
    uv run python examples/classic_interactions_pygame.py

Adjust the CONFIG CONSTANTS section to change scenario, number of episodes,
recording behavior, or paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

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

# ---------------------------------------------------------------------------
# CONFIG CONSTANTS (edit these to adjust behavior)
# ---------------------------------------------------------------------------
MODEL_PATH = Path("model/ppo_model_retrained_10m_2025-02-01.zip")
SCENARIO_MATRIX_PATH = Path("configs/scenarios/classic_interactions.yaml")
SCENARIO_NAME: str | None = None  # e.g., "classic_crossing_low" or None for first
MAX_EPISODES = 1
ENABLE_RECORDING = False
OUTPUT_DIR = Path("results/vis_runs")
DRY_RUN = False  # If True, validates resources then exits

# ---------------------------------------------------------------------------


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


def run_demo() -> List[dict[str, Any]]:
    """Run the configured demo and return episode summaries."""
    scenarios = load_classic_matrix(str(SCENARIO_MATRIX_PATH))
    scenario = select_scenario(scenarios, SCENARIO_NAME)
    seeds = iter_episode_seeds(scenario)
    if DRY_RUN:
        print(
            f"Dry run OK. Scenario={scenario.get('name')} seeds={seeds} model_exists={MODEL_PATH.exists()}"
        )
        return []
    model = _load_policy(str(MODEL_PATH))
    sim_cfg = RobotSimulationConfig()
    env = make_robot_env(config=sim_cfg, debug=False)
    results: List[dict[str, Any]] = []
    for ep_index, seed in enumerate(seeds):
        if ep_index >= MAX_EPISODES:
            break
        print(
            f"Running scenario={scenario.get('name')} seed={seed} ({ep_index + 1}/{MAX_EPISODES})"
        )
        res = run_episode(
            env=env,
            policy=model,
            record=ENABLE_RECORDING,
            out_dir=OUTPUT_DIR,
            scenario_name=scenario.get("name", "unknown"),
            seed=seed,
            episode_index=ep_index,
        )
        results.append(res)
    print("Summary:")
    for r in results:
        print(r)
    env.close()
    return results


if __name__ == "__main__":  # pragma: no cover
    run_demo()
