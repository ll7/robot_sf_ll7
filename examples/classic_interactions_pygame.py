"""Classic Interaction Scenario Visualization with PPO (Feature 128).

Implemented per spec/tasks (FR-001..FR-021). Constants configure behavior; no CLI.

Quick start:
    uv run python examples/classic_interactions_pygame.py

Public API:
    run_demo(dry_run: bool | None = None,
             scenario_name: str | None = None,
             max_episodes: int | None = None,
             enable_recording: bool | None = None) -> list[EpisodeSummary]

Return schema (EpisodeSummary TypedDict):
    {
      'scenario': str,
      'seed': int,
      'steps': int,
      'outcome': str,              # one of {'success','collision','timeout','done'}
      'success': bool,
      'collision': bool,
      'timeout': bool,
      'recorded': bool,
    }

Dry run (`dry_run=True`) validates resources & prints a summary but does not load the model
or step the environment (FR-004). Deterministic ordering of seeds is guaranteed as listed
in scenario YAML (FR-002/FR-003). Recording gracefully degrades when moviepy/ffmpeg are
unavailable (FR-008/FR-009). Logging verbosity controlled by LOGGING_ENABLED (FR-014).
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any, Iterable, List, TypedDict, cast

import numpy as np  # type: ignore
from loguru import logger

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
SCENARIO_NAME: str | None = None  # default = first scenario
MAX_EPISODES = 8
ENABLE_RECORDING = True
OUTPUT_DIR = Path("tmp/vis_runs/" + time.strftime("%Y%m%d_%H%M%S"))
DRY_RUN = False  # global default dry_run if run_demo not passed a value
LOGGING_ENABLED = True  # toggle for non-essential prints (FR-014)

# Internal: headless detection (FR-012). If SDL_VIDEODRIVER=dummy treat as headless.
HEADLESS = False  # os.environ.get("SDL_VIDEODRIVER", "").lower() == "dummy"

_CACHED_MODEL_SENTINEL = object()


class EpisodeSummary(TypedDict):  # (FR-020)
    scenario: str
    seed: int
    steps: int
    outcome: str
    success: bool
    collision: bool
    timeout: bool
    recorded: bool


if LOGGING_ENABLED:
    logger.level("DEBUG")
else:
    logger.level("INFO")

# ---------------------------------------------------------------------------


def _validate_constants() -> None:  # (FR-019)
    if not SCENARIO_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Scenario matrix not found: {SCENARIO_MATRIX_PATH}. Adjust SCENARIO_MATRIX_PATH constant."
        )
    # MODEL_PATH validated lazily (allows dry_run to pass even if missing when not executing episodes)
    if ENABLE_RECORDING and not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_policy(path: str):  # (FR-004, FR-007 guidance)
    cache = getattr(_load_policy, "_cache", _CACHED_MODEL_SENTINEL)
    if cache is not _CACHED_MODEL_SENTINEL:
        return cache  # type: ignore[return-value]
    if PPO is None:
        raise RuntimeError(
            "stable_baselines3 PPO import failed. Install with 'uv add stable-baselines3' to use this demo."
        )
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"  # explicit newline
            "Download or place the pre-trained PPO model at this path. "
            "See docs/dev/issues/classic-interactions-ppo/ for guidance."
        )
    model = PPO.load(path)
    setattr(_load_policy, "_cache", model)
    return model


def _determine_outcome(info: dict[str, Any]) -> str:
    if info.get("collision"):
        return "collision"
    if info.get("success"):
        return "success"
    if info.get("timeout"):
        return "timeout"
    return "done"


def _maybe_record(
    frames: list[Any], scenario_name: str, seed: int, episode_index: int, out_dir: Path
) -> bool:
    if not frames:
        return False
    try:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore

        out_dir.mkdir(parents=True, exist_ok=True)
        video_name = f"{scenario_name}_seed{seed}_ep{episode_index}.mp4"
        clip = ImageSequenceClip(frames, fps=10)
        # Some moviepy versions differ in signature; use minimal args
        clip.write_videofile(str(out_dir / video_name), codec="libx264", fps=10)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[recording skipped] error: {exc}")
        return False


def run_episode(
    env,
    policy,
    record: bool,
    out_dir: Path,
    scenario_name: str,
    seed: int,
    episode_index: int,
) -> EpisodeSummary:  # noqa: D401
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0
    last_info: dict[str, Any] = {}
    frames: list[Any] = [] if (record and MOVIEPY_AVAILABLE) else []
    # Performance: capture frames only if recording enabled & moviepy available (FR-025)
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        last_info = info
        if frames is not None and record and MOVIEPY_AVAILABLE:
            # Attempt to reuse env frames if present; else sample minimal frame at small stride
            if hasattr(env, "sim_ui") and getattr(env.sim_ui, "frames", None):  # type: ignore[attr-defined]
                # Defer using env-managed frames after loop; no per-step copy
                pass
            else:
                if step % 5 == 0:  # light sampling to limit memory (future FR-023 could refine)
                    frames.append(np.zeros((360, 640, 3), dtype=np.uint8))

    outcome = _determine_outcome(last_info)
    success = bool(last_info.get("success"))
    collision = bool(last_info.get("collision"))
    timeout = bool(last_info.get("timeout"))

    recorded_flag = False
    if record:
        if MOVIEPY_AVAILABLE:
            # If env stored frames, prefer them
            if hasattr(env, "sim_ui") and getattr(env.sim_ui, "frames", None):  # type: ignore[attr-defined]
                frames = getattr(env.sim_ui, "frames")  # type: ignore[assignment]
            recorded_flag = _maybe_record(frames, scenario_name, seed, episode_index, out_dir)
        else:
            logger.warning("[recording skipped] moviepy/ffmpeg not available")

    return EpisodeSummary(
        scenario=scenario_name,
        seed=int(seed),
        steps=step,
        outcome=outcome,
        success=success,
        collision=collision,
        timeout=timeout,
        recorded=recorded_flag,
    )


def _warn_if_no_frames(env, record: bool, frames: list[Any]) -> None:
    """Emit a diagnostic warning if recording was requested but no real frames were captured.

    Conditions triggering the warning:
    - `record` is True AND MOVIEPY_AVAILABLE
    - Either the environment has no `sim_ui` (not created with debug=True / record_video=True)
      OR the collected `frames` list is empty / only placeholder zero arrays.

    This helps users understand why produced videos may be black or missing (root cause of
    recent issue). The guidance is actionable and points to remediation steps.
    """
    if not (record and MOVIEPY_AVAILABLE):  # nothing to do
        return

    has_sim_ui = hasattr(env, "sim_ui") and getattr(env, "sim_ui") is not None  # type: ignore[attr-defined]
    env_frames = []
    if has_sim_ui:
        env_frames = getattr(env.sim_ui, "frames", [])  # type: ignore[attr-defined]

    candidate_frames = env_frames if env_frames else frames

    if not candidate_frames:
        logger.warning(
            "[recording warning] Recording enabled but zero frames were captured. "
            "Likely causes: (1) Environment created without debug=True or record_video=True so no SimulationView present; "
            "(2) You never called env.render() inside the episode loop. Remediation: create env with debug=True or record_video and call env.render() each step."
        )
        return

    # Detect placeholder zero frames (heuristic: all frames have sum == 0)
    try:

        def frame_sum(frame_obj):
            # Prefer ndarray.sum if present, else construct array
            if hasattr(frame_obj, "sum") and callable(getattr(frame_obj, "sum")):
                try:
                    return frame_obj.sum()  # type: ignore[no-any-return]
                except Exception:  # pragma: no cover - defensive inner
                    return np.array(frame_obj).sum()
            return np.array(frame_obj).sum()

        if candidate_frames and all(frame_sum(f) == 0 for f in candidate_frames):
            logger.warning(
                "[recording warning] All captured frames appear empty (sum=0). Videos will look black. "
                "Ensure the rendering path draws entities before frame capture and avoid placeholder arrays."
            )
    except Exception:  # pragma: no cover - defensive outer
        pass


def _overlay_text(scenario: str, seed: int, step: int, outcome: str | None = None) -> str:
    """Format overlay text (FR-016 helper). Not yet wired into a renderer; provided for future integration."""
    base = f"{scenario} | seed={seed} | step={step}"
    if outcome:
        return base + f" | {outcome}"
    return base


def _format_summary_table(rows: Iterable[EpisodeSummary]) -> str:
    rows = list(rows)
    if not rows:
        return "(no episodes)"
    headers: list[str] = ["scenario", "seed", "steps", "outcome", "recorded"]
    col_widths = {h: max(len(h), *(len(str(cast(Any, r)[h])) for r in rows)) for h in headers}

    def fmt_row(r: EpisodeSummary | dict[str, Any]) -> str:
        return " | ".join(str(cast(Any, r)[h]).ljust(col_widths[h]) for h in headers)

    header_row: EpisodeSummary | dict[str, Any] = {h: h for h in headers}
    lines = [fmt_row(header_row)]
    lines.append("-|-".join("-" * col_widths[h] for h in headers))
    lines.extend(fmt_row(r) for r in rows)
    return "\n" + "\n".join(lines)


def run_demo(
    dry_run: bool | None = None,
    scenario_name: str | None = None,
    max_episodes: int | None = None,
    enable_recording: bool | None = None,
) -> List[EpisodeSummary]:
    """Execute the PPO visualization demo.

    Parameters
    ----------
    dry_run: bool | None
        If True, validate resources and exit without loading model or running episodes.
        If None, fall back to module-level DRY_RUN constant.
    scenario_name: str | None
        Override scenario selection; defaults to SCENARIO_NAME constant (first scenario if None).
    max_episodes: int | None
        Limit number of episodes (defaults to MAX_EPISODES constant).
    enable_recording: bool | None
        Override recording toggle; defaults to ENABLE_RECORDING constant.

    Returns
    -------
    list[EpisodeSummary]
        Ordered list of episode summaries (deterministic), possibly empty when dry_run.
    """
    _validate_constants()
    effective_dry = DRY_RUN if dry_run is None else dry_run
    eff_name = SCENARIO_NAME if scenario_name is None else scenario_name
    eff_max = MAX_EPISODES if max_episodes is None else max_episodes
    eff_record = ENABLE_RECORDING if enable_recording is None else enable_recording

    scenarios = load_classic_matrix(str(SCENARIO_MATRIX_PATH))
    scenario = select_scenario(scenarios, eff_name)
    seeds = iter_episode_seeds(scenario)
    # Deterministic ordering explicit (FR-017) — convert to list to avoid reuse side-effects
    seeds = list(seeds)

    if effective_dry:
        logger.debug(
            f"Dry run OK. Scenario={scenario.get('name')} seeds={seeds} model_exists={MODEL_PATH.exists()}"
        )
        return []

    # Model load (FR-004, FR-007)
    model_start = time.time()
    model = _load_policy(str(MODEL_PATH))
    logger.info(f"Loaded model in {time.time() - model_start:.2f}s: {MODEL_PATH}")

    # Env creation
    sim_cfg = RobotSimulationConfig()
    env = make_robot_env(config=sim_cfg, debug=False)
    logger.info(
        "Environment created (reward fallback active if custom reward not provided)."
    )  # (T022)

    results: List[EpisodeSummary] = []
    with contextlib.ExitStack() as stack:  # ensures close even on error
        stack.callback(env.close)
        for ep_index, seed in enumerate(seeds):
            if ep_index >= eff_max:
                break
            logger.info(
                f"Running scenario={scenario.get('name')} seed={seed} ({ep_index + 1}/{eff_max})"
            )
            summary = run_episode(
                env=env,
                policy=model,
                record=eff_record,
                out_dir=OUTPUT_DIR,
                scenario_name=scenario.get("name", "unknown"),
                seed=seed,
                episode_index=ep_index,
            )
            results.append(summary)
        # Post-run diagnostics: warn if user expected a video but no real frames were captured.
        if eff_record:
            # Try to surface the frames reference (frames stored on env.sim_ui if present)
            frames_ref = []
            if hasattr(env, "sim_ui") and getattr(env, "sim_ui") is not None:  # type: ignore[attr-defined]
                frames_ref = getattr(env.sim_ui, "frames", [])  # type: ignore[attr-defined]
            _warn_if_no_frames(env, eff_record, frames_ref)
    if LOGGING_ENABLED:
        print("Summary:")
        print(_format_summary_table(results))
    return results


if __name__ == "__main__":  # pragma: no cover
    run_demo()
