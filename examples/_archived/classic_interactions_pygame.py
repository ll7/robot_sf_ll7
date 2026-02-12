"""Archived: Classic interactions pygame demo (see benchmarks/demo_full_classic_benchmark.py).

Usage:
    uv run python examples/_archived/classic_interactions_pygame.py

Prerequisites:
    - model/run_043.zip
    - configs/scenarios/classic_interactions.yaml

Expected Output:
    - Interactive pygame window plus optional MP4 recordings per episode.

Limitations:
    - Heavy interactive workflow retained for reference; prefer benchmark runner for CI.

References:
    - examples/benchmarks/demo_full_classic_benchmark.py

Classic Interaction Scenario Visualization with PPO (Feature 128).

Implemented per spec/tasks (FR-001..FR-021). Constants configure behavior; no CLI.

Quick start:
    uv run python examples/_archived/classic_interactions_pygame.py

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
import json
import os
import time
from pathlib import Path
from typing import Any, TypedDict

import numpy as np  # type: ignore
from loguru import logger

from robot_sf.benchmark.classic_interactions_loader import (
    iter_episode_seeds,
    load_classic_matrix,
    select_scenario,
)
from robot_sf.benchmark.helper_catalog import load_trained_policy
from robot_sf.benchmark.utils import (
    compute_fast_mode_and_cap,
    determine_episode_outcome,
    format_episode_summary_table,
)
from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.reward import simple_reward
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.map_config import MapDefinitionPool, serialize_map
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.render.helper_catalog import ensure_output_dir
from robot_sf.render.sim_view import MOVIEPY_AVAILABLE

# Stable-baselines3 import moved to helper catalog
PPO = None  # Retained for compatibility with existing checks
_PPO_IMPORT_ERROR = None

# TYPE_CHECKING imports removed - no longer needed

# ---------------------------------------------------------------------------
# CONFIG CONSTANTS (edit these to adjust behavior)
# ---------------------------------------------------------------------------
MODEL_PATH = Path("model/run_043.zip")
SCENARIO_MATRIX_PATH = Path("configs/scenarios/classic_interactions.yaml")
SCENARIO_NAME: str | None = None  # default = first scenario
# When True (or when run_demo(scenario_name="ALL") / run_demo(sweep=True)), iterate over all
# scenarios defined in the classic interaction matrix instead of selecting only one.
SWEEP_ALL = False
MAX_EPISODES = 8
ENABLE_RECORDING = True


def _default_output_dir() -> Path:
    """Return a timestamped artifact directory for demo outputs."""
    return resolve_artifact_path(Path("tmp/vis_runs") / time.strftime("%Y%m%d_%H%M%S"))


OUTPUT_DIR = _default_output_dir()
DRY_RUN = False  # global default dry_run if run_demo not passed a value
LOGGING_ENABLED = True  # toggle for non-essential prints (FR-014)

# Placeholder frame defaults used only when env-managed frames are unavailable.
# Prefer deriving width/height from SimulationView when present (avoids mismatched encodes).
DEFAULT_FRAME_HEIGHT = 360
DEFAULT_FRAME_WIDTH = 640
FRAME_CHANNELS = 3

# Internal: headless detection (FR-012). If SDL_VIDEODRIVER=dummy treat as headless.
HEADLESS = False  # retained placeholder; dynamic headless detection handled in run_demo

_CACHED_MODEL_SENTINEL = object()


class EpisodeSummary(TypedDict):  # (FR-020)
    """Typed schema for per-episode demo results."""

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
    """Validate required paths/constants before executing a non-dry demo run."""
    if not SCENARIO_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Scenario matrix not found: {SCENARIO_MATRIX_PATH}. Adjust SCENARIO_MATRIX_PATH constant.",
        )
    # MODEL_PATH validated lazily (allows dry_run to pass even if missing when not executing episodes)
    # Use helper catalog to ensure output directory exists
    if ENABLE_RECORDING:
        ensure_output_dir(OUTPUT_DIR)


# _load_policy function removed - now using load_trained_policy from helper catalog


# Removed: _determine_outcome - now using determine_episode_outcome from utils


def _placeholder_frame_shape(env) -> tuple[int, int, int]:
    """Return (H, W, C) for placeholder frames.

    Tries to read dimensions from env.sim_ui when available to match the actual render size.
    Falls back to DEFAULT_FRAME_HEIGHT/DEFAULT_FRAME_WIDTH when not available.
    """
    try:
        ui = getattr(env, "sim_ui", None)  # type: ignore[attr-defined]
        if ui is not None:
            h = int(getattr(ui, "height", DEFAULT_FRAME_HEIGHT))
            w = int(getattr(ui, "width", DEFAULT_FRAME_WIDTH))
            if h > 0 and w > 0:
                return (h, w, FRAME_CHANNELS)
    except Exception:  # pragma: no cover - defensive
        pass
    return (DEFAULT_FRAME_HEIGHT, DEFAULT_FRAME_WIDTH, FRAME_CHANNELS)


def _maybe_record(
    frames: list[Any],
    scenario_name: str,
    seed: int,
    episode_index: int,
    out_dir: Path,
) -> bool:
    """Write collected frames as an MP4 and report success.

    Args:
        frames: Ordered RGB frames to encode.
        scenario_name: Scenario name used in output filename.
        seed: Episode seed used in output filename.
        episode_index: Episode index used in output filename.
        out_dir: Destination directory for rendered video.

    Returns:
        True when a video was written successfully, otherwise False.
    """
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
    except Exception as exc:
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
) -> EpisodeSummary:
    """Run one episode for a scenario and return a normalized summary.

    Args:
        env: Initialized robot environment instance.
        policy: Policy object exposing `predict(obs, deterministic=True)`.
        record: Whether to capture frames and attempt MP4 export.
        out_dir: Output directory for optional video recording.
        scenario_name: Name used in result summary and output filename.
        seed: Random seed for env reset.
        episode_index: Index used for deterministic output naming.

    Returns:
        EpisodeSummary containing outcome flags, steps, and recording status.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    step = 0
    last_info: dict[str, Any] = {}
    frames: list[Any] | None = [] if record else None
    # Performance: capture frames only if recording enabled & moviepy available (FR-025)
    fast_demo_mode = bool(int(os.getenv("ROBOT_SF_FAST_DEMO", "0") or "0"))
    fast_step_cap = 2 if fast_demo_mode else None
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        # Only attempt render when a Simulation UI is present (debug or recording-enabled env)
        # This prevents RuntimeError in headless/debug=False test paths while still capturing
        # frames when visualization is available. (Fix for failing classic_interactions tests.)
        if getattr(env, "sim_ui", None):  # type: ignore[attr-defined]
            with contextlib.suppress(RuntimeError):
                env.render()  # ensure frame capture if env has sim_ui (FR-013)
        done = terminated or truncated
        step += 1
        last_info = info
        if frames is not None and record and MOVIEPY_AVAILABLE:
            # Attempt to reuse env frames if present; else sample minimal frame at small stride
            if hasattr(env, "sim_ui") and getattr(env.sim_ui, "frames", None):  # type: ignore[attr-defined]
                # Defer using env-managed frames after loop; no per-step copy
                pass
            elif step % 5 == 0:  # light sampling to limit memory (future FR-023 could refine)
                frames.append(np.zeros(_placeholder_frame_shape(env), dtype=np.uint8))

        # Fast demo early break to keep performance smoke test under threshold
        if fast_step_cap is not None and step >= fast_step_cap:
            done = True

    outcome = determine_episode_outcome(last_info)
    success = bool(last_info.get("success"))
    collision = bool(last_info.get("collision"))
    timeout = bool(last_info.get("timeout"))

    recorded_flag = False
    if record:
        if MOVIEPY_AVAILABLE:
            # If env stored frames, prefer them
            if hasattr(env, "sim_ui") and getattr(env.sim_ui, "frames", None):  # type: ignore[attr-defined]
                frames = env.sim_ui.frames  # type: ignore[assignment]
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

    has_sim_ui = hasattr(env, "sim_ui") and env.sim_ui is not None  # type: ignore[attr-defined]
    env_frames = []
    if has_sim_ui:
        env_frames = getattr(env.sim_ui, "frames", [])  # type: ignore[attr-defined]

    candidate_frames = env_frames if env_frames else frames

    if not candidate_frames:
        logger.warning(
            "[recording warning] Recording enabled but zero frames were captured. "
            "Likely causes: (1) Environment created without debug=True or record_video=True so no SimulationView present; "
            "(2) You never called env.render() inside the episode loop. Remediation: create env with debug=True or record_video and call env.render() each step.",
        )
        return

    # Detect placeholder zero frames (heuristic: all frames have sum == 0)
    try:

        def frame_sum(frame_obj):
            # Prefer ndarray.sum if present, else construct array
            """Return a scalar frame intensity sum for placeholder-frame detection."""
            if hasattr(frame_obj, "sum") and callable(frame_obj.sum):
                try:
                    return frame_obj.sum()  # type: ignore[no-any-return]
                except Exception:  # pragma: no cover - defensive inner
                    return np.array(frame_obj).sum()
            return np.array(frame_obj).sum()

        if candidate_frames and all(frame_sum(f) == 0 for f in candidate_frames):
            logger.warning(
                "[recording warning] All captured frames appear empty (sum=0). Videos will look black. "
                "Ensure the rendering path draws entities before frame capture and avoid placeholder arrays.",
            )
    except Exception:  # pragma: no cover - defensive outer
        pass


# Removed: _overlay_text - now using format_overlay_text from utils


# Removed: _format_summary_table - now using format_episode_summary_table from utils


# --- Helper extraction to reduce run_demo complexity (addresses Ruff C901) ---
# Removed: _compute_fast_mode_and_cap - now using compute_fast_mode_and_cap from utils


def _prepare_scenarios(scenario_name: str | None, sweep: bool) -> list[dict[str, Any]]:
    """Return a list of scenario dicts based on requested name & sweep flag.

    Rules:
      - If sweep==True OR scenario_name == "ALL" (case-sensitive) → return all scenarios in file order.
      - Else resolve a single scenario (None selects the first) via select_scenario.
      - Treat the literal string "None" as an unspecified scenario (historic quirk recovery).
    """
    scenarios = load_classic_matrix(str(SCENARIO_MATRIX_PATH))
    if sweep or scenario_name == "ALL":  # ALL sentinel triggers full sweep
        return scenarios
    effective_name: str | None = scenario_name
    if effective_name == "None":  # produced by str(None); treat as unset
        effective_name = None
    single = select_scenario(scenarios, effective_name)  # type: ignore[arg-type]
    return [single]


def _log_dry_run(scenario: dict[str, Any], seeds: list[int]) -> None:
    """Log a concise dry-run status line for one scenario.

    Args:
        scenario: Scenario definition dictionary.
        seeds: Deterministic seed list for the scenario.
    """
    logger.debug(
        f"Dry run OK. Scenario={scenario.get('name')} seeds={seeds} model_exists={MODEL_PATH.exists()}",
    )


def _load_or_stub_model(fast_mode: bool, eff_max: int):  # type: ignore[return-any]
    """Load the trained policy or return a stub policy in explicit fast mode.

    Args:
        fast_mode: Whether test/demo fast mode is active.
        eff_max: Effective episode limit after fast-mode capping.
    """
    explicit_fast_flag = bool(int(os.getenv("ROBOT_SF_FAST_DEMO", "0") or "0"))
    if fast_mode and eff_max <= 1:
        # Only fall back to stub if user explicitly requested fast demo; otherwise enforce model presence.
        if not MODEL_PATH.exists() and not explicit_fast_flag:
            raise FileNotFoundError(
                f"Model path does not exist: {MODEL_PATH}. Download a pre-trained PPO model or set ROBOT_SF_FAST_DEMO=1 for stub policy.",
            )
        if explicit_fast_flag:

            class _StubPolicy:  # pragma: no cover - trivial
                """Deterministic zero-action policy used for explicit fast-demo runs."""

                def predict(self, _obs, **_kwargs):
                    """Return a zero action and a placeholder state tuple."""
                    return np.zeros(2, dtype=float), None

            logger.info("FAST DEMO: Using stub policy (ROBOT_SF_FAST_DEMO=1)")
            return _StubPolicy()
    # Real model load path (non-fast or explicit model present)
    if not MODEL_PATH.exists():  # Surface actionable error for tests (model_path_failure)
        raise FileNotFoundError(
            f"Model path does not exist: {MODEL_PATH}. Download a pre-trained PPO model or set ROBOT_SF_FAST_DEMO=1 for stub policy.",
        )
    model_start = time.time()
    model = load_trained_policy(str(MODEL_PATH))
    logger.info(f"Loaded model in {time.time() - model_start:.2f}s: {MODEL_PATH}")
    return model


def _create_demo_env(fast_mode: bool):
    """Create base simulation config and apply fast-mode horizon reduction.

    Args:
        fast_mode: Whether fast demo constraints should be applied.
    """
    sim_cfg = RobotSimulationConfig()
    if fast_mode:
        sim_cfg.sim_config.sim_time_in_secs = min(sim_cfg.sim_config.sim_time_in_secs, 1.0)
    return sim_cfg


_MAP_CACHE: dict[str, object] = {}


def _load_map_definition(map_file: str):  # type: ignore[return-any]
    """Load a scenario-specific map definition.

    Supports:
      - SVG map paths (converted via convert_map)
      - JSON map definition files (serialized via serialize_map)

    Results cached by absolute path to avoid repeated parsing across seeds.
    Returns MapDefinition (not pooled) for caller to wrap.
    """
    abs_path = str(Path(map_file).resolve())
    if abs_path in _MAP_CACHE:
        return _MAP_CACHE[abs_path]
    p = Path(abs_path)
    if not p.exists():
        raise FileNotFoundError(f"Scenario map file not found: {abs_path}")
    if p.suffix.lower() == ".svg":
        md = convert_map(abs_path)
        if md is None:
            raise RuntimeError(f"Failed to convert SVG map: {abs_path}")
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        md = serialize_map(data)
    else:
        raise ValueError(f"Unsupported map file extension for scenario map: {p.suffix}")
    _MAP_CACHE[abs_path] = md
    return md


def _run_episodes(
    env,  # type: ignore[override]
    model,
    scenario: dict[str, Any],
    seeds: list[int],
    eff_max: int,
    eff_record: bool,
    out_dir: Path,
) -> list[EpisodeSummary]:
    """Run a bounded set of episodes for one scenario and collect summaries.

    Args:
        env: Initialized environment (closed by this helper).
        model: Loaded/stub policy used for action selection.
        scenario: Scenario metadata dictionary.
        seeds: Ordered seed list for repeated episodes.
        eff_max: Maximum number of episodes to execute.
        eff_record: Whether to record episode videos.
        out_dir: Destination directory for optional recordings.

    Returns:
        Ordered list of EpisodeSummary records.
    """
    results: list[EpisodeSummary] = []
    with contextlib.ExitStack() as stack:  # ensures close even on error
        stack.callback(env.close)
        for ep_index, seed in enumerate(seeds):
            if ep_index >= eff_max:
                break
            logger.info(
                f"Running scenario={scenario.get('name')} seed={seed} ({ep_index + 1}/{eff_max})",
            )
            summary = run_episode(
                env=env,
                policy=model,
                record=eff_record,
                out_dir=out_dir,
                scenario_name=scenario.get("name", "unknown"),
                seed=seed,
                episode_index=ep_index,
            )
            results.append(summary)
        if eff_record:
            frames_ref = []
            if hasattr(env, "sim_ui") and env.sim_ui is not None:  # type: ignore[attr-defined]
                frames_ref = getattr(env.sim_ui, "frames", [])  # type: ignore[attr-defined]
            _warn_if_no_frames(env, eff_record, frames_ref)
    return results


def _warn_if_recording_without_ui(env, record: bool) -> None:  # type: ignore[override]
    """Warn when recording is requested but no simulation view is available.

    Args:
        env: Environment potentially carrying a `sim_ui`.
        record: Whether recording was requested.
    """
    if record and not getattr(env, "sim_ui", None):  # user expects video but debug disabled
        logger.warning(
            "Recording enabled but environment created with debug=False: no real frames will be captured. "
            "Recreate with debug=True to enable visualization and non-empty video frames.",
        )


def _maybe_print_summary(results: list[EpisodeSummary]) -> None:
    """Print a formatted summary table when logging output is enabled.

    Args:
        results: Episode summaries to display.
    """
    if LOGGING_ENABLED:
        print("Summary:")
        # Guarantee at least a header line so logging toggle test observes difference.
        print(format_episode_summary_table(results))


def run_demo(
    dry_run: bool | None = None,
    scenario_name: str | None = None,
    max_episodes: int | None = None,
    enable_recording: bool | None = None,
    sweep: bool | None = None,
) -> list[EpisodeSummary]:
    """Execute the PPO visualization demo.

    Parameters
    ----------
    dry_run: bool | None
        If True, validate resources and exit without loading model or running episodes.
        If None, fall back to module-level DRY_RUN constant.
    scenario_name: str | None
        Override scenario selection; defaults to SCENARIO_NAME constant (first scenario if None).
        Special value: "ALL" (or sweep=True) runs every scenario in the matrix.
    max_episodes: int | None
        Limit number of episodes (defaults to MAX_EPISODES constant).
    enable_recording: bool | None
        Override recording toggle; defaults to ENABLE_RECORDING constant.
    sweep: bool | None
        If True, iterate over all scenarios (overrides scenario_name unless scenario_name names a
        single scenario not equal to "ALL"). If None, falls back to SWEEP_ALL constant.

    Returns
    -------
    list[EpisodeSummary]
        Ordered list of episode summaries (deterministic), possibly empty when dry_run.
    """
    _validate_constants()
    explicit_fast_flag = bool(int(os.getenv("ROBOT_SF_FAST_DEMO", "0") or "0"))
    effective_dry = DRY_RUN if dry_run is None else dry_run
    eff_name = SCENARIO_NAME if scenario_name is None else scenario_name
    eff_max = MAX_EPISODES if max_episodes is None else max_episodes
    eff_record = ENABLE_RECORDING if enable_recording is None else enable_recording
    eff_sweep = SWEEP_ALL if sweep is None else sweep

    fast_mode, eff_max = compute_fast_mode_and_cap(max_episodes=eff_max)
    scenarios = _prepare_scenarios(eff_name, sweep=eff_sweep or (eff_name == "ALL"))
    effective_output_dir = resolve_artifact_path(OUTPUT_DIR)

    if effective_dry:
        dry_msgs: list[str] = []
        for sc in scenarios:
            seeds_list = list(iter_episode_seeds(sc))
            dry_msgs.append(f"{sc.get('name')} seeds={seeds_list}")
        logger.debug(
            "Dry run OK. Scenarios="
            + "; ".join(dry_msgs)
            + f" model_exists={MODEL_PATH.exists()} sweep={len(scenarios) > 1}",
        )
        return []

    model = _load_or_stub_model(fast_mode=fast_mode, eff_max=eff_max)
    all_results: list[EpisodeSummary] = []
    for scenario in scenarios:
        map_file = scenario.get("map_file")
        sim_cfg = _create_demo_env(fast_mode=fast_mode)
        if isinstance(map_file, str):
            try:
                md = _load_map_definition(map_file)
                start_pos = getattr(md, "num_start_pos", 0)
                if start_pos and start_pos > 0:
                    sim_cfg.map_pool = MapDefinitionPool(map_defs={Path(map_file).stem: md})  # type: ignore[attr-defined]
                    logger.info(
                        "Loaded scenario map file: {mf} (start_positions={sp})",
                        mf=map_file,
                        sp=start_pos,
                    )
                else:
                    logger.warning(
                        "Scenario map '{mf}' has zero start positions (no robot routes); using default map pool instead.",
                        mf=map_file,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to load scenario map '{mf}' ({err}); falling back to default map pool.",
                    mf=map_file,
                    err=exc,
                )
        # Prefer config-driven render scaling to avoid factory API changes.
        try:
            sim_cfg.render_scaling = 20  # supported by both legacy and unified configs
        except Exception:
            pass
        env = make_robot_env(
            config=sim_cfg,
            reward_func=simple_reward,
            debug=not (explicit_fast_flag and not eff_record),
            record_video=eff_record,
            video_path=str(effective_output_dir) if eff_record else None,
        )
        logger.info(
            "Environment created (reward fallback active if custom reward not provided).",
        )  # (T022)
        _warn_if_recording_without_ui(env, eff_record)
        seeds = list(iter_episode_seeds(scenario))
        scenario_results = _run_episodes(
            env,
            model,
            scenario,
            seeds,
            eff_max,
            eff_record,
            out_dir=effective_output_dir,
        )
        all_results.extend(scenario_results)
    _maybe_print_summary(all_results)
    return all_results


if __name__ == "__main__":  # pragma: no cover
    run_demo(sweep=True)
