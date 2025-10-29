"""Environment factory module (ergonomic public entry points).

Overview
--------
Provides explicit, typed factory functions for creating navigation and
pedestrian simulation environments. Replaces legacy ad‑hoc kwargs surface
with structured option objects while preserving backward compatibility via
``apply_legacy_kwargs`` (T029). Deterministic seeding added in T030.

Key Features
------------
* Explicit signatures: discoverable parameters; no ``**kwargs`` reliance for
    new behavior (only legacy capture path retained for deprecation window).
* Option normalization: boolean convenience flags (``record_video``,
    ``video_path``, ``video_fps``) map into :class:`RecordingOptions` and
    :class:`RenderOptions` with documented precedence rules.
* Determinism: optional ``seed`` seeds Python ``random``, NumPy, and PyTorch
    (if available) plus ``PYTHONHASHSEED``; applied before environment class
    construction. The applied seed is attached to the returned environment as
    ``applied_seed``.
* Pedestrian divergence: pedestrian factory honors an explicit
    ``RecordingOptions(record=False)`` even if ``record_video=True`` convenience
    flag is passed (opt‑out preserved). Robot/image factories flip to True in
    that scenario for user convenience.
* Performance guard: lightweight import strategy and timing test (T015/T031)
    ensure mean creation time regression is limited to +5% vs baseline.

Precedence Summary
------------------
1. Legacy mapped kwargs → merged first (if provided) producing / modifying option objects.
2. Explicit ``render_options`` / ``recording_options`` objects override boolean flags.
3. Robot/Image factories: ``record_video=True`` forces ``RecordingOptions.record=True``.
4. Pedestrian factory: preserves explicit opt‑out (``record=False``) even with ``record_video=True``.
5. ``video_fps`` populates ``RenderOptions.max_fps_override`` only when that field is unset.

Environment Variables
---------------------
* ``ROBOT_SF_FACTORY_LEGACY`` truthy → permissive legacy mapping (warnings only).
* ``ROBOT_SF_FACTORY_STRICT`` truthy → strict legacy mode (unknown legacy params raise).

This module contains no top‑level heavy imports beyond core types to minimize cold‑start
overhead (FR‑017). Rendering classes for image observations are imported lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

# Hoisted imports (avoid per-call import overhead for performance regression guard)
try:  # pragma: no cover - import errors would surface in tests
    from robot_sf.gym_env.robot_env import RobotEnv  # type: ignore

    # Intentionally do NOT import robot_env_with_image here to avoid heavy first-call cost.
    RobotEnvWithImage = None  # type: ignore
except (ImportError, ModuleNotFoundError):
    # Defer errors until factory invocation (lazy fallback)
    RobotEnv = None  # type: ignore
    RobotEnvWithImage = None  # type: ignore

from robot_sf.gym_env._factory_compat import LEGACY_PERMISSIVE_ENV, apply_legacy_kwargs
from robot_sf.gym_env.config_validation import get_resolved_config_dict, validate_config
from robot_sf.gym_env.options import RecordingOptions, RenderOptions
from robot_sf.gym_env.reward import simple_ped_reward
from robot_sf.gym_env.unified_config import (
    ImageRobotConfig,
    MultiRobotConfig,
    PedestrianSimulationConfig,
    RobotSimulationConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from robot_sf.gym_env.abstract_envs import MultiAgentEnv, SingleAgentEnv


class EnvironmentFactory:
    """Internal helpers to construct concrete environments (not exported)."""

    @staticmethod
    def create_robot_env(
        config: RobotSimulationConfig | None = None,
        *,
        use_image_obs: bool,
        peds_have_obstacle_forces: bool,
        reward_func: Callable | None,
        debug: bool,
        recording_enabled: bool,
        record_video: bool,
        video_path: str | None,
        video_fps: float | None,
        use_jsonl_recording: bool = False,
        recording_dir: str = "recordings",
        suite_name: str = "robot_sim",
        scenario_name: str = "default",
        algorithm_name: str = "manual",
        recording_seed: int | None = None,
    ) -> SingleAgentEnv:
        if config is None:
            config = ImageRobotConfig() if use_image_obs else RobotSimulationConfig()
        config.use_image_obs = use_image_obs
        config.peds_have_obstacle_forces = peds_have_obstacle_forces
        if use_image_obs:
            from robot_sf.gym_env.robot_env_with_image import (  # type: ignore
                RobotEnvWithImage as EnvCls,
            )
        else:
            EnvCls = RobotEnv  # type: ignore[assignment]
        if EnvCls is None:  # pragma: no cover - defensive
            raise RuntimeError("Environment classes failed to import; check installation.")
        return EnvCls(
            env_config=config,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
            use_jsonl_recording=use_jsonl_recording,
            recording_dir=recording_dir,
            suite_name=suite_name,
            scenario_name=scenario_name,
            algorithm_name=algorithm_name,
            recording_seed=recording_seed,
        )  # type: ignore[return-value]

    @staticmethod
    def create_pedestrian_env(
        robot_model,
        config: PedestrianSimulationConfig | None = None,
        reward_func: Callable[[dict], float] | None = None,
        debug: bool = False,
        recording_enabled: bool = False,
        peds_have_obstacle_forces: bool = False,
    ) -> SingleAgentEnv:
        if config is None:
            config = PedestrianSimulationConfig()
        from robot_sf.gym_env.pedestrian_env import PedestrianEnv

        # Allow None to be passed through from ergonomic factories and
        # fall back to the canonical internal simple_ped_reward.
        if reward_func is None:
            reward_func = simple_ped_reward

        return PedestrianEnv(
            env_config=config,  # type: ignore[arg-type]
            robot_model=robot_model,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )  # type: ignore[return-value]

    @staticmethod
    def create_multi_robot_env(
        config: MultiRobotConfig | None = None,
        *,
        num_robots: int,
        reward_func: Callable | None,
        debug: bool,
    ) -> MultiAgentEnv:
        if config is None:
            config = MultiRobotConfig()
        if config.num_robots != num_robots:
            config.num_robots = num_robots
        from robot_sf.gym_env.multi_robot_env import MultiRobotEnv

        return MultiRobotEnv(
            env_config=config,
            reward_func=reward_func,
            debug=debug,
            num_robots=num_robots,
        )  # type: ignore[return-value]


def _apply_render(mapped: dict[str, Any], render: RenderOptions | None):
    if "render_options.max_fps_override" in mapped:
        ro = render or RenderOptions()
        ro.max_fps_override = mapped.pop("render_options.max_fps_override")
        return ro
    return render


def _apply_recording(mapped: dict[str, Any], rec: RecordingOptions | None):
    keys = ("recording_options.record", "recording_options.video_path")
    if any(k in mapped for k in keys):
        out = rec or RecordingOptions()
        if keys[0] in mapped:
            out.record = mapped.pop(keys[0])
        if keys[1] in mapped:
            out.video_path = mapped.pop(keys[1])
        return out
    return rec


def _normalize_factory_inputs(
    *,
    record_video: bool,
    video_path: str | None,
    video_fps: float | None,
    render_options: RenderOptions | None,
    recording_options: RecordingOptions | None,
):
    """Normalize convenience flags into structured option objects.

    Legacy kwargs support was removed from the public factories to provide
    strict, explicit signatures (T009 expectation). This helper now focuses
    purely on boolean/primitive convenience mapping which keeps complexity
    low (addresses C901 exceedance after refactor).
    """
    if recording_options is None and (record_video or video_path):
        recording_options = RecordingOptions.from_bool_and_path(record_video, video_path, None)
    if video_fps is not None and (
        render_options is None or render_options.max_fps_override is None
    ):
        render_options = render_options or RenderOptions()
        render_options.max_fps_override = int(video_fps)
    if recording_options and record_video and not recording_options.record:
        logger.warning(
            "record_video=True but RecordingOptions.record is False; enabling recording (boolean convenience precedence).",
        )
        recording_options.record = True
    eff_record = record_video if recording_options is None else recording_options.record
    eff_path = video_path if recording_options is None else recording_options.video_path
    eff_fps = (
        video_fps
        if (render_options is None or render_options.max_fps_override is None)
        else float(render_options.max_fps_override)
    )
    return render_options, recording_options, eff_record, eff_path, eff_fps


def make_robot_env(
    config: RobotSimulationConfig | None = None,
    *,
    seed: int | None = None,
    peds_have_obstacle_forces: bool = False,
    reward_func: Callable | None = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: str | None = None,
    video_fps: float | None = None,
    render_options: RenderOptions | None = None,
    recording_options: RecordingOptions | None = None,
    use_jsonl_recording: bool = False,
    recording_dir: str = "recordings",
    suite_name: str = "robot_sim",
    scenario_name: str = "default",
    algorithm_name: str = "manual",
    recording_seed: int | None = None,
    **legacy_kwargs,
) -> SingleAgentEnv:
    """Create a standard robot environment (non-image observations).

    Parameters
    ----------
    config : RobotSimulationConfig | None
        Optional pre-constructed config; a default instance is created if ``None``.
    seed : int | None
        Deterministic seed (Python ``random``, NumPy, PyTorch, hash seed). Stored on the
        returned env as ``applied_seed``.
    peds_have_obstacle_forces : bool
        Whether pedestrians perceive the robot as an obstacle (interaction forces enabled).
    reward_func : Callable | None
        Optional custom reward function; falls back to internal simple reward with warning.
    debug : bool
        Enable debug / visual features (may trigger view creation when recording).
    recording_enabled : bool
        Master switch gating recording runtime even if options request it (feature flag style).
    record_video : bool
        Convenience flag; if True and no explicit ``RecordingOptions`` provided one is synthesized.
    video_path : str | None
        Convenience output path used if ``RecordingOptions`` absent or lacks path.
    video_fps : float | None
        Convenience FPS; sets ``RenderOptions.max_fps_override`` if unset.
    render_options : RenderOptions | None
        Advanced rendering options; takes precedence over convenience flags.
    recording_options : RecordingOptions | None
        Advanced recording options. For robot/image factories, convenience flag may upgrade
        ``record`` to True if conflicting (precedence rule #3).
    use_jsonl_recording : bool
        Enable JSONL episode recording (per-episode JSONL + metadata outputs) when recording is on.
    recording_dir : str
        Directory where JSONL recorder stores per-episode files if enabled.
    suite_name : str
        Suite identifier stored in JSONL metadata for downstream grouping.
    scenario_name : str
        Scenario identifier stored in JSONL metadata.
    algorithm_name : str
        Algorithm identifier stored in JSONL metadata.
    recording_seed : int | None
        Optional stable seed used by the recorder to derive deterministic episode names.
    legacy_kwargs : dict
        Deprecated legacy surface (mapped via :func:`apply_legacy_kwargs`). Unknown params
        rejected unless ``{env}`` is truthy (permissive).

    Returns
    -------
    SingleAgentEnv
        Initialized robot environment instance with ``applied_seed`` attribute set.

    Notes
    -----
    * Side-effects: seeds RNGs (idempotent for same seed), logs creation line.
    * Performance: heavy image rendering imports are avoided along this path.
    """.replace("{env}", LEGACY_PERMISSIVE_ENV)
    # Apply legacy parameter mapping FIRST so explicit new-style arguments win.
    if legacy_kwargs:
        mapped, _warnings = apply_legacy_kwargs(legacy_kwargs, strict=True)
        # Fold mapped flattened keys into current option objects.
        render_options_local = _apply_render(mapped, render_options)
        recording_options_local = _apply_recording(mapped, recording_options)
        # Any leftover keys after application are ignored (already validated by strict handling).
        render_options = render_options_local
        recording_options = recording_options_local

    _apply_global_seed(seed)

    # Validate configuration if provided
    if config is not None:
        validate_config(config, strict=True)
        resolved = get_resolved_config_dict(config)
        logger.debug("Resolved config: backend={}", resolved.get("backend", "fast-pysf"))

    render_options, recording_options, eff_record_video, eff_video_path, eff_video_fps = (
        _normalize_factory_inputs(
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            render_options=render_options,
            recording_options=recording_options,
        )
    )
    logger.info(
        "Creating robot env debug={debug} record_video={record} video_path={path} fps={fps}",
        debug=debug,
        record=eff_record_video,
        path=eff_video_path,
        fps=eff_video_fps,
    )
    env = EnvironmentFactory.create_robot_env(
        config=config,
        use_image_obs=False,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        record_video=eff_record_video,
        video_path=eff_video_path,
        video_fps=eff_video_fps,
        use_jsonl_recording=use_jsonl_recording,
        recording_dir=recording_dir,
        suite_name=suite_name,
        scenario_name=scenario_name,
        algorithm_name=algorithm_name,
        recording_seed=recording_seed,
    )
    env.applied_seed = seed
    return env


def make_image_robot_env(
    config: ImageRobotConfig | None = None,
    *,
    seed: int | None = None,
    peds_have_obstacle_forces: bool = False,
    reward_func: Callable | None = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: str | None = None,
    video_fps: float | None = None,
    render_options: RenderOptions | None = None,
    recording_options: RecordingOptions | None = None,
    use_jsonl_recording: bool = False,
    recording_dir: str = "recordings",
    suite_name: str = "robot_sim",
    scenario_name: str = "default",
    algorithm_name: str = "manual",
    recording_seed: int | None = None,
    **legacy_kwargs,
) -> SingleAgentEnv:
    """Create a robot environment with image observations.

    Mirrors :func:`make_robot_env` adding an internal switch to select the image-capable
    environment implementation. All parameter semantics and precedence are identical.
    Lazy import defers expensive view initialization until first creation.
    """
    if legacy_kwargs:
        mapped, _warnings = apply_legacy_kwargs(legacy_kwargs, strict=True)
        render_options = _apply_render(mapped, render_options)
        recording_options = _apply_recording(mapped, recording_options)
    _apply_global_seed(seed)
    render_options, recording_options, eff_record_video, eff_video_path, eff_video_fps = (
        _normalize_factory_inputs(
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            render_options=render_options,
            recording_options=recording_options,
        )
    )
    logger.info(
        "Creating image robot env debug={debug} record_video={record} video_path={path} fps={fps}",
        debug=debug,
        record=eff_record_video,
        path=eff_video_path,
        fps=eff_video_fps,
    )
    env = EnvironmentFactory.create_robot_env(
        config=config,
        use_image_obs=True,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        record_video=eff_record_video,
        video_path=eff_video_path,
        video_fps=eff_video_fps,
        use_jsonl_recording=use_jsonl_recording,
        recording_dir=recording_dir,
        suite_name=suite_name,
        scenario_name=scenario_name,
        algorithm_name=algorithm_name,
        recording_seed=recording_seed,
    )
    env.applied_seed = seed
    return env


def make_pedestrian_env(
    config: PedestrianSimulationConfig | None = None,
    *,
    seed: int | None = None,
    robot_model=None,
    reward_func: Callable | None = None,
    debug: bool = False,
    recording_enabled: bool = False,
    peds_have_obstacle_forces: bool = False,
    record_video: bool = False,
    video_path: str | None = None,
    video_fps: float | None = None,
    render_options: RenderOptions | None = None,
    recording_options: RecordingOptions | None = None,
    **legacy_kwargs,
) -> SingleAgentEnv:
    """Create a pedestrian (adversarial) environment.

    Differences vs. :func:`make_robot_env` / :func:`make_image_robot_env`:
    * Honors explicit recording opt-out (``RecordingOptions(record=False)``) even if
      ``record_video=True`` (no implicit flip). Required by T012 test.
    * May auto-enable ``debug`` when effective recording is active to ensure frames
      are produced.
    * Injects a stub robot model when ``robot_model`` is ``None`` for backward compatibility.

    Parameters follow the same semantics; additional ones:
    robot_model : Any | None
        Trained policy / model providing robot actions. A lightweight stub is injected if
        absent so tests and simple demos can still run.
    peds_have_obstacle_forces : bool
        Interaction force toggle for pedestrian physics.
    """
    # Capture explicit override intent BEFORE normalization mutates structures.
    if legacy_kwargs:
        mapped, _warnings = apply_legacy_kwargs(legacy_kwargs, strict=True)
        render_options = _apply_render(mapped, render_options)
        recording_options = _apply_recording(mapped, recording_options)

    _apply_global_seed(seed)

    _explicit_no_record = (
        recording_options is not None and recording_options.record is False and record_video
    )

    render_options, recording_options, eff_record_video, _eff_video_path, _eff_video_fps = (
        _normalize_factory_inputs(
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            render_options=render_options,
            recording_options=recording_options,
        )
    )

    # Reinstate explicit opt-out if requested (override robot-style convenience precedence).
    if _explicit_no_record:
        eff_record_video = False
        if recording_options is not None:
            recording_options.record = False  # ensure consistency for any downstream logic

    if robot_model is None:
        # Maintain previous behavior (tests rely on automatic stub when not provided)
        from robot_sf.gym_env._stub_robot_model import StubRobotModel

        robot_model = StubRobotModel()

    # Determine effective debug flag (enable view only if actually recording here)
    if eff_record_video and not _explicit_no_record:
        if recording_options is None or recording_options.record:
            debug = True

    logger.info(
        "Creating pedestrian env debug={debug} record_video={record} robot_model={has_model}",
        debug=debug,
        record=eff_record_video,
        has_model=robot_model is not None,
    )
    env = EnvironmentFactory.create_pedestrian_env(
        config=config,
        robot_model=robot_model,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
    )
    env.applied_seed = seed
    return env


def _apply_global_seed(seed: int | None) -> None:
    """Apply a deterministic seed to common RNG sources.

    Positioned at end of module to keep import block contiguous (PEP8). Seeds Python's
    ``random`` module, NumPy, and PyTorch (if available); sets PYTHONHASHSEED. Silent
    failures for optional libraries keep dependency surface minimal for lightweight usages.
    """
    if seed is None:
        return
    import os as _os
    import random as _random

    _random.seed(seed)
    try:  # NumPy optional
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except (ImportError, ModuleNotFoundError):
        # NumPy not available; skip seeding it
        pass
    try:  # PyTorch optional
        import torch as _torch  # type: ignore

        if hasattr(_torch, "manual_seed"):
            _torch.manual_seed(seed)
        if hasattr(_torch, "cuda") and hasattr(_torch.cuda, "manual_seed_all"):
            _torch.cuda.manual_seed_all(seed)
    except (ImportError, ModuleNotFoundError):
        # Torch not available; skip seeding it
        pass
    _os.environ["PYTHONHASHSEED"] = str(seed)


def make_multi_robot_env(
    num_robots: int = 2,
    *,
    config: MultiRobotConfig | None = None,
    reward_func: Callable | None = None,
    debug: bool = False,
) -> MultiAgentEnv:
    """Create a multi-robot environment."""
    return EnvironmentFactory.create_multi_robot_env(
        config=config,
        num_robots=num_robots,
        reward_func=reward_func,
        debug=debug,
    )


def _make_multi_robot_env_future(*_args, **_kwargs):  # pragma: no cover
    """Future ergonomic multi-robot factory placeholder (T013)."""
    raise NotImplementedError(
        "Future multi-robot ergonomic factory not yet implemented; see task T013.",
    )
