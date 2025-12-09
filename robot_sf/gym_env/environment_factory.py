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

import importlib
import os
import random
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

from robot_sf.gym_env._factory_compat import apply_legacy_kwargs
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


def _load_robot_env_with_image():
    """Lazy-load the image-capable robot environment class.

    This lazy import defers the heavy Pygame/rendering dependencies until first use,
    reducing cold-start overhead for headless or non-visual workloads.

    Returns:
        type: RobotEnvWithImage class from robot_env_with_image module.

    Raises:
        ModuleNotFoundError: If robot_env_with_image module is not available.
    """
    module = importlib.import_module("robot_sf.gym_env.robot_env_with_image")
    return module.RobotEnvWithImage


def _load_pedestrian_env():
    """Lazy-load the pedestrian (adversarial) environment class.

    This lazy import defers expensive initialization until first use, enabling
    fast instantiation of robot-only environments without pedestrian dependencies.

    Returns:
        type: PedestrianEnv class from pedestrian_env module.

    Raises:
        ModuleNotFoundError: If pedestrian_env module is not available.
    """
    module = importlib.import_module("robot_sf.gym_env.pedestrian_env")
    return module.PedestrianEnv


def _load_multi_robot_env():
    """Lazy-load the multi-robot environment class.

    This lazy import defers initialization of multi-agent infrastructure until needed,
    keeping single-agent workflows lightweight.

    Returns:
        type: MultiRobotEnv class from multi_robot_env module.

    Raises:
        ModuleNotFoundError: If multi_robot_env module is not available.
    """
    module = importlib.import_module("robot_sf.gym_env.multi_robot_env")
    return module.MultiRobotEnv


def _load_stub_robot_model():
    """Lazy-load the stub robot model class for testing.

    Returns a minimal robot model that produces zero actions, enabling pedestrian
    environment initialization without requiring a trained policy.

    Returns:
        type: StubRobotModel class (callable) providing zero-action fallback.

    Raises:
        ModuleNotFoundError: If _stub_robot_model module is not available.
    """
    module = importlib.import_module("robot_sf.gym_env._stub_robot_model")
    return module.StubRobotModel


def _optional_import(module_name: str):
    """Attempt to import a module, returning None if unavailable.

    Args:
        module_name: Fully qualified module name to import.

    Returns:
        module | None: The imported module or None if import fails.
    """
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


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
        enable_telemetry_panel: bool = False,
        telemetry_metrics: list[str] | None = None,
        telemetry_record: bool = False,
        telemetry_refresh_hz: float = 1.0,
        telemetry_pane_layout: str = "vertical_split",
        telemetry_decimation: int = 1,
    ) -> SingleAgentEnv:
        """Construct a robot environment with specified observation and recording configuration.

        Internal factory method used by public ergonomic factories (make_robot_env and
        make_image_robot_env). Selects the appropriate environment class based on the
        use_image_obs flag and applies all configuration settings.

        Args:
            config: RobotSimulationConfig instance defining physics, maps, and sensors.
            use_image_obs: If True, select image-capable environment; else standard lidar-only.
            peds_have_obstacle_forces: Enable pedestrian-robot interaction forces.
            reward_func: Custom reward function; falls back to internal default if None.
            debug: Enable visual debug features and view creation.
            recording_enabled: Master gate for video/JSONL recording even if options request it.
            record_video: Enable video recording to disk.
            video_path: Output path for recorded video files.
            video_fps: Target frames-per-second for video output.
            use_jsonl_recording: Enable per-episode JSONL metadata recording.
            recording_dir: Directory for JSONL episode outputs.
            suite_name: Metadata identifier for episode grouping.
            scenario_name: Scenario identifier for episode metadata.
            algorithm_name: Algorithm identifier for episode metadata.
            recording_seed: Deterministic seed for recorder's episode naming.
            enable_telemetry_panel: Enable docked telemetry visualization panel.
            telemetry_metrics: Metrics to display in the pane; defaults if None.
            telemetry_record: Persist telemetry to JSONL under artifact root.
            telemetry_refresh_hz: Target refresh rate for telemetry (Hz).
            telemetry_pane_layout: Pane docking layout (vertical_split or horizontal_split).
            telemetry_decimation: Decimation factor for telemetry persistence (>=1).

        Returns:
            SingleAgentEnv: Fully initialized robot environment instance.

        Raises:
            RuntimeError: If environment class import fails.
        """
        if config is None:
            config = ImageRobotConfig() if use_image_obs else RobotSimulationConfig()
        config.use_image_obs = use_image_obs
        config.peds_have_obstacle_forces = peds_have_obstacle_forces
        config.enable_telemetry_panel = enable_telemetry_panel
        config.telemetry_record = telemetry_record
        config.telemetry_refresh_hz = telemetry_refresh_hz
        config.telemetry_pane_layout = telemetry_pane_layout
        config.telemetry_decimation = telemetry_decimation
        if telemetry_metrics is not None:
            config.telemetry_metrics = list(telemetry_metrics)
        if use_image_obs:
            EnvCls = _load_robot_env_with_image()
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
        """Construct a pedestrian (adversarial) environment.

        Internal factory method used by make_pedestrian_env. Initializes an ego pedestrian
        agent navigating among crowds controlled by a provided robot policy.

        Args:
            robot_model: Trained policy or model providing robot actions in the scene.
            config: PedestrianSimulationConfig instance; defaults to standard if None.
            reward_func: Custom reward function for pedestrian agent; uses canonical
                simple_ped_reward if None.
            debug: Enable visual debug features and view creation.
            recording_enabled: Master gate for video recording.
            peds_have_obstacle_forces: Enable pedestrian-robot interaction forces.

        Returns:
            SingleAgentEnv: Initialized pedestrian environment for training/evaluation.
        """
        if config is None:
            config = PedestrianSimulationConfig()
        PedestrianEnv = _load_pedestrian_env()

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
        """Construct a multi-robot environment.

        Internal factory method used by make_multi_robot_env. Initializes a multi-agent
        environment where multiple robots navigate and interact in a shared scene.

        Args:
            config: MultiRobotConfig instance; defaults to standard if None.
            num_robots: Number of robot agents in the environment.
            reward_func: Custom reward function for agents; internal default if None.
            debug: Enable visual debug features and view creation.

        Returns:
            MultiAgentEnv: Initialized multi-agent environment instance.
        """
        if config is None:
            config = MultiRobotConfig()
        if config.num_robots != num_robots:
            config.num_robots = num_robots
        MultiRobotEnv = _load_multi_robot_env()

        return MultiRobotEnv(
            env_config=config,
            reward_func=reward_func,
            debug=debug,
            num_robots=num_robots,
        )  # type: ignore[return-value]


def _apply_render(mapped: dict[str, Any], render: RenderOptions | None):
    """Apply legacy render option overrides from mapped kwargs.

    Args:
        mapped: Dictionary of mapped legacy kwargs (modified in-place).
        render: Existing RenderOptions instance or None.

    Returns:
        RenderOptions | None: Updated or new RenderOptions if overrides found, else original.
    """
    if "render_options.max_fps_override" in mapped:
        ro = render or RenderOptions()
        ro.max_fps_override = mapped.pop("render_options.max_fps_override")
        return ro
    return render


def _apply_recording(mapped: dict[str, Any], rec: RecordingOptions | None):
    """Apply legacy recording option overrides from mapped kwargs.

    Args:
        mapped: Dictionary of mapped legacy kwargs (modified in-place).
        rec: Existing RecordingOptions instance or None.

    Returns:
        RecordingOptions | None: Updated or new RecordingOptions if overrides found, else original.
    """
    keys = ("recording_options.record", "recording_options.video_path")
    if any(k in mapped for k in keys):
        out = rec or RecordingOptions()
        if keys[0] in mapped:
            out.record = mapped.pop(keys[0])
        if keys[1] in mapped:
            out.video_path = mapped.pop(keys[1])
        return out
    return rec


def _validate_and_log_config(config: Any) -> None:
    """Validate configuration and log resolved config for reproducibility.

    Extracted helper to avoid duplication across factory functions (addresses
    CodeRabbit review feedback on T030/T031 duplication).
    """
    if config is None:
        return

    validate_config(config, strict=True)
    resolved = get_resolved_config_dict(config)
    # Log resolved config for reproducibility (T031)
    logger.info(
        "Resolved config: type={} backend={} sensors={}",
        type(config).__name__,
        resolved.get("backend", "fast-pysf"),
        len(resolved.get("sensors", [])),
    )
    logger.debug("Full resolved config: {}", resolved)


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

    Returns:
        Tuple of (render_options, recording_options, eff_record, eff_path, eff_fps).
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
    enable_telemetry_panel: bool = False,
    telemetry_metrics: list[str] | None = None,
    telemetry_record: bool = False,
    telemetry_refresh_hz: float = 1.0,
    telemetry_pane_layout: str = "vertical_split",
    telemetry_decimation: int = 1,
    **legacy_kwargs,
) -> SingleAgentEnv:
    """Create a robot environment with lidar-based observations (non-image).

    Primary ergonomic factory for standard robot navigation environments. Supports
    deterministic seeding, optional video recording, custom rewards, and live telemetry
    visualization. Configuration validation and logging ensure reproducibility.

    Args:
        config: Optional pre-constructed config; a default instance is created if None.
        seed: Deterministic seed (Python random, NumPy, PyTorch, hash seed). Stored on
            the returned env as ``applied_seed``.
        peds_have_obstacle_forces: Whether pedestrians perceive the robot as an obstacle
            (interaction forces enabled).
        reward_func: Optional custom reward function; falls back to internal simple reward
            with warning.
        debug: Enable debug/visual features (may trigger view creation when recording).
        recording_enabled: Master switch gating recording runtime even if options request
            it (feature flag style).
        record_video: Convenience flag; if True and no explicit RecordingOptions provided,
            one is synthesized.
        video_path: Convenience output path used if RecordingOptions absent or lacks path.
        video_fps: Convenience FPS; sets RenderOptions.max_fps_override if unset.
        render_options: Advanced rendering options; takes precedence over convenience flags.
        recording_options: Advanced recording options. For robot/image factories, convenience
            flag may upgrade ``record`` to True if conflicting (precedence rule #3).
        use_jsonl_recording: Enable JSONL episode recording (per-episode JSONL + metadata
            outputs) when recording is on.
        recording_dir: Directory where JSONL recorder stores per-episode files if enabled.
        suite_name: Suite identifier stored in JSONL metadata for downstream grouping.
        scenario_name: Scenario identifier stored in JSONL metadata.
        algorithm_name: Algorithm identifier stored in JSONL metadata.
        recording_seed: Optional stable seed used by the recorder to derive deterministic
            episode names.
        enable_telemetry_panel: When True, initialize docked telemetry pane configuration
            (charts blitted into SDL).
        telemetry_metrics: Metrics to render in the telemetry pane; defaults to core metrics
            when None.
        telemetry_record: Enable telemetry JSONL recording for the run (append-only under
            artifact root).
        telemetry_refresh_hz: Target refresh rate for telemetry sampling/chart updates (Hz).
        telemetry_pane_layout: Docking layout for the telemetry pane (vertical_split or
            horizontal_split).
        telemetry_decimation: Decimation factor applied when persisting/visualizing telemetry
            (>=1).
        **legacy_kwargs: Deprecated legacy surface (mapped via apply_legacy_kwargs). Unknown
            params rejected unless ROBOT_SF_FACTORY_LEGACY env var is truthy (permissive mode).

    Returns:
        SingleAgentEnv: Initialized robot environment instance with ``applied_seed`` attribute set.

    Note:
        Side-effects: seeds RNGs (idempotent for same seed), logs creation line.
        Performance: heavy image rendering imports are avoided along this path.
    """
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

    # Validate configuration and log resolved config (T030/T031)
    _validate_and_log_config(config)

    (
        render_options,
        recording_options,
        eff_record_video,
        eff_video_path,
        eff_video_fps,
    ) = _normalize_factory_inputs(
        record_video=record_video,
        video_path=video_path,
        video_fps=video_fps,
        render_options=render_options,
        recording_options=recording_options,
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
        enable_telemetry_panel=enable_telemetry_panel,
        telemetry_metrics=telemetry_metrics,
        telemetry_record=telemetry_record,
        telemetry_refresh_hz=telemetry_refresh_hz,
        telemetry_pane_layout=telemetry_pane_layout,
        telemetry_decimation=telemetry_decimation,
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

    Returns:
        Initialized SingleAgentEnv with image observation capabilities.
    """
    if legacy_kwargs:
        mapped, _warnings = apply_legacy_kwargs(legacy_kwargs, strict=True)
        render_options = _apply_render(mapped, render_options)
        recording_options = _apply_recording(mapped, recording_options)
    _apply_global_seed(seed)

    # Validate configuration and log resolved config (T030/T031)
    _validate_and_log_config(config)

    (
        render_options,
        recording_options,
        eff_record_video,
        eff_video_path,
        eff_video_fps,
    ) = _normalize_factory_inputs(
        record_video=record_video,
        video_path=video_path,
        video_fps=video_fps,
        render_options=render_options,
        recording_options=recording_options,
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

    Returns:
        Initialized SingleAgentEnv for adversarial pedestrian training.
    """
    # Capture explicit override intent BEFORE normalization mutates structures.
    if legacy_kwargs:
        mapped, _warnings = apply_legacy_kwargs(legacy_kwargs, strict=True)
        render_options = _apply_render(mapped, render_options)
        recording_options = _apply_recording(mapped, recording_options)

    _apply_global_seed(seed)

    # Validate configuration and log resolved config (T030/T031)
    _validate_and_log_config(config)

    _explicit_no_record = (
        recording_options is not None and recording_options.record is False and record_video
    )

    (
        render_options,
        recording_options,
        eff_record_video,
        _eff_video_path,
        _eff_video_fps,
    ) = _normalize_factory_inputs(
        record_video=record_video,
        video_path=video_path,
        video_fps=video_fps,
        render_options=render_options,
        recording_options=recording_options,
    )

    # Reinstate explicit opt-out if requested (override robot-style convenience precedence).
    if _explicit_no_record:
        eff_record_video = False
        if recording_options is not None:
            recording_options.record = False  # ensure consistency for any downstream logic

    if robot_model is None:
        # Maintain previous behavior (tests rely on automatic stub when not provided)
        robot_model = _load_stub_robot_model()()

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
    """Apply a deterministic seed to all common RNG sources for reproducibility.

    Ensures consistent behavior across Python's random module, NumPy, PyTorch (if
    available), and hash-based randomization. Gracefully handles optional dependencies
    to keep the lightweight factories accessible without unnecessary imports.

    Args:
        seed: Seed value to apply; if None, this is a no-op (seeds already random).

    Side Effects:
        - Sets Python ``random`` seed.
        - Sets NumPy random seed (if available).
        - Sets PyTorch manual seeds and CUDA seeds (if available).
        - Sets ``PYTHONHASHSEED`` environment variable.

    Note:
        Positioned at end of module to keep import block contiguous (PEP8).
        Silent failures for optional dependencies keep the factory lightweight.
    """
    if seed is None:
        return
    random.seed(seed)
    numpy_mod = _optional_import("numpy")
    if numpy_mod is not None:
        numpy_mod.random.seed(seed)
    torch_mod = _optional_import("torch")
    if torch_mod is not None:
        if hasattr(torch_mod, "manual_seed"):
            torch_mod.manual_seed(seed)
        if hasattr(torch_mod, "cuda") and hasattr(torch_mod.cuda, "manual_seed_all"):
            torch_mod.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_multi_robot_env(
    num_robots: int = 2,
    *,
    config: MultiRobotConfig | None = None,
    reward_func: Callable | None = None,
    debug: bool = False,
) -> MultiAgentEnv:
    """Create a multi-robot environment with independent agents.

    Factory for multi-agent coordination and training scenarios. Multiple robots
    navigate and interact in a shared environment, useful for decentralized policies
    and coordination algorithms.

    Parameters
    ----------
    num_robots : int
        Number of robot agents to spawn in the environment (default: 2).
    config : MultiRobotConfig | None
        Optional multi-robot configuration; default instance created if ``None``.
    reward_func : Callable | None
        Custom reward function applied to each agent; falls back to internal default.
    debug : bool
        Enable visual debug features.

    Returns
    -------
    MultiAgentEnv
        Configured multi-agent environment ready for training/evaluation.
    """
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
