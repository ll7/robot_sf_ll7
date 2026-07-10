"""Environment factory module (ergonomic public entry points).

Overview
--------
Provides explicit, typed factory functions for creating navigation, pedestrian,
and crowd-only simulation environments. Replaces legacy ad‑hoc kwargs surface
with structured option objects. Deterministic seeding added in T030.

Key Features
------------
* Explicit signatures: discoverable parameters; no ``**kwargs`` reliance for
    new behavior.
* Option normalization: boolean convenience flags (``record_video``,
    ``video_path``, ``video_fps``), JSONL metadata kwargs, and telemetry kwargs
    map into focused option objects with documented precedence rules.
* Determinism: optional ``seed`` seeds Python ``random``, NumPy, and PyTorch
    (if available) plus ``PYTHONHASHSEED``; applied before environment class
    construction. The applied seed is attached to the returned environment as
    ``applied_seed``.
* Pedestrian divergence: pedestrian factory honors an explicit
    ``RecordingOptions(record=False)`` even if ``record_video=True`` convenience
    flag is passed (opt‑out preserved). Robot/image factories flip to True in
    that scenario for user convenience.
* Crowd-only simulation: ``make_crowd_sim_env`` exposes robot-free Social Force
    pedestrian stepping with optional rendering/recording and no robot action input.
* Performance guard: lightweight import strategy and timing test (T015/T031)
    ensure mean creation time regression is limited to +5% vs baseline.

Precedence Summary
------------------
1. Explicit ``render_options`` / ``recording_options`` objects override boolean flags.
2. Robot/Image factories: ``record_video=True`` forces ``RecordingOptions.record=True``.
3. Pedestrian factory: preserves explicit opt‑out (``record=False``) even with ``record_video=True``.
4. ``video_fps`` populates ``RenderOptions.max_fps_override`` only when that field is unset.

This module contains no top‑level heavy imports beyond core types to minimize cold‑start
overhead (FR‑017). Rendering classes for image observations are imported lazily.
"""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from loguru import logger

# Hoisted imports (avoid per-call import overhead for performance regression guard)
try:  # pragma: no cover - import errors would surface in tests
    from robot_sf.gym_env.robot_env import RobotEnv  # type: ignore

    # Intentionally do NOT import robot_env_with_image here to avoid heavy first-call cost.
    RobotEnvWithImage = None  # type: ignore
except ImportError:
    # Defer errors until factory invocation (lazy fallback)
    RobotEnv = None  # type: ignore
    RobotEnvWithImage = None  # type: ignore

from robot_sf.gym_env.config_validation import get_resolved_config_dict, validate_config
from robot_sf.gym_env.options import (
    JsonlRecordingOptions,
    RecordingOptions,
    RenderOptions,
    TelemetryOptions,
)
from robot_sf.gym_env.reward import (
    build_reward_curriculum_function,
    build_reward_function,
    simple_ped_reward,
)
from robot_sf.gym_env.unified_config import (
    ImageRobotConfig,
    MultiRobotConfig,
    PedestrianSimulationConfig,
    RobotSimulationConfig,
    sync_pedestrian_obstacle_force_alias,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from robot_sf.gym_env.abstract_envs import MultiAgentEnv, SingleAgentEnv
    from robot_sf.gym_env.crowd_sim_env import CrowdSimEnv, CrowdSimulationConfig


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


def _load_crowd_sim_env():
    """Lazy-load the crowd-only simulation environment class.

    Returns:
        tuple[type, type]: ``CrowdSimEnv`` and ``CrowdSimulationConfig`` classes.
    """
    module = importlib.import_module("robot_sf.gym_env.crowd_sim_env")
    return module.CrowdSimEnv, module.CrowdSimulationConfig


def _load_stub_robot_model():
    """Lazy-load the runtime fallback stub robot model class.

    Returns a minimal robot model that produces zero actions, enabling pedestrian
    environment initialization in backward-compatible runtime paths without a trained
    policy.

    Returns:
        type: StubRobotModel class (callable) providing zero-action fallback.

    Raises:
        ModuleNotFoundError: If ``robot_sf.gym_env._stub_robot_model`` is unavailable.
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
    except ImportError:
        return None


class EnvironmentFactory:
    """Internal helpers to construct concrete environments (not exported)."""

    @staticmethod
    def create_robot_env(  # noqa: PLR0913
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
        jsonl_recording_options: JsonlRecordingOptions | None = None,
        telemetry_options: TelemetryOptions | None = None,
        asymmetric_critic: bool = False,
    ) -> SingleAgentEnv:
        """Construct a robot environment with specified observation and recording configuration.

        Internal factory method used by public ergonomic factories (make_robot_env and
        make_image_robot_env). Selects the appropriate environment class based on the
        use_image_obs flag and applies all configuration settings.

        Args:
            config: RobotSimulationConfig instance defining physics, maps, and sensors.
            use_image_obs: If True, select image-capable environment; else standard lidar-only.
            peds_have_obstacle_forces: (Deprecated) Controls static obstacle forces for pedestrians.
            reward_func: Custom reward function; falls back to internal default if None.
            debug: Enable visual debug features and view creation.
            recording_enabled: Master gate for video/JSONL recording even if options request it.
            record_video: Enable video recording to disk.
            video_path: Output path for recorded video files.
            video_fps: Target frames-per-second for video output.
            jsonl_recording_options: Per-episode JSONL recording metadata.
            telemetry_options: Live telemetry panel and telemetry recording options.
            asymmetric_critic: When True, add a critic-only privileged state vector to the
                observation space for asymmetric actor-critic training.

        Returns:
            SingleAgentEnv: Fully initialized robot environment instance.

        Raises:
            RuntimeError: If environment class import fails.
        """
        if config is None:
            config = ImageRobotConfig() if use_image_obs else RobotSimulationConfig()
        config.use_image_obs = use_image_obs
        legacy_override = None if peds_have_obstacle_forces is True else peds_have_obstacle_forces
        sync_pedestrian_obstacle_force_alias(config, legacy_override)
        jsonl_recording_options = jsonl_recording_options or JsonlRecordingOptions()
        jsonl_recording_options.validate()
        if telemetry_options is None:
            telemetry_options = TelemetryOptions()
        else:
            telemetry_options = _copy_telemetry_options(telemetry_options)
        telemetry_options.validate()
        config.enable_telemetry_panel = telemetry_options.enable_panel
        config.telemetry_record = telemetry_options.record
        config.telemetry_refresh_hz = telemetry_options.refresh_hz
        config.telemetry_pane_layout = telemetry_options.pane_layout
        config.telemetry_decimation = telemetry_options.decimation
        config.telemetry_metrics = list(telemetry_options.metrics)
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
            use_jsonl_recording=jsonl_recording_options.enabled,
            recording_dir=jsonl_recording_options.recording_dir,
            suite_name=jsonl_recording_options.suite_name,
            scenario_name=jsonl_recording_options.scenario_name,
            algorithm_name=jsonl_recording_options.algorithm_name,
            recording_seed=jsonl_recording_options.recording_seed,
            asymmetric_critic=asymmetric_critic,
        )  # type: ignore[return-value]

    @staticmethod
    def create_pedestrian_env(  # noqa: PLR0913
        robot_model,
        config: PedestrianSimulationConfig | None = None,
        reward_func: Callable[[dict], float] | None = None,
        debug: bool = False,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float | None = None,
        peds_have_obstacle_forces: bool = True,
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
            record_video: Enable video recording to disk.
            video_path: Output path for recorded video files.
            video_fps: Target frames-per-second for video output.
            peds_have_obstacle_forces: (Deprecated) Controls static obstacle forces for pedestrians.

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

        legacy_override = None if peds_have_obstacle_forces is True else peds_have_obstacle_forces
        sync_pedestrian_obstacle_force_alias(config, legacy_override)

        return PedestrianEnv(
            env_config=config,  # type: ignore[arg-type]
            robot_model=robot_model,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )  # type: ignore[return-value]

    @staticmethod
    def create_multi_robot_env(  # noqa: PLR0913
        config: MultiRobotConfig | None = None,
        *,
        num_robots: int,
        seed: int | None = None,
        reward_func: Callable | None,
        debug: bool,
        recording_enabled: bool = False,
        record_video: bool = False,
        video_path: str | None = None,
        video_fps: float | None = None,
    ) -> MultiAgentEnv:
        """Construct a multi-robot environment.

        Internal factory method used by make_multi_robot_env. Initializes a multi-agent
        environment where multiple robots navigate and interact in a shared scene.

        Args:
            config: MultiRobotConfig instance; defaults to standard if None.
            num_robots: Number of robot agents in the environment.
            seed: Deterministic seed applied before environment construction.
            reward_func: Custom reward function for agents; internal default if None.
            debug: Enable visual debug features and view creation.
            recording_enabled: Master gate for per-robot recording.
            record_video: Enable per-robot video recording to disk.
            video_path: Base output path for recorded videos.
            video_fps: Target frames-per-second for video output.

        Returns:
            MultiAgentEnv: Initialized multi-agent environment instance.
        """
        if config is None:
            config = MultiRobotConfig()
        if config.num_robots != num_robots:
            config.num_robots = num_robots
        _apply_global_seed(seed)
        MultiRobotEnv = _load_multi_robot_env()

        env = MultiRobotEnv(
            env_config=config,
            reward_func=reward_func,
            debug=debug,
            num_robots=num_robots,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
        )  # type: ignore[return-value]
        env.applied_seed = seed
        return env  # type: ignore[return-value]


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
    logger.debug(
        "Resolved config: type={} backend={} sensors={}",
        type(config).__name__,
        resolved.get("backend", "fast-pysf"),
        len(resolved.get("sensors", [])),
    )


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


def _normalize_jsonl_recording_options(
    *,
    jsonl_recording_options: JsonlRecordingOptions | None,
    use_jsonl_recording: bool,
    recording_dir: str,
    suite_name: str,
    scenario_name: str,
    algorithm_name: str,
    recording_seed: int | None,
) -> JsonlRecordingOptions:
    """Return canonical JSONL recording options from object or compatibility kwargs."""
    if jsonl_recording_options is None:
        jsonl_recording_options = JsonlRecordingOptions(
            enabled=use_jsonl_recording,
            recording_dir=recording_dir,
            suite_name=suite_name,
            scenario_name=scenario_name,
            algorithm_name=algorithm_name,
            recording_seed=recording_seed,
        )
    jsonl_recording_options.validate()
    return jsonl_recording_options


def _copy_telemetry_options(telemetry_options: TelemetryOptions) -> TelemetryOptions:
    """Return a shallow telemetry option copy with caller-owned metrics detached."""
    return replace(telemetry_options, metrics=list(telemetry_options.metrics or []))


def _normalize_telemetry_options(
    *,
    telemetry_options: TelemetryOptions | None,
    enable_telemetry_panel: bool,
    telemetry_metrics: list[str] | None,
    telemetry_record: bool,
    telemetry_refresh_hz: float,
    telemetry_pane_layout: str,
    telemetry_decimation: int,
) -> TelemetryOptions:
    """Return canonical telemetry options from object or compatibility kwargs."""
    if telemetry_options is None:
        telemetry_options = TelemetryOptions(
            enable_panel=enable_telemetry_panel,
            metrics=list(telemetry_metrics) if telemetry_metrics is not None else [],
            record=telemetry_record,
            refresh_hz=telemetry_refresh_hz,
            pane_layout=telemetry_pane_layout,
            decimation=telemetry_decimation,
        )
    else:
        telemetry_options = _copy_telemetry_options(telemetry_options)
    telemetry_options.validate()
    return telemetry_options


def make_robot_env(  # noqa: PLR0913
    config: RobotSimulationConfig | None = None,
    *,
    seed: int | None = None,
    peds_have_obstacle_forces: bool = True,
    reward_func: Callable | None = None,
    reward_name: str | None = None,
    reward_kwargs: Mapping[str, object] | None = None,
    reward_curriculum: Mapping[str, object] | None = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: str | None = None,
    video_fps: float | None = None,
    render_options: RenderOptions | None = None,
    recording_options: RecordingOptions | None = None,
    jsonl_recording_options: JsonlRecordingOptions | None = None,
    use_jsonl_recording: bool = False,
    recording_dir: str = "recordings",
    suite_name: str = "robot_sim",
    scenario_name: str = "default",
    algorithm_name: str = "manual",
    recording_seed: int | None = None,
    telemetry_options: TelemetryOptions | None = None,
    enable_telemetry_panel: bool = False,
    telemetry_metrics: list[str] | None = None,
    telemetry_record: bool = False,
    telemetry_refresh_hz: float = 1.0,
    telemetry_pane_layout: str = "vertical_split",
    telemetry_decimation: int = 1,
    asymmetric_critic: bool = False,
) -> SingleAgentEnv:
    """Create a robot environment with lidar-based observations (non-image).

    Primary ergonomic factory for standard robot navigation environments. Supports
    deterministic seeding, optional video recording, custom rewards, and live telemetry
    visualization. Configuration validation and logging ensure reproducibility.

    Args:
        config: Optional pre-constructed config; a default instance is created if None.
        seed: Deterministic seed (Python random, NumPy, PyTorch, hash seed). Stored on
            the returned env as ``applied_seed``.
        peds_have_obstacle_forces: Deprecated. Controls static obstacle forces for pedestrians.
            Use ``config.peds_have_static_obstacle_forces`` (obstacle forces) and
            ``config.peds_have_robot_repulsion`` (robot repulsion) going forward.
        reward_func: Optional custom reward function; when omitted, a named profile
            is resolved via ``reward_name`` (default: ``route_completion_v2``).
        reward_name: Optional named reward preset (`route_completion_v2`, `social_quality_v1`,
            `simple`, `punish_action`, `snqi_step`, `alyassi`).
            Ignored when ``reward_func`` is provided.
        reward_kwargs: Optional keyword arguments passed to the named reward preset builder.
        reward_curriculum: Optional staged reward curriculum. When provided, the reward callable
            is wrapped so stage selection advances after terminal episodes.
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
        jsonl_recording_options: Advanced per-episode JSONL recording metadata; when provided,
            it takes precedence over the flat JSONL compatibility kwargs below.
        use_jsonl_recording: Enable JSONL episode recording (per-episode JSONL + metadata
            outputs) when recording is on.
        recording_dir: Directory where JSONL recorder stores per-episode files if enabled.
        suite_name: Suite identifier stored in JSONL metadata for downstream grouping.
        scenario_name: Scenario identifier stored in JSONL metadata.
        algorithm_name: Algorithm identifier stored in JSONL metadata.
        recording_seed: Optional stable seed used by the recorder to derive deterministic
            episode names.
        telemetry_options: Advanced telemetry panel/recording options; when provided, it takes
            precedence over the flat telemetry compatibility kwargs below.
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
        asymmetric_critic: When True, add a critic-only privileged state vector to the
            observation space for asymmetric actor-critic training.

    Returns:
        SingleAgentEnv: Initialized robot environment instance with ``applied_seed`` attribute set.

    Note:
        Side-effects: seeds RNGs (idempotent for same seed), logs creation line.
        Performance: heavy image rendering imports are avoided along this path.
    """
    if reward_func is None:
        reward_name = reward_name or "route_completion_v2"
        if reward_curriculum is not None:
            reward_func = build_reward_curriculum_function(
                reward_curriculum,
                default_reward_name=reward_name,
                default_reward_kwargs=reward_kwargs,
            )
        else:
            reward_func = build_reward_function(reward_name, reward_kwargs=reward_kwargs)

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
    jsonl_recording_options = _normalize_jsonl_recording_options(
        jsonl_recording_options=jsonl_recording_options,
        use_jsonl_recording=use_jsonl_recording,
        recording_dir=recording_dir,
        suite_name=suite_name,
        scenario_name=scenario_name,
        algorithm_name=algorithm_name,
        recording_seed=recording_seed,
    )
    telemetry_options = _normalize_telemetry_options(
        telemetry_options=telemetry_options,
        enable_telemetry_panel=enable_telemetry_panel,
        telemetry_metrics=telemetry_metrics,
        telemetry_record=telemetry_record,
        telemetry_refresh_hz=telemetry_refresh_hz,
        telemetry_pane_layout=telemetry_pane_layout,
        telemetry_decimation=telemetry_decimation,
    )
    logger.debug(
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
        jsonl_recording_options=jsonl_recording_options,
        telemetry_options=telemetry_options,
        asymmetric_critic=asymmetric_critic,
    )
    env.applied_seed = seed
    return env


def make_image_robot_env(  # noqa: PLR0913
    config: ImageRobotConfig | None = None,
    *,
    seed: int | None = None,
    peds_have_obstacle_forces: bool = True,
    reward_func: Callable | None = None,
    reward_name: str | None = None,
    reward_kwargs: Mapping[str, object] | None = None,
    reward_curriculum: Mapping[str, object] | None = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: str | None = None,
    video_fps: float | None = None,
    render_options: RenderOptions | None = None,
    recording_options: RecordingOptions | None = None,
    jsonl_recording_options: JsonlRecordingOptions | None = None,
    use_jsonl_recording: bool = False,
    recording_dir: str = "recordings",
    suite_name: str = "robot_sim",
    scenario_name: str = "default",
    algorithm_name: str = "manual",
    recording_seed: int | None = None,
    asymmetric_critic: bool = False,
) -> SingleAgentEnv:
    """Create a robot environment with image observations.

    Mirrors :func:`make_robot_env` adding an internal switch to select the image-capable
    environment implementation. All parameter semantics and precedence are identical.
    Lazy import defers expensive view initialization until first creation.

    Returns:
        Initialized SingleAgentEnv with image observation capabilities.
    """
    if reward_func is None:
        reward_name = reward_name or "route_completion_v2"
        if reward_curriculum is not None:
            reward_func = build_reward_curriculum_function(
                reward_curriculum,
                default_reward_name=reward_name,
                default_reward_kwargs=reward_kwargs,
            )
        else:
            reward_func = build_reward_function(reward_name, reward_kwargs=reward_kwargs)
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
    jsonl_recording_options = _normalize_jsonl_recording_options(
        jsonl_recording_options=jsonl_recording_options,
        use_jsonl_recording=use_jsonl_recording,
        recording_dir=recording_dir,
        suite_name=suite_name,
        scenario_name=scenario_name,
        algorithm_name=algorithm_name,
        recording_seed=recording_seed,
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
        jsonl_recording_options=jsonl_recording_options,
        asymmetric_critic=asymmetric_critic,
    )
    env.applied_seed = seed
    return env


def make_pedestrian_env(  # noqa: PLR0913
    config: PedestrianSimulationConfig | None = None,
    *,
    seed: int | None = None,
    robot_model=None,
    reward_func: Callable | None = None,
    debug: bool = False,
    recording_enabled: bool = False,
    peds_have_obstacle_forces: bool = True,
    record_video: bool = False,
    video_path: str | None = None,
    video_fps: float | None = None,
    render_options: RenderOptions | None = None,
    recording_options: RecordingOptions | None = None,
) -> SingleAgentEnv:
    """Create a pedestrian (adversarial) environment.

    Differences vs. :func:`make_robot_env` / :func:`make_image_robot_env`:
    * Honors explicit recording opt-out (``RecordingOptions(record=False)``) even if
      ``record_video=True`` (no implicit flip). Required by T012 test.
    * May auto-enable ``debug`` when effective recording is active to ensure frames
      are produced.
    * Injects the shared runtime fallback stub robot model when ``robot_model`` is
      ``None`` for backward compatibility and low-friction debugging.

    Parameters follow the same semantics; additional ones:
    robot_model : Any | None
        Trained policy / model providing robot actions. A lightweight stub is injected if
        absent so tests and simple demos can still run.
    peds_have_obstacle_forces : bool
        Deprecated. Controls static obstacle forces for pedestrians (not robot repulsion).

    Returns:
        Initialized SingleAgentEnv for adversarial pedestrian training.
    """
    # Capture explicit override intent BEFORE normalization mutates structures.
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
        eff_video_path,
        eff_video_fps,
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
        record_video=eff_record_video,
        video_path=eff_video_path,
        video_fps=eff_video_fps,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
    )
    env.applied_seed = seed
    return env


def make_crowd_sim_env(  # noqa: PLR0913
    config: CrowdSimulationConfig | None = None,
    *,
    seed: int | None = None,
    map_id: str | None = None,
    peds_have_obstacle_forces: bool | None = None,
    render_mode: str | None = None,
    recording_enabled: bool | None = None,
    recording_dir: str | None = None,
    recording_path: str | None = None,
    video_path: str | None = None,
    video_fps: float | None = None,
) -> CrowdSimEnv:
    """Create a robot-free crowd simulation environment.

    The returned env exposes ``reset``/``step`` for fast pedestrian-only Social Force
    simulation. ``step(action=None)`` is accepted for Gymnasium compatibility; non-``None``
    actions are ignored with a warning because there is no robot or ego policy.

    Returns:
        CrowdSimEnv: Configured crowd-only Gymnasium environment.
    """
    _apply_global_seed(seed)
    CrowdSimEnv, CrowdSimulationConfig = _load_crowd_sim_env()
    if config is None:
        config = CrowdSimulationConfig()
    if map_id is not None:
        config.map_id = map_id
    if peds_have_obstacle_forces is not None:
        config.peds_have_obstacle_forces = peds_have_obstacle_forces
    if render_mode is not None:
        config.render_mode = render_mode
    if recording_enabled is not None:
        config.recording_enabled = recording_enabled
    if recording_dir is not None:
        config.recording_dir = recording_dir
    if recording_path is not None:
        config.recording_path = recording_path
    if video_path is not None:
        config.video_path = video_path
    if video_fps is not None:
        config.video_fps = video_fps
    env = CrowdSimEnv(config)
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


def make_multi_robot_env(  # noqa: PLR0913
    num_robots: int = 2,
    *,
    config: MultiRobotConfig | None = None,
    seed: int | None = None,
    reward_func: Callable | None = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: str | None = None,
    video_fps: float | None = None,
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
    seed : int | None
        Deterministic seed (Python random, NumPy, PyTorch, hash seed). Stored on
        the returned environment and passed through reset metadata.
    reward_func : Callable | None
        Custom reward function applied to each agent; falls back to internal default.
    debug : bool
        Enable visual debug features.
    recording_enabled : bool
        Master switch that enables simulation state recording hooks.
    record_video : bool
        Enable per-robot video recording.
    video_path : str | None
        Optional base output path. Multi-robot env appends a robot suffix per view.
    video_fps : float | None
        Optional FPS for recorded video files.

    Returns
    -------
    MultiAgentEnv
        Configured multi-agent environment ready for training/evaluation.
    """
    return EnvironmentFactory.create_multi_robot_env(
        config=config,
        num_robots=num_robots,
        seed=seed,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        record_video=record_video,
        video_path=video_path,
        video_fps=video_fps,
    )


def _make_multi_robot_env_future(*_args, **_kwargs):  # pragma: no cover
    """Future ergonomic multi-robot factory placeholder (T013)."""
    raise NotImplementedError(
        "Future multi-robot ergonomic factory not yet implemented; see task T013.",
    )
