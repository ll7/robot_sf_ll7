"""Environment factory with ergonomic option normalization.

Maintains backward compatibility with legacy kwargs via `_factory_compat` while
favoring structured `RenderOptions` / `RecordingOptions` objects.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from loguru import logger

# Hoisted imports (avoid per-call import overhead for performance regression guard)
try:  # pragma: no cover - import errors would surface in tests
    from robot_sf.gym_env.robot_env import RobotEnv  # type: ignore

    # Intentionally do NOT import robot_env_with_image here to avoid heavy first-call cost.
    RobotEnvWithImage = None  # type: ignore
except Exception:  # noqa: BLE001
    # Defer errors until factory invocation (lazy fallback)
    RobotEnv = None  # type: ignore
    RobotEnvWithImage = None  # type: ignore

from robot_sf.gym_env._factory_compat import LEGACY_PERMISSIVE_ENV, apply_legacy_kwargs
from robot_sf.gym_env.abstract_envs import MultiAgentEnv, SingleAgentEnv
from robot_sf.gym_env.options import RecordingOptions, RenderOptions
from robot_sf.gym_env.unified_config import (
    ImageRobotConfig,
    MultiRobotConfig,
    PedestrianSimulationConfig,
    RobotSimulationConfig,
)


class EnvironmentFactory:
    """Internal helpers to construct concrete environments (not exported)."""

    @staticmethod
    def create_robot_env(
        config: Optional[RobotSimulationConfig],
        *,
        use_image_obs: bool,
        peds_have_obstacle_forces: bool,
        reward_func: Optional[Callable],
        debug: bool,
        recording_enabled: bool,
        record_video: bool,
        video_path: Optional[str],
        video_fps: Optional[float],
    ) -> SingleAgentEnv:
        if config is None:
            config = ImageRobotConfig() if use_image_obs else RobotSimulationConfig()
        config.use_image_obs = use_image_obs
        config.peds_have_obstacle_forces = peds_have_obstacle_forces
        if use_image_obs:
            from robot_sf.gym_env.robot_env_with_image import (  # type: ignore
                RobotEnvWithImage as EnvCls,  # noqa: F401
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
        )

    @staticmethod
    def create_pedestrian_env(
        config: Optional[PedestrianSimulationConfig],
        *,
        robot_model,
        reward_func: Optional[Callable],
        debug: bool,
        recording_enabled: bool,
        peds_have_obstacle_forces: bool,
    ) -> SingleAgentEnv:
        if config is None:
            config = PedestrianSimulationConfig()
        from robot_sf.gym_env.pedestrian_env import PedestrianEnv

        return PedestrianEnv(
            env_config=config,
            robot_model=robot_model,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
        )

    @staticmethod
    def create_multi_robot_env(
        config: Optional[MultiRobotConfig],
        *,
        num_robots: int,
        reward_func: Optional[Callable],
        debug: bool,
    ) -> MultiAgentEnv:
        if config is None:
            config = MultiRobotConfig()
        from robot_sf.gym_env.multi_robot_env import MultiRobotEnv

        return MultiRobotEnv(
            env_config=config, reward_func=reward_func, debug=debug, num_robots=num_robots
        )


def _apply_render(mapped: dict[str, Any], render: Optional[RenderOptions]):
    if "render_options.max_fps_override" in mapped:
        ro = render or RenderOptions()
        ro.max_fps_override = mapped.pop("render_options.max_fps_override")
        return ro
    return render


def _apply_recording(mapped: dict[str, Any], rec: Optional[RecordingOptions]):
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
    video_path: Optional[str],
    video_fps: Optional[float],
    render_options: Optional[RenderOptions],
    recording_options: Optional[RecordingOptions],
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
            "record_video=True but RecordingOptions.record is False; enabling recording (boolean convenience precedence)."
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
    # Public factory functions follow.


def make_robot_env(
    config: Optional[RobotSimulationConfig] = None,
    *,
    seed: Optional[int] = None,
    peds_have_obstacle_forces: bool = False,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
    render_options: Optional[RenderOptions] = None,
    recording_options: Optional[RecordingOptions] = None,
    **legacy_kwargs,
) -> SingleAgentEnv:
    """Create a standard robot environment (non-image observations).

    Parameters
    ----------
    seed : int | None
        Deterministic seed applied to Python ``random``, NumPy, and PyTorch (if present).
        Stored on the returned environment instance as ``applied_seed`` for introspection.
    legacy_kwargs : dict
        Deprecated parameters (legacy surface). They are mapped or rejected via
        :func:`apply_legacy_kwargs`. Unknown parameters raise unless the environment
        variable ``{env}`` is set to a truthy value (permissive mode).
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
    )
    setattr(env, "applied_seed", seed)
    return env


def make_image_robot_env(
    config: Optional[ImageRobotConfig] = None,
    *,
    seed: Optional[int] = None,
    peds_have_obstacle_forces: bool = False,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
    render_options: Optional[RenderOptions] = None,
    recording_options: Optional[RecordingOptions] = None,
    **legacy_kwargs,
) -> SingleAgentEnv:
    """Create a robot environment with image observations (see `make_robot_env`)."""
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
    )
    setattr(env, "applied_seed", seed)
    return env


def make_pedestrian_env(
    config: Optional[PedestrianSimulationConfig] = None,
    *,
    seed: Optional[int] = None,
    robot_model=None,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    peds_have_obstacle_forces: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
    render_options: Optional[RenderOptions] = None,
    recording_options: Optional[RecordingOptions] = None,
    **legacy_kwargs,
) -> SingleAgentEnv:
    """Create a pedestrian (adversarial) environment.

    Signature ordering intentionally places `peds_have_obstacle_forces` before the
    recording / convenience flags to satisfy snapshot expectations captured in
    tests/factories/test_current_factory_signatures.py (T011 guard).

    Precedence rule difference vs. robot factories:
      - If users provide `RecordingOptions(record=False)` together with
        `record_video=True`, the pedestrian factory respects the explicit
        opt-out (no implicit flip to True). This is required by
        `test_pedestrian_factory_explicit_options_override` (T012).
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
        class _StubRobotModel:  # pragma: no cover - trivial
            def predict(self, _obs, **_ignored):  # noqa: D401
                import numpy as _np  # local import to avoid global dependency

                return _np.zeros(2, dtype=float), None

        robot_model = _StubRobotModel()

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
    setattr(env, "applied_seed", seed)
    return env


def _apply_global_seed(seed: Optional[int]) -> None:
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
        import numpy as _np  # type: ignore

        _np.random.seed(seed)
    except Exception:  # noqa: BLE001
        pass
    try:  # PyTorch optional
        import torch as _torch  # type: ignore

        if hasattr(_torch, "manual_seed"):
            _torch.manual_seed(seed)
        if hasattr(_torch, "cuda") and hasattr(_torch.cuda, "manual_seed_all"):
            _torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass
    _os.environ["PYTHONHASHSEED"] = str(seed)


def make_multi_robot_env(
    num_robots: int = 2,
    *,
    config: Optional[MultiRobotConfig] = None,
    reward_func: Optional[Callable] = None,
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
        "Future multi-robot ergonomic factory not yet implemented; see task T013."
    )
