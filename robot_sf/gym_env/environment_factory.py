"""Environment factory with ergonomic option normalization.

Maintains backward compatibility with legacy kwargs via `_factory_compat` while
favoring structured `RenderOptions` / `RecordingOptions` objects.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from loguru import logger

from robot_sf.gym_env._factory_compat import apply_legacy_kwargs
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
        **kwargs: Any,
    ) -> SingleAgentEnv:
        if config is None:
            config = ImageRobotConfig() if use_image_obs else RobotSimulationConfig()
        config.use_image_obs = use_image_obs
        config.peds_have_obstacle_forces = peds_have_obstacle_forces
        if use_image_obs:
            from robot_sf.gym_env.robot_env_with_image import RobotEnvWithImage as EnvCls
        else:
            from robot_sf.gym_env.robot_env import RobotEnv as EnvCls
        return EnvCls(
            env_config=config,
            reward_func=reward_func,
            debug=debug,
            recording_enabled=recording_enabled,
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            peds_have_obstacle_forces=peds_have_obstacle_forces,
            **kwargs,
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
        **kwargs: Any,
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
            **kwargs,
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
    kwargs: dict[str, Any],
):
    mapped, _warns = apply_legacy_kwargs(kwargs, strict=True) if kwargs else ({}, [])
    render_options = _apply_render(mapped, render_options)
    recording_options = _apply_recording(mapped, recording_options)
    if recording_options is None and (record_video or video_path):
        recording_options = RecordingOptions.from_bool_and_path(record_video, video_path, None)
    if video_fps is not None and (
        render_options is None or render_options.max_fps_override is None
    ):
        render_options = render_options or RenderOptions()
        render_options.max_fps_override = int(video_fps)
    if recording_options and record_video and not recording_options.record:
        logger.warning(
            "record_video=True but RecordingOptions.record is False; enabling recording (precedence to boolean convenience)."
        )
        recording_options.record = True
    eff_record = record_video if recording_options is None else recording_options.record
    eff_path = video_path if recording_options is None else recording_options.video_path
    eff_fps = (
        video_fps
        if (render_options is None or render_options.max_fps_override is None)
        else float(render_options.max_fps_override)
    )
    return render_options, recording_options, eff_record, eff_path, eff_fps, mapped
    # Public factory functions follow.


def make_robot_env(
    config: Optional[RobotSimulationConfig] = None,
    *,
    peds_have_obstacle_forces: bool = False,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
    render_options: Optional[RenderOptions] = None,
    recording_options: Optional[RecordingOptions] = None,
    **kwargs: Any,
) -> SingleAgentEnv:
    """Create a standard robot environment (non-image observations)."""
    render_options, recording_options, eff_record_video, eff_video_path, eff_video_fps, mapped = (
        _normalize_factory_inputs(
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            render_options=render_options,
            recording_options=recording_options,
            kwargs=kwargs,
        )
    )
    logger.info(
        "Creating robot env debug={debug} record_video={record} video_path={path} fps={fps}",
        debug=debug,
        record=eff_record_video,
        path=eff_video_path,
        fps=eff_video_fps,
    )
    return EnvironmentFactory.create_robot_env(
        config=config,
        use_image_obs=False,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        record_video=eff_record_video,
        video_path=eff_video_path,
        video_fps=eff_video_fps,
        **mapped,
    )


def make_image_robot_env(
    config: Optional[ImageRobotConfig] = None,
    *,
    peds_have_obstacle_forces: bool = False,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
    render_options: Optional[RenderOptions] = None,
    recording_options: Optional[RecordingOptions] = None,
    **kwargs: Any,
) -> SingleAgentEnv:
    """Create a robot environment with image observations."""
    render_options, recording_options, eff_record_video, eff_video_path, eff_video_fps, mapped = (
        _normalize_factory_inputs(
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            render_options=render_options,
            recording_options=recording_options,
            kwargs=kwargs,
        )
    )
    logger.info(
        "Creating image robot env debug={debug} record_video={record} video_path={path} fps={fps}",
        debug=debug,
        record=eff_record_video,
        path=eff_video_path,
        fps=eff_video_fps,
    )
    return EnvironmentFactory.create_robot_env(
        config=config,
        use_image_obs=True,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        record_video=eff_record_video,
        video_path=eff_video_path,
        video_fps=eff_video_fps,
        **mapped,
    )


def make_pedestrian_env(
    config: Optional[PedestrianSimulationConfig] = None,
    *,
    robot_model=None,
    reward_func: Optional[Callable] = None,
    debug: bool = False,
    recording_enabled: bool = False,
    record_video: bool = False,
    video_path: Optional[str] = None,
    video_fps: Optional[float] = None,
    render_options: Optional[RenderOptions] = None,
    recording_options: Optional[RecordingOptions] = None,
    peds_have_obstacle_forces: bool = False,
    **kwargs: Any,
) -> SingleAgentEnv:
    """Create a pedestrian (adversarial) environment."""
    render_options, recording_options, eff_record_video, eff_video_path, eff_video_fps, mapped = (
        _normalize_factory_inputs(
            record_video=record_video,
            video_path=video_path,
            video_fps=video_fps,
            render_options=render_options,
            recording_options=recording_options,
            kwargs=kwargs,
        )
    )
    logger.info(
        "Creating pedestrian env debug={debug} record_video={record} video_path={path} fps={fps} robot_model={has_model}",
        debug=debug,
        record=eff_record_video,
        path=eff_video_path,
        fps=eff_video_fps,
        has_model=robot_model is not None,
    )
    return EnvironmentFactory.create_pedestrian_env(
        config=config,
        robot_model=robot_model,
        reward_func=reward_func,
        debug=debug,
        recording_enabled=recording_enabled,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
        **mapped,
    )


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
