"""Demonstration of ergonomic environment factory options (Feature 130).

Shows legacy convenience flags alongside structured RenderOptions / RecordingOptions.
Run:
    uv run python examples/demo_factory_options.py
"""

from __future__ import annotations

from loguru import logger

from robot_sf.gym_env.environment_factory import (
    RecordingOptions,
    RenderOptions,
    make_image_robot_env,
    make_robot_env,
)


def demo_minimal():
    env = make_robot_env()
    env.reset()
    env.close()
    logger.info("Minimal env created and closed.")


def demo_convenience_recording(tmp_path: str = "./results"):
    env = make_robot_env(record_video=True, video_path=f"{tmp_path}/episode.mp4")
    env.reset()
    env.close()
    logger.info("Convenience recording env created.")


def demo_structured():
    render_opts = RenderOptions(max_fps_override=20)
    rec_opts = RecordingOptions(record=True, video_path="results/structured_episode.mp4")
    env = make_robot_env(render_options=render_opts, recording_options=rec_opts, debug=True)
    env.reset()
    env.close()
    logger.info("Structured options env created.")


def demo_image():
    img_env = make_image_robot_env(render_options=RenderOptions(max_fps_override=15))
    img_env.reset()
    img_env.close()
    logger.info("Image observation env created.")


if __name__ == "__main__":  # pragma: no cover
    demo_minimal()
    demo_convenience_recording()
    demo_structured()
    demo_image()
    logger.info("All demos complete.")
