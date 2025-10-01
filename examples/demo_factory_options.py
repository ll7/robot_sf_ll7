"""Demonstration of ergonomic environment factory options (Feature 130).

Shows legacy convenience flags alongside structured RenderOptions / RecordingOptions.
Refactored to use helper catalog utilities for common setup patterns.

Run:
    uv run python examples/demo_factory_options.py
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from robot_sf.benchmark.helper_catalog import prepare_classic_env
from robot_sf.gym_env.environment_factory import (
    RecordingOptions,
    RenderOptions,
    make_image_robot_env,
    make_robot_env,
)
from robot_sf.render.helper_catalog import ensure_output_dir


def demo_minimal():
    # Use helper catalog for basic environment setup
    env, _ = prepare_classic_env()
    env.reset()
    env.close()
    logger.info("Minimal env created using helper catalog and closed.")


def demo_convenience_recording(tmp_path: str = "./results"):
    # Use render helper to ensure output directory exists
    output_dir = ensure_output_dir(Path(tmp_path))
    env = make_robot_env(record_video=True, video_path=f"{output_dir}/episode.mp4")
    env.reset()
    env.close()
    logger.info(f"Convenience recording env created with output dir: {output_dir}")


def demo_structured():
    # Use render helper to ensure output directory exists
    output_dir = ensure_output_dir(Path("results"))
    render_opts = RenderOptions(max_fps_override=20)
    rec_opts = RecordingOptions(record=True, video_path=f"{output_dir}/structured_episode.mp4")
    env = make_robot_env(render_options=render_opts, recording_options=rec_opts, debug=True)
    env.reset()
    env.close()
    logger.info(f"Structured options env created with managed output dir: {output_dir}")


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
