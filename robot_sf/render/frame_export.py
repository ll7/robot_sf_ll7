"""Headless frame and filmstrip export for recorded simulation pickle states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import SimulationView

if TYPE_CHECKING:
    from robot_sf.nav.map_config import MapDefinition
    from robot_sf.render.sim_state import VisualizableSimState


def select_evenly_spaced_indices(total_frames: int, count: int) -> list[int]:
    """Select deterministic, unique frame indices spanning a recording.

    Returns:
        Frame indices in ascending order.
    """
    if total_frames < 0:
        raise ValueError(f"total_frames must be non-negative, got {total_frames}")
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if total_frames == 0:
        return []
    if count >= total_frames:
        return list(range(total_frames))
    if count == 1:
        return [0]

    last_index = total_frames - 1
    return [round(i * last_index / (count - 1)) for i in range(count)]


def render_selected_frames(
    states: list[VisualizableSimState],
    map_def: MapDefinition,
    indices: list[int],
    *,
    width: int = 1280,
    height: int = 720,
    scaling: float = 10,
) -> list[np.ndarray]:
    """Render selected states to RGB arrays using the headless-safe renderer path.

    Returns:
        Rendered RGB frames in the same order as ``indices``.
    """
    if not states or not indices:
        return []

    invalid_indices = [index for index in indices if index < 0 or index >= len(states)]
    if invalid_indices:
        raise IndexError(f"Frame indices out of range for {len(states)} states: {invalid_indices}")

    sim_view = SimulationView(
        width=width,
        height=height,
        scaling=scaling,
        map_def=map_def,
        caption="RobotSF Frame Export",
        record_video=True,
        video_path=None,
        display_text=False,
    )
    try:
        for index in indices:
            sim_view.render(states[index])
        return sim_view.exit_simulation(return_frames=True) or []
    finally:
        if not sim_view.is_exit_requested:
            sim_view.exit_simulation()


def write_png_frames(
    frames: list[np.ndarray],
    output_dir: Path | str,
    *,
    prefix: str = "frame",
    indices: list[int] | None = None,
) -> list[Path]:
    """Write RGB frame arrays as PNG files.

    Returns:
        Paths to written PNG files in frame order.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if indices is not None and len(indices) != len(frames):
        raise ValueError("indices length must match frames length")

    written_paths: list[Path] = []
    for ordinal, frame in enumerate(frames):
        frame_index = indices[ordinal] if indices is not None else ordinal
        path = output_path / f"{prefix}_{frame_index:06d}.png"
        Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
        written_paths.append(path)
    return written_paths


def write_filmstrip(
    frames: list[np.ndarray],
    output_path: Path | str,
    *,
    columns: int | None = None,
) -> Path:
    """Write selected frames into a simple PNG filmstrip grid.

    Returns:
        Path to the written filmstrip PNG.
    """
    if not frames:
        raise ValueError("Cannot write a filmstrip without frames")
    if columns is not None and columns < 1:
        raise ValueError(f"columns must be positive, got {columns}")

    frame_images = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
    frame_width, frame_height = frame_images[0].size
    for image in frame_images:
        if image.size != (frame_width, frame_height):
            raise ValueError("All filmstrip frames must have the same dimensions")

    resolved_columns = columns or len(frame_images)
    rows = int(np.ceil(len(frame_images) / resolved_columns))
    filmstrip = Image.new("RGB", (resolved_columns * frame_width, rows * frame_height), "white")
    for ordinal, image in enumerate(frame_images):
        x = (ordinal % resolved_columns) * frame_width
        y = (ordinal // resolved_columns) * frame_height
        filmstrip.paste(image, (x, y))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    filmstrip.save(path)
    return path


def export_pickle_frames(
    state_file: Path | str,
    output_dir: Path | str,
    *,
    count: int = 8,
    prefix: str = "frame",
    filmstrip_path: Path | str | None = None,
    filmstrip_columns: int | None = None,
    render_size: tuple[int, int] = (1280, 720),
    scaling: float = 10,
) -> tuple[list[Path], Path | None]:
    """Export evenly spaced PNG frames, optionally with a filmstrip, from a pickle recording.

    Returns:
        Tuple containing written frame paths and optional filmstrip path.
    """
    states, map_def = load_states(str(state_file))
    indices = select_evenly_spaced_indices(len(states), count)
    frames = render_selected_frames(
        states,
        map_def,
        indices,
        width=render_size[0],
        height=render_size[1],
        scaling=scaling,
    )
    frame_paths = write_png_frames(frames, output_dir, prefix=prefix, indices=indices)
    filmstrip = (
        write_filmstrip(frames, filmstrip_path, columns=filmstrip_columns)
        if filmstrip_path
        else None
    )
    return frame_paths, filmstrip


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export deterministic PNG frames from Robot SF pickle recordings.",
    )
    parser.add_argument("state_file", type=Path, help="Pickle file containing (states, map_def).")
    parser.add_argument("output_dir", type=Path, help="Directory for exported PNG frames.")
    parser.add_argument("--count", type=int, default=8, help="Number of evenly spaced frames.")
    parser.add_argument("--prefix", default="frame", help="PNG filename prefix.")
    parser.add_argument("--filmstrip", type=Path, help="Optional output PNG filmstrip path.")
    parser.add_argument("--filmstrip-columns", type=int, help="Optional filmstrip column count.")
    parser.add_argument("--width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Render height in pixels.")
    parser.add_argument(
        "--scaling", type=float, default=10, help="Renderer world-to-pixel scaling."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ``python -m robot_sf.render.frame_export``.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    frame_paths, filmstrip_path = export_pickle_frames(
        args.state_file,
        args.output_dir,
        count=args.count,
        prefix=args.prefix,
        filmstrip_path=args.filmstrip,
        filmstrip_columns=args.filmstrip_columns,
        render_size=(args.width, args.height),
        scaling=args.scaling,
    )
    for path in frame_paths:
        sys.stdout.write(f"{path}\n")
    if filmstrip_path:
        sys.stdout.write(f"{filmstrip_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
