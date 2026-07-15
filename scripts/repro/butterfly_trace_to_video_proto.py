"""Prototype: render one job-13334 exemplar trace episode to a watchable video.

STATUS: prototype for Stages 0-1 of the "butterfly worked-example pipeline" plan
(diss repo: docs/context/plan/2026-07-15_butterfly_worked_example_pipeline.md).
Not wired into any benchmark/CI path. Reuses existing robot_sf render components:

- ``robot_sf.render.jsonl_playback.JSONLPlaybackLoader`` to parse a JSONL episode
  into ``VisualizableSimState`` objects + an estimated ``MapDefinition`` (no map
  metadata is embedded in the exemplar trace bundles, so bounds are inferred from
  the trajectory data itself, exactly the loader's built-in fallback path).
- ``robot_sf.render.frame_export.render_selected_frames`` to draw every frame with
  the real headless ``SimulationView`` (pygame, ``SDL_VIDEODRIVER=dummy``) -- the
  same renderer used by ``frame_export``'s pickle-recording CLI.

The only new code is the glue: (1) convert the exemplar bundle's
``trace_series.json`` (schema ``issue-4891-trace-series.v1``) into the JSONL
record schema ``JSONLPlaybackLoader`` already understands, and (2) encode the
rendered RGB frames to an mp4 with imageio (already a repo dependency, used
elsewhere for artifact export).

Usage:
    uv run python scripts/repro/butterfly_trace_to_video_proto.py \\
        docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07/orca/classic_head_on_corridor_medium_seed24_best \\
        --out /tmp/headon_orca_seed24.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Must be set before any robot_sf.render module imports pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def trace_series_to_jsonl(trace_series_path: Path, jsonl_out: Path) -> int:
    """Convert one exemplar ``trace_series.json`` into a JSONL episode file.

    Maps the ``issue-4891-trace-series.v1`` per-step ``frames`` array onto the
    record schema ``JSONLPlaybackLoader.load_single_episode`` parses natively:
    ``{"episode_id", "event": "step", "step_idx", "state": {...}}`` with
    ``state.robot_pose = [[x, y], heading]`` and
    ``state.pedestrian_positions = [[x, y], ...]``.

    Returns:
        Number of step records written.
    """
    with trace_series_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload["frames"]
    episode_id = payload.get("metadata", {}).get("episode_id", 0)

    written = 0
    with jsonl_out.open("w", encoding="utf-8") as out:
        for frame in frames:
            robot = frame["robot"]
            peds = frame.get("pedestrians", [])
            record = {
                "episode_id": episode_id,
                "event": "step",
                "step_idx": frame["step"],
                "state": {
                    "timestep": frame["step"],
                    "robot_pose": [list(robot["position"]), robot["heading"]],
                    "pedestrian_positions": [list(p["position"]) for p in peds],
                },
            }
            out.write(json.dumps(record) + "\n")
            written += 1
    return written


def render_episode_to_video(
    jsonl_path: Path,
    video_out: Path,
    *,
    fps: int,
    width: int,
    height: int,
    scaling: float,
) -> int:
    """Render every step of a JSONL episode to an mp4 via the real SimulationView.

    Returns:
        Number of frames rendered and encoded.
    """
    # Local imports: keep SDL_VIDEODRIVER=dummy set before pygame is touched.
    import imageio.v3 as iio

    from robot_sf.render import frame_export
    from robot_sf.render.jsonl_playback import JSONLPlaybackLoader

    loader = JSONLPlaybackLoader()
    episode, map_def = loader.load_single_episode(jsonl_path)
    states = episode.states
    if not states:
        raise RuntimeError(f"No renderable states parsed from {jsonl_path}")

    indices = list(range(len(states)))
    frames = frame_export.render_selected_frames(
        states,
        map_def,
        indices,
        width=width,
        height=height,
        scaling=scaling,
    )
    if not frames:
        raise RuntimeError("render_selected_frames returned no frames")

    video_out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(video_out, frames, fps=fps)
    return len(frames)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bundle_dir",
        type=Path,
        help="Exemplar bundle dir containing trace_series.json.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output mp4 path.")
    parser.add_argument(
        "--jsonl-out",
        type=Path,
        default=None,
        help="Optional path to keep the intermediate JSONL episode file "
        "(default: a temp file, deleted after rendering).",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video frame rate.")
    parser.add_argument("--width", type=int, default=1280, help="Render width in pixels.")
    parser.add_argument("--height", type=int, default=720, help="Render height in pixels.")
    parser.add_argument(
        "--scaling", type=float, default=10.0, help="World-to-pixel scaling (SimulationView)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    trace_series_path = args.bundle_dir / "trace_series.json"
    if not trace_series_path.exists():
        print(f"error: {trace_series_path} not found", file=sys.stderr)
        return 1

    if args.jsonl_out is not None:
        jsonl_path = args.jsonl_out
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        n_steps = trace_series_to_jsonl(trace_series_path, jsonl_path)
        n_frames = render_episode_to_video(
            jsonl_path,
            args.out,
            fps=args.fps,
            width=args.width,
            height=args.height,
            scaling=args.scaling,
        )
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "episode.jsonl"
            n_steps = trace_series_to_jsonl(trace_series_path, jsonl_path)
            n_frames = render_episode_to_video(
                jsonl_path,
                args.out,
                fps=args.fps,
                width=args.width,
                height=args.height,
                scaling=args.scaling,
            )

    print(f"steps converted: {n_steps}")
    print(f"frames rendered: {n_frames}")
    print(f"video: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
