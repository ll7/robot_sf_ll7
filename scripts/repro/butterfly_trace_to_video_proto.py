"""Prototype: render one job-13334 exemplar trace episode to a watchable video.

STATUS: prototype for Stages 0-2 of the "butterfly worked-example pipeline" plan
(diss repo: docs/context/plan/2026-07-15_butterfly_worked_example_pipeline.md).
Not wired into any benchmark/CI path. Reuses existing robot_sf render components:

- ``robot_sf.render.jsonl_playback.JSONLPlaybackLoader`` to parse a JSONL episode
  into ``VisualizableSimState`` objects + an estimated ``MapDefinition`` (no map
  metadata is embedded in the exemplar trace bundles, so bounds are inferred from
  the trajectory data itself, exactly the loader's built-in fallback path).
- ``robot_sf.render.frame_export.render_selected_frames`` to draw every frame with
  the real headless ``SimulationView`` (pygame, ``SDL_VIDEODRIVER=dummy``) -- the
  same renderer used by ``frame_export``'s pickle-recording CLI.

Stage 1 glue (video only): (1) convert the exemplar bundle's ``trace_series.json``
(schema ``issue-4891-trace-series.v1``) into the JSONL record schema
``JSONLPlaybackLoader`` already understands, and (2) encode the rendered RGB
frames to an mp4 with imageio (already a repo dependency, used elsewhere for
artifact export).

Stage 2 glue (metric minimap, on by default): compute per-step robot speed and
robot-nearest-pedestrian clearance directly from ``trace_series.json`` frames
(``robot.velocity`` / ``robot.position`` / ``pedestrians[].position``, all
already present, no new simulation data needed), render a two-panel matplotlib
strip (speed, clearance-with-thresholds) with a moving time cursor, and stack it
beneath each scene frame before encoding. Near-miss/collision thresholds are
imported from the repo's own trace-failure-predicate defaults
(``robot_sf.analysis_workbench.trace_failure_predicates``) rather than
re-invented here -- see the module docstring of that file and
``robot_sf/evidence/distance_convention.py`` for why: this trace's per-step
frames carry no per-agent ``radius`` field, so the predicate library's own
"missing-radius" near-contact fallback is the closest, most-honest threshold
available (as opposed to a geometry threshold assuming default robot/pedestrian
radii that this trace does not actually record).

Usage:
    uv run python scripts/repro/butterfly_trace_to_video_proto.py \\
        docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07/orca/classic_head_on_corridor_medium_seed24_best \\
        --out /tmp/headon_orca_seed24.mp4

    # Stage-1-only video, no minimap strip:
    uv run python scripts/repro/butterfly_trace_to_video_proto.py \\
        docs/context/evidence/issue_4891_head_on_corridor_exemplars_2026-07/orca/classic_head_on_corridor_medium_seed24_best \\
        --out /tmp/headon_orca_seed24_sceneonly.mp4 --no-minimap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Must be set before any robot_sf.render module imports pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Matplotlib must not try to open a GUI backend in this headless/CPU-only context.
import matplotlib

matplotlib.use("Agg")

from robot_sf.analysis_workbench.trace_failure_predicates import (
    DEFAULT_COLLISION_MISSING_RADIUS_NEAR_DISTANCE_M,
    DEFAULT_NEAR_MISS_THRESHOLD_M,
)

#: Center-to-center clearance at/below which a step counts as a near-miss.
#: Sourced from ``trace_failure_predicates.DEFAULT_NEAR_MISS_THRESHOLD_M`` (0.5 m):
#: the same threshold + distance convention (raw robot-pedestrian center distance,
#: no radius subtraction) the repo's own ``occlusion_triggered_near_miss`` /
#: ``late_evasive_reaction`` predicates use.
NEAR_MISS_THRESHOLD_M: float = DEFAULT_NEAR_MISS_THRESHOLD_M

#: Center-to-center clearance at/below which a step counts as a collision.
#: This trace's frames carry no per-agent ``radius`` field, so the true
#: geometric collision distance (robot_radius + pedestrian_radius) cannot be
#: computed from the trace itself. We therefore use
#: ``trace_failure_predicates.DEFAULT_COLLISION_MISSING_RADIUS_NEAR_DISTANCE_M``
#: (0.2 m) -- the exact fallback the predicate library uses for frames missing
#: radius data (see ``_collision_events`` in that module). For reference, the
#: repo-wide default *geometric* collision distance would be
#: ``DEFAULT_ROBOT_RADIUS`` (1.0 m, robot_sf/common/robot_defaults.py) +
#: pedestrian radius default (0.4 m, robot_sf/nav/occupancy.py:262) = 1.4 m;
#: this trace's global minimum clearance (2.27 m at step 127) clears both
#: thresholds, so neither marker fires for this particular episode.
COLLISION_THRESHOLD_M: float = DEFAULT_COLLISION_MISSING_RADIUS_NEAR_DISTANCE_M


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


def _render_scene_frames(
    jsonl_path: Path,
    *,
    width: int,
    height: int,
    scaling: float,
) -> list[np.ndarray]:
    """Render every step of a JSONL episode with the real SimulationView.

    Returns:
        RGB frames (``uint8``, shape ``(height, width, 3)``), one per episode step.
    """
    # Local imports: keep SDL_VIDEODRIVER=dummy set before pygame is touched.
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
    return frames


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

    Stage-1 scene-only path (no metric minimap). Returns:
        Number of frames rendered and encoded.
    """
    import imageio.v3 as iio

    frames = _render_scene_frames(jsonl_path, width=width, height=height, scaling=scaling)
    video_out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(video_out, frames, fps=fps)
    return len(frames)


def compute_trace_metrics(trace_series_path: Path) -> dict[str, np.ndarray]:
    """Compute per-step robot speed and nearest-pedestrian clearance from a trace.

    Reads ``frames[].robot.{position,velocity}`` and ``frames[].pedestrians[].position``
    directly from the exemplar ``trace_series.json`` (schema
    ``issue-4891-trace-series.v1``) -- the same file ``trace_series_to_jsonl`` reads,
    just without discarding velocity/pedestrian-identity fields the JSONL playback
    schema drops. No new simulation compute; this is a pure re-derivation of values
    already present in the trace.

    Returns:
        Dict with ``time_s``, ``speed_mps``, ``clearance_m`` (nearest-pedestrian
        center-to-center distance, ``nan`` for steps with zero pedestrians), and
        ``nearest_pedestrian_id`` (``float`` array, ``nan`` where undefined)
        arrays, one entry per trace step, in trace order.
    """
    with trace_series_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    frames = payload["frames"]
    n = len(frames)
    time_s = np.empty(n, dtype=np.float64)
    speed_mps = np.empty(n, dtype=np.float64)
    clearance_m = np.empty(n, dtype=np.float64)
    nearest_pedestrian_id = np.full(n, np.nan, dtype=np.float64)

    for i, frame in enumerate(frames):
        robot = frame["robot"]
        time_s[i] = frame["time_s"]
        vx, vy = robot["velocity"]
        speed_mps[i] = math.hypot(vx, vy)

        rx, ry = robot["position"]
        peds = frame.get("pedestrians", [])
        best_dist = math.inf
        best_id: float = math.nan
        for ped in peds:
            px, py = ped["position"]
            dist = math.hypot(rx - px, ry - py)
            if dist < best_dist:
                best_dist = dist
                best_id = float(ped.get("id", math.nan))
        clearance_m[i] = best_dist if peds else math.nan
        nearest_pedestrian_id[i] = best_id

    return {
        "time_s": time_s,
        "speed_mps": speed_mps,
        "clearance_m": clearance_m,
        "nearest_pedestrian_id": nearest_pedestrian_id,
    }


def render_minimap_frames(
    metrics: dict[str, np.ndarray],
    *,
    near_miss_threshold_m: float,
    collision_threshold_m: float,
    width: int,
    height: int,
    dpi: int = 100,
) -> list[np.ndarray]:
    """Render one metric-strip frame per trace step, cursor advancing in lockstep.

    Draws the full (speed, clearance) strip once and, per step, moves a vertical
    cursor line to that step's ``time_s`` before re-rasterizing the same Matplotlib
    canvas (Agg backend). This keeps a single deterministic figure/axes layout for
    every frame -- the cursor position is read directly off the shared time axis,
    so cursor-frame alignment holds by construction rather than by a separate
    pixel-offset calculation.

    Returns:
        RGB frames (``uint8``, shape ``(height, width, 3)``), one per trace step,
        each with the cursor at that step's time.
    """
    import matplotlib.pyplot as plt

    time_s = metrics["time_s"]
    speed = metrics["speed_mps"]
    clearance = metrics["clearance_m"]
    near_miss_mask = clearance <= near_miss_threshold_m
    collision_mask = clearance <= collision_threshold_m

    fig, (ax_speed, ax_clear) = plt.subplots(
        2,
        1,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        sharex=True,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.16, hspace=0.55)

    ax_speed.plot(time_s, speed, color="#1f77b4", linewidth=1.2)
    ax_speed.set_ylabel("speed (m/s)", fontsize=8)
    ax_speed.set_title("robot speed  /  nearest-pedestrian clearance", fontsize=9)
    ax_speed.tick_params(labelsize=7)
    ax_speed.grid(alpha=0.3)

    ax_clear.plot(time_s, clearance, color="#2ca02c", linewidth=1.2, zorder=3)
    ax_clear.axhline(
        near_miss_threshold_m,
        color="#ff7f0e",
        linestyle="--",
        linewidth=0.9,
        label=f"near-miss <= {near_miss_threshold_m:g} m",
        zorder=2,
    )
    ax_clear.axhline(
        collision_threshold_m,
        color="#d62728",
        linestyle="--",
        linewidth=0.9,
        label=f"collision <= {collision_threshold_m:g} m",
        zorder=2,
    )
    if np.any(near_miss_mask):
        ax_clear.scatter(
            time_s[near_miss_mask],
            clearance[near_miss_mask],
            color="#ff7f0e",
            s=14,
            zorder=5,
            label="near-miss step",
        )
    if np.any(collision_mask):
        ax_clear.scatter(
            time_s[collision_mask],
            clearance[collision_mask],
            color="#d62728",
            s=18,
            zorder=6,
            label="collision step",
        )
    ax_clear.set_ylabel("clearance (m)", fontsize=8)
    ax_clear.set_xlabel("time (s)", fontsize=8)
    ax_clear.tick_params(labelsize=7)
    ax_clear.legend(loc="upper right", fontsize=6, framealpha=0.85)
    ax_clear.grid(alpha=0.3)

    t_min, t_max = float(time_s[0]), float(time_s[-1])
    for ax in (ax_speed, ax_clear):
        ax.set_xlim(t_min, t_max)

    canvas = fig.canvas
    frames: list[np.ndarray] = []
    cursor_lines: list[Any] = []
    for t in time_s:
        for line in cursor_lines:
            line.remove()
        cursor_lines = [
            ax_speed.axvline(float(t), color="black", linewidth=1.1, zorder=10),
            ax_clear.axvline(float(t), color="black", linewidth=1.1, zorder=10),
        ]
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(np.ascontiguousarray(rgba[:, :, :3]))

    plt.close(fig)
    return frames


def compose_scene_and_minimap(
    scene_frames: list[np.ndarray],
    minimap_frames: list[np.ndarray],
) -> list[np.ndarray]:
    """Stack each scene frame above its synchronized minimap-strip frame.

    Returns:
        Composed RGB frames, one per step, scene on top / metric strip below.
    """
    if len(scene_frames) != len(minimap_frames):
        raise ValueError(
            f"frame-count mismatch: {len(scene_frames)} scene frames vs "
            f"{len(minimap_frames)} minimap frames"
        )
    composed: list[np.ndarray] = []
    for scene, strip in zip(scene_frames, minimap_frames, strict=True):
        if scene.shape[1] != strip.shape[1]:
            raise ValueError(
                f"width mismatch: scene width {scene.shape[1]} != strip width {strip.shape[1]}"
            )
        composed.append(np.vstack([scene, strip]))
    return composed


@dataclass(frozen=True)
class MinimapVideoConfig:
    """Rendering knobs for the Stage-2 scene+minimap video."""

    fps: int = 10
    width: int = 1280
    height: int = 720
    scaling: float = 10.0
    minimap_height: int = 320
    minimap_dpi: int = 100
    near_miss_threshold_m: float = NEAR_MISS_THRESHOLD_M
    collision_threshold_m: float = COLLISION_THRESHOLD_M


def render_episode_with_minimap(
    trace_series_path: Path,
    jsonl_path: Path,
    video_out: Path,
    config: MinimapVideoConfig,
) -> dict[str, Any]:
    """Render the Stage-2 scene+minimap video and encode it to mp4.

    Returns:
        Summary dict: frame count, near-miss/collision step counts, and the
        (step, clearance_m) of the trace's global minimum clearance.
    """
    import imageio.v3 as iio

    scene_frames = _render_scene_frames(
        jsonl_path, width=config.width, height=config.height, scaling=config.scaling
    )
    metrics = compute_trace_metrics(trace_series_path)
    if len(scene_frames) != len(metrics["time_s"]):
        raise RuntimeError(
            f"scene frame count {len(scene_frames)} != metric step count "
            f"{len(metrics['time_s'])} -- trace/JSONL step mismatch"
        )

    minimap_frames = render_minimap_frames(
        metrics,
        near_miss_threshold_m=config.near_miss_threshold_m,
        collision_threshold_m=config.collision_threshold_m,
        width=config.width,
        height=config.minimap_height,
        dpi=config.minimap_dpi,
    )
    composed = compose_scene_and_minimap(scene_frames, minimap_frames)

    video_out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(video_out, composed, fps=config.fps)

    clearance = metrics["clearance_m"]
    min_step = int(np.nanargmin(clearance))
    return {
        "n_frames": len(composed),
        "near_miss_steps": int(np.sum(clearance <= config.near_miss_threshold_m)),
        "collision_steps": int(np.sum(clearance <= config.collision_threshold_m)),
        "min_clearance_step": min_step,
        "min_clearance_m": float(clearance[min_step]),
    }


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
    parser.add_argument(
        "--no-minimap",
        action="store_true",
        help="Stage-1 behavior: scene-only video, no synchronized metric strip.",
    )
    parser.add_argument(
        "--minimap-height",
        type=int,
        default=320,
        help="Metric-strip height in pixels (strip width always matches --width).",
    )
    parser.add_argument(
        "--minimap-dpi", type=int, default=100, help="Matplotlib render DPI for the metric strip."
    )
    parser.add_argument(
        "--near-miss-threshold-m",
        type=float,
        default=NEAR_MISS_THRESHOLD_M,
        help=(
            "Center-to-center clearance (m) at/below which a step is marked "
            f"near-miss (default: repo trace_failure_predicates default, {NEAR_MISS_THRESHOLD_M} m)."
        ),
    )
    parser.add_argument(
        "--collision-threshold-m",
        type=float,
        default=COLLISION_THRESHOLD_M,
        help=(
            "Center-to-center clearance (m) at/below which a step is marked "
            f"collision (default: repo missing-radius fallback, {COLLISION_THRESHOLD_M} m)."
        ),
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
        summary = _render(args, trace_series_path, jsonl_path)
    else:
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "episode.jsonl"
            n_steps = trace_series_to_jsonl(trace_series_path, jsonl_path)
            summary = _render(args, trace_series_path, jsonl_path)

    print(f"steps converted: {n_steps}")
    print(f"frames rendered: {summary['n_frames']}")
    if not args.no_minimap:
        print(
            f"near-miss steps (<= {args.near_miss_threshold_m:g} m): {summary['near_miss_steps']}"
        )
        print(
            f"collision steps (<= {args.collision_threshold_m:g} m): {summary['collision_steps']}"
        )
        print(
            f"global min clearance: {summary['min_clearance_m']:.6f} m "
            f"at step {summary['min_clearance_step']}"
        )
    print(f"video: {args.out}")
    return 0


def _render(args: argparse.Namespace, trace_series_path: Path, jsonl_path: Path) -> dict[str, Any]:
    """Dispatch to the Stage-1 (scene-only) or Stage-2 (scene+minimap) renderer.

    Returns:
        Summary dict with at least an ``n_frames`` key.
    """
    if args.no_minimap:
        n_frames = render_episode_to_video(
            jsonl_path,
            args.out,
            fps=args.fps,
            width=args.width,
            height=args.height,
            scaling=args.scaling,
        )
        return {"n_frames": n_frames}
    config = MinimapVideoConfig(
        fps=args.fps,
        width=args.width,
        height=args.height,
        scaling=args.scaling,
        minimap_height=args.minimap_height,
        minimap_dpi=args.minimap_dpi,
        near_miss_threshold_m=args.near_miss_threshold_m,
        collision_threshold_m=args.collision_threshold_m,
    )
    return render_episode_with_minimap(trace_series_path, jsonl_path, args.out, config)


if __name__ == "__main__":
    raise SystemExit(main())
