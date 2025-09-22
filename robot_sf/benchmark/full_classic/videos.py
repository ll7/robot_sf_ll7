"""Annotated video generation for representative episodes.

Tasks:
    - T037: Initial stub returning skipped artifacts in smoke mode.
    - T038: Add lightweight annotated video generation when not in smoke mode.

Design (T038):
    - Keep generation inexpensive: simple synthetic path derived from seed (deterministic).
    - Use matplotlib to render frames; compose MP4 via moviepy if available.
    - Graceful degradation:
            * If smoke mode -> always skipped (fast tests)
            * If matplotlib or moviepy/ffmpeg unavailable -> skipped with note
            * Any runtime error during render -> status "error" with note
    - Episode selection: first N records (cfg.max_videos or default 1)
    - Annotations: path trail, current position marker, title with scenario/episode id, outcome tag (SUCCESS/COLLISION/TIMEOUT heuristic via record fields).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np  # type: ignore

try:  # Optional dependencies
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # noqa: BLE001
    plt = None  # type: ignore

try:  # moviepy for encoding
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
except Exception:  # noqa: BLE001
    ImageSequenceClip = None  # type: ignore


@dataclass
class _VideoArtifact:  # minimal internal representation until full T038
    artifact_id: str
    scenario_id: str
    episode_id: str
    path_mp4: str
    status: str  # generated|skipped|error
    note: str | None = None


def _canvas_to_rgb_simple(fig) -> "np.ndarray":  # type: ignore[name-defined]
    """Return RGB array from figure using the simplest, backend-agnostic path.

    We intentionally avoid complex HiDPI / ARGB logic here; on macOS the default
    interactive backend (`macosx`) lacks `tostring_rgb`. Users (or the benchmark
    harness) should set MPLBACKEND=Agg before importing matplotlib to ensure
    `tostring_rgb` is available. If absent we raise a clear error directing the
    user to set MPLBACKEND=Agg.
    """
    import numpy as _np  # local import

    fig.canvas.draw()  # type: ignore[attr-defined]
    w, h = fig.canvas.get_width_height()  # type: ignore[attr-defined]
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()  # type: ignore[attr-defined]
        return _np.frombuffer(buf, dtype="uint8").reshape((h, w, 3))
    # Portable fallback: render figure to PNG in-memory and decode with matplotlib.image
    import io

    from matplotlib import image as mpimg  # type: ignore

    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=100)  # type: ignore[attr-defined]
    bio.seek(0)
    png_arr = mpimg.imread(bio)
    # mpimg may return float [0,1] array; convert to uint8 RGB (drop alpha if present)
    if png_arr.dtype != _np.uint8:
        png_arr = (png_arr * 255).astype("uint8")
    if png_arr.shape[-1] == 4:  # RGBA -> RGB
        png_arr = png_arr[:, :, :3]
    return png_arr


def _render_episode_frames(seed: int, N: int) -> tuple[list, List[float], List[float]]:
    """Generate synthetic (x,y) path coordinates for episode rendering."""
    xs = [math.cos((seed + i) * 0.15) for i in range(N)]
    ys = [math.sin((seed + i) * 0.15) for i in range(N)]
    return [], xs, ys


def _build_outcome(rec) -> str:
    collision_flag = bool(rec.get("collisions") or rec.get("collision"))
    success_flag = bool(rec.get("success", not collision_flag))
    timeout_flag = bool(rec.get("timeout")) and not success_flag and not collision_flag
    return (
        "COLLISION"
        if collision_flag
        else "SUCCESS"
        if success_flag
        else "TIMEOUT"
        if timeout_flag
        else "EPISODE"
    )


## Removed unused _make_skip_artifacts helper (original attempt to simplify skip path)


def generate_videos(records, out_dir, cfg):  # T037 + T038 (refactored for HiDPI)
    """Generate representative episode videos (synthetic path).

    Status values:
      - generated: MP4 successfully written
      - skipped: smoke/disabled/missing dependency
      - error: unexpected failure (benchmark continues)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not records:
        return []

    smoke = bool(getattr(cfg, "smoke", False))
    disable_videos = bool(getattr(cfg, "disable_videos", False))
    max_videos = int(getattr(cfg, "max_videos", 1)) or 1
    selected = records[:max_videos]

    if smoke or disable_videos:
        note = "smoke mode" if smoke else "video generation disabled"
        return [
            _VideoArtifact(
                artifact_id=f"video_{r.get('episode_id', 'unknown')}",
                scenario_id=r.get("scenario_id", "unknown"),
                episode_id=r.get("episode_id", "unknown"),
                path_mp4=str(out_path / f"{r.get('episode_id', 'unknown')}.mp4"),
                status="skipped",
                note=note,
            )
            for r in selected
        ]

    if plt is None or ImageSequenceClip is None:
        reason = "matplotlib missing" if plt is None else "moviepy missing"
        return [
            _VideoArtifact(
                artifact_id=f"video_{r.get('episode_id', 'unknown')}",
                scenario_id=r.get("scenario_id", "unknown"),
                episode_id=r.get("episode_id", "unknown"),
                path_mp4=str(out_path / f"{r.get('episode_id', 'unknown')}.mp4"),
                status="skipped",
                note=reason,
            )
            for r in selected
        ]

    artifacts: List[_VideoArtifact] = []
    for rec in selected:
        episode_id = rec.get("episode_id", "unknown")
        scenario_id = rec.get("scenario_id", "unknown")
        seed = int(rec.get("seed", 0))
        random.seed(seed)
        mp4_path = out_path / f"{episode_id}.mp4"
        try:
            N = 40
            _, xs_full, ys_full = _render_episode_frames(seed, N)
            outcome = _build_outcome(rec)
            frames = []
            for idx in range(N):
                fig, ax = plt.subplots(figsize=(3, 3))  # type: ignore
                ax.plot(xs_full[: idx + 1], ys_full[: idx + 1], color="tab:blue", linewidth=1.5)
                ax.scatter(xs_full[idx], ys_full[idx], color="red", s=20)
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{scenario_id} / {episode_id}\n{outcome}")
                fig.tight_layout()
                frame = _canvas_to_rgb_simple(fig)
                frames.append(frame)
                plt.close(fig)  # type: ignore
            if not frames:
                raise RuntimeError("No frames generated")
            clip = ImageSequenceClip(frames, fps=10)  # type: ignore
            # moviepy version in environment does not accept verbose/logger kwargs
            clip.write_videofile(
                str(mp4_path),
                codec="libx264",
                fps=10,
            )
            artifacts.append(
                _VideoArtifact(
                    artifact_id=f"video_{episode_id}",
                    scenario_id=scenario_id,
                    episode_id=episode_id,
                    path_mp4=str(mp4_path),
                    status="generated",
                    note="synthetic annotated path",
                )
            )
        except Exception as e:  # noqa: BLE001
            artifacts.append(
                _VideoArtifact(
                    artifact_id=f"video_{episode_id}",
                    scenario_id=scenario_id,
                    episode_id=episode_id,
                    path_mp4=str(mp4_path),
                    status="error",
                    note=f"render failed: {e.__class__.__name__}",
                )
            )
    return artifacts
