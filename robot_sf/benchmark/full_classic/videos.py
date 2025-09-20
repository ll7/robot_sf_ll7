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


def generate_videos(records, out_dir, cfg):  # T037 + T038
    """Generate representative episode videos.

    Returns list of ``_VideoArtifact`` items with status among:
      - generated: successful MP4 creation
      - skipped: smoke mode, missing deps, or disabled
      - error: unexpected render failure (continues benchmark)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    artifacts: List[_VideoArtifact] = []
    if not records:
        return artifacts

    smoke = bool(getattr(cfg, "smoke", False))
    disable_videos = bool(getattr(cfg, "disable_videos", False))
    max_videos = int(getattr(cfg, "max_videos", 1))
    # Ensure deterministic selection order
    selected = records[: max_videos or 1]

    if smoke or disable_videos:
        for rec in selected:
            episode_id = rec.get("episode_id", "unknown")
            scenario_id = rec.get("scenario_id", "unknown")
            mp4_path = out_path / f"{episode_id}.mp4"
            artifacts.append(
                _VideoArtifact(
                    artifact_id=f"video_{episode_id}",
                    scenario_id=scenario_id,
                    episode_id=episode_id,
                    path_mp4=str(mp4_path),
                    status="skipped",
                    note="smoke mode" if smoke else "video generation disabled",
                )
            )
        return artifacts

    if plt is None or ImageSequenceClip is None:
        reason = "matplotlib missing" if plt is None else "moviepy missing"
        for rec in selected:
            episode_id = rec.get("episode_id", "unknown")
            scenario_id = rec.get("scenario_id", "unknown")
            mp4_path = out_path / f"{episode_id}.mp4"
            artifacts.append(
                _VideoArtifact(
                    artifact_id=f"video_{episode_id}",
                    scenario_id=scenario_id,
                    episode_id=episode_id,
                    path_mp4=str(mp4_path),
                    status="skipped",
                    note=reason,
                )
            )
        return artifacts

    # Render videos
    for rec in selected:
        episode_id = rec.get("episode_id", "unknown")
        scenario_id = rec.get("scenario_id", "unknown")
        seed = int(rec.get("seed", 0))
        random.seed(seed)
        mp4_path = out_path / f"{episode_id}.mp4"
        frames = []
        try:
            # Generate synthetic path (deterministic) length N
            N = 40
            xs = [math.cos((seed + i) * 0.15) for i in range(N)]
            ys = [math.sin((seed + i) * 0.15) for i in range(N)]

            # Heuristic outcome inference
            collision_flag = bool(rec.get("collisions") or rec.get("collision"))
            success_flag = bool(rec.get("success", not collision_flag))
            timeout_flag = bool(rec.get("timeout")) and not success_flag and not collision_flag
            outcome = (
                "COLLISION"
                if collision_flag
                else "SUCCESS"
                if success_flag
                else "TIMEOUT"
                if timeout_flag
                else "EPISODE"
            )

            for idx in range(N):
                fig, ax = plt.subplots(figsize=(3, 3))  # type: ignore
                ax.plot(xs[: idx + 1], ys[: idx + 1], color="tab:blue", linewidth=1.5)
                ax.scatter(xs[idx], ys[idx], color="red", s=20)
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylim(-1.2, 1.2)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{scenario_id} / {episode_id}\n{outcome}")
                fig.tight_layout()
                # Convert to RGB array
                fig.canvas.draw()  # type: ignore
                width, height = fig.canvas.get_width_height()  # type: ignore
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")  # type: ignore
                image = image.reshape((height, width, 3))
                frames.append(image)
                plt.close(fig)  # type: ignore

            if not frames:
                raise RuntimeError("No frames generated")

            clip = ImageSequenceClip(frames, fps=10)  # type: ignore
            # Suppress verbose moviepy logging
            clip.write_videofile(str(mp4_path), codec="libx264", fps=10, verbose=False, logger=None)  # type: ignore
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
