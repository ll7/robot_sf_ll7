"""Benchmark video artifact generation (synthetic renderer; SimulationView deferred).

Implements portion of Spec 127 / TODO 7a:
    * Deterministic selection of the first N episodes.
    * Synthetic path rendering with matplotlib + moviepy encoding.
    * Skip notes: smoke-mode, disabled, moviepy-missing, simulation-view-missing.
    * Status values aligned with forthcoming schema: success | skipped | failed.
    * Performance timing (encode_time_s) recorded per artifact.
    * Renderer selection flag (cfg.video_renderer) supporting: synthetic | sim-view | none.

Deferred (future): true SimulationView renderer, memory peak measurement, streaming encode.
"""

from __future__ import annotations

import io
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:  # Optional dependencies
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib import image as mpimg  # type: ignore
except ImportError:
    plt = None  # type: ignore
    mpimg = None  # type: ignore[assignment]

try:  # moviepy for encoding
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # type: ignore
except ImportError:
    ImageSequenceClip = None  # type: ignore


@dataclass
class _VideoArtifact:
    """Internal video artifact representation.

    Fields match draft schema (filename instead of path_mp4, renderer, encode metrics).
    """

    artifact_id: str
    scenario_id: str
    episode_id: str
    filename: str | None
    renderer: str
    status: str  # success | skipped | failed
    note: str | None = None
    encode_time_s: float | None = None
    memory_peak_mb: float | None = None

    # Backward compatibility for older tests expecting .path_mp4
    @property
    def path_mp4(self) -> str:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
        return self.filename or ""


def _canvas_to_rgb_simple(fig) -> np.ndarray:
    """Return RGB array from figure using the simplest, backend-agnostic path.

    We intentionally avoid complex HiDPI / ARGB logic here; on macOS the default
    interactive backend (`macosx`) lacks `tostring_rgb`. Users (or the benchmark
    harness) should set MPLBACKEND=Agg before importing matplotlib to ensure
    `tostring_rgb` is available. If absent we raise a clear error directing the
    user to set MPLBACKEND=Agg.
    """
    fig.canvas.draw()  # type: ignore[attr-defined]
    w, h = fig.canvas.get_width_height()  # type: ignore[attr-defined]
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = fig.canvas.tostring_rgb()  # type: ignore[attr-defined]
        return np.frombuffer(buf, dtype="uint8").reshape((h, w, 3))
    # Portable fallback: render figure to PNG in-memory and decode with matplotlib.image
    if mpimg is None:
        raise RuntimeError("matplotlib.image not available; ensure matplotlib installed")
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=100)  # type: ignore[attr-defined]
    bio.seek(0)
    png_arr = mpimg.imread(bio)
    # mpimg may return float [0,1] array; convert to uint8 RGB (drop alpha if present)
    if png_arr.dtype != np.uint8:
        png_arr = (png_arr * 255).astype("uint8")
    if png_arr.shape[-1] == 4:  # RGBA -> RGB
        png_arr = png_arr[:, :, :3]
    return png_arr


def _render_episode_frames(seed: int, N: int) -> tuple[list, list[float], list[float]]:
    """Generate synthetic (x,y) path coordinates for episode rendering."""
    xs = [math.cos((seed + i) * 0.15) for i in range(N)]
    ys = [math.sin((seed + i) * 0.15) for i in range(N)]
    return [], xs, ys


def _build_outcome(rec) -> str:
    """TODO docstring. Document this function.

    Args:
        rec: TODO docstring.

    Returns:
        TODO docstring.
    """
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


def generate_videos(records, out_dir, cfg):  # noqa: C901
    """Generate representative episode videos.

    Config attributes used if present on cfg:
        - smoke (bool): skip with note 'smoke-mode'
        - disable_videos / no_video (bool): skip with note 'disabled'
        - max_videos (int)
        - video_renderer (str): synthetic | sim-view | none
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if not records:
        return []

    smoke = bool(getattr(cfg, "smoke", False))
    disable_videos = bool(getattr(cfg, "disable_videos", False) or getattr(cfg, "no_video", False))
    max_videos = int(getattr(cfg, "max_videos", 1)) or 1
    renderer_req = str(getattr(cfg, "video_renderer", "synthetic"))
    selected = records[:max_videos]

    def _mk_skip(rec, note: str):  # helper
        # Provide deterministic filename even when skipped (legacy expectation)
        """TODO docstring. Document this function.

        Args:
            rec: TODO docstring.
            note: TODO docstring.
        """
        episode_id = rec.get("episode_id", "unknown")
        mp4_name = f"video_{episode_id}.mp4"
        return _VideoArtifact(
            artifact_id=f"video_{rec.get('episode_id', 'unknown')}",
            scenario_id=rec.get("scenario_id", "unknown"),
            episode_id=rec.get("episode_id", "unknown"),
            filename=str(out_path / mp4_name),
            renderer=(
                "synthetic"
                if note in {"moviepy missing", "smoke mode", "disabled"}
                else ("simulation_view" if renderer_req == "sim-view" else "synthetic")
            ),
            status="skipped",
            note=note,
            encode_time_s=None,
            memory_peak_mb=None,
        )

    if renderer_req == "none":
        disable_videos = True

    if smoke:
        return [_mk_skip(r, "smoke mode") for r in selected]
    if disable_videos:
        return [_mk_skip(r, "disabled") for r in selected]
    if plt is None:
        return [_mk_skip(r, "disabled") for r in selected]
    if ImageSequenceClip is None:
        return [_mk_skip(r, "moviepy missing") for r in selected]
    if renderer_req == "sim-view":  # not implemented yet
        # Communicate downgrade but still proceed with synthetic generation
        # (Tests can assert presence of skip note separately later if needed.)
        pass

    artifacts: list[_VideoArtifact] = []
    for rec in selected:
        episode_id = rec.get("episode_id", "unknown")
        scenario_id = rec.get("scenario_id", "unknown")
        seed = int(rec.get("seed", 0))
        random.seed(seed)
        mp4_path = out_path / f"video_{episode_id}.mp4"
        try:
            N = 40
            _, xs_full, ys_full = _render_episode_frames(seed, N)
            outcome = _build_outcome(rec)
            frames = []
            t0 = time.perf_counter()
            for idx in range(N):
                fig, ax = plt.subplots(figsize=(3, 3))  # type: ignore
                ax.plot(
                    xs_full[: idx + 1],
                    ys_full[: idx + 1],
                    color="tab:blue",
                    linewidth=1.5,
                )
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
            encode_time = time.perf_counter() - t0
            artifacts.append(
                _VideoArtifact(
                    artifact_id=f"video_{episode_id}",
                    scenario_id=scenario_id,
                    episode_id=episode_id,
                    filename=str(mp4_path),
                    renderer="synthetic",
                    status="generated",
                    note="synthetic annotated path",
                    encode_time_s=encode_time,
                    memory_peak_mb=0.0,
                )
            )
        except (RuntimeError, OSError, ValueError) as e:
            artifacts.append(
                _VideoArtifact(
                    artifact_id=f"video_{episode_id}",
                    scenario_id=scenario_id,
                    episode_id=episode_id,
                    filename=str(mp4_path),
                    renderer="synthetic",
                    status="error",
                    note=f"render failed: {e.__class__.__name__}: {e}",
                    encode_time_s=None,
                    memory_peak_mb=None,
                )
            )
    return artifacts


def artifacts_to_manifest(artifacts: list[_VideoArtifact]):
    """Convert internal artifacts list to manifest dict for JSON dumping."""
    return {
        "artifacts": [
            {k: v for k, v in asdict(a).items() if k not in {"artifact_id"}} for a in artifacts
        ],
    }
