"""Visual artifact generation (plots + videos) for Full Classic Benchmark.

Implements spec FR-001..FR-015:
 - Deterministic plot & video artifact manifests
 - SimulationView-first (placeholder stub: fallback to synthetic video module)
 - Graceful degradation on optional dependency absence
 - Performance timing meta
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from loguru import logger

from .encode import encode_frames
from .plots import generate_plots
from .render_sim_view import generate_frames
from .render_synthetic import generate_fallback_videos
from .replay import extract_replay_episodes, validate_replay_episode
from .validation import validate_visual_manifests
from .visual_constants import (
    NOTE_DISABLED,
    NOTE_INSUFFICIENT_REPLAY,
    NOTE_SMOKE_MODE,
    RENDERER_SIM_VIEW,
    RENDERER_SYNTHETIC,
)
from .visual_deps import simulation_view_ready

try:  # Try to import SimulationView lazily (primary renderer)
    from robot_sf.render.sim_view import SimulationView  # type: ignore  # noqa: F401

    _SIM_VIEW_CLS = SimulationView  # touch to silence unused import warning

    _SIM_VIEW_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SIM_VIEW_AVAILABLE = False


@dataclass
class PlotArtifact:
    kind: str
    path_pdf: str
    status: str
    note: str | None = None


@dataclass
class VideoArtifact:
    artifact_id: str
    scenario_id: str
    episode_id: str
    path_mp4: str
    status: str
    renderer: str
    note: str | None = None
    encode_time_s: float | None = None
    peak_rss_mb: float | None = None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def _convert_plot_artifacts(raw_list) -> List[dict]:
    out: List[dict] = []
    for a in raw_list:
        out.append(
            {
                "kind": getattr(a, "kind", "unknown"),
                "path_pdf": getattr(a, "path_pdf", ""),
                "status": getattr(a, "status", "skipped"),
                "note": getattr(a, "note", None),
            }
        )
    return out


def _attempt_sim_view_videos(_records, _out_dir: Path, _cfg, replay_map) -> List[VideoArtifact]:
    """Attempt to render + encode videos using SimulationView first path (FR-001, FR-008, T036).

    Behavior summary:
    1. If SimulationView import or readiness probe fails → return [] (caller triggers fallback).
    2. For each selected record with valid replay episode, stream frames via generate_frames
       into encode_frames (moviepy). Memory sampling + encode time captured.
    3. Invalid / insufficient replay → skipped artifact (NOTE_INSUFFICIENT_REPLAY) using SIM_VIEW renderer.
    4. Encoding failure → failed artifact (status=failed, note propagated) and partial file (<1KB) removed.
    5. If moviepy missing (encode_frames returns skipped) we short‑circuit returning [] so synthetic path
       can still attempt (maintains prior behavior tests expect for moviepy-missing path). We only do this
       if *all* attempted encodes report moviepy-missing; mixed states still return produced artifacts.
    """
    if not _SIM_VIEW_AVAILABLE or not simulation_view_ready():
        return []

    capture_enabled = bool(getattr(_cfg, "capture_replay", False))
    if not capture_enabled:
        return []  # Without replay we cannot reconstruct frames

    fps = int(getattr(_cfg, "video_fps", 10))
    smoke = bool(getattr(_cfg, "smoke", False))
    max_frames = int(getattr(_cfg, "sim_view_max_frames", 0)) or None
    artifacts: List[VideoArtifact] = []
    moviepy_missing_all = True
    for rec in _records:
        ep_id = rec.get("episode_id", "unknown")
        sc_id = rec.get("scenario_id", "unknown")
        mp4_path = _out_dir / f"video_{ep_id}.mp4"
        ep = replay_map.get(ep_id) if isinstance(replay_map, dict) else None
        if ep is None or not validate_replay_episode(ep, min_length=2):
            artifacts.append(
                VideoArtifact(
                    artifact_id=f"video_{ep_id}",
                    scenario_id=sc_id,
                    episode_id=ep_id,
                    path_mp4=str(mp4_path),
                    status="skipped",
                    renderer=RENDERER_SIM_VIEW,
                    note=NOTE_INSUFFICIENT_REPLAY,
                )
            )
            continue
        try:
            frame_iter = generate_frames(
                ep, fps=fps, max_frames=(10 if smoke and max_frames is None else max_frames)
            )
            enc = encode_frames(frame_iter, mp4_path, fps=fps, sample_memory=True)
        except Exception as exc:  # noqa: BLE001
            # Best effort cleanup of tiny partial file
            try:
                if mp4_path.exists() and mp4_path.stat().st_size < 1024:
                    mp4_path.unlink()
            except Exception:  # noqa: BLE001
                pass
            artifacts.append(
                VideoArtifact(
                    artifact_id=f"video_{ep_id}",
                    scenario_id=sc_id,
                    episode_id=ep_id,
                    path_mp4=str(mp4_path),
                    status="failed",
                    renderer=RENDERER_SIM_VIEW,
                    note=f"render-error:{exc.__class__.__name__}",
                )
            )
            moviepy_missing_all = False
            continue
        if enc.status == "skipped" and enc.note == "moviepy-missing":
            # Defer decision until we know if all are missing
            continue
        moviepy_missing_all = False
        artifacts.append(
            VideoArtifact(
                artifact_id=f"video_{ep_id}",
                scenario_id=sc_id,
                episode_id=ep_id,
                path_mp4=str(mp4_path),
                status=enc.status,
                renderer=RENDERER_SIM_VIEW,
                note=enc.note,
                encode_time_s=enc.encode_time_s,
                peak_rss_mb=enc.peak_rss_mb,
            )
        )

    # If we produced only insufficient or no artifacts and every encode reported moviepy missing → fallback
    if moviepy_missing_all and not artifacts:
        return []
    return artifacts


def _synthetic_fallback_videos(records, out_dir: Path, cfg) -> List[VideoArtifact]:
    """Generate synthetic fallback + encode with moviepy if possible.

    The existing synthetic generator currently produces mp4s itself; for integration
    with the new encoding pipeline we treat its output as pre-encoded if present.
    Future refactor could switch synthetic path to yield raw frames and call encode_frames.
    """
    raw = generate_fallback_videos(records, out_dir, cfg)
    out: List[VideoArtifact] = []
    for a in raw:
        out.append(
            VideoArtifact(
                artifact_id=getattr(a, "artifact_id", "unknown"),
                scenario_id=getattr(a, "scenario_id", "unknown"),
                episode_id=getattr(a, "episode_id", "unknown"),
                path_mp4=getattr(a, "path_mp4", ""),
                status=getattr(a, "status", "skipped"),
                renderer=RENDERER_SYNTHETIC,
                note=getattr(a, "note", None),
            )
        )
    return out


def _select_records(records, cfg) -> List[dict]:
    max_videos = int(getattr(cfg, "max_videos", 1))
    return records[: max_videos or 1]


def _build_video_artifacts(
    cfg, records: List[dict], videos_dir: Path, replay_map: dict
) -> List[VideoArtifact]:
    smoke = bool(getattr(cfg, "smoke", False))
    disable_videos = bool(getattr(cfg, "disable_videos", False))
    if not records:
        return []
    video_artifacts: List[VideoArtifact] = []
    if not disable_videos and not smoke:
        sim_view_attempt = _attempt_sim_view_videos(records, videos_dir, cfg, replay_map)
        if sim_view_attempt:
            return sim_view_attempt
        video_artifacts = _synthetic_fallback_videos(records, videos_dir, cfg)
        if bool(getattr(cfg, "capture_replay", False)):
            # Reclassify insufficient replay episodes
            rec_index = {r.get("episode_id"): r for r in records}
            for a in video_artifacts:
                rec = rec_index.get(a.episode_id, {})
                steps_raw = rec.get("replay_steps")
                if isinstance(steps_raw, list) and len(steps_raw) < 2:
                    a.renderer = RENDERER_SIM_VIEW
                    a.status = "skipped"
                    a.note = NOTE_INSUFFICIENT_REPLAY
        return video_artifacts
    # Direct skip path
    reason = NOTE_DISABLED if disable_videos else NOTE_SMOKE_MODE
    for rec in records:
        ep_id = rec.get("episode_id", "unknown")
        sc_id = rec.get("scenario_id", "unknown")
        mp4_path = videos_dir / f"{ep_id}.mp4"
        video_artifacts.append(
            VideoArtifact(
                artifact_id=f"video_{ep_id}",
                scenario_id=sc_id,
                episode_id=ep_id,
                path_mp4=str(mp4_path),
                status="skipped",
                renderer=RENDERER_SIM_VIEW if _SIM_VIEW_AVAILABLE else RENDERER_SYNTHETIC,
                note=reason,
            )
        )
    return video_artifacts


def _final_normalize_insufficient(cfg, records: List[dict], video_artifacts: List[VideoArtifact]):
    if not bool(getattr(cfg, "capture_replay", False)):
        return
    rec_index = {r.get("episode_id"): r for r in records}
    for a in video_artifacts:
        rec = rec_index.get(a.episode_id)
        if rec is None:
            continue
        steps_raw = rec.get("replay_steps")
        if isinstance(steps_raw, list) and len(steps_raw) < 2:
            a.renderer = RENDERER_SIM_VIEW
            a.status = "skipped"
            a.note = NOTE_INSUFFICIENT_REPLAY


def generate_visual_artifacts(root: Path, cfg, groups, records) -> dict:  # noqa: C901 (refactored helpers keep body simple)
    """Generate plots and videos and write manifests.

    Returns dict with keys: plots, videos, performance.
    """
    plots_dir = root / "plots"
    videos_dir = root / "videos"
    reports_dir = root / "reports"
    for d in (plots_dir, videos_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Plots timing
    t0 = time.perf_counter()
    raw_plots = generate_plots(groups, records, plots_dir, cfg)
    t1 = time.perf_counter()
    plot_artifacts = _convert_plot_artifacts(raw_plots)

    # Videos timing
    selected_records = _select_records(records, cfg)
    replay_map: dict = {}
    if bool(getattr(cfg, "capture_replay", False)):
        replay_map = extract_replay_episodes(selected_records)
    video_start = time.perf_counter()
    video_artifacts = _build_video_artifacts(cfg, selected_records, videos_dir, replay_map)
    # Attach encode timing placeholders: synthetic path already encoded; SimulationView path not yet implemented
    # For now we only fill encode_time_s / peak_rss_mb if file exists and not skipped.
    for va in video_artifacts:
        if va.status == "success" and Path(va.path_mp4).exists():
            # Can't derive encode_time from legacy path; leave None
            pass
    video_end = time.perf_counter()
    _final_normalize_insufficient(cfg, selected_records, video_artifacts)

    # Performance aggregation (T040/T041)
    first_success = next((v for v in video_artifacts if v.status == "success"), None)
    first_video_time = first_success.encode_time_s if first_success else None
    first_video_peak = first_success.peak_rss_mb if first_success else None
    # Performance meta naming aligned with spec/data-model (plots_time_s, first_video_time_s, etc.)
    plots_time_s = round(t1 - t0, 4)
    videos_time_s = round(video_end - video_start, 4)
    first_video_time_s = first_video_time  # already encode wall time when available
    first_video_peak_mb = first_video_peak
    perf_meta = {
        # Canonical field names (preferred going forward)
        "plots_time_s": plots_time_s,
        "videos_time_s": videos_time_s,
        "first_video_time_s": first_video_time_s,
        "first_video_peak_rss_mb": first_video_peak_mb,
        "plots_over_budget": plots_time_s > 2.0,
        "video_over_budget": (first_video_time_s or 0) > 5.0
        if first_video_time_s is not None
        else False,
        "memory_over_budget": (first_video_peak_mb or 0) > 100
        if first_video_peak_mb is not None
        else False,
        # Backward‑compat legacy keys (scheduled for removal after downstream update window)
        "plots_runtime_sec": plots_time_s,
        "videos_runtime_sec": videos_time_s,
        "first_video_encode_time_s": first_video_time_s,
    }

    _write_json(reports_dir / "plot_artifacts.json", plot_artifacts)
    _write_json(
        reports_dir / "video_artifacts.json",
        [
            {
                "artifact_id": a.artifact_id,
                "scenario_id": a.scenario_id,
                "episode_id": a.episode_id,
                "path_mp4": a.path_mp4,
                "status": a.status,
                "renderer": a.renderer,
                "note": a.note,
                "encode_time_s": a.encode_time_s,
                "peak_rss_mb": a.peak_rss_mb,
            }
            for a in video_artifacts
        ],
    )
    _write_json(reports_dir / "performance_visuals.json", perf_meta)

    # Optional schema validation (T043) triggered by env var
    if os.environ.get("ROBOT_SF_VALIDATE_VISUALS") == "1":
        contracts_dir = (
            Path(__file__).parent / "../../../../specs/127-enhance-benchmark-visual/contracts"
        ).resolve()
        try:
            validated = validate_visual_manifests(reports_dir, contracts_dir)
            logger.info("Validated visual manifests: %s", validated)
        except Exception as exc:  # noqa: BLE001
            logger.error("Visual manifest validation failed: %s", exc)
            raise

    smoke = bool(getattr(cfg, "smoke", False))
    disable_videos = bool(getattr(cfg, "disable_videos", False))
    logger.info(
        "Visual artifacts generated: plots={} videos={} (disable_videos={} smoke={})",
        len(plot_artifacts),
        len(video_artifacts),
        disable_videos,
        smoke,
    )
    return {"plots": plot_artifacts, "videos": video_artifacts, "performance": perf_meta}


__all__ = ["generate_visual_artifacts"]
