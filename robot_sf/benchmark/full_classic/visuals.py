"""Visual artifact generation (plots + videos) for Full Classic Benchmark.

Implements spec FR-001..FR-015:
 - Deterministic plot & video artifact manifests
 - SimulationView-first (placeholder stub: fallback to synthetic video module)
 - Graceful degradation on optional dependency absence
 - Performance timing meta
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from loguru import logger

from .plots import generate_plots
from .render_synthetic import generate_fallback_videos
from .replay import extract_replay_episodes, validate_replay_episode
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
    """Attempt to render videos using SimulationView (placeholder).

    Ordering rationale (FR-008 vs readiness probe):
    - If SimulationView module import failed entirely → immediate fallback (empty list).
    - If capture enabled, we FIRST emit skip artifacts for insufficient replay even if
      the runtime readiness probe would fail. This ensures the skip note uses the
      SimulationView renderer identifier, matching test expectations and providing
      clearer diagnostics ("why no SimulationView video was produced").
    - Only if we have no skip artifacts do we check runtime readiness; failing that
      we fall back (empty list) letting synthetic path proceed.
    - Actual frame rendering not implemented yet (future tasks T031+T035).
    """
    artifacts: List[VideoArtifact] = []
    capture_enabled = bool(getattr(_cfg, "capture_replay", False))
    if capture_enabled:
        for rec in _records:
            ep_id = rec.get("episode_id", "unknown")
            sc_id = rec.get("scenario_id", "unknown")
            rep = replay_map.get(ep_id) if isinstance(replay_map, dict) else None
            if rep is None or not validate_replay_episode(rep, min_length=2):
                artifacts.append(
                    VideoArtifact(
                        artifact_id=f"video_{ep_id}",
                        scenario_id=sc_id,
                        episode_id=ep_id,
                        path_mp4=str(_out_dir / f"video_{ep_id}.mp4"),
                        status="skipped",
                        renderer=RENDERER_SIM_VIEW,
                        note=NOTE_INSUFFICIENT_REPLAY,
                    )
                )
        if artifacts:
            return artifacts
    # If SimulationView import failed entirely we cannot proceed further; return empty to trigger fallback
    if not _SIM_VIEW_AVAILABLE:
        return []
    # No skip artifacts produced; verify readiness for a (future) render attempt
    if not simulation_view_ready():  # probe failure → fallback
        return []
    # Placeholder: real rendering path not implemented yet; return empty list to trigger fallback
    return []


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

    perf_meta = {
        "plots_runtime_sec": round(t1 - t0, 4),
        "videos_runtime_sec": round(video_end - video_start, 4),
        "plots_over_budget": (t1 - t0) > 2.0,
        "videos_over_budget": (video_end - video_start) > 5.0,
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
