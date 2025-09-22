"""Visual artifact generation (plots + videos) for Full Classic Benchmark.

Features (FR-001..FR-015):
 - Deterministic manifests for plots & videos
 - SimulationView-first rendering with graceful synthetic fallback
 - Optional dependency degradation (pygame / moviepy)
 - Performance timing metadata
 - Replay‑based frame reconstruction
 - Renderer toggle flag: cfg.video_renderer in {auto, synthetic, sim-view}

This module was reconstructed after refactor to add a renderer toggle; corruption
was resolved by recreating the file (see git history for prior version).
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
    NOTE_MOVIEPY_MISSING,
    NOTE_SIM_VIEW_MISSING,
    NOTE_SMOKE_MODE,
    RENDERER_SIM_VIEW,
    RENDERER_SYNTHETIC,
)
from .visual_deps import moviepy_ready, simulation_view_ready

try:  # Lazy import SimulationView
    from robot_sf.render.sim_view import SimulationView  # type: ignore  # noqa: F401

    _SIM_VIEW_CLS = SimulationView  # touch (lint silence)
    _SIM_VIEW_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SIM_VIEW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _attempt_sim_view_videos(records, out_dir: Path, cfg, replay_map) -> List[VideoArtifact]:
    if not _SIM_VIEW_AVAILABLE or not simulation_view_ready():
        return []
    if not bool(getattr(cfg, "capture_replay", False)):
        return []
    fps = int(getattr(cfg, "video_fps", 10))
    smoke = bool(getattr(cfg, "smoke", False))
    max_frames = int(getattr(cfg, "sim_view_max_frames", 0)) or None
    artifacts: List[VideoArtifact] = []
    moviepy_missing_all = True
    for rec in records:
        ep_id = rec.get("episode_id", "unknown")
        sc_id = rec.get("scenario_id", "unknown")
        mp4_path = out_dir / f"video_{ep_id}.mp4"
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
        if enc.status == "skipped" and enc.note == NOTE_MOVIEPY_MISSING:
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
    if moviepy_missing_all and not artifacts:
        # Signal to caller that SimulationView path attempted but moviepy missing
        # Caller (sim-view forced) will interpret empty attempt as NOTE_MOVIEPY_MISSING.
        return []
    return artifacts


def _synthetic_fallback_videos(records, out_dir: Path, cfg) -> List[VideoArtifact]:
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


def _build_video_artifacts(  # noqa: C901 - explicit branching for clarity
    cfg, records: List[dict], videos_dir: Path, replay_map: dict
) -> List[VideoArtifact]:
    # --- Inner helpers (local to keep namespace clean) ------------------
    def _normalize_mode(raw) -> str:
        m = str(raw or "auto").strip().lower()
        if m == "sim_view":  # alias
            m = "sim-view"
        if m not in {"auto", "synthetic", "sim-view"}:
            logger.warning("Unknown video_renderer '%s' -> auto", m)
            m = "auto"
        return m

    def _build_skipped(reason: str) -> List[VideoArtifact]:
        out: List[VideoArtifact] = []
        for r in records:
            ep_id = r.get("episode_id", "unknown")
            sc_id = r.get("scenario_id", "unknown")
            # Note: historical path naming (no 'video_' prefix) preserved
            mp4_path = videos_dir / f"{ep_id}.mp4"
            out.append(
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
        return out

    def _build_forced_sim_view() -> List[VideoArtifact]:
        attempt = _attempt_sim_view_videos(records, videos_dir, cfg, replay_map)
        if attempt:
            return attempt
        # Forced sim-view but unavailable / no encodes: decide note
        note = NOTE_SIM_VIEW_MISSING
        if (
            bool(getattr(cfg, "capture_replay", False))
            and _SIM_VIEW_AVAILABLE
            and simulation_view_ready()
            and not moviepy_ready()
        ):
            note = NOTE_MOVIEPY_MISSING
        out: List[VideoArtifact] = []
        for r in records:
            ep_id = r.get("episode_id", "unknown")
            sc_id = r.get("scenario_id", "unknown")
            mp4_path = videos_dir / f"video_{ep_id}.mp4"
            out.append(
                VideoArtifact(
                    artifact_id=f"video_{ep_id}",
                    scenario_id=sc_id,
                    episode_id=ep_id,
                    path_mp4=str(mp4_path),
                    status="skipped",
                    renderer=RENDERER_SIM_VIEW,
                    note=note,
                )
            )
        return out

    def _build_auto() -> List[VideoArtifact]:
        sim_attempt = _attempt_sim_view_videos(records, videos_dir, cfg, replay_map)
        if sim_attempt:
            return sim_attempt
        # Use synthetic; insufficient replay normalization is deferred to _final_normalize_insufficient
        return _synthetic_fallback_videos(records, videos_dir, cfg)

    # --- Main logic -----------------------------------------------------
    if not records:
        return []
    smoke = bool(getattr(cfg, "smoke", False))
    disable_videos = bool(getattr(cfg, "disable_videos", False))
    mode = _normalize_mode(getattr(cfg, "video_renderer", "auto"))

    if disable_videos or smoke:
        return _build_skipped(NOTE_DISABLED if disable_videos else NOTE_SMOKE_MODE)
    if mode == "synthetic":
        return _synthetic_fallback_videos(records, videos_dir, cfg)
    if mode == "sim-view":
        return _build_forced_sim_view()
    # auto
    return _build_auto()


def _final_normalize_insufficient(cfg, records: List[dict], video_artifacts: List[VideoArtifact]):
    if not bool(getattr(cfg, "capture_replay", False)):
        return
    mode = str(getattr(cfg, "video_renderer", "auto")).strip().lower()
    if mode in {"sim-view", "sim_view", "synthetic"}:
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_visual_artifacts(root: Path, cfg, groups, records) -> dict:  # noqa: C901
    plots_dir = root / "plots"
    videos_dir = root / "videos"
    reports_dir = root / "reports"
    for d in (plots_dir, videos_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Plots
    t0 = time.perf_counter()
    raw_plots = generate_plots(groups, records, plots_dir, cfg)
    t1 = time.perf_counter()
    plot_artifacts = _convert_plot_artifacts(raw_plots)

    # Videos
    selected_records = _select_records(records, cfg)
    replay_map: dict = {}
    if bool(getattr(cfg, "capture_replay", False)):
        replay_map = extract_replay_episodes(selected_records)
    video_start = time.perf_counter()
    video_artifacts = _build_video_artifacts(cfg, selected_records, videos_dir, replay_map)
    for va in video_artifacts:  # legacy synthetic path cannot supply encode_time
        if va.status == "success" and Path(va.path_mp4).exists():
            pass
    video_end = time.perf_counter()
    _final_normalize_insufficient(cfg, selected_records, video_artifacts)

    first_success = next((v for v in video_artifacts if v.status == "success"), None)
    first_video_time = first_success.encode_time_s if first_success else None
    first_video_peak = first_success.peak_rss_mb if first_success else None
    # Render vs encode split (T040A): approximate render time as total video phase minus encode time
    first_video_render_time = None
    if (
        first_success
        and first_success.encode_time_s is not None
        and first_success.renderer == RENDERER_SIM_VIEW
    ):
        total_video_phase = video_end - video_start
        enc_time = first_success.encode_time_s
        if total_video_phase >= enc_time:
            first_video_render_time = round(total_video_phase - enc_time, 4)
    plots_time_s = round(t1 - t0, 4)
    videos_time_s = round(video_end - video_start, 4)
    perf_meta = {
        "plots_time_s": plots_time_s,
        "videos_time_s": videos_time_s,
        "first_video_time_s": first_video_time,
        "first_video_render_time_s": first_video_render_time,
        "first_video_peak_rss_mb": first_video_peak,
        "plots_over_budget": plots_time_s > 2.0,
        "video_over_budget": (first_video_time or 0) > 5.0
        if first_video_time is not None
        else False,
        "memory_over_budget": (first_video_peak or 0) > 100
        if first_video_peak is not None
        else False,
        # Legacy keys
        "plots_runtime_sec": plots_time_s,
        "videos_runtime_sec": videos_time_s,
        "first_video_encode_time_s": first_video_time,
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

    logger.info(
        "Visual artifacts generated: plots=%d videos=%d (disable_videos=%s smoke=%s mode=%s)",
        len(plot_artifacts),
        len(video_artifacts),
        bool(getattr(cfg, "disable_videos", False)),
        bool(getattr(cfg, "smoke", False)),
        getattr(cfg, "video_renderer", "auto"),
    )
    return {"plots": plot_artifacts, "videos": video_artifacts, "performance": perf_meta}


__all__ = ["generate_visual_artifacts"]
