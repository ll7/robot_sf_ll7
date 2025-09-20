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

from . import videos as synthetic_videos
from .plots import generate_plots

try:  # Try to import SimulationView lazily (primary renderer)
    from robot_sf.render.sim_view import SimulationView  # type: ignore  # noqa: F401

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


def _attempt_sim_view_videos(_records, _out_dir: Path, _cfg) -> List[VideoArtifact]:
    """Attempt to render videos using SimulationView.

    Current placeholder implementation: falls back immediately if SimulationView not available.
    Future extension: reconstruct environment state replay per episode.
    """
    artifacts: List[VideoArtifact] = []
    if not _SIM_VIEW_AVAILABLE:
        return artifacts
    # Placeholder: we currently do not have replay state; return empty list to force fallback.
    return artifacts


def _synthetic_fallback_videos(records, out_dir: Path, cfg) -> List[VideoArtifact]:
    raw = synthetic_videos.generate_videos(records, out_dir, cfg)
    out: List[VideoArtifact] = []
    for a in raw:
        out.append(
            VideoArtifact(
                artifact_id=getattr(a, "artifact_id", "unknown"),
                scenario_id=getattr(a, "scenario_id", "unknown"),
                episode_id=getattr(a, "episode_id", "unknown"),
                path_mp4=getattr(a, "path_mp4", ""),
                status=getattr(a, "status", "skipped"),
                renderer="synthetic",  # synthetic fallback
                note=getattr(a, "note", None),
            )
        )
    return out


def generate_visual_artifacts(root: Path, cfg, groups, records) -> dict:
    """Generate plots and videos and write manifests.

    Returns dict with keys: plots, videos, performance
    """
    plots_dir = root / "plots"
    videos_dir = root / "videos"
    reports_dir = root / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    t0 = time.perf_counter()
    raw_plots = generate_plots(groups, records, plots_dir, cfg)
    t1 = time.perf_counter()
    plot_artifacts = _convert_plot_artifacts(raw_plots)

    # Videos (SimulationView-first then fallback)
    smoke = bool(getattr(cfg, "smoke", False))
    disable_videos = bool(getattr(cfg, "disable_videos", False))
    max_videos = int(getattr(cfg, "max_videos", 1))
    selected_records = records[: max_videos or 1]
    video_artifacts: List[VideoArtifact] = []

    if selected_records and not disable_videos and not smoke:
        sim_view_attempt = _attempt_sim_view_videos(selected_records, videos_dir, cfg)
        if sim_view_attempt:
            video_artifacts = sim_view_attempt
        else:
            video_artifacts = _synthetic_fallback_videos(selected_records, videos_dir, cfg)
    else:
        # Directly mark skipped cases
        reason = "video generation disabled" if disable_videos else "smoke mode"
        for rec in selected_records:
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
                    renderer="synthetic" if not _SIM_VIEW_AVAILABLE else "simulation_view",
                    note=reason,
                )
            )

    t2 = time.perf_counter()

    perf_meta = {
        "plots_runtime_sec": round(t1 - t0, 4),
        "videos_runtime_sec": round(t2 - t1, 4),
        "plots_over_budget": (t1 - t0) > 2.0,
        "videos_over_budget": (t2 - t1) > 5.0,
    }

    # Serialize manifests
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
            }
            for a in video_artifacts
        ],
    )
    _write_json(reports_dir / "performance_visuals.json", perf_meta)

    logger.info(
        "Visual artifacts generated: plots={} videos={} (disable_videos={} smoke={})",
        len(plot_artifacts),
        len(video_artifacts),
        disable_videos,
        smoke,
    )
    return {"plots": plot_artifacts, "videos": video_artifacts, "performance": perf_meta}


__all__ = ["generate_visual_artifacts"]
