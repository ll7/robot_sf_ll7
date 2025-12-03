"""Synthetic fallback video generation wrapper (T032).

Purpose
-------
Provides a thin indirection layer around the existing ``videos.generate_videos``
implementation so the higher‑level orchestration in ``visuals.py`` can treat
both SimulationView (``render_sim_view``) and synthetic fallback paths via
similarly named, future‑extensible modules.

Why a wrapper now?
------------------
T032 aims to align interfaces before introducing a unified encoding pipeline
(T033–T035). By centralizing synthetic logic here we can later swap out the
current matplotlib/moviepy approach for a streaming frame generator or shared
encoder without touching orchestration code.

Current behavior
----------------
Simply delegates to ``videos.generate_videos`` and returns its list of
artifacts. No additional processing is performed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import videos as _legacy_videos

if TYPE_CHECKING:
    from pathlib import Path


def generate_fallback_videos(records: list[dict[str, Any]], out_dir: Path, cfg):  # T032
    """Delegate to legacy synthetic video generator.

    Parameters
    ----------
    records : list[dict]
        Selected episode records.
    out_dir : Path
        Target directory for mp4 outputs.
    cfg : Any
        Benchmark configuration object (fields inspected by legacy generator).

    Returns
    -------
    list
        Video artifact metadata from the legacy generator.
    """
    return _legacy_videos.generate_videos(records, out_dir, cfg)


__all__ = ["generate_fallback_videos"]
