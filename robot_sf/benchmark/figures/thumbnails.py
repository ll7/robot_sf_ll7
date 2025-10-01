"""Scenario thumbnails montage builder module.

This module provides functionality for building montages of scenario
thumbnails and integrating with the figure orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.scenario_thumbnails import (
    ThumbMeta,
)
from robot_sf.benchmark.scenario_thumbnails import (
    save_montage as _save_montage,
)
from robot_sf.benchmark.scenario_thumbnails import (
    save_scenario_thumbnails as _save_scenario_thumbnails,
)

if TYPE_CHECKING:
    from pathlib import Path


def save_scenario_thumbnails(
    scenarios: list[dict[str, Any]],
    *,
    out_dir: Path | str,
    out_pdf: bool = True,
) -> list[ThumbMeta]:
    """Save individual scenario thumbnails.

    Args:
        scenarios: List of scenario configurations.
        out_dir: Output directory for thumbnails.
        out_pdf: Whether to generate PDF versions.

    Returns:
        List of thumbnail metadata.
    """
    return _save_scenario_thumbnails(scenarios, out_dir=out_dir, out_pdf=out_pdf)


def save_montage(
    thumb_metas: list[ThumbMeta],
    *,
    out_png: str | Path,
    cols: int = 4,
    out_pdf: str | Path | None = None,
) -> None:
    """Save a montage of scenario thumbnails.

    Args:
        thumb_metas: List of thumbnail metadata.
        out_png: Output PNG file path.
        cols: Number of columns in the montage.
        out_pdf: Optional output PDF file path.
    """
    _save_montage(
        thumb_metas,
        out_png=str(out_png),
        cols=cols,
        out_pdf=str(out_pdf) if out_pdf else None,
    )


__all__ = ["ThumbMeta", "save_montage", "save_scenario_thumbnails"]
