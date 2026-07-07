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
    from collections.abc import Sequence
    from pathlib import Path


def save_scenario_thumbnails(  # noqa: PLR0913
    scenarios: list[dict[str, Any]],
    *,
    out_dir: Path | str,
    out_pdf: bool = True,
    publication: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
    caption: bool = False,
    campaign: str | None = None,
    size: str = "single",
    generator_command: str | None = None,
) -> list[ThumbMeta]:
    """Save individual scenario thumbnails.

    Args:
        scenarios: List of scenario configurations.
        out_dir: Output directory for thumbnails.
        out_pdf: Whether to generate PDF versions.
        publication: Render in publication style with provenance sidecars.
        formats: Output formats for publication mode (pdf/png/svg).
        caption: Write per-scenario ``.caption.tex`` sidecars.
        campaign: Campaign name used in provenance and caption fragments.
        size: Publication size preset ("single" or "double").
        generator_command: Generator command recorded in provenance.

    Returns:
        List of thumbnail metadata.
    """
    return _save_scenario_thumbnails(
        scenarios,
        out_dir=out_dir,
        out_pdf=out_pdf,
        publication=publication,
        formats=formats,
        caption=caption,
        campaign=campaign,
        size=size,
        generator_command=generator_command,
    )


def save_montage(  # noqa: PLR0913
    thumb_metas: list[ThumbMeta],
    *,
    out_png: str | Path,
    cols: int = 4,
    out_pdf: str | Path | None = None,
    publication: bool = False,
    formats: Sequence[str] = ("pdf", "png"),
    caption_fragment: str | None = None,
    size: str = "single",
    source_artifacts: list[dict[str, Any]] | None = None,
    generator_command: str | None = None,
) -> dict[str, str]:
    """Save a montage of scenario thumbnails.

    Args:
        thumb_metas: List of thumbnail metadata.
        out_png: Output PNG file path.
        cols: Number of columns in the montage.
        out_pdf: Optional output PDF file path.
        publication: Save as a publication figure with provenance sidecar.
        formats: Output formats for publication mode (pdf/png/svg).
        caption_fragment: Optional LaTeX-ready caption fragment.
        size: Publication size preset ("single" or "double").
        source_artifacts: Source artifact metadata for provenance.
        generator_command: Generator command recorded in provenance.

    Returns:
        Dict mapping written format keys (png/pdf/svg) to file paths.
    """
    return _save_montage(
        thumb_metas,
        out_png=str(out_png),
        cols=cols,
        out_pdf=str(out_pdf) if out_pdf else None,
        publication=publication,
        formats=formats,
        caption_fragment=caption_fragment,
        size=size,
        source_artifacts=source_artifacts,
        generator_command=generator_command,
    )


__all__ = ["ThumbMeta", "save_montage", "save_scenario_thumbnails"]
