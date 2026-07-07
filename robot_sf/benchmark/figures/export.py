"""Vector and raster export helper for publication figures.

This module provides utilities for saving matplotlib figures in multiple
formats (PDF, PNG, SVG) with embedded metadata and provenance sidecars.

Usage:
    from robot_sf.benchmark.figures.export import save_publication_figure

    paths = save_publication_figure(
        fig,
        output_base=Path("output/figure"),
        formats=("pdf", "png"),
        provenance={"source_artifacts": [...]},
        caption_fragment="Campaign: example | Episodes: ep1, ep2",
    )
"""

from __future__ import annotations

import hashlib
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.figures.provenance import (
    build_provenance,
    write_caption_fragment,
    write_provenance,
)


def save_publication_figure(
    fig,
    output_base: Path,
    *,
    formats: tuple[str, ...] = ("pdf",),
    provenance: dict[str, Any] | None = None,
    caption_fragment: str | None = None,
    timestamp=None,
) -> list[Path]:
    """Save a figure in multiple formats with provenance sidecar.

    Args:
        fig: Matplotlib figure object.
        output_base: Base path for output (without extension).
        formats: Tuple of output formats ("pdf", "png", "svg").
        provenance: Optional provenance metadata dictionary.
        caption_fragment: Optional LaTeX-ready caption fragment.
        timestamp: Optional timestamp override for deterministic testing.

    Returns:
        List of paths to generated files (including sidecar).

    Raises:
        ValueError: If no valid formats specified.
        RuntimeError: If matplotlib is not available.
    """
    if not formats:
        raise ValueError("At least one format must be specified")

    # Ensure matplotlib is available
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ImportError:
        raise RuntimeError("matplotlib is required for figure export")

    # Ensure output directory exists
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Build provenance if not provided
    if provenance is None:
        provenance = build_provenance(
            generator_command="save_publication_figure",
            figure_formats=list(formats),
        )

    # Save in each format
    saved_files: list[Path] = []
    for fmt in formats:
        if fmt not in ("pdf", "png", "svg"):
            raise ValueError(f"Unsupported format: {fmt!r}. Must be 'pdf', 'png', or 'svg'.")

        output_path = output_base.parent / f"{output_base.name}.{fmt}"

        # Save with appropriate settings
        save_kwargs: dict[str, Any] = {
            "format": fmt,
            "bbox_inches": "tight",
            "pad_inches": 0.05,
        }

        # Add metadata for PDF and PNG
        if fmt == "pdf":
            save_kwargs["metadata"] = {
                "Title": output_base.name,
                "Author": "Robot SF Benchmark",
                "Subject": "Publication figure",
                "Creator": "robot_sf benchmark visualization",
            }
        elif fmt == "png":
            save_kwargs["dpi"] = 300

        plt.savefig(output_path, **save_kwargs)
        saved_files.append(output_path)

        # Add to provenance output hashes
        if "output_hashes" not in provenance:
            provenance["output_hashes"] = {}
        provenance["output_hashes"][output_path.name] = _file_sha256(output_path)

    # Write provenance sidecar
    provenance_path = write_provenance(output_base, provenance, timestamp=timestamp)
    saved_files.append(provenance_path)

    # Write caption fragment if provided
    if caption_fragment is not None:
        caption_path = write_caption_fragment(output_base, caption_fragment)
        saved_files.append(caption_path)

    return saved_files


def _file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Args:
        path: Path to file.

    Returns:
        Hex digest of SHA-256 hash.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
