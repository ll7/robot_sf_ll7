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
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.figures.provenance import (
    build_provenance,
    write_caption_fragment,
    write_provenance,
)

_MAX_METADATA_VALUE_LENGTH = 1024


def save_publication_figure(  # noqa: C901
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

    # Build compact provenance summary for metadata embedding (must be short)
    prov_json = _compact_provenance_json(provenance)

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

        # Add metadata for PDF
        if fmt == "pdf":
            save_kwargs["metadata"] = _build_pdf_metadata(
                output_base.name, prov_json
            )
        elif fmt == "png":
            save_kwargs["dpi"] = 300

        plt.savefig(output_path, **save_kwargs)

        # Embed provenance in PNG metadata via Pillow
        if fmt == "png":
            _embed_provenance_in_png(output_path, provenance)
            # Update hash after Pillow modifies the file
            provenance.setdefault("output_hashes", {})[
                output_path.name
            ] = _file_sha256(output_path)
        else:
            saved_files.append(output_path)
            if "output_hashes" not in provenance:
                provenance["output_hashes"] = {}
            provenance["output_hashes"][output_path.name] = _file_sha256(output_path)

        if fmt == "png":
            saved_files.append(output_path)

    # Write provenance sidecar
    provenance_path = write_provenance(output_base, provenance, timestamp=timestamp)
    saved_files.append(provenance_path)

    # Write caption fragment if provided
    if caption_fragment is not None:
        caption_path = write_caption_fragment(output_base, caption_fragment)
        saved_files.append(caption_path)

    return saved_files


def _compact_provenance_json(provenance: dict[str, Any]) -> str:
    """Build a compact JSON string of provenance for metadata embedding.

    Keeps only fields small enough for file format metadata limits.

    Args:
        provenance: Provenance metadata dictionary.

    Returns:
        Compact JSON string truncated to metadata-safe length.
    """
    compact: dict[str, Any] = {"gen": provenance.get("generator_command", "unknown")}
    if provenance.get("repo_commit"):
        compact["commit"] = provenance["repo_commit"]
    if provenance.get("source_artifacts"):
        compact["artifacts"] = [
            {"p": a.get("path", "?"), "h": a.get("hash", "?")[:12]}
            for a in provenance["source_artifacts"][:5]
        ]
    if provenance.get("seeds"):
        compact["seeds"] = provenance["seeds"][:10]
    if provenance.get("episode_ids"):
        compact["episodes"] = provenance["episode_ids"][:10]
    raw = json.dumps(compact, separators=(",", ":"), sort_keys=True)
    return raw[:_MAX_METADATA_VALUE_LENGTH]


def _build_pdf_metadata(
    figure_name: str,
    provenance_json: str,
) -> dict[str, str]:
    """Build PDF metadata dictionary for matplotlib savefig.

    Uses only XMP/PDF-standard keys recognized by matplotlib's PDF backend.
    Embeds compact provenance in the Creator field to preserve discoverability
    while remaining within the standard key set.

    Args:
        figure_name: Name of the figure (without extension).
        provenance_json: Compact provenance JSON string.

    Returns:
        Metadata dict suitable for matplotlib's savefig metadata parameter.
    """
    prov = provenance_json[:_MAX_METADATA_VALUE_LENGTH]
    return {
        "Title": figure_name,
        "Author": "Robot SF Benchmark",
        "Subject": "Publication figure",
        "Creator": f"robot_sf benchmark visualization | provenance:{prov}",
    }


def _embed_provenance_in_png(path: Path, provenance: dict[str, Any]) -> None:
    """Embed provenance metadata into a PNG file using Pillow PngInfo.

    Reads the PNG, adds metadata chunks, and writes back.

    Args:
        path: Path to the PNG file.
        provenance: Provenance metadata dictionary.
    """
    try:
        from PIL import Image  # noqa: PLC0415
        from PIL.PngImagePlugin import PngInfo  # noqa: PLC0415
    except ImportError:
        return

    prov_data = _compact_provenance_json(provenance)
    try:
        img = Image.open(path)
        png_info = PngInfo()
        png_info.add_text(
            "RobotSF-Provenance", prov_data[:_MAX_METADATA_VALUE_LENGTH]
        )
        png_info.add_text("Software", "robot_sf benchmark visualization")
        png_info.add_text("Title", path.stem)

        # Copy existing info if present
        existing = getattr(img, "text", None)
        if isinstance(existing, dict):
            for k, v in existing.items():
                val = str(v) if not isinstance(v, str) else v
                png_info.add_text(k, val)

        img.save(path, "PNG", pnginfo=png_info)
    except (OSError, ValueError, KeyError, TypeError):
        # Non-fatal: sidecar JSON remains canonical source
        pass


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
