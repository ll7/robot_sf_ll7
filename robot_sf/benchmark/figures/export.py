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

# Provenance fields embedded inside PDF/PNG file metadata (best-effort).
# The ``.provenance.json`` sidecar remains the canonical machine-readable record.
_PROVENANCE_METADATA_FIELDS = (
    "source_artifacts",
    "seeds",
    "episode_ids",
    "config_hash",
    "scenario_matrix_hash",
    "repo_commit",
    "generator_command",
    "figure_formats",
    "claim_boundary",
)


def _provenance_embedded_payload(provenance: dict[str, Any]) -> dict[str, Any]:
    """Return a compact provenance subset suitable for file-metadata embedding.

    Omits volatile or output-specific fields (timestamps, output hashes) so the
    embedded payload carries the stable provenance (source hashes, seeds, config
    hash, repo commit, generator command) that must travel inside the file.

    Args:
        provenance: Full provenance metadata dictionary.

    Returns:
        Compact dictionary of stable provenance fields.
    """
    payload: dict[str, Any] = {}
    for key in _PROVENANCE_METADATA_FIELDS:
        value = provenance.get(key)
        if value not in (None, "", [], {}):
            payload[key] = value
    return payload


def _embedded_metadata(
    provenance: dict[str, Any],
    figure_name: str,
    fmt: str,
) -> dict[str, Any]:
    """Build a ``savefig`` metadata dict embedding provenance for pdf or png.

    PDF uses the standard info-dict keys (Title/Author/Subject/Keywords/Creator);
    the compact provenance JSON is stored in ``Keywords``. PNG uses arbitrary
    Pillow text chunks; the compact JSON is stored in a ``Provenance`` chunk.

    Args:
        provenance: Full provenance metadata dictionary.
        figure_name: Figure base name (used as PDF Title).
        fmt: Output format (``"pdf"`` or ``"png"``).

    Returns:
        Metadata dict accepted by ``Figure.savefig(metadata=...)`` for the format.
    """
    compact = json.dumps(
        _provenance_embedded_payload(provenance),
        sort_keys=True,
        separators=(",", ":"),
    )

    summary_parts: list[str] = []
    sources = provenance.get("source_artifacts") or []
    if sources:
        summary_parts.append(f"sources={len(sources)}")
    episodes = provenance.get("episode_ids") or []
    if episodes:
        summary_parts.append(f"episodes={len(episodes)}")
    if provenance.get("repo_commit"):
        summary_parts.append(f"commit={provenance['repo_commit']}")
    summary = " ".join(summary_parts) or "Publication figure"
    creator = str(provenance.get("generator_command") or "robot_sf benchmark visualization")

    if fmt == "pdf":
        return {
            "Title": figure_name,
            "Author": "Robot SF Benchmark",
            "Subject": summary,
            "Keywords": compact,
            "Creator": creator,
        }
    # png: Pillow text chunks accept arbitrary string keys
    return {
        "Software": creator,
        "Description": summary,
        "Provenance": compact,
    }


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

        # Embed provenance in PDF/PNG metadata (best-effort; sidecar is canonical).
        # SVG has no comparable metadata channel here, so it relies on the sidecar.
        if fmt == "pdf":
            save_kwargs["metadata"] = _embedded_metadata(provenance, output_base.name, "pdf")
        elif fmt == "png":
            save_kwargs["dpi"] = 300
            save_kwargs["metadata"] = _embedded_metadata(provenance, output_base.name, "png")

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
