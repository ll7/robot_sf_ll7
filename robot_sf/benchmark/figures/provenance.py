"""Provenance sidecar builder for publication figures.

This module handles creation of provenance metadata files (.provenance.json)
for generated figures, including source artifact hashes, seeds, config hashes,
and generator commands.

Usage:
    from robot_sf.benchmark.figures.provenance import build_provenance, write_provenance

    provenance = build_provenance(
        source_artifacts=[{"path": "data.jsonl", "hash": "abc123"}],
        generator_command="scripts/generate_figures.py --episodes data.jsonl",
    )
    write_provenance(Path("output/figure"), provenance)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_sha_short(length: int = 7) -> str:
    """Get short git SHA for current HEAD.

    Returns:
        Short git SHA string or "unknown" if unavailable.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", f"--short={length}", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        return sha or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


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


@dataclass
class ProvenanceConfig:
    """Configuration for building provenance metadata."""

    source_artifacts: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    config_path: str | None = None
    scenario_matrix_path: str | None = None
    generator_command: str | None = None
    figure_formats: list[str] = field(default_factory=list)
    output_files: list[Path] = field(default_factory=list)
    claim_boundary: str | None = None


def build_provenance(config: ProvenanceConfig | None = None, **kwargs: Any) -> dict[str, Any]:
    """Build provenance metadata dictionary.

    Args:
        config: Optional ProvenanceConfig object. If provided, kwargs are ignored.
        **kwargs: Individual configuration values (for backward compatibility).

    Returns:
        Dictionary of provenance metadata.
    """
    if config is None:
        config = ProvenanceConfig(**kwargs)

    provenance: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_commit": _git_sha_short(),
        "generator_command": config.generator_command or "unknown",
        "source_artifacts": config.source_artifacts or [],
        "seeds": config.seeds or [],
        "episode_ids": config.episode_ids or [],
        "figure_formats": config.figure_formats or [],
        "output_hashes": {},
    }

    if config.config_path:
        config_path = Path(config.config_path)
        if config_path.exists():
            provenance["config_path"] = str(config_path)
            provenance["config_hash"] = _file_sha256(config_path)

    if config.scenario_matrix_path:
        matrix_path = Path(config.scenario_matrix_path)
        if matrix_path.exists():
            provenance["scenario_matrix_path"] = str(matrix_path)
            provenance["scenario_matrix_hash"] = _file_sha256(matrix_path)

    if config.output_files:
        for output_file in config.output_files:
            if output_file.exists():
                provenance["output_hashes"][output_file.name] = _file_sha256(output_file)

    if config.claim_boundary:
        provenance["claim_boundary"] = config.claim_boundary

    return provenance


def write_provenance(
    output_base: Path,
    provenance: dict[str, Any],
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Write provenance sidecar file.

    Args:
        output_base: Base path for the figure (without extension).
        provenance: Provenance metadata dictionary.
        timestamp: Optional timestamp override for deterministic testing.

    Returns:
        Path to written provenance file.
    """
    provenance_path = output_base.parent / f"{output_base.name}.provenance.json"

    # Use provided timestamp or current time
    provenance_copy = dict(provenance)
    if timestamp is not None:
        provenance_copy["generated_at"] = timestamp.isoformat()

    provenance_path.write_text(
        json.dumps(provenance_copy, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return provenance_path


def latex_escape(value: str) -> str:
    """Escape special LaTeX characters in a string.

    Escapes at minimum: ``_ % & # $ { } ~ ^ \\``

    Args:
        value: Raw string that may contain LaTeX special characters.

    Returns:
        LaTeX-safe string.
    """
    # Order matters: backslash must be escaped first to avoid double-escaping
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("_", "\\_"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    result = value
    for char, escaped in replacements:
        result = result.replace(char, escaped)
    return result


def build_caption_fragment(
    *,
    campaign_name: str | None = None,
    episode_ids: list[str] | None = None,
    scenario_id: str | None = None,
) -> str:
    """Build a LaTeX-ready caption fragment.

    Args:
        campaign_name: Name of the benchmark campaign.
        episode_ids: List of episode identifiers.
        scenario_id: Scenario identifier.

    Returns:
        LaTeX-ready one-liner caption.
    """
    parts = []
    if campaign_name:
        parts.append(f"Campaign: {latex_escape(str(campaign_name))}")
    if scenario_id:
        parts.append(f"Scenario: {latex_escape(str(scenario_id))}")
    if episode_ids:
        escaped_ids = [latex_escape(str(eid)) for eid in episode_ids[:3]]
        ep_str = ", ".join(escaped_ids)
        if len(episode_ids) > 3:
            ep_str += f", ... ({len(episode_ids)} total)"
        parts.append(f"Episodes: {ep_str}")

    if not parts:
        return "\\textit{No provenance data}"

    return " | ".join(parts)


def write_caption_fragment(
    output_base: Path,
    caption: str,
) -> Path:
    """Write caption fragment file.

    Args:
        output_base: Base path for the figure (without extension).
        caption: LaTeX-ready caption text.

    Returns:
        Path to written caption file.
    """
    caption_path = output_base.parent / f"{output_base.name}.caption.tex"
    caption_path.write_text(caption, encoding="utf-8")
    return caption_path
