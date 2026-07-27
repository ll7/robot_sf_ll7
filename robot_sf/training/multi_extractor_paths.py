"""Filesystem helpers for the multi-extractor training workflow."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.common.artifact_paths import resolve_artifact_path

if TYPE_CHECKING:
    from collections.abc import Iterable

ENV_TMP_OVERRIDE = "ROBOT_SF_MULTI_EXTRACTOR_TMP"
DEFAULT_TMP_ROOT = Path("tmp/multi_extractor_training")


def _normalize_artifact_component(component: str, *, parameter_name: str) -> str:
    """Return a filesystem-safe, single-component artifact path component."""

    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", component.strip()).strip("._-")
    if not normalized:
        raise ValueError(f"{parameter_name} must contain at least one alphanumeric character")
    return normalized


def _normalize_extractor_name(extractor_name: str) -> str:
    """Return a filesystem-safe, single-component extractor directory name."""

    return _normalize_artifact_component(extractor_name, parameter_name="extractor_name")


def validate_unique_extractor_names(extractor_names: Iterable[str]) -> None:
    """Reject names that collide after directory normalization or case folding."""

    names_by_directory: dict[str, str] = {}
    for extractor_name in extractor_names:
        normalized = _normalize_extractor_name(extractor_name)
        directory_key = normalized.casefold()
        first_name = names_by_directory.get(directory_key)
        if first_name is not None:
            raise ValueError(
                "Extractor names collide after normalization/case folding: "
                f"{first_name!r} and {extractor_name!r} -> {normalized!r}"
            )
        names_by_directory[directory_key] = extractor_name


def resolve_base_output_root(env: dict[str, str] | None = None) -> Path:
    """Return the base output directory, honoring environment overrides."""

    env = env or {}
    override = env.get(ENV_TMP_OVERRIDE)
    if override:
        return Path(override).expanduser().resolve()
    return resolve_artifact_path(DEFAULT_TMP_ROOT)


def make_run_directory(
    run_id: str, *, env: dict[str, str] | None = None, timestamp: str | None = None
) -> Path:
    """Create a timestamped training-run directory with a safe run-id component.

    Returns:
        Path: The created run directory path.
    """

    if not run_id:
        raise ValueError("run_id must be a non-empty string")

    safe_run_id = _normalize_artifact_component(run_id, parameter_name="run_id")
    base_root = resolve_base_output_root(env)
    base_root.mkdir(parents=True, exist_ok=True)

    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = base_root / f"{stamp}-{safe_run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "extractors").mkdir(exist_ok=True)
    return run_dir


def make_extractor_directory(run_dir: Path, extractor_name: str) -> Path:
    """Ensure the normalized per-extractor subdirectory exists and return it.

    Returns:
        Path: The path to the extractor-specific directory.
    """

    extractor_dir = run_dir / "extractors" / _normalize_extractor_name(extractor_name)
    extractor_dir.mkdir(parents=True, exist_ok=True)
    return extractor_dir


def summary_paths(run_dir: Path) -> dict[str, Path]:
    """Return the canonical summary artifact locations for a run."""

    return {
        "json": run_dir / "summary.json",
        "markdown": run_dir / "summary.md",
    }
