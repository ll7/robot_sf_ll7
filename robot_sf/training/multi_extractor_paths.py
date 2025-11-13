"""Filesystem helpers for the multi-extractor training workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from robot_sf.common.artifact_paths import resolve_artifact_path

ENV_TMP_OVERRIDE = "ROBOT_SF_MULTI_EXTRACTOR_TMP"
DEFAULT_TMP_ROOT = Path("tmp/multi_extractor_training")


def resolve_base_output_root(env: Optional[dict[str, str]] = None) -> Path:
    """Return the base output directory, honoring environment overrides."""

    env = env or {}
    override = env.get(ENV_TMP_OVERRIDE)
    if override:
        return Path(override).expanduser().resolve()
    return resolve_artifact_path(DEFAULT_TMP_ROOT)


def make_run_directory(
    run_id: str, *, env: Optional[dict[str, str]] = None, timestamp: Optional[str] = None
) -> Path:
    """Create and return the timestamped directory for a training run."""

    if not run_id:
        raise ValueError("run_id must be a non-empty string")

    base_root = resolve_base_output_root(env)
    base_root.mkdir(parents=True, exist_ok=True)

    stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = base_root / f"{stamp}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "extractors").mkdir(exist_ok=True)
    return run_dir


def make_extractor_directory(run_dir: Path, extractor_name: str) -> Path:
    """Ensure the per-extractor subdirectory exists and return it."""

    if not extractor_name:
        raise ValueError("extractor_name must be provided")

    extractor_dir = run_dir / "extractors" / extractor_name
    extractor_dir.mkdir(parents=True, exist_ok=True)
    return extractor_dir


def summary_paths(run_dir: Path) -> dict[str, Path]:
    """Return the canonical summary artifact locations for a run."""

    return {
        "json": run_dir / "summary.json",
        "markdown": run_dir / "summary.md",
    }
