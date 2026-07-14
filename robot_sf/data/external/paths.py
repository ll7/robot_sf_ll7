"""Library-owned paths for locally staged external data."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_DATA_ROOT_ENV = "ROBOT_SF_EXTERNAL_DATA_ROOT"


def external_data_root() -> Path | None:
    """Return the configured shared external-data root, when one is set."""

    raw_root = os.environ.get(EXTERNAL_DATA_ROOT_ENV)
    if raw_root is None or not raw_root.strip():
        return None
    return Path(raw_root).expanduser().resolve()


def resolve_external_data_path(
    asset_id: str,
    default_path: Path,
    *,
    root: Path | None = None,
) -> Path:
    """Resolve an asset path using the shared root when configured.

    Returns:
        The shared-root path when configured, otherwise ``default_path``.
    """

    shared_root = external_data_root() if root is None else root
    if shared_root is None:
        return default_path
    return shared_root / asset_id
