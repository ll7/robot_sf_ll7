"""Helpers for rerouting artifact outputs during automated tests.

When the environment variable ``ROBOT_SF_ARTIFACT_ROOT`` is set, artifact paths
inside the repository are remapped beneath that directory. This allows pytest
runs to execute examples without polluting the working tree while keeping the
default behavior untouched for normal usage.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_OVERRIDE_ENV = "ROBOT_SF_ARTIFACT_ROOT"


@lru_cache(maxsize=1)
def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_artifact_override_root() -> Path | None:
    """Return the artifact override root when configured via the environment."""

    override = os.environ.get(_OVERRIDE_ENV)
    if not override:
        return None
    return Path(override).expanduser().resolve()


def resolve_artifact_path(path: str | Path) -> Path:
    """Resolve ``path`` to its on-disk location, honoring the override root."""

    candidate = Path(path)
    override_root = get_artifact_override_root()
    if override_root is None:
        return candidate.resolve()

    target_root = override_root
    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(_repository_root())
        except ValueError:
            # Path outside repository – leave untouched.
            return candidate
    else:
        relative = candidate

    return (target_root / relative).resolve()
