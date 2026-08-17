#!/usr/bin/env python3
"""Deterministic selector for the repository's base-sensitive test surfaces."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

SELECTOR_VERSION = "pytest-marker-files.v1"
BASE_SENSITIVE = "base_sensitive"
ORDINARY = "ordinary"
UNKNOWN = "unknown"


def _normalize_repo_path(path: str) -> str:
    """Normalize only repeated ``./`` prefixes without changing hidden directories."""
    normalized = path
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def find_base_sensitive_test_files(repo_root: Path) -> list[str]:
    """Return repository-relative test files declaring the ``base_sensitive`` marker.

    The marker is the deliberately small, explicit contract from issue #5559.  Keeping the
    selector in one module lets the gate and queue policy use identical file classification.
    """
    matches: list[str] = []
    ignored_dirs = {".git", ".venv", "venv", "build", "dist", "__pycache__", "third_party"}
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [directory for directory in dirs if directory not in ignored_dirs]
        for filename in files:
            if not (filename.startswith("test_") and filename.endswith(".py")):
                continue
            path = Path(root) / filename
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if BASE_SENSITIVE in text:
                matches.append(path.relative_to(repo_root).as_posix())
    return sorted(matches)


def classify_changed_files(
    changed_files: Sequence[str] | None,
    *,
    sensitive_files: Sequence[str],
) -> dict[str, Any]:
    """Classify a changed-file list using the explicit marker-file selector.

    ``None`` means the changed-file inventory was unavailable and therefore cannot be treated as
    an ordinary PR.  A present inventory with no marker-file intersection is ordinary and may use
    the immediate current-base compare-and-swap path at merge time.
    """
    normalized_sensitive = {_normalize_repo_path(str(path)) for path in sensitive_files}
    if changed_files is None:
        return {
            "status": UNKNOWN,
            "selector": SELECTOR_VERSION,
            "changed_files": [],
            "changed_sensitive_files": [],
            "all_sensitive_files": sorted(normalized_sensitive),
            "reason": "changed_file_inventory_unavailable",
        }

    normalized_changed = sorted(
        {_normalize_repo_path(str(path)) for path in changed_files if str(path)}
    )
    changed_sensitive = sorted(path for path in normalized_changed if path in normalized_sensitive)
    return {
        "status": BASE_SENSITIVE if changed_sensitive else ORDINARY,
        "selector": SELECTOR_VERSION,
        "changed_files": normalized_changed,
        "changed_sensitive_files": changed_sensitive,
        "all_sensitive_files": sorted(normalized_sensitive),
        "reason": (
            "changed_files_intersect_marker_selector"
            if changed_sensitive
            else "no_changed_file_intersects_marker_selector"
        ),
    }
