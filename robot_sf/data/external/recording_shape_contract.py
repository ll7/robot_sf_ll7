"""Reusable skip-if-absent shape contract for recording-style external datasets.

Plain-language summary: several license-gated robot-in-crowd datasets (CrowdBot,
SCAND) ship as ROS bag recordings plus exported CSV/JSON tables and a license or
README file, rather than as flat numeric trajectory matrices like ETH/UCY. This
module provides one cheap, structural shape contract shared by those datasets. It
confirms that the registry-declared recording + license/readme layout is present
and that every staged CSV/JSON export parses as a well-formed, non-empty
table/object and that no ROS bag is empty. It never downloads, vendors, or
redistributes dataset bytes, and it asserts no dataset content (no exact frame
counts, sensor values, trajectories, or scene identities).

Availability resolution reuses the canonical registry contract in
``scripts/tools/manage_external_data.py`` (the ``recording`` and
``license_or_readme`` required-path groups) so the loader and the registry never
drift. Per-dataset wrappers (``crowdbot.py``, ``scand.py``) bind a
:class:`RecordingDatasetSpec` and expose the thin module-level API used by the
skip-if-absent shape-contract tests.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

from scripts.tools.manage_external_data import (
    EXTERNAL_DATA_ROOT_ENV,
    check_asset,
    resolve_asset_local_path_by_id,
)

# Recording/export file extensions recognized by the structural contract. These
# mirror the registry ``recording`` required-path group for the recording-style
# assets (ROS bags plus exported CSV/JSON tables).
_BAG_SUFFIX = ".bag"
_CSV_SUFFIX = ".csv"
_JSON_SUFFIX = ".json"

# Filename prefixes accepted for the license/terms/readme required-path group.
_LICENSE_README_PREFIXES = ("readme", "license", "terms")

__all__ = [
    "EXTERNAL_DATA_ROOT_ENV",
    "RecordingDatasetError",
    "RecordingDatasetPaths",
    "RecordingDatasetSpec",
    "availability_report",
    "is_available",
    "load_shape_contract",
    "require_available",
    "resolve_root",
]


class RecordingDatasetError(RuntimeError):
    """Raised when a staged recording-style dataset is present but invalid.

    Per-dataset modules subclass this so their tests and callers can match a
    dataset-specific error type while sharing this module's validation logic.
    """


@dataclass(frozen=True)
class RecordingDatasetSpec:
    """Binding descriptor for one recording-style external dataset.

    Attributes:
        asset_id: Canonical registry id in ``manage_external_data.py``.
        title: Human-facing dataset title used in error messages.
        docs_path: Repository-relative acquisition/layout doc page.
        error_cls: Concrete :class:`RecordingDatasetError` subclass to raise.
    """

    asset_id: str
    title: str
    docs_path: str
    error_cls: type[RecordingDatasetError]


@dataclass(frozen=True)
class RecordingDatasetPaths:
    """Resolved recording and license/readme files for a staged dataset."""

    root: Path
    bag_files: tuple[Path, ...]
    csv_files: tuple[Path, ...]
    json_files: tuple[Path, ...]
    license_or_readme: tuple[Path, ...]
    docs_path: str


def resolve_root(spec: RecordingDatasetSpec, root: Path | str | None = None) -> Path:
    """Return the dataset root, honoring the shared external-data registry.

    Args:
        spec: Dataset binding descriptor.
        root: Explicit dataset root. When ``None`` the shared external-data
            registry resolves the path from ``ROBOT_SF_EXTERNAL_DATA_ROOT`` (or
            the repo-local default). The path is not required to exist.
    """

    if root is not None:
        return Path(root).expanduser().resolve()
    return resolve_asset_local_path_by_id(spec.asset_id).expanduser().resolve()


def availability_report(
    spec: RecordingDatasetSpec, root: Path | str | None = None
) -> dict[str, Any]:
    """Return the canonical registry availability report for the dataset."""

    return check_asset(spec.asset_id, source_path=resolve_root(spec, root))


def is_available(spec: RecordingDatasetSpec, root: Path | str | None = None) -> bool:
    """Return whether the required recording + license/readme layout is staged.

    Returns ``False`` for an absent or incomplete layout so skip-if-absent tests
    can skip cleanly. A present-but-malformed recording/export file does not fail
    this check; it is reported by :func:`load_shape_contract`.
    """

    return bool(availability_report(spec, root).get("ok", False))


def require_available(
    spec: RecordingDatasetSpec, root: Path | str | None = None
) -> RecordingDatasetPaths:
    """Return resolved recording files or raise an actionable acquisition error."""

    report = availability_report(spec, root)
    resolved_root = Path(report["source_path"])
    if not report.get("ok", False):
        missing = ", ".join(str(path) for path in report.get("missing_required_paths") or [])
        action = str(report.get("action") or f"stage the official {spec.asset_id} assets")
        raise spec.error_cls(
            f"{spec.asset_id} is not staged or is incomplete at {resolved_root}. "
            f"Missing required group(s): {missing or 'unknown'}. {action} "
            f"See {spec.docs_path}. You may set {EXTERNAL_DATA_ROOT_ENV} to a shared "
            "external-data root."
        )
    return _resolve_paths(spec, resolved_root)


def _resolve_paths(spec: RecordingDatasetSpec, root: Path) -> RecordingDatasetPaths:
    """Enumerate recording and license/readme files under a staged root.

    Returns:
        The resolved :class:`RecordingDatasetPaths` with bag/csv/json and
        license-or-readme files sorted by path.
    """

    bag_files: list[Path] = []
    csv_files: list[Path] = []
    json_files: list[Path] = []
    license_or_readme: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == _BAG_SUFFIX:
            bag_files.append(path)
        elif suffix == _CSV_SUFFIX:
            csv_files.append(path)
        elif suffix == _JSON_SUFFIX:
            json_files.append(path)
        if path.name.lower().startswith(_LICENSE_README_PREFIXES):
            license_or_readme.append(path)
    return RecordingDatasetPaths(
        root=root,
        bag_files=tuple(bag_files),
        csv_files=tuple(csv_files),
        json_files=tuple(json_files),
        license_or_readme=tuple(license_or_readme),
        docs_path=spec.docs_path,
    )


def _read_text(spec: RecordingDatasetSpec, path: Path) -> str:
    """Read a staged export file as UTF-8 text or fail closed.

    Returns:
        The decoded file text.
    """

    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise spec.error_cls(
            f"{spec.asset_id} file {path} could not be read as UTF-8 text. "
            f"Re-stage per {spec.docs_path}."
        ) from exc


def _validate_csv(spec: RecordingDatasetSpec, path: Path) -> dict[str, Any]:
    """Validate one staged CSV export as a non-empty rectangular table.

    This is a structural check only: it confirms the export parses as CSV, holds
    at least one non-empty row, and every non-empty row shares one column count.
    It asserts no column names or cell values.

    Returns:
        A mapping with the ``row_count`` and ``column_count`` of the table.

    Raises:
        RecordingDatasetError subclass: If the file is empty or non-rectangular.
    """

    text = _read_text(spec, path)
    rows = [row for row in csv.reader(StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows:
        raise spec.error_cls(
            f"{spec.asset_id} CSV export {path} has no non-empty data rows. See {spec.docs_path}."
        )
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise spec.error_cls(
            f"{spec.asset_id} CSV export {path} is not rectangular; rows have "
            f"varying column counts. See {spec.docs_path}."
        )
    return {"row_count": len(rows), "column_count": column_count}


def _validate_json(spec: RecordingDatasetSpec, path: Path) -> dict[str, Any]:
    """Validate one staged JSON export as parseable and summarize its top type.

    Returns:
        A mapping with the ``top_level_type`` name of the parsed JSON payload.

    Raises:
        RecordingDatasetError subclass: If the file does not parse as JSON.
    """

    text = _read_text(spec, path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise spec.error_cls(
            f"{spec.asset_id} JSON export {path} is not valid JSON: {exc}. See {spec.docs_path}."
        ) from exc
    return {"top_level_type": type(payload).__name__}


def _validate_bag(spec: RecordingDatasetSpec, path: Path) -> dict[str, Any]:
    """Validate one staged ROS bag structurally (present and non-empty).

    Full ROS bag decoding needs optional heavy dependencies, so the shape
    contract only requires a non-empty file. Raises when the bag is zero bytes.

    Returns:
        A mapping with the ``size_bytes`` of the staged bag.
    """

    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        raise spec.error_cls(
            f"{spec.asset_id} ROS bag {path} is empty (0 bytes). Re-stage per {spec.docs_path}."
        )
    return {"size_bytes": size_bytes}


def load_shape_contract(
    spec: RecordingDatasetSpec, root: Path | str | None = None
) -> dict[str, Any]:
    """Load and validate the cheap recording-style shape contract.

    Args:
        spec: Dataset binding descriptor.
        root: Explicit dataset root, or ``None`` to resolve via the shared
            external-data registry.

    Returns:
        A structural contract mapping with recording counts and per-file shape
        metadata (CSV row/column counts, JSON top-level type, bag byte sizes) and
        the resolved license/readme files. No dataset content is asserted.

    Raises:
        RecordingDatasetError subclass: If the dataset is absent/incomplete or any
            staged recording/export file is malformed.
    """

    paths = require_available(spec, root)
    csv_shapes = {
        path.relative_to(paths.root).as_posix(): _validate_csv(spec, path)
        for path in paths.csv_files
    }
    json_shapes = {
        path.relative_to(paths.root).as_posix(): _validate_json(spec, path)
        for path in paths.json_files
    }
    bag_shapes = {
        path.relative_to(paths.root).as_posix(): _validate_bag(spec, path)
        for path in paths.bag_files
    }
    return {
        "asset_id": spec.asset_id,
        "root": str(paths.root),
        "docs_path": paths.docs_path,
        "recording_counts": {
            "bag": len(paths.bag_files),
            "csv": len(paths.csv_files),
            "json": len(paths.json_files),
        },
        "csv_files": csv_shapes,
        "json_files": json_shapes,
        "bag_files": bag_shapes,
        "license_or_readme": [
            path.relative_to(paths.root).as_posix() for path in paths.license_or_readme
        ],
    }
