"""inD intersection-drone external-data shape contract.

Plain-language summary: this loader only inspects locally staged inD files
(naturalistic road-user trajectories at German intersections, Bock et al. 2020).
It never downloads, vendors, or redistributes the request-gated inD dataset
bytes. The contract is cheap and structural: it confirms the documented
per-recording file group exists (``*_tracks.csv``, ``*_tracksMeta.csv``,
``*_recordingMeta.csv``, and a background image) and that each CSV parses as a
non-empty, rectangular table carrying the header columns inD publishes. It
intentionally avoids any content assertion (exact frame counts, coordinates,
class distributions, or scene-specific values) and makes no benchmark or
prediction-comparability claim.

The expected layout, sources, and staging workflow are documented in
``docs/datasets/ind.md``; keep the recording-group descriptors below aligned with
that page rather than duplicating layout rules elsewhere.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.tools.manage_external_data import (
    EXTERNAL_DATA_ROOT_ENV,
    resolve_asset_local_path_by_id,
)

IND_ASSET_ID = "ind-crossings"
ACQUISITION_DOC = "docs/datasets/ind.md"

# inD publishes one group of files per recording, keyed by a leading recording id
# in the file stem, for example ``00_tracks.csv`` / ``07_recordingMeta.csv``. The
# suffix after that id names the file role. A recording is only considered
# complete when all three CSV roles and a background image are present, so a lone
# stray CSV never satisfies the contract.
_TRACKS_SUFFIX = "_tracks.csv"
_TRACKS_META_SUFFIX = "_tracksMeta.csv"
_RECORDING_META_SUFFIX = "_recordingMeta.csv"
_BACKGROUND_SUFFIX = "_background.png"

# Header columns each inD CSV role is documented to carry. These are the stable,
# published schema names (not scene content), so requiring them distinguishes a
# real inD-shaped CSV from an arbitrary table without asserting any values. Only a
# defensible subset is required so upstream column additions do not break staging.
_TRACKS_REQUIRED_COLUMNS = ("trackId", "frame", "xCenter", "yCenter")
_TRACKS_META_REQUIRED_COLUMNS = ("trackId", "class")
_RECORDING_META_REQUIRED_COLUMNS = ("recordingId", "frameRate")

# Coordinate/id columns in ``*_tracks.csv`` that must parse as finite floats. This
# is a structural parseability floor matching the sibling ETH/UCY contract, not a
# content claim about trajectory geometry.
_TRACKS_FINITE_COLUMNS = ("frame", "xCenter", "yCenter")


class IndDataError(RuntimeError):
    """Raised when staged inD data is present but structurally invalid."""


@dataclass(frozen=True)
class IndRecordingPaths:
    """Resolved file group for one staged inD recording.

    Attributes:
        recording_id: Recording key parsed from the shared file-stem prefix.
        tracks: ``*_tracks.csv`` trajectory-sample table.
        tracks_meta: ``*_tracksMeta.csv`` per-track class/size metadata table.
        recording_meta: ``*_recordingMeta.csv`` recording-level metadata table.
        background: Aerial background image for the recording.
    """

    recording_id: str
    tracks: Path
    tracks_meta: Path
    recording_meta: Path
    background: Path


@dataclass(frozen=True)
class IndDatasetPaths:
    """Resolved recordings for a staged inD dataset."""

    root: Path
    recordings: tuple[IndRecordingPaths, ...]
    docs_path: str = ACQUISITION_DOC


def dataset_root(root: Path | str | None = None) -> Path:
    """Return the inD dataset root, honoring the shared external-data root.

    Args:
        root: Explicit dataset root. When ``None`` the shared external-data
            registry resolves the path from ``ROBOT_SF_EXTERNAL_DATA_ROOT`` (or the
            repo-local default).

    Returns:
        The absolute dataset root path. The path is not required to exist.
    """

    if root is not None:
        return Path(root).expanduser().resolve()
    return resolve_asset_local_path_by_id(IND_ASSET_ID).expanduser().resolve()


def _recording_id_for(path: Path, suffix: str) -> str:
    """Return the recording-id prefix for ``path`` given its role ``suffix``.

    inD keys each recording group by the file-stem text preceding the role suffix
    (``00`` in ``00_tracks.csv``). The recording id may be empty for
    suffix-only names; grouping still works because every role shares the prefix.
    """

    name = path.name
    return name[: -len(suffix)]


def _resolve_background(root: Path, recording_id: str) -> Path | None:
    """Resolve the background image for ``recording_id``.

    Prefers the documented ``<id>_background.png`` name, then falls back to any
    ``<id>*.png`` staged without the original suffix, mirroring the registry's
    grouped background required-path alternatives.

    Returns:
        The resolved background image path, or ``None`` when none is staged.
    """

    preferred = sorted(
        match for match in root.rglob(f"{recording_id}{_BACKGROUND_SUFFIX}") if match.is_file()
    )
    if preferred:
        return preferred[0]
    fallback = sorted(match for match in root.rglob(f"{recording_id}*.png") if match.is_file())
    return fallback[0] if fallback else None


def _resolve_dataset_paths(root: Path) -> tuple[IndDatasetPaths | None, list[str]]:
    """Resolve every complete inD recording group under ``root``.

    A recording is complete only when its tracks, tracksMeta, recordingMeta CSVs
    and a background image are all present. Recordings missing any role are
    reported so callers can surface an actionable message.

    Returns:
        A ``(dataset, issues)`` tuple. ``dataset`` is the resolved
        :class:`IndDatasetPaths` when at least one complete recording exists, else
        ``None``; ``issues`` lists human-readable reasons no complete recording was
        found (empty when ``dataset`` is not ``None``).
    """

    tracks_by_id: dict[str, Path] = {}
    for match in sorted(root.rglob(f"*{_TRACKS_SUFFIX}")):
        if match.is_file():
            tracks_by_id.setdefault(_recording_id_for(match, _TRACKS_SUFFIX), match)

    if not tracks_by_id:
        return None, ["no *_tracks.csv recording table found"]

    recordings: list[IndRecordingPaths] = []
    issues: list[str] = []
    for recording_id, tracks in sorted(tracks_by_id.items()):
        tracks_meta = _first_sibling(root, recording_id, _TRACKS_META_SUFFIX)
        recording_meta = _first_sibling(root, recording_id, _RECORDING_META_SUFFIX)
        background = _resolve_background(root, recording_id)
        missing = [
            role
            for role, resolved in (
                ("tracksMeta", tracks_meta),
                ("recordingMeta", recording_meta),
                ("background", background),
            )
            if resolved is None
        ]
        if missing:
            issues.append(f"recording '{recording_id or '<root>'}' missing {', '.join(missing)}")
            continue
        # The ``missing`` guard above guarantees these three siblings resolved
        # to real paths; narrow ``Path | None`` -> ``Path`` for the constructor.
        assert tracks_meta is not None
        assert recording_meta is not None
        assert background is not None
        recordings.append(
            IndRecordingPaths(
                recording_id=recording_id,
                tracks=tracks,
                tracks_meta=tracks_meta,
                recording_meta=recording_meta,
                background=background,
            )
        )

    if not recordings:
        return None, issues
    return IndDatasetPaths(root=root, recordings=tuple(recordings)), []


def _first_sibling(root: Path, recording_id: str, suffix: str) -> Path | None:
    """Return the first staged file matching ``<recording_id><suffix>``."""

    matches = sorted(match for match in root.rglob(f"{recording_id}{suffix}") if match.is_file())
    return matches[0] if matches else None


def is_available(root: Path | str | None = None) -> bool:
    """Return whether at least one complete inD recording group is staged.

    Returns ``False`` for an absent or incomplete layout so skip-if-absent tests
    can skip cleanly. A present-but-malformed CSV does not fail this check; it is
    reported by :func:`load_shape_contract`.
    """

    resolved_root = dataset_root(root)
    if not resolved_root.is_dir():
        return False
    dataset, _issues = _resolve_dataset_paths(resolved_root)
    return dataset is not None


def require_available(root: Path | str | None = None) -> IndDatasetPaths:
    """Return resolved recording paths or raise an actionable acquisition error."""

    resolved_root = dataset_root(root)
    if not resolved_root.is_dir():
        raise IndDataError(
            f"{IND_ASSET_ID} is not staged: {resolved_root} does not exist. "
            f"Follow the acquisition and staging steps in {ACQUISITION_DOC}. "
            f"You may set {EXTERNAL_DATA_ROOT_ENV} to a shared external-data root."
        )
    dataset, issues = _resolve_dataset_paths(resolved_root)
    if dataset is None:
        detail = "; ".join(issues) if issues else "no complete inD recording group found"
        raise IndDataError(
            f"{IND_ASSET_ID} layout at {resolved_root} is incomplete: {detail}. "
            f"Each recording needs *_tracks.csv, *_tracksMeta.csv, *_recordingMeta.csv, "
            f"and a background image. Expected the documented layout in {ACQUISITION_DOC}."
        )
    return dataset


def _read_csv_rows(path: Path, role: str) -> tuple[list[str], list[list[str]]]:
    """Parse a headered inD CSV into ``(header, data_rows)`` string tables.

    Blank trailing lines are ignored. Every retained data row must share the
    header's column count.

    Returns:
        A ``(header, data_rows)`` tuple of the parsed header columns and the
        remaining string data rows.

    Raises:
        IndDataError: If the file cannot be read, has no header, has no data
            rows, or is non-rectangular.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IndDataError(
            f"inD {role} file {path} could not be read as text. Re-stage per {ACQUISITION_DOC}."
        ) from exc

    reader = csv.reader(line for line in text.splitlines() if line.strip())
    records = [row for row in reader if row]
    if not records:
        raise IndDataError(f"inD {role} file {path} is empty. See {ACQUISITION_DOC}.")
    header = [column.strip() for column in records[0]]
    data_rows = records[1:]
    if not data_rows:
        raise IndDataError(
            f"inD {role} file {path} has a header but no data rows. See {ACQUISITION_DOC}."
        )
    column_count = len(header)
    if any(len(row) != column_count for row in data_rows):
        raise IndDataError(
            f"inD {role} file {path} is not rectangular; rows do not match the "
            f"{column_count}-column header. See {ACQUISITION_DOC}."
        )
    return header, data_rows


def _require_columns(header: list[str], required: tuple[str, ...], role: str, path: Path) -> None:
    """Raise when ``header`` is missing any documented inD column for ``role``."""

    missing = [column for column in required if column not in header]
    if missing:
        raise IndDataError(
            f"inD {role} file {path} is missing expected column(s): "
            f"{', '.join(missing)}. This does not look like an inD {role} table. "
            f"See {ACQUISITION_DOC}."
        )


def _tracks_contract(path: Path) -> dict[str, Any]:
    """Validate a ``*_tracks.csv`` table structurally and summarize its shape.

    Confirms the documented header columns are present and that the coordinate/id
    columns parse as finite floats across every data row.

    Returns:
        A mapping with ``row_count`` and ``column_count`` for the table.
    """

    header, data_rows = _read_csv_rows(path, "tracks")
    _require_columns(header, _TRACKS_REQUIRED_COLUMNS, "tracks", path)
    finite_indices = {column: header.index(column) for column in _TRACKS_FINITE_COLUMNS}
    for row in data_rows:
        for column, index in finite_indices.items():
            token = row[index].strip()
            try:
                value = float(token)
            except ValueError as exc:
                raise IndDataError(
                    f"inD tracks file {path} has a non-numeric '{column}' value "
                    f"'{token}'. See {ACQUISITION_DOC}."
                ) from exc
            if not math.isfinite(value):
                raise IndDataError(
                    f"inD tracks file {path} has a non-finite '{column}' value. "
                    f"See {ACQUISITION_DOC}."
                )
    return {
        "row_count": len(data_rows),
        "column_count": len(header),
    }


def _table_contract(path: Path, role: str, required: tuple[str, ...]) -> dict[str, Any]:
    """Validate a headered inD metadata CSV structurally and summarize its shape.

    Returns:
        A mapping with ``row_count`` and ``column_count`` for the table.
    """

    header, data_rows = _read_csv_rows(path, role)
    _require_columns(header, required, role, path)
    return {
        "row_count": len(data_rows),
        "column_count": len(header),
    }


def load_shape_contract(root: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the cheap inD per-recording shape contract.

    Args:
        root: Explicit dataset root, or ``None`` to resolve via the shared
            external-data registry.

    Returns:
        A structural contract mapping with per-recording shape metadata (row and
        column counts for each CSV role plus resolved relative paths). No
        trajectory content is asserted.

    Raises:
        IndDataError: If the dataset is absent/incomplete or any staged CSV is
            malformed or not inD-shaped.
    """

    dataset = require_available(root)
    recordings: dict[str, dict[str, Any]] = {}
    for recording in dataset.recordings:
        recordings[recording.recording_id] = {
            "tracks": {
                "path": recording.tracks.relative_to(dataset.root).as_posix(),
                **_tracks_contract(recording.tracks),
            },
            "tracks_meta": {
                "path": recording.tracks_meta.relative_to(dataset.root).as_posix(),
                **_table_contract(
                    recording.tracks_meta, "tracksMeta", _TRACKS_META_REQUIRED_COLUMNS
                ),
            },
            "recording_meta": {
                "path": recording.recording_meta.relative_to(dataset.root).as_posix(),
                **_table_contract(
                    recording.recording_meta,
                    "recordingMeta",
                    _RECORDING_META_REQUIRED_COLUMNS,
                ),
            },
            "background": {
                "path": recording.background.relative_to(dataset.root).as_posix(),
            },
        }
    return {
        "asset_id": IND_ASSET_ID,
        "root": str(dataset.root),
        "docs_path": dataset.docs_path,
        "recordings": recordings,
    }
