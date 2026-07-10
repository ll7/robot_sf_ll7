"""Real-track trajectory parsing for staged ETH/UCY pedestrian data.

Plain-language summary: the existing ``robot_sf/data/external/eth_ucy.py`` loader
only checks a *structural* shape contract (row/column counts). It does not turn
the staged bytes into usable per-pedestrian position-time arrays. Issue #4975
needs real reference trajectories to compute trajectory RMSE and distribution
comparisons against simulated tracks. This module adds that parsing layer on top
of the existing layout resolver, without changing the license-safe acquisition
contract or asserting any dataset content claim.

The parser is deliberately format-aware but content-agnostic:

- ETH BIWI ``obsmat.txt``: eight whitespace columns ``frame id x z y vx vz vy``,
  world coordinates in meters. The canonical BIWI frame rate is 2.5 Hz (0.4 s per
  frame); the parser derives time from the frame column and an explicit frame
  period rather than assuming a fixed sample count, and never asserts any
  coordinate value.
- Normalized UCY ``.txt``: four columns ``frame id x y``.

It never downloads, vendors, or redistributes dataset bytes, fails closed with
``EthUcyDataError`` when data is absent or malformed, and every error points back
to ``docs/datasets/eth-ucy.md``. UCY ``.vsp`` spline files are intentionally not
parsed here (full spline decoding is out of scope for a trajectory-level harness;
they are skipped with a recorded reason, not a failure).

This module is the real-track reference input for
``robot_sf/benchmark/pedestrian_realism_validation.py``. It is distinct from the
issue #3971 synthetic flow harness, which never compares against real tracks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.data.external.eth_ucy import (
    ACQUISITION_DOC,
    ETH_UCY_ASSET_ID,
    EthUcyDataError,
    EthUcySplitPath,
    require_available,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "ETH_BIWI_OBSMAT_FRAME_PERIOD_S",
    "EthUcyTrack",
    "EthUcyTrackSet",
    "load_split_tracks",
    "load_track_set",
]

#: Canonical ETH BIWI ``obsmat.txt`` frame period (2.5 Hz). Used only to convert
#: the integer frame column into seconds. This is a documented dataset property,
#: not a content assertion about any particular scene.
ETH_BIWI_OBSMAT_FRAME_PERIOD_S = 0.4

#: Column indices for the ETH BIWI ``obsmat.txt`` layout (frame id x z y vx vz vy).
_OBSMAT_FRAME_COL = 0
_OBSMAT_ID_COL = 1
_OBSMAT_X_COL = 2
_OBSMAT_Y_COL = 4

#: Column indices for the normalized four-column ``frame id x y`` layout.
_TXT_FRAME_COL = 0
_TXT_ID_COL = 1
_TXT_X_COL = 2
_TXT_Y_COL = 3

#: Minimum finite samples for a parsed pedestrian track to be retained. A single
#: point carries no displacement information and cannot contribute to RMSE or
#: speed/density metrics, so it is dropped with a recorded reason rather than
#: distorting downstream distributions.
_MIN_TRACK_SAMPLES = 2


@dataclass(frozen=True)
class EthUcyTrack:
    """One parsed pedestrian track from a staged ETH/UCY split.

    Attributes:
        pedestrian_id: Integer pedestrian id from the source file.
        time_s: Strictly increasing sample times in seconds, shape ``(T,)``.
        positions: World-frame ``(x, y)`` positions in meters, shape ``(T, 2)``.
    """

    pedestrian_id: int
    time_s: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class EthUcyTrackSet:
    """Parsed per-pedestrian tracks for one ETH/UCY split.

    Attributes:
        asset_id: Canonical registry id (``"eth-ucy"``).
        group: Source family, ``"eth"`` (BIWI) or ``"ucy"`` (Crowds-by-Example).
        split: Canonical split id (``"eth"``, ``"hotel"``, ``"univ"``, ...).
        format: Detected trajectory format (``"obsmat"`` or ``"txt"``).
        docs_path: Acquisition/layout doc page for error messages.
        tracks: Parsed :class:`EthUcyTrack` entries, ordered by pedestrian id.
        skipped_formats: Formats encountered but not parsed (e.g. ``"vsp"``),
            with the reason recorded so callers can report partial coverage.
        frame_period_s: Frame period in seconds used to derive ``time_s``.
    """

    asset_id: str
    group: str
    split: str
    format: str
    docs_path: str
    tracks: tuple[EthUcyTrack, ...]
    skipped_formats: tuple[str, ...]
    frame_period_s: float


def load_split_tracks(
    split_path: EthUcySplitPath,
    *,
    frame_period_s: float = ETH_BIWI_OBSMAT_FRAME_PERIOD_S,
) -> EthUcyTrackSet:
    """Parse one resolved ETH/UCY split into per-pedestrian tracks.

    Args:
        split_path: A resolved split from :func:`eth_ucy.require_available`.
        frame_period_s: Frame period in seconds used to convert the integer frame
            column to times. Defaults to the canonical ETH BIWI value.

    Returns:
        A :class:`EthUcyTrackSet` with parsed tracks (``obsmat``/``txt``) or an
        empty track set with ``skipped_formats`` recording an unsupported format.

    Raises:
        EthUcyDataError: If a parseable file is malformed (non-numeric,
            non-finite, or narrower than the required columns).
    """

    _validate_frame_period(frame_period_s)
    if split_path.format == "obsmat":
        tracks = _parse_obsmat(split_path, frame_period_s=frame_period_s)
        return EthUcyTrackSet(
            asset_id=ETH_UCY_ASSET_ID,
            group=split_path.group,
            split=split_path.split,
            format=split_path.format,
            docs_path=ACQUISITION_DOC,
            tracks=tuple(tracks),
            skipped_formats=(),
            frame_period_s=frame_period_s,
        )
    if split_path.format == "txt":
        tracks = _parse_txt(split_path, frame_period_s=frame_period_s)
        return EthUcyTrackSet(
            asset_id=ETH_UCY_ASSET_ID,
            group=split_path.group,
            split=split_path.split,
            format=split_path.format,
            docs_path=ACQUISITION_DOC,
            tracks=tuple(tracks),
            skipped_formats=(),
            frame_period_s=frame_period_s,
        )
    return EthUcyTrackSet(
        asset_id=ETH_UCY_ASSET_ID,
        group=split_path.group,
        split=split_path.split,
        format=split_path.format,
        docs_path=ACQUISITION_DOC,
        tracks=(),
        skipped_formats=(split_path.format,),
        frame_period_s=frame_period_s,
    )


def _validate_frame_period(frame_period_s: float) -> None:
    """Reject a frame period that cannot produce a valid time axis."""

    if (
        isinstance(frame_period_s, bool)
        or not math.isfinite(frame_period_s)
        or frame_period_s <= 0.0
    ):
        raise ValueError("frame_period_s must be finite and positive")


def load_track_set(
    split: str,
    *,
    root: Path | str | None = None,
    frame_period_s: float = ETH_BIWI_OBSMAT_FRAME_PERIOD_S,
) -> EthUcyTrackSet:
    """Resolve and parse one named ETH/UCY split from the staged dataset.

    Args:
        split: Canonical split id (``"eth"``, ``"hotel"``, ``"univ"``,
            ``"zara01"``, ``"zara02"``).
        root: Explicit dataset root, or ``None`` to resolve via the shared
            external-data registry.
        frame_period_s: Frame period in seconds used to derive sample times.

    Returns:
        The parsed :class:`EthUcyTrackSet` for ``split``.

    Raises:
        EthUcyDataError: If the dataset is absent/incomplete or the split file is
            malformed.
        KeyError: If ``split`` is not a documented ETH/UCY split id.
    """

    dataset = require_available(root)
    for split_path in dataset.splits:
        if split_path.split == split:
            return load_split_tracks(split_path, frame_period_s=frame_period_s)
    raise KeyError(
        f"Unknown ETH/UCY split '{split}'. Documented splits: "
        f"{', '.join(sorted({p.split for p in dataset.splits}))}. See {ACQUISITION_DOC}."
    )


def _parse_obsmat(
    split_path: EthUcySplitPath,
    *,
    frame_period_s: float,
) -> list[EthUcyTrack]:
    """Parse an ETH BIWI ``obsmat.txt`` (8-col) file into per-pedestrian tracks.

    The BIWI layout is ``frame id x z y vx vz vy``. We extract ``(frame, id, x,
    y)`` and derive time in seconds from the frame column.

    Returns:
        Parsed tracks ordered by pedestrian id, each with at least two samples.

    Raises:
        EthUcyDataError: If the file is narrower than the required y column or
            contains a non-numeric/non-finite value in a used column.
    """

    rows = _read_numeric_rows(split_path, required_columns=_OBSMAT_Y_COL + 1)
    frames = rows[:, _OBSMAT_FRAME_COL]
    ids = rows[:, _OBSMAT_ID_COL]
    positions = np.stack(
        (rows[:, _OBSMAT_X_COL], rows[:, _OBSMAT_Y_COL]),
        axis=1,
    )
    time_s = (frames - frames.min()) * frame_period_s
    return _group_by_pedestrian(ids, time_s, positions, split_path)


def _parse_txt(
    split_path: EthUcySplitPath,
    *,
    frame_period_s: float,
) -> list[EthUcyTrack]:
    """Parse a normalized 4-col ``frame id x y`` trajectory file into tracks.

    Returns:
        Parsed tracks ordered by pedestrian id, each with at least two samples.

    Raises:
        EthUcyDataError: If the file is narrower than the required y column or
            contains a non-numeric/non-finite value.
    """

    rows = _read_numeric_rows(split_path, required_columns=_TXT_Y_COL + 1)
    frames = rows[:, _TXT_FRAME_COL]
    ids = rows[:, _TXT_ID_COL]
    positions = np.stack(
        (rows[:, _TXT_X_COL], rows[:, _TXT_Y_COL]),
        axis=1,
    )
    time_s = (frames - frames.min()) * frame_period_s
    return _group_by_pedestrian(ids, time_s, positions, split_path)


def _read_numeric_rows(
    split_path: EthUcySplitPath,
    *,
    required_columns: int,
) -> np.ndarray:
    """Read a trajectory file into a finite float matrix.

    Args:
        split_path: Resolved split file to read.
        required_columns: Minimum column count the format needs.

    Returns:
        A ``(N, C)`` float matrix.

    Raises:
        EthUcyDataError: If the file cannot be read, is empty, contains a
            non-numeric or non-finite value, or is narrower than
            ``required_columns``.
    """

    path = split_path.path
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} could not be read as "
            f"text. Re-stage per {ACQUISITION_DOC}."
        ) from exc

    delimiter = "comma" if "," in text else "whitespace"
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split(",") if delimiter == "comma" else line.split()
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise EthUcyDataError(
                f"ETH/UCY split '{split_path.split}' file {path} contains a "
                f"non-numeric value in row '{line}'. See {ACQUISITION_DOC}."
            ) from exc
        if any(not np.isfinite(value) for value in values):
            raise EthUcyDataError(
                f"ETH/UCY split '{split_path.split}' file {path} contains a "
                f"non-finite value in row '{line}'. See {ACQUISITION_DOC}."
            )
        rows.append(values)

    if not rows:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} has no numeric data "
            f"rows. See {ACQUISITION_DOC}."
        )
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} is not rectangular; "
            f"rows have varying column counts. See {ACQUISITION_DOC}."
        )
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape[1] < required_columns:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} has {matrix.shape[1]} "
            f"columns; the '{split_path.format}' format needs at least "
            f"{required_columns}. See {ACQUISITION_DOC}."
        )
    return matrix


def _group_by_pedestrian(
    ids: np.ndarray,
    time_s: np.ndarray,
    positions: np.ndarray,
    split_path: EthUcySplitPath,
) -> list[EthUcyTrack]:
    """Group flat rows into per-pedestrian tracks ordered by pedestrian id.

    Pedestrian ids are integer-cast (floor) to match the BIWI/UCY convention.
    Within each pedestrian, samples are sorted by time and de-duplicated so RMSE
    matching is stable. Tracks with fewer than two samples are dropped.

    Returns:
        Parsed :class:`EthUcyTrack` entries ordered by pedestrian id.
    """

    ped_ids = np.floor(ids).astype(np.int64)
    tracks: list[EthUcyTrack] = []
    for ped_id in sorted(set(ped_ids.tolist())):
        mask = ped_ids == ped_id
        order = np.argsort(time_s[mask], kind="stable")
        times = np.asarray(time_s[mask], dtype=float)[order]
        pts = np.asarray(positions[mask], dtype=float)[order]
        # De-duplicate repeated frame timestamps (stable first-wins).
        keep = np.ones(times.shape, dtype=bool)
        if times.shape[0] > 1:
            keep[1:] = times[1:] != times[:-1]
        times = times[keep]
        pts = pts[keep]
        if times.shape[0] < _MIN_TRACK_SAMPLES:
            continue
        tracks.append(EthUcyTrack(pedestrian_id=int(ped_id), time_s=times, positions=pts))
    return tracks


def track_set_summary(track_set: EthUcyTrackSet) -> dict[str, Any]:
    """Return a JSON-safe structural summary for a parsed track set.

    The summary reports counts and timing extents only; it asserts no dataset
    content values (no coordinates, speeds, or scene identities).

    Returns:
        A JSON-safe mapping with pedestrian/track counts and time extents.
    """

    track_lengths = np.asarray([track.positions.shape[0] for track in track_set.tracks])
    all_times: Iterable[float] = (
        float(track.time_s[-1]) for track in track_set.tracks if track.time_s.size
    )
    extents = list(all_times)
    return {
        "asset_id": track_set.asset_id,
        "split": track_set.split,
        "group": track_set.group,
        "format": track_set.format,
        "docs_path": track_set.docs_path,
        "pedestrian_count": len(track_set.tracks),
        "total_samples": int(track_lengths.sum()) if track_lengths.size else 0,
        "min_track_samples": int(track_lengths.min()) if track_lengths.size else 0,
        "max_track_samples": int(track_lengths.max()) if track_lengths.size else 0,
        "time_extent_s": {
            "min": float(min(extents)) if extents else 0.0,
            "max": float(max(extents)) if extents else 0.0,
        },
        "skipped_formats": list(track_set.skipped_formats),
        "frame_period_s": float(track_set.frame_period_s),
    }
