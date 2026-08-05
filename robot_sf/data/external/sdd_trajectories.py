"""Fail-closed parsing of synthetic or locally supplied SDD annotations.

The parser consumes the documented ten-field ``annotations.txt`` format.  It
does not download, stage, or make provenance claims about Socially Compliant
SDD data; callers provide the annotation path and every coordinate conversion
parameter explicitly.
"""

from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "SDD_TRAJECTORY_SCHEMA_VERSION",
    "SddTrack",
    "SddTrackSet",
    "SddTrajectoryDataError",
    "SddTrajectoryTrack",
    "SddTrajectoryTrackSet",
    "load_sdd_track_set",
    "parse_sdd_annotations",
]

SDD_TRAJECTORY_SCHEMA_VERSION = "sdd_trajectory.v1"
_DEFAULT_ACCEPTED_LABELS = ("Pedestrian",)
_ANNOTATION_FIELD_COUNT = 10


class SddTrajectoryDataError(ValueError):
    """Raised when an SDD annotation input cannot satisfy the trajectory contract."""


@dataclass(frozen=True, slots=True)
class _SddRow:
    """One validated, retained annotation row before coordinate conversion."""

    track_id: int
    frame: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    lost: int
    occluded: int
    generated: int
    label: str
    label_key: str

    @property
    def center_px(self) -> tuple[float, float]:
        """Return the bounding-box center in source pixels."""

        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)


@dataclass(frozen=True, slots=True)
class SddTrajectoryTrack:
    """One immutable pedestrian trajectory from an SDD scene/split."""

    track_id: int
    time_s: np.ndarray
    positions: np.ndarray
    occluded_count: int
    generated_count: int
    label: str

    schema_version: ClassVar[str] = SDD_TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Copy and freeze arrays so source or caller arrays cannot mutate a record."""

        if isinstance(self.track_id, bool) or int(self.track_id) != self.track_id:
            raise ValueError("track_id must be a positive integer")
        if self.track_id <= 0:
            raise ValueError("track_id must be a positive integer")
        if any(
            isinstance(value, bool) or int(value) != value or value < 0
            for value in (self.occluded_count, self.generated_count)
        ):
            raise ValueError("occluded_count and generated_count must be non-negative integers")
        if not self.label.strip():
            raise ValueError("label must be non-empty")

        times = np.array(self.time_s, dtype=float, copy=True).reshape(-1)
        positions = np.array(self.positions, dtype=float, copy=True)
        if times.shape[0] < 2 or positions.shape != (times.shape[0], 2):
            raise ValueError("track requires matching time_s and positions with at least 2 rows")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
            raise ValueError("track arrays must contain only finite values")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("track time_s must be strictly increasing")
        times.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "occluded_count", int(self.occluded_count))
        object.__setattr__(self, "generated_count", int(self.generated_count))
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "positions", positions)

    @property
    def pedestrian_id(self) -> int:
        """Expose the existing realism-harness track identity name."""

        return self.track_id

    @property
    def metadata(self) -> Mapping[str, int | str]:
        """Return immutable source-quality metadata for this track."""

        return MappingProxyType(
            {
                "source_track_id": self.track_id,
                "source_label": self.label,
                "occluded_count": self.occluded_count,
                "generated_count": self.generated_count,
            }
        )


@dataclass(frozen=True, slots=True)
class SddTrajectoryTrackSet:
    """Immutable parsed SDD tracks for exactly one scene and split."""

    scene: str
    split: str
    tracks: tuple[SddTrajectoryTrack, ...]
    frame_rate_hz: float
    meters_per_pixel: float
    image_height_px: float | None
    accepted_labels: tuple[str, ...]
    annotation_path: Path
    asset_id: str = "sdd"
    format: str = "annotations.txt"
    docs_path: str = "scripts/tools/import_sdd_scenarios.py"

    schema_version: ClassVar[str] = SDD_TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate immutable set identity and conversion metadata."""

        for name, value in (("scene", self.scene), ("split", self.split)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        _validate_positive_finite(self.frame_rate_hz, "frame_rate_hz")
        _validate_positive_finite(self.meters_per_pixel, "meters_per_pixel")
        if self.image_height_px is not None:
            _validate_positive_finite(self.image_height_px, "image_height_px")
        if tuple(track.track_id for track in self.tracks) != tuple(
            sorted(track.track_id for track in self.tracks)
        ):
            raise ValueError("tracks must be sorted by normalized track_id")
        if len({track.track_id for track in self.tracks}) != len(self.tracks):
            raise ValueError("track ids must be unique")
        labels = tuple(label.strip() for label in self.accepted_labels)
        if not labels or any(not label for label in labels):
            raise ValueError("accepted_labels must contain at least one non-empty label")
        object.__setattr__(self, "scene", self.scene.strip())
        object.__setattr__(self, "split", self.split.strip())
        object.__setattr__(self, "tracks", tuple(self.tracks))
        object.__setattr__(self, "frame_rate_hz", float(self.frame_rate_hz))
        object.__setattr__(self, "meters_per_pixel", float(self.meters_per_pixel))
        object.__setattr__(
            self,
            "image_height_px",
            None if self.image_height_px is None else float(self.image_height_px),
        )
        object.__setattr__(self, "accepted_labels", labels)
        object.__setattr__(self, "annotation_path", Path(self.annotation_path))

    @property
    def scene_split(self) -> str:
        """Return the explicit scene/split identity used by callers."""

        return f"{self.scene}/{self.split}"


SddTrack = SddTrajectoryTrack
SddTrackSet = SddTrajectoryTrackSet


def load_sdd_track_set(  # noqa: C901 - the parser validates each contract field explicitly
    annotation_path: Path | str,
    *,
    scene: str,
    split: str,
    frame_rate_hz: float,
    meters_per_pixel: float,
    image_height_px: float | None = None,
    accepted_labels: tuple[str, ...] | list[str] | set[str] | str | None = None,
) -> SddTrajectoryTrackSet:
    """Parse one explicit SDD ``annotations.txt`` file into immutable tracks.

    Rows with nonzero ``lost`` or labels outside ``accepted_labels`` are not
    reference tracks.  The default accepted label is ``Pedestrian``.

    Returns:
        An immutable track set ordered by normalized track id and frame.
    """

    path = Path(annotation_path).expanduser()
    _validate_identity(scene, "scene")
    _validate_identity(split, "split")
    _validate_positive_finite(frame_rate_hz, "frame_rate_hz")
    _validate_positive_finite(meters_per_pixel, "meters_per_pixel")
    if image_height_px is not None:
        _validate_positive_finite(image_height_px, "image_height_px")
    labels = _normalize_accepted_labels(accepted_labels)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SddTrajectoryDataError(f"SDD annotation file is unavailable: {path}") from exc

    grouped: dict[int, list[_SddRow]] = {}
    seen: dict[tuple[int, int], _SddRow] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        row = _parse_row(raw_line, line_number=line_number)
        if row.lost != 0 or row.label_key not in labels:
            continue
        key = (row.track_id, row.frame)
        previous = seen.get(key)
        if previous is not None:
            if row != previous:
                raise SddTrajectoryDataError(
                    f"SDD annotation line {line_number} conflicts with duplicate track/frame "
                    f"identity ({row.track_id}, {row.frame})"
                )
            continue
        seen[key] = row
        grouped.setdefault(row.track_id, []).append(row)

    if not grouped:
        raise SddTrajectoryDataError(
            f"SDD annotation file {path} has no usable accepted pedestrian rows"
        )

    tracks: list[SddTrajectoryTrack] = []
    for track_id in sorted(grouped):
        rows = sorted(grouped[track_id], key=lambda row: row.frame)
        if len(rows) < 2:
            raise SddTrajectoryDataError(
                f"SDD track {track_id} has fewer than 2 unique usable frames"
            )
        try:
            frame_values = np.asarray([row.frame for row in rows], dtype=float)
        except OverflowError as exc:
            raise SddTrajectoryDataError(
                f"SDD track {track_id} has frame identities outside finite time range"
            ) from exc
        times = frame_values / float(frame_rate_hz)
        if not np.all(np.isfinite(times)):
            raise SddTrajectoryDataError(f"SDD track {track_id} produced non-finite times")
        centers = np.asarray([row.center_px for row in rows], dtype=float)
        if image_height_px is not None:
            centers[:, 1] = float(image_height_px) - centers[:, 1]
        positions = centers * float(meters_per_pixel)
        if not np.all(np.isfinite(positions)):
            raise SddTrajectoryDataError(f"SDD track {track_id} produced non-finite positions")
        tracks.append(
            SddTrajectoryTrack(
                track_id=track_id,
                time_s=times,
                positions=positions,
                occluded_count=sum(row.occluded != 0 for row in rows),
                generated_count=sum(row.generated != 0 for row in rows),
                label=rows[0].label,
            )
        )

    return SddTrajectoryTrackSet(
        scene=scene,
        split=split,
        tracks=tuple(tracks),
        frame_rate_hz=frame_rate_hz,
        meters_per_pixel=meters_per_pixel,
        image_height_px=image_height_px,
        accepted_labels=tuple(label for label in labels.values()),
        annotation_path=path,
    )


def parse_sdd_annotations(
    annotation_path: Path | str,
    **kwargs: object,
) -> SddTrajectoryTrackSet:
    """Compatibility-named entry point for :func:`load_sdd_track_set`.

    Returns:
        The parsed immutable SDD track set.
    """

    return load_sdd_track_set(annotation_path, **kwargs)  # type: ignore[arg-type]


def _parse_row(raw_line: str, *, line_number: int) -> _SddRow:
    """Parse and validate one ten-field annotation row.

    Returns:
        A validated source row, before filtering and coordinate conversion.
    """

    try:
        fields = shlex.split(raw_line, comments=False, posix=True)
    except ValueError as exc:
        raise SddTrajectoryDataError(
            f"SDD annotation line {line_number} has invalid quoting"
        ) from exc
    if len(fields) != _ANNOTATION_FIELD_COUNT:
        raise SddTrajectoryDataError(
            f"SDD annotation line {line_number} must have 10 fields, got {len(fields)}"
        )
    track_id = _parse_integer(fields[0], "track_id", line_number, minimum=1)
    frame = _parse_integer(fields[5], "frame", line_number, minimum=0)
    xmin = _parse_float(fields[1], "xmin", line_number)
    ymin = _parse_float(fields[2], "ymin", line_number)
    xmax = _parse_float(fields[3], "xmax", line_number)
    ymax = _parse_float(fields[4], "ymax", line_number)
    if xmax < xmin or ymax < ymin:
        raise SddTrajectoryDataError(
            f"SDD annotation line {line_number} has an invalid bounding box"
        )
    lost = _parse_integer(fields[6], "lost", line_number, minimum=0)
    occluded = _parse_integer(fields[7], "occluded", line_number, minimum=0)
    generated = _parse_integer(fields[8], "generated", line_number, minimum=0)
    label = fields[9].strip()
    if not label:
        raise SddTrajectoryDataError(f"SDD annotation line {line_number} has an empty label")
    return _SddRow(
        track_id=track_id,
        frame=frame,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        lost=lost,
        occluded=occluded,
        generated=generated,
        label=label,
        label_key=label.casefold(),
    )


def _parse_float(value: str, field: str, line_number: int) -> float:
    """Parse one finite floating-point field.

    Returns:
        The finite parsed value.
    """

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise SddTrajectoryDataError(
            f"SDD annotation line {line_number} has a non-numeric {field}"
        ) from exc
    if not math.isfinite(parsed):
        raise SddTrajectoryDataError(f"SDD annotation line {line_number} has a non-finite {field}")
    return parsed


def _parse_integer(value: str, field: str, line_number: int, *, minimum: int) -> int:
    """Parse an integer identity/flag without accepting fractional values.

    Returns:
        The parsed integer.
    """

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SddTrajectoryDataError(
            f"SDD annotation line {line_number} has an invalid {field} identity"
        ) from exc
    if parsed < minimum:
        raise SddTrajectoryDataError(
            f"SDD annotation line {line_number} has an invalid {field} identity"
        )
    return parsed


def _normalize_accepted_labels(
    accepted_labels: tuple[str, ...] | list[str] | set[str] | str | None,
) -> dict[str, str]:
    """Normalize accepted labels while retaining deterministic display spelling.

    Returns:
        A case-folded lookup mapping to deterministic display labels.
    """

    source = _DEFAULT_ACCEPTED_LABELS if accepted_labels is None else accepted_labels
    if isinstance(source, str):
        values = (source,)
    elif isinstance(source, set):
        values = tuple(sorted(source))
    else:
        values = tuple(source)
    normalized: dict[str, str] = {}
    for label in values:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("accepted_labels must contain non-empty strings")
        display = label.strip()
        normalized.setdefault(display.casefold(), display)
    if not normalized:
        raise ValueError("accepted_labels must contain at least one label")
    return normalized


def _validate_identity(value: object, name: str) -> None:
    """Reject implicit or empty scene/split identities."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _validate_positive_finite(value: object, name: str) -> None:
    """Reject booleans, non-numeric values, and non-positive parameters."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
