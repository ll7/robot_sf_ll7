"""ATC pedestrian tracking external-data shape contract.

Plain-language summary: this loader only inspects locally staged ATC (Osaka
shopping-center) pedestrian-trajectory CSV files. It never downloads, vendors, or
redistributes the license-gated ATC dataset bytes. The contract is cheap and
structural: it confirms the documented daily-CSV layout is present (via the
canonical external-data registry) and that each staged trajectory CSV parses as
finite numeric rows with the documented ATC column width. It intentionally avoids
any content assertion (exact frame counts, coordinates, pedestrian ids, or
scene-specific values) and makes no prediction-comparability or benchmark claim.

The expected layout, sources, chosen-subset policy, and staging workflow are
documented in ``docs/datasets/atc.md``; keep the column contract below aligned
with that page and the ``atc-pedestrian`` registry entry rather than duplicating
layout rules elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.errors import RobotSfError
from scripts.tools.manage_external_data import (
    EXTERNAL_DATA_ROOT_ENV,
    check_asset,
    resolve_asset_local_path_by_id,
)

ASSET_ID = "atc-pedestrian"
ACQUISITION_DOC = "docs/datasets/atc.md"

# Documented ATC daily-CSV column contract (docs/datasets/atc.md). The upstream
# daily files are headerless, comma-delimited, and carry exactly these eight
# numeric columns in this order. This is a structural width contract, not a
# content claim about specific trajectories.
ATC_COLUMN_NAMES: tuple[str, ...] = (
    "time",
    "person_id",
    "x",
    "y",
    "z",
    "velocity",
    "motion_angle",
    "facing_angle",
)
_ATC_COLUMNS = len(ATC_COLUMN_NAMES)

# A single ATC day can hold millions of samples. The shape contract only needs a
# structural sample, so scanning is bounded by default; callers wanting a full
# scan pass ``max_rows=None``. Truncation is reported so a bounded scan is never
# mistaken for a whole-file validation.
_DEFAULT_MAX_SCAN_ROWS = 10_000


class AtcDataError(RobotSfError, RuntimeError):
    """Raised when staged ATC data is present but structurally invalid."""


@dataclass(frozen=True)
class AtcDatasetPaths:
    """Resolved trajectory CSV files for a staged ATC dataset."""

    root: Path
    csv_files: tuple[Path, ...]
    docs_path: str = ACQUISITION_DOC


def dataset_root(root: Path | str | None = None) -> Path:
    """Return the ATC dataset root, honoring the shared external-data root.

    Args:
        root: Explicit dataset root. When ``None`` the shared external-data
            registry resolves the path from ``ROBOT_SF_EXTERNAL_DATA_ROOT`` (or
            the repo-local default).

    Returns:
        The absolute dataset root path. The path is not required to exist.
    """

    if root is not None:
        return Path(root).expanduser().resolve()
    return resolve_asset_local_path_by_id(ASSET_ID).expanduser().resolve()


def _resolve_csv_files(root: Path) -> list[Path]:
    """Resolve staged ATC trajectory CSVs under ``root``.

    Prefers the canonical ``atc-*.csv`` daily filenames and falls back to any
    ``*.csv`` staged without the original name, mirroring the registry
    ``trajectory`` required-path group. Nested subdirectories are allowed.

    Returns:
        Sorted staged trajectory CSV paths, or an empty list when none resolve.
    """

    preferred = sorted(path for path in root.glob("**/atc-*.csv") if path.is_file())
    if preferred:
        return preferred
    return sorted(path for path in root.glob("**/*.csv") if path.is_file())


def availability_report(root: Path | str | None = None) -> dict[str, Any]:
    """Return the canonical external-data registry availability report.

    Reuses ``manage_external_data.check_asset`` so the loader and the registry
    agree on what a staged ATC layout must contain (at least one trajectory CSV
    plus a local README/LICENSE/TERMS note).
    """

    return check_asset(ASSET_ID, source_path=dataset_root(root))


def is_available(root: Path | str | None = None) -> bool:
    """Return whether the documented ATC layout is staged.

    Returns ``False`` for an absent or incomplete layout so skip-if-absent tests
    can skip cleanly. A present-but-malformed CSV does not fail this check; it is
    reported by :func:`load_shape_contract`.
    """

    return bool(availability_report(root).get("ok", False))


def require_available(root: Path | str | None = None) -> AtcDatasetPaths:
    """Return resolved trajectory files or raise an actionable acquisition error."""

    report = availability_report(root)
    if not report.get("ok", False):
        missing = report.get("missing_required_paths") or []
        missing_text = ", ".join(str(path) for path in missing) or "unknown"
        action = str(
            report.get("action")
            or "Stage the official ATC daily CSV(s) plus a local terms/README note."
        )
        raise AtcDataError(
            f"{ASSET_ID} is not staged or is incomplete at {report['source_path']}. "
            f"Missing required path group(s): {missing_text}. {action} "
            f"See {ACQUISITION_DOC}. You may set {EXTERNAL_DATA_ROOT_ENV} to a shared "
            "external-data root."
        )

    resolved_root = dataset_root(root)
    csv_files = _resolve_csv_files(resolved_root)
    if not csv_files:
        # Defensive: the registry check reported ok, so a trajectory CSV should
        # resolve. Guard the invariant rather than silently returning empty.
        raise AtcDataError(
            f"{ASSET_ID} reported available at {resolved_root} but no trajectory CSV "
            f"resolved. Re-stage per {ACQUISITION_DOC}."
        )
    return AtcDatasetPaths(
        root=resolved_root,
        csv_files=tuple(csv_files),
        docs_path=ACQUISITION_DOC,
    )


def _scan_atc_csv(path: Path, *, max_rows: int | None) -> tuple[int, bool]:
    """Structurally validate one ATC daily CSV and summarize its scanned rows.

    Comment lines (``#`` prefix) and blank lines are ignored. Every retained data
    row must be comma-delimited, exactly :data:`_ATC_COLUMNS` wide, and parse to
    finite floats. Scanning stops after ``max_rows`` data rows when a bound is set.

    Returns:
        A ``(scanned_rows, truncated)`` tuple. ``truncated`` is ``True`` when a
        row bound stopped the scan before end-of-file.

    Raises:
        AtcDataError: If the file cannot be read, has no numeric rows, or any
            scanned row is malformed (wrong width, non-numeric, or non-finite).
    """

    scanned = 0
    truncated = False
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if max_rows is not None and scanned >= max_rows:
                    truncated = True
                    break
                tokens = line.split(",")
                if len(tokens) != _ATC_COLUMNS:
                    raise AtcDataError(
                        f"ATC CSV {path} row '{line}' has {len(tokens)} columns; the "
                        f"documented ATC layout has exactly {_ATC_COLUMNS} "
                        f"({', '.join(ATC_COLUMN_NAMES)}). See {ACQUISITION_DOC}."
                    )
                try:
                    values = [float(token) for token in tokens]
                except ValueError as exc:
                    raise AtcDataError(
                        f"ATC CSV {path} contains a non-numeric value in row '{line}'. "
                        f"See {ACQUISITION_DOC}."
                    ) from exc
                if any(not math.isfinite(value) for value in values):
                    raise AtcDataError(
                        f"ATC CSV {path} contains a non-finite value in row '{line}'. "
                        f"See {ACQUISITION_DOC}."
                    )
                scanned += 1
    except (OSError, UnicodeDecodeError) as exc:
        raise AtcDataError(
            f"ATC CSV {path} could not be read as text. Re-stage per {ACQUISITION_DOC}."
        ) from exc

    if scanned == 0:
        raise AtcDataError(f"ATC CSV {path} has no numeric data rows. See {ACQUISITION_DOC}.")
    return scanned, truncated


def load_shape_contract(
    root: Path | str | None = None,
    *,
    max_rows: int | None = _DEFAULT_MAX_SCAN_ROWS,
) -> dict[str, Any]:
    """Load and validate the cheap ATC daily-CSV shape contract.

    Args:
        root: Explicit dataset root, or ``None`` to resolve via the shared
            external-data registry.
        max_rows: Maximum data rows scanned per CSV. ``None`` scans the whole
            file; the bounded default keeps the contract cheap on multi-million
            row daily files while still validating structure.

    Returns:
        A structural contract mapping with the documented column contract and a
        per-file summary (relative path, scanned row count, and whether the scan
        was truncated by ``max_rows``). No trajectory content is asserted.

    Raises:
        AtcDataError: If the dataset is absent/incomplete or any staged CSV is
            malformed.
    """

    dataset = require_available(root)
    files: list[dict[str, Any]] = []
    for csv_path in dataset.csv_files:
        scanned, truncated = _scan_atc_csv(csv_path, max_rows=max_rows)
        files.append(
            {
                "path": csv_path.relative_to(dataset.root).as_posix(),
                "scanned_rows": scanned,
                "scan_truncated": truncated,
            }
        )
    return {
        "asset_id": ASSET_ID,
        "root": str(dataset.root),
        "docs_path": dataset.docs_path,
        "delimiter": "comma",
        "column_count": _ATC_COLUMNS,
        "column_names": list(ATC_COLUMN_NAMES),
        "file_count": len(files),
        "max_rows": max_rows,
        "files": files,
    }
