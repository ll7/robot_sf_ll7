"""ETH/UCY external-data shape contract.

Plain-language summary: this loader only inspects locally staged ETH/UCY
pedestrian-trajectory files. It never downloads, vendors, or redistributes the
license-gated ETH BIWI or UCY Crowds-by-Example dataset bytes. The contract is
cheap and structural: it confirms the documented per-split layout exists and
that each trajectory file is parseable as finite numeric rows. It intentionally
avoids any content assertion (exact frame counts, coordinates, pedestrian ids,
or scene-specific values) and makes no prediction-comparability claim.

The expected layout, sources, and staging workflow are documented in
``docs/datasets/eth-ucy.md``; keep the split descriptors below aligned with that
page rather than duplicating layout rules elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.tools.manage_external_data import (
    EXTERNAL_DATA_ROOT_ENV,
    resolve_asset_local_path_by_id,
)

ETH_UCY_ASSET_ID = "eth-ucy"
ACQUISITION_DOC = "docs/datasets/eth-ucy.md"

# Minimum structural column floor per trajectory format. The loader consumes at
# least (frame, pedestrian id, x, y), so four finite numeric columns is the
# smallest defensible width. This is a structural floor, not a content claim: the
# canonical BIWI ``obsmat.txt`` carries eight columns and the Social-GAN
# normalized ``.txt`` layout carries four.
_MIN_NUMERIC_COLUMNS = 4


class EthUcyDataError(RuntimeError):
    """Raised when staged ETH/UCY data is present but structurally invalid."""


@dataclass(frozen=True)
class EthUcySplitSpec:
    """Expected layout descriptor for one ETH/UCY split.

    Attributes:
        group: Source family, ``"eth"`` (BIWI) or ``"ucy"`` (Crowds-by-Example).
        split: Canonical split id documented in ``docs/datasets/eth-ucy.md``.
        directory: Split subdirectory relative to the staged dataset root.
        file_candidates: Ordered ``(glob, format)`` pairs; the first glob that
            matches a regular file under ``directory`` selects the split file and
            its declared trajectory format.
    """

    group: str
    split: str
    directory: str
    file_candidates: tuple[tuple[str, str], ...]


# Single source of truth for the expected ETH/UCY splits. The five canonical
# scenes map onto the directory names documented in docs/datasets/eth-ucy.md:
#   ETH BIWI  -> eth (university entrance), hotel
#   UCY       -> univ (students), zara01, zara02
# ETH scenes ship as whitespace ``obsmat.txt`` matrices; UCY scenes ship either
# as native ``.vsp`` spline files or normalized trajectory ``.txt`` files.
_SPLIT_SPECS: tuple[EthUcySplitSpec, ...] = (
    EthUcySplitSpec("eth", "eth", "eth", (("obsmat.txt", "obsmat"),)),
    EthUcySplitSpec("eth", "hotel", "hotel", (("obsmat.txt", "obsmat"),)),
    EthUcySplitSpec("ucy", "univ", "univ", (("*.vsp", "vsp"), ("*.txt", "txt"))),
    EthUcySplitSpec("ucy", "zara01", "zara01", (("*.vsp", "vsp"), ("*.txt", "txt"))),
    EthUcySplitSpec("ucy", "zara02", "zara02", (("*.vsp", "vsp"), ("*.txt", "txt"))),
)


@dataclass(frozen=True)
class EthUcySplitPath:
    """Resolved trajectory file for one staged ETH/UCY split."""

    group: str
    split: str
    path: Path
    format: str


@dataclass(frozen=True)
class EthUcyDatasetPaths:
    """Resolved per-split trajectory files for a staged ETH/UCY dataset."""

    root: Path
    splits: tuple[EthUcySplitPath, ...]
    docs_path: str = ACQUISITION_DOC


def dataset_root(root: Path | str | None = None) -> Path:
    """Return the ETH/UCY dataset root, honoring the shared external-data root.

    Args:
        root: Explicit dataset root. When ``None`` the shared external-data
            registry resolves the path from ``ROBOT_SF_EXTERNAL_DATA_ROOT`` (or the
            repo-local default).

    Returns:
        The absolute dataset root path. The path is not required to exist.
    """

    if root is not None:
        return Path(root).expanduser().resolve()
    return resolve_asset_local_path_by_id(ETH_UCY_ASSET_ID).expanduser().resolve()


def _resolve_split_file(root: Path, spec: EthUcySplitSpec) -> EthUcySplitPath | None:
    """Resolve one split's trajectory file under ``root``.

    Returns:
        The resolved :class:`EthUcySplitPath`, or ``None`` when the split
        directory or an accepted trajectory file is absent.
    """

    split_dir = root / spec.directory
    if not split_dir.is_dir():
        return None
    for pattern, fmt in spec.file_candidates:
        matches = sorted(match for match in split_dir.glob(pattern) if match.is_file())
        if matches:
            return EthUcySplitPath(
                group=spec.group,
                split=spec.split,
                path=matches[0],
                format=fmt,
            )
    return None


def _resolve_dataset_paths(root: Path) -> tuple[EthUcyDatasetPaths | None, list[str]]:
    """Resolve every split file and report which documented splits are missing.

    Returns:
        A ``(dataset, missing)`` tuple. ``dataset`` is the resolved
        :class:`EthUcyDatasetPaths` when every split is present, else ``None``;
        ``missing`` lists the split ids without a resolvable trajectory file.
    """

    resolved: list[EthUcySplitPath] = []
    missing: list[str] = []
    for spec in _SPLIT_SPECS:
        split_path = _resolve_split_file(root, spec)
        if split_path is None:
            missing.append(spec.split)
        else:
            resolved.append(split_path)
    if missing:
        return None, missing
    return EthUcyDatasetPaths(root=root, splits=tuple(resolved)), missing


def is_available(root: Path | str | None = None) -> bool:
    """Return whether all documented ETH/UCY split files are staged.

    Returns ``False`` for an absent or incomplete layout so skip-if-absent tests
    can skip cleanly. A present-but-malformed file does not fail this check; it is
    reported by :func:`load_shape_contract`.
    """

    resolved_root = dataset_root(root)
    if not resolved_root.is_dir():
        return False
    dataset, _missing = _resolve_dataset_paths(resolved_root)
    return dataset is not None


def require_available(root: Path | str | None = None) -> EthUcyDatasetPaths:
    """Return resolved split paths or raise an actionable acquisition error."""

    resolved_root = dataset_root(root)
    if not resolved_root.is_dir():
        raise EthUcyDataError(
            f"{ETH_UCY_ASSET_ID} is not staged: {resolved_root} does not exist. "
            f"Follow the acquisition and staging steps in {ACQUISITION_DOC}. "
            f"You may set {EXTERNAL_DATA_ROOT_ENV} to a shared external-data root."
        )
    dataset, missing = _resolve_dataset_paths(resolved_root)
    if dataset is None:
        raise EthUcyDataError(
            f"{ETH_UCY_ASSET_ID} layout at {resolved_root} is incomplete. Missing "
            f"trajectory files for split(s): {', '.join(missing)}. Expected the "
            f"documented layout in {ACQUISITION_DOC}."
        )
    return dataset


def _read_numeric_rows(path: Path, split: str) -> tuple[list[list[float]], str]:
    """Parse a whitespace/comma numeric trajectory file into finite float rows.

    Comment lines (``#`` prefix) and blank lines are ignored. Every retained row
    must parse to finite floats and share a consistent column count.

    Returns:
        A ``(rows, delimiter)`` tuple where ``delimiter`` is ``"comma"`` or
        ``"whitespace"``.

    Raises:
        EthUcyDataError: If the file is empty, non-rectangular, contains a
            non-numeric or non-finite value, or is narrower than the structural
            column floor.
    """

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EthUcyDataError(
            f"ETH/UCY split '{split}' file {path} could not be read as text. "
            f"Re-stage per {ACQUISITION_DOC}."
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
                f"ETH/UCY split '{split}' file {path} contains a non-numeric value "
                f"in row '{line}'. See {ACQUISITION_DOC}."
            ) from exc
        if any(not math.isfinite(value) for value in values):
            raise EthUcyDataError(
                f"ETH/UCY split '{split}' file {path} contains a non-finite value "
                f"in row '{line}'. See {ACQUISITION_DOC}."
            )
        rows.append(values)

    if not rows:
        raise EthUcyDataError(
            f"ETH/UCY split '{split}' file {path} has no numeric data rows. See {ACQUISITION_DOC}."
        )
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise EthUcyDataError(
            f"ETH/UCY split '{split}' file {path} is not rectangular; rows have "
            f"varying column counts. See {ACQUISITION_DOC}."
        )
    if column_count < _MIN_NUMERIC_COLUMNS:
        raise EthUcyDataError(
            f"ETH/UCY split '{split}' file {path} has {column_count} columns; at "
            f"least {_MIN_NUMERIC_COLUMNS} (frame, id, x, y) are required. "
            f"See {ACQUISITION_DOC}."
        )
    return rows, delimiter


def _numeric_split_contract(split_path: EthUcySplitPath) -> dict[str, Any]:
    """Build the structural contract entry for an obsmat/txt numeric split.

    Returns:
        A mapping with ``row_count``, ``column_count``, and detected ``delimiter``.
    """

    rows, delimiter = _read_numeric_rows(split_path.path, split_path.split)
    return {
        "row_count": len(rows),
        "column_count": len(rows[0]),
        "delimiter": delimiter,
    }


def _vsp_split_contract(split_path: EthUcySplitPath) -> dict[str, Any]:
    """Validate a UCY ``.vsp`` file structurally and summarize its shape.

    The UCY Crowds-by-Example ``.vsp`` format opens with an integer agent/spline
    count header followed by control-point rows. Full spline parsing is out of
    scope for a shape contract, so this validates the structural header plus the
    presence of numeric control-point rows without asserting scene content.

    Returns:
        A mapping with ``row_count``, ``column_count``, ``delimiter``, and the
        declared ``agent_count`` header value.
    """

    path = split_path.path
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} could not be read as "
            f"text. Re-stage per {ACQUISITION_DOC}."
        ) from exc

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} is empty. See {ACQUISITION_DOC}."
        )
    header_token = lines[0].split()[0]
    try:
        agent_count = int(header_token)
    except ValueError as exc:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} does not start with an "
            f"integer agent/spline count header. See {ACQUISITION_DOC}."
        ) from exc
    if agent_count < 0:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} declares a negative "
            f"agent/spline count. See {ACQUISITION_DOC}."
        )

    numeric_rows = 0
    max_columns = 0
    for line in lines[1:]:
        tokens = line.split()
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in values):
            raise EthUcyDataError(
                f"ETH/UCY split '{split_path.split}' file {path} contains a "
                f"non-finite control-point value. See {ACQUISITION_DOC}."
            )
        numeric_rows += 1
        max_columns = max(max_columns, len(values))
    if numeric_rows == 0:
        raise EthUcyDataError(
            f"ETH/UCY split '{split_path.split}' file {path} has no numeric "
            f"control-point rows after its header. See {ACQUISITION_DOC}."
        )
    return {
        "row_count": numeric_rows,
        "column_count": max_columns,
        "delimiter": "whitespace",
        "agent_count": agent_count,
    }


def load_shape_contract(root: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the cheap ETH/UCY per-split shape contract.

    Args:
        root: Explicit dataset root, or ``None`` to resolve via the shared
            external-data registry.

    Returns:
        A structural contract mapping with per-split shape metadata (row count,
        column count, detected delimiter, resolved format, and relative path).
        No trajectory content is asserted.

    Raises:
        EthUcyDataError: If the dataset is absent/incomplete or any staged split
            file is malformed.
    """

    dataset = require_available(root)
    splits: dict[str, dict[str, Any]] = {}
    for split_path in dataset.splits:
        if split_path.format == "vsp":
            shape = _vsp_split_contract(split_path)
        else:
            shape = _numeric_split_contract(split_path)
        splits[split_path.split] = {
            "group": split_path.group,
            "format": split_path.format,
            "path": split_path.path.relative_to(dataset.root).as_posix(),
            **shape,
        }
    return {
        "asset_id": ETH_UCY_ASSET_ID,
        "root": str(dataset.root),
        "docs_path": dataset.docs_path,
        "splits": splits,
    }
