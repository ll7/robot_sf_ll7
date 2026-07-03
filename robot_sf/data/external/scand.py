"""SCAND socially-compliant navigation demonstrations shape contract.

Plain-language summary: this loader only inspects locally staged SCAND files
(ROS bags plus exported CSV/JSON tables and a license/README copy). It never
downloads, vendors, or redistributes the terms-gated SCAND (Socially CompliAnt
Navigation Dataset, Karnan et al. 2022) bytes. The contract is cheap and
structural: it confirms the documented recording + license/readme layout exists
and that each staged CSV/JSON export parses and each ROS bag is non-empty. It
asserts no dataset content and makes no oracle-imitation claim.

The expected layout, source, and staging workflow are documented in
``docs/datasets/scand.md``; the availability contract is the ``scand-demos``
registry entry in ``scripts/tools/manage_external_data.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robot_sf.data.external import recording_shape_contract as _engine
from robot_sf.data.external.recording_shape_contract import (
    EXTERNAL_DATA_ROOT_ENV,
    RecordingDatasetError,
    RecordingDatasetPaths,
    RecordingDatasetSpec,
)

if TYPE_CHECKING:
    from pathlib import Path

SCAND_ASSET_ID = "scand-demos"
ACQUISITION_DOC = "docs/datasets/scand.md"

__all__ = [
    "ACQUISITION_DOC",
    "EXTERNAL_DATA_ROOT_ENV",
    "SCAND_ASSET_ID",
    "ScandDataError",
    "dataset_root",
    "is_available",
    "load_shape_contract",
    "require_available",
]


class ScandDataError(RecordingDatasetError):
    """Raised when staged SCAND data is present but structurally invalid."""


_SPEC = RecordingDatasetSpec(
    asset_id=SCAND_ASSET_ID,
    title="SCAND socially compliant navigation demonstrations",
    docs_path=ACQUISITION_DOC,
    error_cls=ScandDataError,
)


def dataset_root(root: Path | str | None = None) -> Path:
    """Return the SCAND dataset root, honoring the shared external-data root."""

    return _engine.resolve_root(_SPEC, root)


def is_available(root: Path | str | None = None) -> bool:
    """Return whether the documented SCAND recording layout is staged."""

    return _engine.is_available(_SPEC, root)


def require_available(root: Path | str | None = None) -> RecordingDatasetPaths:
    """Return resolved SCAND files or raise an actionable acquisition error."""

    return _engine.require_available(_SPEC, root)


def load_shape_contract(root: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the cheap SCAND recording-style shape contract.

    Returns:
        A structural contract mapping with recording counts and per-file shape
        metadata; see :func:`recording_shape_contract.load_shape_contract`.
    """

    return _engine.load_shape_contract(_SPEC, root)
