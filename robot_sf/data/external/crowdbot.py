"""CrowdBot robot-in-crowd external-data shape contract.

Plain-language summary: this loader only inspects locally staged CrowdBot files
(ROS bags plus exported CSV/JSON tables and a license/README copy). It never
downloads, vendors, or redistributes the research-access-gated EPFL LASA CrowdBot
dataset bytes. The contract is cheap and structural: it confirms the documented
recording + license/readme layout exists and that each staged CSV/JSON export
parses and each ROS bag is non-empty. It asserts no dataset content and makes no
robot-reaction-realism claim.

The expected layout, source, and staging workflow are documented in
``docs/datasets/crowdbot.md``; the availability contract is the ``crowdbot``
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

CROWDBOT_ASSET_ID = "crowdbot"
ACQUISITION_DOC = "docs/datasets/crowdbot.md"

__all__ = [
    "ACQUISITION_DOC",
    "CROWDBOT_ASSET_ID",
    "EXTERNAL_DATA_ROOT_ENV",
    "CrowdBotDataError",
    "dataset_root",
    "is_available",
    "load_shape_contract",
    "require_available",
]


class CrowdBotDataError(RecordingDatasetError):
    """Raised when staged CrowdBot data is present but structurally invalid."""


_SPEC = RecordingDatasetSpec(
    asset_id=CROWDBOT_ASSET_ID,
    title="CrowdBot robot-in-crowd dataset",
    docs_path=ACQUISITION_DOC,
    error_cls=CrowdBotDataError,
)


def dataset_root(root: Path | str | None = None) -> Path:
    """Return the CrowdBot dataset root, honoring the shared external-data root."""

    return _engine.resolve_root(_SPEC, root)


def is_available(root: Path | str | None = None) -> bool:
    """Return whether the documented CrowdBot recording layout is staged."""

    return _engine.is_available(_SPEC, root)


def require_available(root: Path | str | None = None) -> RecordingDatasetPaths:
    """Return resolved CrowdBot files or raise an actionable acquisition error."""

    return _engine.require_available(_SPEC, root)


def load_shape_contract(root: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the cheap CrowdBot recording-style shape contract.

    Returns:
        A structural contract mapping with recording counts and per-file shape
        metadata; see :func:`recording_shape_contract.load_shape_contract`.
    """

    return _engine.load_shape_contract(_SPEC, root)
