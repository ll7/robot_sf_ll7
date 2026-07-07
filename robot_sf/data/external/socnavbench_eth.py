"""SocNavBench ETH external-data shape contract.

The loader only inspects locally staged files. It never downloads or vendors the
license-gated SocNavBench/S3DIS ETH dataset bytes.

Security:
    The traversible pickle is loaded through a restricted unpickler that only allows
    NumPy reconstruction symbols. Arbitrary pickle files remain untrusted.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.common.safe_pickle import (
    SOCNAVBENCH_TRAVERSIBLE_ALLOWED_GLOBALS,
    UnsafePickleError,
    restricted_pickle_load,
)
from scripts.tools.manage_external_data import (
    EXTERNAL_DATA_ROOT_ENV,
    check_asset,
    resolve_asset_local_path_by_id,
)

ASSET_ID = "socnavbench-s3dis-eth"
_S3DIS_BASE = Path("sd3dis") / "stanford_building_parser_dataset"
ETH_MESH_RELATIVE_PATH = _S3DIS_BASE / "mesh" / "ETH"
ETH_TRAVERSIBLE_RELATIVE_PATH = _S3DIS_BASE / "traversibles" / "ETH" / "data.pkl"
ACQUISITION_DOC = "docs/datasets/socnavbench-s3dis-eth.md"


class SocNavBenchEthDataError(RuntimeError):
    """Raised when staged SocNavBench ETH data is absent or structurally invalid."""


@dataclass(frozen=True)
class SocNavBenchEthLayout:
    """Resolved filesystem layout for the locally staged SocNavBench ETH asset."""

    root: Path
    mesh_dir: Path
    traversible_pickle: Path


@dataclass(frozen=True)
class SocNavBenchEthShapeContract:
    """Cheap structural summary of the ETH traversible pickle."""

    resolution: float
    traversible_shape: tuple[int, int]
    traversible_dtype: str


def resolve_root(root: Path | str | None = None) -> Path:
    """Return the SocNavBench root, honoring the shared external-data registry."""

    if root is not None:
        return Path(root).expanduser().resolve()
    return resolve_asset_local_path_by_id(ASSET_ID).expanduser().resolve()


def expected_layout(root: Path | str | None = None) -> SocNavBenchEthLayout:
    """Return expected ETH paths without asserting that data is staged."""

    resolved = resolve_root(root)
    return SocNavBenchEthLayout(
        root=resolved,
        mesh_dir=resolved / ETH_MESH_RELATIVE_PATH,
        traversible_pickle=resolved / ETH_TRAVERSIBLE_RELATIVE_PATH,
    )


def availability_report(root: Path | str | None = None) -> dict[str, Any]:
    """Return the canonical external-data registry availability report."""

    return check_asset(ASSET_ID, source_path=resolve_root(root))


def is_available(root: Path | str | None = None) -> bool:
    """Return whether all required SocNavBench ETH source assets are staged."""

    return bool(availability_report(root).get("ok", False))


def require_available(root: Path | str | None = None) -> SocNavBenchEthLayout:
    """Return staged layout or raise an actionable acquisition error."""

    report = availability_report(root)
    if report.get("ok", False):
        return expected_layout(root)

    missing_paths = report.get("missing_required_paths") or [
        str(ETH_MESH_RELATIVE_PATH),
        str(ETH_TRAVERSIBLE_RELATIVE_PATH),
    ]
    missing = ", ".join(str(path) for path in missing_paths)
    action = str(report.get("action") or "stage the official SocNavBench ETH assets")
    raise SocNavBenchEthDataError(
        f"{ASSET_ID} is not staged or is incomplete at {report['source_path']}. "
        f"Missing required paths: {missing or 'unknown'}. {action} "
        f"See {ACQUISITION_DOC}. You may set {EXTERNAL_DATA_ROOT_ENV} to a shared "
        "external-data root."
    )


def load_shape_contract(root: Path | str | None = None) -> SocNavBenchEthShapeContract:
    """Load and validate the cheap traversible-map shape contract.

    Returns:
        A structural summary of the staged ETH traversible pickle.
    """

    layout = require_available(root)
    try:
        with layout.traversible_pickle.open("rb") as handle:
            payload = restricted_pickle_load(
                io.BytesIO(handle.read()),
                allowed_globals=SOCNAVBENCH_TRAVERSIBLE_ALLOWED_GLOBALS,
                label="SocNavBench ETH traversible",
            )
    except UnsafePickleError as exc:
        raise SocNavBenchEthDataError(
            f"Unsafe pickle rejected for {layout.traversible_pickle}: {exc}. "
            f"Re-stage official {ASSET_ID} assets if this is a legitimate file."
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on malformed external bytes.
        raise SocNavBenchEthDataError(
            f"Could not load {layout.traversible_pickle}. Re-stage official "
            f"{ASSET_ID} assets and re-run the shape test."
        ) from exc

    if not isinstance(payload, dict):
        raise SocNavBenchEthDataError(
            f"{layout.traversible_pickle} must contain a mapping with 'resolution' "
            "and 'traversible' keys."
        )
    if "resolution" not in payload or "traversible" not in payload:
        raise SocNavBenchEthDataError(
            f"{layout.traversible_pickle} missing required keys: resolution, traversible."
        )

    resolution = payload["resolution"]
    if not isinstance(resolution, (int, float, np.integer, np.floating)):
        raise SocNavBenchEthDataError("SocNavBench ETH resolution must be numeric.")
    resolution_float = float(resolution)
    if not np.isfinite(resolution_float) or resolution_float <= 0:
        raise SocNavBenchEthDataError("SocNavBench ETH resolution must be finite and positive.")

    traversible = np.asarray(payload["traversible"])
    if traversible.ndim != 2 or 0 in traversible.shape:
        raise SocNavBenchEthDataError("SocNavBench ETH traversible must be a non-empty 2D array.")
    if not (
        np.issubdtype(traversible.dtype, np.bool_) or np.issubdtype(traversible.dtype, np.number)
    ):
        raise SocNavBenchEthDataError("SocNavBench ETH traversible dtype must be bool or numeric.")
    if np.issubdtype(traversible.dtype, np.number) and not np.isfinite(traversible).all():
        raise SocNavBenchEthDataError(
            "SocNavBench ETH traversible array contains non-finite values."
        )

    return SocNavBenchEthShapeContract(
        resolution=resolution_float,
        traversible_shape=tuple(int(axis) for axis in traversible.shape),
        traversible_dtype=str(traversible.dtype),
    )
