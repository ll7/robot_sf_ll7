"""User-facing ``robot-sf datasets`` UX glue.

Thin, beginner-facing surface over the existing external-data registry and
provenance/checksum machinery in :mod:`scripts.tools.manage_external_data`.
The functions here produce structured payloads so the CLI dispatcher
(:mod:`robot_sf.cli`) and tests can share one implementation.

Plain-language summary: ``datasets list`` shows registered datasets and whether
the local layout is staged; ``datasets verify`` checks each staged asset's
required-path layout and, when a provenance manifest pins a checksum, recomputes
the aggregate tree checksum and reports pass/fail; ``datasets prepare <id>``
prints exact acquisition instructions for license-restricted data and then
verifies the local layout WITHOUT downloading.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.data.external.provenance import DEFAULT_MANIFEST_DIR
from scripts.tools.manage_external_data import (
    ASSETS,
    AssetSpec,
    _get_asset,
    _matched_paths_for_report,
    _tree_checksum,
    check_asset,
    list_assets,
)

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from collections.abc import Sequence

__all__ = [
    "list_datasets",
    "prepare_dataset",
    "verify_datasets",
]

# Per-artifact verify statuses.
STATUS_AVAILABLE = "available"
STATUS_INCOMPLETE = "incomplete"
STATUS_MISSING = "missing"
STATUS_CHECKSUM_OK = "checksum_ok"
STATUS_CHECKSUM_MISMATCH = "checksum_mismatch"
STATUS_NO_PINNED_CHECKSUM = "no_pinned_checksum"


def _manifest_path_for(asset: AssetSpec) -> Path:
    """Return the default provenance-manifest path for an asset."""
    return DEFAULT_MANIFEST_DIR / f"{asset.asset_id}.provenance.json"


def _asset_list_summary(asset: AssetSpec, report: dict[str, Any]) -> dict[str, Any]:
    """Build one ``list_datasets`` row from an asset spec plus a check report.

    Returns:
        dict[str, Any]: Per-dataset summary row.
    """
    return {
        "asset_id": asset.asset_id,
        "title": asset.title,
        "source_url": asset.source_url,
        "license_note": asset.license_note,
        "license_url": asset.license_url,
        "auto_download_allowed": asset.auto_download_allowed,
        "status": report["status"],
        "ok": report["ok"],
        "expected_local_path": report["expected_local_path"],
    }


def list_datasets() -> list[dict[str, Any]]:
    """Return one summary row per registered dataset, with local staging status.

    Returns:
        list[dict[str, Any]]: Per-dataset summary rows in registry order.
    """
    rows: list[dict[str, Any]] = []
    for asset in list_assets():
        report = check_asset(asset.asset_id)
        rows.append(_asset_list_summary(asset, report))
    return rows


def _required_paths_view(asset: AssetSpec) -> list[dict[str, Any]]:
    """Return the required-path descriptors for an asset, for instruction output."""
    return [
        {
            "pattern": required.pattern,
            "kind": required.kind,
            "description": required.description,
            "group": required.group,
        }
        for required in asset.required_paths
    ]


def prepare_dataset(asset_id: str, *, source_path: str | Path | None = None) -> dict[str, Any]:
    """Print acquisition instructions and verify local layout WITHOUT downloading.

    For license-restricted datasets, this is the safe path: it never attempts to
    fetch restricted data. It surfaces the official source URL, license note, and
    access instructions, then runs the layout check over the local path so a
    user can confirm they staged the right files.

    Returns:
        dict[str, Any]: Acquisition instructions plus the local layout verdict.
    """
    asset = _get_asset(asset_id)
    report = check_asset(asset_id, source_path=Path(source_path) if source_path else None)
    return {
        "schema": "robot_sf_datasets_prepare.v1",
        "asset_id": asset.asset_id,
        "title": asset.title,
        "source_url": asset.source_url,
        "license_note": asset.license_note,
        "license_url": asset.license_url,
        "access_note": asset.access_note,
        "auto_download_allowed": asset.auto_download_allowed,
        "required_paths": _required_paths_view(asset),
        "acquisition_doc": _asset_acquisition_doc(asset),
        "local_layout": {
            "ok": report["ok"],
            "status": report["status"],
            "source_path": report["source_path"],
            "expected_local_path": report["expected_local_path"],
            "missing_required_paths": report["missing_required_paths"],
            "matched_required_paths": report["matched_required_paths"],
            "action": report.get("action"),
        },
    }


def _asset_acquisition_doc(asset: AssetSpec) -> str | None:
    """Return the dataset's docs/datasets acquisition path if declared, else None."""
    candidate = Path("docs/datasets") / f"{asset.asset_id}.md"
    if candidate.exists():
        return str(candidate)
    return None


def verify_datasets(
    *,
    asset_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify each dataset's layout and pinned checksum, returning pass/fail.

    For every registered asset this runs the required-path layout check
    (:func:`scripts.tools.manage_external_data.check_asset`). When the staged
    layout is available AND a provenance manifest pins a ``tree_sha256``, the
    aggregate tree checksum is recomputed over the matched files and compared
    against the pin, yielding a ``checksum_ok`` / ``checksum_mismatch`` verdict.
    Available layouts without a pinned checksum report ``no_pinned_checksum``.

    Returns:
        dict[str, Any]: Report with ``ok`` (all pinned checksums passed),
        per-artifact ``results``, and aggregate counts.
    """
    if asset_ids is not None:
        # Validate ids via _get_asset (raises on unknown).
        requested = [_get_asset(asset_id).asset_id for asset_id in asset_ids]
        targets = [asset for asset in ASSETS if asset.asset_id in requested]
    else:
        targets = list(ASSETS)

    results: list[dict[str, Any]] = []
    pinned_total = 0
    pass_count = 0
    for asset in targets:
        result = _verify_one_dataset(asset)
        results.append(result)
        if result["pinned_checksum"]:
            pinned_total += 1
            if result["checksum_status"] == STATUS_CHECKSUM_OK:
                pass_count += 1

    return {
        "schema": "robot_sf_datasets_verify.v1",
        "ok": pinned_total > 0 and pass_count == pinned_total,
        "checked": len(results),
        "pinned_checksums": pinned_total,
        "passed": pass_count,
        "results": results,
    }


def _verify_one_dataset(asset: AssetSpec) -> dict[str, Any]:
    """Build the layout + checksum verdict for a single dataset.

    Returns:
        dict[str, Any]: Per-dataset verify result with layout and checksum status.
    """
    report = check_asset(asset.asset_id)
    result: dict[str, Any] = {
        "asset_id": asset.asset_id,
        "title": asset.title,
        "layout_status": report["status"],
        "layout_ok": report["ok"],
        "source_path": report["source_path"],
        "expected_tree_sha256": None,
        "observed_tree_sha256": None,
        "pinned_checksum": False,
        "checksum_status": STATUS_NO_PINNED_CHECKSUM,
        "missing_required_paths": report["missing_required_paths"],
    }
    if not report["ok"]:
        # Layout incomplete: no checksum to verify.
        return result

    source_root = Path(report["source_path"])
    manifest_path = _manifest_path_for(asset)
    pinned = _load_pinned_tree_sha256(manifest_path)
    if pinned is None:
        # Available layout, but no manifest/pinned checksum to compare.
        return result

    result["pinned_checksum"] = True
    result["expected_tree_sha256"] = pinned
    matched_paths = _matched_paths_for_report(source_root, report)
    try:
        recomputed = _tree_checksum(source_root, matched_paths)
    except OSError as exc:
        result["checksum_status"] = "error"
        result["error"] = str(exc)
        return result
    observed = recomputed["tree_sha256"]
    result["observed_tree_sha256"] = observed
    result["checksum_status"] = (
        STATUS_CHECKSUM_OK if observed == pinned else STATUS_CHECKSUM_MISMATCH
    )
    return result


def _load_pinned_tree_sha256(manifest_path: Path) -> str | None:
    """Return the pinned ``tree_sha256`` from a provenance manifest, if present."""
    import json  # noqa: PLC0415

    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    value = manifest.get("tree_sha256")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()
