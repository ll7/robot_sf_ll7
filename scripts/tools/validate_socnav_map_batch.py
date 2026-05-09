#!/usr/bin/env python3
"""Validate staged SocNavBench map-import source assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "maps" / "socnavbench_import_batches.yaml"
DEFAULT_SOCNAV_ROOT = REPO_ROOT / "third_party" / "socnavbench"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _find_batch(manifest: dict[str, Any], batch_id: str) -> dict[str, Any]:
    """Return one batch from the SocNav import manifest."""
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise TypeError("Manifest must contain a 'batches' list")
    for batch in batches:
        if isinstance(batch, dict) and str(batch.get("batch_id")) == batch_id:
            return batch
    raise KeyError(f"Unknown SocNavBench import batch: {batch_id}")


def _asset_exists(path: Path, kind: str) -> bool:
    """Check whether a staged source asset exists with the expected filesystem kind."""
    if kind == "directory":
        return path.is_dir()
    if kind == "file":
        return path.is_file()
    raise ValueError(f"Unsupported asset kind: {kind}")


def validate_batch(
    *,
    manifest_path: Path,
    socnav_root: Path,
    batch_id: str,
) -> dict[str, Any]:
    """Validate staged source assets for one SocNavBench map-import batch."""
    manifest = _load_yaml(manifest_path)
    batch = _find_batch(manifest, batch_id)
    maps = batch.get("maps")
    if not isinstance(maps, list) or not maps:
        raise TypeError(f"Batch '{batch_id}' must contain at least one map")

    map_reports: list[dict[str, Any]] = []
    missing_required: list[dict[str, str]] = []
    for map_entry in maps:
        if not isinstance(map_entry, dict):
            raise TypeError(f"Map entry in batch '{batch_id}' must be a mapping")
        asset_reports: list[dict[str, Any]] = []
        assets = map_entry.get("source_assets")
        if not isinstance(assets, list) or not assets:
            raise TypeError(f"Map '{map_entry.get('map_id')}' must declare source assets")
        for asset in assets:
            if not isinstance(asset, dict):
                raise TypeError(f"Source asset in map '{map_entry.get('map_id')}' is invalid")
            rel = str(asset.get("relative_path", "")).strip()
            kind = str(asset.get("kind", "")).strip()
            required = bool(asset.get("required_for_conversion", False))
            path = socnav_root / rel
            exists = _asset_exists(path, kind)
            asset_report = {
                "key": str(asset.get("key", "")),
                "relative_path": rel,
                "kind": kind,
                "required_for_conversion": required,
                "exists": exists,
                "path": str(path),
            }
            asset_reports.append(asset_report)
            if required and not exists:
                missing_required.append(
                    {
                        "map_id": str(map_entry.get("map_id", "")),
                        "asset_key": asset_report["key"],
                        "relative_path": rel,
                    }
                )
        map_reports.append(
            {
                "map_id": str(map_entry.get("map_id", "")),
                "upstream_map_name": str(map_entry.get("upstream_map_name", "")),
                "assets": asset_reports,
                "planned_outputs": map_entry.get("planned_outputs", {}),
            }
        )

    return {
        "manifest": str(manifest_path),
        "socnav_root": str(socnav_root),
        "batch_id": batch_id,
        "batch_status": str(batch.get("status", "unknown")),
        "ok": not missing_required,
        "missing_required": missing_required,
        "maps": map_reports,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--socnav-root", type=Path, default=DEFAULT_SOCNAV_ROOT)
    parser.add_argument("--batch-id", default="eth_first")
    parser.add_argument("--report-json", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate one SocNavBench import batch and return a process exit code."""
    args = parse_args()
    report = validate_batch(
        manifest_path=args.manifest.resolve(),
        socnav_root=args.socnav_root.resolve(),
        batch_id=str(args.batch_id),
    )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
