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


def _validated_relative_asset_path(asset: dict[str, Any]) -> str:
    """Return a safe non-empty relative path from one asset manifest entry."""
    rel_val = asset.get("relative_path")
    if not isinstance(rel_val, str) or not rel_val.strip():
        raise ValueError(f"Asset '{asset.get('key')}' must have a non-empty relative path")
    rel = rel_val.strip()
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"Asset '{asset.get('key')}' must be a relative path within the root")
    return rel


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
            rel = _validated_relative_asset_path(asset)
            kind = str(asset.get("kind") or "").strip()
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


# Conversion-readiness statuses returned by ``conversion_readiness``. ``ready``
# means every conversion-required source asset is staged and conversion may
# proceed. ``blocked_pending_source_assets`` means at least one required asset
# (notably the ETH traversible ``data.pkl``) is missing, so conversion must fail
# closed instead of fabricating a placeholder map.
READY = "ready"
BLOCKED = "blocked_pending_source_assets"

# Planned-output keys that name a generated *map artifact* (the conversion's
# product). These are the outputs forbidden while the batch is blocked, because
# a pre-existing copy may be a hand-authored or inferred placeholder. Other
# planned outputs such as the provenance note are documentation that is expected
# to exist during the blocked phase, so they are not treated as placeholder risk.
CONVERSION_ARTIFACT_OUTPUT_KEYS = frozenset({"map_svg", "scenario_config"})


def _existing_planned_outputs(
    map_reports: list[dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, str]]:
    """Return planned map artifacts that already exist on disk while blocked.

    The map-conversion issue (#1134) explicitly forbids committing a
    hand-authored or inferred ETH-like SVG while the official source assets are
    not staged. When the batch is blocked, a pre-existing conversion *artifact*
    (for example ``maps/svg_maps/socnavbench/socnavbench_eth.svg`` or the smoke
    scenario) is a provenance risk: it may be a placeholder masquerading as an
    official conversion. Only artifact outputs in
    :data:`CONVERSION_ARTIFACT_OUTPUT_KEYS` are flagged; the provenance note and
    other documentation outputs are expected to exist during the blocked phase.
    """
    found: list[dict[str, str]] = []
    for map_report in map_reports:
        planned = map_report.get("planned_outputs")
        if not isinstance(planned, dict):
            continue
        for output_key, rel in planned.items():
            if str(output_key) not in CONVERSION_ARTIFACT_OUTPUT_KEYS:
                continue
            if not isinstance(rel, str) or not rel.strip():
                continue
            candidate = repo_root / rel.strip()
            if candidate.exists():
                found.append(
                    {
                        "map_id": str(map_report.get("map_id", "")),
                        "output_key": str(output_key),
                        "relative_path": rel.strip(),
                        "path": str(candidate),
                    }
                )
    return found


def conversion_readiness(
    *,
    manifest_path: Path,
    socnav_root: Path,
    batch_id: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Decide whether SocNavBench map conversion may begin for one batch.

    This is a fail-closed preflight gate layered on top of :func:`validate_batch`.
    The underlying validator reports raw source-asset existence; this function
    narrows that to the conversion decision:

    - ``conversion_ready`` is ``True`` only when every asset marked
      ``required_for_conversion`` is staged. Any missing required asset (the ETH
      traversible ``data.pkl`` in particular) yields ``status`` ``blocked`` and
      ``conversion_ready`` ``False``.
    - When blocked, planned outputs that already exist are reported as
      ``placeholder_risk`` so a hand-authored or inferred map cannot be silently
      mistaken for an official conversion.
    - ``next_action`` names the smallest concrete step toward unblocking.

    The function never converts assets, downloads data, or writes maps; it only
    inspects staged state and returns a verdict.
    """
    report = validate_batch(
        manifest_path=manifest_path,
        socnav_root=socnav_root,
        batch_id=batch_id,
    )
    missing = report["missing_required"]
    ready = not missing
    placeholder_risk = [] if ready else _existing_planned_outputs(report["maps"], repo_root)

    if ready:
        next_action = (
            "Conversion-required source assets are staged. Record source "
            "checksums/provenance, then run the official conversion."
        )
    else:
        missing_paths = ", ".join(sorted(item["relative_path"] for item in missing))
        next_action = (
            "Stage the official SocNavBench source assets under the socnav root "
            f"({missing_paths}), then re-run this preflight. Do not author a "
            "placeholder map while blocked."
        )

    return {
        "manifest": report["manifest"],
        "socnav_root": report["socnav_root"],
        "repo_root": str(repo_root),
        "batch_id": report["batch_id"],
        "batch_status": report["batch_status"],
        "conversion_ready": ready,
        "status": READY if ready else BLOCKED,
        "missing_required": missing,
        "placeholder_risk": placeholder_risk,
        "next_action": next_action,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--socnav-root", type=Path, default=DEFAULT_SOCNAV_ROOT)
    parser.add_argument("--batch-id", default="eth_first")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Emit a fail-closed map-conversion readiness verdict instead of the "
            "raw asset-existence report. Exits non-zero while conversion is "
            "blocked on missing source assets."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Validate or preflight one SocNavBench import batch and return an exit code."""
    args = parse_args()
    if args.preflight:
        report = conversion_readiness(
            manifest_path=args.manifest.resolve(),
            socnav_root=args.socnav_root.resolve(),
            batch_id=str(args.batch_id),
        )
        ok = bool(report["conversion_ready"])
    else:
        report = validate_batch(
            manifest_path=args.manifest.resolve(),
            socnav_root=args.socnav_root.resolve(),
            batch_id=str(args.batch_id),
        )
        ok = bool(report["ok"])
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
