"""Prepare and validate SocNavBench third-party data assets.

This helper is intentionally license-safe:
- it does not download third-party datasets automatically,
- it can optionally copy already-downloaded local assets into
  ``third_party/socnavbench``,
- it validates required directory layout for benchmark preflight.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOCNAV_ROOT = REPO_ROOT / "third_party" / "socnavbench"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "output" / "SocNavBench"


@dataclass(frozen=True)
class AssetPath:
    """Logical SocNav asset path with requirement semantics."""

    key: str
    relative_path: str
    required_for_schematic: bool
    required_for_full_render: bool
    description: str


ASSET_PATHS: tuple[AssetPath, ...] = (
    AssetPath(
        key="wayptnav_data",
        relative_path="wayptnav_data",
        required_for_schematic=True,
        required_for_full_render=True,
        description="Precomputed control-pipeline/waypoint navigation data",
    ),
    AssetPath(
        key="sbpd_dataset",
        relative_path="sd3dis/stanford_building_parser_dataset",
        required_for_schematic=True,
        required_for_full_render=True,
        description="S3DIS/SBPD dataset root used by SocNav map loader",
    ),
    AssetPath(
        key="sbpd_traversibles",
        relative_path="sd3dis/stanford_building_parser_dataset/traversibles",
        required_for_schematic=True,
        required_for_full_render=True,
        description="Precomputed traversible maps",
    ),
    AssetPath(
        key="surreal_meshes",
        relative_path="surreal/code/human_meshes",
        required_for_schematic=False,
        required_for_full_render=True,
        description="SURREAL-generated human meshes (full-render only)",
    ),
    AssetPath(
        key="surreal_textures",
        relative_path="surreal/code/human_textures",
        required_for_schematic=False,
        required_for_full_render=True,
        description="SURREAL-generated human textures (full-render only)",
    ),
)

SOURCE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "wayptnav_data": ("wayptnav_data",),
    "sbpd_dataset": (
        "sd3dis/sd3dis-public/stanford_building_parser_dataset",
        "sd3dis/stanford_building_parser_dataset",
    ),
    "sbpd_traversibles": (
        "sd3dis/sd3dis-public/stanford_building_parser_dataset/traversibles",
        "sd3dis/stanford_building_parser_dataset/traversibles",
    ),
    "surreal_meshes": ("surreal/code/human_meshes",),
    "surreal_textures": ("surreal/code/human_textures",),
}


def _required(asset: AssetPath, render_mode: str) -> bool:
    if render_mode == "full-render":
        return asset.required_for_full_render
    return asset.required_for_schematic


def evaluate_assets(socnav_root: Path, *, render_mode: str) -> dict[str, Any]:
    """Return a validation report for SocNav data layout."""
    entries: list[dict[str, Any]] = []
    missing_required: list[str] = []
    for asset in ASSET_PATHS:
        target = socnav_root / asset.relative_path
        exists = target.is_dir()
        required = _required(asset, render_mode)
        if required and not exists:
            missing_required.append(asset.key)
        entries.append(
            {
                "key": asset.key,
                "relative_path": asset.relative_path,
                "exists": exists,
                "required": required,
                "description": asset.description,
            }
        )
    return {
        "socnav_root": str(socnav_root),
        "render_mode": render_mode,
        "ok": not missing_required,
        "missing_required": missing_required,
        "assets": entries,
    }


def _find_source(source_root: Path, key: str) -> Path | None:
    candidates = SOURCE_CANDIDATES.get(key, ())
    for rel in candidates:
        candidate = source_root / rel
        if candidate.is_dir():
            return candidate
    return None


def copy_available_assets(
    *,
    socnav_root: Path,
    source_root: Path,
    render_mode: str,
    overwrite: bool,
) -> list[str]:
    """Copy available assets from local source root into SocNav root.

    Returns:
        list[str]: Human-readable copy actions taken.
    """
    actions: list[str] = []
    resolved_socnav_root = socnav_root.resolve()
    for asset in ASSET_PATHS:
        if not _required(asset, render_mode):
            continue
        src = _find_source(source_root, asset.key)
        if src is None:
            continue
        src_resolved = src.resolve()
        dst = socnav_root / asset.relative_path
        dst_resolved = (resolved_socnav_root / asset.relative_path).resolve()
        if src_resolved == dst_resolved:
            actions.append(f"skip self-copy: {asset.relative_path}")
            continue
        if dst.exists():
            if not overwrite:
                actions.append(f"skip existing: {asset.relative_path}")
                continue
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
        actions.append(f"copied: {src} -> {dst}")
    return actions


def _print_report(report: dict[str, Any]) -> None:
    status = "OK" if report["ok"] else "MISSING_REQUIRED_ASSETS"
    print(f"[socnav-assets] status={status}")
    print(f"[socnav-assets] socnav_root={report['socnav_root']}")
    print(f"[socnav-assets] render_mode={report['render_mode']}")
    for entry in report["assets"]:
        req = "required" if entry["required"] else "optional"
        state = "present" if entry["exists"] else "missing"
        print(f"- {entry['key']}: {state} ({req}) -> {entry['relative_path']}")
    if report["missing_required"]:
        print("")
        print("Missing required assets:")
        for key in report["missing_required"]:
            print(f"- {key}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SocNav asset staging/validation."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate or stage SocNavBench assets for third_party/socnavbench "
            "without downloading licensed datasets."
        )
    )
    parser.add_argument(
        "--socnav-root",
        type=Path,
        default=DEFAULT_SOCNAV_ROOT,
        help="Target SocNavBench root (default: third_party/socnavbench).",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Local source root to copy from (default: output/SocNavBench).",
    )
    parser.add_argument(
        "--render-mode",
        choices=("schematic", "full-render"),
        default="schematic",
        help="Asset requirement profile. full-render additionally requires SURREAL assets.",
    )
    parser.add_argument(
        "--copy-from-source",
        action="store_true",
        help="Copy available assets from --source-root into --socnav-root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing asset directories when copying.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path for machine-readable validation report.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the SocNav asset helper and return a process exit code."""
    args = parse_args()
    socnav_root = args.socnav_root.resolve()
    source_root = args.source_root.resolve()
    socnav_root.mkdir(parents=True, exist_ok=True)

    if args.copy_from_source:
        actions = copy_available_assets(
            socnav_root=socnav_root,
            source_root=source_root,
            render_mode=args.render_mode,
            overwrite=bool(args.overwrite),
        )
        if actions:
            print("[socnav-assets] copy actions:")
            for action in actions:
                print(f"- {action}")
        else:
            print("[socnav-assets] no assets copied (none found or all already present).")

    report = evaluate_assets(socnav_root=socnav_root, render_mode=args.render_mode)
    _print_report(report)

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[socnav-assets] wrote report: {args.report_json}")

    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
