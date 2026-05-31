#!/usr/bin/env python3
"""License-safe assistant for local external data staging and provenance.

The helper deliberately does not bypass dataset license gates. For restricted assets it explains
the official acquisition path, validates a user-provided local path, and writes a compact manifest
that records provenance and aggregate checksums while raw data stays local.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "output" / "external_data" / "manifests"


class ExternalDataError(RuntimeError):
    """Raised when external data cannot be safely staged or validated."""


@dataclass(frozen=True)
class RequiredPath:
    """One required path pattern for an external asset."""

    pattern: str
    kind: str
    description: str
    group: str | None = None


@dataclass(frozen=True)
class AssetSpec:
    """Static registry entry for one supported external asset group."""

    asset_id: str
    title: str
    expected_local_path: Path
    source_url: str
    source_note: str
    license_note: str
    access_note: str
    required_paths: tuple[RequiredPath, ...]
    related_issues: tuple[int, ...]
    auto_download_allowed: bool = False


ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(
        asset_id="sdd",
        title="Stanford Drone Dataset annotations",
        expected_local_path=REPO_ROOT / "output" / "external_data" / "sdd",
        source_url="https://cvgl.stanford.edu/projects/uav_data/",
        source_note=(
            "Official Stanford Drone Dataset project page. Stage only files obtained under the "
            "dataset terms; Robot SF currently needs annotation text files for the SDD importer."
        ),
        license_note="Creative Commons Attribution-NonCommercial-ShareAlike 3.0.",
        access_note=(
            "License-gated/manual acquisition. Download from the official SDD page, keep the "
            "license with your local copy, then stage a directory containing annotations.txt."
        ),
        required_paths=(
            RequiredPath(
                pattern="**/annotations.txt",
                kind="file",
                description="Original SDD annotation text file.",
            ),
        ),
        related_issues=(1497, 1126),
    ),
    AssetSpec(
        asset_id="socnavbench-s3dis-eth",
        title="SocNavBench/S3DIS ETH traversible assets",
        expected_local_path=REPO_ROOT / "third_party" / "socnavbench",
        source_url="https://github.com/CMU-TBD/SocNavBench",
        source_note=(
            "SocNavBench code and install docs point to the official curated asset package; "
            "S3DIS source data has separate access terms."
        ),
        license_note=(
            "SocNavBench code is vendored separately; S3DIS/SBPD meshes and traversibles are "
            "external licensed data and are not redistributed by Robot SF."
        ),
        access_note=(
            "License-gated/manual acquisition. Follow docs/socnav_assets_setup.md and upstream "
            "SocNavBench/S3DIS instructions, then stage the ETH mesh directory and traversible."
        ),
        required_paths=(
            RequiredPath(
                pattern="sd3dis/stanford_building_parser_dataset/mesh/ETH",
                kind="directory",
                description="Curated SocNavBench ETH mesh directory.",
            ),
            RequiredPath(
                pattern="sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl",
                kind="file",
                description="Precomputed ETH traversible pickle.",
            ),
        ),
        related_issues=(1498, 1134, 334),
    ),
    AssetSpec(
        asset_id="socnavbench-control",
        title="SocNavBench control-pipeline assets",
        expected_local_path=REPO_ROOT / "third_party" / "socnavbench",
        source_url="https://github.com/CMU-TBD/SocNavBench",
        source_note=(
            "SocNavBench wayptnav_data assets used by the control/waypoint navigation pipeline."
        ),
        license_note=(
            "External SocNavBench data assets are not redistributed by Robot SF; verify local "
            "staging before benchmark use."
        ),
        access_note=(
            "License-gated/manual acquisition. Follow docs/socnav_assets_setup.md, then stage "
            "wayptnav_data under third_party/socnavbench."
        ),
        required_paths=(
            RequiredPath(
                pattern="wayptnav_data",
                kind="directory",
                description="Precomputed SocNavBench waypoint/control-pipeline data.",
            ),
        ),
        related_issues=(1456, 562),
    ),
    AssetSpec(
        asset_id="amv-calibration",
        title="AMV calibration source provenance",
        expected_local_path=REPO_ROOT / "output" / "external_data" / "amv_calibration",
        source_url="https://github.com/ll7/robot_sf_ll7/issues/1585",
        source_note=(
            "Local-only accepted source bundle for AMV actuation calibration provenance. "
            "Acceptable source classes are an official platform/controller specification, a "
            "maintainer-accepted platform-class source, or a command-response trace bundle."
        ),
        license_note=(
            "Access and redistribution depend on the selected source. Private traces and vendor "
            "documents must stay local unless explicit redistribution rights are recorded."
        ),
        access_note=(
            "Manual provenance handoff only. Stage the accepted source file or trace bundle after "
            "#1585 identifies it; this helper records checksums and access notes but does not "
            "promote synthetic AMV diagnostics into calibrated evidence."
        ),
        required_paths=(
            RequiredPath(
                pattern="**/*.json",
                kind="file",
                description="Accepted JSON manifest, official spec extract, or trace metadata.",
                group="source",
            ),
            RequiredPath(
                pattern="**/*.yaml",
                kind="file",
                description="Accepted YAML manifest, official spec extract, or trace metadata.",
                group="source",
            ),
            RequiredPath(
                pattern="**/*.yml",
                kind="file",
                description="Accepted YAML manifest, official spec extract, or trace metadata.",
                group="source",
            ),
            RequiredPath(
                pattern="**/*.csv",
                kind="file",
                description="Accepted command-response trace table.",
                group="source",
            ),
            RequiredPath(
                pattern="**/*.pdf",
                kind="file",
                description="Accepted official specification document.",
                group="source",
            ),
        ),
        related_issues=(1585, 1559),
    ),
)


def list_assets() -> tuple[AssetSpec, ...]:
    """Return the supported external asset registry."""
    return ASSETS


def _get_asset(asset_id: str) -> AssetSpec:
    """Resolve an asset id or raise a user-facing error."""
    for asset in ASSETS:
        if asset.asset_id == asset_id:
            return asset
    supported = ", ".join(asset.asset_id for asset in ASSETS)
    raise ExternalDataError(f"Unknown asset id '{asset_id}'. Supported assets: {supported}")


def _match_required_path(source_root: Path, required: RequiredPath) -> list[Path]:
    """Return paths under source_root matching one required asset pattern."""
    matches = sorted(source_root.glob(required.pattern))
    if required.kind == "file":
        return [path for path in matches if path.is_file()]
    if required.kind == "directory":
        return [path for path in matches if path.is_dir()]
    raise ExternalDataError(f"Unsupported required path kind: {required.kind}")


def _required_groups(asset: AssetSpec) -> dict[str, list[RequiredPath]]:
    """Group required paths; entries in the same group are alternatives."""
    groups: dict[str, list[RequiredPath]] = {}
    for required in asset.required_paths:
        groups.setdefault(required.group or required.pattern, []).append(required)
    return groups


def _relative_to_repo(path: Path, repo_root: Path) -> Path | None:
    """Return a repo-relative path when path is inside repo_root."""
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None


def _git_check_ignored(path: Path, repo_root: Path) -> bool:
    """Return whether git ignore rules cover a repo-local path."""
    rel = _relative_to_repo(path, repo_root)
    if rel is None:
        return True
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(rel)],
        cwd=repo_root,
        check=False,
    )
    return result.returncode == 0


def _iter_files(paths: list[Path]) -> list[Path]:
    """Expand matched required paths into a deterministic file list."""
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*") if child.is_file())
    return sorted({path.resolve() for path in files})


def _tree_checksum(source_root: Path, matched_paths: list[Path]) -> dict[str, Any]:
    """Compute an aggregate checksum over matched raw files without persisting a huge hash list."""
    digest = hashlib.sha256()
    sample_files: list[dict[str, Any]] = []
    total_size = 0
    files = _iter_files(matched_paths)
    for file_path in files:
        file_digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        size = file_path.stat().st_size
        total_size += size
        rel = file_path.relative_to(source_root).as_posix()
        file_sha = file_digest.hexdigest()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
        if len(sample_files) < 20:
            sample_files.append({"path": rel, "size_bytes": size, "sha256": file_sha})
    return {
        "file_count": len(files),
        "total_size_bytes": total_size,
        "tree_sha256": digest.hexdigest(),
        "sample_files": sample_files,
    }


def check_asset(asset_id: str, *, source_path: Path | None = None) -> dict[str, Any]:
    """Validate the local path for one external asset."""
    asset = _get_asset(asset_id)
    root = (source_path or asset.expected_local_path).expanduser().resolve()
    report: dict[str, Any] = {
        "asset_id": asset.asset_id,
        "title": asset.title,
        "source_path": str(root),
        "expected_local_path": str(asset.expected_local_path),
        "source_url": asset.source_url,
        "license_note": asset.license_note,
        "access_note": asset.access_note,
        "auto_download_allowed": asset.auto_download_allowed,
        "required_paths": [
            {
                "pattern": required.pattern,
                "kind": required.kind,
                "description": required.description,
                "group": required.group,
            }
            for required in asset.required_paths
        ],
        "ok": False,
        "status": "missing",
        "missing_required_paths": [],
        "matched_required_paths": [],
    }
    if not root.exists():
        report["action"] = (
            "Missing local source path. Follow the official acquisition instructions, then run "
            f"`uv run python scripts/tools/manage_external_data.py stage {asset.asset_id} "
            "--source <path>`."
        )
        return report

    matched_paths: list[Path] = []
    missing: list[str] = []
    for group_name, requirements in _required_groups(asset).items():
        group_matches: list[Path] = []
        for required in requirements:
            group_matches.extend(_match_required_path(root, required))
        if not group_matches:
            if len(requirements) == 1:
                missing.append(requirements[0].pattern)
            else:
                alternatives = ", ".join(required.pattern for required in requirements)
                missing.append(f"{group_name}: one of {alternatives}")
            continue
        matched_paths.extend(group_matches)

    report["missing_required_paths"] = missing
    report["matched_required_paths"] = [
        path.relative_to(root).as_posix() for path in sorted(matched_paths)
    ]
    if missing:
        report["status"] = "incomplete"
        report["action"] = (
            "Required files are missing. Re-check the official acquisition instructions and the "
            "expected local layout before staging."
        )
        return report

    report["ok"] = True
    report["status"] = "available"
    report["action"] = "Path satisfies the declared local staging contract."
    return report


def _matched_paths_for_report(source_root: Path, report: dict[str, Any]) -> list[Path]:
    """Resolve matched required paths from a check report."""
    return [source_root / rel for rel in report["matched_required_paths"]]


def _ensure_repo_local_raw_paths_ignored(
    *,
    matched_paths: list[Path],
    repo_root: Path,
) -> None:
    """Fail closed if any repo-local raw path is not covered by gitignore."""
    unsafe: list[str] = []
    for path in matched_paths:
        if _relative_to_repo(path, repo_root) is None:
            continue
        if not _git_check_ignored(path, repo_root):
            unsafe.append(str(_relative_to_repo(path, repo_root)))
    if unsafe:
        raise ExternalDataError(
            "Raw external data path is not covered by gitignore: "
            f"{', '.join(unsafe)}. Move it under output/ or add an explicit ignore rule first."
        )


def stage_asset(
    asset_id: str,
    *,
    source_path: Path,
    manifest_out: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Validate source_path and write a compact provenance manifest."""
    asset = _get_asset(asset_id)
    source_root = source_path.expanduser().resolve()
    report = check_asset(asset_id, source_path=source_root)
    if not report["ok"]:
        raise ExternalDataError(f"Cannot stage {asset_id}: {report['action']}")

    matched_paths = _matched_paths_for_report(source_root, report)
    _ensure_repo_local_raw_paths_ignored(matched_paths=matched_paths, repo_root=repo_root)
    checksum = _tree_checksum(source_root, matched_paths)
    manifest = {
        "schema": "robot_sf_external_data_manifest.v1",
        "asset_id": asset.asset_id,
        "title": asset.title,
        "source_url": asset.source_url,
        "source_note": asset.source_note,
        "license_note": asset.license_note,
        "access_note": asset.access_note,
        "local_path": str(source_root),
        "required_paths": report["required_paths"],
        "matched_required_paths": report["matched_required_paths"],
        "file_count": checksum["file_count"],
        "total_size_bytes": checksum["total_size_bytes"],
        "tree_sha256": checksum["tree_sha256"],
        "checksum_policy": (
            "aggregate_tree_sha256 over relative path, size, and sha256 for every matched file; "
            "sample file hashes are included for review while raw data remains local."
        ),
        "sample_files": checksum["sample_files"],
        "related_issues": list(asset.related_issues),
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "validation_command": (
            f"uv run python scripts/tools/manage_external_data.py check {asset_id} "
            f"--source {source_root}"
        ),
    }
    output_path = manifest_out or DEFAULT_MANIFEST_DIR / f"{asset.asset_id}.provenance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def download_asset(asset_id: str) -> None:
    """Download an asset only when the registry declares an allowed direct download path."""
    asset = _get_asset(asset_id)
    if not asset.auto_download_allowed:
        raise ExternalDataError(
            f"{asset.asset_id} is license-gated or lacks a repository-approved direct download "
            f"path. {asset.access_note}"
        )
    raise ExternalDataError(f"No downloader is implemented for {asset.asset_id}.")


def _print_json(payload: Any) -> None:
    """Print a stable JSON payload."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _asset_summary(asset: AssetSpec, *, include_status: bool) -> dict[str, Any]:
    """Return one list/explain payload."""
    payload: dict[str, Any] = {
        "asset_id": asset.asset_id,
        "title": asset.title,
        "expected_local_path": str(asset.expected_local_path),
        "source_url": asset.source_url,
        "license_note": asset.license_note,
        "access_note": asset.access_note,
        "auto_download_allowed": asset.auto_download_allowed,
        "related_issues": list(asset.related_issues),
    }
    if include_status:
        status = check_asset(asset.asset_id)
        payload["status"] = status["status"]
        payload["ok"] = status["ok"]
        payload["action"] = status["action"]
        payload["missing_required_paths"] = status["missing_required_paths"]
    return payload


def _print_asset_summary(payload: dict[str, Any]) -> None:
    """Print a concise human-readable asset summary."""
    print(f"{payload['asset_id']}: {payload['title']}")
    if "status" in payload:
        print(f"  status: {payload['status']}")
        print(f"  action: {payload['action']}")
    print(f"  expected path: {payload['expected_local_path']}")
    print(f"  source: {payload['source_url']}")
    print(f"  license/access: {payload['license_note']} {payload['access_note']}")
    print(
        f"  download: {'allowed' if payload['auto_download_allowed'] else 'manual/license-gated'}"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List supported assets and current local status.")

    explain_parser = subparsers.add_parser("explain", help="Explain one asset's source contract.")
    explain_parser.add_argument("asset_id")

    check_parser = subparsers.add_parser("check", help="Check one staged asset path.")
    check_parser.add_argument("asset_id", nargs="?", default="all")
    check_parser.add_argument("--source", type=Path, help="Override local source path.")

    stage_parser = subparsers.add_parser("stage", help="Validate a local path and write manifest.")
    stage_parser.add_argument("asset_id")
    stage_parser.add_argument("--source", type=Path, required=True)
    stage_parser.add_argument("--manifest-out", type=Path)

    download_parser = subparsers.add_parser(
        "download", help="Download only approved direct assets."
    )
    download_parser.add_argument("asset_id")

    return parser.parse_args()


def _emit_asset_entries(payload: list[dict[str, Any]], *, as_json: bool) -> None:
    """Print list/check payload entries."""
    if as_json:
        _print_json(payload)
        return
    for index, entry in enumerate(payload):
        if index:
            print("")
        _print_asset_summary(entry)


def _handle_list(args: argparse.Namespace) -> int:
    """Handle the list subcommand."""
    payload = [_asset_summary(asset, include_status=True) for asset in ASSETS]
    _emit_asset_entries(payload, as_json=args.json)
    return 0


def _handle_explain(args: argparse.Namespace) -> int:
    """Handle the explain subcommand."""
    asset = _get_asset(args.asset_id)
    payload = _asset_summary(asset, include_status=False)
    payload["required_paths"] = [
        {
            "pattern": required.pattern,
            "kind": required.kind,
            "description": required.description,
            "group": required.group,
        }
        for required in asset.required_paths
    ]
    if args.json:
        _print_json(payload)
        return 0
    _print_asset_summary(payload)
    print("  required paths:")
    for required in payload["required_paths"]:
        print(f"  - {required['pattern']} ({required['kind']}): {required['description']}")
    return 0


def _handle_check(args: argparse.Namespace) -> int:
    """Handle the check subcommand."""
    if args.asset_id == "all":
        payload = [check_asset(asset.asset_id) for asset in ASSETS]
        _emit_asset_entries(payload, as_json=args.json)
        return 0 if all(entry["ok"] for entry in payload) else 2
    payload = check_asset(args.asset_id, source_path=args.source)
    if args.json:
        _print_json(payload)
    else:
        _print_asset_summary(payload)
    return 0 if payload["ok"] else 2


def _handle_stage(args: argparse.Namespace) -> int:
    """Handle the stage subcommand."""
    payload = stage_asset(
        args.asset_id,
        source_path=args.source,
        manifest_out=args.manifest_out,
    )
    if args.json:
        _print_json(payload)
        return 0
    output_path = args.manifest_out or DEFAULT_MANIFEST_DIR / (
        payload["asset_id"] + ".provenance.json"
    )
    print(f"wrote manifest for {payload['asset_id']}: {output_path}")
    return 0


def _handle_download(args: argparse.Namespace) -> int:
    """Handle the download subcommand."""
    download_asset(args.asset_id)
    return 0


def _handle_error(exc: ExternalDataError, *, as_json: bool) -> int:
    """Print one user-facing CLI error."""
    if as_json:
        _print_json({"ok": False, "error": str(exc)})
    else:
        print(f"error: {exc}")
    return 2


def main() -> int:
    """Run the external-data assistant."""
    args = parse_args()
    handlers = {
        "list": _handle_list,
        "explain": _handle_explain,
        "check": _handle_check,
        "stage": _handle_stage,
        "download": _handle_download,
    }
    try:
        return handlers[args.command](args)
    except ExternalDataError as exc:
        return _handle_error(exc, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
