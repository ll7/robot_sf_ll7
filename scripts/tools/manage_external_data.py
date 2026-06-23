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
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "output" / "external_data" / "manifests"

# Canonical SDD staging manifest. This is the single editable provenance contract for the SDD
# asset (issues #2657, #3473): maintainers pin `expected_tree_sha256` and tune the disk-check size
# here, while all behavior lives in this module (the former scripts/data wrapper just delegates).
DEFAULT_SDD_STAGING_MANIFEST = REPO_ROOT / "configs" / "data" / "sdd_staging_manifest.yaml"
DEFAULT_STAGING_ROOT = (REPO_ROOT / "output").resolve()
SDD_STAGING_STATUS_FILENAME = "sdd_staging_status.json"

# Scenario-prior mode tags surfaced to scenario-prior generation. These names are part of the
# public gate contract consumed by scripts/analysis/calibrate_scenario_priors_from_traces_*.
SDD_MODE_PROXY = "proxy_schema_smoke"
SDD_MODE_DATASET_BACKED = "dataset_backed_prior"


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
    # Optional pointer to a checksum/availability staging manifest. When set, this asset carries a
    # pinned-checksum policy and proxy-vs-dataset-backed availability states (see the SDD asset).
    staging_manifest_path: Path | None = None
    # Availability/proxy-vs-dataset-backed gate mode tags for assets that gate downstream evidence.
    proxy_mode: str | None = None
    dataset_backed_mode: str | None = None


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
        related_issues=(2657, 1497, 1126, 3161, 3473),
        # Single source of truth for the SDD checksum/availability policy (issues #2657, #3473):
        # the staging manifest pins `expected_tree_sha256` and the disk-check size, while the
        # availability states + scenario-prior gate live in this module.
        staging_manifest_path=DEFAULT_SDD_STAGING_MANIFEST,
        proxy_mode=SDD_MODE_PROXY,
        dataset_backed_mode=SDD_MODE_DATASET_BACKED,
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
        return [
            path
            for path in matches
            if path.is_dir() and any(child.is_file() for child in path.rglob("*"))
        ]
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


# --------------------------------------------------------------------------------------------
# Canonical SDD staging subsystem (consolidated from scripts/data/stage_sdd_dataset_issue_2657.py
# for issue #3473). This module is the single source of truth for the SDD checksum policy, the
# proxy-vs-dataset-backed availability gate, and the no-auto-download safety contract from #2657.
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SddExpectedFile:
    """One expected-file pattern declared by the SDD staging manifest."""

    pattern: str
    kind: str
    description: str
    min_count: int = 1


@dataclass(frozen=True)
class SddStagingSpec:
    """Parsed SDD staging manifest contract (checksum policy + availability gate)."""

    asset_id: str
    title: str
    version_tag: str
    source_url: str
    license: str
    license_url: str
    readme_pointer: str
    staging_dir: Path
    download_url: str | None
    access_note: str
    expected_files: tuple[SddExpectedFile, ...]
    expected_total_size_bytes: int
    checksum_algorithm: str
    expected_tree_sha256: str | None
    local_availability_declared: str

    @property
    def expected_total_size_mib(self) -> float:
        """Expected footprint in MiB for human-readable reporting."""
        return self.expected_total_size_bytes / (1024 * 1024)


def _sdd_manifest_path(spec: AssetSpec | None = None) -> Path:
    """Resolve the canonical SDD staging manifest path from the asset registry."""
    asset = spec or _get_asset("sdd")
    if asset.staging_manifest_path is None:
        raise ExternalDataError(f"Asset '{asset.asset_id}' declares no staging manifest path.")
    return asset.staging_manifest_path


def _resolve_sdd_staging_dir(staging_dir_raw: str, *, manifest_path: Path) -> Path:
    """Resolve a manifest staging dir while rejecting unsafe path escapes."""
    staging_dir = Path(staging_dir_raw).expanduser()
    if ".." in staging_dir.parts:
        raise ExternalDataError("Manifest staging_dir must not contain path traversal (`..`).")

    unresolved = staging_dir if staging_dir.is_absolute() else REPO_ROOT / staging_dir
    if unresolved.is_symlink():
        raise ExternalDataError(f"Manifest staging_dir must not be a symlink: {unresolved}")

    resolved = unresolved.resolve(strict=False)
    allowed_roots = (DEFAULT_STAGING_ROOT, manifest_path.resolve(strict=False).parent)
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        allowed = ", ".join(str(root) for root in allowed_roots)
        raise ExternalDataError(
            f"Manifest staging_dir resolves outside allowed roots: {resolved}. "
            f"Allowed roots: {allowed}."
        )
    return resolved


def _parse_sdd_expected_files(expected_files_raw: list[Any]) -> tuple[SddExpectedFile, ...]:
    """Parse manifest expected-file entries with fail-closed error messages."""
    expected_files: list[SddExpectedFile] = []
    for index, entry in enumerate(expected_files_raw):
        if not isinstance(entry, dict):
            raise ExternalDataError(f"`expected_files[{index}]` must be a mapping.")
        pattern_value = entry.get("pattern", "")
        pattern = "" if pattern_value is None else str(pattern_value).strip()
        if not pattern:
            raise ExternalDataError(f"`expected_files[{index}].pattern` is required.")
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise ExternalDataError(
                f"`expected_files[{index}].pattern` must stay within staging_dir."
            )
        kind_value = entry.get("kind", "file")
        kind = "file" if kind_value is None else str(kind_value).strip()
        if kind not in {"file", "directory"}:
            raise ExternalDataError(
                f"`expected_files[{index}].kind` must be one of: file, directory."
            )
        try:
            min_count = int(entry.get("min_count", 1))
        except (TypeError, ValueError) as exc:
            raise ExternalDataError(
                f"`expected_files[{index}].min_count` must be an integer."
            ) from exc
        if min_count <= 0:
            raise ExternalDataError(f"`expected_files[{index}].min_count` must be positive.")
        expected_files.append(
            SddExpectedFile(
                pattern=pattern,
                kind=kind,
                description=str(entry.get("description") or ""),
                min_count=min_count,
            )
        )
    return tuple(expected_files)


def _manifest_text(raw: dict[str, Any], key: str, default: str = "") -> str:
    """Return a stripped manifest string, treating explicit YAML null as missing."""
    value = raw.get(key, default)
    if value is None:
        value = default
    return str(value).strip()


def _required_manifest_text(raw: dict[str, Any], key: str) -> str:
    """Return a required manifest string or fail closed when it is absent/empty/null."""
    value = _manifest_text(raw, key)
    if not value:
        raise ExternalDataError(f"Manifest is missing required `{key}`.")
    return value


def _optional_checksum_text(checksums: dict[str, Any], key: str) -> str | None:
    """Return an optional checksum string, preserving null/empty as unpinned."""
    value = checksums.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_sdd_staging_spec(manifest_path: Path | None = None) -> SddStagingSpec:
    """Load and validate the canonical SDD staging manifest."""
    path = manifest_path or _sdd_manifest_path()
    if not path.is_file():
        raise ExternalDataError(f"Staging manifest not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ExternalDataError(f"Manifest is not a mapping: {path}")

    staging_dir_raw = _required_manifest_text(raw, "staging_dir")
    staging_dir = _resolve_sdd_staging_dir(staging_dir_raw, manifest_path=path)

    expected_files_raw = raw.get("expected_files") or []
    if not isinstance(expected_files_raw, list) or not expected_files_raw:
        raise ExternalDataError("Manifest must declare a non-empty `expected_files` list.")
    expected_files = _parse_sdd_expected_files(expected_files_raw)

    try:
        size_bytes = int(raw.get("expected_total_size_bytes", 0))
    except (TypeError, ValueError) as exc:
        raise ExternalDataError("Manifest `expected_total_size_bytes` must be an integer.") from exc
    if size_bytes <= 0:
        raise ExternalDataError("Manifest must declare a positive `expected_total_size_bytes`.")

    checksums_raw = raw.get("checksums")
    if checksums_raw is None:
        checksums: dict[str, Any] = {}
    elif isinstance(checksums_raw, dict):
        checksums = checksums_raw
    else:
        raise ExternalDataError("Manifest `checksums` must be a mapping when provided.")
    download_url = _manifest_text(raw, "download_url")

    return SddStagingSpec(
        asset_id=_manifest_text(raw, "asset_id", "sdd"),
        title=_manifest_text(raw, "title", "Stanford Drone Dataset"),
        version_tag=_manifest_text(raw, "version_tag", "unknown"),
        source_url=_manifest_text(raw, "source_url"),
        license=_manifest_text(raw, "license"),
        license_url=_manifest_text(raw, "license_url"),
        readme_pointer=_manifest_text(raw, "readme_pointer"),
        staging_dir=staging_dir,
        download_url=download_url or None,
        access_note=_manifest_text(raw, "access_note"),
        expected_files=expected_files,
        expected_total_size_bytes=size_bytes,
        checksum_algorithm=_manifest_text(checksums, "algorithm", "SHA-256"),
        expected_tree_sha256=_optional_checksum_text(checksums, "expected_tree_sha256"),
        local_availability_declared=_manifest_text(raw, "local_availability", "missing"),
    )


def _match_sdd_expected(staging_dir: Path, expected: SddExpectedFile) -> list[Path]:
    """Return files under staging_dir matching one expected-file pattern."""
    if not staging_dir.exists():
        return []
    matches = sorted(staging_dir.glob(expected.pattern))
    if expected.kind == "file":
        return [path for path in matches if path.is_file()]
    if expected.kind == "directory":
        return [path for path in matches if path.is_dir()]
    raise ExternalDataError(f"Unsupported expected-file kind: {expected.kind}")


def validate_sdd_staging(spec: SddStagingSpec, *, compute_checksum: bool = True) -> dict[str, Any]:
    """Validate the locally-staged SDD copy against the manifest contract.

    ``ok`` is True only when every expected-file group meets its minimum count AND the staged tree
    matches the pinned ``expected_tree_sha256``; ``local_availability`` is ``staged`` only then.
    """
    staging_dir = spec.staging_dir
    report: dict[str, Any] = {
        "asset_id": spec.asset_id,
        "version_tag": spec.version_tag,
        "staging_dir": str(staging_dir),
        "staging_dir_exists": staging_dir.exists(),
        "expected_files": [
            {
                "pattern": expected.pattern,
                "kind": expected.kind,
                "description": expected.description,
                "min_count": expected.min_count,
            }
            for expected in spec.expected_files
        ],
        "matched_files": [],
        "missing_expected": [],
        "ok": False,
        "local_availability": "missing",
        "mode": SDD_MODE_PROXY,
    }

    matched_paths: list[Path] = []
    for expected in spec.expected_files:
        group_matches = _match_sdd_expected(staging_dir, expected)
        if len(group_matches) < expected.min_count:
            report["missing_expected"].append(
                f"{expected.pattern} (need >= {expected.min_count}, found {len(group_matches)})"
            )
            continue
        matched_paths.extend(group_matches)

    if matched_paths and staging_dir.exists():
        report["matched_files"] = [
            path.relative_to(staging_dir).as_posix() for path in sorted(matched_paths)
        ]

    if report["missing_expected"]:
        report["action"] = (
            "SDD is not staged or is incomplete. Follow the manifest access_note to acquire SDD "
            "under its license, then re-run validation. Until then, scenario-prior generation is "
            f"forced to `{SDD_MODE_PROXY}`."
        )
        return report

    if not compute_checksum:
        report["checksum_skipped"] = (
            "tree checksum omitted for plan/status speed; run `sdd-validate` for checksum proof"
        )
        report["local_availability"] = "present_unvalidated"
        report["action"] = (
            "SDD expected files are present. Run `sdd-validate` to compute checksums before "
            f"treating the copy as `{SDD_MODE_DATASET_BACKED}` evidence."
        )
        return report

    checksum = _tree_checksum(staging_dir, matched_paths)
    report["file_count"] = checksum["file_count"]
    report["total_size_bytes"] = checksum["total_size_bytes"]
    report["tree_sha256"] = checksum["tree_sha256"]
    report["sample_files"] = checksum["sample_files"]
    report["checksum_algorithm"] = spec.checksum_algorithm

    if spec.expected_tree_sha256 is None:
        report["local_availability"] = "present_unpinned_checksum"
        report["action"] = (
            "Expected SDD files are present and a tree checksum was computed, but the manifest does "
            "not pin `expected_tree_sha256`. Refusing to mark SDD as dataset-backed until a trusted "
            f"checksum is pinned. Scenario-prior generation remains forced to `{SDD_MODE_PROXY}`."
        )
        return report

    report["expected_tree_sha256"] = spec.expected_tree_sha256
    report["checksum_match"] = checksum["tree_sha256"] == spec.expected_tree_sha256
    if not report["checksum_match"]:
        report["action"] = (
            "Staged files do not match the pinned `expected_tree_sha256` in the manifest. "
            "Refusing to mark SDD as staged; treat as untrusted and re-acquire. Scenario-prior "
            f"generation remains forced to `{SDD_MODE_PROXY}`."
        )
        return report

    report["ok"] = True
    report["local_availability"] = "staged"
    report["mode"] = SDD_MODE_DATASET_BACKED
    report["action"] = (
        f"SDD is staged and validated. Scenario-prior generation may run in "
        f"`{SDD_MODE_DATASET_BACKED}` mode for dataset-backed input."
    )
    return report


def _current_sdd_tree_matches_cached_status(
    spec: SddStagingSpec,
    status: dict[str, Any],
    matched_paths: list[Path],
) -> bool:
    """Return True only when cached status still matches the current staged tree."""
    cached_tree_sha256 = status.get("tree_sha256")
    if not isinstance(cached_tree_sha256, str) or not cached_tree_sha256:
        return False
    if spec.expected_tree_sha256 is None or cached_tree_sha256 != spec.expected_tree_sha256:
        return False
    current_checksum = _tree_checksum(spec.staging_dir, matched_paths)
    return current_checksum["tree_sha256"] == cached_tree_sha256


def _cached_sdd_staging_gate(spec: SddStagingSpec) -> dict[str, Any] | None:
    """Return a dataset-backed gate from cached status when it is still structurally valid."""
    status_path = spec.staging_dir / SDD_STAGING_STATUS_FILENAME
    if not status_path.is_file():
        return None
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(status, dict):
        return None
    if status.get("local_availability") != "staged":
        return None
    if status.get("asset_id") != spec.asset_id or status.get("version_tag") != spec.version_tag:
        return None

    matched_paths: list[Path] = []
    for expected in spec.expected_files:
        group_matches = _match_sdd_expected(spec.staging_dir, expected)
        if len(group_matches) < expected.min_count:
            return None
        matched_paths.extend(group_matches)

    if not _current_sdd_tree_matches_cached_status(spec, status, matched_paths):
        return None

    return {
        "mode": SDD_MODE_DATASET_BACKED,
        "dataset_backed": True,
        "reason": f"SDD is staged and validated (cached via {SDD_STAGING_STATUS_FILENAME}).",
        "asset_id": spec.asset_id,
        "version_tag": spec.version_tag,
        "tree_sha256": status["tree_sha256"],
        "staging_dir": str(spec.staging_dir),
        "report": status,
    }


def resolve_sdd_scenario_prior_mode(
    spec: SddStagingSpec | None = None,
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve the scenario-prior generation mode from canonical SDD staging state.

    This is the gate scenario-prior generation consumes. A missing or unvalidated SDD copy forces
    ``proxy_schema_smoke``; only a staged-and-validated copy unlocks ``dataset_backed_prior``.
    """
    spec = spec or load_sdd_staging_spec(manifest_path)
    if cached_gate := _cached_sdd_staging_gate(spec):
        return cached_gate

    report = validate_sdd_staging(spec)
    dataset_backed = bool(report["ok"])
    return {
        "mode": SDD_MODE_DATASET_BACKED if dataset_backed else SDD_MODE_PROXY,
        "dataset_backed": dataset_backed,
        "reason": report["action"],
        "asset_id": spec.asset_id,
        "version_tag": spec.version_tag,
        "tree_sha256": report.get("tree_sha256"),
        "staging_dir": str(spec.staging_dir),
        "report": report,
    }


def check_sdd_disk_space(spec: SddStagingSpec) -> dict[str, Any]:
    """Check free disk space at the staging location against the expected dataset size."""
    probe = spec.staging_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    required = spec.expected_total_size_bytes
    sufficient = usage.free >= required
    return {
        "probe_path": str(probe),
        "required_bytes": required,
        "required_mib": round(required / (1024 * 1024), 1),
        "available_bytes": usage.free,
        "available_mib": round(usage.free / (1024 * 1024), 1),
        "sufficient": sufficient,
    }


def build_sdd_plan(spec: SddStagingSpec) -> dict[str, Any]:
    """Build the plan/report payload for a default (non-downloading) invocation."""
    validation = validate_sdd_staging(spec, compute_checksum=False)
    disk = check_sdd_disk_space(spec)
    return {
        "schema": "sdd_staging_plan.v1",
        "asset_id": spec.asset_id,
        "title": spec.title,
        "version_tag": spec.version_tag,
        "source_url": spec.source_url,
        "license": spec.license,
        "license_url": spec.license_url,
        "readme_pointer": spec.readme_pointer,
        "staging_dir": str(spec.staging_dir),
        "download_url": spec.download_url,
        "access_note": spec.access_note,
        "expected_total_size_bytes": spec.expected_total_size_bytes,
        "expected_total_size_mib": round(spec.expected_total_size_mib, 1),
        "disk_check": disk,
        "would_download": False,
        "auto_download": False,
        "local_availability": validation["local_availability"],
        "scenario_prior_mode": validation["mode"],
        "validation": validation,
        "next_step": (
            "This is a PLAN ONLY; nothing was downloaded. To stage SDD you must (1) obtain it under "
            "its license per access_note, (2) configure a download URL or place files under the "
            "staging_dir, and (3) re-run with `sdd-download --confirm-download` to explicitly "
            "confirm."
        ),
    }


def write_sdd_staging_status(spec: SddStagingSpec, report: dict[str, Any]) -> Path:
    """Write a staging-status manifest into the staging dir after validation."""
    spec.staging_dir.mkdir(parents=True, exist_ok=True)
    status_path = spec.staging_dir / SDD_STAGING_STATUS_FILENAME
    payload = {
        "schema": "sdd_staging_status.v1",
        "asset_id": spec.asset_id,
        "version_tag": spec.version_tag,
        "source_url": spec.source_url,
        "license": spec.license,
        "checksum_algorithm": spec.checksum_algorithm,
        "tree_sha256": report.get("tree_sha256"),
        "file_count": report.get("file_count"),
        "total_size_bytes": report.get("total_size_bytes"),
        "sample_files": report.get("sample_files", []),
        "local_availability": report["local_availability"],
        "validated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return status_path


def confirm_sdd_download(*, confirm_download: bool, yes: bool) -> bool:
    """Return True only when the user has explicitly confirmed a network download.

    Requires the ``--confirm-download`` flag AND an interactive y/N (or ``--yes`` for
    non-interactive confirmation). Default behavior never confirms.
    """
    if not confirm_download:
        return False
    if yes:
        return True
    try:
        answer = input(
            "About to download the Stanford Drone Dataset to a git-ignored folder. "
            "This consumes disk and network. Proceed? [y/N]: "
        )
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def run_sdd_download(spec: SddStagingSpec, *, confirm_download: bool, yes: bool) -> dict[str, Any]:
    """Run the guarded SDD download path. Fails closed at every safety gate."""
    if not confirm_sdd_download(confirm_download=confirm_download, yes=yes):
        raise ExternalDataError(
            "Download not confirmed. This tool never auto-downloads: pass `--confirm-download` and "
            "answer the y/N prompt (or add `--yes` for non-interactive confirmation). A default run "
            "only plans and reports."
        )

    disk = check_sdd_disk_space(spec)
    if not disk["sufficient"]:
        raise ExternalDataError(
            "Insufficient disk space at "
            f"{disk['probe_path']}: required {disk['required_mib']} MiB, "
            f"available {disk['available_mib']} MiB. Refusing to download (fail closed)."
        )

    if not spec.download_url:
        raise ExternalDataError(
            "No download URL is configured for SDD. SDD is license-gated and the repository encodes "
            "no approved direct download URL. Obtain it from the official project page "
            f"({spec.source_url}) under its license, place the annotation files under "
            f"{spec.staging_dir}, then run validation. {spec.access_note}"
        )

    # A real network fetch would happen here using spec.download_url. It is intentionally NOT
    # implemented because SDD is license-gated and no approved URL is encoded; reaching this point
    # requires a maintainer to add a URL to the manifest first.
    raise ExternalDataError(
        "Confirmed and disk-checked, but no downloader is wired for the configured URL. A "
        "maintainer must implement the fetch for the license-approved URL before this path runs."
    )


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

    # Canonical SDD staging subsystem (issues #2657, #3473). These commands own the no-auto-download
    # safety contract and the proxy-vs-dataset-backed scenario-prior gate.
    sdd_plan = subparsers.add_parser(
        "sdd-plan", help="Plan/report only for SDD staging (default; never downloads)."
    )
    sdd_plan.add_argument("--manifest", type=Path, default=None)

    sdd_status = subparsers.add_parser(
        "sdd-status", help="Report SDD local availability/checksums WITHOUT downloading."
    )
    sdd_status.add_argument("--manifest", type=Path, default=None)

    sdd_validate = subparsers.add_parser(
        "sdd-validate", help="Validate the locally-staged SDD copy and write its status file."
    )
    sdd_validate.add_argument("--manifest", type=Path, default=None)

    sdd_mode = subparsers.add_parser(
        "sdd-mode", help="Print the SDD scenario-prior mode gate (proxy vs dataset-backed)."
    )
    sdd_mode.add_argument("--manifest", type=Path, default=None)

    sdd_download = subparsers.add_parser(
        "sdd-download", help="Guarded SDD staging: requires explicit confirmation; fails closed."
    )
    sdd_download.add_argument("--manifest", type=Path, default=None)
    sdd_download.add_argument("--confirm-download", action="store_true")
    sdd_download.add_argument("--yes", action="store_true")

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


def _handle_sdd_plan(args: argparse.Namespace) -> int:
    """Handle the sdd-plan subcommand."""
    spec = load_sdd_staging_spec(args.manifest)
    _print_json(build_sdd_plan(spec))
    return 0


def _handle_sdd_status(args: argparse.Namespace) -> int:
    """Handle the sdd-status subcommand."""
    spec = load_sdd_staging_spec(args.manifest)
    report = validate_sdd_staging(spec)
    _print_json(report)
    return 0 if report["ok"] else 2


def _handle_sdd_validate(args: argparse.Namespace) -> int:
    """Handle the sdd-validate subcommand."""
    spec = load_sdd_staging_spec(args.manifest)
    report = validate_sdd_staging(spec)
    if report["ok"]:
        write_sdd_staging_status(spec, report)
    _print_json(report)
    return 0 if report["ok"] else 2


def _handle_sdd_mode(args: argparse.Namespace) -> int:
    """Handle the sdd-mode subcommand."""
    spec = load_sdd_staging_spec(args.manifest)
    gate = resolve_sdd_scenario_prior_mode(spec)
    _print_json(gate)
    return 0 if gate["dataset_backed"] else 2


def _handle_sdd_download(args: argparse.Namespace) -> int:
    """Handle the sdd-download subcommand."""
    spec = load_sdd_staging_spec(args.manifest)
    report = run_sdd_download(spec, confirm_download=args.confirm_download, yes=args.yes)
    _print_json(report)
    return 0 if report.get("ok") else 2


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
        "sdd-plan": _handle_sdd_plan,
        "sdd-status": _handle_sdd_status,
        "sdd-validate": _handle_sdd_validate,
        "sdd-mode": _handle_sdd_mode,
        "sdd-download": _handle_sdd_download,
    }
    try:
        return handlers[args.command](args)
    except ExternalDataError as exc:
        return _handle_error(exc, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
