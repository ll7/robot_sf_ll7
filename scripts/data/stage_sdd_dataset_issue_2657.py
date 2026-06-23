#!/usr/bin/env python3
"""Reproducible Stanford Drone Dataset (SDD) staging for Issue #2657.

This tool makes local SDD staging reproducible and connects staging state to scenario-prior
generation, so dataset-backed claims cannot be implied without dataset-backed input.

SAFETY CONTRACT (all enforced here):

* **Never auto-download.** A default invocation only PLANS/REPORTS (what would be downloaded,
  where, expected size, disk check) and exits WITHOUT touching the network.
* A network fetch requires an explicit ``--confirm-download`` flag AND an interactive ``y/N``
  prompt. The prompt is skipped only when ``--yes`` is also given for non-interactive use.
* Available disk space is checked against the manifest's expected size BEFORE any fetch; the tool
  fails closed (refuses) if free space is insufficient, reporting required vs available.
* After a download completes the tool validates the expected files/structure and records an
  aggregate SHA-256 tree checksum, reporting success or failure clearly.
* Raw data is staged into a **git-ignored** subfolder (``output/external_data/sdd`` by default).
* ``--check`` reports local availability (``staged`` vs ``missing``) and checksums WITHOUT
  downloading.

The manifest at ``configs/data/sdd_staging_manifest.yaml`` is the provenance contract: source URL,
license/readme pointer, version tag, expected files, expected size, checksum placeholders, and the
``local_availability`` gate.

Usage::

    # Plan only -- NEVER downloads (default):
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py
    # Availability status (no download):
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py --check
    # Validate a locally-staged copy (no download):
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py validate
    # Real staging requires explicit confirmation AND a configured download URL:
    uv run python scripts/data/stage_sdd_dataset_issue_2657.py download --confirm-download
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# parent 1 is scripts/data, parent 2 is scripts, parent 3 is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "configs" / "data" / "sdd_staging_manifest.yaml"
STAGING_STATUS_FILENAME = "sdd_staging_status.json"

# Mode tags surfaced to scenario-prior generation. These names are part of the public contract.
MODE_PROXY = "proxy_schema_smoke"
MODE_DATASET_BACKED = "dataset_backed_prior"


class SddStagingError(RuntimeError):
    """Raised when SDD staging cannot be planned, validated, or completed safely."""


@dataclass(frozen=True)
class ExpectedFile:
    """One expected-file pattern declared by the staging manifest."""

    pattern: str
    kind: str
    description: str
    min_count: int = 1


@dataclass(frozen=True)
class SddManifest:
    """Parsed SDD staging manifest contract."""

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
    expected_files: tuple[ExpectedFile, ...]
    expected_total_size_bytes: int
    checksum_algorithm: str
    expected_tree_sha256: str | None
    local_availability_declared: str

    @property
    def expected_total_size_mib(self) -> float:
        """Expected footprint in MiB for human-readable reporting."""
        return self.expected_total_size_bytes / (1024 * 1024)


def load_manifest(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> SddManifest:
    """Load and validate the SDD staging manifest."""
    if not manifest_path.is_file():
        raise SddStagingError(f"Staging manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SddStagingError(f"Manifest is not a mapping: {manifest_path}")

    staging_dir_raw = str(raw.get("staging_dir", "")).strip()
    if not staging_dir_raw:
        raise SddStagingError("Manifest is missing required `staging_dir`.")
    staging_dir = Path(staging_dir_raw)
    if not staging_dir.is_absolute():
        staging_dir = (REPO_ROOT / staging_dir).resolve()

    expected_files_raw = raw.get("expected_files") or []
    if not isinstance(expected_files_raw, list) or not expected_files_raw:
        raise SddStagingError("Manifest must declare a non-empty `expected_files` list.")
    expected_files = tuple(
        ExpectedFile(
            pattern=str(entry["pattern"]),
            kind=str(entry.get("kind", "file")),
            description=str(entry.get("description", "")),
            min_count=int(entry.get("min_count", 1)),
        )
        for entry in expected_files_raw
    )

    size_bytes = int(raw.get("expected_total_size_bytes", 0))
    if size_bytes <= 0:
        raise SddStagingError("Manifest must declare a positive `expected_total_size_bytes`.")

    checksums = raw.get("checksums") or {}
    download_url = raw.get("download_url")

    return SddManifest(
        asset_id=str(raw.get("asset_id", "sdd")),
        title=str(raw.get("title", "Stanford Drone Dataset")),
        version_tag=str(raw.get("version_tag", "unknown")),
        source_url=str(raw.get("source_url", "")),
        license=str(raw.get("license", "")),
        license_url=str(raw.get("license_url", "")),
        readme_pointer=str(raw.get("readme_pointer", "")),
        staging_dir=staging_dir,
        download_url=str(download_url) if download_url else None,
        access_note=str(raw.get("access_note", "")),
        expected_files=expected_files,
        expected_total_size_bytes=size_bytes,
        checksum_algorithm=str(checksums.get("algorithm", "SHA-256")),
        expected_tree_sha256=(
            str(checksums["expected_tree_sha256"])
            if checksums.get("expected_tree_sha256")
            else None
        ),
        local_availability_declared=str(raw.get("local_availability", "missing")),
    )


def _match_expected(staging_dir: Path, expected: ExpectedFile) -> list[Path]:
    """Return files under staging_dir matching one expected-file pattern."""
    if not staging_dir.exists():
        return []
    matches = sorted(staging_dir.glob(expected.pattern))
    if expected.kind == "file":
        return [path for path in matches if path.is_file()]
    if expected.kind == "directory":
        return [path for path in matches if path.is_dir()]
    raise SddStagingError(f"Unsupported expected-file kind: {expected.kind}")


def _iter_files(paths: list[Path]) -> list[Path]:
    """Expand matched paths into a deterministic, de-duplicated file list."""
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(child for child in path.rglob("*") if child.is_file())
    return sorted({path.resolve() for path in files})


def _tree_checksum(staging_dir: Path, matched_paths: list[Path]) -> dict[str, Any]:
    """Compute an aggregate SHA-256 tree checksum over matched files."""
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
        rel = file_path.relative_to(staging_dir).as_posix()
        file_sha = file_digest.hexdigest()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
        if len(sample_files) < 10:
            sample_files.append({"path": rel, "size_bytes": size, "sha256": file_sha})
    return {
        "file_count": len(files),
        "total_size_bytes": total_size,
        "tree_sha256": digest.hexdigest() if files else None,
        "sample_files": sample_files,
    }


def validate_staging(manifest: SddManifest) -> dict[str, Any]:
    """Validate the locally-staged SDD copy against the manifest contract.

    Returns a report dict. ``ok`` is True only when every expected-file group meets its minimum
    count; ``local_availability`` is ``staged`` only when ``ok`` is True.
    """
    staging_dir = manifest.staging_dir
    report: dict[str, Any] = {
        "asset_id": manifest.asset_id,
        "version_tag": manifest.version_tag,
        "staging_dir": str(staging_dir),
        "staging_dir_exists": staging_dir.exists(),
        "expected_files": [
            {
                "pattern": expected.pattern,
                "kind": expected.kind,
                "description": expected.description,
                "min_count": expected.min_count,
            }
            for expected in manifest.expected_files
        ],
        "matched_files": [],
        "missing_expected": [],
        "ok": False,
        "local_availability": "missing",
        "mode": MODE_PROXY,
    }

    matched_paths: list[Path] = []
    for expected in manifest.expected_files:
        group_matches = _match_expected(staging_dir, expected)
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
            f"forced to `{MODE_PROXY}`."
        )
        return report

    checksum = _tree_checksum(staging_dir, matched_paths)
    report["file_count"] = checksum["file_count"]
    report["total_size_bytes"] = checksum["total_size_bytes"]
    report["tree_sha256"] = checksum["tree_sha256"]
    report["sample_files"] = checksum["sample_files"]
    report["checksum_algorithm"] = manifest.checksum_algorithm

    if manifest.expected_tree_sha256 is not None:
        report["expected_tree_sha256"] = manifest.expected_tree_sha256
        report["checksum_match"] = checksum["tree_sha256"] == manifest.expected_tree_sha256
        if not report["checksum_match"]:
            report["action"] = (
                "Staged files do not match the pinned `expected_tree_sha256` in the manifest. "
                "Refusing to mark SDD as staged; treat as untrusted and re-acquire. Scenario-prior "
                f"generation remains forced to `{MODE_PROXY}`."
            )
            return report

    report["ok"] = True
    report["local_availability"] = "staged"
    report["mode"] = MODE_DATASET_BACKED
    report["action"] = (
        f"SDD is staged and validated. Scenario-prior generation may run in `{MODE_DATASET_BACKED}` "
        "mode for dataset-backed input."
    )
    return report


def resolve_scenario_prior_mode(
    manifest: SddManifest | None = None,
    *,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Resolve the scenario-prior generation mode from SDD staging state.

    This is the gate scenario-prior generation imports. A missing or unvalidated SDD copy forces
    ``proxy_schema_smoke``; only a staged-and-validated copy unlocks ``dataset_backed_prior``.

    Returns a compact dict: ``mode``, ``dataset_backed`` (bool), ``reason``, and the validation
    ``report`` for provenance.
    """
    manifest = manifest or load_manifest(manifest_path)
    report = validate_staging(manifest)
    dataset_backed = bool(report["ok"])
    return {
        "mode": MODE_DATASET_BACKED if dataset_backed else MODE_PROXY,
        "dataset_backed": dataset_backed,
        "reason": report["action"],
        "asset_id": manifest.asset_id,
        "version_tag": manifest.version_tag,
        "tree_sha256": report.get("tree_sha256"),
        "staging_dir": str(manifest.staging_dir),
        "report": report,
    }


def check_disk_space(manifest: SddManifest) -> dict[str, Any]:
    """Check free disk space at the staging location against the expected dataset size."""
    # Walk up to the first existing ancestor so disk_usage works before the dir is created.
    probe = manifest.staging_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    required = manifest.expected_total_size_bytes
    sufficient = usage.free >= required
    return {
        "probe_path": str(probe),
        "required_bytes": required,
        "required_mib": round(required / (1024 * 1024), 1),
        "available_bytes": usage.free,
        "available_mib": round(usage.free / (1024 * 1024), 1),
        "sufficient": sufficient,
    }


def build_plan(manifest: SddManifest) -> dict[str, Any]:
    """Build the plan/report payload for a default (non-downloading) invocation."""
    validation = validate_staging(manifest)
    disk = check_disk_space(manifest)
    return {
        "schema": "sdd_staging_plan.v1",
        "asset_id": manifest.asset_id,
        "title": manifest.title,
        "version_tag": manifest.version_tag,
        "source_url": manifest.source_url,
        "license": manifest.license,
        "license_url": manifest.license_url,
        "readme_pointer": manifest.readme_pointer,
        "staging_dir": str(manifest.staging_dir),
        "download_url": manifest.download_url,
        "access_note": manifest.access_note,
        "expected_total_size_bytes": manifest.expected_total_size_bytes,
        "expected_total_size_mib": round(manifest.expected_total_size_mib, 1),
        "disk_check": disk,
        "would_download": False,
        "auto_download": False,
        "local_availability": validation["local_availability"],
        "scenario_prior_mode": validation["mode"],
        "validation": validation,
        "next_step": (
            "This is a PLAN ONLY; nothing was downloaded. To stage SDD you must (1) obtain it under "
            "its license per access_note, (2) configure a download URL or place files under the "
            "staging_dir, and (3) re-run with `download --confirm-download` to explicitly confirm."
        ),
    }


def _write_staging_status(manifest: SddManifest, report: dict[str, Any]) -> Path:
    """Write a staging-status manifest into the staging dir after validation."""
    manifest.staging_dir.mkdir(parents=True, exist_ok=True)
    status_path = manifest.staging_dir / STAGING_STATUS_FILENAME
    payload = {
        "schema": "sdd_staging_status.v1",
        "asset_id": manifest.asset_id,
        "version_tag": manifest.version_tag,
        "source_url": manifest.source_url,
        "license": manifest.license,
        "checksum_algorithm": manifest.checksum_algorithm,
        "tree_sha256": report.get("tree_sha256"),
        "file_count": report.get("file_count"),
        "total_size_bytes": report.get("total_size_bytes"),
        "sample_files": report.get("sample_files", []),
        "local_availability": report["local_availability"],
        "validated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
    }
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return status_path


def _confirm_download(args: argparse.Namespace) -> bool:
    """Return True only when the user has explicitly confirmed a network download.

    Requires the ``--confirm-download`` flag AND an interactive y/N (or ``--yes`` for
    non-interactive confirmation). Default behavior never confirms.
    """
    if not args.confirm_download:
        return False
    if args.yes:
        return True
    try:
        answer = input(
            "About to download the Stanford Drone Dataset to a git-ignored folder. "
            "This consumes disk and network. Proceed? [y/N]: "
        )
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def run_download(manifest: SddManifest, args: argparse.Namespace) -> dict[str, Any]:
    """Run the guarded download path. Fails closed at every safety gate."""
    if not _confirm_download(args):
        raise SddStagingError(
            "Download not confirmed. This tool never auto-downloads: pass `--confirm-download` and "
            "answer the y/N prompt (or add `--yes` for non-interactive confirmation). A default run "
            "only plans and reports."
        )

    disk = check_disk_space(manifest)
    if not disk["sufficient"]:
        raise SddStagingError(
            "Insufficient disk space at "
            f"{disk['probe_path']}: required {disk['required_mib']} MiB, "
            f"available {disk['available_mib']} MiB. Refusing to download (fail closed)."
        )

    if not manifest.download_url:
        raise SddStagingError(
            "No download URL is configured for SDD. SDD is license-gated and the repository encodes "
            "no approved direct download URL. Obtain it from the official project page "
            f"({manifest.source_url}) under its license, place the annotation files under "
            f"{manifest.staging_dir}, then run validation. {manifest.access_note}"
        )

    # A real network fetch would happen here using manifest.download_url. It is intentionally NOT
    # implemented because SDD is license-gated and no approved URL is encoded; reaching this point
    # requires a maintainer to add a URL to the manifest first.
    raise SddStagingError(
        "Confirmed and disk-checked, but no downloader is wired for the configured URL. A "
        "maintainer must implement the fetch for the license-approved URL before this path runs."
    )


def _emit(payload: dict[str, Any], *, as_json: bool, human_lines: list[str]) -> None:
    """Print JSON or human-readable output."""
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("\n".join(human_lines))


def _plan_lines(plan: dict[str, Any]) -> list[str]:
    """Human-readable lines for a plan/report payload."""
    disk = plan["disk_check"]
    return [
        f"SDD staging PLAN (asset_id={plan['asset_id']}, version={plan['version_tag']})",
        f"  source:        {plan['source_url']}",
        f"  license:       {plan['license']} ({plan['license_url']})",
        f"  readme:        {plan['readme_pointer']}",
        f"  staging_dir:   {plan['staging_dir']} (git-ignored)",
        f"  expected size: {plan['expected_total_size_mib']} MiB",
        f"  disk check:    required {disk['required_mib']} MiB, "
        f"available {disk['available_mib']} MiB -> "
        f"{'OK' if disk['sufficient'] else 'INSUFFICIENT'}",
        f"  availability:  {plan['local_availability']}",
        f"  prior mode:    {plan['scenario_prior_mode']}",
        "  download:      NO (plan only; never auto-downloads)",
        f"  next step:     {plan['next_step']}",
    ]


def _validation_lines(report: dict[str, Any]) -> list[str]:
    """Human-readable lines for a validation/check payload."""
    lines = [
        f"SDD staging status (asset_id={report['asset_id']}, version={report['version_tag']})",
        f"  staging_dir:   {report['staging_dir']} (exists={report['staging_dir_exists']})",
        f"  availability:  {report['local_availability']}",
        f"  prior mode:    {report['mode']}",
    ]
    if report.get("tree_sha256"):
        lines.append(f"  tree_sha256:   {report['tree_sha256']}")
    if report["missing_expected"]:
        lines.append(f"  missing:       {', '.join(report['missing_expected'])}")
    lines.append(f"  action:        {report['action']}")
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Staging manifest path (default: {DEFAULT_MANIFEST_PATH}).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report local availability status and checksums WITHOUT downloading.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("plan", help="Plan/report only (default); never downloads.")
    sub.add_parser("validate", help="Validate the locally-staged SDD copy; never downloads.")
    sub.add_parser("status", help="Alias for --check: availability status; never downloads.")
    sub.add_parser("mode", help="Print the scenario-prior mode gate (proxy vs dataset-backed).")
    download_parser = sub.add_parser(
        "download", help="Guarded staging: requires explicit confirmation; fails closed."
    )
    download_parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Explicitly confirm a network download (still prompts unless --yes).",
    )
    download_parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive y/N prompt for non-interactive confirmation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the SDD staging tool."""
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        command = args.command
        if args.check and command in (None, "plan"):
            command = "status"

        if command in (None, "plan"):
            plan = build_plan(manifest)
            _emit(plan, as_json=args.json, human_lines=_plan_lines(plan))
            return 0

        if command in ("status", "validate"):
            report = validate_staging(manifest)
            if command == "validate" and report["ok"]:
                _write_staging_status(manifest, report)
            _emit(report, as_json=args.json, human_lines=_validation_lines(report))
            return 0 if report["ok"] else 2

        if command == "mode":
            gate = resolve_scenario_prior_mode(manifest)
            _emit(
                gate,
                as_json=args.json,
                human_lines=[
                    f"scenario_prior_mode: {gate['mode']}",
                    f"dataset_backed: {gate['dataset_backed']}",
                    f"reason: {gate['reason']}",
                ],
            )
            return 0 if gate["dataset_backed"] else 2

        if command == "download":
            report = run_download(manifest, args)
            _emit(report, as_json=args.json, human_lines=_validation_lines(report))
            return 0 if report.get("ok") else 2

        raise SddStagingError(f"Unknown command: {command}")
    except SddStagingError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
