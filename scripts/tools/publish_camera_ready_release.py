#!/usr/bin/env python3
"""Validate and publish a camera-ready benchmark bundle as a release asset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    """Create argument parser for guided release publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Path to camera-ready campaign output directory.",
    )
    parser.add_argument(
        "--repo",
        default="ll7/robot_sf_ll7",
        help="GitHub repository in owner/name format used for release commands.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="GitHub release tag to upload assets to.",
    )
    parser.add_argument(
        "--execute-upload",
        action="store_true",
        help="Execute `gh release upload` after validation (default: dry-run plan only).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path for writing the publish plan JSON payload.",
    )
    return parser


def _load_json(path: Path) -> dict[str, object]:
    """Load and validate JSON object from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _resolve_publication_path(publication: dict[str, object], key: str, repo_root: Path) -> Path:
    """Resolve and validate one required publication path field."""
    raw_value = publication.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(
            f"publication_bundle.{key} must be a non-empty string path in campaign_summary.json."
        )
    return repo_root / raw_value


def _validate_prerequisites(campaign_root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    """Validate campaign publication artifacts and return core paths plus campaign summary."""
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing campaign summary: {summary_path}")

    summary = _load_json(summary_path)
    publication = summary.get("publication_bundle")
    if not isinstance(publication, dict):
        raise ValueError(
            "campaign_summary.json is missing publication_bundle metadata. "
            "Run the campaign with publication bundle export enabled."
        )

    repo_root = get_repository_root().resolve()
    archive_path = _resolve_publication_path(publication, "archive_path", repo_root)
    checksums_path = _resolve_publication_path(publication, "checksums_path", repo_root)
    manifest_path = _resolve_publication_path(publication, "manifest_path", repo_root)

    for path in (archive_path, checksums_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required publication artifact: {path}")

    checksum_lines = [
        line.strip()
        for line in checksums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not checksum_lines:
        raise ValueError(f"Checksums file is empty: {checksums_path}")

    return archive_path, checksums_path, manifest_path, summary


def _build_release_payload(
    *,
    campaign_root: Path,
    repo: str,
    tag: str,
    archive_path: Path,
    checksums_path: Path,
    manifest_path: Path,
    summary: dict[str, object],
) -> dict[str, object]:
    """Build release publication metadata and command plan."""
    campaign = summary.get("campaign") if isinstance(summary.get("campaign"), dict) else {}
    repository_url = str(campaign.get("repository_url", f"https://github.com/{repo}"))
    doi = str(campaign.get("doi", "10.5281/zenodo.<record-id>"))

    upload_cmd = [
        "gh",
        "release",
        "upload",
        tag,
        str(archive_path),
        str(checksums_path),
        str(manifest_path),
        "--repo",
        repo,
        "--clobber",
    ]

    return {
        "campaign_root": str(campaign_root),
        "repo": repo,
        "tag": tag,
        "archive_path": str(archive_path),
        "checksums_path": str(checksums_path),
        "manifest_path": str(manifest_path),
        "release_url": f"{repository_url.rstrip('/')}/releases/tag/{tag}",
        "release_asset_url": (
            f"{repository_url.rstrip('/')}/releases/download/{tag}/{archive_path.name}"
        ),
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
        "upload_command": upload_cmd,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run guided publication workflow and return POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    campaign_root = args.campaign_root.resolve()
    archive_path, checksums_path, manifest_path, summary = _validate_prerequisites(campaign_root)
    payload = _build_release_payload(
        campaign_root=campaign_root,
        repo=str(args.repo),
        tag=str(args.tag),
        archive_path=archive_path,
        checksums_path=checksums_path,
        manifest_path=manifest_path,
        summary=summary,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.execute_upload:
        logger.info("Executing release upload for tag={} repo={}", args.tag, args.repo)
        subprocess.run(payload["upload_command"], check=True)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
