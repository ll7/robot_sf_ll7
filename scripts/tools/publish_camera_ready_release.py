#!/usr/bin/env python3
"""Validate and publish a camera-ready benchmark bundle as a release asset."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from loguru import logger

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from robot_sf.benchmark.release_protocol import resolve_campaign_artifact_path
from robot_sf.benchmark.release_publication_contract import (
    validate_release_publication_contract,
)
from robot_sf.benchmark.release_tag_identity import check_tag_source_consistency
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
        "--create-draft",
        action="store_true",
        help=(
            "Create the missing tag-targeted draft GitHub Release before upload. "
            "Requires --expected-source-sha; fails closed when the tag already "
            "exists on a different target or a non-draft release is present."
        ),
    )
    parser.add_argument(
        "--expected-source-sha",
        default=None,
        help=(
            "Exact 40-character source SHA that the release tag must resolve to. "
            "Required with --create-draft so the draft binds one immutable target."
        ),
    )
    parser.add_argument(
        "--release-title",
        default=None,
        help="Optional GitHub Release title (default: derived from the campaign release id).",
    )
    parser.add_argument(
        "--release-notes",
        default=None,
        help="Optional GitHub Release notes (default: derived from the campaign summary).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path for writing the publish plan JSON payload.",
    )
    return parser


def _resolve_publication_path(publication: dict[str, object], key: str, repo_root: Path) -> Path:
    """Resolve and validate one required publication path field."""
    raw_value = publication.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(
            f"publication_bundle.{key} must be a non-empty string path in campaign_summary.json."
        )
    relative = raw_value.strip()
    try:
        return resolve_campaign_artifact_path(repo_root, relative)
    except ValueError as exc:
        # Preserve the command's established missing-artifact error while still
        # routing every candidate through the fail-closed path validator first.
        candidate = Path(repo_root).absolute() / relative
        if (
            "not a regular file" in str(exc)
            and not candidate.exists()
            and not candidate.is_symlink()
        ):
            raise FileNotFoundError(f"Missing required publication artifact: {candidate}") from exc
        raise


def _validate_prerequisites(
    campaign_root: Path, *, expected_release_tag: str
) -> tuple[Path, Path, Path, dict[str, object]]:
    """Validate campaign publication artifacts and return core paths plus campaign summary."""
    summary_path = resolve_campaign_artifact_path(campaign_root, "reports/campaign_summary.json")

    summary = _load_json(summary_path)
    publication = summary.get("publication_bundle")
    if not isinstance(publication, dict):
        raise ValueError(
            "campaign_summary.json is missing publication_bundle metadata. "
            "Run the campaign with publication bundle export enabled."
        )

    repo_root = get_repository_root()
    archive_path = _resolve_publication_path(publication, "archive_path", repo_root)
    checksums_path = _resolve_publication_path(publication, "checksums_path", repo_root)
    manifest_path = _resolve_publication_path(publication, "manifest_path", repo_root)
    bundle_dir = manifest_path.parent
    if not bundle_dir.is_dir():
        raise ValueError("publication bundle path must name a regular directory")
    if checksums_path.parent != bundle_dir:
        raise ValueError("publication bundle checksums must be in the manifest bundle directory")
    if archive_path.parent != bundle_dir.parent:
        raise ValueError("publication bundle archive must be beside the manifest bundle directory")

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

    contract = validate_release_publication_contract(
        campaign_root,
        manifest_path.parent,
        expected_release_tag=expected_release_tag,
    )
    if contract["status"] != "pass":
        blockers = "\n".join(f"- {item}" for item in contract["blockers"])
        raise ValueError(f"Release publication contract blocked:\n{blockers}")

    return archive_path, checksums_path, manifest_path, summary


def _summary_source_sha(summary: dict[str, object]) -> str | None:  # noqa: C901
    """Return one exact source SHA declared by the campaign summary.

    Returns:
        One normalized source SHA, or ``None`` when no source is declared.
    """
    candidates: list[str] = []
    campaign = summary.get("campaign")
    if isinstance(campaign, dict):
        for key in ("source_sha", "source_commit", "git_hash"):
            value = campaign.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip().lower())
    release = summary.get("benchmark_release")
    if isinstance(release, dict):
        for key in ("source_sha", "source_commit"):
            value = release.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip().lower())
    acceptance = summary.get("full_release_acceptance")
    if isinstance(acceptance, dict):
        values = acceptance.get("source_commits")
        if isinstance(values, list):
            candidates.extend(
                value.strip().lower()
                for value in values
                if isinstance(value, str) and value.strip()
            )
    if not candidates:
        return None
    if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in candidates):
        raise ValueError("campaign summary source identity must use exact 40-character Git SHAs")
    distinct = set(candidates)
    if len(distinct) != 1:
        raise ValueError("campaign summary source identity fields disagree")
    return next(iter(distinct))


def _validate_source_identity(
    summary: dict[str, object], *, tag: str, expected_source_sha: str | None
) -> str | None:
    """Bind the requested tag to the final source recorded by the campaign."""
    expected = expected_source_sha.strip().lower() if expected_source_sha else None
    if expected is not None and re.fullmatch(r"[0-9a-f]{40}", expected) is None:
        raise ValueError("--expected-source-sha must be an exact 40-character lowercase SHA")
    declared = _summary_source_sha(summary)
    if expected is not None and declared is not None and declared != expected:
        raise ValueError(
            f"campaign summary source SHA {declared!r} does not match expected source {expected!r}"
        )
    source_sha = expected or declared
    if source_sha is None:
        raise ValueError(
            "publication source SHA is missing; record the final immutable source before upload"
        )
    problems = check_tag_source_consistency(tag, source_sha)
    if problems:
        raise ValueError("Release tag/source identity blocked: " + "; ".join(problems))
    return source_sha


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


def _resolve_release_identity(
    summary: dict[str, object],
    *,
    tag: str,
    release_title: str | None,
    release_notes: str | None,
) -> tuple[str, str]:
    """Return the draft release title and notes bound to the exact campaign identity.

    The release tag must already match the publication contract; this derives a
    deterministic human-readable title and notes from the campaign summary so a
    first publication has no hidden manual release-creation step.
    """
    campaign = summary.get("campaign") if isinstance(summary.get("campaign"), dict) else {}
    release_id = str(campaign.get("release_id") or "").strip() or tag
    repository_url = (
        str(campaign.get("repository_url", "")) or "https://github.com/ll7/robot_sf_ll7"
    )
    doi = str(campaign.get("doi", "")).strip()
    title = (release_title or "").strip() or f"Benchmark data release {release_id}"
    notes_lines = [f"Benchmark data release `{release_id}` for tag `{tag}`."]
    if repository_url:
        notes_lines.append(f"Repository: {repository_url}")
    if doi:
        notes_lines.append(f"DOI: https://doi.org/{doi}")
    notes = (release_notes or "").strip() or "\n".join(notes_lines)
    return title, notes


def _build_draft_create_command(
    *,
    repo: str,
    tag: str,
    source_sha: str,
    release_title: str,
    release_notes: str,
) -> list[str]:
    """Build the `gh release create --draft` command binding tag to one SHA."""
    return [
        "gh",
        "release",
        "create",
        tag,
        "--repo",
        repo,
        "--draft",
        "--title",
        release_title,
        "--notes",
        release_notes,
        "--target",
        source_sha,
    ]


def _check_release_collision(
    *,
    repo: str,
    tag: str,
    expected_source_sha: str,
    dry_run: bool,
) -> tuple[str | None, bool]:
    """Fail closed when the tag already has a release on a different target.

    In dry-run mode the live GitHub state is not queried: the plan reports the
    create-then-upload command order and the execute path performs the
    collision check immediately before any mutation.

    Returns:
        A ``(blocker, exists_at_target)`` pair: the blocker message when the
        existing release would be mutated or is non-draft, and whether an
        exact-SHA draft already exists (in which case creation is skipped and
        only the upload proceeds).
    """
    if dry_run:
        return None, False
    view_cmd = ["gh", "release", "view", tag, "--repo", repo, "--json", "isDraft,targetCommitish"]
    try:
        result = subprocess.run(
            view_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        # Only an explicit not-found response means that creation is safe.
        # Authentication, transport, or malformed-query errors are ambiguous
        # live state and must not be treated as an unused tag.
        detail = str(exc.stderr or exc.output or "")
        if not re.search(r"(?:not found|does not exist|HTTP 404|status 404)", detail, re.I):
            return (
                f"cannot determine whether release {tag} exists: {detail or 'unknown error'}",
                False,
            )
        return None, False
    try:
        existing = json.loads(result.stdout)
    except json.JSONDecodeError:
        return f"cannot parse `gh release view {tag}` output; refusing to create or upload", False
    target = str(existing.get("targetCommitish") or "").strip()
    is_draft = bool(existing.get("isDraft"))
    if target and target != expected_source_sha:
        return (
            f"release {tag} already exists at target {target!r}, not the required "
            f"{expected_source_sha!r}; refusing to create or upload",
            False,
        )
    if not is_draft:
        return f"release {tag} already exists and is not a draft; refusing to mutate it", False
    return None, True


def _check_tag_collision(*, repo: str, tag: str, dry_run: bool) -> str | None:
    """Fail closed when GitHub already has the tag planned for creation."""
    if dry_run:
        return None
    endpoint = f"repos/{repo}/git/ref/tags/{quote(tag, safe='')}"
    result = subprocess.run(
        ["gh", "api", endpoint],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return f"tag {tag} already exists; refusing to retarget or overwrite it"
    detail = str(result.stderr or result.stdout or "")
    if re.search(r"(?:not found|does not exist|HTTP 404|status 404)", detail, re.I):
        return None
    return f"cannot determine whether tag {tag} exists: {detail or 'unknown error'}"


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    """Run guided publication workflow and return POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.create_draft and not args.expected_source_sha:
        parser.error("--create-draft requires --expected-source-sha")
    if args.expected_source_sha and len(args.expected_source_sha) != 40:
        parser.error("--expected-source-sha must be an exact 40-character SHA")

    campaign_root = Path(args.campaign_root).absolute()
    archive_path, checksums_path, manifest_path, summary = _validate_prerequisites(
        campaign_root, expected_release_tag=str(args.tag)
    )
    source_sha = _validate_source_identity(
        summary,
        tag=str(args.tag),
        expected_source_sha=(
            str(args.expected_source_sha) if args.expected_source_sha is not None else None
        ),
    )
    payload = _build_release_payload(
        campaign_root=campaign_root,
        repo=str(args.repo),
        tag=str(args.tag),
        archive_path=archive_path,
        checksums_path=checksums_path,
        manifest_path=manifest_path,
        summary=summary,
    )
    payload["source_sha"] = source_sha

    draft_create_command: list[str] | None = None
    if args.create_draft:
        assert source_sha is not None
        title, notes = _resolve_release_identity(
            summary,
            tag=str(args.tag),
            release_title=str(args.release_title) if args.release_title else None,
            release_notes=str(args.release_notes) if args.release_notes else None,
        )
        blocker, exists_at_target = _check_release_collision(
            repo=str(args.repo),
            tag=str(args.tag),
            expected_source_sha=source_sha,
            dry_run=not args.execute_upload,
        )
        if blocker is not None:
            raise SystemExit(f"Release draft admission blocked: {blocker}")
        if not exists_at_target:
            blocker = _check_tag_collision(
                repo=str(args.repo),
                tag=str(args.tag),
                dry_run=not args.execute_upload,
            )
            if blocker is not None:
                raise SystemExit(f"Release tag admission blocked: {blocker}")
        draft_create_command = (
            None
            if exists_at_target
            else _build_draft_create_command(
                repo=str(args.repo),
                tag=str(args.tag),
                source_sha=source_sha,
                release_title=title,
                release_notes=notes,
            )
        )
        payload["expected_source_sha"] = source_sha
        payload["release_title"] = title
        if draft_create_command is not None:
            payload["draft_create_command"] = draft_create_command

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.create_draft and args.execute_upload and draft_create_command is not None:
        logger.info("Creating draft release for tag={} repo={}", args.tag, args.repo)
        subprocess.run(draft_create_command, check=True)

    if args.execute_upload:
        logger.info("Executing release upload for tag={} repo={}", args.tag, args.repo)
        subprocess.run(payload["upload_command"], check=True)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
