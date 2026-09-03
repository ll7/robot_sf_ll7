#!/usr/bin/env python3
"""Validate and publish a camera-ready benchmark bundle as a release asset."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from loguru import logger

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json
from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.benchmark.release_protocol import resolve_campaign_artifact_path
from robot_sf.benchmark.release_publication_contract import (
    validate_release_publication_contract,
)
from robot_sf.benchmark.release_tag_identity import check_tag_source_consistency
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from collections.abc import Sequence

ERRATUM_CUSTODY_ASSET = "publication_custody.json"
_ERRATUM_TAG_RE = re.compile(r".+-[0-9a-f]{40}-erratum\.[1-9][0-9]*$")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_HTTP_HEADER_STATUS_RE = re.compile(r"^\s*HTTP(?:/\d+(?:\.\d+)?)?\s+(\d{3})(?!\d)", re.I)
_HTTP_ERROR_STATUS_RE = re.compile(
    r"(?:^\s*HTTP(?:/\d+(?:\.\d+)?)?\s+|\(HTTP(?:/\d+(?:\.\d+)?)?\s+)"
    r"(?P<status>\d{3})(?!\d)",
    re.I,
)
_CUSTODY_SCHEMA = "benchmark-publication-custody.v1"
_CUSTODY_DIGEST_POLICY = "archive digest is external to the bundle; no cycle"
_DRAFT_READBACK_ATTEMPTS = 10
_DRAFT_READBACK_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class _LocalAsset:
    """Immutable local asset identity used for remote draft admission."""

    path: Path
    name: str
    size: int
    sha256: str


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
        help=(
            "Execute a fail-closed upload into an exact unpublished draft after validation "
            "(default: dry-run plan only)."
        ),
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
            "Required for every mutating mode so the draft binds one immutable target."
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


def _resolve_publication_path(
    publication: dict[str, object],
    key: str,
    *,
    campaign_root: Path,
    repo_root: Path,
) -> Path:
    """Resolve one publication path against its declared campaign or repository root."""
    raw_value = publication.get(key)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(
            f"publication_bundle.{key} must be a non-empty string path in campaign_summary.json."
        )
    relative = raw_value.strip()
    roots = tuple(dict.fromkeys((Path(campaign_root).absolute(), Path(repo_root).absolute())))
    matches: list[Path] = []
    missing_candidates: list[Path] = []
    for root in roots:
        try:
            matches.append(resolve_campaign_artifact_path(root, relative))
        except ValueError as exc:
            # An absent path may belong to the other supported root. Security
            # violations are root-independent and must still fail immediately.
            if "not a regular file" not in str(exc):
                raise
            candidate = root / relative
            if candidate.exists() or candidate.is_symlink():
                raise
            missing_candidates.append(candidate)
    distinct_matches = tuple(dict.fromkeys(matches))
    if len(distinct_matches) > 1:
        raise ValueError(
            f"publication_bundle.{key} is ambiguous between campaign and repository roots"
        )
    if distinct_matches:
        return distinct_matches[0]
    candidate_text = ", ".join(str(path) for path in missing_candidates)
    raise FileNotFoundError(f"Missing required publication artifact: {candidate_text}")


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
    archive_path = _resolve_publication_path(
        publication,
        "archive_path",
        campaign_root=campaign_root,
        repo_root=repo_root,
    )
    checksums_path = _resolve_publication_path(
        publication,
        "checksums_path",
        campaign_root=campaign_root,
        repo_root=repo_root,
    )
    manifest_path = _resolve_publication_path(
        publication,
        "manifest_path",
        campaign_root=campaign_root,
        repo_root=repo_root,
    )
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


def _resolve_upload_assets(
    *,
    tag: str,
    archive_path: Path,
    checksums_path: Path,
    manifest_path: Path,
) -> tuple[Path, ...]:
    """Return the complete, ordered GitHub asset set for this release.

    Canonical errata carry a detached custody receipt beside the archive. It
    cannot live inside the archive because it binds the completed archive
    digest, so omitting it would make the two-channel cold audit impossible.

    Returns:
        The archive, checksum manifest, publication manifest, and (for an
        erratum) detached custody receipt.
    """
    assets = [archive_path, checksums_path, manifest_path]
    if _ERRATUM_TAG_RE.fullmatch(tag) is None:
        return tuple(assets)
    custody_path = resolve_campaign_artifact_path(
        archive_path.parent,
        ERRATUM_CUSTODY_ASSET,
    )
    if custody_path.parent != archive_path.parent:
        raise ValueError("erratum custody receipt must be beside the publication archive")
    assets.append(custody_path)
    return tuple(assets)


def _require_mapping(value: object, *, label: str) -> Mapping[str, object]:
    """Return one JSON mapping or reject an ambiguous publication receipt."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    """Return a canonical lowercase SHA-256 value."""
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.strip()) is None:
        raise ValueError(f"{label} must be an exact 64-character SHA-256")
    return value.strip().lower()


def _require_positive_size(value: object, *, label: str) -> int:
    """Return a positive JSON integer size, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _validate_custody_paths(
    custody: Mapping[str, object], *, archive_path: Path, manifest_path: Path
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Validate canonical archive and bundle path fields in a custody receipt."""
    archive = _require_mapping(custody.get("archive"), label="erratum custody archive")
    bundle = _require_mapping(custody.get("bundle"), label="erratum custody bundle")
    archive_name = archive.get("path")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or "/" in archive_name
        or "\\" in archive_name
    ):
        raise ValueError("erratum custody archive path is not a canonical file name")
    if archive_name != archive_path.name:
        raise ValueError("erratum custody archive path does not match the local archive name")
    bundle_name = bundle.get("path")
    if (
        not isinstance(bundle_name, str)
        or Path(bundle_name).name != bundle_name
        or "/" in bundle_name
        or "\\" in bundle_name
    ):
        raise ValueError("erratum custody bundle path is not a canonical directory name")
    if bundle_name != manifest_path.parent.name:
        raise ValueError("erratum custody bundle path does not match the local bundle")
    declared_bundle_name = custody.get("bundle_name")
    if declared_bundle_name != bundle_name:
        raise ValueError("erratum custody bundle_name differs from the local bundle")
    return archive, bundle


def _validate_custody_archive(archive: Mapping[str, object], *, archive_path: Path) -> None:
    """Validate the detached receipt's archive size and digest."""
    expected_size = _require_positive_size(
        archive.get("size_bytes"), label="erratum custody archive size"
    )
    if archive_path.stat().st_size != expected_size:
        raise ValueError("erratum custody archive size is stale")
    expected_sha = _require_sha256(archive.get("sha256"), label="erratum custody archive SHA-256")
    if _sha256_file(archive_path) != expected_sha:
        raise ValueError("erratum custody archive SHA-256 is stale")


def _validate_custody_bundle(
    bundle: Mapping[str, object], *, checksums_path: Path, manifest_path: Path
) -> None:
    """Validate the detached receipt's publication sidecar digests."""
    expected_manifest_sha = _require_sha256(
        bundle.get("publication_manifest_sha256"),
        label="erratum custody publication manifest SHA-256",
    )
    expected_checksums_sha = _require_sha256(
        bundle.get("checksums_sha256"), label="erratum custody checksums SHA-256"
    )
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("erratum custody publication manifest SHA-256 is stale")
    if _sha256_file(checksums_path) != expected_checksums_sha:
        raise ValueError("erratum custody checksums SHA-256 is stale")


def _validate_erratum_custody(
    custody_path: Path,
    *,
    archive_path: Path,
    checksums_path: Path,
    manifest_path: Path,
    source_sha: str,
) -> None:
    """Bind detached erratum custody to the exact local bytes before mutation.

    The detached receipt is deliberately checked before any GitHub lookup that
    could lead to a draft creation or upload.  This closes the gap where a
    valid-looking campaign could mutate a remote release while its archive
    custody sidecar described different local bytes.
    """
    if custody_path.is_symlink() or not custody_path.is_file():
        raise ValueError("erratum custody receipt must be a regular non-symlink file")
    custody = _require_mapping(_load_json(custody_path), label="erratum custody receipt")
    if custody.get("schema_version") != _CUSTODY_SCHEMA:
        raise ValueError("erratum custody receipt has an unsupported schema")
    if custody.get("source_execution_commit") != source_sha:
        raise ValueError("erratum custody source commit differs from expected source SHA")
    if custody.get("credentials") != "not_recorded":
        raise ValueError("erratum custody receipt has an invalid credential policy")
    if custody.get("archive_self_digest_policy") != _CUSTODY_DIGEST_POLICY:
        raise ValueError("erratum custody receipt has a noncanonical archive digest policy")
    archive, bundle = _validate_custody_paths(
        custody, archive_path=archive_path, manifest_path=manifest_path
    )
    _validate_custody_archive(archive, archive_path=archive_path)
    _validate_custody_bundle(bundle, checksums_path=checksums_path, manifest_path=manifest_path)


def _local_asset_records(upload_assets: tuple[Path, ...]) -> tuple[_LocalAsset, ...]:
    """Compute the immutable local name/size/digest set before remote mutation."""
    records: list[_LocalAsset] = []
    names: set[str] = set()
    for path in upload_assets:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"release asset must be a regular non-symlink file: {path}")
        name = path.name
        if not name or name in names:
            raise ValueError(f"release asset names must be unique: {name!r}")
        names.add(name)
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"release asset must not be empty: {path}")
        records.append(_LocalAsset(path=path, name=name, size=size, sha256=_sha256_file(path)))
    return tuple(records)


def _validate_local_asset_snapshot(local_assets: tuple[_LocalAsset, ...]) -> str | None:
    """Reject local asset changes observed after the initial digest snapshot."""
    for asset in local_assets:
        if asset.path.is_symlink() or not asset.path.is_file():
            return f"local release asset changed or disappeared: {asset.name!r}"
        if asset.path.stat().st_size != asset.size or _sha256_file(asset.path) != asset.sha256:
            return f"local release asset changed after preflight: {asset.name!r}"
    return None


def _build_release_payload(
    *,
    campaign_root: Path,
    repo: str,
    tag: str,
    archive_path: Path,
    checksums_path: Path,
    manifest_path: Path,
    upload_assets: tuple[Path, ...],
    summary: dict[str, object],
) -> dict[str, object]:
    """Build release publication metadata and command plan."""
    campaign = summary.get("campaign") if isinstance(summary.get("campaign"), dict) else {}
    repository_url = str(campaign.get("repository_url", f"https://github.com/{repo}"))
    doi = str(campaign.get("doi", "10.5281/zenodo.<record-id>"))

    upload_cmd = _build_upload_command(
        repo=repo,
        tag=tag,
        upload_assets=upload_assets,
    )

    return {
        "campaign_root": str(campaign_root),
        "repo": repo,
        "tag": tag,
        "archive_path": str(archive_path),
        "checksums_path": str(checksums_path),
        "manifest_path": str(manifest_path),
        "upload_assets": [str(path) for path in upload_assets],
        "release_url": f"{repository_url.rstrip('/')}/releases/tag/{tag}",
        "release_asset_url": (
            f"{repository_url.rstrip('/')}/releases/download/{tag}/{archive_path.name}"
        ),
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
        "upload_command": upload_cmd,
    }


def _build_upload_command(
    *,
    repo: str,
    tag: str,
    upload_assets: tuple[Path, ...],
) -> list[str]:
    """Build a non-clobbering upload command for exactly the missing assets."""
    if not upload_assets:
        return []
    return [
        "gh",
        "release",
        "upload",
        tag,
        *(str(path) for path in upload_assets),
        "--repo",
        repo,
    ]


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


def _text_output(value: object) -> str:
    """Normalize subprocess output without ever rendering credentials."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _http_status_code(result: subprocess.CompletedProcess[object]) -> int | None:
    """Extract an explicit HTTP status line from a ``gh api`` response.

    A bare ``Not Found`` message is intentionally not enough to establish a
    missing tag.  The caller may treat only an explicit REST 404 as absence.
    """
    for line in _text_output(result.stdout).splitlines():
        match = _HTTP_HEADER_STATUS_RE.match(line)
        if match is not None:
            return int(match.group(1))
    match = _HTTP_ERROR_STATUS_RE.search(_text_output(result.stderr))
    if match is not None:
        return int(match.group("status"))
    return None


def _parse_json_output(raw: object, *, label: str) -> object:
    """Parse JSON from plain or ``gh api --include`` output."""
    text = _text_output(raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                payload, end = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if not text[index + end :].strip():
                return payload
    raise ValueError(f"{label} returned malformed JSON")


def _run_tag_api(repo: str, endpoint: str) -> subprocess.CompletedProcess[str]:
    """Read one GitHub tag endpoint with headers for exact status handling."""
    return subprocess.run(
        ["gh", "api", "--include", f"repos/{repo}/{endpoint}"],
        check=False,
        capture_output=True,
        text=True,
    )


def _parse_tag_object(payload: object, *, tag: str, label: str) -> tuple[str, str] | str:
    """Extract a typed tag object identity or a fail-closed error."""
    if not isinstance(payload, dict):
        return f"tag {tag} {label} returned an invalid response shape"
    object_value = payload.get("object")
    if not isinstance(object_value, dict):
        return f"tag {tag} {label} has no unambiguous object"
    object_type = object_value.get("type")
    object_sha = object_value.get("sha")
    if not isinstance(object_type, str) or not isinstance(object_sha, str):
        return f"tag {tag} {label} has an invalid object identity"
    object_sha = object_sha.strip().lower()
    if _SHA1_RE.fullmatch(object_sha) is None:
        return f"tag {tag} {label} has an invalid object SHA"
    return object_type, object_sha


def _peel_annotated_tag(
    *,
    repo: str,
    tag: str,
    object_sha: str,
) -> tuple[str | None, str | None]:
    """Peel an annotated tag object through bounded tag-object lookups."""
    visited: set[str] = set()
    current_sha = object_sha
    for _ in range(8):
        if current_sha in visited:
            return None, f"tag {tag} annotated target contains a cycle"
        visited.add(current_sha)
        result = _run_tag_api(repo, f"git/tags/{current_sha}")
        status = _http_status_code(result)
        if result.returncode != 0 or (status is not None and not 200 <= status < 300):
            detail = _text_output(result.stderr or result.stdout).strip()
            return None, f"cannot peel annotated tag {tag}: {detail or 'unknown REST/API error'}"
        try:
            payload = _parse_json_output(result.stdout, label=f"tag {tag} peel")
        except ValueError as exc:
            return None, str(exc)
        parsed = _parse_tag_object(payload, tag=tag, label="peel")
        if isinstance(parsed, str):
            return None, parsed
        object_type, current_sha = parsed
        if object_type == "commit":
            return current_sha, None
        if object_type != "tag":
            return None, f"tag {tag} peels to unsupported object type {object_type!r}"
    return None, f"tag {tag} annotated target is too deeply nested"


def _is_not_found_result(
    result: subprocess.CompletedProcess[str],
    status: int | None,
) -> bool:
    """Return True when a GitHub response represents a missing resource.

    A ``gh api`` call can exit non-zero with an unparseable status line when a
    remote resource is absent.  Treat both an explicit 404 and a non-zero exit
    whose error body quotes a 404 as absence so the caller can fail closed on
    genuine transport/server errors while still admitting missing refs.
    """
    if status == 404:
        return True
    if status is not None:
        return False
    if result.returncode == 0:
        return False
    error_text = _text_output(result.stderr or result.stdout)
    return bool(re.search(r"\b404\b", error_text))


def _resolve_tag_ref_target(
    *,
    repo: str,
    tag: str,
    allow_absent: bool,
) -> tuple[str | None, str | None]:
    """Resolve a GitHub tag ref to its peeled commit SHA.

    GitHub represents a lightweight tag with an object of type ``commit`` and
    an annotated tag with an object of type ``tag``.  Annotated objects are
    peeled through the tag-object endpoint; ambiguous or malformed responses
    are never interpreted as absence.
    """
    result = _run_tag_api(repo, f"git/ref/tags/{quote(tag, safe='')}")
    status = _http_status_code(result)
    if result.returncode != 0 or (status is not None and not 200 <= status < 300):
        if allow_absent and _is_not_found_result(result, status):
            return None, None
        detail = _text_output(result.stderr or result.stdout).strip()
        return None, f"cannot resolve tag {tag}: {detail or 'unknown REST/API error'}"
    try:
        payload = _parse_json_output(result.stdout, label=f"tag {tag} lookup")
    except ValueError as exc:
        return None, str(exc)
    if isinstance(payload, dict):
        response_ref = payload.get("ref")
        if response_ref != f"refs/tags/{tag}":
            return None, f"tag {tag} lookup returned a different ref"
    parsed = _parse_tag_object(payload, tag=tag, label="lookup")
    if isinstance(parsed, str):
        return None, parsed
    object_type, object_sha = parsed
    if object_type == "commit":
        return object_sha, None
    if object_type != "tag":
        return None, f"tag {tag} resolves to unsupported object type {object_type!r}"
    return _peel_annotated_tag(repo=repo, tag=tag, object_sha=object_sha)


def _query_release_listing(
    *,
    repo: str,
    tag: str,
) -> tuple[dict[str, object] | None, str | None]:
    """Read the complete release listing and return the exact tag record."""
    endpoint = f"repos/{repo}/releases?per_page=100"
    result = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", endpoint],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = _text_output(result.stderr or result.stdout).strip()
        return None, f"cannot determine whether release {tag} exists: {detail or 'unknown error'}"
    try:
        existing, parse_blocker = _parse_release_listing(result.stdout, tag=tag)
    except (TypeError, ValueError) as exc:
        return None, f"release lookup for {tag} is malformed: {exc}"
    return existing, parse_blocker


def _release_target_blocker(release: dict[str, object], *, tag: str, source_sha: str) -> str | None:
    """Validate the release's explicit target and unpublished draft state."""
    target_value = release.get("target_commitish")
    if not isinstance(target_value, str) or not target_value.strip():
        return f"release {tag} does not declare an exact target commit; refusing to mutate"
    target = target_value.strip().lower()
    if _SHA1_RE.fullmatch(target) is None:
        return f"release {tag} target is not an exact commit SHA; refusing to mutate"
    if target != source_sha:
        return (
            f"release {tag} already exists at target {target!r}, not the required "
            f"{source_sha!r}; refusing to create or upload"
        )
    draft_value = release.get("draft")
    if not isinstance(draft_value, bool):
        return f"release {tag} has a malformed draft flag; refusing to mutate"
    if draft_value is not True:
        return f"release {tag} already exists and is not a draft; refusing to mutate it"
    return None


def _require_exact_draft(
    *,
    repo: str,
    tag: str,
    expected_source_sha: str,
) -> tuple[dict[str, object] | None, str | None]:
    """Require one exact unpublished draft and validate any existing tag target.

    GitHub normally does not create ``refs/tags/<tag>`` until a draft release is
    published.  The draft's exact ``target_commitish`` therefore binds its
    intended source while an absent tag ref is expected.  If the ref already
    exists, it must still peel to the same source SHA so drift remains blocking.
    """
    release, blocker = _query_release_listing(repo=repo, tag=tag)
    if blocker is not None:
        return None, blocker
    if release is None:
        return None, f"release {tag} does not exist as an unpublished draft"
    blocker = _release_target_blocker(release, tag=tag, source_sha=expected_source_sha)
    if blocker is not None:
        return None, blocker
    tag_target, tag_blocker = _resolve_tag_ref_target(repo=repo, tag=tag, allow_absent=True)
    if tag_blocker is not None:
        return None, tag_blocker
    if tag_target is not None and tag_target != expected_source_sha:
        return (
            None,
            f"tag {tag} resolves to {tag_target!r}, not the required "
            f"{expected_source_sha!r}; refusing to mutate",
        )
    return release, None


def _check_release_collision(
    *,
    repo: str,
    tag: str,
    expected_source_sha: str,
    dry_run: bool,
) -> tuple[str | None, bool]:
    """Fail closed when an existing release is not an exact unpublished draft."""
    if dry_run:
        return None, False
    release, blocker = _query_release_listing(repo=repo, tag=tag)
    if blocker is not None:
        return blocker, False
    if release is None:
        return None, False
    blocker = _release_target_blocker(release, tag=tag, source_sha=expected_source_sha)
    if blocker is not None:
        return blocker, False
    return None, True


def _parse_release_listing(
    stdout: str,
    *,
    tag: str,
) -> tuple[dict[str, object] | None, str | None]:
    """Return one exact-tag release from paginated ``gh api --slurp`` JSON."""
    try:
        pages = json.loads(stdout)
    except json.JSONDecodeError:
        return None, f"cannot parse release lookup for {tag}; refusing to create or upload"
    if not isinstance(pages, list) or any(not isinstance(page, list) for page in pages):
        return None, f"release lookup for {tag} has an invalid response shape"
    if any(not isinstance(release, dict) for page in pages for release in page):
        return None, f"release lookup for {tag} contains an invalid release record"
    matches = [release for page in pages for release in page if release.get("tag_name") == tag]
    if len(matches) > 1:
        return None, f"release lookup found multiple releases for tag {tag}; refusing to mutate"
    return (matches[0] if matches else None), None


def _check_tag_collision(*, repo: str, tag: str, dry_run: bool) -> str | None:
    """Fail closed when GitHub already has the tag planned for creation."""
    if dry_run:
        return None
    target, blocker = _resolve_tag_ref_target(repo=repo, tag=tag, allow_absent=True)
    if blocker is not None:
        return blocker
    if target is None:
        return None
    return f"tag {tag} already exists at target {target}; refusing to retarget or overwrite it"


def _validate_remote_asset(
    raw_asset: object,
    *,
    expected: Mapping[str, _LocalAsset],
    seen: set[str],
) -> str | None:
    """Validate one GitHub release asset against the local asset set."""
    if not isinstance(raw_asset, dict):
        return "release asset inventory contains a malformed asset"
    name = raw_asset.get("name")
    if not isinstance(name, str) or not name:
        return "release asset inventory contains an asset without a valid name"
    if name in seen:
        return f"release draft contains duplicate asset {name!r}; refusing to mutate"
    seen.add(name)
    local = expected.get(name)
    if local is None:
        return f"release draft contains unexpected asset {name!r}; refusing to mutate"
    if raw_asset.get("state") != "uploaded":
        return f"release draft asset {name!r} is not uploaded; refusing to mutate"
    size = raw_asset.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return f"release draft asset {name!r} has an invalid size; refusing to mutate"
    digest = raw_asset.get("digest")
    digest_match = (
        re.fullmatch(r"sha256:([0-9a-f]{64})", digest, re.I) if isinstance(digest, str) else None
    )
    if digest_match is None:
        return f"release draft asset {name!r} has an invalid SHA-256 digest; refusing to mutate"
    if size != local.size or digest_match.group(1).lower() != local.sha256:
        return f"release draft asset {name!r} differs from the local publication asset"
    return None


def _validate_remote_asset_inventory(
    release: Mapping[str, object],
    local_assets: tuple[_LocalAsset, ...],
) -> tuple[str | None, tuple[_LocalAsset, ...]]:
    """Validate remote assets and return only local assets still missing.

    Matching assets are safe to leave in place, while partial drafts are
    retryable by uploading only the absent assets.  Existing assets are never
    clobbered: extras, duplicate names, stale sizes, bad states, and digest
    mismatches block before an upload command is built.
    """
    if "assets" not in release:
        return "release asset inventory is missing the assets list", ()
    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        return "release asset inventory is not a JSON list", ()
    expected = {asset.name: asset for asset in local_assets}
    seen: set[str] = set()
    for raw_asset in raw_assets:
        blocker = _validate_remote_asset(raw_asset, expected=expected, seen=seen)
        if blocker is not None:
            return blocker, ()
    return None, tuple(asset for asset in local_assets if asset.name not in seen)


def _prepare_draft_plan(
    *,
    repo: str,
    tag: str,
    source_sha: str,
    summary: dict[str, object],
    execute_upload: bool,
    release_title: str | None,
    release_notes: str | None,
) -> tuple[list[str] | None, str]:
    """Prepare a draft command after checking live release/tag collisions."""
    title, notes = _resolve_release_identity(
        summary,
        tag=tag,
        release_title=release_title,
        release_notes=release_notes,
    )
    blocker, exists_at_target = _check_release_collision(
        repo=repo,
        tag=tag,
        expected_source_sha=source_sha,
        dry_run=not execute_upload,
    )
    if blocker is not None:
        raise SystemExit(f"Release draft admission blocked: {blocker}")
    if exists_at_target:
        return None, title
    blocker = _check_tag_collision(repo=repo, tag=tag, dry_run=not execute_upload)
    if blocker is not None:
        raise SystemExit(f"Release tag admission blocked: {blocker}")
    return (
        _build_draft_create_command(
            repo=repo,
            tag=tag,
            source_sha=source_sha,
            release_title=title,
            release_notes=notes,
        ),
        title,
    )


def _set_upload_inventory_payload(
    payload: dict[str, object],
    *,
    local_assets: tuple[_LocalAsset, ...],
    missing_assets: tuple[_LocalAsset, ...],
) -> None:
    """Record deterministic remote inventory results in the publication plan."""
    payload["remote_asset_count"] = len(local_assets) - len(missing_assets)
    payload["missing_upload_assets"] = [asset.name for asset in missing_assets]


def _admit_draft_assets(
    *,
    repo: str,
    tag: str,
    source_sha: str,
    local_assets: tuple[_LocalAsset, ...],
    error_prefix: str,
    readback_attempts: int = 1,
) -> tuple[dict[str, object], tuple[_LocalAsset, ...]]:
    """Re-read one exact draft and validate its assets before any upload."""
    missing_draft = f"release {tag} does not exist as an unpublished draft"
    release: dict[str, object] | None = None
    blocker: str | None = None
    for attempt in range(readback_attempts):
        release, blocker = _require_exact_draft(
            repo=repo,
            tag=tag,
            expected_source_sha=source_sha,
        )
        if blocker != missing_draft or attempt + 1 == readback_attempts:
            break
        logger.info(
            "Waiting for draft release readback tag={} repo={} attempt={}/{}",
            tag,
            repo,
            attempt + 1,
            readback_attempts,
        )
        time.sleep(_DRAFT_READBACK_DELAY_SECONDS)
    if blocker is not None or release is None:
        raise SystemExit(f"{error_prefix}: {blocker or 'missing release'}")
    blocker, missing_assets = _validate_remote_asset_inventory(release, local_assets)
    if blocker is not None:
        raise SystemExit(f"{error_prefix}: {blocker}")
    local_blocker = _validate_local_asset_snapshot(local_assets)
    if local_blocker is not None:
        raise SystemExit(f"{error_prefix}: {local_blocker}")
    return release, missing_assets


def _execute_release_upload(
    *,
    payload: dict[str, object],
    repo: str,
    tag: str,
    source_sha: str,
    create_draft: bool,
    draft_create_command: list[str] | None,
    local_assets: tuple[_LocalAsset, ...],
) -> None:
    """Create/reuse, revalidate, and safely upload a GitHub draft release."""
    if draft_create_command is not None:
        logger.info("Creating draft release for tag={} repo={}", tag, repo)
        subprocess.run(draft_create_command, check=True)
        error_prefix = "Release draft readback blocked after creation"
    else:
        error_prefix = (
            "Release draft admission blocked"
            if create_draft
            else "Release upload admission blocked"
        )
    _, missing_assets = _admit_draft_assets(
        repo=repo,
        tag=tag,
        source_sha=source_sha,
        local_assets=local_assets,
        error_prefix=error_prefix,
        readback_attempts=_DRAFT_READBACK_ATTEMPTS if draft_create_command is not None else 1,
    )
    _set_upload_inventory_payload(payload, local_assets=local_assets, missing_assets=missing_assets)
    if not missing_assets:
        payload["upload_command"] = []
        payload["upload_skipped"] = True
        return

    # Re-read both the release and the peeled tag immediately before building
    # the mutating command. This catches publication, retargeting, and asset
    # drift observed after the first preflight.
    _, missing_assets = _admit_draft_assets(
        repo=repo,
        tag=tag,
        source_sha=source_sha,
        local_assets=local_assets,
        error_prefix="Release upload readback blocked immediately before upload",
    )
    _set_upload_inventory_payload(payload, local_assets=local_assets, missing_assets=missing_assets)
    if not missing_assets:
        payload["upload_command"] = []
        payload["upload_skipped"] = True
        return
    upload_command = _build_upload_command(
        repo=repo,
        tag=tag,
        upload_assets=tuple(asset.path for asset in missing_assets),
    )
    payload["upload_command"] = upload_command
    logger.info("Executing release upload for tag={} repo={}", tag, repo)
    subprocess.run(upload_command, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Run guided publication workflow and return POSIX exit code."""
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.create_draft and not args.expected_source_sha:
        parser.error("--create-draft requires --expected-source-sha")
    if args.execute_upload and not args.expected_source_sha:
        parser.error("--execute-upload requires --expected-source-sha")
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
    upload_assets = _resolve_upload_assets(
        tag=str(args.tag),
        archive_path=archive_path,
        checksums_path=checksums_path,
        manifest_path=manifest_path,
    )
    local_asset_records: tuple[_LocalAsset, ...] = ()
    if args.execute_upload:
        if _ERRATUM_TAG_RE.fullmatch(str(args.tag)) is not None:
            _validate_erratum_custody(
                upload_assets[-1],
                archive_path=archive_path,
                checksums_path=checksums_path,
                manifest_path=manifest_path,
                source_sha=source_sha,
            )
        local_asset_records = _local_asset_records(upload_assets)
    payload = _build_release_payload(
        campaign_root=campaign_root,
        repo=str(args.repo),
        tag=str(args.tag),
        archive_path=archive_path,
        checksums_path=checksums_path,
        manifest_path=manifest_path,
        upload_assets=upload_assets,
        summary=summary,
    )
    payload["source_sha"] = source_sha

    draft_create_command: list[str] | None = None
    if args.create_draft:
        draft_create_command, title = _prepare_draft_plan(
            repo=str(args.repo),
            tag=str(args.tag),
            source_sha=source_sha,
            summary=summary,
            execute_upload=args.execute_upload,
            release_title=str(args.release_title) if args.release_title else None,
            release_notes=str(args.release_notes) if args.release_notes else None,
        )
        payload["expected_source_sha"] = source_sha
        payload["release_title"] = title
        if draft_create_command is not None:
            payload["draft_create_command"] = draft_create_command

    if args.execute_upload:
        _execute_release_upload(
            payload=payload,
            repo=str(args.repo),
            tag=str(args.tag),
            source_sha=source_sha,
            create_draft=args.create_draft,
            draft_create_command=draft_create_command,
            local_assets=local_asset_records,
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
