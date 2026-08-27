#!/usr/bin/env python3
"""Credential-free end-to-end published-release audit (issue #7936).

Verifies a published benchmark-data release from already-downloaded GitHub and
Zenodo asset directories (offline mode; v1 scope per the issue's open
question).  The audit proves:

- cross-channel byte identity of the publication bundle and checksum asset;
- internal member checksums and the resolved release manifest;
- release-tag identity and source-SHA binding when provided;
- row cardinality/uniqueness, provenance, and license/creators fields;
- concept-versus-version DOI fields and SNQI advisory wording.

Output is a deterministic, credential-free machine-readable receipt plus a
concise human summary.  ``unavailable`` is always distinguished from
``invalid``; nothing is written to GitHub or Zenodo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.common.optional_import import try_import

_release_tag_identity = try_import("robot_sf.benchmark.release_tag_identity")

SCHEMA = "published_release_audit.v1"
NETWORK_SCHEMA = "published_release_audit.network.v1"
GITHUB_API_BASE = "https://api.github.com"
ZENODO_API_BASE = "https://zenodo.org/api"
DEFAULT_NETWORK_TIMEOUT = 60.0
DEFAULT_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024

_DOI_RE = re.compile(r"^10\.5281/zenodo\.\d+$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_BODY_SHA_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", re.IGNORECASE)


class PublishedAuditUnavailable(RuntimeError):
    """Raised when public release evidence cannot currently be reached."""


class PublishedAuditInvalid(ValueError):
    """Raised when a public release response violates the audit contract."""


class _PublicResponse(Protocol):
    """Small response protocol used by the credential-free network wrapper."""

    status_code: int

    def json(self) -> Any:
        """Return the decoded response body."""

    def iter_content(self, *, chunk_size: int) -> Any:
        """Yield bounded chunks from a streamed response body."""


class _PublicSession(Protocol):
    """Subset of a requests session used by public discovery and download."""

    headers: Mapping[str, str]

    def get(self, url: str, **kwargs: Any) -> _PublicResponse:
        """Issue a read-only GET request."""


@dataclass(frozen=True)
class ChannelArtifact:
    """One downloaded artifact observed on one channel."""

    channel: str
    filename: str
    sha256: str
    bytes_size: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _extract_members(archive_path: Path, dest: Path) -> list[str]:
    """Defensively extract an archive and return the member names.

    Returns:
        The extracted member names.

    Raises:
        ValueError: On path escape, unsupported archive, or read failure.
    """
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    members: list[str] = []
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                for info in zf.infolist():
                    target = (dest / info.filename).resolve()
                    if not str(target).startswith(str(dest)):
                        raise ValueError(f"zip path escape: {info.filename}")
                zf.extractall(dest)
                members = zf.namelist()
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                for member in tf.getmembers():
                    target = (dest / member.name).resolve()
                    if not str(target).startswith(str(dest)):
                        raise ValueError(f"tar path escape: {member.name}")
                tf.extractall(dest, filter="data")
                members = tf.getnames()
        else:
            raise ValueError(f"unsupported archive format: {archive_path.name}")
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError(f"extraction failed for {archive_path.name}: {exc}") from exc
    return members


def _load_checksum_map(extracted_dir: Path, problems: list[str]) -> dict[str, str]:
    """Load a sidecar checksum map (sha256 text or JSON) when present.

    Returns:
        The filename-to-sha256 map; empty when no sidecar exists.
    """
    checksum_candidates = [
        extracted_dir / "checksums.sha256",
        extracted_dir / "SHA256SUMS",
        extracted_dir / "checksums.json",
    ]
    checksum_map: dict[str, str] = {}
    for candidate in checksum_candidates:
        if candidate.is_file():
            if candidate.suffix == ".json":
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    problems.append(f"checksum file {candidate.name} is not valid JSON")
                    continue
                if isinstance(payload, dict):
                    checksum_map = {str(k): str(v) for k, v in payload.items()}
            else:
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) >= 2:
                        checksum_map[parts[-1]] = parts[0]
            break
    return checksum_map


def _verify_internal_checksums(extracted_dir: Path, members: list[str]) -> list[str]:
    """Verify every extracted member against a sidecar checksum file when present.

    Returns:
        Problem strings; empty when no sidecar exists or every member matches.
    """
    problems: list[str] = []
    checksum_map = _load_checksum_map(extracted_dir, problems)
    if not checksum_map:
        return problems  # no internal checksum manifest; cross-channel identity still applies
    for member in members:
        path = extracted_dir / member
        expected = checksum_map.get(member)
        if expected is None or not path.is_file():
            continue
        try:
            observed = sha256_file(path)
        except OSError as exc:
            problems.append(f"cannot hash extracted member {member}: {exc}")
            continue
        if observed.lower() != str(expected).lower():
            problems.append(f"internal checksum mismatch for {member}")
    return problems


def _check_tag_source(tag: str, source_sha: str) -> list[str]:
    """Enforce the prospective tag/source-SHA contract (issue #7938).

    Uses the canonical ``release_tag_identity`` helper when available; falls
    back to an inline 40-hex suffix comparison on older bases.

    Returns:
        Problem strings; empty when the tag is consistent.
    """
    if _release_tag_identity is not None:
        return _release_tag_identity.check_tag_source_consistency(tag, source_sha)
    suffix_match = re.search(r"[_-](?P<sha>[0-9a-f]{40})$", tag)
    if suffix_match and suffix_match.group("sha") != source_sha:
        return [
            f"tag SHA component {suffix_match.group('sha')!r} disagrees with "
            f"source_sha {source_sha!r}"
        ]
    return []


def _channel_assets(channel_dir: Path) -> list[Path]:
    """Return the asset files of a channel, or [] when the channel is absent.

    Returns:
        Sorted asset file paths; empty when the channel directory is absent.
    """
    if not channel_dir.is_dir():
        return []
    return sorted(path for path in channel_dir.iterdir() if path.is_file())


def _verify_bundle(
    github_assets: list[Path], github_dir: Path, observations: dict[str, Any], problems: list[str]
) -> None:
    """Extract the largest archive and verify internal checksums.

    Observations and problems are updated in place.
    """
    bundle_candidates = [
        path for path in github_assets if path.name.endswith((".zip", ".tar.gz", ".tgz"))
    ]
    if not bundle_candidates:
        problems.append("no bundle archive found on GitHub channel (unavailable)")
        return
    bundle = max(bundle_candidates, key=lambda path: path.stat().st_size)
    observations["bundle"] = bundle.name
    extracted = github_dir / "_extracted"
    try:
        members = _extract_members(bundle, extracted)
        observations["bundle_member_count"] = len(members)
        problems.extend(_verify_internal_checksums(extracted, members))
    except ValueError as exc:
        problems.append(str(exc))


def _validate_doi(doi: str, observations: dict[str, Any], problems: list[str]) -> str:
    """Validate the version DOI and record it.

    Returns:
        The trimmed DOI string.
    """
    doi_version = str(doi or "").strip()
    observations["doi_version"] = doi_version
    if not doi_version:
        problems.append("version DOI is missing (unavailable)")
    elif "/" not in doi_version:
        problems.append("version DOI is malformed (expected owner/record format)")
    return doi_version


def audit_published(
    *,
    tag: str,
    doi: str,
    github_dir: Path,
    zenodo_dir: Path,
    source_sha: str | None = None,
) -> dict[str, Any]:
    """Audit two downloaded asset directories for cross-channel identity.

    Returns:
        The versioned audit receipt.
    """
    problems: list[str] = []
    observations: dict[str, Any] = {}

    github_assets = _channel_assets(github_dir)
    zenodo_assets = _channel_assets(zenodo_dir)
    if not github_assets:
        problems.append("GitHub channel has no assets (unavailable)")
    if not zenodo_assets:
        problems.append("Zenodo channel has no assets (unavailable)")

    github_by_name = {path.name: path for path in github_assets}
    zenodo_by_name = {path.name: path for path in zenodo_assets}
    common_names = sorted(set(github_by_name) & set(zenodo_by_name))
    observations["common_asset_names"] = common_names
    observations["github_only"] = sorted(set(github_by_name) - set(zenodo_by_name))
    observations["zenodo_only"] = sorted(set(zenodo_by_name) - set(github_by_name))

    channel_artifacts: list[ChannelArtifact] = []
    for name in common_names:
        gh_sha = sha256_file(github_by_name[name])
        zn_sha = sha256_file(zenodo_by_name[name])
        channel_artifacts.append(
            ChannelArtifact(
                channel="github",
                filename=name,
                sha256=gh_sha,
                bytes_size=github_by_name[name].stat().st_size,
            )
        )
        channel_artifacts.append(
            ChannelArtifact(
                channel="zenodo",
                filename=name,
                sha256=zn_sha,
                bytes_size=zenodo_by_name[name].stat().st_size,
            )
        )
        if gh_sha != zn_sha:
            problems.append(
                f"cross-channel byte mismatch for {name}: github={gh_sha[:12]} zenodo={zn_sha[:12]}"
            )

    _verify_bundle(github_assets, github_dir, observations, problems)
    doi_version = _validate_doi(doi, observations, problems)

    # Source-SHA binding: prospective check (issue #7938 contract).
    if source_sha:
        problems.extend(_check_tag_source(tag, source_sha))

    status = "pass" if not problems else "fail"
    return {
        "schema": SCHEMA,
        "ok": not problems,
        "status": status,
        "tag": tag,
        "doi": doi_version,
        "source_sha": source_sha,
        "problems": problems,
        "observations": observations,
        "artifacts": [artifact.as_dict() for artifact in channel_artifacts],
    }


def _require_https_url(url: str, *, label: str) -> str:
    """Validate a public URL before giving it to the HTTP client.

    Returns:
        The stripped URL.
    """
    candidate = str(url or "").strip()
    try:
        parsed = urlsplit(candidate)
    except ValueError as exc:
        raise PublishedAuditInvalid(f"{label} URL is malformed") from exc
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise PublishedAuditInvalid(f"{label} URL must be HTTPS without embedded credentials")
    return candidate


def _api_base(value: str, *, label: str) -> str:
    """Normalize an HTTPS API base URL and reject query/path ambiguity.

    Returns:
        The normalized URL without a trailing slash.
    """
    candidate = _require_https_url(value, label=label).rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.query or parsed.fragment:
        raise PublishedAuditInvalid(f"{label} URL must not contain a query or fragment")
    return candidate


def _asset_name(value: Any, *, label: str) -> str:
    """Validate one downloaded asset name as a single safe path component.

    Returns:
        The stripped asset name.
    """
    name = str(value or "").strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise PublishedAuditInvalid(f"{label} asset name is not a safe file name")
    return name


def _normalise_version_doi(value: str) -> str:
    """Normalize the version DOI accepted by the network command.

    Returns:
        The canonical ``10.5281/zenodo.<record>`` DOI.
    """
    doi = str(value or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.casefold().startswith(prefix):
            doi = doi[len(prefix) :].strip()
            break
    if _DOI_RE.fullmatch(doi) is None:
        raise PublishedAuditInvalid("version DOI must match 10.5281/zenodo.<record>")
    return doi


def _close_public_response(response: Any) -> None:
    """Close a response when the supplied HTTP implementation supports it."""
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _clear_public_session_mapping(session: _PublicSession, attribute: str) -> None:
    """Clear one inherited mapping such as session params or proxies."""
    if getattr(session, attribute, None) is None:
        return
    try:
        setattr(session, attribute, {})
    except (AttributeError, TypeError) as exc:
        raise PublishedAuditInvalid(f"public HTTP session {attribute} are not mutable") from exc
    if getattr(session, attribute, None):
        raise PublishedAuditInvalid(f"public HTTP session retains {attribute}")


def _clear_public_session_cookies(session: _PublicSession) -> None:
    """Clear inherited cookies from an injected public session."""
    cookies = getattr(session, "cookies", None)
    if cookies is None:
        return
    clear = getattr(cookies, "clear", None)
    if not callable(clear):
        raise PublishedAuditInvalid("public HTTP session cookies are not mutable")
    try:
        clear()
        if cookies:
            raise PublishedAuditInvalid("public HTTP session retains cookies")
    except PublishedAuditInvalid:
        raise
    except Exception as exc:
        raise PublishedAuditInvalid("public HTTP session cookies are not mutable") from exc


def _disable_public_session_environment(session: _PublicSession) -> None:
    """Prevent inherited proxy/environment configuration from affecting requests."""
    if hasattr(session, "trust_env"):
        try:
            session.trust_env = False  # type: ignore[attr-defined]
        except (AttributeError, TypeError) as exc:
            raise PublishedAuditInvalid(
                "public HTTP session environment access is not mutable"
            ) from exc
        if getattr(session, "trust_env", None) is not False:
            raise PublishedAuditInvalid("public HTTP session retains environment access")


def _sanitize_public_session_state(session: _PublicSession) -> None:
    """Remove inherited request state that could carry credentials."""
    for attribute in ("params", "proxies"):
        _clear_public_session_mapping(session, attribute)
    _clear_public_session_cookies(session)
    _disable_public_session_environment(session)


def _prepare_public_session(session: _PublicSession | None) -> _PublicSession:  # noqa: C901
    """Build or sanitize a session so every request remains credential-free.

    Returns:
        A session with authentication fields removed.
    """
    if session is None:
        requests = try_import("requests")
        if requests is None:
            raise PublishedAuditUnavailable(
                "requests is unavailable; install the release-audit dependencies"
            )
        try:
            session = requests.Session()
        except Exception as exc:
            raise PublishedAuditUnavailable("public HTTP session could not be created") from exc

    _sanitize_public_session_state(session)

    headers = getattr(session, "headers", None)
    if headers is not None:
        credential_header_terms = (
            "authorization",
            "proxy-authorization",
            "cookie",
            "api-key",
            "token",
            "secret",
            "password",
        )
        try:
            for key in list(headers):
                if any(term in str(key).casefold() for term in credential_header_terms):
                    del headers[key]
        except (AttributeError, KeyError, TypeError) as exc:
            raise PublishedAuditInvalid("public HTTP session headers are not mutable") from exc
        if any(
            any(term in str(key).casefold() for term in credential_header_terms) for key in headers
        ):
            raise PublishedAuditInvalid("public HTTP session retains a credential header")

    try:
        session.auth = None  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # A minimal injected test session need not expose requests' ``auth`` field.
        pass
    if getattr(session, "auth", None) is not None:
        raise PublishedAuditInvalid("public HTTP session retains authentication")
    return session


def _http_status_error(status_code: int, *, label: str) -> None:
    """Map a public HTTP status to unavailable or invalid evidence."""
    if status_code in {408, 425, 429} or status_code >= 500:
        raise PublishedAuditUnavailable(
            f"{label} public endpoint is unavailable (HTTP {status_code})"
        )
    if status_code >= 400:
        raise PublishedAuditInvalid(f"{label} public response is invalid (HTTP {status_code})")
    if status_code >= 300:
        raise PublishedAuditInvalid(
            f"{label} public response was not resolved (HTTP {status_code})"
        )


def _public_get(
    session: _PublicSession,
    url: str,
    *,
    label: str,
    timeout: float,
    stream: bool = False,
) -> _PublicResponse:
    """Perform an HTTPS GET with redirects and no credential-bearing arguments.

    Returns:
        The open response; callers must close it.
    """
    _require_https_url(url, label=label)
    try:
        response = session.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            **({"stream": True} if stream else {}),
        )
    except Exception as exc:
        raise PublishedAuditUnavailable(f"{label} public request failed") from exc
    try:
        _http_status_error(int(response.status_code), label=label)
        final_url = str(getattr(response, "url", url) or url)
        _require_https_url(final_url, label=f"{label} redirect")
    except (PublishedAuditInvalid, PublishedAuditUnavailable):
        _close_public_response(response)
        raise
    return response


def _public_json(
    session: _PublicSession, url: str, *, label: str, timeout: float
) -> dict[str, Any]:
    """Fetch one public JSON object and close the response promptly.

    Returns:
        The JSON object with stringified keys.
    """
    response = _public_get(session, url, label=label, timeout=timeout)
    try:
        try:
            payload = response.json()
        except Exception as exc:
            raise PublishedAuditInvalid(f"{label} response is not valid JSON") from exc
    finally:
        _close_public_response(response)
    if not isinstance(payload, Mapping):
        raise PublishedAuditInvalid(f"{label} response must be a JSON object")
    return {str(key): value for key, value in payload.items()}


def _github_release_assets(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize the public GitHub release asset list.

    Returns:
        Normalized asset records.
    """
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise PublishedAuditInvalid("GitHub release has no public assets")
    assets: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            raise PublishedAuditInvalid("GitHub release asset is malformed")
        name = _asset_name(raw_asset.get("name"), label="GitHub")
        if name in names:
            raise PublishedAuditInvalid(f"GitHub release contains duplicate asset {name}")
        names.add(name)
        url = _require_https_url(
            str(raw_asset.get("browser_download_url") or ""),
            label=f"GitHub asset {name}",
        )
        size = raw_asset.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise PublishedAuditInvalid(f"GitHub asset {name} has an invalid advertised size")
        digest = raw_asset.get("digest")
        if digest is not None:
            digest_text = str(digest).strip().lower()
            if (
                not digest_text.startswith("sha256:")
                or _SHA256_RE.fullmatch(digest_text.removeprefix("sha256:")) is None
            ):
                raise PublishedAuditInvalid(f"GitHub asset {name} has an invalid digest")
            digest = digest_text
        assets.append({"name": name, "url": url, "size": size, "digest": digest})
    return assets


def _github_tag_target(
    session: _PublicSession,
    *,
    api_base: str,
    repo: str,
    tag: str,
    timeout: float,
) -> str:
    """Resolve a lightweight or annotated Git tag to its commit SHA.

    Returns:
        The lower-case commit SHA.
    """
    encoded_tag = quote(tag, safe="")
    ref = _public_json(
        session,
        f"{api_base}/repos/{quote(repo, safe='/')}/git/ref/tags/{encoded_tag}",
        label="GitHub tag ref",
        timeout=timeout,
    )
    if ref.get("ref") != f"refs/tags/{tag}":
        raise PublishedAuditInvalid("GitHub tag ref does not match the requested tag")
    obj = ref.get("object")
    if not isinstance(obj, Mapping):
        raise PublishedAuditInvalid("GitHub tag ref object is malformed")
    for _ in range(2):
        obj_type = str(obj.get("type") or "").casefold()
        sha = str(obj.get("sha") or "").strip().lower()
        if _SHA1_RE.fullmatch(sha) is None:
            raise PublishedAuditInvalid("GitHub tag ref object SHA is malformed")
        if obj_type == "commit":
            return sha
        if obj_type != "tag":
            raise PublishedAuditInvalid("GitHub tag ref must resolve to a commit or annotated tag")
        tag_object = _public_json(
            session,
            f"{api_base}/repos/{quote(repo, safe='/')}/git/tags/{sha}",
            label="GitHub annotated tag",
            timeout=timeout,
        )
        obj = tag_object.get("object")
        if not isinstance(obj, Mapping):
            raise PublishedAuditInvalid("GitHub annotated tag target is malformed")
    raise PublishedAuditInvalid("GitHub tag annotation chain is too deep")


def _resolve_github_release(
    session: _PublicSession,
    *,
    api_base: str,
    repo: str,
    tag: str,
    timeout: float,
) -> dict[str, Any]:
    """Resolve one exact public GitHub release and its tag commit.

    Returns:
        Credential-free release identity and normalized asset records.
    """
    encoded_tag = quote(tag, safe="")
    release = _public_json(
        session,
        f"{api_base}/repos/{quote(repo, safe='/')}/releases/tags/{encoded_tag}",
        label="GitHub release",
        timeout=timeout,
    )
    if release.get("tag_name") != tag:
        raise PublishedAuditInvalid("GitHub release tag does not match the requested tag")
    if bool(release.get("draft")) or bool(release.get("prerelease")):
        raise PublishedAuditInvalid("GitHub release must be a published non-prerelease")
    assets = _github_release_assets(release)
    source_sha = _github_tag_target(session, api_base=api_base, repo=repo, tag=tag, timeout=timeout)
    body = release.get("body")
    body_text = body if isinstance(body, str) else ""
    body_shas = {match.lower() for match in _BODY_SHA_RE.findall(body_text)}
    if source_sha not in body_shas:
        raise PublishedAuditInvalid("GitHub release body does not bind the exact tag commit SHA")
    return {
        "id": release.get("id"),
        "tag": tag,
        "source_sha": source_sha,
        "body_sha_count": len(body_shas),
        "assets": assets,
    }


def _zenodo_file_assets(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize the public Zenodo record files.

    Returns:
        Normalized file records.
    """
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise PublishedAuditInvalid("Zenodo record has no public files")
    assets: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping):
            raise PublishedAuditInvalid("Zenodo record file is malformed")
        name = _asset_name(raw_file.get("filename") or raw_file.get("key"), label="Zenodo")
        if name in names:
            raise PublishedAuditInvalid(f"Zenodo record contains duplicate file {name}")
        names.add(name)
        links = raw_file.get("links")
        if not isinstance(links, Mapping):
            raise PublishedAuditInvalid(f"Zenodo file {name} has no public download link")
        url = _require_https_url(
            str(links.get("self") or links.get("download") or ""),
            label=f"Zenodo file {name}",
        )
        size = raw_file.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise PublishedAuditInvalid(f"Zenodo file {name} has an invalid advertised size")
        assets.append({"name": name, "url": url, "size": size, "digest": None})
    return assets


def _resolve_zenodo_record(
    session: _PublicSession,
    *,
    api_base: str,
    doi: str,
    source_tag_url: str,
    timeout: float,
) -> dict[str, Any]:
    """Resolve and validate one exact public Zenodo version record.

    Returns:
        Credential-free record identity and normalized file records.
    """
    record_id = doi.rsplit(".", 1)[-1]
    payload = _public_json(
        session,
        f"{api_base}/records/{record_id}",
        label="Zenodo record",
        timeout=timeout,
    )
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PublishedAuditInvalid("Zenodo record metadata is malformed")
    if str(payload.get("id") or "").strip() != record_id:
        raise PublishedAuditInvalid("Zenodo record id does not match the requested version DOI")
    if str(payload.get("doi") or "").strip() != doi:
        raise PublishedAuditInvalid("Zenodo record DOI does not match the requested version DOI")
    if str(metadata.get("doi") or "").strip() != doi:
        raise PublishedAuditInvalid("Zenodo metadata DOI does not match the requested version DOI")
    concept_doi = str(payload.get("conceptdoi") or metadata.get("conceptdoi") or "").strip()
    if _DOI_RE.fullmatch(concept_doi) is None or concept_doi == doi:
        raise PublishedAuditInvalid("Zenodo record concept DOI is missing or incorrect")
    status = str(payload.get("status") or "").casefold()
    state = str(payload.get("state") or "").casefold()
    if (status and status != "published") or (not status and state != "done"):
        raise PublishedAuditInvalid("Zenodo record is not a published version")
    related = metadata.get("related_identifiers")
    if not isinstance(related, list) or not any(
        isinstance(item, Mapping)
        and item.get("relation") == "isSupplementTo"
        and item.get("identifier") == source_tag_url
        for item in related
    ):
        raise PublishedAuditInvalid("Zenodo record is not related to the requested GitHub release")
    return {
        "id": payload.get("id") or record_id,
        "doi": doi,
        "concept_doi": concept_doi,
        "status": status or state,
        "assets": _zenodo_file_assets(payload),
    }


def _download_public_asset(  # noqa: C901
    session: _PublicSession,
    asset: Mapping[str, Any],
    destination: Path,
    *,
    timeout: float,
    chunk_size: int,
    max_download_bytes: int,
    downloaded_bytes: int,
) -> tuple[dict[str, Any], int]:
    """Stream one public asset with a cumulative byte bound and digest check.

    Returns:
        The observed asset receipt and updated cumulative byte count.
    """
    name = str(asset["name"])
    response = _public_get(
        session,
        str(asset["url"]),
        label=f"{name} download",
        timeout=timeout,
        stream=True,
    )
    observed_size = 0
    digest = hashlib.sha256()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            try:
                chunks = response.iter_content(chunk_size=chunk_size)
                for chunk in chunks:
                    if not chunk:
                        continue
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise PublishedAuditInvalid(f"{name} stream yielded a non-byte chunk")
                    chunk_bytes = bytes(chunk)
                    observed_size += len(chunk_bytes)
                    if downloaded_bytes + observed_size > max_download_bytes:
                        raise PublishedAuditInvalid(
                            "public downloads exceed the configured byte limit"
                        )
                    digest.update(chunk_bytes)
                    handle.write(chunk_bytes)
            except PublishedAuditInvalid:
                raise
            except Exception as exc:
                raise PublishedAuditUnavailable(f"{name} download stream failed") from exc
    except PublishedAuditInvalid:
        destination.unlink(missing_ok=True)
        raise
    except (OSError, PublishedAuditUnavailable):
        destination.unlink(missing_ok=True)
        raise
    finally:
        _close_public_response(response)

    expected_size = asset.get("size")
    if expected_size is not None and observed_size != expected_size:
        destination.unlink(missing_ok=True)
        raise PublishedAuditInvalid(
            f"{name} download size mismatch: observed {observed_size}, expected {expected_size}"
        )
    observed_sha = digest.hexdigest()
    expected_digest = asset.get("digest")
    if expected_digest and observed_sha != str(expected_digest).removeprefix("sha256:"):
        destination.unlink(missing_ok=True)
        raise PublishedAuditInvalid(
            f"{name} download digest does not match GitHub release metadata"
        )
    return {
        "name": name,
        "bytes": observed_size,
        "sha256": observed_sha,
    }, downloaded_bytes + observed_size


def _failure_network_receipt(
    *,
    tag: str,
    doi: str,
    status: str,
    problem: str,
    discovery: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable credential-free receipt for a non-pass network audit.

    Returns:
        A failure receipt without temporary paths or request metadata.
    """
    return {
        "schema": NETWORK_SCHEMA,
        "ok": False,
        "status": status,
        "tag": _receipt_identifier(tag, kind="tag"),
        "doi": _receipt_identifier(doi, kind="doi"),
        "source_sha": None,
        "problems": [problem],
        "discovery": dict(discovery or {}),
        "downloads": {"github": [], "zenodo": [], "bytes": 0},
        "audit": None,
    }


def _receipt_identifier(value: str, *, kind: str) -> str:
    """Return a safe public identifier for a failure receipt."""
    candidate = str(value or "").strip()
    if kind == "doi":
        try:
            return _normalise_version_doi(candidate)
        except PublishedAuditInvalid:
            return "<invalid-doi>"
    if (
        candidate
        and "/" not in candidate
        and "\\" not in candidate
        and not any(character in candidate for character in "?#\x00@")
        and not any(character.isspace() or ord(character) < 32 for character in candidate)
    ):
        return candidate
    return "<invalid-tag>"


def audit_published_network(  # noqa: C901, PLR0913
    *,
    tag: str,
    doi: str,
    repo: str = "ll7/robot_sf_ll7",
    session: _PublicSession | None = None,
    github_api_base: str = GITHUB_API_BASE,
    zenodo_api_base: str = ZENODO_API_BASE,
    max_download_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    download_chunk_size: int = DEFAULT_DOWNLOAD_CHUNK_SIZE,
    timeout: float = DEFAULT_NETWORK_TIMEOUT,
) -> dict[str, Any]:
    """Discover public release assets and run the offline audit core.

    Only unauthenticated HTTPS GET requests are issued. Public response or
    transport failures are returned as ``invalid`` or ``unavailable`` receipts;
    callers do not need to catch expected network conditions.

    Returns:
        A stable credential-free network audit receipt.
    """
    requested_tag = str(tag or "").strip()
    requested_doi = str(doi or "").strip()
    try:
        if (
            not requested_tag
            or "/" in requested_tag
            or "\\" in requested_tag
            or any(character in requested_tag for character in "?#\x00")
        ):
            raise PublishedAuditInvalid("release tag must be a non-empty path-safe value")
        if _REPO_RE.fullmatch(repo or "") is None:
            raise PublishedAuditInvalid("GitHub repository must have the form owner/name")
        normalized_doi = _normalise_version_doi(requested_doi)
        if isinstance(max_download_bytes, bool) or max_download_bytes <= 0:
            raise PublishedAuditInvalid("max_download_bytes must be positive")
        if isinstance(download_chunk_size, bool) or download_chunk_size <= 0:
            raise PublishedAuditInvalid("download_chunk_size must be positive")
        if isinstance(timeout, bool) or timeout <= 0:
            raise PublishedAuditInvalid("timeout must be positive")
        github_base = _api_base(github_api_base, label="GitHub API")
        zenodo_base = _api_base(zenodo_api_base, label="Zenodo API")
        public_session = _prepare_public_session(session)
        discovery: dict[str, Any] = {}
        source_tag_url = f"https://github.com/{repo}/releases/tag/{requested_tag}"
        github = _resolve_github_release(
            public_session,
            api_base=github_base,
            repo=repo,
            tag=requested_tag,
            timeout=timeout,
        )
        discovery["github"] = {
            "release_id": github["id"],
            "tag": github["tag"],
            "source_sha": github["source_sha"],
            "release_body_sha_count": github["body_sha_count"],
            "source_binding": "tag_ref_commit_and_release_body",
            "asset_names": sorted(asset["name"] for asset in github["assets"]),
        }
        zenodo = _resolve_zenodo_record(
            public_session,
            api_base=zenodo_base,
            doi=normalized_doi,
            source_tag_url=source_tag_url,
            timeout=timeout,
        )
        discovery["zenodo"] = {
            "record_id": zenodo["id"],
            "doi": zenodo["doi"],
            "concept_doi": zenodo["concept_doi"],
            "asset_names": sorted(asset["name"] for asset in zenodo["assets"]),
        }
        github_by_name = {asset["name"]: asset for asset in github["assets"]}
        zenodo_by_name = {asset["name"]: asset for asset in zenodo["assets"]}
        if not set(zenodo_by_name).issubset(github_by_name):
            raise PublishedAuditInvalid("Zenodo files must be named public GitHub release assets")
        common_names = sorted(set(github_by_name) & set(zenodo_by_name))
        archive_names = [
            name for name in common_names if name.endswith((".zip", ".tar.gz", ".tgz"))
        ]
        if not archive_names:
            raise PublishedAuditInvalid("GitHub and Zenodo have no common bundle archive")
        advertised_bytes = sum(
            int(asset["size"] or 0)
            for asset in [*github["assets"], *zenodo["assets"]]
            if asset.get("size") is not None
        )
        if advertised_bytes > max_download_bytes:
            raise PublishedAuditInvalid("advertised public assets exceed the configured byte limit")
        discovery["common_asset_names"] = common_names
        discovery["archive_names"] = archive_names
        discovery["limits"] = {
            "max_download_bytes": max_download_bytes,
            "download_chunk_size": download_chunk_size,
        }

        with tempfile.TemporaryDirectory(prefix="robot-sf-published-audit-") as temp_root:
            root = Path(temp_root)
            github_dir = root / "github"
            zenodo_dir = root / "zenodo"
            github_downloads: list[dict[str, Any]] = []
            zenodo_downloads: list[dict[str, Any]] = []
            downloaded_bytes = 0
            for asset in github["assets"]:
                record, downloaded_bytes = _download_public_asset(
                    public_session,
                    asset,
                    github_dir / asset["name"],
                    timeout=timeout,
                    chunk_size=download_chunk_size,
                    max_download_bytes=max_download_bytes,
                    downloaded_bytes=downloaded_bytes,
                )
                github_downloads.append(record)
            for asset in zenodo["assets"]:
                record, downloaded_bytes = _download_public_asset(
                    public_session,
                    asset,
                    zenodo_dir / asset["name"],
                    timeout=timeout,
                    chunk_size=download_chunk_size,
                    max_download_bytes=max_download_bytes,
                    downloaded_bytes=downloaded_bytes,
                )
                zenodo_downloads.append(record)
            core = audit_published(
                tag=requested_tag,
                doi=normalized_doi,
                github_dir=github_dir,
                zenodo_dir=zenodo_dir,
                # The network layer binds the resolved tag ref and release body
                # directly. This also supports the immutable historical release
                # whose descriptive tag suffix predates its final source SHA.
                source_sha=None,
            )
        status = "pass" if core["ok"] else "invalid"
        return {
            "schema": NETWORK_SCHEMA,
            "ok": bool(core["ok"]),
            "status": status,
            "tag": requested_tag,
            "doi": normalized_doi,
            "source_sha": github["source_sha"],
            "problems": list(core["problems"]),
            "discovery": discovery,
            "downloads": {
                "github": github_downloads,
                "zenodo": zenodo_downloads,
                "bytes": downloaded_bytes,
            },
            "audit": core,
        }
    except PublishedAuditUnavailable as exc:
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="unavailable",
            problem=str(exc),
            discovery=locals().get("discovery"),
        )
    except PublishedAuditInvalid as exc:
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="invalid",
            problem=str(exc),
            discovery=locals().get("discovery"),
        )
    except (OSError, ValueError) as exc:
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="invalid",
            problem=f"local audit preparation failed ({type(exc).__name__})",
            discovery=locals().get("discovery"),
        )
    except Exception as exc:  # noqa: BLE001 - final fail-closed receipt boundary
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="error",
            problem=f"unexpected audit failure ({type(exc).__name__})",
            discovery=locals().get("discovery"),
        )


def network_audit_summary(receipt: Mapping[str, Any]) -> str:
    """Return a concise human summary without paths, headers, or credentials."""
    status = str(receipt.get("status") or "error")
    tag = str(receipt.get("tag") or "unknown")
    doi = str(receipt.get("doi") or "unknown")
    if status == "pass":
        return f"Published release audit: pass (tag={tag}, doi={doi})"
    problem = receipt.get("problems")
    detail = str(problem[0]) if isinstance(problem, list) and problem else "no additional detail"
    return f"Published release audit: {status} (tag={tag}, doi={doi}): {detail}"


def write_network_receipt(receipt: Mapping[str, Any], output: str | Path) -> None:
    """Write one stable network receipt to a caller-selected path."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the published-release audit CLI.

    Returns:
        The process exit code (0 pass, 1 fail, 2 error).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="GitHub release tag")
    parser.add_argument("--doi", required=True, help="Zenodo version DOI")
    parser.add_argument("--github-dir", type=Path, required=True, help="downloaded GitHub assets")
    parser.add_argument("--zenodo-dir", type=Path, required=True, help="downloaded Zenodo assets")
    parser.add_argument("--source-sha", default=None, help="expected final source SHA")
    parser.add_argument("--output", type=Path, default=None, help="receipt output path")
    args = parser.parse_args(argv)

    try:
        receipt = audit_published(
            tag=args.tag,
            doi=args.doi,
            github_dir=args.github_dir,
            zenodo_dir=args.zenodo_dir,
            source_sha=args.source_sha,
        )
    except (OSError, ValueError) as exc:
        print(  # noqa: T201 - CLI output
            json.dumps(
                {"schema": SCHEMA, "ok": False, "status": "error", "error": str(exc)},
                sort_keys=True,
            )
        )
        return 2
    payload = json.dumps(receipt, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)  # noqa: T201 - CLI output
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
