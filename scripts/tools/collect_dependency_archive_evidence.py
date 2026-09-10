#!/usr/bin/env python3
"""Generic, non-executing dependency archive evidence collector proposal.

This task-local proposal deliberately derives ownership and row identity from
the supplied immutable batch manifest.  It downloads only target-compatible
artifacts, verifies bytes before bounded archive-member reads, and emits
unresolved interpretation questions rather than rights conclusions.
"""

from __future__ import annotations

import argparse
import email
import email.policy
import hashlib
import json
import re
import stat
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_BATCH_DOWNLOAD_BYTES = 3 * 1024 * 1024 * 1024
MAX_MEMBER_COUNT = 50_000
MAX_MEMBER_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_EVIDENCE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_EVIDENCE_TOTAL_BYTES = 64 * 1024 * 1024
MAX_METADATA_JSON_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_JSON_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_MEMBER_COUNT = 50_000
CHUNK_BYTES = 1024 * 1024
MANIFEST_SCHEMA = "robot_sf.p04.p05_batch.v1"
PYPI_REGISTRY_URL = "https://pypi.org/simple"
PYPI_ARCHIVE_HOSTS = frozenset({"files.pythonhosted.org", "pypi.org"})
KNOWN_SOURCE_TYPES = frozenset({"editable", "registry"})
CANONICAL_OWNER_REPOSITORY = "ll7/robot_sf_ll7"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]*$")
ARTIFACT_KINDS = frozenset({"sdist", "wheel"})
METADATA_RE = re.compile(r"(?:^|/)[^/]+\.dist-info/metadata$|(?:^|/)pkg-info$", re.IGNORECASE)
LICENSE_RE = re.compile(r"(?:^|/)(?:license|licence|copying|notice)(?:[^/]*)$", re.IGNORECASE)
THIRD_PARTY_RE = re.compile(r"(?:^|/)(?:third[-_ ]party|_vendor|vendor)(?:/|$)", re.IGNORECASE)


def now() -> str:
    """Return a UTC timestamp suitable for run metadata."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write deterministic, UTF-8 JSON beneath the caller-owned output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_bounded_bytes(path: Path, limit: int, description: str) -> bytes:
    """Read one input file without allowing its in-memory representation to grow unbounded."""
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{description} exceeds {limit} byte bound")
    return data


def _https_url(value: object) -> bool:
    """Return whether a value is an absolute HTTPS URL."""
    if not isinstance(value, str):
        return False
    try:
        parsed = urllib.parse.urlsplit(value)
        return (
            parsed.scheme.lower() == "https"
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment
        )
    except ValueError:
        return False


def _pypi_archive_url(value: object) -> bool:
    """Return whether an archive URL stays on the public PyPI file hosts."""
    if not _https_url(value):
        return False
    try:
        parsed = urllib.parse.urlsplit(str(value))
        return (
            parsed.hostname.lower() in PYPI_ARCHIVE_HOSTS
            and parsed.port in {None, 443}
            and not parsed.query
        )
    except (AttributeError, ValueError):
        return False


def _pypi_json_url(value: object) -> bool:
    """Return whether a registry URL stays on the canonical HTTPS PyPI host."""
    if not _https_url(value):
        return False
    try:
        parsed = urllib.parse.urlsplit(str(value))
        return (
            parsed.hostname.lower() == "pypi.org"
            and parsed.port in {None, 443}
            and not parsed.query
        )
    except (AttributeError, ValueError):
        return False


def _safe_output_path(root: Path, candidate: Path) -> bool:
    """Return whether a cache path stays below ``root`` without symlink hops."""
    root_path = root.resolve()
    candidate_path = candidate.absolute()
    try:
        candidate_path.relative_to(root_path)
    except ValueError:
        return False
    current = candidate_path
    while current != root_path:
        if current.is_symlink():
            return False
        current = current.parent
    return True


def _relative_output_path(path: Path, root: Path) -> str:
    """Return a stable output-relative path without exposing host directories."""
    try:
        return path.resolve(strict=False).relative_to(root.resolve()).as_posix()
    except ValueError:
        return "<external-local-path>"


def _input_display_path(path: Path, output: Path) -> str:
    """Return a privacy-safe input path label; the byte digest remains authoritative."""
    if _safe_output_path(output, path):
        return _relative_output_path(path, output)
    return f"<external-input>/{path.name}"


def _redact_cache_record(record: dict[str, Any], output: Path) -> dict[str, Any]:
    """Replace local cache paths in a durable network observation with relative labels."""
    redacted = dict(record)
    if isinstance(record.get("path"), str):
        redacted["path"] = _relative_output_path(Path(record["path"]), output)
    if isinstance(record.get("error"), str):
        redacted["error"] = record["error"].replace(str(output), "<output>")
    return redacted


def _canonical_name(value: str) -> str:
    """Normalize a distribution name using the PEP 503 spelling."""
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _safe_relative_path(value: object) -> bool:
    """Return whether a manifest path is a canonical relative POSIX path."""
    if not isinstance(value, str) or not value or "\\" in value or "//" in value:
        return False
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and (not path.parts or ":" not in path.parts[0])
    )


def _validate_manifest(  # noqa: C901, PLR0912, PLR0915 - closed batch contract checks
    manifest: dict[str, Any], args: argparse.Namespace
) -> int:
    """Validate the immutable batch contract before any network or cache read."""
    issues: list[str] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        issues.append("unsupported batch manifest schema")
    if manifest.get("profile") != args.expected_profile:
        issues.append(
            f"manifest profile {manifest.get('profile')!r} != requested {args.expected_profile!r}"
        )
    owner_issue = manifest.get("owner_issue")
    if (
        not isinstance(owner_issue, int)
        or isinstance(owner_issue, bool)
        or owner_issue != args.expected_owner_issue
    ):
        issues.append("manifest owner_issue does not match requested owner")
    owner_url = manifest.get("owner_url")
    expected_owner_url = f"https://github.com/{CANONICAL_OWNER_REPOSITORY}/issues/{owner_issue}"
    if owner_url != expected_owner_url:
        issues.append("manifest owner_url is not an HTTPS issue URL for owner_issue")
    for field, value in (
        ("audit_source_sha", manifest.get("audit_source_sha")),
        ("candidate_commit_sha", manifest.get("candidate_commit_sha")),
        ("candidate_tree_sha", manifest.get("candidate_tree_sha")),
    ):
        if not _sha(value, SHA1_RE):
            issues.append(f"manifest {field} is not a lowercase SHA-1")
    expected_ids = {
        "audit_source_sha": args.expected_audit_source_sha,
        "candidate_commit_sha": args.expected_candidate_commit_sha,
        "candidate_tree_sha": args.expected_candidate_tree_sha,
    }
    for field, expected in expected_ids.items():
        if expected is not None and manifest.get(field) != expected:
            issues.append(f"manifest {field} does not match requested identity")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        issues.append("manifest members must be a non-empty list")
        members = []
    member_count = manifest.get("member_count")
    if not isinstance(member_count, int) or isinstance(member_count, bool):
        issues.append("manifest member_count must be a typed integer")
    elif member_count != len(members):
        issues.append("manifest member_count does not match members")
    if (
        isinstance(member_count, int)
        and not isinstance(member_count, bool)
        and member_count > MAX_MANIFEST_MEMBER_COUNT
    ) or len(members) > MAX_MANIFEST_MEMBER_COUNT:
        issues.append("manifest member count exceeds bounded collector limit")
    package_ids: set[str] = set()
    identity_ids: set[str] = set()
    for index, member in enumerate(members):
        prefix = f"member[{index}]"
        if not isinstance(member, dict):
            issues.append(f"{prefix} is not an object")
            continue
        name = member.get("name")
        version = member.get("version")
        package_id = member.get("package_id")
        identity = member.get("identity_sha256")
        if not isinstance(name, str) or PACKAGE_NAME_RE.fullmatch(name) is None:
            issues.append(f"{prefix} has an invalid name")
            name = "?"
        if member.get("normalized_name") != _canonical_name(name):
            issues.append(f"{prefix} normalized_name is not the canonical package name")
        if version is not None and (
            not isinstance(version, str) or VERSION_RE.fullmatch(version) is None
        ):
            issues.append(f"{prefix} has a non-string version")
        if not isinstance(package_id, str) or "@" not in package_id or "#" not in package_id:
            issues.append(f"{prefix} has an invalid package_id")
        else:
            expected_prefix = (
                f"{_canonical_name(name)}@{version if version is not None else 'editable'}#"
            )
            if not package_id.startswith(expected_prefix):
                issues.append(f"{prefix} package_id is not bound to name/version")
            else:
                suffix = package_id.rsplit("#", 1)[-1]
                if not re.fullmatch(r"[0-9a-f]{16,64}", suffix) or not str(identity).startswith(
                    suffix
                ):
                    issues.append(f"{prefix} package_id suffix is not bound to identity_sha256")
            if package_id in package_ids:
                issues.append(f"duplicate package_id: {package_id}")
            package_ids.add(package_id)
        if not _sha(identity, SHA256_RE):
            issues.append(f"{prefix} identity_sha256 is invalid")
        elif identity in identity_ids:
            issues.append(f"duplicate identity_sha256: {identity}")
        identity_ids.add(str(identity))
        if member.get("lockfile") != "uv.lock":
            issues.append(f"{prefix} lockfile must be the typed path uv.lock")
        assignment = member.get("assignment")
        if not isinstance(assignment, dict):
            issues.append(f"{prefix} has no assignment object")
        else:
            if (
                assignment.get("owner_issue") != owner_issue
                or assignment.get("owner_url") != expected_owner_url
            ):
                issues.append(f"{prefix} assignment owner is not bound to the manifest owner")
            if assignment.get("owner_url") != owner_url:
                issues.append(f"{prefix} assignment owner_url differs from manifest owner_url")
        if member.get("selected_profiles") != [args.expected_profile]:
            issues.append(f"{prefix} selected_profiles must be exactly the requested profile")
        if (
            not isinstance(member.get("profiles"), list)
            or not member.get("profiles")
            or not all(isinstance(item, str) and item for item in member.get("profiles", []))
        ):
            issues.append(f"{prefix} profiles are not a non-empty string list")
        source_type = member.get("source_type")
        if source_type not in KNOWN_SOURCE_TYPES:
            issues.append(f"{prefix} source_type is unknown")
        elif source_type == "registry":
            source = member.get("source")
            if not isinstance(source, dict) or source.get("registry") != PYPI_REGISTRY_URL:
                issues.append(f"{prefix} registry source must be the canonical PyPI URL")
            if not isinstance(version, str):
                issues.append(f"{prefix} registry version must be a typed string")
        else:
            source = member.get("source")
            if not isinstance(source, dict) or not _safe_relative_path(source.get("editable")):
                issues.append(f"{prefix} editable source must be a safe relative path")
        artifacts = member.get("artifacts")
        if not isinstance(artifacts, list):
            issues.append(f"{prefix} artifacts must be a list")
            artifacts = []
        artifact_names: set[str] = set()
        for artifact_index, artifact in enumerate(artifacts):
            artifact_prefix = f"{prefix}.artifact[{artifact_index}]"
            if not isinstance(artifact, dict):
                issues.append(f"{artifact_prefix} is not an object")
                continue
            filename = artifact.get("filename")
            kind = artifact.get("kind")
            size = artifact.get("size")
            if (
                not isinstance(filename, str)
                or ARTIFACT_NAME_RE.fullmatch(filename) is None
                or "/" in filename
                or "\\" in filename
            ):
                issues.append(f"{artifact_prefix} filename is unsafe")
            filename_key = filename if isinstance(filename, str) else repr(filename)
            if filename_key in artifact_names:
                issues.append(f"{artifact_prefix} duplicates filename")
            artifact_names.add(filename_key)
            if kind not in ARTIFACT_KINDS:
                issues.append(f"{artifact_prefix} has unsupported kind")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                issues.append(f"{artifact_prefix} size is not a positive typed integer")
            if not _sha(artifact.get("sha256"), SHA256_RE):
                issues.append(f"{artifact_prefix} sha256 is invalid")
            if not _pypi_archive_url(artifact.get("url")):
                issues.append(
                    f"{artifact_prefix} URL must be HTTPS on an approved PyPI archive host"
                )
            tags = artifact.get("platform_tags", [])
            if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
                issues.append(f"{artifact_prefix} platform_tags are invalid")
        if member.get("source_type") == "registry" and version is not None and not artifacts:
            issues.append(f"{prefix} registry row has no artifacts")
    if issues:
        raise ValueError("invalid immutable dependency batch: " + "; ".join(issues))
    return int(owner_issue)


def safe_member_name(name: str) -> bool:
    """Return whether an archive member path is safe to inspect in place."""
    if not isinstance(name, str) or not name or "\\" in name or "//" in name:
        return False
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        return False
    path = PurePosixPath(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and (not path.parts or ":" not in path.parts[0])
    )


def _zip_info_is_symlink(info: zipfile.ZipInfo) -> bool:
    """Return whether a ZIP member advertises a POSIX symlink type."""
    mode = (info.external_attr >> 16) & 0o170000
    return stat.S_ISLNK(mode)


def basename(name: str) -> str:
    """Return the final POSIX path component."""
    return name.rsplit("/", 1)[-1]


def member_kind(name: str) -> str | None:
    """Classify a member that may contain metadata, notices, or bundled paths."""
    lower = name.lower()
    if METADATA_RE.search(lower):
        return "metadata"
    if LICENSE_RE.search(lower):
        return "license_or_notice"
    if THIRD_PARTY_RE.search(lower) or basename(lower).startswith("third_party"):
        return "bundled_component_path"
    return None


def text_summary(data: bytes | None) -> dict[str, Any]:
    """Return a bounded textual summary without preserving full member contents."""
    if data is None:
        return {"summary_lines": [], "copyright_lines": [], "encoding": "unread"}
    text = data.decode("utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "summary_lines": [line[:300] for line in lines[:8]],
        "copyright_lines": [
            line[:300]
            for line in lines
            if re.search(r"copyright|©|license|permission|eula|end user", line, re.IGNORECASE)
        ][:20],
        "encoding": "utf-8-replace",
    }


def parse_metadata(data: bytes) -> dict[str, Any]:
    """Parse the selected archive metadata fields into a JSON-safe mapping."""
    message = email.message_from_bytes(data, policy=email.policy.default)
    urls = []
    for value in message.get_all("Project-URL", []):
        if "," in value:
            label, url = value.split(",", 1)
            urls.append({"label": label.strip(), "url": url.strip()})
        else:
            urls.append({"label": "Project-URL", "url": value.strip()})
    return {
        "metadata_version": message.get("Metadata-Version"),
        "name": message.get("Name"),
        "version": message.get("Version"),
        "summary": message.get("Summary"),
        "license": message.get("License"),
        "license_expression": message.get("License-Expression"),
        "license_files": message.get_all("License-File", []),
        "license_classifiers": [
            value
            for value in message.get_all("Classifier", [])
            if value.lower().startswith("license ::")
        ],
        "project_urls": urls,
        "requires_python": message.get("Requires-Python"),
    }


def bounded_read(handle: BinaryIO) -> tuple[bytes | None, str]:
    """Read one evidence member while enforcing its individual byte bound."""
    data = handle.read(MAX_EVIDENCE_MEMBER_BYTES + 1)
    if len(data) > MAX_EVIDENCE_MEMBER_BYTES:
        return None, f"too_large>{MAX_EVIDENCE_MEMBER_BYTES}"
    return data, "read"


def _metadata_identity_match(
    fields: dict[str, Any], expected_name: str, expected_version: str | None
) -> bool:
    """Check archive metadata identity without turning it into a rights decision."""
    observed_name = fields.get("name")
    if not isinstance(observed_name, str) or _canonical_name(observed_name) != _canonical_name(
        expected_name
    ):
        return False
    return expected_version is None or fields.get("version") == expected_version


def fetch_json(  # noqa: C901, PLR0912 - cache and network bounds remain explicit
    url: str,
    path: Path,
    *,
    offline: bool = False,
    cache_root: Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Reuse or fetch bounded registry JSON and report malformed responses."""
    if not _https_url(url):
        return None, {
            "url": url,
            "path": str(path),
            "status": "unavailable",
            "error": "metadata URL must be HTTPS",
        }
    if not _pypi_json_url(url):
        return None, {
            "url": url,
            "path": str(path),
            "status": "unavailable",
            "error": "metadata URL host is not the canonical pypi.org host",
        }
    if cache_root is not None and not _safe_output_path(cache_root, path):
        return None, {
            "url": url,
            "path": str(path),
            "status": "unavailable",
            "error": "metadata cache path escapes the output directory",
        }
    cache_error = None
    try:
        cache_is_file = path.is_file()
    except OSError as exc:
        cache_is_file = False
        cache_error = f"{type(exc).__name__}: cannot inspect metadata cache"
    if cache_is_file:
        try:
            cached_size = path.stat().st_size
        except OSError as exc:
            cache_error = f"{type(exc).__name__}: cannot stat metadata cache"
        else:
            if cached_size > MAX_METADATA_JSON_BYTES:
                return None, {
                    "url": url,
                    "path": str(path),
                    "status": "unavailable",
                    "error": "metadata cache exceeds 16 MiB bound",
                }
            try:
                with path.open("rb") as handle:
                    data = handle.read(MAX_METADATA_JSON_BYTES + 1)
                if len(data) > MAX_METADATA_JSON_BYTES:
                    return None, {
                        "url": url,
                        "path": str(path),
                        "status": "unavailable",
                        "error": "metadata cache exceeds 16 MiB bound",
                    }
                payload = json.loads(data)
                if not isinstance(payload, dict):
                    raise TypeError("metadata cache is not a JSON object")
                return payload, {
                    "url": url,
                    "path": str(path),
                    "status": "cache_reused",
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            except (OSError, TypeError, UnicodeError) as exc:
                cache_error = f"{type(exc).__name__}: invalid metadata cache"
            except json.JSONDecodeError:
                cache_error = "JSONDecodeError: invalid metadata cache"
    if offline:
        return None, {
            "url": url,
            "path": str(path),
            "status": "unavailable",
            "error": "offline mode: metadata cache is missing or invalid"
            if cache_error is None
            else f"offline mode: {cache_error}",
        }
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "robot-sf-p05-generic-evidence/1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_METADATA_JSON_BYTES + 1)
            status = getattr(response, "status", 200)
            final_url = getattr(response, "geturl", lambda: url)()
        if not _pypi_json_url(final_url):
            raise ValueError("metadata redirect did not remain on pypi.org over HTTPS")
        if len(data) > MAX_METADATA_JSON_BYTES:
            raise ValueError("metadata JSON exceeds 16 MiB bound")
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise TypeError("metadata response is not a JSON object")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return payload, {
            "url": url,
            "path": str(path),
            "status": "fetched",
            "http_status": status,
            "final_url": final_url,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        }
    except (
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        UnicodeError,
    ) as exc:
        return None, {
            "url": url,
            "path": str(path),
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }


def fetch_archive(  # noqa: C901, PLR0912 - bounded cache and streamed fetch states stay explicit
    artifact: dict[str, Any],
    path: Path,
    downloaded: int,
    *,
    offline: bool = False,
    cache_root: Path | None = None,
) -> tuple[Path | None, dict[str, Any], int]:
    """Fetch or verify one manifest archive under the batch download bound."""
    expected_size = int(artifact["size"])
    expected_sha = artifact["sha256"]
    if not _https_url(artifact.get("url")):
        return (
            None,
            {
                "status": "unavailable",
                "error": "archive URL must be HTTPS",
            },
            downloaded,
        )
    if not _pypi_archive_url(artifact.get("url")):
        return (
            None,
            {
                "status": "unavailable",
                "error": "archive URL host is not an approved PyPI archive host",
            },
            downloaded,
        )
    if cache_root is not None and not _safe_output_path(cache_root, path):
        return (
            None,
            {
                "status": "unavailable",
                "error": "archive cache path escapes the output directory",
            },
            downloaded,
        )
    if expected_size > MAX_ARCHIVE_BYTES:
        return (
            None,
            {
                "status": "unavailable",
                "error": "manifest artifact exceeds per-archive bound",
            },
            downloaded,
        )
    if downloaded + expected_size > MAX_BATCH_DOWNLOAD_BYTES:
        return (
            None,
            {
                "status": "unavailable",
                "error": "target selections exceed total batch download bound",
            },
            downloaded,
        )
    cache_verified = False
    try:
        cache_verified = (
            path.is_file()
            and path.stat().st_size == expected_size
            and sha256_file(path) == expected_sha
        )
    except OSError:
        cache_verified = False
    if cache_verified:
        return (
            path,
            {
                "status": "cache_verified",
                "expected_size": expected_size,
                "observed_size": expected_size,
                "expected_sha256": expected_sha,
                "observed_sha256": expected_sha,
                "path": str(path),
            },
            downloaded + expected_size,
        )
    if offline:
        return (
            None,
            {
                "status": "unavailable",
                "expected_size": expected_size,
                "expected_sha256": expected_sha,
                "path": str(path),
                "error": "offline mode: archive cache is missing or digest-mismatched",
            },
            downloaded,
        )
    part = path.with_name(path.name + ".part")
    if cache_root is not None and not _safe_output_path(cache_root, part):
        return (
            None,
            {
                "status": "unavailable",
                "expected_size": expected_size,
                "expected_sha256": expected_sha,
                "path": str(path),
                "error": "archive partial path escapes the output directory",
            },
            downloaded,
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            artifact["url"], headers={"User-Agent": "robot-sf-p05-generic-evidence/1"}
        )
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            part.open("wb") as handle,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) != expected_size:
                raise ValueError(
                    f"content length {content_length} != manifest size {expected_size}"
                )
            final_url = getattr(response, "geturl", lambda: artifact["url"])()
            if not _pypi_archive_url(final_url):
                raise ValueError("archive redirect did not remain on an approved PyPI host")
            digest = hashlib.sha256()
            observed = 0
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                observed += len(chunk)
                if observed > MAX_ARCHIVE_BYTES or observed > expected_size:
                    raise ValueError("download exceeds bounded expected archive size")
                digest.update(chunk)
                handle.write(chunk)
        observed_sha = digest.hexdigest()
        if observed != expected_size or observed_sha != expected_sha:
            raise ValueError(f"download digest/size mismatch: {observed_sha}/{observed}")
        part.replace(path)
        return (
            path,
            {
                "status": "fetched_verified",
                "expected_size": expected_size,
                "observed_size": observed,
                "expected_sha256": expected_sha,
                "observed_sha256": observed_sha,
                "final_url": final_url,
                "path": str(path),
            },
            downloaded + observed,
        )
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        try:
            part.unlink(missing_ok=True)
        except OSError:
            pass
        return (
            None,
            {
                "status": "unavailable",
                "expected_size": expected_size,
                "expected_sha256": expected_sha,
                "path": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            },
            downloaded,
        )


def inspect_archive(  # noqa: C901, PLR0912, PLR0915 - archive safety states are explicit
    path: Path,
    artifact: dict[str, Any],
    *,
    expected_name: str | None = None,
    expected_version: str | None = None,
) -> dict[str, Any]:
    """Inspect bounded metadata and notice members without extracting an archive."""
    result: dict[str, Any] = {
        "filename": artifact["filename"],
        "kind": artifact["kind"],
        "url": artifact["url"],
        "expected_size": artifact["size"],
        "expected_sha256": artifact["sha256"],
        "observed_size": None,
        "observed_sha256": None,
        "digest_verified": False,
        "inspection_status": "unavailable",
        "member_count": None,
        "total_uncompressed_bytes": None,
        "evidence_members": [],
        "metadata": [],
        "errors": [],
    }
    try:
        result["observed_size"] = path.stat().st_size
        result["observed_sha256"] = sha256_file(path)
    except OSError as exc:
        result["errors"].append(f"archive file could not be read: {type(exc).__name__}: {exc}")
        return result
    result["digest_verified"] = (
        result["observed_size"] == artifact["size"]
        and result["observed_sha256"] == artifact["sha256"]
    )
    if not result["digest_verified"]:
        result["errors"].append("archive digest or size differs from immutable manifest")
        return result
    try:
        archive_kind = "zip" if path.suffix.lower() == ".whl" else "tar"
        if artifact.get("kind") == "wheel" or path.suffix.lower() == ".zip":
            archive_kind = "zip"
        evidence_total = 0
        if archive_kind == "zip":
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                result["member_count"] = len(infos)
                result["total_uncompressed_bytes"] = sum(
                    info.file_size for info in infos if not info.is_dir()
                )
                if (
                    len(infos) > MAX_MEMBER_COUNT
                    or result["total_uncompressed_bytes"] > MAX_MEMBER_UNCOMPRESSED_BYTES
                ):
                    result["errors"].append("archive member/expanded-size bound exceeded")
                    return result
                seen_names: set[str] = set()
                for info in infos:
                    if info.filename in seen_names:
                        result["errors"].append(f"duplicate archive member: {info.filename}")
                        continue
                    seen_names.add(info.filename)
                    if not safe_member_name(info.filename):
                        result["errors"].append(f"unsafe archive member: {info.filename}")
                        continue
                    if _zip_info_is_symlink(info):
                        result["errors"].append(f"unsafe archive member: {info.filename}")
                        continue
                    if info.is_dir():
                        continue
                    kind = member_kind(info.filename)
                    if not kind:
                        continue
                    if evidence_total + info.file_size > MAX_EVIDENCE_TOTAL_BYTES:
                        result["errors"].append("selected evidence-member byte bound exceeded")
                        return result
                    with archive.open(info) as handle:
                        data, status = bounded_read(handle)
                    record = {
                        "path": info.filename,
                        "kind": kind,
                        "size": info.file_size,
                        "sha256": sha256_bytes(data) if data is not None else None,
                        "read_status": status,
                    }
                    if status != "read":
                        result["errors"].append(
                            f"evidence member could not be read: {info.filename}: {status}"
                        )
                    record.update(text_summary(data))
                    result["evidence_members"].append(record)
                    evidence_total += min(info.file_size, MAX_EVIDENCE_MEMBER_BYTES)
                    if kind == "metadata" and data is not None:
                        fields = parse_metadata(data)
                        identity_match = expected_name is None or _metadata_identity_match(
                            fields, expected_name, expected_version
                        )
                        record["identity_match"] = identity_match
                        if not identity_match:
                            result["errors"].append(
                                f"archive metadata identity differs from manifest: {info.filename}"
                            )
                        result["metadata"].append(
                            {
                                "path": info.filename,
                                "sha256": record["sha256"],
                                "fields": fields,
                            }
                        )
        else:
            with tarfile.open(path, mode="r:*") as archive:
                infos = archive.getmembers()
                result["member_count"] = len(infos)
                result["total_uncompressed_bytes"] = sum(
                    info.size for info in infos if info.isfile()
                )
                if (
                    len(infos) > MAX_MEMBER_COUNT
                    or result["total_uncompressed_bytes"] > MAX_MEMBER_UNCOMPRESSED_BYTES
                ):
                    result["errors"].append("archive member/expanded-size bound exceeded")
                    return result
                seen_names: set[str] = set()
                for info in infos:
                    if info.name in seen_names:
                        result["errors"].append(f"duplicate archive member: {info.name}")
                        continue
                    seen_names.add(info.name)
                    if not safe_member_name(info.name):
                        result["errors"].append(f"unsafe archive member: {info.name}")
                        continue
                    if info.isdir():
                        continue
                    if not info.isfile():
                        result["errors"].append(f"non-regular archive member: {info.name}")
                        continue
                    kind = member_kind(info.name)
                    if not kind:
                        continue
                    if evidence_total + info.size > MAX_EVIDENCE_TOTAL_BYTES:
                        result["errors"].append("selected evidence-member byte bound exceeded")
                        return result
                    handle = archive.extractfile(info)
                    data, status = (
                        bounded_read(handle) if handle is not None else (None, "unavailable")
                    )
                    record = {
                        "path": info.name,
                        "kind": kind,
                        "size": info.size,
                        "sha256": sha256_bytes(data) if data is not None else None,
                        "read_status": status,
                    }
                    if status != "read":
                        result["errors"].append(
                            f"evidence member could not be read: {info.name}: {status}"
                        )
                    record.update(text_summary(data))
                    result["evidence_members"].append(record)
                    evidence_total += min(info.size, MAX_EVIDENCE_MEMBER_BYTES)
                    if kind == "metadata" and data is not None:
                        fields = parse_metadata(data)
                        identity_match = expected_name is None or _metadata_identity_match(
                            fields, expected_name, expected_version
                        )
                        record["identity_match"] = identity_match
                        if not identity_match:
                            result["errors"].append(
                                f"archive metadata identity differs from manifest: {info.name}"
                            )
                        result["metadata"].append(
                            {
                                "path": info.name,
                                "sha256": record["sha256"],
                                "fields": fields,
                            }
                        )
        result["inspection_status"] = "inspected" if not result["errors"] else "partial"
    except (
        AttributeError,
        OSError,
        EOFError,
        IndexError,
        KeyError,
        RuntimeError,
        tarfile.TarError,
        TypeError,
        UnicodeError,
        zipfile.BadZipFile,
        ValueError,
    ) as exc:
        result["errors"].append(f"archive inspection failed: {type(exc).__name__}: {exc}")
    return result


def _python_tag_compatible(tag: str, python_version: str) -> bool:
    """Check the Python portion of a wheel tag for the requested runtime."""
    major, _, minor = python_version.partition(".")
    target = f"{major}{minor}"
    for value in tag.lower().split("."):
        if value in {f"py{major}", f"cp{target}"}:
            return True
        if value.startswith("cp") and value[2:] == target:
            return True
    return False


def target_compatible(
    artifact: dict[str, Any],
    os_name: str,
    architecture: str,
    python_version: str = "3.13",
) -> bool:
    """Return whether an artifact is compatible with the requested target."""
    if artifact.get("kind") == "sdist":
        return True
    tag_set = {str(tag).lower() for tag in artifact.get("platform_tags", [])}
    if not any(_python_tag_compatible(tag, python_version) for tag in tag_set):
        return False
    if {"none", "any"}.issubset(tag_set):
        return True
    arch = architecture.lower()
    has_target_arch = any(tag == arch or tag.endswith("_" + arch) for tag in tag_set)
    if os_name.lower() == "linux":
        return has_target_arch and any(tag.startswith("manylinux") for tag in tag_set)
    return has_target_arch


def select_target_artifacts(
    member: dict[str, Any], *, os_name: str, architecture: str, python_version: str
) -> list[dict[str, Any]]:
    """Select a deterministic target partition and reject ambiguous wheels."""
    artifacts = member.get("artifacts", [])
    selected = [
        artifact
        for artifact in artifacts
        if target_compatible(artifact, os_name, architecture, python_version)
    ]
    wheels = [artifact for artifact in selected if artifact.get("kind") == "wheel"]
    sdists = [artifact for artifact in selected if artifact.get("kind") == "sdist"]
    if len(wheels) > 1:
        raise ValueError(
            f"{member.get('package_id')}: multiple target-compatible wheels; exact selection is ambiguous"
        )
    if len(sdists) > 1:
        raise ValueError(
            f"{member.get('package_id')}: multiple target-compatible sdists; exact selection is ambiguous"
        )
    return [*sdists, *wheels]


def metadata_summary(archives: list[dict[str, Any]], pypi: dict[str, Any]) -> dict[str, Any]:
    """Summarize observed metadata without making a rights or custody decision."""
    fields = [item["fields"] for archive in archives for item in archive.get("metadata", [])]
    info = pypi.get("info")
    if not isinstance(info, dict):
        info = {}
    info_observation = _pypi_info_observation(info)
    classifiers = info_observation["classifiers"] or []
    project_urls = info_observation["project_urls"] or {}
    return {
        "authority_boundary": "descriptive_observation_only",
        "archive_license": sorted({item.get("license") for item in fields if item.get("license")}),
        "archive_license_expression": sorted(
            {item.get("license_expression") for item in fields if item.get("license_expression")}
        ),
        "archive_license_classifiers": sorted(
            {c for item in fields for c in item.get("license_classifiers", [])}
        ),
        "archive_license_files": sorted(
            {f for item in fields for f in item.get("license_files", [])}
        ),
        "pypi_license": info_observation.get("license"),
        "pypi_license_classifiers": [
            c for c in classifiers if isinstance(c, str) and c.lower().startswith("license ::")
        ],
        "pypi_project_urls": project_urls,
        "pypi_home_page": info_observation.get("home_page"),
    }


def _pypi_info_errors(info: dict[str, Any]) -> list[str]:
    """Validate only the registry metadata fields consumed by the ledger."""
    errors: list[str] = []
    for field in ("name", "version"):
        if not isinstance(info.get(field), str):
            errors.append(f"registry metadata {field} is not a string")
    classifiers = info.get("classifiers")
    if not isinstance(classifiers, list) or not all(isinstance(item, str) for item in classifiers):
        errors.append("registry metadata classifiers are not a string list")
    project_urls = info.get("project_urls")
    if not isinstance(project_urls, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in project_urls.items()
    ):
        errors.append("registry metadata project_urls are not a string mapping")
    for field in ("license", "home_page"):
        if info.get(field) is not None and not isinstance(info.get(field), str):
            errors.append(f"registry metadata {field} is not a string or null")
    return errors


def _pypi_info_observation(info: dict[str, Any]) -> dict[str, Any]:
    """Keep only registry fields consumed by this diagnostic collector."""
    observed = {}
    for field in ("name", "version", "license", "home_page"):
        value = info.get(field)
        if field in info and (value is None or isinstance(value, str)):
            observed[field] = value
    classifiers = info.get("classifiers")
    observed["classifiers"] = (
        sorted(item for item in classifiers if isinstance(item, str))
        if isinstance(classifiers, list)
        else None
    )
    project_urls = info.get("project_urls")
    observed["project_urls"] = (
        {
            key: value
            for key, value in project_urls.items()
            if isinstance(key, str) and isinstance(value, str)
        }
        if isinstance(project_urls, dict)
        else None
    )
    return observed


def _pypi_release_file_observation(item: dict[str, Any]) -> dict[str, Any]:
    """Keep exact-match and release-routing fields, excluding arbitrary registry payloads."""
    observed = {}
    for field in (
        "filename",
        "packagetype",
        "python_version",
        "url",
        "upload_time_iso_8601",
        "requires_python",
        "yanked_reason",
    ):
        value = item.get(field)
        if field in item and (value is None or isinstance(value, str)):
            observed[field] = value
    size = item.get("size")
    if isinstance(size, int) and not isinstance(size, bool):
        observed["size"] = size
    yanked = item.get("yanked")
    if isinstance(yanked, bool):
        observed["yanked"] = yanked
    digests = item.get("digests")
    sha256 = digests.get("sha256") if isinstance(digests, dict) else None
    observed["digests"] = {"sha256": sha256 if isinstance(sha256, str) else None}
    return observed


def _pypi_release_file_sort_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return a stable key for release-file observations."""
    return tuple(
        str(item.get(field, "")) for field in ("filename", "packagetype", "python_version", "url")
    )


def _pypi_exact_file_match(item: object, artifact: dict[str, Any]) -> bool:
    """Compare one PyPI release-file observation without trusting nested shapes."""
    if not isinstance(item, dict):
        return False
    digests = item.get("digests")
    return (
        isinstance(digests, dict)
        and item.get("filename") == artifact["filename"]
        and digests.get("sha256") == artifact["sha256"]
        and isinstance(item.get("size"), int)
        and not isinstance(item.get("size"), bool)
        and item["size"] == artifact["size"]
        and item.get("url") == artifact["url"]
    )


def _collection_status(
    *,
    selected: list[dict[str, Any]],
    manifest_artifacts: list[dict[str, Any]],
    archive_complete: bool,
    pypi_complete: bool,
) -> str:
    """Classify target collection without hiding unsupported manifest artifacts."""
    if archive_complete and pypi_complete:
        return "complete_factual_evidence"
    if not selected and not manifest_artifacts:
        return "not_applicable_no_archive"
    return "partial_or_unavailable"


def collect(  # noqa: C901, PLR0912, PLR0915 - diagnostic row assembly is fail-closed
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Collect exact target artifacts and bounded factual evidence for one batch."""
    output = Path(args.output).resolve()
    cache = output / "cache"
    pypi_cache = output / "pypi-json"
    manifest_path = Path(args.batch_manifest).resolve()
    manifest_bytes = read_bounded_bytes(manifest_path, MAX_MANIFEST_JSON_BYTES, "batch manifest")
    manifest_sha = sha256_bytes(manifest_bytes)
    if manifest_sha != args.expected_batch_sha256:
        raise SystemExit(
            f"batch manifest digest mismatch: {manifest_sha} != {args.expected_batch_sha256}"
        )
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise SystemExit("batch manifest must contain a JSON object")
    owner_issue = _validate_manifest(manifest, args)
    offline = bool(getattr(args, "offline", False))
    rows = []
    network = []
    all_archives = []
    downloaded = 0
    for member in manifest["members"]:
        package_id = member["package_id"]
        pypi_url = f"https://pypi.org/pypi/{urllib.parse.quote(member['name'])}/{urllib.parse.quote(str(member['version']))}/json"
        pypi_path = pypi_cache / f"{member['name'].replace('-', '_')}-{member['version']}.json"
        if member.get("source_type") == "registry" and member.get("version") is not None:
            pypi_payload, pypi_record = fetch_json(
                pypi_url, pypi_path, offline=offline, cache_root=output
            )
            network.append({"package_id": package_id, **_redact_cache_record(pypi_record, output)})
        else:
            pypi_payload, pypi_record = (
                None,
                {
                    "status": "not_applicable",
                    "reason": "manifest source is not registry",
                },
            )
        selection_error = None
        try:
            selected = select_target_artifacts(
                member,
                os_name=args.target_os,
                architecture=args.target_architecture,
                python_version=args.python_version,
            )
        except ValueError as exc:
            selected = []
            selection_error = f"{type(exc).__name__}: {exc}"
        row = {
            "package_id": package_id,
            "name": member["name"],
            "version": member["version"],
            "identity_sha256": member["identity_sha256"],
            "owner_issue": owner_issue,
            "owner_url": member.get("assignment", {}).get("owner_url"),
            "distribution_mode": member.get("distribution_mode"),
            "source_type": member.get("source_type"),
            "surface_membership": member.get("surface_membership"),
            "selected_profiles": member.get("selected_profiles", []),
            "profiles": member.get("profiles", []),
            "originating_extras": member.get("originating_extras", []),
            "lockfile": member.get("lockfile"),
            "dependencies": member.get("dependencies", []),
            "target_context": {
                "target": {
                    "os": args.target_os,
                    "architecture": args.target_architecture,
                    "python": args.python_version,
                    "resolver": {
                        "name": args.resolver_name,
                        "version": args.resolver_version,
                        "lock_mode": "frozen",
                    },
                },
                "profile": manifest.get("profile"),
                "artifact_selection": "at most one target-compatible sdist and wheel; all other manifest/release files remain uninspected",
                "artifact_selection_status": "unresolved" if selection_error else "resolved",
                "artifact_selection_error": selection_error,
                "selected_artifacts": selected,
                "uninspected_manifest_artifacts": [
                    artifact for artifact in member.get("artifacts", []) if artifact not in selected
                ],
                "uninspected_release_files": [],
            },
            "canonical_input": {
                "batch_manifest_sha256": manifest_sha,
                "audit_source_sha": manifest.get("audit_source_sha"),
                "candidate_commit_sha": manifest.get("candidate_commit_sha"),
                "candidate_tree_sha": manifest.get("candidate_tree_sha"),
            },
            "pypi_evidence": None,
            "archive_evidence": [],
            "license_facts": {},
            "shipping_boundary": {
                "distribution_mode": member.get("distribution_mode"),
                "surface_membership": member.get("surface_membership"),
                "source": member.get("source"),
                "candidate_archive_shipped": member.get(
                    "candidate_archive_shipped", "not_recorded"
                ),
                "manifest_routing_only": True,
            },
            "interpretation": {
                "status": "unresolved_diagnostic",
                "rights_review_owner_issue": owner_issue,
                "rights_review_url": member.get("assignment", {}).get("owner_url"),
                "policy_disposition_from_manifest": member.get("policy_disposition"),
                "mechanical_rule_id_from_manifest": member.get("policy_rule_id"),
                "rights_conclusion": "not_assessed",
                "permission_inferred": False,
            },
            "unresolved_questions": [
                {
                    "question": "Which release-surface and rights disposition applies to this exact selected dependency identity?",
                    "owner": owner_issue,
                }
            ],
            "collection_status": "unavailable",
        }
        if selection_error is not None:
            row["unresolved_questions"].append(
                {
                    "question": f"Target artifact selection is unresolved: {selection_error}",
                    "owner": owner_issue,
                }
            )
        pypi_identity_match = False
        if pypi_payload is not None:
            pypi_info = pypi_payload.get("info")
            raw_release_files = pypi_payload.get("urls")
            pypi_errors: list[str] = []
            if not isinstance(pypi_info, dict):
                pypi_info = {}
                pypi_errors.append("registry metadata info is not an object")
            pypi_errors.extend(_pypi_info_errors(pypi_info))
            if not isinstance(raw_release_files, list) or not all(
                isinstance(item, dict) for item in raw_release_files
            ):
                release_files: list[dict[str, Any]] = []
                pypi_errors.append("registry metadata urls is not a list of objects")
            else:
                release_files = sorted(raw_release_files, key=_pypi_release_file_sort_key)
            release_file_observations = [
                _pypi_release_file_observation(item) for item in release_files
            ]
            observed_name = pypi_info.get("name")
            observed_version = pypi_info.get("version")
            pypi_identity_match = (
                isinstance(observed_name, str)
                and _canonical_name(observed_name) == _canonical_name(member["name"])
                and isinstance(observed_version, str)
                and observed_version == member["version"]
            )
            if not pypi_identity_match:
                pypi_errors.append(
                    "registry metadata name/version does not match manifest identity"
                )
            selected_names = {artifact["filename"] for artifact in selected}
            row["pypi_evidence"] = {
                "json_url": pypi_url,
                "json_sha256": pypi_record.get("sha256"),
                "release_page_url": f"https://pypi.org/project/{urllib.parse.quote(member['name'])}/{urllib.parse.quote(str(member['version']))}/",
                "metadata_identity_match": pypi_identity_match,
                "errors": pypi_errors,
                "release_date": next(
                    (
                        item.get("upload_time_iso_8601")
                        for item in release_files
                        if item.get("upload_time_iso_8601")
                    ),
                    None,
                ),
                "metadata": {
                    "info": _pypi_info_observation(pypi_info),
                    "release_files": release_file_observations,
                },
                "file_matches": [
                    {
                        "filename": artifact["filename"],
                        "manifest_sha256": artifact["sha256"],
                        "manifest_size": artifact["size"],
                        "manifest_url": artifact["url"],
                        "pypi_match": next(
                            (
                                _pypi_release_file_observation(item)
                                for item in release_files
                                if item.get("filename") == artifact["filename"]
                            ),
                            None,
                        ),
                        "exact_match": any(
                            _pypi_exact_file_match(item, artifact) for item in release_files
                        ),
                    }
                    for artifact in selected
                ],
            }
            row["target_context"]["uninspected_release_files"] = [
                observation
                for item, observation in zip(release_files, release_file_observations, strict=False)
                if item.get("filename") not in selected_names
            ]
        package_cache = cache / member["name"].replace("-", "_") / str(member["version"])
        for artifact in selected:
            match = None
            if member.get("source_type") == "registry":
                match = next(
                    (
                        item
                        for item in (row["pypi_evidence"] or {}).get("file_matches", [])
                        if item["filename"] == artifact["filename"]
                    ),
                    None,
                )
            if member.get("source_type") == "registry" and (not match or not match["exact_match"]):
                row["unresolved_questions"].append(
                    {
                        "question": f"Can the exact manifest artifact {artifact['filename']} be bound to the same PyPI release bytes before review?",
                        "owner": owner_issue,
                    }
                )
                continue
            path, download_record, downloaded = fetch_archive(
                artifact,
                package_cache / artifact["filename"],
                downloaded,
                offline=offline,
                cache_root=output,
            )
            download_record = _redact_cache_record(download_record, output)
            network.append(
                {
                    "package_id": package_id,
                    "artifact": artifact["filename"],
                    **download_record,
                }
            )
            if path is not None:
                archive = inspect_archive(
                    path,
                    artifact,
                    expected_name=member["name"],
                    expected_version=member.get("version"),
                )
                archive["download_status"] = download_record["status"]
                archive["local_cache_path"] = _relative_output_path(path, output)
            else:
                archive = {
                    "filename": artifact["filename"],
                    "kind": artifact["kind"],
                    "url": artifact["url"],
                    "expected_size": artifact["size"],
                    "expected_sha256": artifact["sha256"],
                    "digest_verified": False,
                    "inspection_status": "unavailable",
                    "errors": [download_record.get("error", "download failed")],
                }
            row["archive_evidence"].append(archive)
            all_archives.append(archive)
        row["license_facts"] = metadata_summary(row["archive_evidence"], pypi_payload or {})
        if selected and not row["archive_evidence"]:
            row["unresolved_questions"].append(
                {
                    "question": "No selected target-compatible archive was available for bounded evidence; can the exact manifest identity be collected without substitution?",
                    "owner": owner_issue,
                }
            )
        if not selected and member.get("artifacts"):
            row["unresolved_questions"].append(
                {
                    "question": "No target-compatible manifest artifact was selected; which target-specific artifact should be reviewed without substituting bytes?",
                    "owner": owner_issue,
                }
            )
        if selected and not any(
            item.get("kind") in {"license_or_notice", "bundled_component_path"}
            for archive in row["archive_evidence"]
            for item in archive.get("evidence_members", [])
        ):
            row["unresolved_questions"].append(
                {
                    "question": "No readable license, copying, notice, or bundled-component member was found in the selected archive; what authoritative release-surface evidence is required?",
                    "owner": owner_issue,
                }
            )
        archive_complete = (
            bool(selected)
            and bool(row["archive_evidence"])
            and all(
                archive.get("digest_verified") and archive.get("inspection_status") == "inspected"
                for archive in row["archive_evidence"]
            )
        )
        pypi_complete = member.get("source_type") != "registry" or (
            bool(row.get("pypi_evidence"))
            and pypi_identity_match
            and not row["pypi_evidence"].get("errors")
            and all(
                item.get("exact_match") for item in row["pypi_evidence"].get("file_matches", [])
            )
        )
        row["collection_status"] = _collection_status(
            selected=selected,
            manifest_artifacts=member.get("artifacts", []),
            archive_complete=archive_complete,
            pypi_complete=pypi_complete,
        )
        rows.append(row)
    coverage = {
        "assigned_rows": len(manifest["members"]),
        "ledger_rows": len(rows),
        "exact_member_set": [row["package_id"] for row in rows]
        == [member["package_id"] for member in manifest["members"]],
        "target_selected_artifacts": sum(
            len(row["target_context"]["selected_artifacts"]) for row in rows
        ),
        "uninspected_manifest_artifacts": sum(
            len(row["target_context"]["uninspected_manifest_artifacts"]) for row in rows
        ),
        "uninspected_release_files": sum(
            len(row["target_context"]["uninspected_release_files"]) for row in rows
        ),
        "complete_factual_rows": sum(
            row["collection_status"] == "complete_factual_evidence" for row in rows
        ),
        "partial_or_unavailable_rows": sum(
            row["collection_status"] == "partial_or_unavailable" for row in rows
        ),
        "not_applicable_rows": sum(
            row["collection_status"] == "not_applicable_no_archive" for row in rows
        ),
        "unavailable_rows": [
            row["package_id"]
            for row in rows
            if row["collection_status"] == "partial_or_unavailable"
        ],
        "archive_artifacts_expected_target": sum(
            len(row["target_context"]["selected_artifacts"]) for row in rows
        ),
        "archive_artifacts_digest_verified": sum(
            archive.get("digest_verified", False) for archive in all_archives
        ),
        "archive_artifacts_inspected": sum(
            archive.get("inspection_status") == "inspected" for archive in all_archives
        ),
        "pypi_json_rows": sum(row.get("pypi_evidence") is not None for row in rows),
        "pypi_exact_file_match_rows": sum(
            bool(row.get("pypi_evidence"))
            and all(
                item.get("exact_match") for item in row["pypi_evidence"].get("file_matches", [])
            )
            for row in rows
        ),
        "target_download_bytes": sum(
            archive.get("observed_size", 0) or 0 for archive in all_archives
        ),
        "batch_download_limit_bytes": MAX_BATCH_DOWNLOAD_BYTES,
        "per_archive_limit_bytes": MAX_ARCHIVE_BYTES,
    }
    result = {
        "schema_version": "robot_sf.p05.generic_dependency_evidence.v1",
        "task_id": args.task_id,
        "owner_issue": owner_issue,
        "owner_url": manifest["owner_url"],
        "status": "diagnostic_only",
        "claim_boundary": "Exact target-compatible dependency archive, registry metadata, bounded license/notice member, and manifest routing facts only; no rights conclusion, permission, candidate admission, or all-platform clearance.",
        "inputs": {
            "batch_manifest": {
                "path": _input_display_path(manifest_path, output),
                "sha256": manifest_sha,
                "member_count": manifest.get("member_count"),
                "max_bytes": MAX_MANIFEST_JSON_BYTES,
                "max_members": MAX_MANIFEST_MEMBER_COUNT,
            },
            "requested": {
                "owner_issue": args.expected_owner_issue,
                "profile": args.expected_profile,
                "audit_source_sha": args.expected_audit_source_sha,
                "candidate_commit_sha": args.expected_candidate_commit_sha,
                "candidate_tree_sha": args.expected_candidate_tree_sha,
            },
            "validated": {
                "audit_source_sha": manifest.get("audit_source_sha"),
                "candidate_commit_sha": manifest.get("candidate_commit_sha"),
                "candidate_tree_sha": manifest.get("candidate_tree_sha"),
            },
        },
        "target": {
            "os": args.target_os,
            "architecture": args.target_architecture,
            "python": args.python_version,
            "offline": offline,
            "resolver": {
                "name": args.resolver_name,
                "version": args.resolver_version,
                "lock_mode": "frozen",
            },
            "profile": manifest.get("profile"),
            "included_extras": manifest.get("included_extras"),
            "target_only": True,
        },
        "separate_gate_update": manifest.get("separate_gate_update"),
        "coverage": coverage,
        "network_observations": network,
        "rows": rows,
    }
    write_json(output / "dependency_evidence_ledger.json", result)
    cache_files = []
    for path in sorted(cache.rglob("*")) if cache.exists() else []:
        if _safe_output_path(output, path) and path.is_file() and not path.name.endswith(".part"):
            cache_files.append(
                {
                    "path": _relative_output_path(path, output),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(
        output / "private_cache_summary.json",
        {
            "schema_version": "robot_sf.p05.private_archive_cache_summary.v1",
            "classification": "ignored-cache",
            "durable_evidence": "compact ledger only; archive bytes remain private cache",
            "per_archive_limit_bytes": MAX_ARCHIVE_BYTES,
            "total_batch_download_limit_bytes": MAX_BATCH_DOWNLOAD_BYTES,
            "files": cache_files,
            "bytes": sum(item["size"] for item in cache_files),
            "file_count": len(cache_files),
        },
    )
    write_json(
        output / "collector-run.json",
        {
            "schema_version": "robot_sf.p05.generic_collector_run.v1",
            "task_id": args.task_id,
            "owner_issue": owner_issue,
            "argv": "<local invocation omitted; input identities are recorded above>",
            "finished_at": now(),
            "exit_code": 0,
            "coverage": coverage,
            "ledger_sha256": sha256_file(output / "dependency_evidence_ledger.json"),
        },
    )
    return result


def main() -> int:
    """Parse the diagnostic CLI and emit the bounded ledger."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-manifest", required=True)
    parser.add_argument("--expected-batch-sha256", required=True)
    parser.add_argument("--expected-owner-issue", required=True, type=int)
    parser.add_argument("--expected-profile", default="all")
    parser.add_argument("--expected-audit-source-sha", required=True)
    parser.add_argument("--expected-candidate-commit-sha", required=True)
    parser.add_argument("--expected-candidate-tree-sha", required=True)
    parser.add_argument("--target-os", default="Linux")
    parser.add_argument("--target-architecture", default="x86_64")
    parser.add_argument("--python-version", default="3.13")
    parser.add_argument("--resolver-name", default="uv")
    parser.add_argument("--resolver-version", default="0.11.21")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="reuse only bounded local caches; never contact PyPI or download archives",
    )
    args = parser.parse_args()
    args.argv = " ".join(sys.argv)
    try:
        result = collect(args)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"collector failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["coverage"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
