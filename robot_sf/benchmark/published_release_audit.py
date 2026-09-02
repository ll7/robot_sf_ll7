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
import shutil
import stat
import struct
import sys
import tarfile
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

from robot_sf.benchmark.artifact_publication import (
    PUBLICATION_BUNDLE_SCHEMA_VERSION,
    PublicationPreflightError,
    verify_publication_bundle_preflight,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_erratum import (
    PredecessorEvidence,
    ReleaseErratumError,
    validate_erratum_receipt_against_campaign,
)
from robot_sf.common.optional_import import try_import

_release_tag_identity = try_import("robot_sf.benchmark.release_tag_identity")

SCHEMA = "published_release_audit.v1"
NETWORK_SCHEMA = "published_release_audit.network.v1"
GITHUB_API_BASE = "https://api.github.com"
ZENODO_API_BASE = "https://zenodo.org/api"
DEFAULT_NETWORK_TIMEOUT = 60.0
DEFAULT_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
# The largest known publication archive expands to roughly 686 MiB.  These
# limits leave room for that release while bounding metadata and decompressed
# output from untrusted archives before the audit reads it further.
DEFAULT_MAX_ARCHIVE_MEMBERS = 100_000
DEFAULT_MAX_MEMBER_EXPANDED_BYTES = 1 * 1024 * 1024 * 1024
DEFAULT_MAX_ARCHIVE_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_EXTRACTION_CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_ZIP_CENTRAL_DIRECTORY_BYTES = 64 * 1024 * 1024
_ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
_ZIP64_EOCD_LOCATOR_SIGNATURE = b"PK\x06\x07"
_ZIP_EOCD_STRUCT = struct.Struct("<4s4H2LH")
_ZIP64_EOCD_LOCATOR_STRUCT = struct.Struct("<4sLQL")
_ZIP64_EOCD_STRUCT = struct.Struct("<4sQ2H2L4Q")
_ZIP_EOCD_SIZE = _ZIP_EOCD_STRUCT.size
_ZIP64_EOCD_LOCATOR_SIZE = _ZIP64_EOCD_LOCATOR_STRUCT.size
_ZIP64_EOCD_FIXED_SIZE = _ZIP64_EOCD_STRUCT.size
_ZIP_CENTRAL_DIRECTORY_FIXED_SIZE = 46
_MAX_ZIP64_EOCD_RECORD_BYTES = 1 * 1024 * 1024
_ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")
ERRATUM_CUSTODY_ASSET = "publication_custody.json"
ERRATUM_MANIFEST_ASSET = "publication_manifest.json"
ERRATUM_CHECKSUMS_ASSET = "checksums.sha256"
ERRATUM_REPOSITORY_URL = "https://github.com/ll7/robot_sf_ll7"
_ERRATUM_CURRENT_TAG_KEYS = (
    "release_tag",
    "release_id",
    "benchmark_release_tag",
    "benchmark_release_id",
)
_ERRATUM_CURRENT_DOI_KEYS = ("doi", "version_doi")
_ERRATUM_OPTIONAL_DOCUMENTS = (
    "campaign_manifest.json",
    "manifest.json",
    "run_meta.json",
)

_DOI_RE = re.compile(r"^10\.5281/zenodo\.[1-9][0-9]*$")
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_BODY_SHA_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{40})(?![0-9a-f])", re.IGNORECASE)
_CANONICAL_ERRATUM_TAG_RE = re.compile(r"^(?P<predecessor>.+-[0-9a-f]{40})-erratum\.1$")


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


def _safe_extraction_target(
    dest: Path, member_name: str, *, kind: str, directory: bool = False
) -> Path:
    """Return a member target only when its lexical and resolved paths are safe."""
    raw_name = member_name[:-1] if directory and member_name.endswith("/") else member_name
    raw_parts = raw_name.split("/")
    path = Path(raw_name)
    if (
        path.is_absolute()
        or not raw_name
        or any(part in {"", ".", ".."} for part in raw_parts)
        or (member_name.endswith("/") and not directory)
        or "\\" in member_name
        or "\x00" in member_name
    ):
        raise ValueError(f"{kind} path escape: {member_name}")
    target = (dest / path).resolve()
    if not target.is_relative_to(dest):
        raise ValueError(f"{kind} path escape: {member_name}")
    return target


def _validate_extraction_limits(
    *, max_members: int, max_member_expanded_bytes: int, max_expanded_bytes: int, chunk_size: int
) -> None:
    """Reject invalid archive extraction bounds before touching the destination."""
    for name, value in (
        ("max_members", max_members),
        ("max_member_expanded_bytes", max_member_expanded_bytes),
        ("max_expanded_bytes", max_expanded_bytes),
        ("chunk_size", chunk_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _prepare_member_parent(dest: Path, target: Path, *, kind: str, member_name: str) -> None:
    """Create safe parent directories without following archive-created symlinks."""
    try:
        relative_parts = target.parent.relative_to(dest).parts
    except ValueError as exc:
        raise ValueError(f"{kind} path escape: {member_name}") from exc
    current = dest
    for part in relative_parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{kind} member traverses symbolic link: {member_name}")
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"{kind} member conflicts with directory: {member_name}")
        else:
            current.mkdir()


def _prepare_member_directory(dest: Path, target: Path, *, kind: str, member_name: str) -> None:
    """Create one validated archive directory without replacing existing entries."""
    _prepare_member_parent(dest, target, kind=kind, member_name=member_name)
    if target.is_symlink():
        raise ValueError(f"{kind} member is a symbolic link: {member_name}")
    if target.exists() and not target.is_dir():
        raise ValueError(f"{kind} member conflicts with directory: {member_name}")
    target.mkdir(exist_ok=True)


def _stream_archive_member(  # noqa: PLR0913
    source: Any,
    target: Path,
    *,
    kind: str,
    member_name: str,
    declared_size: int,
    expanded_bytes: int,
    max_member_expanded_bytes: int,
    max_expanded_bytes: int,
    chunk_size: int,
) -> int:
    """Copy one archive member while enforcing declared and observed byte bounds.

    Returns:
        Updated cumulative expanded byte count.
    """
    written_bytes = 0
    with target.open("xb") as handle:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            if not isinstance(chunk, (bytes, bytearray, memoryview)):
                raise ValueError(f"{kind} member yielded a non-byte chunk: {member_name}")
            chunk_bytes = bytes(chunk)
            next_written_bytes = written_bytes + len(chunk_bytes)
            if next_written_bytes > max_member_expanded_bytes:
                raise ValueError(
                    f"{kind} member exceeds per-file expanded byte limit: {member_name}"
                )
            if expanded_bytes + next_written_bytes > max_expanded_bytes:
                raise ValueError(f"{kind} archive exceeds cumulative expanded byte limit")
            write_result = handle.write(chunk_bytes)
            if write_result != len(chunk_bytes):
                raise OSError(f"short write for {kind} member {member_name}")
            written_bytes = next_written_bytes
    if written_bytes != declared_size:
        raise ValueError(
            f"{kind} member expanded size mismatch for {member_name}: "
            f"observed {written_bytes}, expected {declared_size}"
        )
    return expanded_bytes + written_bytes


def _validate_zip_central_directory_metadata(
    *,
    entries: int,
    central_directory_size: int,
    central_directory_offset: int,
    central_directory_end: int,
    max_members: int,
) -> None:
    """Reject ZIP central-directory metadata before ``ZipFile`` parses it."""
    if entries > max_members:
        raise ValueError(f"zip member count exceeds limit: {entries} > {max_members}")
    if central_directory_size > DEFAULT_MAX_ZIP_CENTRAL_DIRECTORY_BYTES:
        raise ValueError(
            "zip central directory exceeds limit: "
            f"{central_directory_size} > {DEFAULT_MAX_ZIP_CENTRAL_DIRECTORY_BYTES}"
        )
    if central_directory_size > central_directory_end:
        raise ValueError("zip central directory extends before the archive start")
    central_directory_start = central_directory_end - central_directory_size
    if central_directory_offset > central_directory_start:
        raise ValueError("zip central directory offset is invalid")
    if entries > central_directory_size // _ZIP_CENTRAL_DIRECTORY_FIXED_SIZE:
        raise ValueError("zip central directory is too small for its declared member count")


def _read_zip64_end_record(  # noqa: C901
    handle: Any,
    *,
    file_size: int,
    locator_offset: int,
    locator: bytes,
) -> tuple[int, int, int, int, int, int, int] | None:
    """Read bounded ZIP64 metadata from a structurally valid locator.

    The locator and record offsets are untrusted archive bytes.  They are
    range-checked before any seek, and the variable extensible-data section is
    bounded without being loaded into memory.

    Returns:
        ``(record_offset, disk, directory_disk, entries_on_disk,
        entries_total, central_directory_size, central_directory_offset)``
        when the locator points at a valid ZIP64 end record; otherwise ``None``
        for a non-ZIP64 locator signature.
    """
    if len(locator) != _ZIP64_EOCD_LOCATOR_SIZE:
        raise ValueError("zip64 end-of-central-directory locator is truncated")
    signature, disk, declared_record_offset, disks = _ZIP64_EOCD_LOCATOR_STRUCT.unpack(locator)
    if signature != _ZIP64_EOCD_LOCATOR_SIGNATURE:
        return None
    if disk != 0 or disks != 1:
        raise ValueError("zip archive spans multiple disks")
    if declared_record_offset > file_size - _ZIP64_EOCD_FIXED_SIZE:
        raise ValueError("zip64 end-of-central-directory record is truncated")

    # The locator offset is relative to the archive disk and may not include a
    # self-extracting prefix.  Derive the physical location from the fixed
    # record/locator adjacency first; the untrusted locator offset is only
    # range-checked above and never used to size a read.
    record_offset = locator_offset - _ZIP64_EOCD_FIXED_SIZE
    if record_offset < 0:
        raise ValueError("zip64 end-of-central-directory record is truncated")
    handle.seek(record_offset)
    record = handle.read(_ZIP64_EOCD_FIXED_SIZE)
    if len(record) != _ZIP64_EOCD_FIXED_SIZE:
        raise ValueError("zip64 end-of-central-directory record is truncated")
    if record[:4] != _ZIP64_EOCD_SIGNATURE:
        # A ZIP64 record with extensible data is not fixed-size.  Fall back to
        # the locator's bounded, range-checked offset so valid non-prefixed
        # records remain supported while still rejecting a bad locator.
        record_offset = declared_record_offset
        handle.seek(record_offset)
        record = handle.read(_ZIP64_EOCD_FIXED_SIZE)
        if len(record) != _ZIP64_EOCD_FIXED_SIZE:
            raise ValueError("zip64 end-of-central-directory record is truncated")
    (
        record_signature,
        record_size,
        _create_version,
        _read_version,
        disk_number,
        directory_disk,
        entries_on_disk,
        entries_total,
        central_directory_size,
        central_directory_offset,
    ) = _ZIP64_EOCD_STRUCT.unpack(record)
    if record_signature != _ZIP64_EOCD_SIGNATURE:
        raise ValueError("zip64 end-of-central-directory record has invalid signature")
    if record_size < _ZIP64_EOCD_FIXED_SIZE - 12:
        raise ValueError("zip64 end-of-central-directory record is malformed")
    if record_size > _MAX_ZIP64_EOCD_RECORD_BYTES:
        raise ValueError("zip64 end-of-central-directory record exceeds metadata limit")
    if declared_record_offset > record_offset:
        raise ValueError("zip64 end-of-central-directory locator offset is invalid")
    record_end = record_offset + 12 + record_size
    if record_end != locator_offset:
        raise ValueError("zip64 end-of-central-directory record is truncated")
    if disk_number != 0 or directory_disk != 0 or entries_on_disk != entries_total:
        raise ValueError("zip archive spans multiple disks")
    return (
        record_offset,
        disk_number,
        directory_disk,
        entries_on_disk,
        entries_total,
        central_directory_size,
        central_directory_offset,
    )


def _preflight_zip_metadata(archive_path: Path, *, max_members: int) -> None:  # noqa: C901
    """Bound ZIP metadata before constructing ``zipfile.ZipFile``.

    Only the bounded end-of-archive window plus fixed-size ZIP64 records are
    read.  In particular, the declared central-directory size is checked
    before ``ZipFile`` can allocate/read that directory or instantiate one
    ``ZipInfo`` object per attacker-controlled entry.
    """
    eocd_size = _ZIP_EOCD_SIZE
    max_comment_size = (1 << 16) - 1
    eocd_search_size = eocd_size + max_comment_size
    try:
        file_size = archive_path.stat().st_size
        if file_size < eocd_size:
            raise ValueError("zip end-of-central-directory record is truncated")
        with archive_path.open("rb") as handle:
            tail_size = min(file_size, eocd_search_size)
            tail_start = file_size - tail_size
            handle.seek(tail_start)
            tail = handle.read(tail_size)
            if len(tail) != tail_size:
                raise ValueError("zip end-of-central-directory record is truncated")

            eocd_offset: int | None = None
            eocd: tuple[Any, ...] | None = None
            search_end = len(tail)
            while True:
                candidate = tail.rfind(_ZIP_EOCD_SIGNATURE, 0, search_end)
                if candidate < 0:
                    break
                search_end = candidate
                if candidate + eocd_size > len(tail):
                    continue
                candidate_data = tail[candidate : candidate + eocd_size]
                candidate_eocd = _ZIP_EOCD_STRUCT.unpack(candidate_data)
                comment_size = candidate_eocd[-1]
                if candidate + eocd_size + comment_size == len(tail):
                    eocd_offset = tail_start + candidate
                    eocd = candidate_eocd
                    break
            if eocd_offset is None or eocd is None:
                raise ValueError("zip end-of-central-directory record is missing or truncated")

            (
                _disk_number,
                _directory_disk,
                entries_on_disk,
                entries_total,
                central_directory_size_32,
                central_directory_offset_32,
                _comment_size,
            ) = eocd[1:]
            requires_zip64 = (
                entries_on_disk == (1 << 16) - 1
                or entries_total == (1 << 16) - 1
                or central_directory_size_32 == (1 << 32) - 1
                or central_directory_offset_32 == (1 << 32) - 1
            )
            zip64_metadata: tuple[int, int, int, int, int, int, int] | None = None
            locator_offset = eocd_offset - _ZIP64_EOCD_LOCATOR_SIZE
            if locator_offset >= 0:
                handle.seek(locator_offset)
                locator = handle.read(_ZIP64_EOCD_LOCATOR_SIZE)
                if locator[:4] == _ZIP64_EOCD_LOCATOR_SIGNATURE:
                    try:
                        zip64_metadata = _read_zip64_end_record(
                            handle,
                            file_size=file_size,
                            locator_offset=locator_offset,
                            locator=locator,
                        )
                    except ValueError:
                        if requires_zip64:
                            raise
            if requires_zip64 and zip64_metadata is None:
                raise ValueError("zip64 end-of-central-directory record is missing or truncated")

            if zip64_metadata is None:
                disk_number, directory_disk = eocd[1:3]
                if disk_number != 0 or directory_disk != 0 or entries_on_disk != entries_total:
                    raise ValueError("zip archive spans multiple disks")
                _validate_zip_central_directory_metadata(
                    entries=entries_total,
                    central_directory_size=central_directory_size_32,
                    central_directory_offset=central_directory_offset_32,
                    central_directory_end=eocd_offset,
                    max_members=max_members,
                )
                return

            (
                record_offset,
                _disk_number,
                _directory_disk,
                _entries_on_disk,
                entries_total,
                central_directory_size,
                central_directory_offset,
            ) = zip64_metadata
            _validate_zip_central_directory_metadata(
                entries=entries_total,
                central_directory_size=central_directory_size,
                central_directory_offset=central_directory_offset,
                central_directory_end=record_offset,
                max_members=max_members,
            )
    except OSError as exc:
        raise ValueError("unable to read ZIP metadata") from exc


def _extract_zip_members(  # noqa: C901
    archive_path: Path,
    dest: Path,
    *,
    max_members: int,
    max_member_expanded_bytes: int,
    max_expanded_bytes: int,
    chunk_size: int,
) -> list[str]:
    """Validate and stream one ZIP archive into ``dest``.

    Returns:
        Validated archive member names.
    """
    _preflight_zip_metadata(archive_path, max_members=max_members)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > max_members:
            raise ValueError(f"zip member count exceeds limit: {len(infos)} > {max_members}")
        seen: set[str] = set()
        seen_targets: set[Path] = set()
        declared_expanded_bytes = 0
        for info in infos:
            if info.filename in seen:
                raise ValueError(f"zip contains duplicate member: {info.filename}")
            seen.add(info.filename)
            target = _safe_extraction_target(
                dest, info.filename, kind="zip", directory=info.is_dir()
            )
            if target in seen_targets:
                raise ValueError(f"zip contains colliding member: {info.filename}")
            seen_targets.add(target)
            zip_type = stat.S_IFMT(info.external_attr >> 16)
            if zip_type == stat.S_IFLNK:
                raise ValueError(f"zip contains symbolic link: {info.filename}")
            if zip_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValueError(f"zip contains non-regular member: {info.filename}")
            if info.is_dir():
                continue
            if info.file_size > max_member_expanded_bytes:
                raise ValueError(
                    f"zip member exceeds per-file expanded byte limit: {info.filename}"
                )
            declared_expanded_bytes += info.file_size
            if declared_expanded_bytes > max_expanded_bytes:
                raise ValueError("zip archive exceeds cumulative expanded byte limit")

        expanded_bytes = 0
        for info in infos:
            target = _safe_extraction_target(
                dest, info.filename, kind="zip", directory=info.is_dir()
            )
            if info.is_dir():
                _prepare_member_directory(dest, target, kind="zip", member_name=info.filename)
                continue
            _prepare_member_parent(dest, target, kind="zip", member_name=info.filename)
            with archive.open(info, "r") as source:
                expanded_bytes = _stream_archive_member(
                    source,
                    target,
                    kind="zip",
                    member_name=info.filename,
                    declared_size=info.file_size,
                    expanded_bytes=expanded_bytes,
                    max_member_expanded_bytes=max_member_expanded_bytes,
                    max_expanded_bytes=max_expanded_bytes,
                    chunk_size=chunk_size,
                )
        return [info.filename for info in infos]


def _extract_tar_members(  # noqa: C901
    archive_path: Path,
    dest: Path,
    *,
    max_members: int,
    max_member_expanded_bytes: int,
    max_expanded_bytes: int,
    chunk_size: int,
) -> list[str]:
    """Validate and stream one TAR archive into ``dest``.

    Returns:
        Validated archive member names.
    """
    with tarfile.open(archive_path, mode="r:*") as archive:
        seen: set[str] = set()
        seen_targets: set[Path] = set()
        members: list[str] = []
        expanded_bytes = 0
        for member in archive:
            if len(members) >= max_members:
                raise ValueError(f"tar member count exceeds limit: {max_members}")
            if member.name in seen:
                raise ValueError(f"tar contains duplicate member: {member.name}")
            seen.add(member.name)
            members.append(member.name)
            target = _safe_extraction_target(
                dest, member.name, kind="tar", directory=member.isdir()
            )
            if target in seen_targets:
                raise ValueError(f"tar contains colliding member: {member.name}")
            seen_targets.add(target)
            if not (member.isdir() or member.isreg()):
                raise ValueError(f"tar contains non-regular member: {member.name}")
            if member.isdir():
                _prepare_member_directory(dest, target, kind="tar", member_name=member.name)
                continue
            if isinstance(member.size, bool) or not isinstance(member.size, int) or member.size < 0:
                raise ValueError(f"tar member has an invalid size: {member.name}")
            if member.size > max_member_expanded_bytes:
                raise ValueError(f"tar member exceeds per-file expanded byte limit: {member.name}")
            if expanded_bytes + member.size > max_expanded_bytes:
                raise ValueError("tar archive exceeds cumulative expanded byte limit")
            _prepare_member_parent(dest, target, kind="tar", member_name=member.name)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"tar member has no readable data: {member.name}")
            with source:
                expanded_bytes = _stream_archive_member(
                    source,
                    target,
                    kind="tar",
                    member_name=member.name,
                    declared_size=member.size,
                    expanded_bytes=expanded_bytes,
                    max_member_expanded_bytes=max_member_expanded_bytes,
                    max_expanded_bytes=max_expanded_bytes,
                    chunk_size=chunk_size,
                )
        return members


def _extract_members(  # noqa: C901
    archive_path: Path,
    dest: Path,
    *,
    max_members: int = DEFAULT_MAX_ARCHIVE_MEMBERS,
    max_member_expanded_bytes: int = DEFAULT_MAX_MEMBER_EXPANDED_BYTES,
    max_expanded_bytes: int = DEFAULT_MAX_ARCHIVE_EXPANDED_BYTES,
    chunk_size: int = DEFAULT_EXTRACTION_CHUNK_SIZE,
) -> list[str]:
    """Safely stream an archive and return its member names.

    Extraction is staged and committed only after every member has been read.
    A rejected archive therefore leaves no partial output at ``dest``.

    Returns:
        Validated archive member names.

    Raises:
        ValueError: On path escape, unsupported archive, limit violation, or read failure.
    """
    _validate_extraction_limits(
        max_members=max_members,
        max_member_expanded_bytes=max_member_expanded_bytes,
        max_expanded_bytes=max_expanded_bytes,
        chunk_size=chunk_size,
    )
    lexical_dest = Path(dest)
    if lexical_dest.is_symlink():
        raise ValueError("extraction destination must not be a symlink")
    resolved_dest = lexical_dest.resolve()
    if resolved_dest.exists() and resolved_dest.is_symlink():
        raise ValueError("extraction destination must not be a symlink")
    if resolved_dest.exists() and not resolved_dest.is_dir():
        raise ValueError("extraction destination must be a directory")
    resolved_dest.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = None
    previous: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{resolved_dest.name or 'extraction'}-", dir=resolved_dest.parent
            )
        )
        if zipfile.is_zipfile(archive_path):
            members = _extract_zip_members(
                archive_path,
                staging,
                max_members=max_members,
                max_member_expanded_bytes=max_member_expanded_bytes,
                max_expanded_bytes=max_expanded_bytes,
                chunk_size=chunk_size,
            )
        elif tarfile.is_tarfile(archive_path):
            members = _extract_tar_members(
                archive_path,
                staging,
                max_members=max_members,
                max_member_expanded_bytes=max_member_expanded_bytes,
                max_expanded_bytes=max_expanded_bytes,
                chunk_size=chunk_size,
            )
        else:
            raise ValueError(f"unsupported archive format: {archive_path.name}")

        if lexical_dest.is_symlink() or resolved_dest.is_symlink():
            raise ValueError("extraction destination must not be a symlink")
        if resolved_dest.exists():
            previous = Path(
                tempfile.mkdtemp(
                    prefix=f".{resolved_dest.name or 'extraction'}-old-",
                    dir=resolved_dest.parent,
                )
            )
            previous.rmdir()
            resolved_dest.replace(previous)
        try:
            staging.replace(resolved_dest)
        except Exception:
            if previous is not None and previous.exists() and not resolved_dest.exists():
                previous.replace(resolved_dest)
            raise
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
        staging = None
        return members
    except (OSError, RuntimeError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError(
            f"extraction failed for {archive_path.name}: {type(exc).__name__}"
        ) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if previous is not None and previous.exists():
            shutil.rmtree(previous, ignore_errors=True)


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


def _strict_erratum_checksum_map(bundle_root: Path) -> dict[str, str]:
    """Parse the canonical erratum checksum sidecar without permissive fallbacks.

    Returns:
        Exact bundle-root-relative payload paths and lowercase SHA-256 values.
    """
    path = bundle_root / ERRATUM_CHECKSUMS_ASSET
    if not path.is_file() or path.is_symlink():
        raise ValueError("erratum bundle lacks a safe checksums.sha256")
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2 or _SHA256_RE.fullmatch(parts[0]) is None:
            raise ValueError(f"erratum checksums.sha256:{line_number} is malformed")
        relative = parts[1].lstrip("*")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or relative_path.parts[0] != "payload"
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or "\\" in relative
            or "\x00" in relative
        ):
            raise ValueError(f"erratum checksums.sha256:{line_number} has an unsafe path")
        if relative in entries:
            raise ValueError(f"erratum checksums.sha256:{line_number} repeats {relative!r}")
        entries[relative] = parts[0].lower()
    if not entries:
        raise ValueError("erratum checksums.sha256 is empty")
    return entries


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    """Load a safe JSON object used by the public erratum proof.

    Returns:
        The parsed mapping.
    """
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _normalise_erratum_manifest_path(value: Any, *, label: str) -> str:
    """Return a safe payload-relative path from one manifest file entry."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is missing")
    relative = value.strip().removeprefix("payload/")
    candidate = Path(relative)
    if (
        not relative
        or candidate.is_absolute()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "\\" in relative
        or "\x00" in relative
    ):
        raise ValueError(f"{label} path is unsafe")
    return candidate.as_posix()


def _assert_erratum_url_mapping(
    mapping: Mapping[str, Any],
    *,
    label: str,
    tag: str,
    doi: str,
    repository_url: str,
    archive_name: str,
    require_coordinates: bool,
) -> None:
    """Bind present release/tag/DOI URL aliases to one requested publication."""
    tag_values = [mapping[key] for key in _ERRATUM_CURRENT_TAG_KEYS if key in mapping]
    doi_values = [mapping[key] for key in _ERRATUM_CURRENT_DOI_KEYS if key in mapping]
    if require_coordinates and not tag_values:
        raise ValueError(f"{label} is missing its release tag")
    if require_coordinates and not doi_values:
        raise ValueError(f"{label} is missing its version DOI")
    if require_coordinates and "release_url" not in mapping:
        raise ValueError(f"{label} is missing its release URL")
    if any(value != tag for value in tag_values):
        raise ValueError(f"{label} release tag is not bound to the requested tag")
    if any(value != doi for value in doi_values):
        raise ValueError(f"{label} version DOI is not bound to the requested DOI")

    expected_release_url = f"{repository_url}/releases/tag/{tag}"
    expected_asset_url = f"{repository_url}/releases/download/{tag}/{archive_name}"
    expected_doi_url = f"https://doi.org/{doi}"
    url_values = {
        "release_url": expected_release_url,
        "release_asset_url": expected_asset_url,
        "doi_url": expected_doi_url,
    }
    for key, expected in url_values.items():
        if key in mapping and mapping[key] != expected:
            raise ValueError(f"{label}.{key} is not bound to the requested release")


def _verify_erratum_manifest_coordinates(
    manifest: Mapping[str, Any], *, bundle_root: Path, archive_name: str, tag: str, doi: str
) -> None:
    """Validate publication-channel, bundle, and release URL identity."""
    channels = manifest.get("publication_channels")
    if not isinstance(channels, Mapping):
        raise ValueError("erratum publication manifest publication_channels is missing")
    repository_url = channels.get("repository_url")
    if repository_url != ERRATUM_REPOSITORY_URL:
        raise ValueError("erratum publication channels repository URL is not canonical")
    _assert_erratum_url_mapping(
        channels,
        label="erratum publication_channels",
        tag=tag,
        doi=doi,
        repository_url=ERRATUM_REPOSITORY_URL,
        archive_name=archive_name,
        require_coordinates=True,
    )
    if manifest.get("bundle_name") != bundle_root.name:
        raise ValueError("erratum publication manifest bundle name is stale")
    for key in ("bundle", "release"):
        value = manifest.get(key)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"erratum publication manifest {key} must be an object")
        _assert_erratum_url_mapping(
            value,
            label=f"erratum publication manifest {key}",
            tag=tag,
            doi=doi,
            repository_url=ERRATUM_REPOSITORY_URL,
            archive_name=archive_name,
            require_coordinates=False,
        )


def _verify_erratum_manifest_entry(
    raw_entry: Any,
    *,
    index: int,
    bundle_root: Path,
    payload_files: set[str],
    checksums: Mapping[str, str],
) -> tuple[str, int]:
    """Validate one publication-manifest entry against its payload bytes.

    Returns:
        The normalized payload path and its verified byte size.
    """
    label = f"erratum publication manifest files[{index}]"
    if not isinstance(raw_entry, Mapping):
        raise ValueError(f"{label} must be an object")
    relative = _normalise_erratum_manifest_path(raw_entry.get("path"), label=label)
    if relative not in {path.removeprefix("payload/") for path in payload_files}:
        raise ValueError(f"{label} names a missing payload file: {relative}")
    path = bundle_root / "payload" / relative
    declared_size = raw_entry.get("size_bytes")
    if isinstance(declared_size, bool) or not isinstance(declared_size, int) or declared_size < 0:
        raise ValueError(f"{label} has an invalid size_bytes value")
    actual_size = path.stat().st_size
    if declared_size != actual_size:
        raise ValueError(f"{label} size_bytes disagrees with payload bytes for {relative}")
    declared_digest = raw_entry.get("sha256")
    if not isinstance(declared_digest, str) or _SHA256_RE.fullmatch(declared_digest) is None:
        raise ValueError(f"{label} has an invalid sha256 value")
    actual_digest = sha256_file(path)
    if declared_digest.lower() != actual_digest:
        raise ValueError(f"{label} sha256 disagrees with payload bytes for {relative}")
    checksum_path = f"payload/{relative}"
    if checksums.get(checksum_path) != actual_digest:
        raise ValueError(f"{label} sha256 disagrees with checksums.sha256 for {relative}")
    return relative, declared_size


def _verify_erratum_manifest_files(
    manifest: Mapping[str, Any],
    *,
    bundle_root: Path,
    payload_files: set[str],
    checksums: Mapping[str, str],
) -> dict[str, Any]:
    """Authenticate every publication-manifest file entry against payload bytes.

    Returns:
        Verified file-count and byte-total observations.
    """
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, list):
        raise ValueError("erratum publication manifest files must be a list")
    payload_relative_files = {path.removeprefix("payload/") for path in payload_files}
    observed_entries: set[str] = set()
    declared_bytes = 0
    for index, raw_entry in enumerate(manifest_files):
        label = f"erratum publication manifest files[{index}]"
        relative, declared_size = _verify_erratum_manifest_entry(
            raw_entry,
            index=index,
            bundle_root=bundle_root,
            payload_files=payload_files,
            checksums=checksums,
        )
        if relative in observed_entries:
            raise ValueError(f"{label} repeats {relative!r}")
        observed_entries.add(relative)
        declared_bytes += declared_size

    if observed_entries != payload_relative_files:
        raise ValueError("erratum publication manifest files differ from the payload inventory")
    totals = manifest.get("totals")
    if not isinstance(totals, Mapping):
        raise ValueError("erratum publication manifest totals must be an object")
    if (
        totals.get("file_count") != len(observed_entries)
        or totals.get("total_bytes") != declared_bytes
    ):
        raise ValueError("erratum publication manifest totals disagree with file entries")
    return {
        "file_count": len(observed_entries),
        "total_bytes": declared_bytes,
    }


def _assert_cold_current_aliases(
    payload: Mapping[str, Any],
    *,
    label: str,
    tag: str,
    doi: str,
    concept_doi: str,
    required: bool,
) -> None:
    """Reject predecessor coordinates in one current-publication mapping."""
    tag_values = [payload[key] for key in _ERRATUM_CURRENT_TAG_KEYS if key in payload]
    doi_values = [payload[key] for key in _ERRATUM_CURRENT_DOI_KEYS if key in payload]
    if required and not tag_values:
        raise ValueError(f"{label} is missing its current release tag")
    if required and not doi_values:
        raise ValueError(f"{label} is missing its current version DOI")
    if any(value != tag for value in tag_values):
        raise ValueError(f"{label} contains a stale release-tag alias")
    if any(value != doi for value in doi_values):
        raise ValueError(f"{label} contains a stale version-DOI alias")
    if "concept_doi" in payload and payload["concept_doi"] != concept_doi:
        raise ValueError(f"{label} contains an invalid concept-DOI alias")
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        _assert_cold_current_aliases(
            provenance,
            label=f"{label}.provenance",
            tag=tag,
            doi=doi,
            concept_doi=concept_doi,
            required=False,
        )


def _assert_cold_publication_mapping(
    payload: Mapping[str, Any],
    *,
    label: str,
    tag: str,
    doi: str,
    concept_doi: str,
    predecessor_doi: str,
    required: bool,
) -> None:
    """Validate a nested ``publication`` mapping in a downloaded document."""
    publication = payload.get("publication")
    if publication is None and not required:
        return
    if not isinstance(publication, Mapping):
        raise ValueError(f"{label}.publication must be an object")
    _assert_cold_current_aliases(
        publication,
        label=f"{label}.publication",
        tag=tag,
        doi=doi,
        concept_doi=concept_doi,
        required=True,
    )
    if publication.get("predecessor_version_doi") != predecessor_doi:
        raise ValueError(f"{label}.publication contains a stale predecessor DOI alias")


def _assert_cold_predecessor_aliases(
    payload: Mapping[str, Any],
    *,
    label: str,
    predecessor_tag: str,
    predecessor_doi: str,
    concept_doi: str,
    source_sha: str,
) -> None:
    """Validate an explicitly preserved scientific-execution identity."""
    tags = [payload[key] for key in _ERRATUM_CURRENT_TAG_KEYS if key in payload]
    dois = [payload[key] for key in _ERRATUM_CURRENT_DOI_KEYS if key in payload]
    if not tags or any(value != predecessor_tag for value in tags):
        raise ValueError(f"{label} contains an invalid predecessor tag alias")
    if not dois or any(value != predecessor_doi for value in dois):
        raise ValueError(f"{label} contains an invalid predecessor DOI alias")
    if "concept_doi" in payload and payload["concept_doi"] != concept_doi:
        raise ValueError(f"{label} contains an invalid predecessor concept DOI")
    for key in ("source_sha", "source_commit", "scientific_source_sha"):
        if key in payload and payload[key] != source_sha:
            raise ValueError(f"{label} contains an invalid scientific source SHA")
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping) and any(
        key in provenance
        for key in (*_ERRATUM_CURRENT_TAG_KEYS, *_ERRATUM_CURRENT_DOI_KEYS, "concept_doi")
    ):
        _assert_cold_predecessor_aliases(
            provenance,
            label=f"{label}.provenance",
            predecessor_tag=predecessor_tag,
            predecessor_doi=predecessor_doi,
            concept_doi=concept_doi,
            source_sha=source_sha,
        )


def _assert_cold_publication_document(
    payload: Mapping[str, Any],
    *,
    label: str,
    tag: str,
    doi: str,
    predecessor_doi: str,
    predecessor_tag: str,
    concept_doi: str,
    source_sha: str,
) -> None:
    """Audit current and explicitly preserved identities in one JSON document."""
    _assert_cold_current_aliases(
        payload,
        label=label,
        tag=tag,
        doi=doi,
        concept_doi=concept_doi,
        required=False,
    )
    _assert_cold_publication_mapping(
        payload,
        label=label,
        tag=tag,
        doi=doi,
        concept_doi=concept_doi,
        predecessor_doi=predecessor_doi,
        required=False,
    )
    for key in ("benchmark_release", "resolved_manifest", "campaign"):
        current = payload.get(key)
        if not isinstance(current, Mapping):
            continue
        _assert_cold_current_aliases(
            current,
            label=f"{label}.{key}",
            tag=tag,
            doi=doi,
            concept_doi=concept_doi,
            required=True,
        )
        _assert_cold_publication_mapping(
            current,
            label=f"{label}.{key}",
            tag=tag,
            doi=doi,
            concept_doi=concept_doi,
            predecessor_doi=predecessor_doi,
            required=True,
        )
    for key in (
        "scientific_execution_benchmark_release",
        "scientific_execution_resolved_manifest",
        "scientific_execution_release_identity",
    ):
        execution = payload.get(key)
        if isinstance(execution, Mapping):
            _assert_cold_predecessor_aliases(
                execution,
                label=f"{label}.{key}",
                predecessor_tag=predecessor_tag,
                predecessor_doi=predecessor_doi,
                concept_doi=concept_doi,
                source_sha=source_sha,
            )


def _verify_erratum_cold_publication_documents(
    campaign_root: Path,
    *,
    tag: str,
    doi: str,
    predecessor_tag: str,
    predecessor_doi: str,
    concept_doi: str,
    source_sha: str,
) -> dict[str, Any]:
    """Audit copied optional/current release documents after archive extraction.

    Returns:
        The list of copied documents checked successfully.
    """
    checked: list[str] = []
    relative_documents = (
        *_ERRATUM_OPTIONAL_DOCUMENTS,
        "release/release_manifest.resolved.json",
        "release/release_result.json",
        "reports/campaign_summary.json",
    )
    for relative in relative_documents:
        path = campaign_root / relative
        if path.is_symlink():
            raise ValueError(f"published {relative} must not be a symlink")
        if not path.exists():
            continue
        if not path.is_file():
            raise ValueError(f"published {relative} is not a regular file")
        document = _read_json_mapping(path, label=f"published {relative}")
        _assert_cold_publication_document(
            document,
            label=f"published {relative}",
            tag=tag,
            doi=doi,
            predecessor_doi=predecessor_doi,
            predecessor_tag=predecessor_tag,
            concept_doi=concept_doi,
            source_sha=source_sha,
        )
        checked.append(relative)
    return {"status": "pass", "checked_documents": checked}


def _verify_erratum_bundle_inventory(
    bundle_root: Path, *, archive_name: str, tag: str, doi: str
) -> dict[str, Any]:
    """Authenticate the exact manifest/checksum/payload member inventory.

    Returns:
        Compact inventory evidence for the public audit receipt.
    """
    try:
        preflight = verify_publication_bundle_preflight(bundle_root)
    except PublicationPreflightError as exc:
        raise ValueError(f"erratum publication preflight failed: {exc}") from exc
    manifest_path = bundle_root / ERRATUM_MANIFEST_ASSET
    manifest = _read_json_mapping(manifest_path, label="erratum publication manifest")
    if manifest.get("schema_version") != PUBLICATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError("erratum publication manifest schema is unsupported")
    checksums = _strict_erratum_checksum_map(bundle_root)

    payload_root = bundle_root / "payload"
    if not payload_root.is_dir() or payload_root.is_symlink():
        raise ValueError("erratum bundle payload directory is missing or unsafe")
    payload_files = {
        path.relative_to(bundle_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if set(checksums) != payload_files:
        missing = sorted(payload_files - set(checksums))
        unexpected = sorted(set(checksums) - payload_files)
        raise ValueError(
            "erratum payload/checksum inventory differs "
            f"(unsigned={len(missing)}, missing={len(unexpected)})"
        )

    allowed_root_files = {
        ERRATUM_MANIFEST_ASSET,
        ERRATUM_CHECKSUMS_ASSET,
        "README.md",
    }
    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_files = payload_files | allowed_root_files
    if actual_files != expected_files:
        raise ValueError("erratum archive contains an unlisted or missing bundle member")
    _verify_erratum_manifest_coordinates(
        manifest, bundle_root=bundle_root, archive_name=archive_name, tag=tag, doi=doi
    )
    manifest_totals = _verify_erratum_manifest_files(
        manifest,
        bundle_root=bundle_root,
        payload_files=payload_files,
        checksums=checksums,
    )
    return {
        "status": "pass",
        "preflight_status": preflight.get("status"),
        "payload_file_count": len(payload_files),
        "payload_bytes": manifest_totals["total_bytes"],
        "manifest_file_count": manifest_totals["file_count"],
        "publication_manifest_sha256": sha256_file(manifest_path),
        "checksums_sha256": sha256_file(bundle_root / ERRATUM_CHECKSUMS_ASSET),
    }


def _verify_erratum_custody(
    *,
    custody_path: Path,
    bundle: Path,
    bundle_root: Path,
    receipt_path: Path,
    source_sha: str,
) -> dict[str, Any]:
    """Validate the detached receipt that binds archive and internal sidecars.

    Returns:
        Compact custody evidence for the public audit receipt.
    """
    custody = _read_json_mapping(custody_path, label="erratum publication custody receipt")
    if custody.get("schema_version") != "benchmark-publication-custody.v1":
        raise ValueError("erratum publication custody schema is unsupported")
    if custody.get("source_execution_commit") != source_sha:
        raise ValueError("erratum custody source commit differs from the GitHub tag target")
    if custody.get("credentials") != "not_recorded":
        raise ValueError("erratum custody receipt has an invalid credential policy")

    archive = custody.get("archive")
    bundle_block = custody.get("bundle")
    erratum = custody.get("erratum")
    if not all(isinstance(value, Mapping) for value in (archive, bundle_block, erratum)):
        raise ValueError("erratum custody receipt is incomplete")
    if (
        archive.get("path") != bundle.name
        or archive.get("sha256") != sha256_file(bundle)
        or archive.get("size_bytes") != bundle.stat().st_size
    ):
        raise ValueError("erratum custody archive identity is stale")
    if (
        bundle_block.get("path") != bundle_root.name
        or bundle_block.get("publication_manifest_sha256")
        != sha256_file(bundle_root / ERRATUM_MANIFEST_ASSET)
        or bundle_block.get("checksums_sha256")
        != sha256_file(bundle_root / ERRATUM_CHECKSUMS_ASSET)
    ):
        raise ValueError("erratum custody bundle identity is stale")

    receipt = _read_json_mapping(receipt_path, label="embedded erratum receipt")
    expected_receipt_path = f"{bundle_root.name}/payload/provenance/benchmark_release_erratum.json"
    if (
        erratum.get("embedded_receipt_path") != expected_receipt_path
        or erratum.get("embedded_receipt_sha256") != sha256_file(receipt_path)
        or erratum.get("correction_id") != receipt.get("correction_id")
        or erratum.get("correction_scope") != receipt.get("correction_scope")
        or erratum.get("supersedes") != receipt.get("supersedes")
        or erratum.get("successor") != receipt.get("successor")
        or erratum.get("scientific_equality") != receipt.get("scientific_equality")
    ):
        raise ValueError("erratum custody receipt does not bind the embedded correction proof")
    return {
        "status": "pass",
        "archive_sha256": archive["sha256"],
        "archive_size_bytes": archive["size_bytes"],
        "custody_sha256": sha256_file(custody_path),
    }


def _check_tag_source(tag: str, source_sha: str) -> list[str]:
    """Enforce the prospective tag/source-SHA contract (issue #7938).

    Uses the canonical ``release_tag_identity`` helper when available; falls
    back to an inline 40-hex suffix comparison on older bases.

    Returns:
        Problem strings; empty when the tag is consistent.
    """
    if _release_tag_identity is not None:
        if (
            _release_tag_identity.is_historical_release_tag(tag)
            and source_sha == _release_tag_identity.HISTORICAL_RELEASE_SOURCE_SHA
        ):
            return []
        return _release_tag_identity.check_tag_source_consistency(tag, source_sha)
    suffix_match = re.search(r"[_-](?P<sha>[0-9a-f]{40})$", tag)
    if suffix_match and suffix_match.group("sha") != source_sha:
        return [
            f"tag SHA component {suffix_match.group('sha')!r} disagrees with "
            f"source_sha {source_sha!r}"
        ]
    return []


def _channel_assets(channel_dir: Path, *, channel: str, problems: list[str]) -> list[Path]:
    """Return the asset files of a channel, or [] when the channel is absent.

    Returns:
        Sorted asset file paths; empty when the channel directory is absent.
    """
    if channel_dir.is_symlink():
        problems.append(f"{channel} channel directory must not be a symlink")
        return []
    if not channel_dir.is_dir():
        return []
    assets: list[Path] = []
    for path in sorted(channel_dir.iterdir()):
        if path.is_symlink():
            problems.append(f"{channel} channel asset {path.name} must not be a symlink")
        elif path.is_file():
            assets.append(path)
    return assets


def _erratum_bundle_root(extracted: Path, members: list[str]) -> tuple[Path, list[str]]:
    """Return the sole canonical archive root and its relative members."""
    member_parts = [Path(name).parts for name in members if Path(name).parts]
    roots = {parts[0] for parts in member_parts}
    if len(roots) != 1:
        raise ValueError("erratum bundle must contain exactly one archive root")
    root_name = next(iter(roots))
    return extracted / root_name, [
        Path(*parts[1:]).as_posix() for parts in member_parts if len(parts) > 1
    ]


def _erratum_external_assets(
    github_assets: list[Path], *, source_sha: str | None
) -> dict[str, Path]:
    """Require the detached erratum sidecars and exact tag target.

    Returns:
        The external assets keyed by file name.
    """
    assets_by_name = {path.name: path for path in github_assets}
    required_sidecars = {
        ERRATUM_MANIFEST_ASSET,
        ERRATUM_CHECKSUMS_ASSET,
        ERRATUM_CUSTODY_ASSET,
    }
    if not required_sidecars.issubset(assets_by_name):
        raise ValueError(
            "erratum publication requires external manifest, checksums, and custody assets"
        )
    if source_sha is None:
        raise ValueError("erratum publication requires the exact GitHub tag target SHA")
    return assets_by_name


def _compare_erratum_sidecars(bundle_root: Path, assets_by_name: Mapping[str, Path]) -> None:
    """Require public manifest/checksum sidecars to match archive bytes."""
    for sidecar_name in (ERRATUM_MANIFEST_ASSET, ERRATUM_CHECKSUMS_ASSET):
        internal = bundle_root / sidecar_name
        try:
            matches = internal.read_bytes() == assets_by_name[sidecar_name].read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot compare erratum {sidecar_name}: {exc}") from exc
        if not matches:
            raise ValueError(f"external {sidecar_name} differs from the archived sidecar")


def _erratum_identity_paths(bundle_root: Path, relative_members: list[str]) -> tuple[Path, Path]:
    """Require unique canonical receipt and metadata paths in the signed payload.

    Returns:
        The canonical receipt and Zenodo metadata paths.
    """
    receipt_relative = "payload/provenance/benchmark_release_erratum.json"
    metadata_relative = "payload/release/zenodo_metadata.erratum.json"
    required_members = {receipt_relative, metadata_relative}
    identity_members = {
        name
        for name in relative_members
        if Path(name).name in {"benchmark_release_erratum.json", "zenodo_metadata.erratum.json"}
    }
    if identity_members != required_members:
        raise ValueError("erratum bundle lacks the canonical receipt or metadata path")
    checksum_map = _strict_erratum_checksum_map(bundle_root)
    if not required_members.issubset(checksum_map):
        raise ValueError("erratum receipt and metadata must be listed in checksums.sha256")
    return bundle_root / receipt_relative, bundle_root / metadata_relative


def _verify_canonical_erratum_bundle(
    *,
    bundle: Path,
    extracted: Path,
    members: list[str],
    github_assets: list[Path],
    tag: str,
    doi: str,
    source_sha: str | None,
    predecessor_evidence: PredecessorEvidence | None = None,
) -> dict[str, Any]:
    """Verify one complete canonical erratum archive and detached proof set.

    Returns:
        The authenticated inventory, correction receipt, and custody evidence.
    """
    bundle_root, relative_members = _erratum_bundle_root(extracted, members)
    internal_problems = _verify_internal_checksums(bundle_root, relative_members)
    if internal_problems:
        raise ValueError("; ".join(internal_problems))
    if source_sha is None:
        raise ValueError("erratum publication requires the exact GitHub tag target SHA")
    assets_by_name = _erratum_external_assets(github_assets, source_sha=source_sha)
    _compare_erratum_sidecars(bundle_root, assets_by_name)
    inventory = _verify_erratum_bundle_inventory(
        bundle_root, archive_name=bundle.name, tag=tag, doi=doi
    )
    receipt_path, metadata_path = _erratum_identity_paths(bundle_root, relative_members)
    try:
        receipt = validate_erratum_receipt_against_campaign(
            receipt_path,
            campaign_root=bundle_root / "payload",
            metadata_path=metadata_path,
            predecessor_evidence=predecessor_evidence,
            archive_name=bundle.name,
            expected_tag=tag,
            expected_doi=doi,
            expected_source_sha=source_sha,
        )
    except ReleaseErratumError as exc:
        raise ValueError(f"erratum receipt validation failed: {exc}") from exc
    custody = _verify_erratum_custody(
        custody_path=assets_by_name[ERRATUM_CUSTODY_ASSET],
        bundle=bundle,
        bundle_root=bundle_root,
        receipt_path=receipt_path,
        source_sha=source_sha,
    )
    embedded_receipt = _read_json_mapping(receipt_path, label="embedded erratum receipt")
    supersedes = embedded_receipt.get("supersedes")
    if not isinstance(supersedes, Mapping):
        raise ValueError("embedded erratum receipt lacks predecessor identity")
    predecessor_tag = supersedes.get("github_release_tag")
    predecessor_doi = (
        receipt.get("predecessor_version_doi")
        or embedded_receipt.get("predecessor_version_doi")
        or supersedes.get("version_doi")
    )
    concept_doi = receipt.get("concept_doi") or embedded_receipt.get("concept_doi")
    if not isinstance(predecessor_tag, str) or not isinstance(predecessor_doi, str):
        raise ValueError("embedded erratum receipt has an invalid predecessor identity")
    if not isinstance(concept_doi, str):
        raise ValueError("embedded erratum receipt has an invalid concept DOI")
    cold_documents = _verify_erratum_cold_publication_documents(
        bundle_root / "payload",
        tag=tag,
        doi=doi,
        predecessor_tag=predecessor_tag,
        predecessor_doi=predecessor_doi,
        concept_doi=concept_doi,
        source_sha=source_sha,
    )
    return {
        "inventory": inventory,
        "receipt": receipt,
        "cold_documents": cold_documents,
        "custody": custody,
    }


def _verify_bundle(
    github_assets: list[Path],
    github_dir: Path,
    observations: dict[str, Any],
    problems: list[str],
    *,
    tag: str,
    doi: str,
    source_sha: str | None,
    predecessor_evidence: PredecessorEvidence | None = None,
) -> None:
    """Extract the largest archive and verify internal checksums.

    Observations and problems are updated in place.
    """
    bundle_candidates = [path for path in github_assets if path.name.endswith(_ARCHIVE_SUFFIXES)]
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
        erratum_marker = "-erratum." in tag.casefold()
        canonical_erratum = re.fullmatch(r".+-[0-9a-f]{40}-erratum\.[1-9][0-9]*", tag)
        if erratum_marker and canonical_erratum is None:
            problems.append("erratum tag is malformed or does not carry one lowercase source SHA")
        elif canonical_erratum is not None:
            proof = _verify_canonical_erratum_bundle(
                bundle=bundle,
                extracted=extracted,
                members=members,
                github_assets=github_assets,
                tag=tag,
                doi=doi,
                source_sha=source_sha,
                predecessor_evidence=predecessor_evidence,
            )
            observations["erratum_bundle_inventory"] = proof["inventory"]
            observations["erratum"] = proof["receipt"]
            observations["erratum_cold_documents"] = proof["cold_documents"]
            observations["erratum_custody"] = proof["custody"]
    except RecursionError as exc:
        detail = str(exc) or "recursion depth exceeded"
        problems.append(f"erratum payload traversal exceeded the recursion limit: {detail}")
    except (OSError, ValueError, ReleaseErratumError) as exc:
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


def _check_erratum_channel_assets(
    *,
    tag: str,
    github_by_name: Mapping[str, Path],
    zenodo_by_name: Mapping[str, Path],
) -> list[str]:
    """Return fail-closed two-channel inventory problems for canonical errata."""
    if re.fullmatch(r".+-[0-9a-f]{40}-erratum\.[1-9][0-9]*", tag) is None:
        return []
    problems: list[str] = []
    if set(github_by_name) != set(zenodo_by_name):
        problems.append("erratum GitHub and Zenodo asset inventories must be identical")
    required = {ERRATUM_MANIFEST_ASSET, ERRATUM_CHECKSUMS_ASSET, ERRATUM_CUSTODY_ASSET}
    for channel, assets in (("GitHub", github_by_name), ("Zenodo", zenodo_by_name)):
        if not required.issubset(assets):
            problems.append(
                f"{channel} erratum assets lack manifest, checksums, or custody receipt"
            )
        archives = [name for name in assets if name.endswith(_ARCHIVE_SUFFIXES)]
        if len(archives) != 1:
            problems.append(f"{channel} erratum assets must contain exactly one archive")
    return problems


def audit_published(
    *,
    tag: str,
    doi: str,
    github_dir: Path,
    zenodo_dir: Path,
    source_sha: str | None = None,
    predecessor_evidence: PredecessorEvidence | None = None,
) -> dict[str, Any]:
    """Audit two downloaded asset directories for cross-channel identity.

    Returns:
        The versioned audit receipt.
    """
    problems: list[str] = []
    observations: dict[str, Any] = {}

    github_assets = _channel_assets(github_dir, channel="GitHub", problems=problems)
    zenodo_assets = _channel_assets(zenodo_dir, channel="Zenodo", problems=problems)
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

    problems.extend(
        _check_erratum_channel_assets(
            tag=tag,
            github_by_name=github_by_name,
            zenodo_by_name=zenodo_by_name,
        )
    )

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

    _verify_bundle(
        github_assets,
        github_dir,
        observations,
        problems,
        tag=tag,
        doi=doi,
        source_sha=source_sha,
        predecessor_evidence=predecessor_evidence,
    )
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


def _canonical_erratum_predecessor_tag(tag: str) -> str | None:
    """Return the predecessor tag encoded by an exact ``-erratum.1`` suffix."""
    match = _CANONICAL_ERRATUM_TAG_RE.fullmatch(tag)
    return match.group("predecessor") if match is not None else None


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
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise PublishedAuditInvalid(
                f"published Zenodo file {name} requires a positive advertised size"
            )
        assets.append({"name": name, "url": url, "size": size, "digest": None})
    return assets


def _require_zenodo_source_relation(related: object, *, source_tag_url: str) -> list[object]:
    """Require one exact URL-scheme relation to the GitHub release tag.

    Returns:
        The validated related-identifier list.
    """
    if not isinstance(related, list):
        raise PublishedAuditInvalid("Zenodo record is not related to the requested GitHub release")
    source_matches = [
        item
        for item in related
        if isinstance(item, Mapping)
        and item.get("relation") == "isSupplementTo"
        and item.get("identifier") == source_tag_url
    ]
    if len(source_matches) != 1 or source_matches[0].get("scheme") != "url":
        raise PublishedAuditInvalid("Zenodo record is not related to the requested GitHub release")
    return related


def _zenodo_predecessor_relation(
    related: list[object],
    *,
    source_tag_url: str,
    doi: str,
    concept_doi: str,
) -> str | None:
    """Validate and return the sole predecessor version relation, when present.

    Returns:
        The predecessor DOI, or ``None`` for a non-erratum without that relation.
    """
    relations = [
        item
        for item in related
        if isinstance(item, Mapping) and item.get("relation") == "isNewVersionOf"
    ]
    predecessor_doi = None
    if relations:
        candidate = relations[0]
        identifier = str(candidate.get("identifier") or "")
        if (
            len(relations) != 1
            or candidate.get("scheme") != "doi"
            or _DOI_RE.fullmatch(identifier) is None
        ):
            raise PublishedAuditInvalid("Zenodo predecessor-version relation is malformed")
        predecessor_doi = identifier
    if "-erratum." in source_tag_url and (
        predecessor_doi is None or predecessor_doi in {doi, concept_doi}
    ):
        raise PublishedAuditInvalid(
            "Zenodo erratum metadata lacks one distinct predecessor version DOI"
        )
    return predecessor_doi


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
    payload_concept_doi = str(payload.get("conceptdoi") or "").strip()
    metadata_concept_doi = str(metadata.get("conceptdoi") or "").strip()
    if payload_concept_doi and metadata_concept_doi and payload_concept_doi != metadata_concept_doi:
        raise PublishedAuditInvalid("Zenodo record concept DOI differs between record and metadata")
    concept_doi = payload_concept_doi or metadata_concept_doi
    if _DOI_RE.fullmatch(concept_doi) is None or concept_doi == doi:
        raise PublishedAuditInvalid("Zenodo record concept DOI is missing or incorrect")
    status = str(payload.get("status") or "").casefold()
    state = str(payload.get("state") or "").casefold()
    if (status and status != "published") or (not status and state != "done"):
        raise PublishedAuditInvalid("Zenodo record is not a published version")
    related = _require_zenodo_source_relation(
        metadata.get("related_identifiers"), source_tag_url=source_tag_url
    )
    predecessor_doi = _zenodo_predecessor_relation(
        related,
        source_tag_url=source_tag_url,
        doi=doi,
        concept_doi=concept_doi,
    )
    return {
        "id": payload.get("id") or record_id,
        "doi": doi,
        "concept_doi": concept_doi,
        "predecessor_doi": predecessor_doi,
        "status": status or state,
        "assets": _zenodo_file_assets(payload),
    }


def _require_single_common_archive(
    github_assets: list[dict[str, Any]],
    zenodo_assets: list[dict[str, Any]],
) -> str:
    """Return the one archive shared by independently resolved channels."""
    github_names = {asset["name"] for asset in github_assets}
    zenodo_names = {asset["name"] for asset in zenodo_assets}
    archive_names = sorted(
        name for name in github_names & zenodo_names if name.endswith(_ARCHIVE_SUFFIXES)
    )
    if not archive_names:
        raise PublishedAuditInvalid(
            "predecessor GitHub and Zenodo have no common predecessor archive"
        )
    if len(archive_names) != 1:
        raise PublishedAuditInvalid(
            "predecessor GitHub and Zenodo must have exactly one common predecessor archive"
        )
    return archive_names[0]


def _require_named_asset(assets: list[dict[str, Any]], *, name: str, label: str) -> dict[str, Any]:
    """Return one normalized asset by name or fail with an invalid receipt."""
    for asset in assets:
        if asset.get("name") == name:
            return asset
    raise PublishedAuditInvalid(f"{label} asset {name} is missing")


def _require_positive_advertised_size(asset: Mapping[str, Any], *, label: str) -> int:
    """Require a positive public size before downloading one protected archive.

    Returns:
        The validated advertised byte count.
    """
    size = asset.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise PublishedAuditInvalid(f"{label} requires a positive advertised size")
    return size


def _require_cross_channel_archive_identity(
    github_download: Mapping[str, Any], zenodo_download: Mapping[str, Any]
) -> None:
    """Require matching observed size and digest for a predecessor archive."""
    if github_download.get("bytes") != zenodo_download.get("bytes"):
        raise PublishedAuditInvalid(
            "predecessor archive size mismatch across GitHub and Zenodo channels"
        )
    if github_download.get("sha256") != zenodo_download.get("sha256"):
        raise PublishedAuditInvalid(
            "predecessor archive digest mismatch across GitHub and Zenodo channels"
        )


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


def _sanitize_network_text(value: str, *, private_root: Path | None) -> str:
    """Replace temporary audit paths with stable bundle-relative labels.

    Returns:
        Sanitized text.
    """
    if private_root is None:
        return value
    roots = (private_root, private_root.resolve())
    replacements = tuple(
        (root / suffix, public_path)
        for root in roots
        for suffix, public_path in (
            (Path("github/_extracted"), "github-bundle"),
            (Path("github"), "github"),
            (Path("zenodo"), "zenodo"),
            (Path("predecessor-github"), "predecessor-github"),
            (Path("predecessor-zenodo"), "predecessor-zenodo"),
            (Path(), "audit"),
        )
    )
    sanitized = value
    for private_path, public_path in replacements:
        sanitized = sanitized.replace(str(private_path), public_path)
    return sanitized


def _sanitize_network_value(value: Any, *, private_root: Path | None) -> Any:
    """Recursively remove private temporary paths from a receipt payload.

    Returns:
        A JSON-compatible value with temporary paths sanitized.
    """
    if isinstance(value, str):
        return _sanitize_network_text(value, private_root=private_root)
    if isinstance(value, Mapping):
        return {
            _sanitize_network_text(str(key), private_root=private_root): _sanitize_network_value(
                nested, private_root=private_root
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_network_value(item, private_root=private_root) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_network_value(item, private_root=private_root) for item in value]
    return value


def _failure_network_receipt(
    *,
    tag: str,
    doi: str,
    status: str,
    problem: str,
    discovery: Mapping[str, Any] | None = None,
    predecessor: Mapping[str, Any] | None = None,
    private_root: Path | None = None,
) -> dict[str, Any]:
    """Build a stable credential-free receipt for a non-pass network audit.

    Returns:
        A failure receipt without temporary paths or request metadata.
    """
    receipt = {
        "schema": NETWORK_SCHEMA,
        "ok": False,
        "status": status,
        "tag": _receipt_identifier(tag, kind="tag"),
        "doi": _receipt_identifier(doi, kind="doi"),
        "source_sha": None,
        "predecessor": dict(predecessor) if predecessor is not None else None,
        "problems": [problem],
        "discovery": dict(discovery or {}),
        "downloads": {"github": [], "zenodo": [], "bytes": 0},
        "audit": None,
    }
    return _sanitize_network_value(receipt, private_root=private_root)


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


def _reconcile_zenodo_erratum_lineage(
    core: dict[str, Any], *, tag: str, zenodo: Mapping[str, Any]
) -> None:
    """Fail the core audit when public lineage and embedded proof disagree."""
    if "-erratum." not in tag:
        return
    observations = core.get("observations")
    erratum = observations.get("erratum") if isinstance(observations, Mapping) else None
    if isinstance(erratum, Mapping) and (
        erratum.get("predecessor_version_doi") == zenodo.get("predecessor_doi")
        and erratum.get("concept_doi") == zenodo.get("concept_doi")
    ):
        return
    core["problems"].append("Zenodo API lineage differs from the embedded erratum receipt")
    core["ok"] = False
    core["status"] = "fail"


def audit_published_network(  # noqa: C901, PLR0912, PLR0913, PLR0915
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
    predecessor_receipt: dict[str, Any] | None = None
    private_temp_root: Path | None = None
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
        canonical_predecessor_tag = _canonical_erratum_predecessor_tag(requested_tag)
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
            "predecessor_doi": zenodo["predecessor_doi"],
            "asset_names": sorted(asset["name"] for asset in zenodo["assets"]),
        }

        predecessor_github: dict[str, Any] | None = None
        predecessor_zenodo: dict[str, Any] | None = None
        predecessor_archive_name: str | None = None
        if canonical_predecessor_tag is not None:
            predecessor_doi = zenodo.get("predecessor_doi")
            if not isinstance(predecessor_doi, str) or not predecessor_doi:
                raise PublishedAuditInvalid(
                    "canonical erratum successor lacks its predecessor version DOI"
                )
            predecessor_source_tag_url = (
                f"https://github.com/{repo}/releases/tag/{canonical_predecessor_tag}"
            )
            predecessor_github = _resolve_github_release(
                public_session,
                api_base=github_base,
                repo=repo,
                tag=canonical_predecessor_tag,
                timeout=timeout,
            )
            if predecessor_github["source_sha"] != github["source_sha"]:
                raise PublishedAuditInvalid(
                    "predecessor GitHub tag source SHA differs from the successor source SHA"
                )
            predecessor_zenodo = _resolve_zenodo_record(
                public_session,
                api_base=zenodo_base,
                doi=predecessor_doi,
                source_tag_url=predecessor_source_tag_url,
                timeout=timeout,
            )
            if predecessor_zenodo["concept_doi"] != zenodo["concept_doi"]:
                raise PublishedAuditInvalid(
                    "predecessor Zenodo concept DOI differs from the successor concept DOI"
                )
            predecessor_github_by_name = {
                asset["name"]: asset for asset in predecessor_github["assets"]
            }
            predecessor_zenodo_by_name = {
                asset["name"]: asset for asset in predecessor_zenodo["assets"]
            }
            if not set(predecessor_zenodo_by_name).issubset(predecessor_github_by_name):
                raise PublishedAuditInvalid(
                    "predecessor Zenodo files must be named public GitHub release assets"
                )
            predecessor_archive_name = _require_single_common_archive(
                predecessor_github["assets"], predecessor_zenodo["assets"]
            )
            discovery["predecessor"] = {
                "record_id": predecessor_zenodo["id"],
                "doi": predecessor_zenodo["doi"],
                "concept_doi": predecessor_zenodo["concept_doi"],
                "tag": predecessor_github["tag"],
                "source_sha": predecessor_github["source_sha"],
                "asset_names": sorted(predecessor_github_by_name),
                "archive_name": predecessor_archive_name,
            }
        github_by_name = {asset["name"]: asset for asset in github["assets"]}
        zenodo_by_name = {asset["name"]: asset for asset in zenodo["assets"]}
        if not set(zenodo_by_name).issubset(github_by_name):
            raise PublishedAuditInvalid("Zenodo files must be named public GitHub release assets")
        common_names = sorted(set(github_by_name) & set(zenodo_by_name))
        archive_names = [name for name in common_names if name.endswith(_ARCHIVE_SUFFIXES)]
        if not archive_names:
            raise PublishedAuditInvalid("GitHub and Zenodo have no common bundle archive")
        advertised_assets = [*github["assets"], *zenodo["assets"]]
        if canonical_predecessor_tag is not None:
            if (
                predecessor_github is None
                or predecessor_zenodo is None
                or predecessor_archive_name is None
            ):
                raise PublishedAuditInvalid("predecessor public identity is incomplete")
            predecessor_github_asset = _require_named_asset(
                predecessor_github["assets"],
                name=predecessor_archive_name,
                label="predecessor GitHub archive",
            )
            predecessor_zenodo_asset = _require_named_asset(
                predecessor_zenodo["assets"],
                name=predecessor_archive_name,
                label="predecessor Zenodo archive",
            )
            _require_positive_advertised_size(
                predecessor_github_asset, label="predecessor GitHub archive"
            )
            _require_positive_advertised_size(
                predecessor_zenodo_asset, label="predecessor Zenodo archive"
            )
            advertised_assets.extend([predecessor_github_asset, predecessor_zenodo_asset])
        advertised_bytes = sum(
            int(asset["size"] or 0) for asset in advertised_assets if asset.get("size") is not None
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
            private_temp_root = root
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
            if canonical_predecessor_tag is not None:
                if (
                    predecessor_github is None
                    or predecessor_zenodo is None
                    or predecessor_archive_name is None
                ):
                    raise PublishedAuditInvalid("predecessor public identity is incomplete")
                predecessor_github_asset = _require_named_asset(
                    predecessor_github["assets"],
                    name=predecessor_archive_name,
                    label="predecessor GitHub archive",
                )
                predecessor_zenodo_asset = _require_named_asset(
                    predecessor_zenodo["assets"],
                    name=predecessor_archive_name,
                    label="predecessor Zenodo archive",
                )
                predecessor_github_record, downloaded_bytes = _download_public_asset(
                    public_session,
                    predecessor_github_asset,
                    root / "predecessor-github" / predecessor_archive_name,
                    timeout=timeout,
                    chunk_size=download_chunk_size,
                    max_download_bytes=max_download_bytes,
                    downloaded_bytes=downloaded_bytes,
                )
                predecessor_zenodo_record, downloaded_bytes = _download_public_asset(
                    public_session,
                    predecessor_zenodo_asset,
                    root / "predecessor-zenodo" / predecessor_archive_name,
                    timeout=timeout,
                    chunk_size=download_chunk_size,
                    max_download_bytes=max_download_bytes,
                    downloaded_bytes=downloaded_bytes,
                )
                github_downloads.append(predecessor_github_record)
                zenodo_downloads.append(predecessor_zenodo_record)
                _require_cross_channel_archive_identity(
                    predecessor_github_record, predecessor_zenodo_record
                )
                predecessor_receipt = {
                    "version_doi": predecessor_zenodo["doi"],
                    "concept_doi": predecessor_zenodo["concept_doi"],
                    "github_release_tag": predecessor_github["tag"],
                    "source_sha": predecessor_github["source_sha"],
                    "archive_sha256": predecessor_github_record["sha256"],
                    "archive_size_bytes": predecessor_github_record["bytes"],
                }
                predecessor_evidence = PredecessorEvidence(
                    archive_path=root / "predecessor-github" / predecessor_archive_name,
                    version_doi=predecessor_receipt["version_doi"],
                    concept_doi=predecessor_receipt["concept_doi"],
                    github_release_tag=predecessor_receipt["github_release_tag"],
                    archive_sha256=predecessor_receipt["archive_sha256"],
                    archive_size_bytes=predecessor_receipt["archive_size_bytes"],
                )
            else:
                predecessor_evidence = None
            core = audit_published(
                tag=requested_tag,
                doi=normalized_doi,
                github_dir=github_dir,
                zenodo_dir=zenodo_dir,
                source_sha=github["source_sha"],
                predecessor_evidence=predecessor_evidence,
            )
            _reconcile_zenodo_erratum_lineage(core, tag=requested_tag, zenodo=zenodo)
        status = "pass" if core["ok"] else "invalid"
        receipt = {
            "schema": NETWORK_SCHEMA,
            "ok": bool(core["ok"]),
            "status": status,
            "tag": requested_tag,
            "doi": normalized_doi,
            "source_sha": github["source_sha"],
            "predecessor": predecessor_receipt,
            "problems": list(core["problems"]),
            "discovery": discovery,
            "downloads": {
                "github": github_downloads,
                "zenodo": zenodo_downloads,
                "bytes": downloaded_bytes,
            },
            "audit": core,
        }
        return _sanitize_network_value(receipt, private_root=private_temp_root)
    except PublishedAuditUnavailable as exc:
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="unavailable",
            problem=str(exc),
            discovery=locals().get("discovery"),
            predecessor=predecessor_receipt,
            private_root=private_temp_root,
        )
    except PublishedAuditInvalid as exc:
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="invalid",
            problem=str(exc),
            discovery=locals().get("discovery"),
            predecessor=predecessor_receipt,
            private_root=private_temp_root,
        )
    except (OSError, ValueError) as exc:
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="invalid",
            problem=f"local audit preparation failed ({type(exc).__name__})",
            discovery=locals().get("discovery"),
            predecessor=predecessor_receipt,
            private_root=private_temp_root,
        )
    except Exception as exc:  # noqa: BLE001 - final fail-closed receipt boundary
        return _failure_network_receipt(
            tag=requested_tag,
            doi=requested_doi,
            status="error",
            problem=f"unexpected audit failure ({type(exc).__name__})",
            discovery=locals().get("discovery"),
            predecessor=predecessor_receipt,
            private_root=private_temp_root,
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
    parser.add_argument(
        "--predecessor-archive",
        type=Path,
        default=None,
        help="detached predecessor archive for a canonical erratum cold audit",
    )
    parser.add_argument(
        "--predecessor-doi",
        "--predecessor-version-doi",
        dest="predecessor_doi",
        default=None,
        help="predecessor Zenodo version DOI (required with every predecessor argument)",
    )
    parser.add_argument(
        "--predecessor-concept-doi",
        dest="predecessor_concept_doi",
        default=None,
        help="predecessor Zenodo concept DOI (required with every predecessor argument)",
    )
    parser.add_argument(
        "--predecessor-tag",
        "--predecessor-github-release-tag",
        dest="predecessor_tag",
        default=None,
        help="immutable predecessor GitHub release tag (required with every predecessor argument)",
    )
    parser.add_argument(
        "--predecessor-sha256",
        "--predecessor-archive-sha256",
        dest="predecessor_sha256",
        default=None,
        help="predecessor archive SHA-256 (required with every predecessor argument)",
    )
    parser.add_argument(
        "--predecessor-size-bytes",
        "--predecessor-archive-size-bytes",
        dest="predecessor_size_bytes",
        type=int,
        default=None,
        help="predecessor archive byte count (required with every predecessor argument)",
    )
    parser.add_argument("--output", type=Path, default=None, help="receipt output path")
    args = parser.parse_args(argv)

    try:
        predecessor_values = (
            args.predecessor_archive,
            args.predecessor_doi,
            args.predecessor_concept_doi,
            args.predecessor_tag,
            args.predecessor_sha256,
            args.predecessor_size_bytes,
        )
        if any(value is not None for value in predecessor_values) and not all(
            value is not None for value in predecessor_values
        ):
            raise ValueError(
                "predecessor archive, DOI, concept DOI, tag, SHA-256, and size must be provided "
                "together"
            )
        predecessor_evidence = (
            PredecessorEvidence(
                archive_path=args.predecessor_archive,
                version_doi=args.predecessor_doi,
                concept_doi=args.predecessor_concept_doi,
                github_release_tag=args.predecessor_tag,
                archive_sha256=args.predecessor_sha256,
                archive_size_bytes=args.predecessor_size_bytes,
            )
            if all(value is not None for value in predecessor_values)
            else None
        )
        receipt = audit_published(
            tag=args.tag,
            doi=args.doi,
            github_dir=args.github_dir,
            zenodo_dir=args.zenodo_dir,
            source_sha=args.source_sha,
            predecessor_evidence=predecessor_evidence,
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
