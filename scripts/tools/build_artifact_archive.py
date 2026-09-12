#!/usr/bin/env python3
"""Build and verify deterministic, manifest-owned local artifact archives.

The fixed PAX-tar/gzip contract accepts only digest-pinned regular files from
the existing harvest/staging manifests, normalizes metadata, checks source
identity while streaming, and atomically finalizes outputs.  It never discovers
files, deletes sources, publishes data, or makes a benchmark/scientific claim.
"""

# fmt: off
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robot_sf.benchmark.identity.hash_utils import sha256_file, stable_hash  # noqa: E402
from robot_sf.common.atomic_io import atomic_write_json  # noqa: E402
from scripts.tools.chunk_manifest import ChunkManifestError, normalize_relative_path  # noqa: E402
from scripts.validation.verify_artifact_transfer import _manifest_members  # noqa: E402

ARCHIVE_SCHEMA = "artifact_archive.v1"
FORMAT = {"container": "pax-tar", "compression": "gzip", "compression_level": 9, "gzip_mtime": 0, "contract": "pax-tar-gzip-v1"}
CLAIM_BOUNDARY = "Operational local archive custody only; no source deletion, publication, benchmark result, or scientific claim is implied."
EXIT_OK, EXIT_BLOCKED, EXIT_MALFORMED = 0, 2, 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ACTIVE_STATES = frozenset({"active", "running", "writing", "partial", "in_progress"})
ACTIVE_MARKERS = (
    ".active-writer",
    ".active_writer",
    ".writer-active",
    ".writer_active",
    ".gate_lease.json",
)


class ArchiveError(ValueError):
    """Fail-closed error carrying a stable reason code."""

    def __init__(self, code: str, message: str, *, path: str | None = None) -> None:
        """Create an error with an optional sanitized relative path."""
        super().__init__(message)
        self.code, self.path = code, path

    def as_dict(self) -> dict[str, str]:
        """Return the CLI-safe representation."""
        result = {"code": self.code, "message": str(self)}
        if self.path is not None:
            result["path"] = self.path
        return result


@dataclass(frozen=True, slots=True)
class _Source:
    path: str
    digest: str
    size: int
    role: str
    local: Path | None = None
    identity: tuple[int, int, int, int, int, int] | None = None
    mode: str | None = None


def _ident(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino, value.st_dev, value.st_mode)


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        value = os.lstat(path)
    except OSError as exc:
        raise ArchiveError("source_unavailable", f"cannot stat {label}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise ArchiveError("symlink_rejected", f"symlink is not allowed: {label}", path=label)
    if not stat.S_ISREG(value.st_mode):
        raise ArchiveError("special_file_rejected", f"regular file required: {label}", path=label)
    if value.st_nlink != 1:
        raise ArchiveError("hardlink_rejected", f"hardlink is not allowed: {label}", path=label)
    return value


def _path(raw: Any, location: str) -> str:
    if not isinstance(raw, str) or not raw or any(ord(char) < 32 or ord(char) == 127 for char in raw):
        raise ArchiveError("path_invalid", f"invalid path at {location}")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ArchiveError("path_invalid", f"path is not valid UTF-8 at {location}") from exc
    try:
        normalized = normalize_relative_path(raw)
    except ChunkManifestError as exc:
        raise ArchiveError("path_traversal", str(exc), path=raw) from exc
    if raw != normalized or ":" in normalized:
        raise ArchiveError("path_invalid", f"path is not canonically portable: {raw}", path=raw)
    return normalized


def _digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ArchiveError("manifest_digest_invalid", f"sha256 required at {location}")
    return value


def _role(value: Any, location: str) -> str:
    value = "unspecified" if value in (None, "") else value
    if not isinstance(value, str) or ROLE_RE.fullmatch(value) is None:
        raise ArchiveError("manifest_role_invalid", f"content role invalid at {location}")
    return value


def _active(payload: Mapping[str, Any]) -> bool:
    if any(
        payload.get(key) not in (None, False, "", [], {})
        for key in ("active_writer", "writer_active", "active_writers", "writer_lock")
    ):
        return True
    if payload.get("active_job") is True:
        return True
    writers = payload.get("writers")
    if isinstance(writers, list) and any(
        isinstance(item, Mapping)
        and (item.get("active") is True or str(item.get("state", "")).lower() in ACTIVE_STATES)
        for item in writers
    ):
        return True
    writer = payload.get("writer")
    return (
        isinstance(writer, Mapping)
        and (writer.get("active") is True or str(writer.get("state", "")).lower() in ACTIVE_STATES)
    ) or any(str(payload.get(key, "")).lower() in ACTIVE_STATES for key in ("state", "status"))


def _require_harvest_ready(payload: Mapping[str, Any]) -> None:
    scheduler = payload.get("scheduler")
    if (
        payload.get("status") != "ready"
        or payload.get("artifact_status") != "complete"
        or not isinstance(scheduler, Mapping)
        or scheduler.get("terminal") is not True
        or payload.get("problems") != []
    ):
        raise ArchiveError(
            "source_not_ready",
            "terminal job harvest must be ready, complete, terminal, and problem-free",
        )


def _require_bundle_ready(payload: Mapping[str, Any]) -> None:
    """Reject compute staging bundles that are not explicitly ready and problem-free."""
    if payload.get("status") != "ready" or payload.get("problems") != []:
        raise ArchiveError(
            "source_not_ready",
            "compute staging bundle must be ready and problem-free",
        )


def _read(path: Path) -> tuple[Mapping[str, Any], str, tuple[int, int, int, int, int, int]]:
    before = _regular(path, "source manifest")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveError("manifest_invalid", "source manifest is unreadable") from exc
    if _ident(_regular(path, "source manifest")) != _ident(before):
        raise ArchiveError("source_drift", "source manifest changed while reading")
    if not isinstance(payload, Mapping):
        raise ArchiveError("manifest_invalid", "source manifest must be a JSON object")
    return payload, hashlib.sha256(raw).hexdigest(), _ident(before)


def _source_manifest(path: Path) -> tuple[str, str, tuple[int, int, int, int, int, int], tuple[_Source, ...]]:  # noqa: C901 - one bounded manifest pass
    payload, digest, identity = _read(path)
    schema = payload.get("schema_version")
    if schema not in {"terminal_job_harvest.v1", "compute_staging_bundle.v1"}:
        raise ArchiveError("unsupported_manifest_schema", f"unsupported source schema: {schema!r}")
    if schema == "terminal_job_harvest.v1":
        _require_harvest_ready(payload)
    elif schema == "compute_staging_bundle.v1":
        _require_bundle_ready(payload)
    if _active(payload):
        raise ArchiveError("active_writer", "source manifest reports an active writer")
    problems: list[dict[str, str]] = []
    _schema, parsed = _manifest_members(payload, problems)
    if problems:
        first = problems[0]
        raise ArchiveError(first["code"], first["message"], path=first.get("location"))
    raw_members = payload.get("inventory" if schema == "terminal_job_harvest.v1" else "members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ArchiveError("manifest_invalid", "source manifest needs non-empty members")
    by_path = {item.relative_path: item for item in parsed}
    members: list[_Source] = []
    seen: set[str] = set()
    folded_seen: set[str] = set()
    for index, raw in enumerate(raw_members):
        if not isinstance(raw, Mapping):
            raise ArchiveError("manifest_invalid", f"member {index} is not an object")
        normalized = _path(raw.get("path", raw.get("relative_path")), f"members[{index}].path")
        member = by_path.get(normalized)
        if member is None:
            raise ArchiveError("manifest_invalid", f"member identity differs: {normalized}")
        digest_value = _digest(raw.get("sha256", raw.get("source_sha256")), f"members[{index}].sha256")
        size_value = raw.get("byte_size", raw.get("size_bytes", raw.get("size")))
        if (digest_value, size_value) != (member.sha256, member.byte_size) or not isinstance(size_value, int) or isinstance(size_value, bool) or size_value < 0:
            raise ArchiveError("manifest_invalid", f"member identity differs: {normalized}")
        role_value = raw.get("content_role", raw.get("role", raw.get("retention_class")))
        if schema == "compute_staging_bundle.v1" and role_value in (None, ""):
            roles = raw.get("roles")
            if not isinstance(roles, list) or len(roles) != 1:
                raise ArchiveError("manifest_role_invalid", f"one compute-staging role required at members[{index}]")
            role_value = roles[0]
        role = _role(role_value, f"members[{index}].content_role")
        folded = normalized.casefold()
        if normalized in seen or folded in folded_seen:
            raise ArchiveError("duplicate_normalized_path", f"duplicate member: {normalized}", path=normalized)
        seen.add(normalized)
        folded_seen.add(folded)
        members.append(_Source(normalized, digest_value, size_value, role))
    members.sort(key=lambda item: item.path.encode("utf-8"))
    return str(schema), digest, identity, tuple(members)


def _root(root: Path) -> None:
    try:
        value = os.lstat(root)
    except OSError as exc:
        raise ArchiveError("source_unavailable", "source root is unavailable") from exc
    if stat.S_ISLNK(value.st_mode):
        raise ArchiveError("symlink_root", "source root must not be a symlink")
    if not stat.S_ISDIR(value.st_mode):
        raise ArchiveError("source_invalid", "source root must be a directory")
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            if name in ACTIVE_MARKERS:
                located = os.path.relpath(os.path.join(current, name), root)
                raise ArchiveError(
                    "active_writer",
                    "active writer marker is present",
                    path=located,
                )


def _local(root: Path, relative: str) -> tuple[Path, os.stat_result]:
    current, parts = root, PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current /= part
        try:
            value = os.lstat(current)
        except OSError as exc:
            raise ArchiveError("source_missing", f"source member is absent: {relative}", path=relative) from exc
        if stat.S_ISLNK(value.st_mode):
            raise ArchiveError("symlink_rejected", f"source member traverses a symlink: {relative}", path=relative)
        if index < len(parts) - 1 and not stat.S_ISDIR(value.st_mode):
            raise ArchiveError("source_invalid", f"source parent is not a directory: {relative}", path=relative)
    return current, _regular(current, relative)


def _snapshot(root: Path, member: _Source) -> _Source:
    path, before = _local(root, member.path)
    if before.st_size != member.size:
        raise ArchiveError("source_size_mismatch", f"source size differs: {member.path}", path=member.path)
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise ArchiveError("source_unreadable", f"source could not be hashed: {member.path}", path=member.path) from exc
    after = _regular(path, member.path)
    if _ident(before) != _ident(after):
        raise ArchiveError("source_drift", f"source changed while hashing: {member.path}", path=member.path)
    if actual != member.digest:
        raise ArchiveError("source_checksum_mismatch", f"source checksum differs: {member.path}", path=member.path)
    return _Source(member.path, member.digest, member.size, member.role, path, _ident(before), "executable" if before.st_mode & 0o111 else "regular")


def _reject_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return
    raise ArchiveError("output_inside_source", f"{label} must not be inside source root")


def _reject_output_alias(archive: Path, member_manifest: Path) -> None:
    try:
        resolved_archive = archive.resolve(strict=False)
        resolved_manifest = member_manifest.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ArchiveError("output_alias", "output paths could not be resolved") from exc
    if (
        resolved_archive == resolved_manifest
        or str(resolved_archive).casefold() == str(resolved_manifest).casefold()
    ):
        raise ArchiveError("output_alias", "archive and member manifest outputs must be distinct")


def _partials(path: Path) -> None:
    if path.name.endswith(".partial"):
        raise ArchiveError("partial_archive", f"partial output is not accepted: {path.name}")
    try:
        candidates = [path.with_name(f"{path.name}.partial"), path.with_name(f".{path.name}.partial")]
        candidates += [item for item in path.parent.iterdir() if item.name.startswith(f".{path.name}.") and item.name.endswith(".partial")]
    except OSError as exc:
        raise ArchiveError("output_unavailable", "cannot inspect output directory") from exc
    if any(os.path.lexists(item) for item in candidates):
        raise ArchiveError("partial_archive", f"partial output exists beside {path.name}")


def _safe_parent(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ArchiveError("extraction_symlink", "extraction parent must not be a symlink")
        if current.parent == current:
            return
        current = current.parent


def _capacity(parent: Path, required: int, declared: int | None) -> int:
    if declared is not None:
        if declared < 0:
            raise ArchiveError("capacity_invalid", "capacity must be non-negative")
        free = declared
    else:
        try:
            free = int(shutil.disk_usage(parent).free)
        except OSError as exc:
            raise ArchiveError("capacity_unavailable", "free output capacity could not be measured") from exc
    if free < required:
        raise ArchiveError("capacity_insufficient", f"need {required} bytes, have {free}")
    return free


def _required(sources: Sequence[_Source]) -> int:
    tar_bytes = 64 + sum(512 + ((item.size + 511) // 512) * 512 for item in sources)
    return tar_bytes + 4096 + sum(220 + len(item.path) + len(item.role) for item in sources)


class _Reader:
    def __init__(self, handle: Any) -> None:
        self.handle, self.digest, self.total = handle, hashlib.sha256(), 0

    def read(self, size: int = -1) -> bytes:
        value = self.handle.read(size)
        self.digest.update(value)
        self.total += len(value)
        return value


def _open(source: _Source):
    try:
        handle = os.fdopen(os.open(source.local, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb")
        if _ident(os.fstat(handle.fileno())) != source.identity:
            handle.close()
            raise ArchiveError("source_drift", f"source changed before copy: {source.path}", path=source.path)
        return handle
    except ArchiveError:
        raise
    except OSError as exc:
        raise ArchiveError("source_drift", f"source could not be opened safely: {source.path}", path=source.path) from exc


def _write_tar(path: Path, sources: Sequence[_Source]) -> None:
    try:
        with path.open("w+b") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as compressed, tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for source in sources:
                info = tarfile.TarInfo(source.path)
                info.size, info.mode = source.size, 0o755 if source.mode == "executable" else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime, info.type, info.pax_headers = 0, tarfile.REGTYPE, {}
                handle = _open(source)
                try:
                    reader = _Reader(handle)
                    archive.addfile(info, reader)
                finally:
                    handle.close()
                if reader.total != source.size or reader.digest.hexdigest() != source.digest or _ident(_regular(source.local, source.path)) != source.identity:
                    raise ArchiveError("source_drift", f"source changed during copy: {source.path}", path=source.path)
            raw.flush()
            os.fsync(raw.fileno())
    except ArchiveError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ArchiveError("archive_write_failed", f"archive could not be written: {path.name}") from exc


def _member_manifest(schema: str, source_digest: str, sources: Sequence[_Source], archive_digest: str, archive_size: int) -> dict[str, Any]:
    body: dict[str, Any] = {"schema_version": ARCHIVE_SCHEMA, "format": FORMAT, "source": {"manifest_schema": schema, "manifest_sha256": source_digest}, "members": [{"path": item.path, "size_bytes": item.size, "source_sha256": item.digest, "mode_class": item.mode, "content_role": item.role} for item in sources], "archive": {"sha256": archive_digest, "size_bytes": archive_size}, "claim_boundary": CLAIM_BOUNDARY}
    return {**body, "manifest_id": stable_hash(body)}


def build_archive(source_manifest: Path, source_root: Path, archive: Path, *, member_manifest: Path | None = None, capacity_bytes: int | None = None) -> dict[str, Any]:
    """Build an archive and its deterministic member manifest."""
    source_manifest, source_root, archive = map(Path, (source_manifest, source_root, archive))
    member_manifest = Path(member_manifest or f"{archive}.manifest.json")
    _reject_output_alias(archive, member_manifest)
    _root(source_root)
    for path, label in ((source_manifest, "source manifest"), (archive, "archive"), (member_manifest, "member manifest")):
        _reject_inside(path, source_root, label)
    archive.parent.mkdir(parents=True, exist_ok=True)
    member_manifest.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(archive) or os.path.lexists(member_manifest):
        raise ArchiveError("output_exists", "archive outputs must be new")
    _partials(archive)
    _partials(member_manifest)
    schema, source_digest, identity, members = _source_manifest(source_manifest)
    sources = tuple(_snapshot(source_root, member) for member in members)
    required = _required(sources)
    free = min(_capacity(parent, required, capacity_bytes) for parent in (archive.parent, member_manifest.parent))
    _root(source_root)
    archive_temp = manifest_temp = None
    committed = False
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{archive.name}.", suffix=".partial", dir=archive.parent)
        os.close(descriptor)
        archive_temp = Path(temporary)
        _write_tar(archive_temp, sources)
        archive_digest, archive_size = sha256_file(archive_temp), archive_temp.stat().st_size
        _assert_manifest_unchanged(source_manifest, identity)
        _assert_sources_unchanged(sources)
        _root(source_root)
        payload = _member_manifest(schema, source_digest, sources, archive_digest, archive_size)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{member_manifest.name}.", suffix=".partial", dir=member_manifest.parent)
        os.close(descriptor)
        manifest_temp = Path(temporary)
        atomic_write_json(manifest_temp, payload)
        os.replace(archive_temp, archive)
        committed = True
        os.replace(manifest_temp, member_manifest)
        return {"status": "verified", "schema_version": ARCHIVE_SCHEMA, "archive_sha256": archive_digest, "archive_size_bytes": archive_size, "member_manifest_sha256": sha256_file(member_manifest), "member_count": len(sources), "required_bytes": required, "free_bytes": free}
    except ArchiveError:
        raise
    except OSError as exc:
        raise ArchiveError("archive_finalize_failed", "archive finalization failed") from exc
    finally:
        if archive_temp is not None:
            archive_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)
        if committed and not member_manifest.exists():
            archive.unlink(missing_ok=True)


def _assert_manifest_unchanged(path: Path, identity: tuple[int, int, int, int, int, int]) -> None:
    if _ident(_regular(path, "source manifest")) != identity:
        raise ArchiveError("source_drift", "source manifest changed during archive build")


def _assert_sources_unchanged(sources: Sequence[_Source]) -> None:
    for source in sources:
        if _ident(_regular(source.local, source.path)) != source.identity:
            raise ArchiveError("source_drift", f"source changed during archive build: {source.path}", path=source.path)


def _load_member_manifest(path: Path) -> dict[str, Any]:  # noqa: C901 - compact schema audit
    payload, _digest_value, _identity = _read(Path(path))
    if not isinstance(payload, dict) or payload.get("schema_version") != ARCHIVE_SCHEMA:
        raise ArchiveError("manifest_invalid", "unsupported member manifest")
    manifest_id = _digest(payload.get("manifest_id"), "manifest_id")
    if stable_hash({key: value for key, value in payload.items() if key != "manifest_id"}) != manifest_id:
        raise ArchiveError("manifest_identity_mismatch", "member manifest digest differs")
    if payload.get("format") != FORMAT:
        raise ArchiveError("format_mismatch", "archive format contract differs")
    source = payload.get("source")
    if not isinstance(source, Mapping) or source.get("manifest_schema") not in {"terminal_job_harvest.v1", "compute_staging_bundle.v1"}:
        raise ArchiveError("manifest_invalid", "member manifest source identity is incomplete")
    _digest(source.get("manifest_sha256"), "source.manifest_sha256")
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ArchiveError("manifest_invalid", "member manifest needs non-empty members")
    paths: list[str] = []
    seen: set[str] = set()
    folded_seen: set[str] = set()
    for index, item in enumerate(members):
        if not isinstance(item, Mapping):
            raise ArchiveError("manifest_invalid", f"member {index} is not an object")
        path_value = _path(item.get("path"), f"members[{index}].path")
        folded = path_value.casefold()
        if path_value in seen or folded in folded_seen:
            raise ArchiveError("duplicate_normalized_path", f"duplicate member: {path_value}", path=path_value)
        seen.add(path_value)
        folded_seen.add(folded)
        size = item.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ArchiveError("manifest_invalid", f"invalid size for {path_value}", path=path_value)
        _digest(item.get("source_sha256"), f"members[{index}].source_sha256")
        if item.get("mode_class") not in {"regular", "executable"}:
            raise ArchiveError("manifest_invalid", f"invalid mode class for {path_value}", path=path_value)
        _role(item.get("content_role"), f"members[{index}].content_role")
        paths.append(path_value)
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")):
        raise ArchiveError("manifest_order_invalid", "member manifest paths are not sorted")
    archive = payload.get("archive")
    if not isinstance(archive, Mapping) or not _digest(archive.get("sha256"), "archive.sha256"):
        raise ArchiveError("manifest_invalid", "archive identity is missing")
    if not isinstance(archive.get("size_bytes"), int) or isinstance(archive.get("size_bytes"), bool) or archive["size_bytes"] < 1:
        raise ArchiveError("manifest_invalid", "archive size is invalid")
    return payload


def verify_archive(archive: Path, member_manifest: Path, *, extraction_root: Path | None = None, capacity_bytes: int | None = None) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915 - one fail-closed pass
    """Verify archive identity and member checksums, optionally extracting atomically."""
    archive, member_manifest = Path(archive), Path(member_manifest)
    _partials(archive)
    _partials(member_manifest)
    manifest = _load_member_manifest(member_manifest)
    before = _regular(archive, "archive")
    if before.st_size < 1:
        raise ArchiveError("partial_archive", "archive is empty")
    actual = sha256_file(archive)
    after = _regular(archive, "archive")
    if _ident(before) != _ident(after):
        raise ArchiveError("source_drift", "archive changed while being hashed")
    if actual != manifest["archive"]["sha256"]:
        raise ArchiveError("archive_checksum_mismatch", "archive checksum differs")
    if after.st_size != manifest["archive"]["size_bytes"]:
        raise ArchiveError("archive_size_mismatch", "archive size differs")
    expected = {item["path"]: item for item in manifest["members"]}
    target = Path(extraction_root) if extraction_root is not None else None
    stage = None
    if target is not None:
        if os.path.lexists(target):
            raise ArchiveError("extraction_not_fresh", "extraction root must not already exist")
        _safe_parent(target.parent)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArchiveError("extraction_unavailable", "extraction parent is unavailable") from exc
        _capacity(target.parent, sum(item["size_bytes"] for item in expected.values()) + 4096 * len(expected), capacity_bytes)
        stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=target.parent))
    seen: list[str] = []
    previous = None
    try:
        try:
            with tarfile.open(archive, mode="r:gz") as source:
                for info in source:
                    name = _path(info.name, "archive member")
                    if info.name != name:
                        raise ArchiveError("path_traversal", f"archive member is not canonical: {info.name}", path=info.name)
                    if previous is not None and name.encode("utf-8") < previous.encode("utf-8"):
                        raise ArchiveError("archive_order_invalid", "archive members are not sorted")
                    previous = name
                    if name in seen:
                        raise ArchiveError("duplicate_path", f"duplicate archive member: {name}", path=name)
                    wanted = expected.get(name)
                    if wanted is None:
                        raise ArchiveError("unexpected_member", f"archive member not declared: {name}", path=name)
                    expected_mode = 0o755 if wanted["mode_class"] == "executable" else 0o644
                    if (info.type != tarfile.REGTYPE or info.linkname or info.size != wanted["size_bytes"] or info.mode != expected_mode or info.uid != 0 or info.gid != 0 or info.uname or info.gname or info.mtime != 0):
                        raise ArchiveError("unsafe_archive_member", f"archive member is not the declared regular file: {name}", path=name)
                    stream = source.extractfile(info)
                    if stream is None:
                        raise ArchiveError("partial_archive", f"archive member has no data: {name}", path=name)
                    digest, written = hashlib.sha256(), 0
                    destination = None
                    if stage is not None:
                        destination = stage.joinpath(*PurePosixPath(name).parts)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination = destination.open("xb")
                    try:
                        while chunk := stream.read(1024 * 1024):
                            digest.update(chunk)
                            written += len(chunk)
                            if destination:
                                destination.write(chunk)
                    finally:
                        stream.close()
                        if destination:
                            try:
                                os.fchmod(destination.fileno(), expected_mode)
                            finally:
                                destination.close()
                    if written != wanted["size_bytes"]:
                        raise ArchiveError("partial_archive", f"archive member truncated: {name}", path=name)
                    if digest.hexdigest() != wanted["source_sha256"]:
                        raise ArchiveError("member_checksum_mismatch", f"member checksum differs: {name}", path=name)
                    seen.append(name)
        except ArchiveError:
            raise
        except (OSError, EOFError, tarfile.TarError, gzip.BadGzipFile) as exc:
            raise ArchiveError("partial_archive", "archive is truncated or unreadable") from exc
        missing = sorted(set(expected) - set(seen), key=lambda item: item.encode("utf-8"))
        if missing:
            raise ArchiveError("missing_member", f"archive member is missing: {missing[0]}", path=missing[0])
        if stage is not None:
            if os.path.lexists(target):
                raise ArchiveError("extraction_not_fresh", "extraction root appeared during verification")
            os.replace(stage, target)
            stage = None
        return {"status": "verified", "schema_version": ARCHIVE_SCHEMA, "archive_sha256": actual, "member_count": len(seen), "extracted": target is not None}
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="build_artifact_archive", description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True, help="source or member manifest")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member-manifest", type=Path)
    parser.add_argument("--extract-root", type=Path)
    parser.add_argument("--capacity-bytes", type=int)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the archive CLI."""
    args = _parser().parse_args(argv)
    try:
        if args.build:
            if args.source_root is None:
                raise ArchiveError("command_invalid", "--source-root is required with --build")
            report = build_archive(args.manifest, args.source_root, args.archive, member_manifest=args.member_manifest, capacity_bytes=args.capacity_bytes)
        else:
            report = verify_archive(args.archive, args.manifest, extraction_root=args.extract_root, capacity_bytes=args.capacity_bytes)
    except ArchiveError as exc:
        report = {"status": "blocked", "error": exc.as_dict()}
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(f"build_artifact_archive: blocked [{exc.code}] {exc}")
        return EXIT_MALFORMED if exc.code in {"command_invalid", "manifest_invalid", "unsupported_manifest_schema"} else EXIT_BLOCKED
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"build_artifact_archive: verified members={report['member_count']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
# fmt: on
