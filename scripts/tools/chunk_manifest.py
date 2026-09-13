#!/usr/bin/env python3
"""Deterministic chunked manifests and verification for very large result trees (#8915).

``chunk_manifest.v1``. ``manifest_id`` is the SHA-256 of the canonical manifest body without the
``manifest_id`` key and is the semantic digest other receipts reference. ``tree_sha256``
aggregates member records sorted by normalized relative path (``path``, ``size_bytes``,
``content_sha256``), so it is invariant to traversal order and worker count. Small members
(``size_bytes <= full_digest_threshold_bytes``) use ``mode: full`` with the whole-file SHA-256 in
one chunk; larger members use fixed boundaries ``offset_i = i * chunk_size_bytes`` and
``length_i = min(chunk_size_bytes, size - offset_i)``, independent of filesystem read size,
traversal order, and worker count. Chunk digests are not whole-file digest substitutes.

Modes: ``manifest`` (hash an owned root), ``verify`` (re-hash and compare against a manifest),
``resume`` (build a manifest while reusing identity-guarded cached digests from a
``chunk_manifest.state.v1`` file), and ``compare`` (diff two manifests without a root scan).
``verify --state`` records the same identity-guarded digests. A cached file record is reused only
when the size, mtime, ctime, inode, and device identity are unchanged; any other state fails closed
(``state_invalid``, ``state_schema_unsupported``, ``state_root_mismatch``, ``state_policy_mismatch``)
instead of trusting stale digests.

Symlinks, hardlinks, sparse files, special files, path escapes, duplicate or case-colliding paths,
missing or unexpected members, source mutation/truncation/growth during hashing, and partial or
tampered manifests fail closed with a coded error. Output holds normalized relative paths only;
``root_identity`` defaults to a SHA-256 of the resolved path. Producer manifests are never
rewritten. Stdlib-only; no ``robot_sf`` imports needed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

SCHEMA_VERSION = "chunk_manifest.v1"
STATE_SCHEMA_VERSION = "chunk_manifest.state.v1"
ALGORITHM = "fixed-size-v1"
BOUNDARY_RULE = "offset_i = i * chunk_size_bytes; length_i = min(chunk_size_bytes, size - offset_i)"
CHUNK_SIZE = 8 * 1024 * 1024
READ_SIZE = 1024 * 1024
RETENTION_UNSPECIFIED = "unspecified"
ROLES = ("keep-latest", "long-lived", "short-lived", "disposable", RETENTION_UNSPECIFIED)
DIGEST_KIND_FULL = "full-sha256-v1"
DIGEST_KIND_CHUNKED = "chunked-content-v1"
MAX_FAILURES = 1000
CHECKPOINT_INTERVAL_SECONDS = 15.0
EXIT_OK, EXIT_FAILED = 0, 1
REQUIRED = (
    "schema_version",
    "artifact",
    "chunking",
    "excluded",
    "files",
    "manifest_id",
    "tree_sha256",
)


class ChunkManifestError(Exception):
    """Fail-closed chunk-manifest error carrying a machine-readable code."""

    def __init__(self, code: str, message: str, *, file: str | None = None) -> None:
        """Build an error with an optional exact member location."""
        super().__init__(message)
        self.code, self.file = code, file

    def to_dict(self) -> dict[str, Any]:
        """Return the error as a JSON-safe mapping."""
        payload = {"code": self.code, "message": str(self)}
        if self.file is not None:
            payload["file"] = self.file
        return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically: fsync the temporary sibling, then replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        directory = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _identity(st: os.stat_result) -> tuple[int, int, int, int, int]:
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev)


def _stat_checked(path: Path, relative: str) -> tuple[int, int, int, int, int]:
    st = os.stat(path, follow_symlinks=False)
    if stat.S_ISLNK(st.st_mode):
        raise ChunkManifestError("symlink_rejected", f"symlink member: {relative}", file=relative)
    if not stat.S_ISREG(st.st_mode):
        raise ChunkManifestError(
            "unsupported_special_file", f"special file: {relative}", file=relative
        )
    if st.st_nlink > 1:
        raise ChunkManifestError(
            "hardlink_rejected", f"hardlinked member: {relative}", file=relative
        )
    blocks = getattr(st, "st_blocks", None)
    if blocks is not None and st.st_size > 0 and blocks * 512 < st.st_size:
        raise ChunkManifestError("sparse_file", f"sparse member: {relative}", file=relative)
    return _identity(st)


def normalize_relative_path(path_text: str) -> str:
    """Return a normalized relative POSIX path or raise a coded error."""
    if not path_text or path_text.startswith(("/", "\\")):
        raise ChunkManifestError("path_not_relative", f"not relative: {path_text!r}")
    if "\\" in path_text:
        raise ChunkManifestError("unsupported_path_character", f"backslash: {path_text!r}")
    posix = PurePosixPath(path_text)
    if posix.is_absolute() or any(part == ".." for part in posix.parts):
        raise ChunkManifestError("path_escape", f"escapes root: {path_text!r}")
    normalized = posix.as_posix()
    if normalized in ("", "."):
        raise ChunkManifestError("path_not_relative", f"empty path: {path_text!r}")
    return normalized


def _excluded(relative: str, kind: str, pattern: str) -> dict[str, str]:
    return {"path": relative, "member_type": kind, "reason": f"excluded_by_pattern:{pattern}"}


def scan_root(  # noqa: C901 - single fail-closed walk over path, link, and exclusion rules
    root: Path, *, exclude_patterns: Sequence[str] = ()
):
    """Scan one owned root deterministically; return sorted members and exclusions."""
    root = Path(root)
    if root.is_symlink():
        raise ChunkManifestError("symlink_root", "root itself must not be a symlink")
    if not root.is_dir():
        raise ChunkManifestError("root_not_directory", "root must be an existing directory")
    patterns = tuple(exclude_patterns)
    members: list[tuple[str, Path, tuple[int, int, int, int, int]]] = []
    excluded: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    exact: set[str] = set()

    def on_error(exc: OSError) -> None:
        raise ChunkManifestError("scan_error", f"directory scan failed: {exc.strerror}") from exc

    for current, dirnames, filenames in os.walk(root, topdown=True, onerror=on_error):
        current_path = Path(current)
        kept: list[str] = []
        for dirname in sorted(dirnames):
            candidate = current_path / dirname
            relative = normalize_relative_path(candidate.relative_to(root).as_posix())
            pattern = next((p for p in patterns if fnmatch.fnmatchcase(relative, p)), None)
            if pattern is not None:
                excluded.append(_excluded(relative, "directory", pattern))
            elif candidate.is_symlink():
                raise ChunkManifestError(
                    "symlink_rejected", f"symlink dir: {relative}", file=relative
                )
            else:
                kept.append(dirname)
        dirnames[:] = kept
        for filename in sorted(filenames):
            candidate = current_path / filename
            relative = normalize_relative_path(candidate.relative_to(root).as_posix())
            pattern = next((p for p in patterns if fnmatch.fnmatchcase(relative, p)), None)
            if pattern is not None:
                excluded.append(_excluded(relative, "file", pattern))
                continue
            if relative in exact:
                raise ChunkManifestError(
                    "duplicate_path", f"duplicate path: {relative}", file=relative
                )
            exact.add(relative)
            folded = relative.casefold()
            if folded in seen and seen[folded] != relative:
                raise ChunkManifestError(
                    "case_collision", f"{seen[folded]!r} vs {relative!r}", file=relative
                )
            seen[folded] = relative
            members.append((relative, candidate, _stat_checked(candidate, relative)))
    members.sort(key=lambda member: member[0].encode("utf-8"))
    excluded.sort(key=lambda entry: entry["path"].encode("utf-8"))
    return members, excluded


@dataclass(frozen=True, slots=True)
class ChunkingPolicy:
    """Documented chunking policy shared by manifest and verify runs."""

    chunk_size_bytes: int = CHUNK_SIZE
    full_digest_threshold_bytes: int | None = None
    exclude_patterns: tuple[str, ...] = ()


def _threshold(policy: ChunkingPolicy) -> int:
    limit = policy.full_digest_threshold_bytes
    return policy.chunk_size_bytes if limit is None else int(limit)


def _mode_for_size(size_bytes: int, policy: ChunkingPolicy) -> str:
    return "full" if size_bytes <= _threshold(policy) else "chunked"


def compute_tree_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the order- and worker-count-invariant aggregate tree digest."""
    hasher = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"]).encode("utf-8")):
        fields = (record["path"], record["size_bytes"], record["content_sha256"])
        hasher.update("\0".join(str(field) for field in fields).encode("utf-8") + b"\0")
    return hasher.hexdigest()


def compute_manifest_id(payload: Mapping[str, Any]) -> str:
    """Return the semantic manifest digest (body without ``manifest_id``)."""
    body = {key: value for key, value in payload.items() if key != "manifest_id"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_exact(handle: Any, length: int, relative: str) -> bytes:
    parts: list[bytes] = []
    while length > 0:
        data = handle.read(length)
        if not data:
            raise ChunkManifestError("source_truncated", f"short source: {relative}", file=relative)
        parts.append(data)
        length -= len(data)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def _is_int(value: Any, *, minimum: int | None = None) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return minimum is None or value >= minimum


def _identity_tuple(value: Any) -> tuple[int, int, int, int, int]:
    if not isinstance(value, list) or len(value) != 5 or not all(_is_int(item) for item in value):
        raise ChunkManifestError("state_invalid", "identity must be five integers")
    return (int(value[0]), int(value[1]), int(value[2]), int(value[3]), int(value[4]))


def _validate_cached_chunks(
    value: Any, *, size_bytes: int, policy: ChunkingPolicy
) -> list[dict[str, Any]]:
    """Return a validated chunk prefix or fail closed on a malformed cache entry."""
    if not isinstance(value, list):
        raise ChunkManifestError("state_invalid", "chunks must be a list")
    chunks: list[dict[str, Any]] = []
    offset = 0
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ChunkManifestError("state_invalid", "chunk entries must be mappings")
        if int(item.get("index", -1)) != index or int(item.get("offset", -1)) != offset:
            raise ChunkManifestError("state_invalid", f"chunk {index} breaks the boundary rule")
        length = item.get("length")
        if not _is_int(length, minimum=1) or length > policy.chunk_size_bytes:
            raise ChunkManifestError("state_invalid", f"chunk {index} has an invalid length")
        if offset + int(length) > size_bytes:
            raise ChunkManifestError("state_invalid", f"chunk {index} exceeds the source size")
        if not _hex64(item.get("sha256")):
            raise ChunkManifestError("state_invalid", f"chunk {index} digest is malformed")
        chunks.append(
            {
                "index": index,
                "offset": offset,
                "length": int(length),
                "sha256": str(item["sha256"]),
            }
        )
        offset += int(length)
    if chunks:
        last = chunks[-1]
        if (
            last["offset"] + last["length"] < size_bytes
            and last["length"] != policy.chunk_size_bytes
        ):
            raise ChunkManifestError("state_invalid", "chunk prefix has a short interior chunk")
    return chunks


@dataclass(frozen=True, slots=True)
class CachedFile:
    """One identity-guarded cached file entry from a resume state."""

    identity: tuple[int, int, int, int, int]
    mode: str
    chunks: tuple[Mapping[str, Any], ...]
    content_sha256: str | None
    complete: bool

    def matches(self, identity: tuple[int, int, int, int, int], mode: str) -> bool:
        """Return whether the cached entry still describes the current member identity."""
        return self.identity == identity and self.mode == mode

    def reusable(self) -> bool:
        """Return whether the entry carries a complete, well-formed record."""
        return self.complete and bool(self.content_sha256) and bool(self.chunks)

    def record(self, relative: str, size_bytes: int) -> dict[str, Any]:
        """Return the manifest record rebuilt from this cache entry."""
        chunks = [dict(chunk) for chunk in self.chunks]
        return {
            "path": relative,
            "size_bytes": size_bytes,
            "mode": self.mode,
            "digest_kind": DIGEST_KIND_FULL if self.mode == "full" else DIGEST_KIND_CHUNKED,
            "content_sha256": self.content_sha256,
            "chunks": chunks,
        }


def _state_body(
    *,
    root_identity: str,
    chunk_size_bytes: int,
    full_digest_threshold_bytes: int,
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "root_identity": root_identity,
        "chunking": {
            "algorithm": ALGORITHM,
            "chunk_size_bytes": chunk_size_bytes,
            "full_digest_threshold_bytes": full_digest_threshold_bytes,
            "read_size_bytes": READ_SIZE,
        },
        "files": [files[path] for path in sorted(files, key=lambda item: item.encode("utf-8"))],
    }


def compute_state_id(body: Mapping[str, Any]) -> str:
    """Return the semantic state digest (body without ``state_id``)."""
    payload = {key: value for key, value in body.items() if key != "state_id"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_state_entry(entry: Any) -> tuple[str, Mapping[str, Any]]:
    """Return one validated state file entry or fail closed."""
    if not isinstance(entry, Mapping):
        raise ChunkManifestError("state_invalid", "state file entries must be mappings")
    try:
        relative = normalize_relative_path(str(entry.get("path", "")))
    except ChunkManifestError as exc:
        raise ChunkManifestError("state_invalid", f"bad path: {exc}") from exc
    if entry.get("mode") not in ("full", "chunked"):
        raise ChunkManifestError("state_invalid", f"{relative}: bad mode")
    if not _is_int(entry.get("size_bytes"), minimum=0):
        raise ChunkManifestError("state_invalid", f"{relative}: bad size")
    if not isinstance(entry.get("complete"), bool):
        raise ChunkManifestError("state_invalid", f"{relative}: bad complete flag")
    if entry.get("complete") and not _hex64(entry.get("content_sha256")):
        raise ChunkManifestError("state_invalid", f"{relative}: bad content digest")
    if not entry.get("complete") and entry.get("content_sha256") is not None:
        raise ChunkManifestError("state_invalid", f"{relative}: partial content digest")
    _identity_tuple(entry.get("identity"))
    return relative, entry


def _validate_state_payload(
    payload: Mapping[str, Any],
    *,
    root_identity: str,
    chunk_size_bytes: int,
    full_digest_threshold_bytes: int,
) -> dict[str, Mapping[str, Any]]:
    """Validate one state payload and return its file entries."""
    if payload.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ChunkManifestError("state_schema_unsupported", str(payload.get("schema_version")))
    chunking = payload.get("chunking")
    if not isinstance(chunking, Mapping) or chunking.get("algorithm") != ALGORITHM:
        raise ChunkManifestError("state_invalid", "state chunking is malformed")
    if int(chunking.get("chunk_size_bytes", 0)) != int(chunk_size_bytes):
        raise ChunkManifestError("state_policy_mismatch", "state chunk size differs")
    if int(chunking.get("full_digest_threshold_bytes", -1)) != int(full_digest_threshold_bytes):
        raise ChunkManifestError("state_policy_mismatch", "state digest threshold differs")
    if payload.get("root_identity") != root_identity:
        raise ChunkManifestError("state_root_mismatch", "state belongs to another root")
    if payload.get("state_id") != compute_state_id(payload):
        raise ChunkManifestError("state_invalid", "state_id mismatch")
    files = payload.get("files")
    if not isinstance(files, list):
        raise ChunkManifestError("state_invalid", "state files must be a list")
    return dict(_validate_state_entry(entry) for entry in files)


class ResumeState:
    """Identity-guarded per-file digest cache with bounded checkpoint persistence."""

    def __init__(
        self,
        *,
        root_identity: str,
        chunk_size_bytes: int,
        full_digest_threshold_bytes: int,
        path: Path | None = None,
        files: Mapping[str, Mapping[str, Any]] | None = None,
        checkpoint_interval: float = CHECKPOINT_INTERVAL_SECONDS,
    ) -> None:
        """Build an empty or loaded state for one root and chunking geometry."""
        self.root_identity = root_identity
        self.chunk_size_bytes = int(chunk_size_bytes)
        self.full_digest_threshold_bytes = int(full_digest_threshold_bytes)
        self.path = Path(path) if path is not None else None
        self.files: dict[str, Mapping[str, Any]] = {
            str(key): value for key, value in dict(files or {}).items()
        }
        self.checkpoint_interval = float(checkpoint_interval)
        self.reused_files = 0
        self.reused_chunks = 0
        self.hashed_files = 0
        self._lock = Lock()
        self._last_persist = 0.0

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        root_identity: str,
        chunk_size_bytes: int,
        full_digest_threshold_bytes: int,
        checkpoint_interval: float = CHECKPOINT_INTERVAL_SECONDS,
    ) -> ResumeState:
        """Load and validate a ``chunk_manifest.state.v1`` file, failing closed when malformed."""
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ChunkManifestError(
                "state_unreadable", f"cannot read state: {exc.strerror}"
            ) from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ChunkManifestError("state_invalid", f"state is not valid JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ChunkManifestError("state_invalid", "state must be a JSON object")
        files = _validate_state_payload(
            payload,
            root_identity=root_identity,
            chunk_size_bytes=chunk_size_bytes,
            full_digest_threshold_bytes=full_digest_threshold_bytes,
        )
        return cls(
            root_identity=root_identity,
            chunk_size_bytes=chunk_size_bytes,
            full_digest_threshold_bytes=full_digest_threshold_bytes,
            path=path,
            files=files,
            checkpoint_interval=checkpoint_interval,
        )

    def cached(self, relative: str) -> CachedFile | None:
        """Return the cached entry for one member, if any."""
        entry = self.files.get(relative)
        if entry is None:
            return None
        return CachedFile(
            identity=_identity_tuple(entry.get("identity")),
            mode=str(entry.get("mode")),
            chunks=tuple(entry.get("chunks") or ()),
            content_sha256=entry.get("content_sha256"),
            complete=bool(entry.get("complete")),
        )

    def note_reuse(self, chunks: int) -> None:
        """Record one fully reused member and its reused chunk count."""
        with self._lock:
            self.reused_files += 1
            self.reused_chunks += max(chunks, 0)

    def note_reused_chunks(self, chunks: int) -> None:
        """Record partially reused chunks from an interrupted member."""
        with self._lock:
            self.reused_chunks += max(chunks, 0)

    def record_file(
        self,
        relative: str,
        *,
        identity: tuple[int, int, int, int, int],
        size_bytes: int,
        mode: str,
        chunks: Sequence[Mapping[str, Any]],
        content_sha256: str | None,
        complete: bool,
        force_persist: bool = False,
    ) -> None:
        """Store one member's progress and persist it under the checkpoint interval."""
        with self._lock:
            self.files[relative] = {
                "path": relative,
                "identity": list(identity),
                "size_bytes": int(size_bytes),
                "mode": mode,
                "chunks": [dict(chunk) for chunk in chunks],
                "content_sha256": content_sha256 if complete else None,
                "complete": bool(complete),
            }
            if complete:
                self.hashed_files += 1
            self._maybe_persist_locked(force=force_persist)

    def _maybe_persist_locked(self, *, force: bool) -> None:
        if self.path is None:
            return
        now = time.monotonic()
        if (
            not force
            and self.checkpoint_interval > 0
            and now - self._last_persist < self.checkpoint_interval
        ):
            return
        self._last_persist = now
        body = self._body_locked()
        body["state_id"] = compute_state_id(body)
        _write_json(self.path, body)

    def flush(self) -> None:
        """Persist the current state unconditionally."""
        with self._lock:
            self._maybe_persist_locked(force=True)

    def _body_locked(self) -> dict[str, Any]:
        return _state_body(
            root_identity=self.root_identity,
            chunk_size_bytes=self.chunk_size_bytes,
            full_digest_threshold_bytes=self.full_digest_threshold_bytes,
            files=dict(self.files),
        )

    def body(self) -> dict[str, Any]:
        """Return the canonical state body without ``state_id``."""
        with self._lock:
            return self._body_locked()

    def stats(self) -> dict[str, int]:
        """Return bounded resume counters for receipts."""
        with self._lock:
            return {
                "reused_files": self.reused_files,
                "reused_chunks": self.reused_chunks,
                "hashed_files": self.hashed_files,
                "cached_files": len(self.files),
            }


def _compare_chunks(
    relative: str,
    mode: str,
    expected: Mapping[int, Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(expected) != len(chunks):
        failures.append({"code": "chunk_count_mismatch", "file": relative})
    for chunk in chunks:
        wanted = expected.get(int(chunk["index"]))
        if wanted is None or str(wanted.get("sha256")) == chunk["sha256"]:
            continue
        code = "full_digest_mismatch" if mode == "full" else "chunk_digest_mismatch"
        failures.append(
            {
                "code": code,
                "file": relative,
                "chunk_index": int(chunk["index"]),
                "offset": int(chunk["offset"]),
                "expected": wanted.get("sha256"),
                "actual": chunk["sha256"],
            }
        )
    return failures


def _cached_prefix(cached, identity, mode: str, size_bytes: int, policy: ChunkingPolicy, state):
    """Return the reusable chunk prefix and byte offset for one interrupted member."""
    if cached is None or state is None or mode != "chunked" or not cached.matches(identity, mode):
        return [], 0
    if cached.reusable():
        return [], 0
    prefix = _validate_cached_chunks(list(cached.chunks), size_bytes=size_bytes, policy=policy)
    if prefix:
        state.note_reused_chunks(len(prefix))
    return prefix, sum(int(chunk["length"]) for chunk in prefix)


def _content_digest(
    chunks: Sequence[Mapping[str, Any]], mode: str, size_bytes: int
) -> tuple[str, str]:
    """Return the member content digest and its documented digest kind."""
    if mode == "full":
        return str(chunks[0]["sha256"]), DIGEST_KIND_FULL
    hasher = hashlib.sha256()
    hasher.update(b"chunked-content-v1\0" + str(size_bytes).encode("ascii"))
    for chunk in chunks:
        text = f"{chunk['index']}:{chunk['offset']}:{chunk['length']}:{chunk['sha256']}"
        hasher.update(b"\0" + text.encode("ascii"))
    return hasher.hexdigest(), DIGEST_KIND_CHUNKED


def _hash_member(member, *, policy: ChunkingPolicy, cached: CachedFile | None = None, state=None):
    """Hash one member, reusing identity-guarded cached digests and chunk checkpoints."""
    relative, absolute, identity = member
    size_bytes = identity[0]
    mode = _mode_for_size(size_bytes, policy)
    if (
        cached is not None
        and state is not None
        and cached.matches(identity, mode)
        and cached.reusable()
    ):
        record = cached.record(relative, size_bytes)
        state.note_reuse(len(cached.chunks))
        return record
    if _stat_checked(absolute, relative) != identity:
        raise ChunkManifestError(
            "source_mutated", f"changed before hashing: {relative}", file=relative
        )
    chunks, offset = _cached_prefix(cached, identity, mode, size_bytes, policy, state)
    with absolute.open("rb") as handle:
        handle.seek(offset)
        if mode == "full":
            stream = hashlib.sha256()
            for block in iter(lambda: handle.read(READ_SIZE), b""):
                stream.update(block)
            chunks.append(
                {"index": 0, "offset": 0, "length": size_bytes, "sha256": stream.hexdigest()}
            )
        else:
            index = len(chunks)
            while offset < size_bytes:
                length = min(policy.chunk_size_bytes, size_bytes - offset)
                digest = hashlib.sha256(_read_exact(handle, length, relative)).hexdigest()
                chunks.append(
                    {"index": index, "offset": offset, "length": length, "sha256": digest}
                )
                offset += length
                index += 1
                if state is not None:
                    state.record_file(
                        relative,
                        identity=identity,
                        size_bytes=size_bytes,
                        mode=mode,
                        chunks=chunks,
                        content_sha256=None,
                        complete=False,
                    )
        if handle.read(1):
            raise ChunkManifestError("source_mutated", f"source grew: {relative}", file=relative)
    if _stat_checked(absolute, relative) != identity:
        raise ChunkManifestError(
            "source_mutated", f"changed during hashing: {relative}", file=relative
        )
    content, digest_kind = _content_digest(chunks, mode, size_bytes)
    if state is not None:
        state.record_file(
            relative,
            identity=identity,
            size_bytes=size_bytes,
            mode=mode,
            chunks=chunks,
            content_sha256=content,
            complete=True,
        )
    return {
        "path": relative,
        "size_bytes": size_bytes,
        "mode": mode,
        "digest_kind": digest_kind,
        "content_sha256": content,
        "chunks": chunks,
    }


def _process_member(member, *, policy, expected=None, cached=None, state=None):
    relative, _absolute, identity = member
    size_bytes = identity[0]
    mode = _mode_for_size(size_bytes, policy)
    if expected is not None and int(expected.get("size_bytes", -1)) != size_bytes:
        failure = {"code": "size_mismatch", "file": relative}
        return {"path": relative, "record": None, "failures": [failure]}
    if expected is not None and expected.get("mode") != mode:
        failure = {"code": "mode_mismatch", "file": relative}
        return {"path": relative, "record": None, "failures": [failure]}
    record = _hash_member(member, policy=policy, cached=cached, state=state)
    failures: list[dict[str, Any]] = []
    if expected is not None:
        wanted = {int(chunk["index"]): chunk for chunk in expected.get("chunks", [])}
        failures = _compare_chunks(relative, mode, wanted, record["chunks"])
    return {"path": relative, "record": record, "failures": failures}


def _run_members(
    members,
    *,
    expected_files,
    policy,
    workers,
    cached_files=None,
    state=None,
    progress_every=0,
):
    cached_files = cached_files or {}
    completed = 0
    total = len(members)
    total_bytes = 0

    def process(member):
        return _process_member(
            member,
            policy=policy,
            expected=expected_files.get(member[0]),
            cached=cached_files.get(member[0]),
            state=state,
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = []
            for result in pool.map(process, members):
                results.append(result)
                completed += 1
                total_bytes += int(result["record"]["size_bytes"]) if result["record"] else 0
                _maybe_progress(completed, total, total_bytes, progress_every)
            return results
    results = []
    for member in members:
        result = process(member)
        results.append(result)
        completed += 1
        total_bytes += int(result["record"]["size_bytes"]) if result["record"] else 0
        _maybe_progress(completed, total, total_bytes, progress_every)
    return results


def _maybe_progress(completed: int, total: int, total_bytes: int, progress_every: int) -> None:
    """Emit one bounded progress line to stderr every ``progress_every`` members."""
    if progress_every > 0 and completed % progress_every == 0:
        print(
            f"progress: {completed}/{total} members, {total_bytes} bytes",
            file=sys.stderr,
            flush=True,
        )


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Semantic artifact identity carried by one chunk manifest."""

    artifact_id: str
    artifact_version: str
    root_identity: str
    retention_role: str = RETENTION_UNSPECIFIED


def build_manifest(root, *, artifact, policy, workers=1, state=None, progress_every=0):
    """Build a deterministic chunk manifest for one owned root."""
    members, excluded = scan_root(root, exclude_patterns=policy.exclude_patterns)
    cached_files = {}
    if state is not None:
        cached_files = {member[0]: state.cached(member[0]) for member in members}
    outcomes = _run_members(
        members,
        expected_files={},
        policy=policy,
        workers=workers,
        cached_files=cached_files,
        state=state,
        progress_every=progress_every,
    )
    records = [result["record"] for result in outcomes]
    summary = {
        "member_count": len(records),
        "total_bytes": sum(int(record["size_bytes"]) for record in records),
        "chunk_count": sum(len(record["chunks"]) for record in records),
        "chunked_members": sum(1 for record in records if record["mode"] == "chunked"),
        "full_members": sum(1 for record in records if record["mode"] == "full"),
        "excluded_member_count": len(excluded),
    }
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact": asdict(artifact),
        "chunking": {
            "algorithm": ALGORITHM,
            "boundary_rule": BOUNDARY_RULE,
            "chunk_size_bytes": policy.chunk_size_bytes,
            "exclude_patterns": list(policy.exclude_patterns),
            "full_digest_threshold_bytes": _threshold(policy),
            "read_size_bytes": READ_SIZE,
        },
        "files": records,
        "excluded": excluded,
        "summary": summary,
        "tree_sha256": compute_tree_digest(records),
    }
    return {**body, "manifest_id": compute_manifest_id(body)}


def _issue(code: str, message: str, path: str | None = None) -> dict[str, str]:
    issue = {"code": code, "message": message}
    if path is not None:
        issue["path"] = path
    return issue


def _hex64(value: Any) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def validate_manifest(payload: Any) -> list[dict[str, str]]:  # noqa: C901, PLR0912, PLR0915 - full schema audit
    """Validate a ``chunk_manifest.v1`` payload and return coded issues."""
    if not isinstance(payload, Mapping):
        return [_issue("invalid_manifest", "manifest must be a JSON object")]
    issues = [
        _issue("manifest_field_missing", f"missing {key}", key)
        for key in REQUIRED
        if key not in payload
    ]
    if issues:
        return issues
    if payload.get("schema_version") != SCHEMA_VERSION:
        return [_issue("unsupported_schema_version", str(payload.get("schema_version")))]
    artifact = payload.get("artifact")
    if not isinstance(artifact, Mapping):
        return [_issue("invalid_manifest", "artifact must be a mapping")]
    for key in ("artifact_id", "artifact_version", "root_identity"):
        if not artifact.get(key):
            issues.append(_issue("manifest_field_missing", f"artifact.{key}", key))
    if artifact.get("retention_role") not in ROLES:
        issues.append(_issue("manifest_retention_role", str(artifact.get("retention_role"))))
    chunking = payload.get("chunking")
    chunk_size = int(chunking.get("chunk_size_bytes", 0)) if isinstance(chunking, Mapping) else 0
    if (
        chunk_size < 1
        or not isinstance(chunking, Mapping)
        or chunking.get("algorithm") != ALGORITHM
    ):
        issues.append(_issue("manifest_chunk_algorithm", "bad chunking"))
    elif "read_size_bytes" in chunking and not _is_int(chunking.get("read_size_bytes"), minimum=1):
        issues.append(_issue("manifest_chunk_algorithm", "bad read_size_bytes"))
    files = payload.get("files")
    if not isinstance(files, list):
        issues.append(_issue("invalid_manifest", "files must be a list"))
        files = []
    seen: dict[str, str] = {}
    exact: set[str] = set()
    paths: list[str] = []
    for record in files:
        if not isinstance(record, Mapping):
            issues.append(_issue("invalid_manifest", "file entries must be mappings"))
            continue
        raw = str(record.get("path", ""))
        try:
            relative = normalize_relative_path(raw)
        except ChunkManifestError as exc:
            issues.append(_issue(exc.code, str(exc), raw))
            continue
        paths.append(relative)
        if relative in exact:
            issues.append(_issue("duplicate_path", f"duplicate {relative}", relative))
        exact.add(relative)
        folded = relative.casefold()
        if folded in seen and seen[folded] != relative:
            issues.append(_issue("case_collision", f"{seen[folded]!r} vs {relative!r}", relative))
        seen[folded] = relative
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0:
            issues.append(_issue("manifest_field_missing", f"{relative}: size", relative))
        mode = record.get("mode")
        if mode not in ("full", "chunked"):
            issues.append(_issue("manifest_field_missing", f"{relative}: mode", relative))
        elif "digest_kind" in record and record.get("digest_kind") != (
            DIGEST_KIND_FULL if mode == "full" else DIGEST_KIND_CHUNKED
        ):
            issues.append(_issue("manifest_digest_kind", f"{relative}: digest_kind", relative))
        if not _hex64(record.get("content_sha256")):
            issues.append(_issue("manifest_digest_format", f"{relative}: content digest", relative))
        chunks = record.get("chunks")
        if not isinstance(chunks, list):
            issues.append(_issue("manifest_field_missing", f"{relative}: chunks", relative))
            continue
        size_bytes = int(record.get("size_bytes", -1))
        if mode == "full":
            if len(chunks) != 1 or int(chunks[0].get("length", -1)) != size_bytes:
                issues.append(_issue("manifest_chunk_shape", f"{relative}: full chunk", relative))
            continue
        offset = 0
        for index, chunk in enumerate(chunks):
            if int(chunk.get("index", -1)) != index or int(chunk.get("offset", -1)) != offset:
                issues.append(_issue("manifest_chunk_shape", f"{relative}: gap {index}", relative))
            length = int(chunk.get("length", -1))
            if length <= 0 or (index < len(chunks) - 1 and length != chunk_size):
                issues.append(
                    _issue("manifest_chunk_shape", f"{relative}: length {index}", relative)
                )
            if not _hex64(chunk.get("sha256")):
                issues.append(
                    _issue("manifest_digest_format", f"{relative}: digest {index}", relative)
                )
            offset += max(length, 0)
        if offset != size_bytes:
            issues.append(
                _issue(
                    "manifest_chunk_shape", f"{relative}: cover {offset} != {size_bytes}", relative
                )
            )
    if paths != sorted(paths, key=lambda item: item.encode("utf-8")):
        issues.append(_issue("manifest_files_unsorted", "files unsorted"))
    excluded = payload.get("excluded")
    if not isinstance(excluded, list):
        issues.append(_issue("invalid_manifest", "excluded must be a list"))
    else:
        for entry in excluded:
            if not isinstance(entry, Mapping) or not entry.get("reason"):
                issues.append(_issue("excluded_member_reason", "reason required"))
                continue
            try:
                normalize_relative_path(str(entry.get("path", "")))
            except ChunkManifestError as exc:
                issues.append(_issue(exc.code, str(exc), str(entry.get("path", ""))))
    summary = payload.get("summary")
    if summary is not None:
        if not isinstance(summary, Mapping):
            issues.append(_issue("manifest_summary_invalid", "summary must be a mapping"))
        else:
            for key in (
                "member_count",
                "total_bytes",
                "chunk_count",
                "chunked_members",
                "full_members",
                "excluded_member_count",
            ):
                if not _is_int(summary.get(key), minimum=0):
                    issues.append(_issue("manifest_summary_invalid", f"summary.{key}"))
            if all(_is_int(summary.get(key), minimum=0) for key in ("member_count", "total_bytes")):
                observed = {
                    "member_count": len(files),
                    "total_bytes": sum(
                        int(record["size_bytes"])
                        for record in files
                        if isinstance(record, Mapping)
                        and _is_int(record.get("size_bytes"), minimum=0)
                    ),
                    "chunk_count": sum(
                        len(record["chunks"])
                        for record in files
                        if isinstance(record, Mapping) and isinstance(record.get("chunks"), list)
                    ),
                    "chunked_members": sum(
                        1
                        for record in files
                        if isinstance(record, Mapping) and record.get("mode") == "chunked"
                    ),
                    "full_members": sum(
                        1
                        for record in files
                        if isinstance(record, Mapping) and record.get("mode") == "full"
                    ),
                    "excluded_member_count": len(excluded) if isinstance(excluded, list) else 0,
                }
                for key, value in observed.items():
                    if summary.get(key) != value:
                        issues.append(_issue("manifest_summary_mismatch", f"summary.{key}", key))
    if not _hex64(payload.get("manifest_id")):
        issues.append(_issue("manifest_digest_format", "bad manifest_id"))
    elif compute_manifest_id(payload) != payload["manifest_id"]:
        issues.append(_issue("manifest_digest_mismatch", "manifest_id mismatch"))
    if not _hex64(payload.get("tree_sha256")):
        issues.append(_issue("manifest_digest_format", "bad tree_sha256"))
    elif files:
        try:
            expected_tree = compute_tree_digest(files)
        except (KeyError, TypeError, ValueError):
            expected_tree = None
        if expected_tree is not None and expected_tree != payload.get("tree_sha256"):
            issues.append(_issue("tree_digest_mismatch", "tree_sha256 mismatch"))
    return issues


def load_manifest_file(path: Path) -> dict[str, Any]:
    """Load and fully validate a chunk manifest, failing closed when partial."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ChunkManifestError("manifest_unreadable", f"cannot read: {exc.strerror}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ChunkManifestError("partial_manifest", f"manifest incomplete: {exc}") from exc
    issues = validate_manifest(payload)
    if issues:
        first = issues[0]
        raise ChunkManifestError(
            first["code"], f"invalid manifest: {first['message']}", file=first.get("path")
        )
    return payload


def verify_manifest(root, *, manifest, state=None, progress_every=0):
    """Verify one root against a manifest, failing closed on any difference."""
    chunking = manifest["chunking"]
    policy = ChunkingPolicy(
        int(chunking["chunk_size_bytes"]),
        int(chunking["full_digest_threshold_bytes"]),
        tuple(chunking.get("exclude_patterns", ())),
    )
    members, _excluded = scan_root(root, exclude_patterns=policy.exclude_patterns)
    manifest_files = {record["path"]: record for record in manifest["files"]}
    scanned = {member[0] for member in members}
    failures = [
        {"code": "missing_member", "file": path} for path in sorted(set(manifest_files) - scanned)
    ]
    failures += [
        {"code": "unexpected_member", "file": path}
        for path in sorted(scanned - set(manifest_files))
    ]
    targets = [member for member in members if member[0] in manifest_files]
    cached_files = {}
    if state is not None:
        cached_files = {member[0]: state.cached(member[0]) for member in targets}
    outcomes = _run_members(
        targets,
        expected_files=manifest_files,
        policy=policy,
        workers=1,
        cached_files=cached_files,
        state=state,
        progress_every=progress_every,
    )
    for result in outcomes:
        failures.extend(result["failures"])
    failure_count = len(failures)
    truncated = failure_count > MAX_FAILURES
    if state is not None:
        state.flush()
    return {
        "status": "ok" if not failure_count else "failed",
        "failures": failures[:MAX_FAILURES],
        "failure_count": failure_count,
        "failures_truncated": truncated,
        "members_checked": len(outcomes),
        "manifest_id": manifest["manifest_id"],
        "tree_sha256": manifest["tree_sha256"],
    }


def compare_manifests(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two manifests deterministically without touching either root."""
    left_files = {str(record["path"]): record for record in left["files"]}
    right_files = {str(record["path"]): record for record in right["files"]}
    left_paths, right_paths = set(left_files), set(right_files)
    added = sorted(right_paths - left_paths)
    removed = sorted(left_paths - right_paths)
    changed: list[dict[str, Any]] = []
    for path in sorted(left_paths & right_paths):
        before, after = left_files[path], right_files[path]
        if before.get("mode") != after.get("mode"):
            reason = "mode_changed"
        elif int(before.get("size_bytes", -1)) != int(after.get("size_bytes", -1)):
            reason = "size_changed"
        elif before.get("content_sha256") != after.get("content_sha256"):
            reason = "content_changed"
        else:
            continue
        changed.append(
            {
                "path": path,
                "reason": reason,
                "left_size_bytes": int(before.get("size_bytes", -1)),
                "right_size_bytes": int(after.get("size_bytes", -1)),
                "left_sha256": before.get("content_sha256"),
                "right_sha256": after.get("content_sha256"),
            }
        )
    counts = {
        "added": len(added),
        "removed": len(removed),
        "changed": len(changed),
        "unchanged": len(left_paths & right_paths) - len(changed),
    }
    return {
        "status": "identical" if not (added or removed or changed) else "different",
        "counts": counts,
        "added": added,
        "removed": removed,
        "changed": changed,
        "left_manifest_id": left.get("manifest_id"),
        "right_manifest_id": right.get("manifest_id"),
        "left_tree_sha256": left.get("tree_sha256"),
        "right_tree_sha256": right.get("tree_sha256"),
    }


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-id", default=None)
    parser.add_argument("--artifact-version", default=None)
    parser.add_argument("--root-identity", default=None)
    parser.add_argument("--retention-role", choices=ROLES, default=None)
    parser.add_argument("--chunk-size", type=_positive_int, default=CHUNK_SIZE)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--progress-every", type=_non_negative_int, default=0)
    parser.add_argument("--json", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chunk_manifest", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("manifest", help="Write a chunk manifest")
    _add_build_arguments(manifest)

    resume = commands.add_parser(
        "resume", help="Write a chunk manifest, reusing an identity-guarded state file"
    )
    _add_build_arguments(resume)
    resume.add_argument("--state", type=Path, required=True)

    verify = commands.add_parser("verify", help="Verify a root against a manifest")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--state", type=Path, default=None)
    verify.add_argument("--progress-every", type=_non_negative_int, default=0)
    verify.add_argument("--json", action="store_true")

    compare = commands.add_parser("compare", help="Compare two manifests without a root scan")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--json", action="store_true")
    return parser


def _dump(payload: Mapping[str, Any], json_mode: bool) -> None:
    print(json.dumps(payload, indent=2 if json_mode else None, sort_keys=True))


def _reject_inside_root(path: Path | None, root: Path, label: str) -> None:
    if path is None:
        return
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return
    raise ChunkManifestError("path_inside_root", f"{label} must not live inside --root")


def _default_root_identity(root: Path) -> str:
    return "sha256:" + hashlib.sha256(str(root).encode("utf-8")).hexdigest()


def _resolve_artifact(args: argparse.Namespace, root: Path) -> ArtifactIdentity:
    root_identity = args.root_identity or _default_root_identity(root)
    return ArtifactIdentity(
        str(args.artifact_id or root.name),
        str(args.artifact_version or "unspecified"),
        str(root_identity),
        str(args.retention_role or RETENTION_UNSPECIFIED),
    )


def _load_or_init_state(state_path: Path, *, root: Path, policy: ChunkingPolicy) -> ResumeState:
    root_identity = _default_root_identity(root)
    if state_path.exists():
        return ResumeState.load(
            state_path,
            root_identity=root_identity,
            chunk_size_bytes=policy.chunk_size_bytes,
            full_digest_threshold_bytes=_threshold(policy),
        )
    return ResumeState(
        root_identity=root_identity,
        chunk_size_bytes=policy.chunk_size_bytes,
        full_digest_threshold_bytes=_threshold(policy),
        path=state_path,
    )


def _run_manifest(args: argparse.Namespace) -> int:
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    _reject_inside_root(output, root, "--output")
    manifest = build_manifest(
        root,
        artifact=_resolve_artifact(args, root),
        policy=ChunkingPolicy(args.chunk_size, None, tuple(args.exclude)),
        workers=args.workers,
        progress_every=args.progress_every,
    )
    _write_json(output, manifest)
    _dump(
        {
            "mode": "manifest",
            "status": "ok",
            "manifest_id": manifest["manifest_id"],
            "tree_sha256": manifest["tree_sha256"],
            "member_count": len(manifest["files"]),
            "excluded_member_count": len(manifest["excluded"]),
            "receipt_ref": {"kind": SCHEMA_VERSION, "manifest_id": manifest["manifest_id"]},
        },
        args.json,
    )
    return EXIT_OK


def _run_resume(args: argparse.Namespace) -> int:
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    _reject_inside_root(output, root, "--output")
    _reject_inside_root(state_path, root, "--state")
    if state_path == output:
        raise ChunkManifestError("state_manifest_conflict", "--state and --output must differ")
    policy = ChunkingPolicy(args.chunk_size, None, tuple(args.exclude))
    state = _load_or_init_state(state_path, root=root, policy=policy)
    manifest = build_manifest(
        root,
        artifact=_resolve_artifact(args, root),
        policy=policy,
        workers=args.workers,
        state=state,
        progress_every=args.progress_every,
    )
    state.flush()
    _write_json(output, manifest)
    body = state.body()
    _dump(
        {
            "mode": "resume",
            "status": "ok",
            "manifest_id": manifest["manifest_id"],
            "tree_sha256": manifest["tree_sha256"],
            "member_count": len(manifest["files"]),
            "excluded_member_count": len(manifest["excluded"]),
            "resume": state.stats(),
            "state_ref": {
                "kind": STATE_SCHEMA_VERSION,
                "state_id": compute_state_id(body),
            },
            "receipt_ref": {"kind": SCHEMA_VERSION, "manifest_id": manifest["manifest_id"]},
        },
        args.json,
    )
    return EXIT_OK


def _run_verify(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    manifest = load_manifest_file(args.manifest)
    state = None
    if args.state is not None:
        state_path = args.state.expanduser().resolve()
        _reject_inside_root(state_path, root, "--state")
        chunking = manifest["chunking"]
        policy = ChunkingPolicy(
            int(chunking["chunk_size_bytes"]),
            int(chunking["full_digest_threshold_bytes"]),
            tuple(chunking.get("exclude_patterns", ())),
        )
        state = _load_or_init_state(state_path, root=root, policy=policy)
    result = verify_manifest(
        root, manifest=manifest, state=state, progress_every=args.progress_every
    )
    if state is not None:
        result["resume"] = state.stats()
        result["state_ref"] = {
            "kind": STATE_SCHEMA_VERSION,
            "state_id": compute_state_id(state.body()),
        }
    result["receipt_ref"] = {"kind": SCHEMA_VERSION, "manifest_id": manifest["manifest_id"]}
    _dump(result, args.json)
    return EXIT_OK if result["status"] == "ok" else EXIT_FAILED


def _run_compare(args: argparse.Namespace) -> int:
    left = load_manifest_file(args.left.expanduser().resolve())
    right = load_manifest_file(args.right.expanduser().resolve())
    result = compare_manifests(left, right)
    result["receipt_ref"] = {
        "kind": SCHEMA_VERSION,
        "left_manifest_id": result["left_manifest_id"],
        "right_manifest_id": result["right_manifest_id"],
    }
    _dump(result, args.json)
    return EXIT_OK if result["status"] == "identical" else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chunk-manifest CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    runners = {
        "manifest": _run_manifest,
        "resume": _run_resume,
        "verify": _run_verify,
        "compare": _run_compare,
    }
    try:
        return runners[args.command](args)
    except (ChunkManifestError, OSError) as exc:
        error = (
            exc.to_dict()
            if isinstance(exc, ChunkManifestError)
            else {"code": "io_error", "message": str(exc)}
        )
        _dump(
            {"mode": args.command, "status": "failed", "error": error}, getattr(args, "json", False)
        )
        return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
