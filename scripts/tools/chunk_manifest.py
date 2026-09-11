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

Modes: ``manifest`` (hash an owned root) and ``verify`` (re-hash and compare against a manifest).
Symlinks, hardlinks, sparse files, special files, path escapes, duplicate or case-colliding paths,
missing or unexpected members, source mutation/truncation/growth during hashing, and partial or
tampered manifests fail closed with a coded error. Output holds normalized relative paths only;
``root_identity`` defaults to a SHA-256 of the resolved path. Producer manifests are never
rewritten. Identity-guarded ``resume`` mode, verification resume state, per-chunk checkpoints, and
``compare`` mode are deferred follow-ups. Stdlib-only; no ``robot_sf`` imports needed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = "chunk_manifest.v1"
ALGORITHM = "fixed-size-v1"
BOUNDARY_RULE = "offset_i = i * chunk_size_bytes; length_i = min(chunk_size_bytes, size - offset_i)"
CHUNK_SIZE = 8 * 1024 * 1024
READ_SIZE = 1024 * 1024
RETENTION_UNSPECIFIED = "unspecified"
ROLES = ("keep-latest", "long-lived", "short-lived", "disposable", RETENTION_UNSPECIFIED)
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _hash_member(member, *, policy: ChunkingPolicy):
    relative, absolute, identity = member
    size_bytes = identity[0]
    mode = _mode_for_size(size_bytes, policy)
    if _stat_checked(absolute, relative) != identity:
        raise ChunkManifestError(
            "source_mutated", f"changed before hashing: {relative}", file=relative
        )
    chunks: list[dict[str, Any]] = []
    with absolute.open("rb") as handle:
        if mode == "full":
            stream = hashlib.sha256()
            for block in iter(lambda: handle.read(READ_SIZE), b""):
                stream.update(block)
            chunks.append(
                {"index": 0, "offset": 0, "length": size_bytes, "sha256": stream.hexdigest()}
            )
        else:
            offset = index = 0
            while offset < size_bytes:
                length = min(policy.chunk_size_bytes, size_bytes - offset)
                digest = hashlib.sha256(_read_exact(handle, length, relative)).hexdigest()
                chunks.append(
                    {"index": index, "offset": offset, "length": length, "sha256": digest}
                )
                offset += length
                index += 1
        if handle.read(1):
            raise ChunkManifestError("source_mutated", f"source grew: {relative}", file=relative)
    if _stat_checked(absolute, relative) != identity:
        raise ChunkManifestError(
            "source_mutated", f"changed during hashing: {relative}", file=relative
        )
    if mode == "full":
        content = chunks[0]["sha256"]
    else:
        hasher = hashlib.sha256()
        hasher.update(b"chunked-content-v1\0" + str(size_bytes).encode("ascii"))
        for chunk in chunks:
            text = f"{chunk['index']}:{chunk['offset']}:{chunk['length']}:{chunk['sha256']}"
            hasher.update(b"\0" + text.encode("ascii"))
        content = hasher.hexdigest()
    record = {
        "path": relative,
        "size_bytes": size_bytes,
        "mode": mode,
        "content_sha256": content,
        "chunks": chunks,
    }
    return record


def _process_member(member, *, policy, expected=None):
    relative, _absolute, identity = member
    size_bytes = identity[0]
    mode = _mode_for_size(size_bytes, policy)
    if expected is not None and int(expected.get("size_bytes", -1)) != size_bytes:
        failure = {"code": "size_mismatch", "file": relative}
        return {"path": relative, "record": None, "failures": [failure]}
    if expected is not None and expected.get("mode") != mode:
        failure = {"code": "mode_mismatch", "file": relative}
        return {"path": relative, "record": None, "failures": [failure]}
    record = _hash_member(member, policy=policy)
    failures: list[dict[str, Any]] = []
    if expected is not None:
        wanted = {int(chunk["index"]): chunk for chunk in expected.get("chunks", [])}
        failures = _compare_chunks(relative, mode, wanted, record["chunks"])
    return {"path": relative, "record": record, "failures": failures}


def _run_members(members, *, expected_files, policy, workers):
    def process(member):
        return _process_member(member, policy=policy, expected=expected_files.get(member[0]))

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(process, members))
    return [process(member) for member in members]


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Semantic artifact identity carried by one chunk manifest."""

    artifact_id: str
    artifact_version: str
    root_identity: str
    retention_role: str = RETENTION_UNSPECIFIED


def build_manifest(root, *, artifact, policy, workers=1):
    """Build a deterministic chunk manifest for one owned root."""
    members, excluded = scan_root(root, exclude_patterns=policy.exclude_patterns)
    outcomes = _run_members(members, expected_files={}, policy=policy, workers=workers)
    records = [result["record"] for result in outcomes]
    body = {
        "schema_version": SCHEMA_VERSION,
        "artifact": asdict(artifact),
        "chunking": {
            "algorithm": ALGORITHM,
            "boundary_rule": BOUNDARY_RULE,
            "chunk_size_bytes": policy.chunk_size_bytes,
            "exclude_patterns": list(policy.exclude_patterns),
            "full_digest_threshold_bytes": _threshold(policy),
        },
        "files": records,
        "excluded": excluded,
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


def verify_manifest(root, *, manifest):
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
    outcomes = _run_members(targets, expected_files=manifest_files, policy=policy, workers=1)
    for result in outcomes:
        failures.extend(result["failures"])
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "failure_count": len(failures),
        "members_checked": len(outcomes),
        "manifest_id": manifest["manifest_id"],
        "tree_sha256": manifest["tree_sha256"],
    }


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chunk_manifest", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="Write a chunk manifest")
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--artifact-id", default=None)
    manifest.add_argument("--artifact-version", default=None)
    manifest.add_argument("--root-identity", default=None)
    manifest.add_argument("--retention-role", choices=ROLES, default=None)
    manifest.add_argument("--chunk-size", type=_positive_int, default=CHUNK_SIZE)
    manifest.add_argument("--exclude", action="append", default=[])
    manifest.add_argument("--workers", type=_positive_int, default=1)
    manifest.add_argument("--json", action="store_true")
    verify = commands.add_parser("verify", help="Verify a root against a manifest")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--json", action="store_true")
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


def _resolve_artifact(args: argparse.Namespace, root: Path) -> ArtifactIdentity:
    root_identity = args.root_identity
    if not root_identity:
        digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
        root_identity = "sha256:" + digest
    return ArtifactIdentity(
        str(args.artifact_id or root.name),
        str(args.artifact_version or "unspecified"),
        str(root_identity),
        str(args.retention_role or RETENTION_UNSPECIFIED),
    )


def _run_manifest(args: argparse.Namespace) -> int:
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    _reject_inside_root(output, root, "--output")
    manifest = build_manifest(
        root,
        artifact=_resolve_artifact(args, root),
        policy=ChunkingPolicy(args.chunk_size, None, tuple(args.exclude)),
        workers=args.workers,
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


def _run_verify(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    manifest = load_manifest_file(args.manifest)
    result = verify_manifest(root, manifest=manifest)
    result["receipt_ref"] = {"kind": SCHEMA_VERSION, "manifest_id": manifest["manifest_id"]}
    _dump(result, args.json)
    return EXIT_OK if result["status"] == "ok" else EXIT_FAILED


def main(argv: Sequence[str] | None = None) -> int:
    """Run the chunk-manifest CLI and return the process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        return _run_manifest(args) if args.command == "manifest" else _run_verify(args)
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
