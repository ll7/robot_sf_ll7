"""Read-only loading of episode rows from a published benchmark bundle.

The release anomaly detectors operate on the rows that were published with a
release.  This module deliberately stops at the episode-row boundary: it does
not extract traces, run a simulator, or infer a planner identity from a row.
It verifies the publication manifest and the checksummed episode members before
returning rows with the release arm and source member attached.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PUBLICATION_BUNDLE_SCHEMA_VERSION = "benchmark-publication-bundle.v2"

# These limits are intentionally independent of the much larger limits used by
# general archive tooling.  The anomaly loader only needs the episode rows and
# should not become an accidental bulk-artifact reader.
MAX_ARCHIVE_BYTES = 4 * 1024**3
MAX_ARCHIVE_MEMBERS = 20_000
MAX_EXPANDED_BYTES = 4 * 1024**3
MAX_MANIFEST_BYTES = 8 * 1024**2
MAX_EPISODE_FILE_BYTES = 256 * 1024**2
MAX_EPISODE_ROW_BYTES = 16 * 1024**2
MAX_EPISODE_ROWS = 1_000_000

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ARM_SUFFIX = "__differential_drive"
_MANIFEST_NAME = "publication_manifest.json"
_EPISODE_NAME = "episodes.jsonl"


class ReleaseRowBundleError(ValueError):
    """Raised when a publication bundle cannot be admitted as release rows."""


@dataclass(frozen=True, slots=True)
class _EpisodeMember:
    """A verified episode member and its bundle-relative identity."""

    relative_path: str
    arm_directory: str
    member: tarfile.TarInfo | None = None
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class _BundleSource:
    """Input handles and verified source metadata for one bundle."""

    manifest_bytes: bytes
    manifest: Mapping[str, Any]
    episode_members: tuple[_EpisodeMember, ...]
    bundle_sha256: str | None


def _sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of ``value``."""

    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    """Hash a bounded regular file without retaining a second copy.

    Returns:
        Lowercase SHA-256 digest.
    """

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ReleaseRowBundleError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently choosing one.

    Returns:
        The validated object mapping.
    """

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _read_json(raw: bytes, *, label: str) -> Any:
    """Decode UTF-8 JSON with duplicate-key checks.

    Python's JSON decoder intentionally retains ``NaN`` and ``Infinity`` as
    floating-point values.  Published 0.0.7 rows contain those legacy metric
    values, and admission must preserve their bytes and values for downstream
    missing/non-finite handling rather than rewriting or rejecting the source.

    Returns:
        The decoded JSON value.
    """

    try:
        return json.loads(
            raw,
            object_pairs_hook=_strict_object_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ReleaseRowBundleError(f"{label} is malformed JSON") from exc


def _validate_member_path(name: str, *, label: str) -> tuple[str, ...]:
    """Validate a POSIX archive-relative path and return its components.

    Returns:
        Validated POSIX path components.
    """

    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ReleaseRowBundleError(f"{label} has an unsafe path")
    raw_parts = name.split("/")
    if any(part in {".", ".."} for part in raw_parts) or any(not part for part in raw_parts[:-1]):
        raise ReleaseRowBundleError(f"{label} has an unsafe path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseRowBundleError(f"{label} has an unsafe path: {name!r}")
    return path.parts


def _relative_episode_path(value: str, *, label: str) -> tuple[str, str] | None:
    """Return ``(canonical path, arm directory)`` for a manifest episode path.

    Returns:
        Canonical episode path and arm directory, or ``None`` for another file.
    """

    parts = _validate_member_path(value, label=label)
    if parts and parts[0] == "payload":
        parts = parts[1:]
    if len(parts) != 3 or parts[0] != "runs" or parts[2] != _EPISODE_NAME:
        return None
    arm_directory = parts[1]
    if not arm_directory:
        raise ReleaseRowBundleError(f"{label} has an empty release arm")
    return f"payload/runs/{arm_directory}/{_EPISODE_NAME}", arm_directory


def _release_arm(arm_directory: str, *, label: str) -> str:
    """Convert a run-directory name to the stable release arm identity.

    Returns:
        Release arm without the differential-drive suffix.
    """

    arm = arm_directory.removesuffix(_ARM_SUFFIX)
    if not arm or arm in {".", ".."}:
        raise ReleaseRowBundleError(f"{label} has an invalid release arm")
    return arm


def _manifest_entry(entry: object, *, index: int) -> tuple[str, int, str]:
    """Validate one publication-manifest file entry.

    Returns:
        Canonical payload path, declared byte size, and lowercase SHA-256.
    """

    if not isinstance(entry, Mapping):
        raise ReleaseRowBundleError(f"publication manifest file entry {index} is not an object")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ReleaseRowBundleError(f"publication manifest file entry {index} has no path")
    parts = _validate_member_path(raw_path, label=f"manifest file entry {index}")
    # Publication bundles write payload-relative paths (``runs/...``), but
    # older publication tooling and hand-built fixtures may include the
    # explicit ``payload/`` prefix.  Store one canonical form for both.
    canonical_path = "/".join(parts) if parts[0] == "payload" else "payload/" + "/".join(parts)

    raw_size = entry.get("size_bytes")
    if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
        raise ReleaseRowBundleError(f"manifest file entry {index} has an invalid size_bytes")
    if raw_size > MAX_EXPANDED_BYTES:
        raise ReleaseRowBundleError(f"manifest file entry {index} exceeds the bundle size limit")
    raw_sha256 = entry.get("sha256")
    if not isinstance(raw_sha256, str) or _SHA256_RE.fullmatch(raw_sha256) is None:
        raise ReleaseRowBundleError(f"manifest file entry {index} has an invalid sha256")
    return canonical_path, raw_size, raw_sha256.lower()


def _manifest_entry_map(manifest: Mapping[str, Any]) -> dict[str, tuple[int, str]]:
    """Validate and index the v2 publication manifest file entries.

    Returns:
        Mapping from canonical payload path to declared size and digest.
    """

    if manifest.get("schema_version") != PUBLICATION_BUNDLE_SCHEMA_VERSION:
        raise ReleaseRowBundleError(
            "publication_manifest.json has unsupported schema_version "
            f"{manifest.get('schema_version')!r}"
        )
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ReleaseRowBundleError("publication_manifest.json files must be a list")

    indexed: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(entries, start=1):
        canonical_path, raw_size, raw_sha256 = _manifest_entry(entry, index=index)
        if canonical_path in indexed:
            raise ReleaseRowBundleError(f"duplicate publication manifest path: {canonical_path!r}")
        indexed[canonical_path] = (raw_size, raw_sha256)
    return indexed


def _episode_members_from_manifest(
    manifest_entries: Mapping[str, tuple[int, str]],
) -> dict[str, tuple[int, str, str]]:
    """Return checked episode entries keyed by canonical bundle-relative path."""

    result: dict[str, tuple[int, str, str]] = {}
    for path, (size, digest) in manifest_entries.items():
        parsed = _relative_episode_path(path, label="manifest episode path")
        if parsed is None:
            continue
        canonical_path, arm_directory = parsed
        # ``_manifest_entry_map`` already rejects duplicate canonical paths;
        # keep this guard in case the path normalization is changed later.
        if canonical_path in result:
            raise ReleaseRowBundleError(f"duplicate publication episode path: {canonical_path}")
        result[canonical_path] = (size, digest, arm_directory)
    if not result:
        raise ReleaseRowBundleError("publication manifest contains no episode members")
    return result


def _read_manifest_bytes(raw: bytes, *, source_label: str) -> tuple[bytes, Mapping[str, Any]]:
    """Validate the bounded publication manifest bytes.

    Returns:
        The original bytes and their decoded manifest mapping.
    """

    if len(raw) > MAX_MANIFEST_BYTES:
        raise ReleaseRowBundleError(f"{source_label} exceeds the manifest size limit")
    payload = _read_json(raw, label=source_label)
    if not isinstance(payload, Mapping):
        raise ReleaseRowBundleError(f"{source_label} must contain a JSON object")
    # Validate schema and entries before any episode bytes are admitted.
    _episode_members_from_manifest(_manifest_entry_map(payload))
    return raw, payload


def _validate_tar_info(member: tarfile.TarInfo, *, total_bytes: int) -> int:
    """Validate one archive member and update the expanded-size accounting.

    Returns:
        Updated expanded-byte count.
    """

    _validate_member_path(member.name, label="archive member")
    if not (member.isdir() or member.isfile()):
        raise ReleaseRowBundleError("archive contains a non-regular member")
    if member.size < 0:
        raise ReleaseRowBundleError("archive contains a negative member size")
    total_bytes += member.size
    if total_bytes > MAX_EXPANDED_BYTES:
        raise ReleaseRowBundleError("archive expands beyond the safety limit")
    return total_bytes


def _validated_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Read and validate archive members before any member content is used.

    Returns:
        Validated archive members in archive order.
    """

    try:
        members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseRowBundleError("cannot enumerate release archive members") from exc
    if not members or len(members) > MAX_ARCHIVE_MEMBERS:
        raise ReleaseRowBundleError("archive member count is outside the supported limit")

    names: set[str] = set()
    expanded_bytes = 0
    for member in members:
        if member.name in names:
            raise ReleaseRowBundleError(f"duplicate archive member name: {member.name!r}")
        names.add(member.name)
        expanded_bytes = _validate_tar_info(member, total_bytes=expanded_bytes)
    return members


def _archive_relative_names(
    members: list[tarfile.TarInfo],
) -> tuple[dict[str, tarfile.TarInfo], tarfile.TarInfo]:
    """Strip the archive wrapper and index bundle-relative member names.

    Returns:
        Relative member index and the root-level publication manifest member.
    """

    manifest_members = [
        member for member in members if PurePosixPath(member.name).name == _MANIFEST_NAME
    ]
    if len(manifest_members) != 1:
        raise ReleaseRowBundleError("archive must contain exactly one publication_manifest.json")
    manifest_member = manifest_members[0]
    if not manifest_member.isfile() or manifest_member.size > MAX_MANIFEST_BYTES:
        raise ReleaseRowBundleError("archive publication manifest member is invalid")

    manifest_parts = _validate_member_path(manifest_member.name, label="archive manifest member")
    root_prefix = "/".join(manifest_parts[:-1])
    prefix = root_prefix + "/" if root_prefix else ""
    relative_names: dict[str, tarfile.TarInfo] = {}
    for member in members:
        if root_prefix and member.name == root_prefix:
            relative = ""
        elif root_prefix and member.name.startswith(prefix):
            relative = member.name[len(prefix) :]
        elif root_prefix:
            raise ReleaseRowBundleError("archive contains members outside its bundle root")
        else:
            relative = member.name
        if not relative:
            continue
        if relative in relative_names:
            raise ReleaseRowBundleError(f"duplicate bundle-relative archive member: {relative!r}")
        relative_names[relative] = member

    if _MANIFEST_NAME not in relative_names:
        raise ReleaseRowBundleError("archive publication manifest is not at its bundle root")
    return relative_names, manifest_member


def _archive_episode_members(
    relative_names: Mapping[str, tarfile.TarInfo],
) -> list[_EpisodeMember]:
    """Collect and validate direct ``payload/runs/<arm>/episodes.jsonl`` files.

    Returns:
        Episode member descriptors sorted by bundle-relative path.
    """

    episode_members: list[_EpisodeMember] = []
    for relative, member in sorted(relative_names.items()):
        if PurePosixPath(relative).name != _EPISODE_NAME:
            continue
        parsed = _relative_episode_path(relative, label="archive episode member")
        if parsed is None:
            raise ReleaseRowBundleError(
                f"archive contains episodes.jsonl outside payload/runs/<arm>: {relative!r}"
            )
        canonical_path, arm_directory = parsed
        if not member.isfile() or member.size > MAX_EPISODE_FILE_BYTES:
            raise ReleaseRowBundleError(f"archive episode member is invalid: {relative!r}")
        _release_arm(arm_directory, label=f"archive episode member {relative!r}")
        episode_members.append(
            _EpisodeMember(
                relative_path=canonical_path,
                arm_directory=arm_directory,
                member=member,
            )
        )
    if not episode_members:
        raise ReleaseRowBundleError("archive contains no payload/runs/*/episodes.jsonl members")
    return episode_members


def _archive_manifest_bytes(
    archive: tarfile.TarFile,
    manifest_member: tarfile.TarInfo,
) -> bytes:
    """Read the bounded publication manifest member.

    Returns:
        Raw publication-manifest bytes.
    """

    try:
        stream = archive.extractfile(manifest_member)
        if stream is None:
            raise ReleaseRowBundleError("archive publication manifest is unreadable")
        try:
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
        finally:
            stream.close()
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseRowBundleError("archive publication manifest is unreadable") from exc
    if len(raw) != manifest_member.size:
        raise ReleaseRowBundleError("archive publication manifest is truncated")
    return raw


def _archive_source(bundle: Path) -> tuple[_BundleSource, tarfile.TarFile]:
    """Open an archive and collect its manifest and episode member handles.

    Returns:
        Verified source metadata and an open archive handle.
    """

    try:
        archive_size = bundle.stat().st_size
    except OSError as exc:
        raise ReleaseRowBundleError(f"cannot stat archive {bundle}: {exc}") from exc
    if archive_size > MAX_ARCHIVE_BYTES:
        raise ReleaseRowBundleError("archive exceeds the input size limit")
    bundle_sha256 = _sha256_file(bundle)
    try:
        archive = tarfile.open(bundle, mode="r:*")
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseRowBundleError(f"cannot open release archive {bundle}") from exc

    try:
        members = _validated_archive_members(archive)
        relative_names, manifest_member = _archive_relative_names(members)
        episode_members = _archive_episode_members(relative_names)
        manifest_bytes = _archive_manifest_bytes(archive, manifest_member)
        manifest_bytes, manifest = _read_manifest_bytes(
            manifest_bytes, source_label="archive publication_manifest.json"
        )
        return (
            _BundleSource(
                manifest_bytes=manifest_bytes,
                manifest=manifest,
                episode_members=tuple(episode_members),
                bundle_sha256=bundle_sha256,
            ),
            archive,
        )
    except (OSError, ReleaseRowBundleError, tarfile.TarError):
        archive.close()
        raise


def _extracted_root(bundle: Path) -> Path:
    """Resolve a bundle directory, allowing one extracted archive wrapper.

    Returns:
        The resolved bundle root containing the publication manifest.
    """

    if bundle.is_symlink() or not bundle.is_dir():
        raise ReleaseRowBundleError(f"release bundle root is not a regular directory: {bundle}")
    root = bundle.resolve()
    manifest = root / _MANIFEST_NAME
    if manifest.is_file() and not manifest.is_symlink():
        return root
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir() and not child.is_symlink() and (child / _MANIFEST_NAME).is_file()
    ]
    if len(candidates) != 1:
        raise ReleaseRowBundleError("extracted bundle lacks a unique publication_manifest.json")
    return candidates[0].resolve()


def _extracted_episode_members(root: Path) -> list[_EpisodeMember]:
    """Collect and validate episode files below an extracted bundle root.

    Returns:
        Episode member descriptors sorted by bundle-relative path.
    """

    payload = root / "payload"
    runs = payload / "runs"
    if payload.is_symlink() or not payload.is_dir() or runs.is_symlink() or not runs.is_dir():
        raise ReleaseRowBundleError("extracted bundle payload/runs directory is missing or unsafe")

    # Symlinked non-episode files are also rejected: otherwise an apparently
    # read-only bundle could make this loader read outside the admitted root.
    try:
        all_entries = list(runs.rglob("*"))
    except OSError as exc:
        raise ReleaseRowBundleError("cannot enumerate extracted payload/runs") from exc
    if any(entry.is_symlink() for entry in all_entries):
        raise ReleaseRowBundleError("extracted payload/runs contains a symlink")

    episode_paths = sorted(runs.rglob(_EPISODE_NAME))
    members: list[_EpisodeMember] = []
    for path in episode_paths:
        if not path.is_file():
            raise ReleaseRowBundleError(f"extracted episode member is not a regular file: {path}")
        relative_to_root = path.relative_to(root).as_posix()
        parsed = _relative_episode_path(relative_to_root, label="extracted episode member")
        if parsed is None:
            raise ReleaseRowBundleError(
                "extracted bundle contains episodes.jsonl outside payload/runs/<arm>"
            )
        canonical_path, arm_directory = parsed
        _release_arm(arm_directory, label=f"extracted episode member {relative_to_root!r}")
        members.append(
            _EpisodeMember(
                relative_path=canonical_path,
                arm_directory=arm_directory,
                path=path,
            )
        )
    if not members:
        raise ReleaseRowBundleError("extracted bundle contains no payload/runs/*/episodes.jsonl")
    return members


def _extracted_source(bundle: Path) -> _BundleSource:
    """Collect and verify manifest and episode files from an extracted root.

    Returns:
        Verified source metadata for the extracted bundle.
    """

    root = _extracted_root(bundle)
    manifest_path = root / _MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ReleaseRowBundleError("extracted bundle publication manifest is missing or unsafe")
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ReleaseRowBundleError("publication_manifest.json exceeds the manifest size limit")
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ReleaseRowBundleError("extracted bundle publication manifest is unreadable") from exc
    manifest_bytes, manifest = _read_manifest_bytes(
        manifest_bytes, source_label="publication_manifest.json"
    )
    members = _extracted_episode_members(root)
    return _BundleSource(
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        episode_members=tuple(members),
        bundle_sha256=None,
    )


def _read_extracted_member(path: Path, *, relative_path: str) -> bytes:
    """Read one bounded extracted episode file.

    Returns:
        Raw episode-member bytes.
    """

    try:
        stat_before = path.stat()
        if path.is_symlink() or not path.is_file():
            raise ReleaseRowBundleError(f"episode member is not a regular file: {path}")
        if stat_before.st_size > MAX_EPISODE_FILE_BYTES:
            raise ReleaseRowBundleError(f"episode member exceeds the safety limit: {path}")
        raw = path.read_bytes()
        stat_after = path.stat()
    except OSError as exc:
        raise ReleaseRowBundleError(f"cannot read episode member {path}") from exc
    if stat_before.st_size != stat_after.st_size or len(raw) != stat_before.st_size:
        raise ReleaseRowBundleError(f"episode member changed while being read: {relative_path}")
    return raw


def _read_archive_member(archive: tarfile.TarFile, member: _EpisodeMember) -> bytes:
    """Read one bounded episode member from an open archive.

    Returns:
        Raw episode-member bytes.
    """

    if member.member is None:
        raise ReleaseRowBundleError("episode member has no readable source")
    if member.member.size > MAX_EPISODE_FILE_BYTES:
        raise ReleaseRowBundleError(
            f"episode member exceeds the safety limit: {member.relative_path}"
        )
    stream = archive.extractfile(member.member)
    if stream is None:
        raise ReleaseRowBundleError(f"episode member is unreadable: {member.relative_path}")
    try:
        raw = stream.read(MAX_EPISODE_FILE_BYTES + 1)
    except OSError as exc:
        raise ReleaseRowBundleError(f"cannot read episode member {member.relative_path}") from exc
    finally:
        stream.close()
    if len(raw) != member.member.size:
        raise ReleaseRowBundleError(f"episode member is truncated: {member.relative_path}")
    return raw


def _read_member_bytes(
    source: _BundleSource,
    member: _EpisodeMember,
    *,
    archive: tarfile.TarFile | None,
) -> bytes:
    """Read one bounded episode member and verify its manifest digest/size.

    Returns:
        The verified episode-member bytes.
    """

    manifest_entries = _episode_members_from_manifest(_manifest_entry_map(source.manifest))
    expected = manifest_entries.get(member.relative_path)
    if expected is None:
        raise ReleaseRowBundleError(
            f"episode member is missing from publication manifest: {member.relative_path}"
        )
    expected_size, expected_digest, _ = expected
    if expected_size > MAX_EPISODE_FILE_BYTES:
        raise ReleaseRowBundleError(
            f"episode member exceeds the safety limit: {member.relative_path}"
        )

    if member.path is not None:
        raw = _read_extracted_member(member.path, relative_path=member.relative_path)
    elif archive is not None:
        raw = _read_archive_member(archive, member)
    else:
        raise ReleaseRowBundleError("episode member has no readable source")
    if len(raw) != expected_size:
        raise ReleaseRowBundleError(
            f"episode member size does not match publication manifest: {member.relative_path}"
        )
    observed_digest = _sha256_bytes(raw)
    if observed_digest != expected_digest:
        raise ReleaseRowBundleError(
            f"episode member SHA-256 does not match publication manifest: {member.relative_path}"
        )
    return raw


def _parse_episode_rows(raw: bytes, *, member_path: str) -> list[dict[str, Any]]:
    """Parse one strict JSONL episode member.

    Returns:
        Parsed episode objects in source order.
    """

    rows: list[dict[str, Any]] = []
    seen_episode_ids: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if len(line) > MAX_EPISODE_ROW_BYTES:
            raise ReleaseRowBundleError(
                f"{member_path} row {line_number} exceeds the row size limit"
            )
        if not line.strip():
            raise ReleaseRowBundleError(f"{member_path} row {line_number} is empty")
        payload = _read_json(line, label=f"{member_path} row {line_number}")
        if not isinstance(payload, dict):
            raise ReleaseRowBundleError(f"{member_path} row {line_number} is not a JSON object")
        episode_id = payload.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ReleaseRowBundleError(
                f"{member_path} row {line_number} has no non-empty episode_id"
            )
        if episode_id in seen_episode_ids:
            raise ReleaseRowBundleError(
                f"{member_path} contains duplicate episode_id {episode_id!r}"
            )
        seen_episode_ids.add(episode_id)
        rows.append(payload)
        if len(rows) > MAX_EPISODE_ROWS:
            raise ReleaseRowBundleError("release bundle contains too many episode rows")
    # ``bytes.splitlines`` returns an empty list for an empty file, which is a
    # missing episode source rather than a valid zero-row release arm.
    if not rows:
        raise ReleaseRowBundleError(f"{member_path} contains no episode rows")
    return rows


def load_release_rows(bundle: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and verify release episode rows from an archive or extracted root.

    Args:
        bundle: A ``.tar.gz`` publication bundle or its extracted bundle root.

    Returns:
        ``(rows, source)`` where rows retain their source fields and have
        ``_release_arm`` and ``_source_member`` attached.  ``source`` carries
        immutable manifest/archive digests and the verified episode-member set.

    Raises:
        ReleaseRowBundleError: If the bundle is malformed, incomplete, unsafe,
            oversized, or differs from its publication manifest.
    """

    path = Path(bundle)
    if path.is_symlink():
        raise ReleaseRowBundleError(f"release bundle path must not be a symlink: {path}")
    if path.is_dir():
        source = _extracted_source(path)
        archive: tarfile.TarFile | None = None
        close_archive = False
    elif path.is_file():
        source, archive = _archive_source(path)
        close_archive = True
    else:
        raise ReleaseRowBundleError(f"release bundle does not exist: {path}")

    try:
        manifest_entries = _episode_members_from_manifest(_manifest_entry_map(source.manifest))
        observed_paths = {member.relative_path for member in source.episode_members}
        expected_paths = set(manifest_entries)
        if observed_paths != expected_paths:
            missing = sorted(expected_paths - observed_paths)
            unexpected = sorted(observed_paths - expected_paths)
            raise ReleaseRowBundleError(
                "publication manifest episode members do not match bundle files: "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )

        rows: list[dict[str, Any]] = []
        for member in source.episode_members:
            raw = _read_member_bytes(source, member, archive=archive)
            member_rows = _parse_episode_rows(raw, member_path=member.relative_path)
            arm = _release_arm(member.arm_directory, label=member.relative_path)
            for row in member_rows:
                enriched = dict(row)
                enriched["_release_arm"] = arm
                enriched["_source_member"] = member.relative_path
                rows.append(enriched)
                if len(rows) > MAX_EPISODE_ROWS:
                    raise ReleaseRowBundleError("release bundle contains too many episode rows")
        if not rows:
            raise ReleaseRowBundleError("release bundle contains no episode rows")
        source_payload = {
            "planner_ids": sorted(
                {
                    _release_arm(member.arm_directory, label=member.relative_path)
                    for member in source.episode_members
                }
            ),
            "bundle_sha256": source.bundle_sha256,
            "manifest_sha256": _sha256_bytes(source.manifest_bytes),
            "row_count": len(rows),
            "episode_members": [member.relative_path for member in source.episode_members],
        }
        return rows, source_payload
    finally:
        if close_archive and archive is not None:
            archive.close()


__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_ARCHIVE_MEMBERS",
    "MAX_EPISODE_FILE_BYTES",
    "MAX_EPISODE_ROWS",
    "MAX_EPISODE_ROW_BYTES",
    "MAX_EXPANDED_BYTES",
    "MAX_MANIFEST_BYTES",
    "PUBLICATION_BUNDLE_SCHEMA_VERSION",
    "ReleaseRowBundleError",
    "load_release_rows",
]
