#!/usr/bin/env python3
"""Revalidate preserved benchmark rows and build a public release bundle.

This helper is intentionally a *derived-artifact* workflow.  It never runs a
planner and never edits the producer campaign.  Instead it verifies the
checksum-pinned retrieval, runs the corrected full-release gate against the
original producer root, copies the verified tree into a fresh staging area,
and makes a public, path-sanitised projection of that tree.

The command is designed for the job-14890 recovery path, but the small pure
helpers are reusable by later recovery runs.  Publication is fail-closed:
producer checksums, the rejected producer result, acceptance, bundle
preflight, and path hygiene all have to pass before any staging directory is
promoted.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark import artifact_publication as artifact_publication_module
from robot_sf.benchmark import release_acceptance as release_acceptance_module
from robot_sf.benchmark import release_protocol as release_protocol_module
from robot_sf.benchmark.artifact_publication import (
    PublicationPreflightError,
    export_publication_bundle,
    verify_publication_bundle_preflight,
)
from robot_sf.benchmark.camera_ready import _config as camera_config_module
from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.camera_ready_campaign import write_campaign_report
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_acceptance import validate_full_benchmark_release_acceptance
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest

FROZEN_SOURCE_SHA = "b1d5ab6de708385c0828c99501a9d1c29727ec11"
EXPECTED_PRODUCER_SUMS_SHA256 = "2408431cef70bd7f7cf96fe0c42c44e84db89a841ea446e27fbb5650be713506"
EXPECTED_PRODUCER_RECEIPT_SHA256 = (
    "cac330ce79261147669843c842868f50ee4becfbc6258e69bdbb0c0357a1d823"
)
# A later read-only retrieval verification refreshed only ``verified_at`` in
# the receipt.  The preserved receipt remains the authoritative pre-refresh
# evidence; this digest is the admitted current retrieval counterpart.
EXPECTED_REFRESHED_PRODUCER_RECEIPT_SHA256 = (
    "1c9f19c066cd3153a1f426bb71bafc1b2b98cec76970128339088a22b80248fe"
)
EXPECTED_REJECTED_RESULT_SHA256 = "be3b7feca2a139565c59c70e36808572ac4378a181046294c6779f81f6d9ae0d"
EXPECTED_PRODUCER_FILE_COUNT = 109
PRODUCER_SUMS_NAME = "SHA256SUMS"
PRODUCER_RECEIPT_NAME = "artifact-verification-receipt.json"
REJECTED_RESULT_RELATIVE = "release/release_result.json"
DERIVATION_RECEIPT_RELATIVE = "provenance/derived_revalidation_receipt.json"
PUBLICATION_CUSTODY_NAME = "publication_custody.json"
SNQI_ADVISORY_BOUNDARY = (
    "SNQI calibration failed under warn and remains advisory only; it is not a planner-ranking "
    "authority for this release."
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PRIVATE_ABSOLUTE_PATH_RE = re.compile(
    rb"(?<![A-Za-z0-9_:/])/(?:home|tmp|scratch|dev/shm|gpfs|lustre|mnt|work)/[^\s\"'<>`]+"
)
_BINARY_SUFFIXES = {
    ".7z",
    ".bin",
    ".bz2",
    ".gif",
    ".jpeg",
    ".jpg",
    ".lz4",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pdf",
    ".pickle",
    ".png",
    ".pt",
    ".pth",
    ".tar",
    ".webp",
    ".xz",
    ".zip",
    ".zst",
    ".gz",
}


class DerivedReleaseError(RuntimeError):
    """Raised when a preserved release cannot be promoted safely."""


def _assert_safe_directory(path: Path, *, label: str) -> Path:
    """Require a real directory with no symlink path component."""
    lexical = Path(path.absolute())
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise DerivedReleaseError(f"{label} contains a symlink component")
    if not lexical.is_dir():
        raise DerivedReleaseError(f"{label} is not a directory")
    return lexical


def _is_text_capable(path: Path) -> bool:
    """Use a strict binary allowlist; every other file is scanned as text."""
    return path.suffix.lower() not in _BINARY_SUFFIXES


@contextlib.contextmanager
def _source_repository_binding(source_root: Path, *, validator_root: Path | None = None):
    """Bind repository-aware protocol/export helpers to the frozen checkout."""
    previous_protocol_root = release_protocol_module.get_repository_root
    previous_publication_root = artifact_publication_module.get_repository_root
    previous_config_root = camera_config_module.get_repository_root
    previous_acceptance_root = release_acceptance_module.get_repository_root
    release_protocol_module.get_repository_root = lambda: source_root
    artifact_publication_module.get_repository_root = lambda: source_root
    camera_config_module.get_repository_root = lambda: source_root
    release_acceptance_module.get_repository_root = lambda: validator_root or source_root
    try:
        yield
    finally:
        release_protocol_module.get_repository_root = previous_protocol_root
        artifact_publication_module.get_repository_root = previous_publication_root
        camera_config_module.get_repository_root = previous_config_root
        release_acceptance_module.get_repository_root = previous_acceptance_root


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and fail closed on malformed input."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DerivedReleaseError(f"invalid JSON input: {path.name}") from exc
    if not isinstance(payload, dict):
        raise DerivedReleaseError(f"JSON input must be an object: {path.name}")
    return payload


def _read_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    """Read a JSON object from bytes without exposing its source path."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DerivedReleaseError(f"invalid JSON input: {label}") from exc
    if not isinstance(payload, dict):
        raise DerivedReleaseError(f"JSON input must be an object: {label}")
    return payload


def _read_single_gzip_member(path: Path) -> bytes:
    """Read exactly one gzip member and reject truncation or trailing data."""
    if path.is_symlink() or not path.is_file():
        raise DerivedReleaseError("preserved receipt source is missing or unsafe")
    try:
        compressed = path.read_bytes()
    except OSError as exc:
        raise DerivedReleaseError("preserved receipt source cannot be read") from exc
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        payload = decompressor.decompress(compressed)
        payload += decompressor.flush()
    except zlib.error as exc:
        raise DerivedReleaseError("preserved receipt gzip is invalid") from exc
    if not decompressor.eof:
        raise DerivedReleaseError("preserved receipt gzip is truncated")
    if decompressor.unused_data or decompressor.unconsumed_tail:
        raise DerivedReleaseError("preserved receipt gzip has trailing data or members")
    if not payload:
        raise DerivedReleaseError("preserved receipt gzip is empty")
    return payload


def _sha256_bytes(data: bytes) -> str:
    """Return a SHA-256 digest for an in-memory receipt payload."""
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable, UTF-8 JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _relative_path(root: Path, path: Path) -> str:
    """Return a POSIX path relative to ``root``."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _all_tree_files(root: Path) -> list[Path]:
    """Return regular files while rejecting symlink components."""
    root = _assert_safe_directory(root, label="artifact root")
    files: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise DerivedReleaseError("producer tree contains a symlink")
        if candidate.is_file():
            files.append(candidate)
        elif not candidate.is_dir():
            raise DerivedReleaseError("producer tree contains a non-regular entry")
    return files


def _parse_sha256sums(path: Path, *, root: Path) -> dict[str, str]:
    """Parse a strict root-relative SHA256SUMS file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DerivedReleaseError("producer SHA256SUMS cannot be read") from exc
    entries: dict[str, str] = {}
    for number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not _SHA256_RE.fullmatch(parts[0]):
            raise DerivedReleaseError(f"producer SHA256SUMS line {number} is malformed")
        raw_relative = parts[1].lstrip("*")
        candidate = Path(raw_relative)
        if candidate.is_absolute() or ".." in candidate.parts or not raw_relative:
            raise DerivedReleaseError("producer SHA256SUMS contains an unsafe path")
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root.resolve()):
            raise DerivedReleaseError("producer SHA256SUMS escapes its root")
        relative = candidate.as_posix()
        if relative in entries:
            raise DerivedReleaseError(f"producer SHA256SUMS duplicates {relative}")
        entries[relative] = parts[0].lower()
    if not entries:
        raise DerivedReleaseError("producer SHA256SUMS is empty")
    return dict(sorted(entries.items()))


def _verify_hashes(root: Path, entries: Mapping[str, str]) -> None:
    """Verify every signed file exists, is regular, and has the expected digest."""
    for relative, expected in entries.items():
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise DerivedReleaseError(f"producer checksum file is missing or unsafe: {relative}")
        if sha256_file(candidate).lower() != expected.lower():
            raise DerivedReleaseError(f"producer checksum mismatch: {relative}")


def _build_file_map(root: Path, relative_paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Build a complete path/size/SHA map for a verified artifact root."""
    file_map: dict[str, dict[str, Any]] = {}
    for relative in sorted(relative_paths):
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise DerivedReleaseError(f"artifact map entry is missing or unsafe: {relative}")
        file_map[relative] = {
            "bytes": candidate.stat().st_size,
            "sha256": sha256_file(candidate).lower(),
        }
    return file_map


def _verify_campaign_file_map(
    campaign_root: Path,
    *,
    expected_sums_sha256: str,
    expected_result_sha256: str,
    expected_file_count: int,
) -> dict[str, Any]:
    """Verify an accepted campaign root, excluding only its optional receipt."""
    root = _assert_safe_directory(campaign_root, label="accepted campaign root")
    all_files = _all_tree_files(root)
    sums_path = root / PRODUCER_SUMS_NAME
    result_path = root / REJECTED_RESULT_RELATIVE
    if sums_path.is_symlink() or not sums_path.is_file():
        raise DerivedReleaseError("accepted campaign SHA256SUMS is missing or unsafe")
    if result_path.is_symlink() or not result_path.is_file():
        raise DerivedReleaseError("accepted campaign release_result is missing or unsafe")
    sums_digest = sha256_file(sums_path).lower()
    if sums_digest != expected_sums_sha256.lower():
        raise DerivedReleaseError("accepted campaign SHA256SUMS digest is not admitted")
    result_digest = sha256_file(result_path).lower()
    if result_digest != expected_result_sha256.lower():
        raise DerivedReleaseError("accepted campaign release_result digest is not admitted")
    entries = _parse_sha256sums(sums_path, root=root)
    if len(entries) != expected_file_count:
        raise DerivedReleaseError(
            f"accepted campaign SHA256SUMS must list {expected_file_count} files"
        )
    _verify_hashes(root, entries)
    listed = set(entries)
    optional_receipt = (
        {PRODUCER_RECEIPT_NAME} if (root / PRODUCER_RECEIPT_NAME).is_file() else set()
    )
    actual = {_relative_path(root, path) for path in all_files}
    expected_actual = listed | {PRODUCER_SUMS_NAME} | optional_receipt
    if actual != expected_actual:
        raise DerivedReleaseError("accepted campaign file inventory does not match SHA256SUMS")
    map_paths = sorted(listed | {PRODUCER_SUMS_NAME})
    return {
        "status": "verified",
        "listed_file_count": len(entries),
        "total_file_count": len(actual),
        "sha256sums_sha256": sums_digest,
        "rejected_release_result_sha256": result_digest,
        "file_map": _build_file_map(root, map_paths),
    }


def _assert_equal_file_maps(accepted: Mapping[str, Any], retrieved: Mapping[str, Any]) -> None:
    """Require accepted and published roots to have identical path/size/SHA maps."""
    if accepted.get("file_map") != retrieved.get("file_map"):
        accepted_map = accepted.get("file_map")
        retrieved_map = retrieved.get("file_map")
        if not isinstance(accepted_map, Mapping) or not isinstance(retrieved_map, Mapping):
            raise DerivedReleaseError("accepted/retrieved file maps are missing")
        paths = sorted(set(accepted_map) | set(retrieved_map))
        for path in paths:
            if accepted_map.get(path) != retrieved_map.get(path):
                raise DerivedReleaseError(f"accepted/retrieved file map mismatch: {path}")
        raise DerivedReleaseError("accepted/retrieved file maps differ")


def _json_difference_paths(left: Any, right: Any, *, prefix: str = "") -> list[str]:
    """Return deterministic JSON paths whose values differ."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_json_difference_paths(left[key], right[key], prefix=path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        for index in range(max(len(left), len(right))):
            path = f"{prefix}[{index}]"
            if index >= len(left) or index >= len(right):
                differences.append(path)
            else:
                differences.extend(_json_difference_paths(left[index], right[index], prefix=path))
        return differences
    return [] if left == right else [prefix or "<root>"]


def _validate_receipt(  # noqa: C901
    receipt: Mapping[str, Any],
    *,
    label: str,
    root: Path,
    sums_digest: str,
    entries: Mapping[str, str],
    expected_file_count: int,
) -> None:
    """Validate one receipt's table independently against current producer bytes."""
    if receipt.get("status") != "verified":
        raise DerivedReleaseError(f"{label} is not verified")
    if receipt.get("file_count") != expected_file_count:
        raise DerivedReleaseError(f"{label} file_count disagrees")
    if str(receipt.get("manifest_sha256", "")).lower() != sums_digest:
        raise DerivedReleaseError(f"{label} manifest hash disagrees")
    receipt_files = receipt.get("files")
    if not isinstance(receipt_files, list) or len(receipt_files) != expected_file_count:
        raise DerivedReleaseError(f"{label} files are incomplete")
    receipt_entries: dict[str, str] = {}
    for item in receipt_files:
        if not isinstance(item, Mapping):
            raise DerivedReleaseError(f"{label} has malformed files")
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DerivedReleaseError(f"{label} has malformed entry")
        if relative in receipt_entries:
            raise DerivedReleaseError(f"{label} duplicates a path")
        receipt_entries[relative] = digest.lower()
        byte_count = item.get("bytes")
        if byte_count is not None:
            candidate = root / relative
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or not candidate.is_file()
                or candidate.stat().st_size != byte_count
            ):
                raise DerivedReleaseError(f"{label} byte count disagrees: {relative}")
    if receipt_entries != dict(entries):
        raise DerivedReleaseError(f"{label} disagrees with SHA256SUMS")
    # Deliberately hash from this receipt's table, rather than relying only on
    # the already checked SHA256SUMS table.  This keeps the preserved and
    # refreshed receipt checks independent if their schemas ever diverge.
    _verify_hashes(root, receipt_entries)


def _validate_preserved_refresh(
    preserved: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Require that a refreshed receipt differs only in its timestamp."""
    differences = _json_difference_paths(preserved, current)
    if differences != ["verified_at"]:
        raise DerivedReleaseError("preserved/current artifact receipts differ outside verified_at")
    previous = preserved.get("verified_at")
    refreshed = current.get("verified_at")
    if not isinstance(previous, str) or not isinstance(refreshed, str) or previous == refreshed:
        raise DerivedReleaseError("artifact receipt verified_at was not refreshed")
    for value in (previous, refreshed):
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise DerivedReleaseError(
                "artifact receipt verified_at is not an ISO timestamp"
            ) from exc
    return {
        "allowed_difference": "verified_at",
        "difference_paths": differences,
        "preserved_verified_at": previous,
        "current_verified_at": refreshed,
    }


def verify_producer_artifacts(  # noqa: C901
    producer_root: Path,
    *,
    expected_sums_sha256: str = EXPECTED_PRODUCER_SUMS_SHA256,
    expected_receipt_sha256: str | None = None,
    preserved_receipt_source: Path | None = None,
    expected_preserved_receipt_sha256: str = EXPECTED_PRODUCER_RECEIPT_SHA256,
    expected_rejected_result_sha256: str = EXPECTED_REJECTED_RESULT_SHA256,
    expected_file_count: int = EXPECTED_PRODUCER_FILE_COUNT,
) -> dict[str, Any]:
    """Verify the immutable retrieval contract before any derived copy.

    The default contract is the preserved job-14890 retrieval: 109 files
    listed by ``SHA256SUMS``, plus the checksum manifest and its verification
    receipt (111 regular files total).  When ``preserved_receipt_source`` is
    supplied, it must be a single gzip member containing the immutable
    pre-refresh receipt.  The live retrieval receipt is then admitted only
    when its semantic JSON differs from that receipt at ``verified_at``.
    Optional expected values remain explicit so small unit fixtures can
    exercise the same fail-closed logic.
    """
    root = producer_root.resolve()
    admitted_receipt_sha256 = expected_receipt_sha256 or (
        EXPECTED_REFRESHED_PRODUCER_RECEIPT_SHA256
        if preserved_receipt_source is not None
        else EXPECTED_PRODUCER_RECEIPT_SHA256
    )
    all_files = _all_tree_files(root)
    sums_path = root / PRODUCER_SUMS_NAME
    receipt_path = root / PRODUCER_RECEIPT_NAME
    result_path = root / REJECTED_RESULT_RELATIVE
    for required in (sums_path, receipt_path, result_path):
        if required.is_symlink() or not required.is_file():
            raise DerivedReleaseError(
                f"producer required file is missing or unsafe: {required.name}"
            )

    sums_digest = sha256_file(sums_path).lower()
    if sums_digest != expected_sums_sha256.lower():
        raise DerivedReleaseError("producer SHA256SUMS digest does not match the admitted receipt")
    current_receipt_bytes = receipt_path.read_bytes()
    receipt_digest = _sha256_bytes(current_receipt_bytes)
    if receipt_digest != admitted_receipt_sha256.lower():
        raise DerivedReleaseError(
            "producer artifact-verification-receipt digest does not match the admitted receipt"
        )
    result_digest = sha256_file(result_path).lower()
    if result_digest != expected_rejected_result_sha256.lower():
        raise DerivedReleaseError("producer rejected release_result digest does not match")

    entries = _parse_sha256sums(sums_path, root=root)
    if len(entries) != expected_file_count:
        raise DerivedReleaseError(
            f"producer SHA256SUMS must list {expected_file_count} files, found {len(entries)}"
        )
    _verify_hashes(root, entries)

    listed = set(entries)
    actual = {_relative_path(root, path) for path in all_files}
    expected_actual = listed | {PRODUCER_SUMS_NAME, PRODUCER_RECEIPT_NAME}
    if actual != expected_actual:
        extras = sorted(actual - expected_actual)
        missing = sorted(expected_actual - actual)
        detail = []
        if extras:
            detail.append(f"unlisted files: {extras[:4]}")
        if missing:
            detail.append(f"missing files: {missing[:4]}")
        raise DerivedReleaseError("producer file inventory mismatch (" + "; ".join(detail) + ")")

    current_receipt = _read_json_bytes(
        current_receipt_bytes, label="producer artifact-verification-receipt.json"
    )
    _validate_receipt(
        current_receipt,
        label="producer artifact-verification-receipt",
        root=root,
        sums_digest=sums_digest,
        entries=entries,
        expected_file_count=expected_file_count,
    )

    preserved_receipt: dict[str, Any] | None = None
    preserved_receipt_bytes: bytes | None = None
    receipt_refresh: dict[str, Any] | None = None
    preserved_receipt_digest: str | None = None
    preserved_receipt_compressed_digest: str | None = None
    if preserved_receipt_source is not None:
        try:
            preserved_receipt_compressed_digest = sha256_file(preserved_receipt_source)
        except OSError as exc:
            raise DerivedReleaseError("preserved receipt gzip cannot be hashed") from exc
        preserved_receipt_bytes = _read_single_gzip_member(preserved_receipt_source)
        preserved_receipt_digest = _sha256_bytes(preserved_receipt_bytes)
        if preserved_receipt_digest != expected_preserved_receipt_sha256.lower():
            raise DerivedReleaseError(
                "preserved artifact-verification-receipt digest does not match the admitted receipt"
            )
        preserved_receipt = _read_json_bytes(
            preserved_receipt_bytes, label="preserved artifact-verification-receipt.json.gz"
        )
        _validate_receipt(
            preserved_receipt,
            label="preserved artifact-verification-receipt",
            root=root,
            sums_digest=sums_digest,
            entries=entries,
            expected_file_count=expected_file_count,
        )
        receipt_refresh = _validate_preserved_refresh(preserved_receipt, current_receipt)

    return {
        "status": "verified",
        "listed_file_count": len(entries),
        "total_file_count": len(actual),
        "sha256sums_sha256": sums_digest,
        "artifact_verification_receipt_sha256": receipt_digest,
        "preserved_artifact_verification_receipt_sha256": preserved_receipt_digest,
        "preserved_artifact_verification_receipt_compressed_sha256": preserved_receipt_compressed_digest,
        "artifact_receipt_refresh": receipt_refresh,
        "rejected_release_result_sha256": result_digest,
        # The receipt is intentionally not part of this cross-root map: the
        # local retrieval carries the refreshed receipt while the accepted
        # producer root has no receipt.  Both receipt payloads are bound below.
        "file_map": _build_file_map(root, sorted(listed | {PRODUCER_SUMS_NAME})),
        "files": dict(entries),
        # Internal-only bytes let the copy step carry exactly the verified
        # preserved payload without rereading a mutable external source.  CLI
        # output filters these keys; they never enter a publication receipt.
        "_current_receipt_bytes": current_receipt_bytes,
        "_preserved_receipt_bytes": preserved_receipt_bytes,
    }


def _copy_tree_without_symlinks(
    source: Path,
    destination: Path,
    *,
    expected_file_map: Mapping[str, Mapping[str, Any]],
) -> None:
    """Copy a producer tree after checking every source entry for symlinks."""
    source_files = _all_tree_files(source)
    source_relatives = {_relative_path(source, path) for path in source_files}
    if source_relatives != set(expected_file_map):
        raise DerivedReleaseError("producer inventory changed before copying")
    destination.mkdir(parents=True, exist_ok=False)
    for source_path in sorted(source.rglob("*")):
        if source_path.is_symlink():
            raise DerivedReleaseError("producer tree changed to include a symlink during copy")
        relative = source_path.relative_to(source)
        destination_path = destination / relative
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            relative_name = relative.as_posix()
            expected = expected_file_map.get(relative_name)
            if expected is None or destination_path.is_symlink():
                raise DerivedReleaseError(f"staged copy has an unexpected file: {relative_name}")
            observed = {
                "bytes": destination_path.stat().st_size,
                "sha256": sha256_file(destination_path).lower(),
            }
            if observed != dict(expected):
                raise DerivedReleaseError(
                    f"staged copy does not match admitted bytes: {relative_name}"
                )
        else:
            raise DerivedReleaseError("producer tree changed to include a non-regular entry")


def _sanitise_bytes(data: bytes, *, source_root: Path, producer_root: Path) -> bytes:
    """Remove private absolute paths while retaining portable relative paths."""
    result = data
    # These two roots are safe to turn into portable paths: sidecar validation
    # accepts paths relative to either the source checkout or campaign root.
    for root in sorted(
        (source_root.resolve(), producer_root.resolve()), key=lambda value: -len(str(value))
    ):
        root_bytes = str(root).encode("utf-8")
        result = result.replace(root_bytes + b"/", b"")
        result = result.replace(root_bytes, b".")
    return _PRIVATE_ABSOLUTE_PATH_RE.sub(b"<external-path>", result)


def _sanitise_tree_paths(root: Path, *, source_root: Path, producer_root: Path) -> None:
    """Rewrite private path strings in copied text artifacts using streaming I/O."""
    for path in _all_tree_files(root):
        if not _is_text_capable(path):
            continue
        temporary = path.with_name(path.name + ".path-sanitising")
        changed = False
        carry = b""
        try:
            with path.open("rb") as source_handle, temporary.open("wb") as target_handle:
                while chunk := source_handle.read(1024 * 1024):
                    data = carry + chunk
                    # Keep enough overlap for a long absolute prefix/path to
                    # cross a chunk boundary.  The output is still streamed.
                    if len(data) > 8192:
                        body, carry = data[:-8192], data[-8192:]
                    else:
                        body, carry = b"", data
                    replaced = _sanitise_bytes(
                        body, source_root=source_root, producer_root=producer_root
                    )
                    changed = changed or replaced != body
                    target_handle.write(replaced)
                replaced = _sanitise_bytes(
                    carry, source_root=source_root, producer_root=producer_root
                )
                changed = changed or replaced != carry
                target_handle.write(replaced)
            if changed:
                os.replace(temporary, path)
            else:
                temporary.unlink()
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise DerivedReleaseError("could not sanitise copied artifact paths") from exc


def _assert_no_private_absolute_paths(root: Path) -> None:
    """Reject private absolute path markers in the public projection."""
    for path in _all_tree_files(root):
        if not _is_text_capable(path):
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DerivedReleaseError("could not inspect public artifact path hygiene") from exc
        if _PRIVATE_ABSOLUTE_PATH_RE.search(data):
            raise DerivedReleaseError("public artifact contains a private absolute path")


def _copy_producer_projection(
    producer_root: Path,
    *,
    staging_root: Path,
    source_root: Path,
    expected_file_map: Mapping[str, Mapping[str, Any]],
    current_receipt_bytes: bytes,
    preserved_receipt_bytes: bytes | None,
) -> None:
    """Copy and prepare a public projection without changing the producer."""
    _copy_tree_without_symlinks(
        producer_root,
        staging_root,
        expected_file_map=expected_file_map,
    )
    provenance_dir = staging_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the rejected result before any path projection.  The public
    # copy is sanitised below; its source digest is bound in the derivation
    # receipt so the rejection cannot be silently rewritten.
    producer_result = staging_root / REJECTED_RESULT_RELATIVE
    shutil.copy2(producer_result, producer_result.parent / "producer_release_result.rejected.json")
    copied_current_receipt = (staging_root / PRODUCER_RECEIPT_NAME).read_bytes()
    if copied_current_receipt != current_receipt_bytes:
        raise DerivedReleaseError("producer receipt changed while copying")
    # The original retrieval manifest/receipt are evidence of the source copy,
    # not the new derived tree.  Preserve them under a stable namespace before
    # path sanitisation and generate a fresh derived SHA256SUMS below.
    source_path = staging_root / PRODUCER_SUMS_NAME
    shutil.copy2(source_path, provenance_dir / "producer_SHA256SUMS")
    source_path.unlink()
    preserved_bytes = preserved_receipt_bytes or current_receipt_bytes
    (provenance_dir / "producer_artifact_verification_receipt.json").write_bytes(preserved_bytes)
    if preserved_receipt_bytes is not None:
        (provenance_dir / "current_producer_artifact_verification_receipt.json").write_bytes(
            current_receipt_bytes
        )
    (staging_root / PRODUCER_RECEIPT_NAME).unlink()
    _sanitise_tree_paths(staging_root, source_root=source_root, producer_root=producer_root)
    _assert_no_private_absolute_paths(staging_root)


def _write_derived_checksums(root: Path) -> dict[str, str]:
    """Write a fresh checksum list for the derived tree (excluding itself)."""
    sums_path = root / PRODUCER_SUMS_NAME
    entries = {
        _relative_path(root, path): sha256_file(path)
        for path in _all_tree_files(root)
        if path != sums_path
    }
    sums_path.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in sorted(entries.items())),
        encoding="utf-8",
    )
    return entries


def _git_value(args: Sequence[str], *, cwd: Path) -> str:
    """Read a short, non-secret Git value."""
    completed = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=15
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not value:
        raise DerivedReleaseError("could not resolve validator Git provenance")
    return value


def _validator_provenance(repo_root: Path, *, expected_commit: str) -> dict[str, str]:
    """Require a clean reviewed validator checkout and return its provenance."""
    _assert_safe_directory(repo_root, label="validator repository root")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_commit):
        raise DerivedReleaseError("expected validator commit must be a 40-character lowercase SHA")
    actual_commit = _git_value(["rev-parse", "HEAD^{commit}"], cwd=repo_root).lower()
    if actual_commit != expected_commit:
        raise DerivedReleaseError("validator checkout is not the expected reviewed commit")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise DerivedReleaseError("validator checkout is not clean")
    relative = Path("robot_sf/benchmark/release_acceptance.py")
    path = repo_root / relative
    if not path.is_file():
        raise DerivedReleaseError("validator implementation is missing")
    return {
        "commit": actual_commit,
        "expected_reviewed_commit": expected_commit,
        "file": relative.as_posix(),
        "file_sha256": sha256_file(path),
    }


def _assert_manifest_paths_from_source(manifest: Any, source_root: Path) -> None:  # noqa: C901
    """Ensure every manifest-resolved asset is owned by the frozen checkout."""
    source_root = _assert_safe_directory(source_root, label="source repository root")
    fields = (
        "path",
        "canonical_campaign_config_path",
        "scenario_matrix_path",
        "citation_path",
        "release_checklist_path",
        "snqi_weights_path",
        "snqi_baseline_path",
        "suite_policy_path",
        "route_certification_path",
        "metadata_path",
        "stress_smoke_suite_policy_path",
        "stress_smoke_seed_sets_path",
        "stress_smoke_route_certification_path",
    )
    for field in fields:
        value = getattr(manifest, field, None)
        if value is None:
            continue
        if not isinstance(value, Path):
            raise DerivedReleaseError(f"manifest {field} is not a resolved Path")
        candidate = Path(value.absolute())
        if any(component.is_symlink() for component in candidate.parents) or candidate.is_symlink():
            raise DerivedReleaseError(f"manifest {field} contains a symlink")
        try:
            candidate.resolve().relative_to(source_root.resolve())
        except ValueError as exc:
            raise DerivedReleaseError(
                f"manifest {field} is not bound to the frozen source repository"
            ) from exc
    for field in ("stress_smoke_scenario_source_pins", "stress_smoke_hybrid_config_pins"):
        pins = getattr(manifest, field, ())
        for pin in pins:
            path = getattr(pin, "path", None)
            if not isinstance(path, Path) or not path.resolve().is_relative_to(
                source_root.resolve()
            ):
                raise DerivedReleaseError(f"manifest {field} contains an external asset")
    seed_policy = getattr(manifest, "seed_policy", {})
    raw_seed_path = seed_policy.get("seed_sets_path") if isinstance(seed_policy, Mapping) else None
    if raw_seed_path:
        candidate = Path(str(raw_seed_path))
        candidate = candidate if candidate.is_absolute() else source_root / candidate
        if not candidate.resolve().is_relative_to(source_root.resolve()):
            raise DerivedReleaseError("manifest seed_sets_path is not bound to frozen source")


def _assert_frozen_source_repository(source_root: Path, expected_sha: str) -> None:
    """Require the explicitly supplied source checkout to be clean and exact."""
    _assert_safe_directory(source_root, label="source repository root")
    actual = _git_value(["rev-parse", "HEAD^{commit}"], cwd=source_root).lower()
    if actual != expected_sha:
        raise DerivedReleaseError("source repository checkout is not the frozen execution SHA")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise DerivedReleaseError("source repository checkout is not clean")


def _safe_publication_descriptor(
    *,
    bundle_name: str,
    result: Any,
    publication_relative_dir: str,
) -> dict[str, Any]:
    """Build a path-portable descriptor without leaking staging paths."""
    prefix = publication_relative_dir.strip("/")
    return {
        "bundle_dir": f"{prefix}/{bundle_name}",
        "archive_path": f"{prefix}/{bundle_name}.tar.gz",
        "manifest_path": f"{prefix}/{bundle_name}/publication_manifest.json",
        "checksums_path": f"{prefix}/{bundle_name}/checksums.sha256",
        "file_count": int(result.file_count),
        "total_bytes": int(result.total_bytes),
    }


def _set_accepted_release_metadata(
    campaign_root: Path,
    *,
    acceptance: Mapping[str, Any],
    producer_result: Mapping[str, Any],
    publication_descriptor: Mapping[str, Any],
) -> None:
    """Turn only the copied release metadata into an accepted derived result."""
    result_path = campaign_root / "release" / "release_result.json"
    result = dict(_read_json(result_path))
    result.update(
        {
            "status": "benchmark_success",
            "status_reason": "preserved campaign passed corrected full-release acceptance",
            "evidence_status": "valid",
            "benchmark_success": True,
            "campaign_benchmark_success": True,
            "release_benchmark_success": True,
            "release_status": "ok",
            "release_status_reason": "derived release acceptance and publication preflight passed",
            "release_exit_code": 0,
            "publication_requested": True,
            "publication_preflight_status": "pass",
            "publication_bundle": dict(publication_descriptor),
            "release_acceptance": dict(acceptance),
            "full_release_acceptance": dict(acceptance),
            "derivation": {
                "mode": "preserved_rows_corrected_validator",
                "producer_release_result_sha256": sha256_file(
                    campaign_root / "release" / "producer_release_result.rejected.json"
                ),
                "producer_release_status": producer_result.get("release_status"),
                "snqi_claim_boundary": SNQI_ADVISORY_BOUNDARY,
            },
        }
    )
    _write_json(result_path, result)

    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = _read_json(summary_path)
    campaign = summary.get("campaign")
    campaign = dict(campaign) if isinstance(campaign, Mapping) else {}
    campaign.update(
        {
            "status": "benchmark_success",
            "evidence_status": "valid",
            "benchmark_success": True,
            "release_benchmark_success": True,
            "full_release_acceptance_status": "valid",
        }
    )
    summary["campaign"] = campaign
    summary["full_release_acceptance"] = dict(acceptance)
    summary["publication_bundle"] = dict(publication_descriptor)
    _write_json(summary_path, summary)
    write_campaign_report(campaign_root / "reports" / "campaign_report.md", summary)


def _write_derivation_receipt(
    campaign_root: Path,
    *,
    producer_evidence: Mapping[str, Any],
    accepted_evidence: Mapping[str, Any],
    manifest_validation: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    validator: Mapping[str, str],
    source_sha: str,
    derived_checksums: Mapping[str, str],
) -> None:
    """Write the credential-free binding receipt for the derived projection."""
    receipt = {
        "schema_version": "benchmark-derived-revalidation.v1",
        "created_at_utc": _utc_now(),
        "mode": "preserved_rows_corrected_validator",
        "source": {
            "execution_commit": source_sha,
            "producer_release_result_sha256": producer_evidence["rejected_release_result_sha256"],
            "producer_sha256sums_sha256": producer_evidence["sha256sums_sha256"],
            "producer_artifact_verification_receipt_sha256": producer_evidence[
                "artifact_verification_receipt_sha256"
            ],
            "preserved_artifact_verification_receipt_sha256": producer_evidence.get(
                "preserved_artifact_verification_receipt_sha256"
            ),
            "preserved_artifact_verification_receipt_compressed_sha256": producer_evidence.get(
                "preserved_artifact_verification_receipt_compressed_sha256"
            ),
            "artifact_receipt_refresh": producer_evidence.get("artifact_receipt_refresh"),
            "producer_listed_file_count": producer_evidence["listed_file_count"],
            "producer_total_file_count": producer_evidence["total_file_count"],
        },
        "cross_root_binding": {
            "accepted_file_map": accepted_evidence.get("file_map"),
            "retrieved_file_map": producer_evidence.get("file_map"),
            "separate_receipt_policy": "verification receipt is excluded from cross-root map",
        },
        "manifest_validation": dict(manifest_validation),
        "validator": dict(validator),
        "acceptance": dict(acceptance),
        "derived_projection": {
            "campaign_root": "(this directory)",
            "sha256sums_file": PRODUCER_SUMS_NAME,
            "sha256sums_entry_count": len(derived_checksums),
            "sha256sums_entry_count_scope": "pre_publication_projection",
            "sha256sums_excludes_only": PRODUCER_SUMS_NAME,
            "final_inventory_rewritten_after_publication": True,
            "final_inventory_includes": [
                "publication subtree",
                "publication archive",
                PUBLICATION_CUSTODY_NAME,
            ],
            "producer_rejected_result": "release/producer_release_result.rejected.json",
        },
        "snqi": {
            "status": "advisory",
            "claim_boundary": SNQI_ADVISORY_BOUNDARY,
        },
        "credentials": "not_recorded",
    }
    _write_json(campaign_root / DERIVATION_RECEIPT_RELATIVE, receipt)


def _write_custody_receipt(
    publication_dir: Path,
    *,
    bundle_name: str,
    archive_path: Path,
    bundle_dir: Path,
    source_sha: str,
) -> None:
    """Bind the final archive outside the bundle, avoiding a checksum cycle."""
    payload = {
        "schema_version": "benchmark-publication-custody.v1",
        "created_at_utc": _utc_now(),
        "bundle_name": bundle_name,
        "source_execution_commit": source_sha,
        "archive": {
            "path": f"{bundle_name}.tar.gz",
            "sha256": sha256_file(archive_path),
            "size_bytes": archive_path.stat().st_size,
        },
        "bundle": {
            "path": bundle_name,
            "publication_manifest_sha256": sha256_file(bundle_dir / "publication_manifest.json"),
            "checksums_sha256": sha256_file(bundle_dir / "checksums.sha256"),
        },
        "archive_self_digest_policy": "archive digest is external to the bundle; no cycle",
        "credentials": "not_recorded",
    }
    _write_json(publication_dir / PUBLICATION_CUSTODY_NAME, payload)


def _export_stabilised_bundle(  # noqa: PLR0913
    campaign_root: Path,
    *,
    publication_stage: Path,
    bundle_name: str,
    acceptance: Mapping[str, Any],
    producer_result: Mapping[str, Any],
    release_tag: str,
    doi: str,
    repository_url: str,
    publication_relative_dir: str,
) -> tuple[dict[str, Any], Path, Path]:
    """Export, write descriptors, and repeat until the descriptor is stable."""
    descriptor: dict[str, Any] = {
        "bundle_dir": f"{publication_relative_dir}/{bundle_name}",
        "archive_path": f"{publication_relative_dir}/{bundle_name}.tar.gz",
        "manifest_path": f"{publication_relative_dir}/{bundle_name}/publication_manifest.json",
        "checksums_path": f"{publication_relative_dir}/{bundle_name}/checksums.sha256",
        "file_count": 0,
        "total_bytes": 0,
    }
    for _ in range(5):
        _set_accepted_release_metadata(
            campaign_root,
            acceptance=acceptance,
            producer_result=producer_result,
            publication_descriptor=descriptor,
        )
        exported = export_publication_bundle(
            campaign_root,
            publication_stage,
            bundle_name=bundle_name,
            include_videos=False,
            repository_url=repository_url,
            release_tag=release_tag,
            doi=doi,
            overwrite=True,
        )
        next_descriptor = _safe_publication_descriptor(
            bundle_name=bundle_name,
            result=exported,
            publication_relative_dir=publication_relative_dir,
        )
        if next_descriptor == descriptor:
            verify_publication_bundle_preflight(exported.bundle_dir)
            _assert_no_private_absolute_paths(exported.bundle_dir)
            return next_descriptor, exported.bundle_dir, exported.archive_path
        descriptor = next_descriptor
    raise PublicationPreflightError("publication bundle descriptor did not stabilize")


def build_derived_release(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    producer_root: Path,
    acceptance_root: Path,
    source_repository_root: Path,
    manifest_path: Path,
    output_root: Path,
    derived_name: str,
    validator_repository_root: Path,
    expected_validator_commit: str,
    publication_name: str | None = None,
    preserved_receipt_source: Path | None = None,
    expected_source_sha: str = FROZEN_SOURCE_SHA,
) -> dict[str, Any]:
    """Run the complete derived validation/build/promotion workflow.

    ``producer_root`` is normally the checksum-bearing retrieval, while
    ``acceptance_root`` is the untouched original producer campaign (possibly
    on the execution host).  Both may be the same path for a local fixture.
    """
    producer_root = _assert_safe_directory(producer_root, label="producer retrieval root")
    acceptance_root = _assert_safe_directory(acceptance_root, label="accepted campaign root")
    source_repository_root = _assert_safe_directory(
        source_repository_root, label="source repository root"
    )
    validator_repository_root = _assert_safe_directory(
        validator_repository_root, label="validator repository root"
    )
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = source_repository_root / manifest_path
    manifest_path = Path(manifest_path.absolute())
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DerivedReleaseError("release manifest is missing or unsafe")
    if any(component.is_symlink() for component in manifest_path.parents):
        raise DerivedReleaseError("release manifest contains a symlink component")
    try:
        manifest_path.resolve().relative_to(source_repository_root.resolve())
    except ValueError as exc:
        raise DerivedReleaseError(
            "release manifest must be supplied from the frozen source repository"
        ) from exc
    output_root = output_root.resolve()
    final_campaign = output_root / derived_name
    publication_name = publication_name or f"{derived_name}_publication"
    final_publication = final_campaign / publication_name
    if final_campaign.exists():
        raise DerivedReleaseError("derived campaign or publication target already exists")
    if not re.fullmatch(r"[0-9a-f]{40}", expected_source_sha):
        raise DerivedReleaseError("expected source SHA must be a 40-character lowercase SHA")

    _assert_frozen_source_repository(source_repository_root, expected_source_sha)
    validator = _validator_provenance(
        validator_repository_root,
        expected_commit=expected_validator_commit,
    )
    helper_validator_path = Path(release_acceptance_module.__file__).resolve()
    if sha256_file(helper_validator_path) != validator["file_sha256"]:
        raise DerivedReleaseError(
            "running validator implementation differs from the reviewed validator checkout"
        )
    producer_evidence = verify_producer_artifacts(
        producer_root,
        expected_receipt_sha256=(
            EXPECTED_REFRESHED_PRODUCER_RECEIPT_SHA256
            if preserved_receipt_source is not None
            else EXPECTED_PRODUCER_RECEIPT_SHA256
        ),
        preserved_receipt_source=preserved_receipt_source,
    )
    accepted_evidence = _verify_campaign_file_map(
        acceptance_root,
        expected_sums_sha256=EXPECTED_PRODUCER_SUMS_SHA256,
        expected_result_sha256=EXPECTED_REJECTED_RESULT_SHA256,
        expected_file_count=EXPECTED_PRODUCER_FILE_COUNT,
    )
    _assert_equal_file_maps(accepted_evidence, producer_evidence)

    with _source_repository_binding(
        source_repository_root,
        validator_root=validator_repository_root,
    ):
        manifest = load_release_manifest(manifest_path)
        _assert_manifest_paths_from_source(manifest, source_repository_root)
        config_path = manifest.canonical_campaign_config_path
        campaign_config = load_campaign_config(config_path)
        manifest_validation = validate_release_manifest(
            manifest,
            campaign_config=campaign_config,
        )
        if manifest_validation.get("status") != "valid":
            raise DerivedReleaseError(
                "release manifest validation failed: "
                + "; ".join(str(item) for item in manifest_validation.get("problems", []))
            )
        acceptance = validate_full_benchmark_release_acceptance(
            acceptance_root,
            manifest=manifest,
            campaign_config=campaign_config,
            source_repository_root=source_repository_root,
        )
    if acceptance.get("status") != "valid":
        raise DerivedReleaseError("corrected full-release acceptance rejected preserved rows")
    source_commits = acceptance.get("source_commits")
    if source_commits != [expected_source_sha]:
        raise DerivedReleaseError("acceptance did not bind every row to the frozen source SHA")

    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(tempfile.mkdtemp(prefix=f".{derived_name}.staging-", dir=output_root))
    staging_campaign = staging_parent / derived_name
    staging_publication = staging_parent / publication_name
    try:
        copy_file_map = dict(producer_evidence["file_map"])
        copy_file_map[PRODUCER_RECEIPT_NAME] = {
            "bytes": len(producer_evidence["_current_receipt_bytes"]),
            "sha256": producer_evidence["artifact_verification_receipt_sha256"],
        }
        _copy_producer_projection(
            producer_root,
            staging_root=staging_campaign,
            source_root=source_repository_root,
            expected_file_map=copy_file_map,
            current_receipt_bytes=producer_evidence["_current_receipt_bytes"],
            preserved_receipt_bytes=producer_evidence["_preserved_receipt_bytes"],
        )
        derived_checksums = _write_derived_checksums(staging_campaign)
        _write_derivation_receipt(
            staging_campaign,
            producer_evidence=producer_evidence,
            accepted_evidence=accepted_evidence,
            manifest_validation=manifest_validation,
            acceptance=acceptance,
            validator=validator,
            source_sha=expected_source_sha,
            derived_checksums=derived_checksums,
        )
        # The receipt itself is part of the derived checksum inventory.  Rewrite
        # once after it is emitted, without ever signing SHA256SUMS itself.
        _write_derived_checksums(staging_campaign)

        resolved_manifest = _read_json(
            staging_campaign / "release" / "release_manifest.resolved.json"
        )
        release_tag = str(resolved_manifest.get("release_tag", "")).strip()
        provenance = resolved_manifest.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        doi = str(
            provenance.get("version_doi")
            or provenance.get("doi")
            or resolved_manifest.get("doi", "")
        ).strip()
        repository_url = str(provenance.get("repository_url", "")).strip()
        if not release_tag or not doi or not repository_url:
            raise DerivedReleaseError("resolved release manifest lacks publication identity")
        with _source_repository_binding(
            source_repository_root,
            validator_root=validator_repository_root,
        ):
            descriptor, bundle_dir, archive_path = _export_stabilised_bundle(
                staging_campaign,
                publication_stage=staging_publication,
                bundle_name=f"{derived_name}_publication_bundle",
                acceptance=acceptance,
                producer_result=_read_json(
                    staging_campaign / "release" / "producer_release_result.rejected.json"
                ),
                release_tag=release_tag,
                doi=doi,
                repository_url=repository_url,
                publication_relative_dir=publication_name,
            )
        _assert_no_private_absolute_paths(staging_campaign)
        _assert_no_private_absolute_paths(staging_publication)
        _write_custody_receipt(
            staging_publication,
            bundle_name=bundle_dir.name,
            archive_path=archive_path,
            bundle_dir=bundle_dir,
            source_sha=expected_source_sha,
        )
        # Detect producer mutation before promotion.  This is deliberately the
        # same strict manifest check used before copying.
        producer_after = verify_producer_artifacts(
            producer_root,
            expected_receipt_sha256=(
                EXPECTED_REFRESHED_PRODUCER_RECEIPT_SHA256
                if preserved_receipt_source is not None
                else EXPECTED_PRODUCER_RECEIPT_SHA256
            ),
            preserved_receipt_source=preserved_receipt_source,
        )
        if producer_after["files"] != producer_evidence["files"]:
            raise DerivedReleaseError("producer tree changed during derived build")
        if producer_after.get("file_map") != producer_evidence.get("file_map"):
            raise DerivedReleaseError("producer file map changed during derived build")
        for key in (
            "artifact_verification_receipt_sha256",
            "preserved_artifact_verification_receipt_sha256",
            "artifact_receipt_refresh",
        ):
            if producer_after.get(key) != producer_evidence.get(key):
                raise DerivedReleaseError("producer receipt changed during derived build")
        accepted_after = _verify_campaign_file_map(
            acceptance_root,
            expected_sums_sha256=EXPECTED_PRODUCER_SUMS_SHA256,
            expected_result_sha256=EXPECTED_REJECTED_RESULT_SHA256,
            expected_file_count=EXPECTED_PRODUCER_FILE_COUNT,
        )
        _assert_equal_file_maps(accepted_after, producer_after)
        # Put campaign and publication artifacts under one staged directory,
        # then perform one directory rename.  This prevents a caller from
        # observing a campaign without its bundle or archive.
        staged_publication_target = staging_campaign / publication_name
        os.replace(staging_publication, staged_publication_target)
        # The final inventory is written only after the publication subtree,
        # archive, and custody receipt are in place.  SHA256SUMS excludes
        # itself, so this is complete without a self-digest cycle.
        _write_derived_checksums(staging_campaign)
        _assert_no_private_absolute_paths(staging_campaign)
        os.replace(staging_campaign, final_campaign)
        staging_parent.rmdir()
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise

    return {
        "status": "published_to_staging",
        "campaign_root": final_campaign,
        "publication_root": final_publication,
        "publication_bundle": final_publication / bundle_dir.name,
        "publication_archive": final_publication / archive_path.name,
        "publication_descriptor": descriptor,
        "producer": producer_evidence,
        "accepted": accepted_evidence,
        "manifest_validation": manifest_validation,
        "acceptance": acceptance,
        "validator": validator,
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        help="Untouched producer campaign root used for acceptance (defaults to producer root).",
    )
    parser.add_argument("--source-repository-root", type=Path, required=True)
    parser.add_argument("--validator-repository-root", type=Path, required=True)
    parser.add_argument("--expected-validator-commit", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--derived-name", required=True)
    parser.add_argument("--publication-name")
    parser.add_argument(
        "--preserved-receipt",
        type=Path,
        help="Single-member gzip containing the immutable pre-refresh receipt.",
    )
    parser.add_argument("--source-sha", default=FROZEN_SOURCE_SHA)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed derived release command."""
    args = _build_parser().parse_args(argv)
    acceptance_root = args.acceptance_root or args.producer_root
    try:
        result = build_derived_release(
            producer_root=args.producer_root,
            acceptance_root=acceptance_root,
            source_repository_root=args.source_repository_root,
            validator_repository_root=args.validator_repository_root,
            expected_validator_commit=args.expected_validator_commit,
            manifest_path=args.manifest,
            output_root=args.output_root,
            derived_name=args.derived_name,
            publication_name=args.publication_name,
            preserved_receipt_source=args.preserved_receipt,
            expected_source_sha=args.source_sha,
        )
    except (DerivedReleaseError, OSError, ValueError, PublicationPreflightError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, indent=2))
        return 2
    # Do not print absolute paths: publication output can live on a private
    # filesystem and the machine-readable receipt is the durable handoff.
    safe = {
        "status": result["status"],
        "publication_descriptor": result["publication_descriptor"],
        "producer": {
            key: value
            for key, value in result["producer"].items()
            if key != "files" and not key.startswith("_")
        },
        "acceptance": result["acceptance"],
        "validator": result["validator"],
    }
    print(json.dumps(safe, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
