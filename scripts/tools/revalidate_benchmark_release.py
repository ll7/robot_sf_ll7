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
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from robot_sf.benchmark.camera_ready import _run_state as camera_run_state_module
from robot_sf.benchmark.camera_ready._artifacts import _write_snqi_diagnostics_artifacts
from robot_sf.benchmark.camera_ready_campaign import write_campaign_report
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_erratum import (
    ErratumContract,
    ReleaseErratumError,
    build_erratum_receipt,
    load_erratum_contract,
    snapshot_campaign,
    snapshot_predecessor_archive,
)
from robot_sf.benchmark.release_protocol import (
    load_release_campaign_config,
    load_release_manifest,
    validate_release_manifest,
)
from robot_sf.benchmark.snqi.campaign_contract import SNQI_FAILED_WARN_RECOMMENDATION

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
ERRATUM_RECEIPT_RELATIVE = "provenance/benchmark_release_erratum.json"
ERRATUM_METADATA_RELATIVE = "release/zenodo_metadata.erratum.json"
PUBLICATION_CUSTODY_NAME = "publication_custody.json"
SNQI_ADVISORY_BOUNDARY = (
    "SNQI calibration failed under warn and remains advisory only; it is not a planner-ranking "
    "authority for this release."
)
GOAL_TIMEOUT_BOUNDARY_NOTE = (
    "Producer recorded goal_reached and timeout in the same terminal row without "
    "reached_goal_step; exact event ordering is unavailable. This row is excluded from "
    "goal/timeout timing-boundary interpretation; status, termination, outcome, and metrics "
    "remain unchanged."
)
EXPECTED_GOAL_TIMEOUT_BOUNDARY_ROWS = frozenset(
    {
        (
            "guarded_ppo__differential_drive",
            "francis2023_parallel_traffic--132--2bf83ad03db6559e",
        )
    }
)
EXPECTED_RELEASE_EPISODE_ROWS = 20_160
EXPECTED_RELEASE_ARMS = 14
EXPECTED_SOURCE_CAMPAIGN_RELATIVE = Path(
    "output/benchmarks/camera_ready/issue7742_release_full-s30-h600-b1d5ab6de708-v1_20260825"
)


@dataclass(frozen=True)
class RecoveryContract:
    """Immutable identities required to derive one rejected producer campaign safely."""

    source_sha: str
    producer_sums_sha256: str
    producer_receipt_sha256: str
    rejected_result_sha256: str
    producer_file_count: int
    source_campaign_relative: Path
    episode_rows: int
    arms: int
    goal_timeout_boundary_rows: frozenset[tuple[str, str]]
    refreshed_producer_receipt_sha256: str | None = None


DEFAULT_RECOVERY_CONTRACT = RecoveryContract(
    source_sha=FROZEN_SOURCE_SHA,
    producer_sums_sha256=EXPECTED_PRODUCER_SUMS_SHA256,
    producer_receipt_sha256=EXPECTED_PRODUCER_RECEIPT_SHA256,
    refreshed_producer_receipt_sha256=EXPECTED_REFRESHED_PRODUCER_RECEIPT_SHA256,
    rejected_result_sha256=EXPECTED_REJECTED_RESULT_SHA256,
    producer_file_count=EXPECTED_PRODUCER_FILE_COUNT,
    source_campaign_relative=EXPECTED_SOURCE_CAMPAIGN_RELATIVE,
    episode_rows=EXPECTED_RELEASE_EPISODE_ROWS,
    arms=EXPECTED_RELEASE_ARMS,
    goal_timeout_boundary_rows=EXPECTED_GOAL_TIMEOUT_BOUNDARY_ROWS,
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PRIVATE_ABSOLUTE_PATH_RE = re.compile(
    rb"(?<![A-Za-z0-9_:/])/(?:home|root|Users|tmp|scratch|dev/shm|gpfs|lustre|mnt|work|worktrees|workspace|var)/[^\s\"'<>`]+"
)
_TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".cfg",
    ".conf",
    ".csv",
    ".html",
    ".htm",
    ".ini",
    ".json",
    ".jsonl",
    ".log",
    ".lock",
    ".md",
    ".py",
    ".rst",
    ".sbatch",
    ".sha256",
    ".sh",
    ".sum",
    ".svg",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_TEXT_NAMES = {"CITATION", "LICENSE", "Makefile", "README", "SHA256SUMS"}
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
    ".db",
    ".sqlite",
    ".sqlite3",
    ".parquet",
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


_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_safe_component(value: str, *, label: str) -> str:
    """Require a generated output name to be one safe path component."""
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise DerivedReleaseError(f"{label} must be a non-empty safe path component")
    candidate = Path(value)
    if candidate.is_absolute() or len(candidate.parts) != 1 or candidate.name != value:
        raise DerivedReleaseError(f"{label} must be a single relative path component")
    if not _SAFE_COMPONENT_RE.fullmatch(value):
        raise DerivedReleaseError(f"{label} contains unsafe characters")
    return value


def load_recovery_contract(path: Path) -> RecoveryContract:  # noqa: C901
    """Load a checksum-pinned recovery identity without release-specific code edits."""
    payload = _read_json(_assert_safe_file(path, label="recovery contract"))
    if payload.get("schema_version") != "benchmark-derived-release-recovery.v1":
        raise DerivedReleaseError("recovery contract schema_version is unsupported")

    def required_text(name: str) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            raise DerivedReleaseError(f"recovery contract {name} must be a non-empty string")
        return value.strip()

    source_sha = required_text("source_sha").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise DerivedReleaseError("recovery contract source_sha must be a full Git SHA")
    digests: dict[str, str] = {}
    for name in (
        "producer_sums_sha256",
        "producer_receipt_sha256",
        "rejected_result_sha256",
    ):
        digest = required_text(name).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise DerivedReleaseError(f"recovery contract {name} must be a SHA-256")
        digests[name] = digest
    refreshed = payload.get("refreshed_producer_receipt_sha256")
    if refreshed is not None:
        if not isinstance(refreshed, str) or not _SHA256_RE.fullmatch(refreshed.lower()):
            raise DerivedReleaseError(
                "recovery contract refreshed_producer_receipt_sha256 must be a SHA-256"
            )
        refreshed = refreshed.lower()

    source_campaign_relative = Path(required_text("source_campaign_relative"))
    if (
        source_campaign_relative.is_absolute()
        or source_campaign_relative == Path(".")
        or ".." in source_campaign_relative.parts
    ):
        raise DerivedReleaseError("recovery contract source_campaign_relative must be safe")

    integers: dict[str, int] = {}
    for name in ("producer_file_count", "episode_rows", "arms"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise DerivedReleaseError(f"recovery contract {name} must be a positive integer")
        integers[name] = value

    raw_boundaries = payload.get("goal_timeout_boundary_rows", [])
    if not isinstance(raw_boundaries, list):
        raise DerivedReleaseError("recovery contract goal_timeout_boundary_rows must be a list")
    boundaries: set[tuple[str, str]] = set()
    for index, row in enumerate(raw_boundaries):
        if not isinstance(row, Mapping):
            raise DerivedReleaseError(f"recovery contract boundary row {index} must be an object")
        arm = row.get("arm")
        episode_id = row.get("episode_id")
        if not isinstance(arm, str) or not arm or not isinstance(episode_id, str) or not episode_id:
            raise DerivedReleaseError(f"recovery contract boundary row {index} is malformed")
        identity = (arm, episode_id)
        if identity in boundaries:
            raise DerivedReleaseError("recovery contract duplicates a boundary-row identity")
        boundaries.add(identity)

    return RecoveryContract(
        source_sha=source_sha,
        producer_sums_sha256=digests["producer_sums_sha256"],
        producer_receipt_sha256=digests["producer_receipt_sha256"],
        refreshed_producer_receipt_sha256=refreshed,
        rejected_result_sha256=digests["rejected_result_sha256"],
        producer_file_count=integers["producer_file_count"],
        source_campaign_relative=source_campaign_relative,
        episode_rows=integers["episode_rows"],
        arms=integers["arms"],
        goal_timeout_boundary_rows=frozenset(boundaries),
    )


def _expected_current_producer_receipt_sha256(
    recovery_contract: RecoveryContract,
    *,
    preserved_receipt_source: Path | None,
) -> str:
    """Select the exact current receipt digest for producer verification.

    A preserved pre-refresh receipt proves the historical bytes. Its paired current receipt must
    be pinned separately so a generalized contract can never inherit the job-14890 default.
    """
    if preserved_receipt_source is None:
        return recovery_contract.producer_receipt_sha256
    refreshed = recovery_contract.refreshed_producer_receipt_sha256
    if refreshed is None:
        raise DerivedReleaseError(
            "recovery contract must pin refreshed_producer_receipt_sha256 when a preserved "
            "receipt is supplied"
        )
    return refreshed


def _assert_safe_directory(path: Path, *, label: str) -> Path:
    """Require a real directory with no symlink path component."""
    lexical = Path(path).absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise DerivedReleaseError(f"{label} contains a symlink component")
    if not lexical.is_dir():
        raise DerivedReleaseError(f"{label} is not a directory")
    return lexical


def _assert_safe_file(path: Path, *, label: str) -> Path:
    """Require a regular file whose parent path contains no symlink."""
    lexical = Path(path).absolute()
    if any(component.is_symlink() for component in lexical.parents) or lexical.is_symlink():
        raise DerivedReleaseError(f"{label} contains a symlink component")
    if not lexical.is_file():
        raise DerivedReleaseError(f"{label} is missing or not a regular file")
    return lexical


def _is_text_capable(path: Path) -> bool:
    """Classify only explicitly supported text/binary formats.

    Unknown suffixes are rejected instead of being decoded as text.  This is
    important for opaque scientific payloads such as SQLite and Parquet: a
    publication projection must never rewrite a binary file merely because it
    happened to contain bytes that look like UTF-8.
    """
    if (
        path.name in _TEXT_NAMES
        or path.name.endswith("SHA256SUMS")
        or path.suffix.lower() in _TEXT_SUFFIXES
    ):
        return True
    if path.suffix.lower() in _BINARY_SUFFIXES:
        return False
    raise DerivedReleaseError(f"unsupported publication file type: {path.name}")


@contextlib.contextmanager
def _source_repository_binding(source_root: Path, *, validator_root: Path | None = None):
    """Bind repository-aware protocol/export helpers to the frozen checkout."""
    previous_protocol_root = release_protocol_module.get_repository_root
    previous_publication_root = artifact_publication_module.get_repository_root
    previous_config_root = camera_config_module.get_repository_root
    previous_run_state_root = camera_run_state_module.get_repository_root
    previous_acceptance_root = release_acceptance_module.get_repository_root
    release_protocol_module.get_repository_root = lambda: source_root
    artifact_publication_module.get_repository_root = lambda: source_root
    camera_config_module.get_repository_root = lambda: source_root
    camera_run_state_module.get_repository_root = lambda: source_root
    release_acceptance_module.get_repository_root = lambda: validator_root or source_root
    try:
        yield
    finally:
        release_protocol_module.get_repository_root = previous_protocol_root
        artifact_publication_module.get_repository_root = previous_publication_root
        camera_config_module.get_repository_root = previous_config_root
        camera_run_state_module.get_repository_root = previous_run_state_root
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
    path = _assert_safe_file(path, label="preserved receipt source")
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
    return path.absolute().relative_to(root.absolute()).as_posix()


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
    receipt_path = root / PRODUCER_RECEIPT_NAME
    if receipt_path.is_symlink():
        raise DerivedReleaseError("accepted campaign receipt is symlinked")
    optional_receipt = {PRODUCER_RECEIPT_NAME} if receipt_path.is_file() else set()
    actual = {_relative_path(root, path) for path in all_files}
    expected_actual = listed | {PRODUCER_SUMS_NAME} | optional_receipt
    if actual != expected_actual:
        raise DerivedReleaseError("accepted campaign file inventory does not match SHA256SUMS")
    map_paths = sorted(listed | {PRODUCER_SUMS_NAME})
    separate_receipt = None
    separate_receipt_payload = None
    if receipt_path.is_file():
        separate_receipt_payload = _read_json(receipt_path)
        _validate_receipt(
            separate_receipt_payload,
            label="accepted campaign artifact-verification-receipt",
            root=root,
            sums_digest=sums_digest,
            entries=entries,
            expected_file_count=expected_file_count,
        )
        separate_receipt = {
            "path": PRODUCER_RECEIPT_NAME,
            "bytes": receipt_path.stat().st_size,
            "sha256": sha256_file(receipt_path).lower(),
            "policy": "accepted-root-receipt-is-separate-from-admitted-row-map",
        }
    return {
        "status": "verified",
        "listed_file_count": len(entries),
        "total_file_count": len(actual),
        "sha256sums_sha256": sums_digest,
        "rejected_release_result_sha256": result_digest,
        "file_map": _build_file_map(root, map_paths),
        "separate_receipt": separate_receipt,
        "_separate_receipt_payload": separate_receipt_payload,
    }


def _verify_acceptance_campaign_subset(
    campaign_root: Path,
    *,
    producer_evidence: Mapping[str, Any],
    expected_rejected_result_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind the untouched acceptance tree to the checksummed producer superset.

    The execution campaign is the only root whose provenance sidecars retain
    their canonical ``raw_artifact`` paths.  Collection adds scheduler and
    custody files to a separate checksum-bearing tree, so requiring the two
    directory inventories to be identical would reject the real layout.  Every
    file read by the acceptance validator must instead be present byte-for-byte
    in the admitted producer inventory.
    """
    root = _assert_safe_directory(campaign_root, label="accepted campaign root")
    producer_map = producer_evidence.get("file_map")
    if not isinstance(producer_map, Mapping) or not producer_map:
        raise DerivedReleaseError("producer file map is unavailable for acceptance binding")
    relative_paths = [_relative_path(root, path) for path in _all_tree_files(root)]
    if not relative_paths:
        raise DerivedReleaseError("accepted campaign tree is empty")
    file_map = _build_file_map(root, relative_paths)
    for relative, identity in file_map.items():
        if producer_map.get(relative) != identity:
            raise DerivedReleaseError(
                f"accepted campaign file is not bound to producer inventory: {relative}"
            )
    result_identity = file_map.get(REJECTED_RESULT_RELATIVE)
    if not isinstance(result_identity, Mapping):
        raise DerivedReleaseError("accepted campaign release_result is missing")
    expected_rejected_result_sha256 = (
        expected_rejected_result_sha256 or EXPECTED_REJECTED_RESULT_SHA256
    )
    if result_identity.get("sha256") != expected_rejected_result_sha256:
        raise DerivedReleaseError("accepted campaign release_result is not the admitted rejection")
    return {
        "status": "verified",
        "file_count": len(file_map),
        "producer_file_count": len(producer_map),
        "producer_extra_file_count": len(set(producer_map) - set(file_map)),
        "rejected_release_result_sha256": result_identity["sha256"],
        "binding_policy": "acceptance-tree-is-byte-identical-subset-of-checksummed-producer",
        "file_map": file_map,
    }


def _assert_accepted_receipt_relation(
    accepted: Mapping[str, Any], retrieved: Mapping[str, Any]
) -> None:
    """Bind an optional accepted receipt to the retrieved receipt semantics."""
    accepted_receipt = accepted.get("_separate_receipt_payload")
    if accepted_receipt is None:
        return
    current_bytes = retrieved.get("_current_receipt_bytes")
    if not isinstance(current_bytes, bytes):
        raise DerivedReleaseError("retrieved artifact receipt is unavailable for relation binding")
    current_receipt = _read_json_bytes(current_bytes, label="retrieved artifact receipt")
    differences = _json_difference_paths(accepted_receipt, current_receipt)
    if differences not in ([], ["verified_at"]):
        raise DerivedReleaseError(
            "accepted artifact receipt differs from retrieved receipt outside verified_at"
        )
    if differences == ["verified_at"]:
        _validate_preserved_refresh(accepted_receipt, current_receipt)


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
    root = _assert_safe_directory(producer_root, label="producer retrieval root")
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


def _sanitise_bytes(
    data: bytes,
    *,
    source_root: Path,
    producer_root: Path,
    validator_root: Path | None = None,
) -> bytes:
    """Remove private absolute paths while retaining portable relative paths."""
    result = data
    # These two roots are safe to turn into portable paths: sidecar validation
    # accepts paths relative to either the source checkout or campaign root.
    for root in sorted(
        (
            root.resolve()
            for root in (source_root, producer_root, validator_root)
            if root is not None
        ),
        key=lambda value: -len(str(value)),
    ):
        root_bytes = str(root).encode("utf-8")
        result = result.replace(root_bytes + b"/", b"")
        result = result.replace(root_bytes, b".")
    return _PRIVATE_ABSOLUTE_PATH_RE.sub(b"<external-path>", result)


def _sanitise_tree_paths(
    root: Path,
    *,
    source_root: Path,
    producer_root: Path,
    validator_root: Path | None = None,
) -> None:
    """Rewrite private path strings in copied text artifacts using streaming I/O."""
    for path in _all_tree_files(root):
        if not _is_text_capable(path):
            continue
        temporary = path.with_name(path.name + ".path-sanitising")
        changed = False
        try:
            with path.open("rb") as source_handle, temporary.open("wb") as target_handle:
                # Text artifacts are structured by lines.  Processing complete
                # lines prevents an absolute path from being split and corrupted
                # at an arbitrary byte-chunk boundary.  JSONL rows can be large,
                # but remain bounded per episode and do not require whole-file
                # buffering.
                for body in source_handle:
                    replaced = _sanitise_bytes(
                        body,
                        source_root=source_root,
                        producer_root=producer_root,
                        validator_root=validator_root,
                    )
                    changed = changed or replaced != body
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
        # Classify every file, including opaque scientific binaries.  Binary
        # payloads are never rewritten, but a known private path marker in one
        # is still a publication leak and must fail closed.
        _is_text_capable(path)
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
    validator_root: Path | None = None,
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
    _sanitise_tree_paths(
        staging_root,
        source_root=source_root,
        producer_root=producer_root,
        validator_root=validator_root,
    )
    _assert_no_private_absolute_paths(staging_root)


def _is_unresolved_goal_timeout_boundary(record: Mapping[str, Any]) -> bool:
    """Return whether a row needs an explicit non-fabricated boundary annotation."""
    ledger = record.get("event_ledger")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    exact_events = ledger.get("exact_events")
    exact_events = exact_events if isinstance(exact_events, Mapping) else {}
    note = record.get("goal_timeout_boundary_note")
    return (
        exact_events.get("goal_reached") is True
        and exact_events.get("timeout") is True
        and record.get("reached_goal_step") is None
        and not (isinstance(note, str) and note.strip())
    )


def _validate_goal_timeout_annotation_candidate(record: Mapping[str, Any]) -> None:
    """Require the reviewed terminal semantics before adding a publication-only note."""
    outcome = record.get("outcome")
    outcome = outcome if isinstance(outcome, Mapping) else {}
    metrics = record.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    if (
        record.get("status") != "success"
        or record.get("termination_reason") != "success"
        or outcome.get("route_complete") is not True
        or outcome.get("timeout_event") is not True
        or metrics.get("success") not in (1, 1.0, True)
    ):
        raise DerivedReleaseError(
            "reviewed goal+timeout row does not retain successful terminal semantics"
        )


def _rebind_sidecar_rows(
    sidecar: Mapping[str, Any],
    *,
    allowed_paths: set[str],
    relative_path: str,
    expected_count: int,
) -> None:
    """Validate and rebind every row-level raw-artifact path."""
    sidecar_rows = sidecar.get("rows")
    if not isinstance(sidecar_rows, list) or len(sidecar_rows) != expected_count:
        raise DerivedReleaseError("episode provenance sidecar row count is stale")
    for sidecar_row in sidecar_rows:
        if (
            not isinstance(sidecar_row, dict)
            or sidecar_row.get("raw_artifact") not in allowed_paths
        ):
            raise DerivedReleaseError("episode sidecar row path is not source/projection bound")
        sidecar_row["raw_artifact"] = relative_path


def _rebind_one_publication_sidecar(
    campaign_root: Path,
    *,
    episodes_path: Path,
    source_file_map: Mapping[str, Mapping[str, Any]],
    boundary_files: Mapping[str, Mapping[str, Any]],
    source_campaign_relative: Path = EXPECTED_SOURCE_CAMPAIGN_RELATIVE,
) -> dict[str, Any]:
    """Rebind one source-validated sidecar to its derived episode path."""
    relative_path = episodes_path.relative_to(campaign_root).as_posix()
    source_relative_path = (source_campaign_relative / relative_path).as_posix()
    allowed_paths = {relative_path, source_relative_path}
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    sidecar_relative = sidecar_path.relative_to(campaign_root).as_posix()
    pre_rebind_sidecar_sha256 = sha256_file(sidecar_path).lower()
    producer_sidecar_entry = source_file_map.get(sidecar_relative)
    producer_sidecar_sha256 = (
        str(producer_sidecar_entry.get("sha256", "")).lower()
        if isinstance(producer_sidecar_entry, Mapping)
        else ""
    )
    if not _SHA256_RE.fullmatch(producer_sidecar_sha256):
        raise DerivedReleaseError("producer map omits a valid episode sidecar digest")
    sidecar = _read_json(sidecar_path)
    raw_artifacts = sidecar.get("raw_artifacts")
    if not isinstance(raw_artifacts, list):
        raise DerivedReleaseError("episode provenance sidecar raw_artifacts is malformed")
    episode_artifacts = [
        item
        for item in raw_artifacts
        if isinstance(item, dict) and item.get("kind") == "episodes_jsonl"
    ]
    if len(episode_artifacts) != 1 or episode_artifacts[0].get("path") not in allowed_paths:
        raise DerivedReleaseError("episode sidecar raw path is not source/projection bound")
    source_entry = source_file_map.get(relative_path)
    source_digest = (
        str(source_entry.get("sha256", "")).lower() if isinstance(source_entry, Mapping) else ""
    )
    if not _SHA256_RE.fullmatch(source_digest):
        raise DerivedReleaseError("producer map omits a valid episode artifact digest")
    if str(episode_artifacts[0].get("sha256", "")).lower() != source_digest:
        raise DerivedReleaseError("episode provenance sidecar disagrees with producer map")
    current_digest = sha256_file(episodes_path).lower()
    episode_artifacts[0]["path"] = relative_path
    episode_artifacts[0]["sha256"] = current_digest
    episode_rows = sum(
        1 for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    _rebind_sidecar_rows(
        sidecar,
        allowed_paths=allowed_paths,
        relative_path=relative_path,
        expected_count=episode_rows,
    )
    derived_artifacts = sidecar.get("derived_artifacts")
    if not isinstance(derived_artifacts, list):
        raise DerivedReleaseError("episode provenance sidecar derived_artifacts is malformed")
    projection_record: dict[str, Any] = {
        "kind": "publication_projection",
        "path": relative_path,
        "producer_sha256": source_digest,
        "sha256": current_digest,
        "path_binding": "projection_relative",
        "producer_sidecar_sha256": producer_sidecar_sha256,
        "scientific_execution_changed": False,
        "simulation_rerun": False,
    }
    boundary = boundary_files.get(relative_path)
    if isinstance(boundary, Mapping):
        if boundary.get("mode") == "excluded_without_row_mutation":
            projection_record["goal_timeout_boundary_exclusion"] = {
                "source_sha256": boundary.get("source_sha256"),
                "derived_sha256": boundary.get("derived_sha256"),
                "excluded_row_count": len(boundary.get("episode_ids", [])),
                "episode_ids": boundary.get("episode_ids", []),
                "raw_episode_rows_unchanged": True,
                "timing_evidence_fabricated": False,
                "scientific_fields_changed": False,
            }
        else:
            projection_record["goal_timeout_boundary_annotation"] = {
                "pre_annotation_projection_sha256": boundary.get("source_sha256"),
                "annotation_count": len(boundary.get("episode_ids", [])),
                "episode_ids": boundary.get("episode_ids", []),
                "timing_evidence_fabricated": False,
                "scientific_fields_changed": False,
            }
    derived_artifacts.append(projection_record)
    _write_json(sidecar_path, sidecar)
    return {
        "path": sidecar_relative,
        "producer_sidecar_sha256": producer_sidecar_sha256,
        "pre_rebind_projection_sidecar_sha256": pre_rebind_sidecar_sha256,
        "derived_sha256": sha256_file(sidecar_path).lower(),
        "episodes_path": relative_path,
        "producer_episodes_sha256": source_digest,
        "episodes_sha256": current_digest,
        "row_count": episode_rows,
    }


def _rebind_publication_sidecars(
    campaign_root: Path,
    *,
    source_file_map: Mapping[str, Mapping[str, Any]],
    boundary_reconciliation: Mapping[str, Any],
    expected_arm_count: int = EXPECTED_RELEASE_ARMS,
    expected_row_count: int = EXPECTED_RELEASE_EPISODE_ROWS,
    source_campaign_relative: Path = EXPECTED_SOURCE_CAMPAIGN_RELATIVE,
) -> dict[str, Any]:
    """Rebind every copied sidecar to the derived tree after strict source-path validation."""
    raw_boundary_files = boundary_reconciliation.get("files")
    raw_boundary_files = raw_boundary_files if isinstance(raw_boundary_files, list) else []
    boundary_files = {
        str(item.get("path")): item for item in raw_boundary_files if isinstance(item, Mapping)
    }
    files = [
        _rebind_one_publication_sidecar(
            campaign_root,
            episodes_path=episodes_path,
            source_file_map=source_file_map,
            boundary_files=boundary_files,
            source_campaign_relative=source_campaign_relative,
        )
        for episodes_path in sorted(campaign_root.glob("runs/*/episodes.jsonl"))
    ]
    row_count = sum(int(item["row_count"]) for item in files)
    if len(files) != expected_arm_count or row_count != expected_row_count:
        raise DerivedReleaseError("projection sidecar rebind did not cover the full release matrix")
    return {
        "status": "projection_relative",
        "arm_count": len(files),
        "row_count": row_count,
        "files": files,
    }


def _find_unresolved_goal_timeout_rows(
    campaign_root: Path,
) -> tuple[dict[Path, list[tuple[int, str, dict[str, Any]]]], set[tuple[str, str]]]:
    """Plan boundary annotations without changing any artifact bytes."""
    planned: dict[Path, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    observed: set[tuple[str, str]] = set()
    for episodes_path in sorted(campaign_root.glob("runs/*/episodes.jsonl")):
        try:
            lines = episodes_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise DerivedReleaseError("could not read publication episode projection") from exc
        for line_index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DerivedReleaseError(
                    "publication episode projection contains invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise DerivedReleaseError("publication episode projection row is not an object")
            if not _is_unresolved_goal_timeout_boundary(record):
                continue
            episode_id = record.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise DerivedReleaseError("goal+timeout row is missing an episode identity")
            observed.add((episodes_path.parent.name, episode_id.strip()))
            planned[episodes_path].append((line_index, line, record))
    return planned, observed


def _annotate_publication_goal_timeout_boundaries(
    campaign_root: Path,
    *,
    expected_rows: set[tuple[str, str]] | frozenset[tuple[str, str]] = (
        EXPECTED_GOAL_TIMEOUT_BOUNDARY_ROWS
    ),
) -> dict[str, Any]:
    """Annotate only the reviewed frozen-row ambiguity without inventing event timing."""
    planned, observed = _find_unresolved_goal_timeout_rows(campaign_root)
    if observed != set(expected_rows):
        raise DerivedReleaseError(
            "unresolved goal+timeout rows differ from reviewed boundary-row set"
        )

    file_evidence: list[dict[str, Any]] = []
    for episodes_path, edits in sorted(planned.items(), key=lambda item: item[0].as_posix()):
        lines = episodes_path.read_text(encoding="utf-8").splitlines()
        source_sha256 = sha256_file(episodes_path).lower()
        episode_ids: list[str] = []
        for line_index, original_line, record in edits:
            if lines[line_index] != original_line:
                raise DerivedReleaseError("episode projection changed during annotation planning")
            _validate_goal_timeout_annotation_candidate(record)
            record["goal_timeout_boundary_note"] = GOAL_TIMEOUT_BOUNDARY_NOTE
            lines[line_index] = json.dumps(record, sort_keys=True, ensure_ascii=False)
            episode_ids.append(str(record["episode_id"]))
        temporary = episodes_path.with_name(f".{episodes_path.name}.boundary-annotation")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, episodes_path)
        file_evidence.append(
            {
                "path": episodes_path.relative_to(campaign_root).as_posix(),
                "source_sha256": source_sha256,
                "derived_sha256": sha256_file(episodes_path).lower(),
                "episode_ids": sorted(episode_ids),
            }
        )

    run_meta_path = campaign_root / "run_meta.json"
    run_meta = _read_json(run_meta_path)
    run_meta["goal_timeout_boundary"] = {
        "annotated_rows": len(observed),
        "unresolved_rows": 0,
        "timing_evidence_fabricated": False,
        "policy": (
            "Frozen rows lacking a reached-goal step carry an explicit note and are excluded "
            "from timing-boundary interpretation; no event timing is inferred."
        ),
    }
    _write_json(run_meta_path, run_meta)
    return {
        "status": "annotated",
        "annotated_row_count": len(observed),
        "rows": [{"arm": arm, "episode_id": episode_id} for arm, episode_id in sorted(observed)],
        "files": file_evidence,
        "timing_evidence_fabricated": False,
        "scientific_fields_changed": False,
    }


def _record_publication_goal_timeout_boundaries_without_row_mutation(
    campaign_root: Path,
    *,
    expected_rows: set[tuple[str, str]] | frozenset[tuple[str, str]] = (
        EXPECTED_GOAL_TIMEOUT_BOUNDARY_ROWS
    ),
) -> dict[str, Any]:
    """Record reviewed timing exclusions while preserving every episode byte.

    A derived-metadata erratum compares complete scientific rows with its
    immutable predecessor.  It therefore cannot add even an explanatory field
    to an episode row.  This path validates the same reviewed identities and
    terminal semantics as the ordinary recovery annotator, then records the
    interpretation boundary only in signed run metadata and provenance.

    Returns:
        Machine-readable reconciliation evidence with unchanged file digests.
    """
    planned, observed = _find_unresolved_goal_timeout_rows(campaign_root)
    if observed != set(expected_rows):
        raise DerivedReleaseError(
            "unresolved goal+timeout rows differ from reviewed boundary-row set"
        )

    file_evidence: list[dict[str, Any]] = []
    for episodes_path, rows in sorted(planned.items(), key=lambda item: item[0].as_posix()):
        source_sha256 = sha256_file(episodes_path).lower()
        episode_ids: list[str] = []
        for _line_index, _original_line, record in rows:
            _validate_goal_timeout_annotation_candidate(record)
            episode_ids.append(str(record["episode_id"]))
        if sha256_file(episodes_path).lower() != source_sha256:
            raise DerivedReleaseError("episode projection changed during boundary review")
        file_evidence.append(
            {
                "path": episodes_path.relative_to(campaign_root).as_posix(),
                "source_sha256": source_sha256,
                "derived_sha256": source_sha256,
                "episode_ids": sorted(episode_ids),
                "mode": "excluded_without_row_mutation",
            }
        )

    excluded_rows = [{"arm": arm, "episode_id": episode_id} for arm, episode_id in sorted(observed)]
    run_meta_path = campaign_root / "run_meta.json"
    run_meta = _read_json(run_meta_path)
    run_meta["goal_timeout_boundary"] = {
        "status": "excluded_from_timing_interpretation",
        "excluded_row_count": len(observed),
        "excluded_rows": excluded_rows,
        "raw_episode_rows_unchanged": True,
        "timing_evidence_fabricated": False,
        "note": GOAL_TIMEOUT_BOUNDARY_NOTE,
        "policy": (
            "Frozen rows lacking a reached-goal step are excluded from timing-boundary "
            "interpretation through publication provenance; no episode field or event timing "
            "is changed or inferred."
        ),
    }
    _write_json(run_meta_path, run_meta)
    return {
        "status": "recorded_without_row_mutation",
        "annotated_row_count": 0,
        "excluded_row_count": len(observed),
        "rows": excluded_rows,
        "files": file_evidence,
        "raw_episode_rows_unchanged": True,
        "timing_evidence_fabricated": False,
        "scientific_fields_changed": False,
    }


def _require_reviewed_snqi_mismatch(
    campaign_root: Path,
    *,
    expected_row_count: int,
    expected_arm_count: int,
) -> dict[str, Any]:
    """Require that stale ordering is the only SNQI publication inconsistency."""
    before = artifact_publication_module._check_snqi_field_consistency(campaign_root)
    counts = before.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    expected_violation = (
        "per-episode metrics.snqi arm ordering disagrees with "
        "snqi_diagnostics.json planner_ordering"
    )
    counts_match = (
        counts.get("rows") == expected_row_count
        and counts.get("episode_field_present") == expected_row_count
        and counts.get("snqi_field_mismatches") == 0
        and counts.get("arms") == expected_arm_count
    )
    if (
        before.get("checked") is not True
        or not counts_match
        or before.get("violations") != [expected_violation]
    ):
        raise DerivedReleaseError(
            "SNQI diagnostics have drift beyond the reviewed ordering mismatch"
        )
    return before


def _stored_snqi_ordering(campaign_root: Path) -> list[dict[str, Any]]:
    """Build planner ordering from already formula-verified stored episode fields."""
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for episodes_path in sorted(campaign_root.glob("runs/*/episodes.jsonl")):
        planner_key, separator, kinematics = episodes_path.parent.name.partition("__")
        if not separator or not planner_key or not kinematics:
            raise DerivedReleaseError("SNQI episode arm directory is malformed")
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            metrics = record.get("metrics") if isinstance(record, Mapping) else None
            raw_snqi = metrics.get("snqi") if isinstance(metrics, Mapping) else None
            if (
                isinstance(raw_snqi, bool)
                or not isinstance(raw_snqi, (int, float))
                or not math.isfinite(float(raw_snqi))
            ):
                raise DerivedReleaseError("publication row lacks a finite stored SNQI value")
            grouped[(planner_key, kinematics)].append(float(raw_snqi))
    ordering = [
        {
            "planner_key": planner_key,
            "kinematics": kinematics,
            "episode_count": len(values),
            "mean_snqi": sum(values) / len(values),
        }
        for (planner_key, kinematics), values in grouped.items()
    ]
    ordering.sort(
        key=lambda row: (
            -float(row["mean_snqi"]),
            str(row["planner_key"]),
            str(row["kinematics"]),
        )
    )
    for rank, row in enumerate(ordering, start=1):
        row["rank"] = rank
    return ordering


def _reconcile_publication_snqi_diagnostics(
    campaign_root: Path,
    *,
    expected_row_count: int = EXPECTED_RELEASE_EPISODE_ROWS,
    expected_arm_count: int = EXPECTED_RELEASE_ARMS,
) -> dict[str, Any]:
    """Reconcile stale diagnostics ordering while keeping failed SNQI calibration advisory."""
    diagnostics_path = campaign_root / "reports" / "snqi_diagnostics.json"
    producer_diagnostics_sha256 = sha256_file(diagnostics_path).lower()
    diagnostics = _read_json(diagnostics_path)
    if (
        diagnostics.get("contract_enabled") is not True
        or diagnostics.get("contract_enforcement") != "warn"
        or diagnostics.get("contract_status") != "fail"
    ):
        raise DerivedReleaseError("SNQI recovery requires the reviewed failed-under-warn contract")

    before = _require_reviewed_snqi_mismatch(
        campaign_root,
        expected_row_count=expected_row_count,
        expected_arm_count=expected_arm_count,
    )
    diagnostics["planner_ordering"] = _stored_snqi_ordering(campaign_root)
    diagnostics["planner_ordering_basis"] = "stored_metrics.snqi"
    diagnostics["score_basis_reconciliation"] = {
        "status": "reconciled_from_verified_stored_fields",
        "canonical_formula": "robot_sf.benchmark.metrics.snqi",
        "verified_episode_rows": expected_row_count,
        "stored_field_disposition": (
            "retained: every stored metrics.snqi value matched the pinned curvature-aware basis"
        ),
        "planner_ordering_disposition": (
            "recomputed from the frozen per-episode fields without changing episode metrics"
        ),
        "integrity": dict(before.get("integrity", {})),
    }
    diagnostics["release_claim_boundary"] = {
        "status": "advisory_only",
        "ranking_authority": False,
        "ranking_claims_admitted": False,
        "calibration_status": "fail",
        "enforcement": "warn",
        "claim_boundary": SNQI_ADVISORY_BOUNDARY,
    }
    positioning = diagnostics.get("positioning")
    positioning = dict(positioning) if isinstance(positioning, Mapping) else {}
    positioning["planner_ordering_informative"] = False
    positioning["recommendation"] = SNQI_FAILED_WARN_RECOMMENDATION
    caveats = positioning.get("caveats")
    caveats = list(caveats) if isinstance(caveats, list) else []
    if SNQI_ADVISORY_BOUNDARY not in caveats:
        caveats.append(SNQI_ADVISORY_BOUNDARY)
    positioning["caveats"] = caveats
    diagnostics["positioning"] = positioning
    _write_snqi_diagnostics_artifacts(campaign_root / "reports", diagnostics)

    after = artifact_publication_module._check_snqi_field_consistency(campaign_root)
    if after.get("violation_count") != 0 or after.get("violations"):
        raise DerivedReleaseError("reconciled SNQI diagnostics remain publication-inconsistent")
    return {
        "status": "reconciled_advisory_only",
        "verified_episode_rows": expected_row_count,
        "arm_count": expected_arm_count,
        "producer_diagnostics_sha256": producer_diagnostics_sha256,
        "derived_diagnostics_sha256": sha256_file(diagnostics_path).lower(),
        "post_reconciliation_violation_count": 0,
        "ranking_authority": False,
        "claim_boundary": SNQI_ADVISORY_BOUNDARY,
    }


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


def _assert_distinct_validator_checkout(validator_root: Path, source_root: Path) -> None:
    """Reject the frozen source or helper checkout as the reviewed validator."""
    validator = Path(validator_root).resolve()
    source = Path(source_root).resolve()
    helper = Path(__file__).resolve().parents[2]
    if validator in {source, helper}:
        raise DerivedReleaseError(
            "validator checkout must be distinct from frozen source and helper checkout"
        )


def _assert_exact_orchestration_checkout(repo_root: Path, expected_commit: str) -> None:
    """Bind the erratum workflow implementation to one clean exact checkout."""
    root = _assert_safe_directory(repo_root, label="orchestration repository root")
    helper = Path(__file__).resolve().parents[2]
    if root.resolve() != helper:
        raise DerivedReleaseError(
            "erratum orchestration root differs from the executing tooling checkout"
        )
    actual = _git_value(["rev-parse", "HEAD^{commit}"], cwd=root).lower()
    if actual != expected_commit:
        raise DerivedReleaseError("erratum orchestration checkout is not the contracted commit")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise DerivedReleaseError("erratum orchestration checkout is not clean")


def _run_exact_validator(
    *,
    validator_root: Path,
    source_root: Path,
    acceptance_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Execute the reviewed validator from its clean checkout in isolation."""
    _assert_distinct_validator_checkout(validator_root, source_root)
    script = r"""
import json
import sys
from pathlib import Path

validator_root = Path(sys.argv[1])
source_root = Path(sys.argv[2])
acceptance_root = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
# Resolve source-relative assets from the frozen checkout while keeping the
# reviewed validator implementation ahead of the working directory on import.
sys.path.insert(0, str(validator_root))

from robot_sf.benchmark import release_protocol as protocol
from robot_sf.benchmark import release_acceptance as acceptance_module
from robot_sf.benchmark.camera_ready import _config as config_module
from robot_sf.benchmark.camera_ready import _run_state as run_state_module

expected_validator_file = validator_root / "robot_sf/benchmark/release_acceptance.py"
if Path(acceptance_module.__file__).resolve() != expected_validator_file.resolve():
    raise RuntimeError("imported validator is not from the reviewed checkout")

protocol.get_repository_root = lambda: source_root
config_module.get_repository_root = lambda: source_root
run_state_module.get_repository_root = lambda: source_root
manifest = protocol.load_release_manifest(manifest_path)
campaign_config = protocol.load_release_campaign_config(
    manifest,
    repository_root=source_root,
)
result = acceptance_module.validate_full_benchmark_release_acceptance(
    acceptance_root,
    manifest=manifest,
    campaign_config=campaign_config,
    source_repository_root=source_root,
)
print(json.dumps(result, sort_keys=True, default=str))
"""
    environment = os.environ.copy()
    # A validator checkout must win over any editable helper checkout in the
    # caller's environment. Source-owned registry paths must likewise resolve
    # from the frozen execution checkout rather than the validator checkout.
    environment["PYTHONPATH"] = str(validator_root)
    source_map_registry = _assert_safe_file(
        source_root / "maps/registry.yaml",
        label="frozen source map registry",
    )
    environment["ROBOT_SF_MAP_REGISTRY"] = str(source_map_registry)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(validator_root),
            str(source_root),
            str(acceptance_root),
            str(manifest_path),
        ],
        cwd=source_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        raise DerivedReleaseError("exact reviewed validator execution failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DerivedReleaseError("exact reviewed validator returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise DerivedReleaseError("exact reviewed validator returned a non-object result")
    return payload


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
            if not isinstance(path, Path):
                raise DerivedReleaseError(f"manifest {field} contains an external asset")
            lexical_pin = Path(path.absolute())
            if (
                any(component.is_symlink() for component in lexical_pin.parents)
                or lexical_pin.is_symlink()
            ):
                raise DerivedReleaseError(f"manifest {field} contains a symlink")
            if not lexical_pin.resolve().is_relative_to(source_root.resolve()):
                raise DerivedReleaseError(f"manifest {field} contains an external asset")
    seed_policy = getattr(manifest, "seed_policy", {})
    raw_seed_path = seed_policy.get("seed_sets_path") if isinstance(seed_policy, Mapping) else None
    if raw_seed_path:
        candidate = Path(str(raw_seed_path))
        # ``load_release_manifest`` resolves stress-contract assets relative to
        # the manifest directory, not the repository root.  Mirror that exact
        # rule here so a helper checkout cannot satisfy a relative seed path by
        # accident.
        candidate = candidate if candidate.is_absolute() else manifest.path.parent / candidate
        if any(component.is_symlink() for component in candidate.parents) or candidate.is_symlink():
            raise DerivedReleaseError("manifest seed_sets_path contains a symlink")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(source_root.resolve()):
            raise DerivedReleaseError("manifest seed_sets_path is not bound to frozen source")


def _assert_publication_inputs_from_manifest(
    manifest: Any,
    resolved_manifest: Mapping[str, Any],
    source_root: Path,
) -> dict[str, dict[str, str]]:
    """Bind publication metadata and SNQI assets to the loaded manifest exactly."""
    source_root = _assert_safe_directory(source_root, label="source repository root")
    expected_fields = {
        "citation": ("citation_path", None),
        "zenodo_metadata": ("metadata_path", "metadata_sha256"),
        "snqi_weights": ("snqi_weights_path", "snqi_weights_sha256"),
        "snqi_baseline": ("snqi_baseline_path", "snqi_baseline_sha256"),
    }
    provenance = resolved_manifest.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    metrics = resolved_manifest.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    raw_by_role = {
        "citation": provenance.get("citation_path"),
        "zenodo_metadata": provenance.get("zenodo_metadata_path")
        or provenance.get("metadata_path"),
        "snqi_weights": metrics.get("snqi_weights_path"),
        "snqi_baseline": metrics.get("snqi_baseline_path"),
    }
    bound: dict[str, dict[str, str]] = {}
    for role, (path_field, digest_field) in expected_fields.items():
        expected_path = getattr(manifest, path_field, None)
        if not isinstance(expected_path, Path):
            raise DerivedReleaseError(f"manifest {path_field} is missing")
        expected_path = Path(expected_path.absolute())
        raw_path = raw_by_role[role]
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise DerivedReleaseError(f"publication manifest omits canonical {role} path")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = source_root / candidate
        if any(component.is_symlink() for component in candidate.parents) or candidate.is_symlink():
            raise DerivedReleaseError(f"publication {role} path contains a symlink")
        candidate = Path(candidate.absolute())
        if candidate != expected_path:
            raise DerivedReleaseError(
                f"publication {role} path differs from canonical loaded manifest asset"
            )
        if not candidate.is_file() or not candidate.absolute().is_relative_to(source_root):
            raise DerivedReleaseError(f"publication {role} path is outside frozen source")
        digest = sha256_file(candidate).lower()
        expected_digest = getattr(manifest, digest_field, None) if digest_field else None
        if expected_digest is not None and digest != str(expected_digest).lower():
            raise DerivedReleaseError(f"publication {role} hash differs from manifest pin")
        bound[role] = {
            "path": candidate.relative_to(source_root).as_posix(),
            "sha256": digest,
        }
    return bound


def _assert_frozen_source_repository(
    source_root: Path, expected_source_sha: str = FROZEN_SOURCE_SHA
) -> None:
    """Require the supplied source checkout to be clean and the fixed release SHA."""
    _assert_safe_directory(source_root, label="source repository root")
    actual = _git_value(["rev-parse", "HEAD^{commit}"], cwd=source_root).lower()
    if actual != expected_source_sha:
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
    prefix = _validate_safe_component(publication_relative_dir, label="publication_relative_dir")
    _validate_safe_component(bundle_name, label="bundle_name")
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
    derivation = result.get("derivation")
    derivation = dict(derivation) if isinstance(derivation, Mapping) else {}
    derivation.update(
        {
            "mode": "preserved_rows_corrected_validator",
            "producer_release_result_sha256": sha256_file(
                campaign_root / "release" / "producer_release_result.rejected.json"
            ),
            "producer_release_status": producer_result.get("release_status"),
            "snqi_claim_boundary": SNQI_ADVISORY_BOUNDARY,
        }
    )
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
            "publication_preflight_violations": [],
            "publication_bundle": dict(publication_descriptor),
            "release_acceptance": dict(acceptance),
            "full_release_acceptance": dict(acceptance),
            "derivation": derivation,
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
    release_tag = str(campaign.get("release_tag") or "").strip()
    doi = str(campaign.get("doi") or "").strip()
    repository_url = str(campaign.get("repository_url") or "").strip().rstrip("/")
    archive_relative = publication_descriptor.get("archive_path")
    archive_name = (
        Path(archive_relative).name
        if isinstance(archive_relative, str) and archive_relative.strip()
        else ""
    )
    if release_tag and repository_url:
        campaign["release_url"] = f"{repository_url}/releases/tag/{release_tag}"
        if archive_name:
            campaign["release_asset_url"] = (
                f"{repository_url}/releases/download/{release_tag}/{archive_name}"
            )
    if doi:
        campaign["doi_url"] = f"https://doi.org/{doi}"
    summary["campaign"] = campaign
    artifacts = summary.get("artifacts")
    artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    for key in ("release_url", "release_asset_url", "doi_url"):
        value = campaign.get(key)
        if isinstance(value, str) and value:
            artifacts[key] = value
    summary["artifacts"] = artifacts
    summary["full_release_acceptance"] = dict(acceptance)
    summary["publication_bundle"] = dict(publication_descriptor)
    _write_json(summary_path, summary)
    write_campaign_report(campaign_root / "reports" / "campaign_report.md", summary)


def _write_derivation_receipt(  # noqa: PLR0913
    campaign_root: Path,
    *,
    producer_evidence: Mapping[str, Any],
    accepted_evidence: Mapping[str, Any],
    manifest_validation: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    validator: Mapping[str, str],
    source_sha: str,
    derived_checksums: Mapping[str, str],
    publication_inputs: Mapping[str, Any],
    publication_reconciliation: Mapping[str, Any],
    projection_acceptance: Mapping[str, Any],
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
            "accepted_separate_receipt": accepted_evidence.get("separate_receipt"),
            "retrieved_separate_receipt": {
                "path": PRODUCER_RECEIPT_NAME,
                "bytes": len(producer_evidence["_current_receipt_bytes"]),
                "sha256": producer_evidence["artifact_verification_receipt_sha256"],
            },
            "separate_receipt_policy": (
                "verification receipts are explicitly recorded separately and excluded from "
                "the acceptance-tree subset and retrieved producer file maps"
            ),
        },
        "manifest_validation": dict(manifest_validation),
        "publication_inputs": dict(publication_inputs),
        "validator": dict(validator),
        "acceptance": dict(projection_acceptance),
        "source_acceptance": dict(acceptance),
        "projection_acceptance": dict(projection_acceptance),
        "publication_reconciliation": dict(publication_reconciliation),
        "derived_projection": {
            "campaign_root": "(this directory)",
            "sha256sums_file": PRODUCER_SUMS_NAME,
            "sha256sums_entry_count": len(derived_checksums),
            "sha256sums_entry_count_scope": "pre_publication_projection",
            "sha256sums_excludes_only": PRODUCER_SUMS_NAME,
            "final_inventory_rewritten_after_publication": True,
            "nested_bundle_root_inventory": "omitted_to_avoid_checksum_cycle; bundle checksums.sha256 is complete",
            "final_inventory_includes": [
                "publication subtree",
                "publication archive",
                PUBLICATION_CUSTODY_NAME,
            ],
            "producer_rejected_result": "release/producer_release_result.rejected.json",
        },
        "snqi": {
            "status": "advisory",
            "ranking_authority": False,
            "claim_boundary": SNQI_ADVISORY_BOUNDARY,
        },
        "credentials": "not_recorded",
    }
    _write_json(campaign_root / DERIVATION_RECEIPT_RELATIVE, receipt)


def _rewrite_publication_provenance(
    payload: Mapping[str, Any] | None, *, contract: ErratumContract
) -> dict[str, Any]:
    """Rewrite one current-publication provenance mapping to successor coordinates."""
    provenance = dict(payload) if isinstance(payload, Mapping) else {}
    execution_metadata_path = provenance.get("metadata_path")
    execution_metadata_sha256 = provenance.get("metadata_sha256")
    if isinstance(execution_metadata_path, str) and execution_metadata_path:
        provenance.setdefault("scientific_execution_metadata_path", execution_metadata_path)
    if isinstance(execution_metadata_sha256, str) and execution_metadata_sha256:
        provenance.setdefault("scientific_execution_metadata_sha256", execution_metadata_sha256)
    provenance.update(
        {
            "release_tag": contract.successor_github_release_tag,
            "release_id": contract.successor_github_release_tag,
            "doi": contract.successor_version_doi,
            "version_doi": contract.successor_version_doi,
            "concept_doi": contract.concept_doi,
            "version_record_id": contract.successor_version_doi.rsplit(".", 1)[-1],
            "concept_record_id": contract.concept_doi.rsplit(".", 1)[-1],
            "bundle_zenodo_metadata_path": ERRATUM_METADATA_RELATIVE,
            "metadata_path": ERRATUM_METADATA_RELATIVE,
            "metadata_sha256": contract.metadata_sha256,
            "scientific_source_sha": contract.source_sha,
            "erratum_builder_sha": contract.builder_sha,
            "erratum_validator_sha": contract.validator_sha,
            "erratum_orchestration_sha": contract.orchestration_sha,
        }
    )
    return provenance


def _rewrite_embedded_publication_identity(
    payload: Mapping[str, Any], *, contract: ErratumContract
) -> dict[str, Any]:
    """Rewrite only publication coordinates in one copied metadata object.

    Returns:
        A new mapping retaining execution provenance and scientific-manifest hashes.
    """
    updated = dict(payload)
    execution_manifest_path = updated.get("manifest_path")
    if isinstance(execution_manifest_path, str) and execution_manifest_path:
        updated["scientific_execution_manifest_path"] = execution_manifest_path
        updated["manifest_path"] = "release/release_manifest.resolved.json"
    updated.update(
        {
            "release_tag": contract.successor_github_release_tag,
            "release_id": contract.successor_github_release_tag,
            "doi": contract.successor_version_doi,
            "concept_doi": contract.concept_doi,
            "version_doi": contract.successor_version_doi,
            "metadata_path": ERRATUM_METADATA_RELATIVE,
            "metadata_sha256": contract.metadata_sha256,
        }
    )
    if isinstance(updated.get("provenance"), Mapping):
        updated["provenance"] = _rewrite_publication_provenance(
            updated["provenance"], contract=contract
        )
    return updated


def _rewrite_resolved_manifest_publication_identity(
    payload: Mapping[str, Any], *, contract: ErratumContract
) -> dict[str, Any]:
    """Return a successor resolved manifest while retaining scientific identity."""
    resolved = dict(payload)
    resolved["release_tag"] = contract.successor_github_release_tag
    resolved["release_id"] = contract.successor_github_release_tag
    if "doi" in resolved:
        resolved["doi"] = contract.successor_version_doi
    if "version_doi" in resolved:
        resolved["version_doi"] = contract.successor_version_doi
    if "concept_doi" in resolved:
        resolved["concept_doi"] = contract.concept_doi
    provenance = _rewrite_publication_provenance(resolved.get("provenance"), contract=contract)
    resolved["provenance"] = provenance
    resolved["publication"] = {
        "channel": provenance.get("publication_channel", "direct_zenodo_benchmark_dataset"),
        "concept_doi": contract.concept_doi,
        "version_doi": contract.successor_version_doi,
        "concept_record_id": contract.concept_doi.rsplit(".", 1)[-1],
        "version_record_id": contract.successor_version_doi.rsplit(".", 1)[-1],
        "bundle_metadata_path": ERRATUM_METADATA_RELATIVE,
        "metadata_sha256": contract.metadata_sha256,
        "correction_scope": "derived_publication_metadata_only",
        "predecessor_version_doi": contract.predecessor_version_doi,
    }
    resolved["erratum"] = {
        "correction_id": contract.correction_id,
        "correction_scope": "derived_publication_metadata_only",
        "predecessor_version_doi": contract.predecessor_version_doi,
        "predecessor_github_release_tag": contract.predecessor_github_release_tag,
        "scientific_source_unchanged": True,
        "simulation_rerun": False,
    }
    return resolved


def _assert_successor_identity_fields(
    payload: Mapping[str, Any], *, contract: ErratumContract, label: str
) -> None:
    """Require one current-publication mapping to name only successor coordinates."""
    tag_keys = ("release_tag", "release_id", "benchmark_release_tag", "benchmark_release_id")
    tag_values = [payload[key] for key in tag_keys if key in payload]
    doi_values = [payload[key] for key in ("version_doi", "doi") if key in payload]
    concept_values = [payload["concept_doi"]] if "concept_doi" in payload else []
    provenance = payload.get("provenance")
    if isinstance(provenance, Mapping):
        tag_values.extend(provenance[key] for key in tag_keys if key in provenance)
        doi_values.extend(provenance[key] for key in ("version_doi", "doi") if key in provenance)
        if "concept_doi" in provenance:
            concept_values.append(provenance["concept_doi"])
    if not tag_values or any(
        value != contract.successor_github_release_tag for value in tag_values
    ):
        raise DerivedReleaseError(f"{label} does not name the successor release tag")
    if not doi_values or any(value != contract.successor_version_doi for value in doi_values):
        raise DerivedReleaseError(f"{label} does not name the successor version DOI")
    if not concept_values or any(value != contract.concept_doi for value in concept_values):
        raise DerivedReleaseError(f"{label} does not name the successor concept DOI")


def _assert_resolved_erratum_provenance(provenance: Any, *, contract: ErratumContract) -> None:
    """Require one resolved manifest's provenance to use successor coordinates."""
    if not isinstance(provenance, Mapping):
        raise DerivedReleaseError("resolved manifest lacks successor provenance")
    if provenance.get("version_doi") != contract.successor_version_doi:
        raise DerivedReleaseError("resolved provenance does not name the successor version DOI")
    if provenance.get("concept_doi") != contract.concept_doi:
        raise DerivedReleaseError("resolved provenance does not name the successor concept DOI")
    if provenance.get("metadata_path") != ERRATUM_METADATA_RELATIVE:
        raise DerivedReleaseError("resolved provenance metadata path is not bundle-local")
    if provenance.get("metadata_sha256") != contract.metadata_sha256:
        raise DerivedReleaseError("resolved provenance metadata digest is stale")


def _assert_erratum_manifest_identities(campaign_root: Path, *, contract: ErratumContract) -> None:
    """Check the resolved and copied manifest identity separations."""
    resolved = _read_json(campaign_root / "release" / "release_manifest.resolved.json")
    _assert_successor_identity_fields(resolved, contract=contract, label="resolved manifest")
    _assert_resolved_erratum_provenance(resolved.get("provenance"), contract=contract)
    publication = resolved.get("publication")
    if not isinstance(publication, Mapping) or any(
        publication.get(key) != value
        for key, value in {
            "concept_doi": contract.concept_doi,
            "version_doi": contract.successor_version_doi,
            "predecessor_version_doi": contract.predecessor_version_doi,
            "bundle_metadata_path": ERRATUM_METADATA_RELATIVE,
            "metadata_sha256": contract.metadata_sha256,
            "correction_scope": "derived_publication_metadata_only",
        }.items()
    ):
        raise DerivedReleaseError("resolved manifest publication identity is stale")

    for relative in ("campaign_manifest.json", "manifest.json", "run_meta.json"):
        path = campaign_root / relative
        if not path.is_file():
            continue
        payload = _read_json(path)
        benchmark_release = payload.get("benchmark_release")
        execution_release = payload.get("scientific_execution_benchmark_release")
        if not isinstance(benchmark_release, Mapping) or not isinstance(execution_release, Mapping):
            raise DerivedReleaseError(f"{relative} does not separate publication and execution")
        _assert_successor_identity_fields(
            benchmark_release, contract=contract, label=f"{relative}.benchmark_release"
        )
        if execution_release.get("release_tag") != contract.predecessor_github_release_tag:
            raise DerivedReleaseError(f"{relative} lost the predecessor execution tag")
        if execution_release.get("version_doi") != contract.predecessor_version_doi:
            raise DerivedReleaseError(f"{relative} lost the predecessor execution DOI")
        if execution_release.get("concept_doi") != contract.concept_doi:
            raise DerivedReleaseError(f"{relative} lost the predecessor concept DOI")


def _assert_erratum_release_result(campaign_root: Path, *, contract: ErratumContract) -> None:
    """Check the successor result verdict and retained execution identity."""
    result = _read_json(campaign_root / "release" / "release_result.json")
    _assert_successor_identity_fields(result, contract=contract, label="release result")
    for current_key, execution_key in (
        ("benchmark_release", "scientific_execution_benchmark_release"),
        ("resolved_manifest", "scientific_execution_resolved_manifest"),
    ):
        current = result.get(current_key)
        execution = result.get(execution_key)
        if not isinstance(current, Mapping) or not isinstance(execution, Mapping):
            raise DerivedReleaseError(f"release result does not separate {current_key} identity")
        _assert_successor_identity_fields(
            current, contract=contract, label=f"release result {current_key}"
        )
        if execution.get("release_tag") != contract.predecessor_github_release_tag:
            raise DerivedReleaseError(f"release result {execution_key} lost predecessor identity")
        if execution.get("version_doi") != contract.predecessor_version_doi:
            raise DerivedReleaseError(f"release result {execution_key} lost predecessor DOI")
        if execution.get("concept_doi") != contract.concept_doi:
            raise DerivedReleaseError(
                f"release result {execution_key} lost predecessor concept DOI"
            )
    derivation = result.get("derivation")
    expected_derivation = {
        "builder_sha": contract.builder_sha,
        "validator_sha": contract.validator_sha,
        "orchestration_sha": contract.orchestration_sha,
        "scientific_source_sha": contract.source_sha,
        "simulation_rerun": False,
        "correction_id": contract.correction_id,
        "predecessor_version_doi": contract.predecessor_version_doi,
    }
    if not isinstance(derivation, Mapping) or any(
        derivation.get(key) != value for key, value in expected_derivation.items()
    ):
        raise DerivedReleaseError("erratum release result derivation identity is stale")
    if (
        result.get("publication_preflight_status") != "pass"
        or result.get("publication_preflight_violations") != []
        or result.get("release_status") != "ok"
        or result.get("ranking_claims_admitted") is not False
    ):
        raise DerivedReleaseError("erratum release result has a contradictory verdict")


def _assert_erratum_summary(campaign_root: Path, *, contract: ErratumContract) -> None:
    """Check all current campaign-summary DOI, tag, and asset coordinates."""
    summary = _read_json(campaign_root / "reports" / "campaign_summary.json")
    summary_release = summary.get("benchmark_release")
    campaign = summary.get("campaign")
    artifacts = summary.get("artifacts")
    if not all(isinstance(value, Mapping) for value in (summary_release, campaign, artifacts)):
        raise DerivedReleaseError("campaign summary lacks successor publication identity")
    _assert_successor_identity_fields(
        summary_release,
        contract=contract,
        label="campaign summary benchmark_release",
    )
    expected_release_url = (
        "https://github.com/ll7/robot_sf_ll7/releases/tag/" + contract.successor_github_release_tag
    )
    expected_doi_url = f"https://doi.org/{contract.successor_version_doi}"
    if campaign.get("release_tag") != contract.successor_github_release_tag:
        raise DerivedReleaseError("campaign summary campaign tag is stale")
    if campaign.get("doi") != contract.successor_version_doi:
        raise DerivedReleaseError("campaign summary campaign DOI is stale")
    execution = campaign.get("scientific_execution_release_identity")
    if (
        not isinstance(execution, Mapping)
        or execution.get("release_tag") != contract.predecessor_github_release_tag
        or execution.get("doi") != contract.predecessor_version_doi
        or execution.get("source_sha") != contract.source_sha
    ):
        raise DerivedReleaseError("campaign summary lost predecessor execution identity")
    for label, current in (("campaign", campaign), ("artifacts", artifacts)):
        if current.get("release_url") != expected_release_url:
            raise DerivedReleaseError(f"campaign summary {label} release URL is stale")
        if current.get("doi_url") != expected_doi_url:
            raise DerivedReleaseError(f"campaign summary {label} DOI URL is stale")
        asset_url = current.get("release_asset_url")
        expected_asset_prefix = (
            "https://github.com/ll7/robot_sf_ll7/releases/download/"
            + contract.successor_github_release_tag
            + "/"
        )
        if not isinstance(asset_url, str) or not asset_url.startswith(expected_asset_prefix):
            raise DerivedReleaseError(f"campaign summary {label} release asset URL is stale")


def _assert_erratum_publication_identity(campaign_root: Path, *, contract: ErratumContract) -> None:
    """Fail closed on contradictory current-publication coordinates."""
    _assert_erratum_manifest_identities(campaign_root, contract=contract)
    _assert_erratum_release_result(campaign_root, contract=contract)
    _assert_erratum_summary(campaign_root, contract=contract)
    metadata = campaign_root / ERRATUM_METADATA_RELATIVE
    if not metadata.is_file() or sha256_file(metadata).lower() != contract.metadata_sha256:
        raise DerivedReleaseError("erratum metadata copy is missing or stale")


def _rewrite_erratum_campaign_summary(campaign_root: Path, *, contract: ErratumContract) -> None:
    """Rewrite current publication coordinates while retaining execution identity."""
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = _read_json(summary_path)
    summary_release = summary.get("benchmark_release")
    if isinstance(summary_release, Mapping):
        summary.setdefault("scientific_execution_benchmark_release", dict(summary_release))
        summary["benchmark_release"] = _rewrite_embedded_publication_identity(
            summary_release, contract=contract
        )
    campaign = summary.get("campaign")
    campaign = dict(campaign) if isinstance(campaign, Mapping) else {}
    campaign.setdefault(
        "scientific_execution_release_identity",
        {
            "release_tag": campaign.get("release_tag") or campaign.get("benchmark_release_tag"),
            "doi": campaign.get("doi"),
            "manifest_path": campaign.get("benchmark_release_manifest_path"),
            "invoked_command": campaign.get("invoked_command"),
            "source_sha": contract.source_sha,
        },
    )
    campaign.update(
        {
            "release_tag": contract.successor_github_release_tag,
            "release_id": contract.successor_github_release_tag,
            "benchmark_release_tag": contract.successor_github_release_tag,
            "benchmark_release_id": contract.successor_github_release_tag,
            "benchmark_release_manifest_path": "release/release_manifest.resolved.json",
            "doi": contract.successor_version_doi,
            "version_doi": contract.successor_version_doi,
            "concept_doi": contract.concept_doi,
            "doi_url": f"https://doi.org/{contract.successor_version_doi}",
            "release_url": (
                "https://github.com/ll7/robot_sf_ll7/releases/tag/"
                + contract.successor_github_release_tag
            ),
        }
    )
    existing_asset_url = campaign.get("release_asset_url")
    if isinstance(existing_asset_url, str) and existing_asset_url.rsplit("/", 1)[-1]:
        campaign["release_asset_url"] = (
            "https://github.com/ll7/robot_sf_ll7/releases/download/"
            + contract.successor_github_release_tag
            + "/"
            + existing_asset_url.rsplit("/", 1)[-1]
        )
    summary["campaign"] = campaign
    artifacts = summary.get("artifacts")
    artifacts = dict(artifacts) if isinstance(artifacts, Mapping) else {}
    artifacts.update(
        {
            "doi_url": f"https://doi.org/{contract.successor_version_doi}",
            "release_url": (
                "https://github.com/ll7/robot_sf_ll7/releases/tag/"
                + contract.successor_github_release_tag
            ),
        }
    )
    if isinstance(campaign.get("release_asset_url"), str):
        artifacts["release_asset_url"] = campaign["release_asset_url"]
    summary["artifacts"] = artifacts
    summary["publication_erratum"] = {
        "correction_id": contract.correction_id,
        "predecessor_version_doi": contract.predecessor_version_doi,
        "predecessor_github_release_tag": contract.predecessor_github_release_tag,
        "builder_sha": contract.builder_sha,
        "validator_sha": contract.validator_sha,
        "orchestration_sha": contract.orchestration_sha,
        "source_sha": contract.source_sha,
        "simulation_rerun": False,
    }
    _write_json(summary_path, summary)


def _apply_erratum_publication_identity(
    campaign_root: Path,
    *,
    contract: ErratumContract,
) -> dict[str, Any]:
    """Materialize a successor identity without changing execution provenance.

    Returns:
        The rewritten resolved manifest used for publication export.
    """
    metadata_target = campaign_root / ERRATUM_METADATA_RELATIVE
    metadata_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract.metadata_path, metadata_target)
    if sha256_file(metadata_target).lower() != contract.metadata_sha256:
        raise DerivedReleaseError("copied erratum metadata does not match its contract")

    resolved_path = campaign_root / "release" / "release_manifest.resolved.json"
    resolved = _rewrite_resolved_manifest_publication_identity(
        _read_json(resolved_path), contract=contract
    )
    _write_json(resolved_path, resolved)

    for relative in ("campaign_manifest.json", "manifest.json", "run_meta.json"):
        path = campaign_root / relative
        if not path.is_file():
            continue
        payload = _read_json(path)
        payload["release_tag"] = contract.successor_github_release_tag
        payload["release_id"] = contract.successor_github_release_tag
        payload["doi"] = contract.successor_version_doi
        payload["version_doi"] = contract.successor_version_doi
        payload["concept_doi"] = contract.concept_doi
        benchmark_release = payload.get("benchmark_release")
        if isinstance(benchmark_release, Mapping):
            payload.setdefault("scientific_execution_benchmark_release", dict(benchmark_release))
            payload["benchmark_release"] = _rewrite_embedded_publication_identity(
                benchmark_release, contract=contract
            )
        payload["publication_erratum"] = {
            "correction_id": contract.correction_id,
            "predecessor_version_doi": contract.predecessor_version_doi,
            "source_sha": contract.source_sha,
            "builder_sha": contract.builder_sha,
            "validator_sha": contract.validator_sha,
            "orchestration_sha": contract.orchestration_sha,
            "simulation_rerun": False,
        }
        _write_json(path, payload)

    result_path = campaign_root / "release" / "release_result.json"
    result = _read_json(result_path)
    benchmark_release = result.get("benchmark_release")
    if isinstance(benchmark_release, Mapping):
        result.setdefault("scientific_execution_benchmark_release", dict(benchmark_release))
        result["benchmark_release"] = _rewrite_embedded_publication_identity(
            benchmark_release, contract=contract
        )
    nested_resolved = result.get("resolved_manifest")
    if isinstance(nested_resolved, Mapping):
        result.setdefault("scientific_execution_resolved_manifest", dict(nested_resolved))
        result["resolved_manifest"] = _rewrite_resolved_manifest_publication_identity(
            nested_resolved, contract=contract
        )
    result.update(
        {
            "release_tag": contract.successor_github_release_tag,
            "release_id": contract.successor_github_release_tag,
            "doi": contract.successor_version_doi,
            "version_doi": contract.successor_version_doi,
            "concept_doi": contract.concept_doi,
            "source_commit": contract.source_sha,
            "publication_preflight_status": "pass",
            "publication_preflight_violations": [],
            "release_status": "ok",
            "ranking_claims_admitted": False,
        }
    )
    derivation = result.get("derivation")
    derivation = dict(derivation) if isinstance(derivation, Mapping) else {}
    derivation.update(
        {
            "builder_sha": contract.builder_sha,
            "validator_sha": contract.validator_sha,
            "orchestration_sha": contract.orchestration_sha,
            "scientific_source_sha": contract.source_sha,
            "simulation_rerun": False,
            "correction_id": contract.correction_id,
            "predecessor_version_doi": contract.predecessor_version_doi,
        }
    )
    result["derivation"] = derivation
    _write_json(result_path, result)
    _rewrite_erratum_campaign_summary(campaign_root, contract=contract)
    _assert_erratum_publication_identity(campaign_root, contract=contract)
    return resolved


def _write_erratum_receipt(
    campaign_root: Path,
    *,
    contract: ErratumContract,
    predecessor_snapshot: Any,
) -> dict[str, Any]:
    """Write scientific-equality proof before the successor archive is built.

    Returns:
        The self-contained receipt payload.
    """
    successor_snapshot = snapshot_campaign(campaign_root, contract=contract)
    receipt = build_erratum_receipt(
        contract=contract,
        predecessor=predecessor_snapshot,
        successor=successor_snapshot,
    )
    _write_json(campaign_root / ERRATUM_RECEIPT_RELATIVE, receipt)
    return receipt


def _write_custody_receipt(
    publication_dir: Path,
    *,
    bundle_name: str,
    archive_path: Path,
    bundle_dir: Path,
    source_sha: str,
    erratum_receipt: Mapping[str, Any] | None = None,
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
    if erratum_receipt is not None:
        payload["erratum"] = {
            "correction_id": erratum_receipt.get("correction_id"),
            "correction_scope": erratum_receipt.get("correction_scope"),
            "supersedes": erratum_receipt.get("supersedes"),
            "successor": erratum_receipt.get("successor"),
            "scientific_equality": erratum_receipt.get("scientific_equality"),
            "embedded_receipt_path": (f"{bundle_name}/payload/{ERRATUM_RECEIPT_RELATIVE}"),
            "embedded_receipt_sha256": sha256_file(
                bundle_dir / "payload" / ERRATUM_RECEIPT_RELATIVE
            ),
            "archive_digest_scope": (
                "detached custody receipt binds the complete archive; the embedded erratum "
                "receipt cannot self-hash its containing archive"
            ),
        }
    _write_json(publication_dir / PUBLICATION_CUSTODY_NAME, payload)


@contextlib.contextmanager
def _exclude_root_checksum_from_bundle(campaign_root: Path):
    """Keep the mutable root inventory out of the nested publication bundle.

    The root inventory is finalized after the publication archive and custody
    receipt exist.  Copying it into the bundle would create a checksum cycle:
    changing the bundle changes the root inventory, which would make the
    bundled copy stale.  The bundle has its own complete ``checksums.sha256``;
    the final campaign inventory remains at the campaign root and explicitly
    covers the publication subtree, archive, and custody receipt.
    """
    source = campaign_root / PRODUCER_SUMS_NAME
    hidden = campaign_root / f".{PRODUCER_SUMS_NAME}.not-bundled"
    if not source.is_file() or hidden.exists():
        raise DerivedReleaseError("root checksum inventory is unavailable for publication export")
    os.replace(source, hidden)
    try:
        yield
    finally:
        if source.exists():
            raise DerivedReleaseError("publication exporter recreated root checksum inventory")
        os.replace(hidden, source)


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
    _validate_safe_component(bundle_name, label="bundle_name")
    _validate_safe_component(publication_relative_dir, label="publication_relative_dir")
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
    recovery_contract: RecoveryContract = DEFAULT_RECOVERY_CONTRACT,
    erratum_contract: ErratumContract | None = None,
    predecessor_archive: Path | None = None,
    orchestration_repository_root: Path | None = None,
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
    if (erratum_contract is None) != (predecessor_archive is None):
        raise DerivedReleaseError(
            "erratum contract and predecessor archive must be supplied together"
        )
    if (erratum_contract is None) != (orchestration_repository_root is None):
        raise DerivedReleaseError(
            "erratum contract and orchestration repository root must be supplied together"
        )
    predecessor_snapshot = None
    if erratum_contract is not None:
        if erratum_contract.source_sha != recovery_contract.source_sha:
            raise DerivedReleaseError("erratum and recovery source SHAs differ")
        if erratum_contract.planner_arms != recovery_contract.arms:
            raise DerivedReleaseError("erratum and recovery planner-arm counts differ")
        if erratum_contract.episode_rows != recovery_contract.episode_rows:
            raise DerivedReleaseError("erratum and recovery episode-row counts differ")
        if erratum_contract.builder_sha != erratum_contract.validator_sha:
            raise DerivedReleaseError(
                "erratum correction builder and validator SHAs must name the same accepted commit"
            )
        if erratum_contract.validator_sha != expected_validator_commit:
            raise DerivedReleaseError(
                "erratum validator SHA differs from the exact validator commit"
            )
        _assert_exact_orchestration_checkout(
            orchestration_repository_root,  # type: ignore[arg-type]
            erratum_contract.orchestration_sha,
        )
        try:
            predecessor_snapshot = snapshot_predecessor_archive(
                predecessor_archive,  # type: ignore[arg-type]
                contract=erratum_contract,
            )
        except ReleaseErratumError as exc:
            raise DerivedReleaseError(str(exc)) from exc
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
    if output_root.exists():
        output_root = _assert_safe_directory(output_root, label="derived output root")
    else:
        output_root = Path(output_root).absolute()
        parent = output_root.parent
        if any(component.is_symlink() for component in parent.parents) or parent.is_symlink():
            raise DerivedReleaseError("derived output root contains a symlink component")
    final_campaign = output_root / derived_name
    _validate_safe_component(derived_name, label="derived_name")
    publication_name = publication_name or f"{derived_name}_publication"
    _validate_safe_component(publication_name, label="publication_name")
    final_publication = final_campaign / publication_name
    if final_campaign.exists():
        raise DerivedReleaseError("derived campaign or publication target already exists")
    _assert_distinct_validator_checkout(validator_repository_root, source_repository_root)
    _assert_frozen_source_repository(source_repository_root, recovery_contract.source_sha)
    validator = _validator_provenance(
        validator_repository_root,
        expected_commit=expected_validator_commit,
    )
    producer_evidence = verify_producer_artifacts(
        producer_root,
        expected_sums_sha256=recovery_contract.producer_sums_sha256,
        expected_receipt_sha256=_expected_current_producer_receipt_sha256(
            recovery_contract,
            preserved_receipt_source=preserved_receipt_source,
        ),
        preserved_receipt_source=preserved_receipt_source,
        expected_preserved_receipt_sha256=recovery_contract.producer_receipt_sha256,
        expected_rejected_result_sha256=recovery_contract.rejected_result_sha256,
        expected_file_count=recovery_contract.producer_file_count,
    )
    accepted_evidence = _verify_acceptance_campaign_subset(
        acceptance_root,
        producer_evidence=producer_evidence,
        expected_rejected_result_sha256=recovery_contract.rejected_result_sha256,
    )

    with _source_repository_binding(
        source_repository_root,
        validator_root=validator_repository_root,
    ):
        manifest = load_release_manifest(manifest_path)
        _assert_manifest_paths_from_source(manifest, source_repository_root)
        campaign_config = load_release_campaign_config(
            manifest,
            repository_root=source_repository_root,
        )
        manifest_validation = validate_release_manifest(
            manifest,
            campaign_config=campaign_config,
        )
        if manifest_validation.get("status") != "valid":
            raise DerivedReleaseError(
                "release manifest validation failed: "
                + "; ".join(str(item) for item in manifest_validation.get("problems", []))
            )
        acceptance = _run_exact_validator(
            validator_root=validator_repository_root,
            source_root=source_repository_root,
            acceptance_root=acceptance_root,
            manifest_path=manifest_path,
        )
    acceptance = dict(acceptance)
    acceptance["validator_execution"] = {
        "commit": validator["commit"],
        "file": validator["file"],
        "file_sha256": validator["file_sha256"],
        "execution_mode": "isolated_exact_validator_checkout",
    }
    if acceptance.get("status") != "valid":
        raise DerivedReleaseError("corrected full-release acceptance rejected preserved rows")
    source_commits = acceptance.get("source_commits")
    if source_commits != [recovery_contract.source_sha]:
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
            validator_root=validator_repository_root,
            expected_file_map=copy_file_map,
            current_receipt_bytes=producer_evidence["_current_receipt_bytes"],
            preserved_receipt_bytes=producer_evidence["_preserved_receipt_bytes"],
        )
        resolved_manifest = _read_json(
            staging_campaign / "release" / "release_manifest.resolved.json"
        )
        publication_inputs = _assert_publication_inputs_from_manifest(
            manifest, resolved_manifest, source_repository_root
        )
        boundary_reconciler = (
            _record_publication_goal_timeout_boundaries_without_row_mutation
            if erratum_contract is not None
            else _annotate_publication_goal_timeout_boundaries
        )
        goal_timeout_reconciliation = boundary_reconciler(
            staging_campaign,
            expected_rows=recovery_contract.goal_timeout_boundary_rows,
        )
        sidecar_reconciliation = _rebind_publication_sidecars(
            staging_campaign,
            source_file_map=producer_evidence["file_map"],
            boundary_reconciliation=goal_timeout_reconciliation,
            expected_arm_count=recovery_contract.arms,
            expected_row_count=recovery_contract.episode_rows,
            source_campaign_relative=recovery_contract.source_campaign_relative,
        )
        with _source_repository_binding(
            source_repository_root,
            validator_root=validator_repository_root,
        ):
            snqi_reconciliation = _reconcile_publication_snqi_diagnostics(
                staging_campaign,
                expected_row_count=recovery_contract.episode_rows,
                expected_arm_count=recovery_contract.arms,
            )
            projection_acceptance = _run_exact_validator(
                validator_root=validator_repository_root,
                source_root=source_repository_root,
                acceptance_root=staging_campaign,
                manifest_path=manifest_path,
            )
        if projection_acceptance.get("status") != "valid":
            raise DerivedReleaseError("derived publication projection failed full acceptance")
        if projection_acceptance.get("source_commits") != [recovery_contract.source_sha]:
            raise DerivedReleaseError("derived publication projection lost frozen source binding")
        publication_reconciliation = {
            "goal_timeout_boundary": goal_timeout_reconciliation,
            "sidecar_path_binding": sidecar_reconciliation,
            "snqi_diagnostics": snqi_reconciliation,
            "scientific_execution_changed": False,
            "simulation_rerun": False,
        }
        erratum_receipt = None
        if erratum_contract is not None:
            resolved_manifest = _apply_erratum_publication_identity(
                staging_campaign,
                contract=erratum_contract,
            )
            publication_inputs["erratum_zenodo_metadata"] = {
                "path": ERRATUM_METADATA_RELATIVE,
                "sha256": erratum_contract.metadata_sha256,
            }
            try:
                erratum_receipt = _write_erratum_receipt(
                    staging_campaign,
                    contract=erratum_contract,
                    predecessor_snapshot=predecessor_snapshot,
                )
            except ReleaseErratumError as exc:
                raise DerivedReleaseError(str(exc)) from exc
        derived_checksums = _write_derived_checksums(staging_campaign)
        _write_derivation_receipt(
            staging_campaign,
            producer_evidence=producer_evidence,
            accepted_evidence=accepted_evidence,
            manifest_validation=manifest_validation,
            acceptance=acceptance,
            validator=validator,
            source_sha=recovery_contract.source_sha,
            derived_checksums=derived_checksums,
            publication_inputs=publication_inputs,
            publication_reconciliation=publication_reconciliation,
            projection_acceptance=projection_acceptance,
        )
        # The receipt itself is part of the derived checksum inventory.  Rewrite
        # once after it is emitted, without ever signing SHA256SUMS itself.
        _write_derived_checksums(staging_campaign)

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
            with _exclude_root_checksum_from_bundle(staging_campaign):
                descriptor, bundle_dir, archive_path = _export_stabilised_bundle(
                    staging_campaign,
                    publication_stage=staging_publication,
                    bundle_name=f"{derived_name}_publication_bundle",
                    acceptance=projection_acceptance,
                    producer_result=_read_json(
                        staging_campaign / "release" / "producer_release_result.rejected.json"
                    ),
                    release_tag=release_tag,
                    doi=doi,
                    repository_url=repository_url,
                    publication_relative_dir=publication_name,
                )
        if erratum_contract is not None:
            _assert_erratum_publication_identity(staging_campaign, contract=erratum_contract)
            _assert_erratum_publication_identity(bundle_dir / "payload", contract=erratum_contract)
        _assert_no_private_absolute_paths(staging_campaign)
        _assert_no_private_absolute_paths(staging_publication)
        _write_custody_receipt(
            staging_publication,
            bundle_name=bundle_dir.name,
            archive_path=archive_path,
            bundle_dir=bundle_dir,
            source_sha=recovery_contract.source_sha,
            erratum_receipt=erratum_receipt,
        )
        # Detect producer mutation before promotion.  This is deliberately the
        # same strict manifest check used before copying.
        producer_after = verify_producer_artifacts(
            producer_root,
            expected_sums_sha256=recovery_contract.producer_sums_sha256,
            expected_receipt_sha256=_expected_current_producer_receipt_sha256(
                recovery_contract,
                preserved_receipt_source=preserved_receipt_source,
            ),
            preserved_receipt_source=preserved_receipt_source,
            expected_preserved_receipt_sha256=recovery_contract.producer_receipt_sha256,
            expected_rejected_result_sha256=recovery_contract.rejected_result_sha256,
            expected_file_count=recovery_contract.producer_file_count,
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
        accepted_after = _verify_acceptance_campaign_subset(
            acceptance_root,
            producer_evidence=producer_after,
            expected_rejected_result_sha256=recovery_contract.rejected_result_sha256,
        )
        if accepted_after.get("file_map") != accepted_evidence.get("file_map"):
            raise DerivedReleaseError("accepted campaign tree changed during derived build")
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
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

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
        "projection_acceptance": projection_acceptance,
        "publication_reconciliation": publication_reconciliation,
        "validator": validator,
        "publication_inputs": publication_inputs,
        "erratum_receipt": erratum_receipt,
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
        "--recovery-contract",
        type=Path,
        help=(
            "JSON benchmark-derived-release-recovery.v1 identity contract. "
            "Omit only for the historical job-14890 recovery."
        ),
    )
    parser.add_argument(
        "--preserved-receipt",
        type=Path,
        help="Single-member gzip containing the immutable pre-refresh receipt.",
    )
    parser.add_argument(
        "--erratum-contract",
        type=Path,
        help="Exact benchmark-release-erratum.v1 successor identity contract.",
    )
    parser.add_argument(
        "--erratum-repository-root",
        type=Path,
        help=(
            "Reviewed identity checkout containing the erratum contract and its "
            "repository-relative metadata. The executing tooling checkout is separate and "
            "must equal the contract's orchestration_sha."
        ),
    )
    parser.add_argument(
        "--predecessor-archive",
        type=Path,
        help="Cold-downloaded immutable predecessor publication archive.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed derived release command."""
    args = _build_parser().parse_args(argv)
    acceptance_root = args.acceptance_root or args.producer_root
    try:
        recovery_contract = (
            load_recovery_contract(args.recovery_contract)
            if args.recovery_contract is not None
            else DEFAULT_RECOVERY_CONTRACT
        )
        erratum_inputs = (
            args.erratum_contract,
            args.erratum_repository_root,
            args.predecessor_archive,
        )
        if any(value is not None for value in erratum_inputs) and not all(
            value is not None for value in erratum_inputs
        ):
            raise DerivedReleaseError(
                "--erratum-contract, --erratum-repository-root, and --predecessor-archive "
                "must be supplied together"
            )
        erratum_contract = (
            load_erratum_contract(
                args.erratum_contract,
                repository_root=args.erratum_repository_root,
            )
            if args.erratum_contract is not None
            else None
        )
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
            recovery_contract=recovery_contract,
            erratum_contract=erratum_contract,
            predecessor_archive=args.predecessor_archive,
            orchestration_repository_root=Path(__file__).resolve().parents[2],
        )
    except (
        DerivedReleaseError,
        OSError,
        ValueError,
        PublicationPreflightError,
        ReleaseErratumError,
    ) as exc:
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
