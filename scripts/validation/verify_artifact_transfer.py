#!/usr/bin/env python3
"""Fixture-based artifact transfer verifier with resumable end-to-end custody proof.

Consumes one existing ``terminal_job_harvest.v1`` receipt from
:mod:`scripts.validation.harvest_terminal_job` or one ``compute_staging_bundle.v1`` receipt
from :mod:`scripts.validation.build_compute_staging_bundle`, copies only manifest-declared
artifacts from an explicit source root to an explicit destination root, verifies byte counts
and SHA-256 at both ends, and writes a deterministic ``artifact_transfer_custody.v1`` receipt.
Fixture/local-copy only: no network, no uploads, no writes outside the destination root, no
source deletion, and no scientific interpretation.

``--check`` verifies without writing; ``--apply`` copies missing members and writes the
custody receipt under the destination root. Every existing destination member is re-hashed
before any copy, so already-correct members are reported ``already_verified`` and never
re-copied, conflicting bytes fail closed and are never overwritten, and interrupted
``.transfer-partial`` files are cleaned before a resumed copy. Exit codes: 0 verified,
2 blocked, 3 malformed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.identity.hash_utils import sha256_file, stable_hash  # noqa: E402
from scripts.tools.chunk_manifest import (  # noqa: E402
    ChunkManifestError,
    normalize_relative_path,
    scan_root,
)
from scripts.validation.build_compute_staging_bundle import BUNDLE_SCHEMA  # noqa: E402
from scripts.validation.harvest_terminal_job import RECEIPT_SCHEMA as HARVEST_SCHEMA  # noqa: E402

CUSTODY_SCHEMA = "artifact_transfer_custody.v1"
DEFAULT_RECEIPT_NAME = "artifact_transfer_custody.v1.json"
PARTIAL_SUFFIX = ".transfer-partial"
CLAIM_BOUNDARY = (
    "Operational local transfer custody only. Byte identity at both ends is recorded; no "
    "scientific interpretation, no private host/account/mount value, and no automatic source "
    "deletion is implied."
)
EXIT_VERIFIED, EXIT_BLOCKED, EXIT_MALFORMED = 0, 2, 3
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PRIVATE_LOCATOR_RE = re.compile(r"(^/|^[A-Za-z]:[\\/]|~|\$|://|@github\.com)")
Problems = list[dict[str, str]]


class TransferContractError(ValueError):
    """Raised when the transfer request cannot be read as a contract at all."""


class TransferCopyError(RuntimeError):
    """Raised when copied bytes do not reproduce the declared manifest digest."""


@dataclass(frozen=True, slots=True)
class Member:
    """One manifest-declared artifact with its normalized path, digest, and storage class."""

    relative_path: str
    sha256: str
    byte_size: int
    retention_class: str


def _add(problems: Problems, code: str, location: str, message: str) -> None:
    problems.append({"code": code, "location": location, "message": message})


def _get(node: Any, *keys: str) -> Any:
    for key in keys:
        node = node.get(key) if isinstance(node, Mapping) else None
    return node


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def load_manifest(path: Path) -> Mapping[str, Any]:
    """Load one transfer manifest JSON object, failing closed on malformed input."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise TransferContractError(f"cannot read manifest: {exc.strerror}") from exc
    except json.JSONDecodeError as exc:
        raise TransferContractError("manifest is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise TransferContractError("manifest must be a JSON object")
    return payload


def _validated_path(raw: Any, location: str, problems: Problems) -> str | None:
    if not isinstance(raw, str) or not raw:
        _add(problems, "manifest_invalid", location, "non-empty path required")
        return None
    if PRIVATE_LOCATOR_RE.search(raw):
        _add(problems, "private_locator", location, "private locator in manifest path")
        return None
    try:
        return normalize_relative_path(raw)
    except ChunkManifestError as exc:
        _add(problems, exc.code, location, str(exc))
        return None


def _member(
    item: Mapping[str, Any], location: str, problems: Problems, *, require_present: bool
) -> Member | None:
    relative = _validated_path(item.get("relative_path", item.get("path")), location, problems)
    digest = item.get("sha256")
    size = item.get("byte_size")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        _add(problems, "manifest_digest_invalid", f"{location}.sha256", "64-hex sha256 required")
        digest = None
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        _add(problems, "manifest_invalid", f"{location}.byte_size", "non-negative byte_size")
        size = None
    if require_present and item.get("status") != "present":
        _add(problems, "manifest_member_unavailable", location, "declared member is not present")
    if relative is None or digest is None or size is None:
        return None
    return Member(relative, digest, size, str(item.get("retention_class", "unspecified")))


def _dedupe(members: list[Member], problems: Problems) -> list[Member]:
    by_path: dict[str, Member] = {}
    for member in members:
        if member.relative_path in by_path:
            _add(problems, "ambiguous_manifest", member.relative_path, "duplicate member path")
        by_path[member.relative_path] = member
    folded: dict[str, str] = {}
    for relative in sorted(by_path):
        key = relative.casefold()
        if key in folded and folded[key] != relative:
            _add(problems, "ambiguous_manifest", relative, "case-colliding member path")
        folded[key] = relative
    return [by_path[relative] for relative in sorted(by_path)]


def _entries(
    payload: Mapping[str, Any], key: str, require_present: bool, problems: Problems
) -> list[Member]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        _add(problems, "manifest_invalid", key, f"non-empty {key} required")
        return []
    if any(not isinstance(item, Mapping) for item in raw):
        _add(problems, "manifest_invalid", key, f"{key} entries must be objects")
    members = [
        _member(item, f"{key}[{index}]", problems, require_present=require_present)
        for index, item in enumerate(raw)
        if isinstance(item, Mapping)
    ]
    return _dedupe([item for item in members if item], problems)


def _manifest_members(
    payload: Mapping[str, Any], problems: Problems
) -> tuple[str | None, list[Member]]:
    schema = payload.get("schema_version")
    if schema == HARVEST_SCHEMA:
        if not _hex64(payload.get("receipt_id")):
            _add(problems, "manifest_invalid", "receipt_id", "64-hex receipt_id required")
        return "terminal_job_harvest", _entries(payload, "inventory", True, problems)
    if schema == BUNDLE_SCHEMA:
        declared = payload.get("manifest_sha256")
        if not _hex64(declared):
            _add(problems, "manifest_digest_invalid", "manifest_sha256", "64-hex identity required")
        elif stable_hash({k: v for k, v in payload.items() if k != "manifest_sha256"}) != declared:
            _add(problems, "manifest_identity_mismatch", "manifest_sha256", "identity differs")
        return "compute_staging_bundle", _entries(payload, "members", False, problems)
    _add(problems, "unsupported_manifest_schema", "schema_version", f"unsupported: {schema!r}")
    return None, []


def _ownership(
    payload: Mapping[str, Any], schema_name: str | None, problems: Problems
) -> str | None:
    if schema_name == "compute_staging_bundle":
        owner = payload.get("owner")
    elif schema_name == "terminal_job_harvest":
        owner = _get(payload, "ownership", "owner")
    else:
        return None
    if not isinstance(owner, str) or SAFE_ID_RE.fullmatch(owner) is None:
        _add(problems, "missing_artifact_ownership", "owner", "manifest must declare an owner")
        return None
    return owner


def _resolve_member(root: Path, relative: str) -> Path | None:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            return None
    return current


def _check_source(root: Path, member: Member) -> tuple[str, str | None]:
    path = _resolve_member(root, member.relative_path)
    if path is None:
        return "symlink_rejected", "member path traverses a symlink"
    if not path.exists():
        return "source_missing", "declared member is absent"
    if not path.is_file():
        return "source_not_file", "member is not a regular file"
    try:
        before = path.stat()
        if before.st_size != member.byte_size:
            return "source_size_mismatch", "declared byte_size differs"
        digest = sha256_file(path)
        after = path.stat()
    except OSError:
        return "source_unreadable", "member could not be read"
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        return "source_mutated", "member changed during hashing"
    if digest != member.sha256:
        return "source_checksum_mismatch", "source digest differs from the manifest"
    return "verified", None


def _scan_destination(root: Path, problems: Problems) -> dict[str, Path] | None:
    try:
        found, _excluded = scan_root(root)
    except ChunkManifestError as exc:
        _add(problems, exc.code, "destination_root", str(exc))
        return None
    return {relative: path for relative, path, _identity in found}


def _hash_destination(path: Path, member: Member, problems: Problems) -> str:
    try:
        if path.stat().st_size != member.byte_size:
            _add(problems, "destination_conflict", member.relative_path, "byte_size differs")
            return "conflict"
        observed = sha256_file(path)
    except OSError:
        _add(problems, "destination_unreadable", member.relative_path, "member could not be read")
        return "conflict"
    if observed == member.sha256:
        return "verified"
    _add(problems, "destination_conflict", member.relative_path, "existing bytes differ")
    return "conflict"


def _copy_member(source: Path, target: Path, member: Member) -> None:
    temp = target.with_name(f".{target.name}.{member.sha256[:12]}{PARTIAL_SUFFIX}")
    try:
        with source.open("rb") as reader, temp.open("wb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if sha256_file(temp) != member.sha256:
            raise TransferCopyError("copied bytes do not reproduce the manifest digest")
        os.replace(temp, target)
    except (OSError, TransferCopyError):
        temp.unlink(missing_ok=True)
        raise


def _copy_missing(
    source_root: Path,
    destination_root: Path,
    members: list[Member],
    destination_states: dict[str, str],
    problems: Problems,
) -> dict[str, str]:
    states = dict(destination_states)
    for member in members:
        if states.get(member.relative_path) == "verified":
            continue
        source = _resolve_member(source_root, member.relative_path)
        target = _resolve_member(destination_root, member.relative_path)
        if source is None or target is None:
            _add(problems, "source_missing", member.relative_path, "member path unavailable")
            states[member.relative_path] = "pending"
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.parent.is_symlink():
                raise TransferCopyError("destination parent is a symlink")
            _copy_member(source, target, member)
        except (OSError, TransferCopyError) as exc:
            _add(problems, "copy_failed", member.relative_path, str(exc))
            states[member.relative_path] = "pending"
            continue
        states[member.relative_path] = "copied"
    return states


def _confirm_destination(
    destination_root: Path, members: list[Member], states: dict[str, str], problems: Problems
) -> dict[str, str]:
    confirmed = dict(states)
    for member in members:
        path = _resolve_member(destination_root, member.relative_path)
        observed = None
        if path is not None and path.is_file():
            try:
                observed = sha256_file(path)
            except OSError:
                observed = None
        if observed == member.sha256:
            continue
        _add(
            problems,
            "destination_checksum_mismatch",
            member.relative_path,
            "destination digest differs after transfer",
        )
        confirmed[member.relative_path] = "conflict"
    return confirmed


def _free_bytes(root: Path, problems: Problems) -> int | None:
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return int(shutil.disk_usage(probe).free)
    except OSError:
        _add(problems, "destination_capacity_unavailable", "destination_root", "not measurable")
        return None


def _existing_receipt_digest(path: Path, problems: Problems) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    digest = _get(payload, "manifest", "manifest_digest")
    if not isinstance(digest, str):
        _add(problems, "receipt_corrupt", path.name, "existing receipt unreadable")
        return None
    return digest


def _finalize(report: dict[str, Any], problems: Problems) -> dict[str, Any]:
    report["problems"] = sorted(
        problems, key=lambda item: (item["location"], item["code"], item["message"])
    )
    report["problem_count"] = len(report["problems"])
    report["reason_codes"] = sorted({item["code"] for item in report["problems"]})
    report["status"] = "verified" if not problems else "blocked"
    report.pop("receipt_id", None)
    report["receipt_id"] = stable_hash(report)
    return report


def build_transfer_report(  # noqa: C901, PLR0912, PLR0915 - one fail-closed transfer pass
    manifest_path: Path,
    source_root: Path,
    destination_root: Path,
    *,
    apply: bool = False,
    capacity_bytes: int | None = None,
) -> dict[str, Any]:
    """Build one deterministic transfer report, copying members when applying."""
    manifest_path, source_root = Path(manifest_path), Path(source_root)
    destination_root = Path(destination_root)
    problems: Problems = []
    payload = load_manifest(manifest_path)
    manifest_digest = sha256_file(manifest_path)
    schema_name, members = _manifest_members(payload, problems)
    owner = _ownership(payload, schema_name, problems) if schema_name else None
    source_states: dict[str, str] = {}
    if source_root.is_dir() and not source_root.is_symlink():
        for member in members:
            state, message = _check_source(source_root, member)
            source_states[member.relative_path] = state
            if message is not None:
                _add(problems, state, member.relative_path, message)
    else:
        _add(problems, "source_unavailable", "source_root", "root must be a directory")
    destination_ready = destination_root.is_dir() and not destination_root.is_symlink()
    if not destination_ready and apply and not destination_root.is_symlink():
        try:
            destination_root.mkdir(parents=False)
        except OSError:
            _add(problems, "destination_unavailable", "destination_root", "root not creatable")
        else:
            destination_ready = True
    elif not destination_ready:
        _add(problems, "destination_unavailable", "destination_root", "root must be a directory")
    receipt_target = destination_root / DEFAULT_RECEIPT_NAME
    allowed = {DEFAULT_RECEIPT_NAME}
    destination_states: dict[str, str] = {}
    partials: list[str] = []
    required, capacity, fits = 0, None, False
    if destination_ready:
        found = _scan_destination(destination_root, problems)
        if found is not None:
            declared = {member.relative_path: member for member in members}
            partials = sorted(path for path in found if path.endswith(PARTIAL_SUFFIX))
            for relative in sorted(found):
                if relative in allowed or relative.endswith(PARTIAL_SUFFIX):
                    continue
                member = declared.get(relative)
                if member is None:
                    _add(problems, "unexpected_destination_member", relative, "not declared")
                else:
                    destination_states[relative] = _hash_destination(
                        found[relative], member, problems
                    )
            for member in members:
                if member.relative_path not in destination_states:
                    destination_states[member.relative_path] = "missing"
        if partials and not apply:
            for relative in partials:
                _add(problems, "partial_transfer", relative, "interrupted member present")
        elif partials:
            for relative in partials:
                (destination_root / relative).unlink(missing_ok=True)
        required = sum(
            member.byte_size
            for member in members
            if destination_states.get(member.relative_path) != "verified"
        )
        measured = (
            capacity_bytes
            if capacity_bytes is not None
            else _free_bytes(destination_root, problems)
        )
        fits = measured is not None and required <= measured
        capacity = capacity_bytes
        if measured is not None and not fits:
            _add(
                problems, "destination_capacity_exceeded", "destination_root", "capacity too small"
            )
    if apply and destination_ready:
        prior = _existing_receipt_digest(receipt_target, problems)
        if prior is not None and prior != manifest_digest:
            _add(problems, "destination_manifest_conflict", receipt_target.name, "other manifest")
    if apply and destination_ready and not problems:
        destination_states = _copy_missing(
            source_root, destination_root, members, destination_states, problems
        )
        if not problems:
            destination_states = _confirm_destination(
                destination_root, members, destination_states, problems
            )
    if destination_ready:
        for member in members:
            state = destination_states.get(member.relative_path)
            if state in (None, "missing", "pending"):
                _add(problems, "destination_incomplete", member.relative_path, "member absent")
                destination_states[member.relative_path] = "pending"
            if (destination_root / member.relative_path).is_dir():
                _add(
                    problems, "destination_conflict", member.relative_path, "directory blocks path"
                )
                destination_states[member.relative_path] = "conflict"
    state_map = {"verified": "already_verified", "copied": "copied", "conflict": "conflict"}
    files = [
        {
            "relative_path": member.relative_path,
            "byte_size": member.byte_size,
            "sha256": member.sha256,
            "retention_class": member.retention_class,
            "source_state": source_states.get(member.relative_path, "not_checked"),
            "state": state_map.get(destination_states.get(member.relative_path), "pending"),
        }
        for member in members
    ]
    counts = Counter(record["state"] for record in files)
    report: dict[str, Any] = {
        "schema_version": CUSTODY_SCHEMA,
        "mode": "apply" if apply else "check",
        "owner": owner,
        "durability_class": "unspecified",
        "transfer": {
            "method": "local_copy",
            "network_access": False,
            "uploads": False,
            "source_deletion": "never_automatic",
        },
        "manifest": {
            "schema_version": schema_name,
            "manifest_digest": manifest_digest,
            "member_count": len(members),
            "total_bytes": sum(member.byte_size for member in members),
        },
        "destination": {
            "capacity_bytes": capacity,
            "required_bytes": required,
            "fits": fits,
            "receipt_written": False,
            "partial_members_cleaned": len(partials) if apply else 0,
            "verified_members": counts.get("already_verified", 0) + counts.get("copied", 0),
        },
        "files": files,
        "counts": {
            "members": len(members),
            "copied": counts.get("copied", 0),
            "already_verified": counts.get("already_verified", 0),
            "pending": counts.get("pending", 0),
            "conflicts": counts.get("conflict", 0),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if (
        apply
        and destination_ready
        and not any(
            problem["code"] in ("destination_manifest_conflict", "receipt_corrupt")
            for problem in problems
        )
    ):
        report["destination"]["receipt_written"] = True
    report = _finalize(report, problems)
    if report["destination"]["receipt_written"]:
        try:
            temp = receipt_target.with_name(f".{receipt_target.name}{PARTIAL_SUFFIX}")
            temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temp, receipt_target)
        except OSError as exc:
            _add(problems, "receipt_write_failed", receipt_target.name, str(exc))
            report["destination"]["receipt_written"] = False
            report = _finalize(report, problems)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_artifact_transfer", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Harvest or staging receipt.")
    parser.add_argument("--source-root", "--source", dest="source_root", type=Path, required=True)
    parser.add_argument(
        "--destination-root", "--destination", dest="destination_root", type=Path, required=True
    )
    parser.add_argument("--check", action="store_true", help="Verify only; never write.")
    parser.add_argument("--apply", action="store_true", help="Copy missing members and record.")
    parser.add_argument("--capacity-bytes", type=int, default=None, help="Declared free capacity.")
    parser.add_argument("--format", choices=("text", "json"), default="text", help="Output format.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the artifact transfer verifier and return the process exit code."""
    args = _parser().parse_args(argv)
    if args.check == args.apply:
        print("error: pass exactly one of --check or --apply", file=sys.stderr)
        return EXIT_MALFORMED
    if args.capacity_bytes is not None and args.capacity_bytes < 0:
        print("error: --capacity-bytes must be non-negative", file=sys.stderr)
        return EXIT_MALFORMED
    try:
        report = build_transfer_report(
            args.manifest,
            args.source_root,
            args.destination_root,
            apply=args.apply,
            capacity_bytes=args.capacity_bytes,
        )
    except (TransferContractError, OSError) as exc:
        print(f"artifact transfer verifier: malformed input: {exc}", file=sys.stderr)
        return EXIT_MALFORMED
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] == "verified":
        counts = report["counts"]
        print(
            "artifact transfer verifier: verified "
            f"members={counts['members']} already_verified={counts['already_verified']} "
            f"copied={counts['copied']}"
        )
    else:
        print(f"artifact transfer verifier: blocked ({', '.join(report['reason_codes'])})")
        for problem in report["problems"]:
            print(f"  [{problem['code']}] {problem['location']}: {problem['message']}")
    return EXIT_VERIFIED if report["status"] == "verified" else EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
