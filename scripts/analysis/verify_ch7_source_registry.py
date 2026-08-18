"""Verify the commit-pinned Chapter 7 source registry and package snapshots.

The registry is custody metadata, not an admission receipt.  It points at the
immutable repository commit that contains the exact v1 source and v2 projection
trees.  Retrieval is deliberately fail-closed: a missing local object is
``unavailable``, malformed or moving references are ``blocked``, and any byte
or identity difference is a ``mismatch``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path("configs/analysis/source_gate_registry.v1.json")
REGISTRY_SCHEMA_VERSION = "case-source-integrity-registry.v1"
REGISTRY_ID = "source_gate_registry.v1"
RECORD_SCHEMA_VERSION = "ch7-source-registry-record.v1"
RECORD_ID = "issue-7411-ch7-v1-v2-source.v1"
RECORD_HASH_ALGORITHM = "canonical-json-sha256-without-record_sha256.v1"
RETRIEVAL_PROTOCOL = "git-commit-tree-v1"
REPOSITORY_NAME = "ll7/robot_sf_ll7"
REPOSITORY_URL = "https://github.com/ll7/robot_sf_ll7"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")


class SourceRegistryError(ValueError):
    """Base error for a fail-closed source-registry check."""

    status = "unavailable"


class SourceRegistryBlockedError(SourceRegistryError):
    """Raised when registry metadata cannot authorize retrieval."""

    status = "blocked"


class SourceRegistryUnavailableError(SourceRegistryError):
    """Raised when the registry or pinned git object cannot be retrieved."""

    status = "unavailable"


class SourceRegistryMismatchError(SourceRegistryError):
    """Raised when retrieved bytes or identities differ from the registry."""

    status = "mismatch"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise SourceRegistryUnavailableError(f"registry is unreadable: {path}") from exc


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SourceRegistryBlockedError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _require_git_object(value: Any, label: str) -> str:
    if not isinstance(value, str) or GIT_OBJECT_RE.fullmatch(value) is None:
        raise SourceRegistryBlockedError(f"{label} is not an immutable 40-character git object")
    return value


def _read_registry(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceRegistryUnavailableError(f"source registry is unavailable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise SourceRegistryBlockedError("source registry must be a JSON object")
    return dict(payload)


def record_sha256(record: Mapping[str, Any]) -> str:
    """Return the self-hash for a record with its ``record_sha256`` removed."""

    body = copy.deepcopy(dict(record))
    body.pop("record_sha256", None)
    return _sha256_bytes(_canonical_bytes(body))


def _safe_relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceRegistryBlockedError(f"{label} is missing")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise SourceRegistryBlockedError(f"{label} is not a safe repository path")
    return path.as_posix()


def _parse_sha256sums(payload: bytes, label: str) -> dict[str, str]:
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise SourceRegistryMismatchError(f"{label} is not ASCII SHA256SUMS data") from exc
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        if "  " not in raw:
            raise SourceRegistryMismatchError(f"{label} has malformed line {line_number}")
        digest, relative = raw.split("  ", 1)
        digest = _require_sha256(digest, f"{label} line {line_number}")
        relative = _safe_relative_path(relative, f"{label} line {line_number} path")
        if relative in entries:
            raise SourceRegistryMismatchError(f"{label} has duplicate member: {relative}")
        entries[relative] = digest
    if not entries:
        raise SourceRegistryMismatchError(f"{label} is empty")
    return entries


def _git_bytes(repository_root: Path, arguments: Sequence[str], label: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise SourceRegistryUnavailableError(f"git retrieval is unavailable: {label}") from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        reason = f": {detail}" if detail else ""
        raise SourceRegistryUnavailableError(f"git object is unavailable for {label}{reason}")
    return result.stdout


def _git_object(repository_root: Path, commit: str, path: str) -> str:
    output = _git_bytes(repository_root, ("rev-parse", f"{commit}:{path}"), path)
    value = output.decode("ascii", errors="replace").strip()
    return _require_git_object(value, f"git object for {path}")


def _validate_registry(registry: Mapping[str, Any]) -> Mapping[str, Any]:
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise SourceRegistryBlockedError("source registry schema version is unsupported")
    if registry.get("registry_id") != REGISTRY_ID:
        raise SourceRegistryBlockedError("source registry identifier is missing or changed")
    records = registry.get("durable_records")
    if not isinstance(records, list):
        raise SourceRegistryBlockedError("source registry durable records are missing")
    matches = [
        record
        for record in records
        if isinstance(record, Mapping) and record.get("record_id") == RECORD_ID
    ]
    if len(matches) != 1:
        raise SourceRegistryBlockedError(
            "source registry must contain exactly one Chapter 7 record"
        )
    return matches[0]


def _validate_record(  # noqa: C901, PLR0912
    record: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], list[Mapping[str, Any]]]:
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        raise SourceRegistryBlockedError("Chapter 7 source record schema version is unsupported")
    if record.get("record_id") != RECORD_ID:
        raise SourceRegistryBlockedError("Chapter 7 source record identifier is missing or changed")
    if record.get("status") != "approved":
        raise SourceRegistryBlockedError("Chapter 7 source record is not approved")
    if record.get("record_hash_algorithm") != RECORD_HASH_ALGORITHM:
        raise SourceRegistryBlockedError("Chapter 7 source record hash algorithm is unsupported")
    recorded_hash = _require_sha256(record.get("record_sha256"), "record_sha256")
    if record_sha256(record) != recorded_hash:
        raise SourceRegistryMismatchError("source registry record self-hash mismatch")
    repository = record.get("repository")
    if not isinstance(repository, Mapping):
        raise SourceRegistryBlockedError("Chapter 7 source record repository metadata is missing")
    if repository.get("name") != REPOSITORY_NAME or repository.get("url") != REPOSITORY_URL:
        raise SourceRegistryBlockedError("Chapter 7 source record repository identity changed")
    if repository.get("ref_policy") != "immutable_commit":
        raise SourceRegistryBlockedError("source record does not require an immutable commit")
    commit = _require_git_object(repository.get("commit"), "repository.commit")
    retrieval = record.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise SourceRegistryBlockedError("Chapter 7 source record retrieval metadata is missing")
    if retrieval.get("protocol") != RETRIEVAL_PROTOCOL:
        raise SourceRegistryBlockedError("source record retrieval protocol is unsupported")
    if retrieval.get("public_access") is not True:
        raise SourceRegistryBlockedError("source record does not declare public access")
    if not isinstance(retrieval.get("retention"), str) or not retrieval["retention"]:
        raise SourceRegistryBlockedError("source record retention metadata is missing")
    if not isinstance(retrieval.get("rights"), Mapping):
        raise SourceRegistryBlockedError("source record rights metadata is missing")
    packages = record.get("packages")
    if not isinstance(packages, list) or len(packages) != 2:
        raise SourceRegistryBlockedError("source record must contain exactly v1 and v2 packages")
    if not all(isinstance(package, Mapping) for package in packages):
        raise SourceRegistryBlockedError("source record contains a malformed package entry")
    package_ids = {package["package_id"] for package in packages}
    if package_ids != {"ch7-evidence-package-v1", "ch7-evidence-package-v2"}:
        raise SourceRegistryBlockedError("source record must contain the v1 and v2 package IDs")
    return commit, retrieval, list(packages)


def _verify_package_snapshot(  # noqa: C901, PLR0912
    repository_root: Path,
    *,
    commit: str,
    package: Mapping[str, Any],
    retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    package_id = package.get("package_id")
    if not isinstance(package_id, str) or not package_id:
        raise SourceRegistryBlockedError("source package identifier is missing")
    path = _safe_relative_path(package.get("path"), f"{package_id}.path")
    if package.get("commit") != commit:
        raise SourceRegistryBlockedError(f"{package_id} is not pinned to the record commit")
    if package.get("media_type") not in {
        "application/vnd.robot-sf.ch7-evidence-package.v1+directory",
        "application/vnd.robot-sf.ch7-evidence-package.v2+directory",
    }:
        raise SourceRegistryBlockedError(f"{package_id} media type is unsupported")
    if package.get("public_access") is not True:
        raise SourceRegistryBlockedError(f"{package_id} does not declare public access")
    if not isinstance(package.get("retention"), str) or not package["retention"]:
        raise SourceRegistryBlockedError(f"{package_id} retention metadata is missing")
    if not isinstance(package.get("rights"), Mapping):
        raise SourceRegistryBlockedError(f"{package_id} rights metadata is missing")
    if package.get("retrieval_protocol") != retrieval["protocol"]:
        raise SourceRegistryBlockedError(f"{package_id} retrieval protocol changed")
    expected_uri = f"{REPOSITORY_URL}/tree/{commit}/{path}"
    if package.get("durable_uri") != expected_uri:
        raise SourceRegistryBlockedError(f"{package_id} durable URI is not commit-pinned")
    tree_sha = _require_git_object(package.get("tree_sha"), f"{package_id}.tree_sha")
    observed_tree_sha = _git_object(repository_root, commit, path)
    if observed_tree_sha != tree_sha:
        raise SourceRegistryMismatchError(f"{package_id} git tree identity mismatch")
    sums_path = _safe_relative_path(package.get("sha256sums_path"), f"{package_id}.sha256sums_path")
    expected_sums_path = f"{path}/SHA256SUMS"
    if sums_path != expected_sums_path:
        raise SourceRegistryBlockedError(f"{package_id} SHA256SUMS path is not package-local")
    sums_bytes = _git_bytes(repository_root, ("show", f"{commit}:{sums_path}"), sums_path)
    sums_sha = _require_sha256(package.get("sha256sums_sha256"), f"{package_id}.sha256sums_sha256")
    if _sha256_bytes(sums_bytes) != sums_sha:
        raise SourceRegistryMismatchError(f"{package_id} SHA256SUMS digest mismatch")
    observed_members = _parse_sha256sums(sums_bytes, f"{package_id} SHA256SUMS")
    recorded_members = package.get("member_sha256sums")
    if not isinstance(recorded_members, Mapping):
        raise SourceRegistryBlockedError(f"{package_id} member SHA-256 map is missing")
    normalized_members = {
        _safe_relative_path(member, f"{package_id} member"): _require_sha256(
            digest, f"{package_id} member {member}"
        )
        for member, digest in recorded_members.items()
    }
    if normalized_members != observed_members:
        raise SourceRegistryMismatchError(f"{package_id} member SHA-256 map mismatch")
    required_members = package.get("required_members")
    if not isinstance(required_members, list) or not required_members:
        raise SourceRegistryBlockedError(f"{package_id} required members are missing")
    for member in required_members:
        member_path = _safe_relative_path(member, f"{package_id} required member")
        if member_path not in observed_members:
            raise SourceRegistryMismatchError(
                f"{package_id} required member is not listed: {member_path}"
            )
    for member, expected_digest in observed_members.items():
        member_bytes = _git_bytes(repository_root, ("show", f"{commit}:{path}/{member}"), member)
        if _sha256_bytes(member_bytes) != expected_digest:
            raise SourceRegistryMismatchError(f"{package_id} member digest mismatch: {member}")
    return {
        "package_id": package_id,
        "path": path,
        "tree_sha": tree_sha,
        "sha256sums_sha256": sums_sha,
        "durable_uri": expected_uri,
    }


def verify_source_registry(
    *,
    repository_root: Path = ROOT,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the canonical registry and return a manifest-safe custody binding."""

    repository_root = repository_root.resolve()
    path = registry_path or repository_root / REGISTRY_PATH
    path = path.resolve()
    registry = _read_registry(path)
    registry_sha = _sha256_file(path)
    record = _validate_registry(registry)
    commit, retrieval, packages = _validate_record(record)
    if len(packages) != 2:
        raise SourceRegistryBlockedError(
            "Chapter 7 source record contains malformed package entries"
        )
    observed_packages = [
        _verify_package_snapshot(
            repository_root,
            commit=commit,
            package=package,
            retrieval=retrieval,
        )
        for package in packages
    ]
    return {
        "registry_id": REGISTRY_ID,
        "record_id": RECORD_ID,
        "record_sha256": record["record_sha256"],
        "registry_path": REGISTRY_PATH.as_posix(),
        "registry_sha256": registry_sha,
        "source_commit": commit,
        "retrieval": {
            "protocol": retrieval["protocol"],
            "repository_url": REPOSITORY_URL,
            "public_access": retrieval["public_access"],
            "retention": retrieval["retention"],
            "rights": dict(retrieval["rights"]),
        },
        "durable_sources": observed_packages,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--check-only", action="store_true", help="verify without changing files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fail-closed registry check and emit a machine-readable result."""

    args = _parser().parse_args(argv)
    try:
        result = verify_source_registry(
            repository_root=args.repository_root,
            registry_path=args.registry,
        )
    except SourceRegistryError as exc:
        print(json.dumps({"status": exc.status, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"status": "verified", **result}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
