"""Shared exact historical evidence binding resolver.

The resolver is used by both runtime evidence consumers and the evidence
registry linter. It validates only the two explicit manifest occurrences and
fails closed when the required Git history, object identities, or JSON pointer
bindings cannot be proven.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HISTORICAL_BINDING_MANIFEST = Path(
    "scripts/validation/evidence_registry_historical_bindings.v1.json"
)
HISTORICAL_BINDING_SCHEMA = "robot_sf.evidence_registry_historical_bindings.v1"
HISTORICAL_BINDING_REPOSITORY = "ll7/robot_sf_ll7"
FULL_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
JSON_POINTER_TOKEN_RE = re.compile(r"(?:[^~]|~[01])*\Z")
JSON_POINTER_ARRAY_INDEX_RE = re.compile(r"(?:0|[1-9][0-9]*)\Z", re.ASCII)


class DuplicateJSONKeyError(ValueError):
    """Raised when strict historical JSON contains a duplicate object key."""


class HistoricalBindingError(RuntimeError):
    """Raised when a versioned historical reference binding cannot be verified."""


def _git_succeeds(repo_root: Path, *args: str) -> bool:
    """Return whether a Git query succeeds without exposing Git diagnostics."""
    return (
        subprocess.run(
            ["git", *args],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _git_bytes(repo_root: Path, *args: str) -> bytes | None:
    """Return Git command output bytes, or ``None`` when the object is unavailable."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _is_tracked(
    repo_root: Path,
    repo_path: str,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> bool:
    """Return whether a normalized path exists in the current or immutable tree."""
    if content_ref is not None:
        if tracked_paths is not None:
            return repo_path in tracked_paths
        if content_cache is not None and repo_path in content_cache:
            return True
        return _git_succeeds(repo_root, "cat-file", "-e", f"{content_ref}:{repo_path}")
    return _git_succeeds(repo_root, "ls-files", "--error-unmatch", "--", repo_path)


def _repository_file_bytes(
    repo_root: Path,
    repo_path: str,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
) -> bytes | None:
    """Read repository bytes from the working tree or an immutable commit tree.

    Returns:
        The file bytes, or ``None`` when unavailable.
    """
    if content_ref is not None:
        if content_cache is not None and repo_path in content_cache:
            return content_cache[repo_path]
        return _git_bytes(repo_root, "show", f"{content_ref}:{repo_path}")
    try:
        return (repo_root / repo_path).read_bytes()
    except OSError:
        return None


def _resolve_repo_path(repo_root: Path, value: str) -> tuple[str, Path] | None:
    """Normalize one repository-relative path and reject URLs or escapes.

    Returns:
        The normalized relative path and absolute path, or ``None`` when unsafe.
    """
    if "://" in value or value.startswith("urn:"):
        return None
    candidate = (repo_root / value).resolve()
    try:
        normalized = candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return normalized.as_posix(), candidate


def _reject_duplicate_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys before pointer evaluation.

    Returns:
        The object mapping when all keys are unique.
    """
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJSONKeyError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _load_document(
    path: Path,
    raw: bytes | None = None,
    *,
    reject_duplicate_json_keys: bool = False,
) -> Any:
    """Load one historical JSON document with optional duplicate-key rejection.

    Returns:
        The parsed JSON value.
    """
    if raw is None:
        raw = path.read_bytes()
    decoded = raw.decode("utf-8")
    if path.suffix.lower() != ".json":
        raise ValueError(f"historical binding document is not JSON: {path}")
    if reject_duplicate_json_keys:
        return json.loads(decoded, object_pairs_hook=_reject_duplicate_json_object)
    return json.loads(decoded)


def _json_pointer_get(value: Any, pointer: str) -> Any:
    """Resolve one RFC 6901 JSON pointer.

    Returns:
        The value addressed by ``pointer``.
    """
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}")
    current = value
    for raw_part in pointer.split("/")[1:]:
        if JSON_POINTER_TOKEN_RE.fullmatch(raw_part) is None:
            raise ValueError(f"invalid JSON pointer escape in {pointer!r}")
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(pointer)
            current = current[part]
        elif isinstance(current, list):
            if JSON_POINTER_ARRAY_INDEX_RE.fullmatch(part) is None:
                raise KeyError(pointer)
            index = int(part)
            try:
                current = current[index]
            except IndexError as exc:
                raise KeyError(pointer) from exc
        else:
            raise KeyError(pointer)
    return current


def _iter_mappings(
    value: Any, ancestors: tuple[Mapping[str, Any], ...] = ()
) -> Iterable[tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]]:
    """Yield nested mappings and their ancestors for exact occurrence counting."""
    if isinstance(value, Mapping):
        yield value, ancestors
        for child in value.values():
            yield from _iter_mappings(child, (*ancestors, value))
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child, ancestors)


HISTORICAL_BINDING_TOP_FIELDS = frozenset(
    {
        "$schema",
        "schema_version",
        "repository",
        "reviewed_ancestry_anchor",
        "bindings",
    }
)
HISTORICAL_BINDING_FIELDS = frozenset(
    {
        "repository",
        "reviewed_ancestry_anchor",
        "consumer_path",
        "consumer_sha256",
        "consumer_blob_sha1",
        "reference_locator",
        "reference_path",
        "declared_sha256",
        "producer_commit",
        "producer_tree",
        "producer_reference_sha256",
        "producer_reference_blob_sha1",
        "producer_consumer_sha256",
        "producer_consumer_blob_sha1",
        "parent_commit",
        "parent_reference_sha256",
    }
)


def _historical_binding_error(message: str) -> HistoricalBindingError:
    """Return a consistently prefixed historical-binding validation error."""
    return HistoricalBindingError(f"{HISTORICAL_BINDING_MANIFEST}: {message}")


def _blob_sha1(value: bytes) -> str:
    """Return Git's SHA-1 for a blob without writing an object to the repository."""
    header = b"blob " + str(len(value)).encode("ascii") + b"\0"
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def _historical_binding_path(repo_root: Path, value: Any, field: str) -> str:
    """Validate one canonical repository-relative path from the binding manifest.

    Returns:
        The canonical repository-relative path.
    """
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise _historical_binding_error(f"{field} must be a canonical repository-relative path")
    if Path(value).is_absolute():
        raise _historical_binding_error(f"{field} must be repository-relative")
    resolved = _resolve_repo_path(repo_root, value)
    if resolved is None or resolved[0] != value:
        raise _historical_binding_error(f"{field} is not a canonical repository-relative path")
    return resolved[0]


def _historical_binding_text(binding: Mapping[str, Any], field: str) -> str:
    """Return one required non-empty textual binding field."""
    value = binding.get(field)
    if not isinstance(value, str) or not value:
        raise _historical_binding_error(f"bindings entry requires non-empty {field}")
    return value


def _historical_binding_sha(
    binding: Mapping[str, Any], field: str, pattern: re.Pattern[str]
) -> str:
    """Return one canonical lowercase hexadecimal binding digest."""
    value = _historical_binding_text(binding, field)
    if not pattern.fullmatch(value) or value != value.lower():
        raise _historical_binding_error(
            f"bindings {field} must be a full hexadecimal digest in lowercase"
        )
    return value.lower()


def _historical_binding_git_text(repo_root: Path, *args: str) -> str | None:
    """Read one UTF-8/ASCII Git identity, returning None for unavailable objects.

    Returns:
        The decoded Git output, or ``None`` when unavailable or non-ASCII.
    """
    raw = _git_bytes(repo_root, *args)
    if raw is None:
        return None
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None


def _historical_binding_git_bytes(repo_root: Path, commit: str, path: str) -> bytes | None:
    """Read one historical path from an exact commit tree.

    Returns:
        The historical file bytes, or ``None`` when unavailable.
    """
    return _git_bytes(repo_root, "show", f"{commit}:{path}")


def _historical_binding_occurrences(
    value: Any, reference_path: str, declared_sha256: str
) -> list[Mapping[str, Any]]:
    """Find exact path/checksum reference mappings in one evaluated consumer.

    Returns:
        All matching nested mappings, in traversal order.
    """
    occurrences: list[Mapping[str, Any]] = []
    if isinstance(value, str):
        return occurrences
    for mapping, _ancestors in _iter_mappings(value):
        path = mapping.get("path")
        digest = mapping.get("sha256")
        if (
            isinstance(path, str)
            and path == reference_path
            and isinstance(digest, str)
            and digest.lower() == declared_sha256
        ):
            occurrences.append(mapping)
    return occurrences


def _historical_binding_locator_is_valid(locator: str) -> bool:
    """Return whether a locator is a syntactically valid RFC 6901 pointer."""
    if not locator.startswith("/"):
        return False
    return all(JSON_POINTER_TOKEN_RE.fullmatch(part) is not None for part in locator.split("/")[1:])


def _historical_binding_file(
    repo_root: Path,
    path: str,
    *,
    content_ref: str | None,
    content_cache: Mapping[str, bytes] | None,
    tracked_paths: set[str] | None,
    label: str,
) -> bytes:
    """Read a tracked file from the evaluated working tree or immutable candidate tree.

    Returns:
        The tracked file bytes.
    """
    if not _is_tracked(repo_root, path, content_ref, content_cache, tracked_paths):
        raise _historical_binding_error(f"{label} is not tracked: {path}")
    value = _repository_file_bytes(repo_root, path, content_ref, content_cache)
    if value is None:
        raise _historical_binding_error(f"{label} cannot be read: {path}")
    return value


def _validate_historical_binding_entry(  # noqa: C901, PLR0912, PLR0913, PLR0915
    repo_root: Path,
    binding: Any,
    index: int,
    repository: str,
    reviewed_ancestry_anchor: str,
    *,
    content_ref: str | None,
    content_cache: Mapping[str, bytes] | None,
    tracked_paths: set[str] | None,
    authority_ref: str,
) -> tuple[tuple[str, str, str], dict[str, str]]:
    """Validate one exact historical binding against evaluated and Git bytes.

    Returns:
        The binding key and its normalized, validated fields.
    """
    if not isinstance(binding, Mapping):
        raise _historical_binding_error(f"bindings[{index}] must be an object")
    unknown = set(binding) - HISTORICAL_BINDING_FIELDS
    missing = HISTORICAL_BINDING_FIELDS - set(binding)
    if unknown:
        raise _historical_binding_error(
            f"bindings[{index}] contains unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise _historical_binding_error(f"bindings[{index}] is missing fields: {sorted(missing)}")
    if binding.get("repository") != repository:
        raise _historical_binding_error(f"bindings[{index}] repository does not match the header")
    if binding.get("reviewed_ancestry_anchor") != reviewed_ancestry_anchor:
        raise _historical_binding_error(
            f"bindings[{index}] reviewed ancestry anchor does not match the header"
        )
    consumer_path = _historical_binding_path(repo_root, binding["consumer_path"], "consumer_path")
    reference_path = _historical_binding_path(
        repo_root, binding["reference_path"], "reference_path"
    )
    locator = _historical_binding_text(binding, "reference_locator")
    if not _historical_binding_locator_is_valid(locator):
        raise _historical_binding_error(
            f"bindings[{index}] reference_locator must be a JSON pointer"
        )
    consumer_sha256 = _historical_binding_sha(binding, "consumer_sha256", SHA256_RE)
    consumer_blob_sha1 = _historical_binding_sha(binding, "consumer_blob_sha1", FULL_SHA1_RE)
    declared_sha256 = _historical_binding_sha(binding, "declared_sha256", SHA256_RE)
    producer_commit = _historical_binding_sha(binding, "producer_commit", FULL_SHA1_RE)
    producer_tree = _historical_binding_sha(binding, "producer_tree", FULL_SHA1_RE)
    producer_reference_sha256 = _historical_binding_sha(
        binding, "producer_reference_sha256", SHA256_RE
    )
    producer_reference_blob_sha1 = _historical_binding_sha(
        binding, "producer_reference_blob_sha1", FULL_SHA1_RE
    )
    producer_consumer_sha256 = _historical_binding_sha(
        binding, "producer_consumer_sha256", SHA256_RE
    )
    producer_consumer_blob_sha1 = _historical_binding_sha(
        binding, "producer_consumer_blob_sha1", FULL_SHA1_RE
    )
    parent_commit = _historical_binding_sha(binding, "parent_commit", FULL_SHA1_RE)
    parent_reference_sha256 = _historical_binding_sha(binding, "parent_reference_sha256", SHA256_RE)

    anchor_type = _historical_binding_git_text(
        repo_root, "cat-file", "-t", reviewed_ancestry_anchor
    )
    if anchor_type != "commit":
        raise _historical_binding_error(
            f"reviewed ancestry anchor is unavailable as a commit: {reviewed_ancestry_anchor}"
        )
    if not _git_succeeds(
        repo_root,
        "merge-base",
        "--is-ancestor",
        reviewed_ancestry_anchor,
        authority_ref,
    ):
        raise _historical_binding_error(
            f"reviewed ancestry anchor {reviewed_ancestry_anchor} is not reachable from "
            f"authority {authority_ref}"
        )
    if _historical_binding_git_text(repo_root, "cat-file", "-t", producer_commit) != "commit":
        raise _historical_binding_error(f"producer commit is unavailable: {producer_commit}")
    if _historical_binding_git_text(repo_root, "cat-file", "-t", parent_commit) != "commit":
        raise _historical_binding_error(f"parent commit is unavailable: {parent_commit}")
    if _historical_binding_git_text(repo_root, "cat-file", "-t", producer_tree) != "tree":
        raise _historical_binding_error(f"producer tree object is unavailable: {producer_tree}")
    for label, object_id in (
        ("producer reference blob", producer_reference_blob_sha1),
        ("producer consumer blob", producer_consumer_blob_sha1),
    ):
        if _historical_binding_git_text(repo_root, "cat-file", "-t", object_id) != "blob":
            raise _historical_binding_error(f"{label} object is unavailable: {object_id}")
    resolved_tree = _historical_binding_git_text(
        repo_root, "rev-parse", f"{producer_commit}^{{tree}}"
    )
    if resolved_tree != producer_tree:
        raise _historical_binding_error(f"producer tree does not match {producer_commit}")
    parents = _historical_binding_git_text(
        repo_root, "rev-list", "--parents", "-n", "1", producer_commit
    )
    if parents is None or parents.split()[1:] != [parent_commit]:
        raise _historical_binding_error(
            f"parent transition for {producer_commit} is not the exact declared parent"
        )
    if not _git_succeeds(
        repo_root, "merge-base", "--is-ancestor", producer_commit, reviewed_ancestry_anchor
    ):
        raise _historical_binding_error(
            f"producer commit {producer_commit} is not reachable from reviewed anchor "
            f"{reviewed_ancestry_anchor}"
        )

    producer_reference = _historical_binding_git_bytes(repo_root, producer_commit, reference_path)
    producer_consumer = _historical_binding_git_bytes(repo_root, producer_commit, consumer_path)
    parent_reference = _historical_binding_git_bytes(repo_root, parent_commit, reference_path)
    if producer_reference is None or producer_consumer is None or parent_reference is None:
        raise _historical_binding_error(
            f"producer/parent paths are unavailable for binding {consumer_path}"
        )
    if _blob_sha1(producer_reference) != producer_reference_blob_sha1:
        raise _historical_binding_error(f"producer reference blob mismatch for {reference_path}")
    if _blob_sha1(producer_consumer) != producer_consumer_blob_sha1:
        raise _historical_binding_error(f"producer consumer blob mismatch for {consumer_path}")
    if hashlib.sha256(producer_reference).hexdigest() != producer_reference_sha256:
        raise _historical_binding_error(f"producer reference digest mismatch for {reference_path}")
    if producer_reference_sha256 != declared_sha256:
        raise _historical_binding_error(
            f"declared digest does not match producer reference {reference_path}"
        )
    if hashlib.sha256(producer_consumer).hexdigest() != producer_consumer_sha256:
        raise _historical_binding_error(f"producer consumer digest mismatch for {consumer_path}")
    try:
        producer_consumer_value = _load_document(
            repo_root / consumer_path,
            raw=producer_consumer,
            reject_duplicate_json_keys=True,
        )
        producer_pointer_value = _json_pointer_get(producer_consumer_value, locator)
    except DuplicateJSONKeyError as exc:
        raise _historical_binding_error(
            f"producer consumer contains duplicate JSON object key: {exc}"
        ) from exc
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise _historical_binding_error(
            f"producer consumer reference_locator does not resolve in {consumer_path}: {locator}"
        ) from exc
    if not isinstance(producer_pointer_value, Mapping):
        raise _historical_binding_error(
            f"producer consumer reference_locator must identify a path/checksum mapping "
            f"in {consumer_path}"
        )
    producer_pointer_digest = producer_pointer_value.get("sha256")
    if (
        producer_pointer_value.get("path") != reference_path
        or not isinstance(producer_pointer_digest, str)
        or producer_pointer_digest.lower() != declared_sha256
    ):
        raise _historical_binding_error(
            f"producer consumer reference_locator does not identify the declared reference "
            f"in {consumer_path}"
        )
    producer_occurrences = _historical_binding_occurrences(
        producer_consumer_value, reference_path, declared_sha256
    )
    if len(producer_occurrences) != 1 or producer_occurrences[0] is not producer_pointer_value:
        raise _historical_binding_error(
            f"producer consumer reference must occur exactly once at {consumer_path}{locator}"
        )
    if hashlib.sha256(parent_reference).hexdigest() != parent_reference_sha256:
        raise _historical_binding_error(f"parent reference digest mismatch for {reference_path}")
    if parent_reference_sha256 == producer_reference_sha256:
        raise _historical_binding_error(f"parent transition did not change {reference_path}")

    consumer_bytes = _historical_binding_file(
        repo_root,
        consumer_path,
        content_ref=content_ref,
        content_cache=content_cache,
        tracked_paths=tracked_paths,
        label="consumer_path",
    )
    if hashlib.sha256(consumer_bytes).hexdigest() != consumer_sha256:
        raise _historical_binding_error(f"evaluated consumer digest mismatch for {consumer_path}")
    if _blob_sha1(consumer_bytes) != consumer_blob_sha1:
        raise _historical_binding_error(f"evaluated consumer blob mismatch for {consumer_path}")
    try:
        consumer_value = _load_document(
            repo_root / consumer_path,
            raw=consumer_bytes,
            reject_duplicate_json_keys=True,
        )
        pointer_value = _json_pointer_get(consumer_value, locator)
    except DuplicateJSONKeyError as exc:
        raise _historical_binding_error(
            f"current consumer contains duplicate JSON object key: {exc}"
        ) from exc
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:
        raise _historical_binding_error(
            f"reference_locator does not resolve in {consumer_path}: {locator}"
        ) from exc
    if not isinstance(pointer_value, Mapping):
        raise _historical_binding_error(
            f"reference_locator must identify a path/checksum mapping in {consumer_path}"
        )
    pointer_digest = pointer_value.get("sha256")
    if pointer_value.get("path") != reference_path or not isinstance(pointer_digest, str):
        raise _historical_binding_error(
            f"reference_locator does not identify the declared reference in {consumer_path}"
        )
    if pointer_digest.lower() != declared_sha256:
        raise _historical_binding_error(
            f"reference_locator does not identify the declared reference in {consumer_path}"
        )
    occurrences = _historical_binding_occurrences(consumer_value, reference_path, declared_sha256)
    if len(occurrences) != 1 or occurrences[0] is not pointer_value:
        raise _historical_binding_error(
            f"reference must occur exactly once at {consumer_path}{locator}"
        )
    key = (consumer_path, reference_path, declared_sha256)
    return key, {
        "consumer_path": consumer_path,
        "consumer_sha256": consumer_sha256,
        "consumer_blob_sha1": consumer_blob_sha1,
        "reference_locator": locator,
        "reference_path": reference_path,
        "declared_sha256": declared_sha256,
        "producer_commit": producer_commit,
        "producer_tree": producer_tree,
        "producer_reference_sha256": producer_reference_sha256,
        "producer_reference_blob_sha1": producer_reference_blob_sha1,
        "producer_consumer_sha256": producer_consumer_sha256,
        "producer_consumer_blob_sha1": producer_consumer_blob_sha1,
        "parent_commit": parent_commit,
        "parent_reference_sha256": parent_reference_sha256,
        "reviewed_ancestry_anchor": reviewed_ancestry_anchor,
        "source": "historical_git_tree",
    }


def _load_historical_bindings(  # noqa: C901
    repo_root: Path,
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
    authority_ref: str = "HEAD",
) -> tuple[dict[tuple[str, str, str], dict[str, str]], dict[str, Any]]:
    """Load and verify the tracked two-record historical binding manifest.

    Returns:
        The validated bindings keyed by consumer/reference identity and a report.
    """
    manifest_path = HISTORICAL_BINDING_MANIFEST.as_posix()
    if not _is_tracked(repo_root, manifest_path, content_ref, content_cache, tracked_paths):
        return {}, {
            "manifest_path": manifest_path,
            "status": "absent",
            "validated": 0,
            "applied": [],
        }
    raw = _historical_binding_file(
        repo_root,
        manifest_path,
        content_ref=content_ref,
        content_cache=content_cache,
        tracked_paths=tracked_paths,
        label="historical binding manifest",
    )
    try:
        value = _load_document(
            repo_root / manifest_path,
            raw=raw,
            reject_duplicate_json_keys=True,
        )
    except DuplicateJSONKeyError as exc:
        raise _historical_binding_error(
            f"manifest contains duplicate JSON object key: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _historical_binding_error("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise _historical_binding_error("manifest root must be an object")
    unknown = set(value) - HISTORICAL_BINDING_TOP_FIELDS
    missing = HISTORICAL_BINDING_TOP_FIELDS - set(value)
    if unknown:
        raise _historical_binding_error(f"manifest contains unknown fields: {sorted(unknown)}")
    if missing:
        raise _historical_binding_error(f"manifest is missing fields: {sorted(missing)}")
    if (
        value.get("$schema")
        != "scripts/validation/evidence_registry_historical_bindings.v1.schema.json"
    ):
        raise _historical_binding_error("manifest $schema does not name the tracked schema")
    if value.get("schema_version") != HISTORICAL_BINDING_SCHEMA:
        raise _historical_binding_error(f"schema_version must be {HISTORICAL_BINDING_SCHEMA}")
    repository = value.get("repository")
    if repository != HISTORICAL_BINDING_REPOSITORY:
        raise _historical_binding_error(
            "manifest repository does not match the repository identity"
        )
    reviewed_ancestry_anchor = value.get("reviewed_ancestry_anchor")
    if not isinstance(reviewed_ancestry_anchor, str) or not FULL_SHA1_RE.fullmatch(
        reviewed_ancestry_anchor
    ):
        raise _historical_binding_error("reviewed_ancestry_anchor must be a full commit SHA-1")
    bindings = value.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise _historical_binding_error("bindings must contain exactly two records")
    validated: dict[tuple[str, str, str], dict[str, str]] = {}
    for index, binding in enumerate(bindings):
        key, checked = _validate_historical_binding_entry(
            repo_root,
            binding,
            index,
            repository,
            reviewed_ancestry_anchor,
            content_ref=content_ref,
            content_cache=content_cache,
            tracked_paths=tracked_paths,
            authority_ref=authority_ref,
        )
        if key in validated:
            raise _historical_binding_error(f"duplicate or conflicting binding for {key[0]}")
        validated[key] = checked
    return validated, {
        "manifest_path": manifest_path,
        "schema_version": HISTORICAL_BINDING_SCHEMA,
        "repository": repository,
        "reviewed_ancestry_anchor": reviewed_ancestry_anchor.lower(),
        "status": "validated",
        "validated": len(validated),
        "applied": [],
    }


load_historical_bindings = _load_historical_bindings
