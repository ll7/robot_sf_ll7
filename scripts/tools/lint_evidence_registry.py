#!/usr/bin/env python3
"""Report traceability gaps in committed evidence-registry campaign records.

The default mode is intentionally non-blocking so existing provenance gaps can be
measured before this checker becomes a CI gate. Use ``--strict`` to make findings
fail the command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Sequence


COMMIT_RE = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40}(?![0-9a-fA-F])")
FULL_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SYNTHETIC_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40,}[^0-9a-fA-F]")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PROJECTION_SCHEMA = "evidence_registry_projection.v1"
PROJECTION_MODE = "frozen_squash_base"
MARKDOWN_CAMPAIGN_RE = re.compile(r"\bcampaign_id\s*[:=]\s*`?([A-Za-z0-9_.-]+)`?")
COMMIT_KEYS = {"commit", "producing_commit", "source_commit", "git_commit"}
CONFIG_PATH_KEYS = {
    "config",
    "config_path",
    "producing_config",
    "source_config",
    "training_config",
}
ARTIFACT_PATH_KEYS = {
    "artifact_path",
    "artifact_uri",
    "file",
    "filename",
    "path",
    "source_path",
}
CH7_PORTFOLIO_COMPANION_BINDING = Path(
    "docs/context/evidence/issue_7047_ch7_portfolio_companion_binding.v1.json"
)
CH7_PORTFOLIO_TARGET_PATH = "configs/analysis/ch7_worked_example_portfolio.v1.yaml"
CH7_PORTFOLIO_TARGET_SHA256 = "2fe0723bbb67eb18d25944b6933575b7c7b5a31836062c0bd56540fe4e3923ec"
CH7_PORTFOLIO_ALLOWED_BINDING_POINTERS = {
    (
        "docs/context/evidence/issue_6792_ch7_evidence_package_v1/manifest.json",
        "hash_without_artifact_path",
        "sha256 lacks an adjacent artifact path",
    ): {
        "document_digest_pointer": "/inputs/portfolio_config/sha256",
        "document_json_pointer": "/inputs/portfolio_config/sha256",
        "document_json_value": CH7_PORTFOLIO_TARGET_SHA256,
    },
    (
        "docs/context/evidence/issue_6792_ch7_evidence_package_v1/publication/"
        "materialization_overlay.json",
        "uncommitted_artifact_missing_location",
        "ch7_worked_example_portfolio.v1.yaml is not tracked and lacks an explicit location marker",
    ): {
        "document_digest_pointer": "/source_portfolio/sha256",
        "document_json_pointer": "/source_portfolio/path",
        "document_json_value": "ch7_worked_example_portfolio.v1.yaml",
    },
}
CH7_PORTFOLIO_ALLOWED_BINDINGS = frozenset(CH7_PORTFOLIO_ALLOWED_BINDING_POINTERS)


@dataclass(frozen=True)
class _DocumentRecord:
    """Parsed registry data needed for bundle-level campaign checks."""

    path: Path
    campaign_ids: list[str]
    config_paths: list[str]
    config_hashes: list[str]
    commits: list[str]
    findings: list[dict[str, str]]


def _issue(path: Path, code: str, message: str) -> dict[str, str]:
    """Return one stable, machine-readable lint finding."""
    return {"path": path.as_posix(), "code": code, "message": message}


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


class ShallowRepositoryError(RuntimeError):
    """Raised when commit reachability cannot be classified completely."""


class CompanionBindingError(RuntimeError):
    """Raised when a companion binding cannot be trusted to resolve linter findings."""


class ProjectionValidationError(RuntimeError):
    """Raised when an explicit frozen-base projection cannot be evaluated safely."""


def _require_full_history(repo_root: Path) -> None:
    """Fail before classifying commits when Git history is shallow."""
    result = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("evidence-registry linter could not determine Git repository depth")
    if result.stdout.strip().lower() == "true":
        raise ShallowRepositoryError(
            "evidence-registry commit reachability requires a full-history Git "
            "repository; shallow repository detected. Run `git fetch --unshallow` "
            "(or use a full-history checkout) before rerunning."
        )


def _commit_is_reachable(
    repo_root: Path,
    commit: str,
    authority_ref: str = "HEAD",
    cache: dict[tuple[str, str], bool] | None = None,
    authority_commits: set[str] | None = None,
) -> bool:
    """Return whether ``commit`` exists and belongs to the checked-out history.

    Git object stores may also contain commits fetched through unrelated branches
    or tags. Treating raw object presence as provenance resolution makes the
    evidence baseline depend on which refs happened to be fetched. Reachability
    from the explicit authority ref gives full-history checkouts one stable
    authority while still rejecting missing commit objects.
    """
    key = (commit, authority_ref)
    if cache is not None and key in cache:
        return cache[key]
    if authority_commits is not None:
        reachable = commit.lower() in authority_commits
    else:
        reachable = _git_succeeds(
            repo_root, "cat-file", "-e", f"{commit}^{{commit}}"
        ) and _git_succeeds(repo_root, "merge-base", "--is-ancestor", commit, authority_ref)
    if cache is not None:
        cache[key] = reachable
    return reachable


def _commit_is_head_reachable(repo_root: Path, commit: str) -> bool:
    """Return whether ``commit`` is reachable from the ordinary checked-out HEAD."""
    return _commit_is_reachable(repo_root, commit)


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


def _config_hash_matches(
    repo_root: Path,
    commit: str,
    path: str,
    declared_hashes: set[str],
    cache: dict[tuple[str, str, tuple[str, ...]], bool] | None = None,
) -> bool:
    """Return whether one committed config blob matches a declared SHA-256 value."""
    key = (commit, path, tuple(sorted(declared_hashes)))
    if cache is not None and key in cache:
        return cache[key]
    blob = _git_bytes(repo_root, "show", f"{commit}:{path}")
    matches = blob is not None and hashlib.sha256(blob).hexdigest() in declared_hashes
    if cache is not None:
        cache[key] = matches
    return matches


def _is_tracked(
    repo_root: Path,
    repo_path: str,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> bool:
    """Return whether a normalized path exists in the current index or an immutable tree."""
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
    """Read repository bytes from the working tree or an immutable commit tree."""
    if content_ref is not None:
        if content_cache is not None and repo_path in content_cache:
            return content_cache[repo_path]
        return _git_bytes(repo_root, "show", f"{content_ref}:{repo_path}")
    try:
        return (repo_root / repo_path).read_bytes()
    except OSError:
        return None


def _resolve_repo_path(repo_root: Path, value: str) -> tuple[str, Path] | None:
    """Normalize a repository-root-relative path, rejecting URLs and escapes.

    Manifest path fields are never relative to the manifest or its evidence bundle.
    A sibling artifact must therefore use its full repository path, such as
    ``docs/context/evidence/<bundle>/README.md``.
    """
    if "://" in value or value.startswith("urn:"):
        return None
    candidate = (repo_root / value).resolve()
    try:
        normalized = candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return normalized.as_posix(), candidate


def _load_document(path: Path, raw: bytes | None = None) -> Any:
    """Load supported structured registry files, returning text for Markdown/CSV."""
    if raw is None:
        raw = path.read_bytes()
    decoded = raw.decode("utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(decoded)
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(decoded)
    if suffix == ".csv":
        return list(csv.DictReader(decoded.splitlines()))
    return decoded


def _json_pointer_get(value: Any, pointer: str) -> Any:
    """Resolve one RFC 6901 JSON pointer from a loaded document."""
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer {pointer!r}")
    current = value
    for raw_part in pointer.split("/")[1:]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if part not in current:
                raise KeyError(pointer)
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise KeyError(pointer) from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise KeyError(pointer) from exc
        else:
            raise KeyError(pointer)
    return current


def _ch7_binding_error(message: str) -> CompanionBindingError:
    """Return a consistently prefixed companion-binding error."""
    return CompanionBindingError(f"{CH7_PORTFOLIO_COMPANION_BINDING}: {message}")


def _load_ch7_companion_payload(path: Path, *, raw: bytes | None = None) -> Mapping[str, Any]:
    """Load and parse the #7047 companion payload."""
    try:
        if raw is None:
            raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ch7_binding_error("could not read JSON") from exc
    if not isinstance(value, Mapping):
        raise _ch7_binding_error("root must be an object")
    return value


def _validate_ch7_companion_header(value: Mapping[str, Any]) -> list[Any]:
    """Validate schema-level companion fields and return the binding list."""
    if value.get("schema_version") != "evidence_registry_companion_binding.v1":
        raise _ch7_binding_error("schema_version must be evidence_registry_companion_binding.v1")
    if value.get("issue") != 7047:
        raise _ch7_binding_error("issue must be 7047")
    if value.get("target_path") != CH7_PORTFOLIO_TARGET_PATH:
        raise _ch7_binding_error(f"target_path must be {CH7_PORTFOLIO_TARGET_PATH}")
    if value.get("target_sha256") != CH7_PORTFOLIO_TARGET_SHA256:
        raise _ch7_binding_error("target_sha256 must match the approved digest")
    bindings = value.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != len(CH7_PORTFOLIO_ALLOWED_BINDINGS):
        raise _ch7_binding_error(
            f"bindings must contain exactly {len(CH7_PORTFOLIO_ALLOWED_BINDINGS)} entries"
        )
    return bindings


def _validate_ch7_target(
    repo_root: Path,
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> None:
    """Validate that the approved source config is tracked and byte-matching."""
    target = _resolve_repo_path(repo_root, CH7_PORTFOLIO_TARGET_PATH)
    if target is None or not _is_tracked(
        repo_root, target[0], content_ref, content_cache, tracked_paths
    ):
        raise _ch7_binding_error(f"target_path is not tracked: {CH7_PORTFOLIO_TARGET_PATH}")
    target_bytes = _repository_file_bytes(repo_root, target[0], content_ref, content_cache)
    if target_bytes is None:
        raise _ch7_binding_error(f"target_path cannot be read: {CH7_PORTFOLIO_TARGET_PATH}")
    actual_target_sha256 = hashlib.sha256(target_bytes).hexdigest()
    if actual_target_sha256 != CH7_PORTFOLIO_TARGET_SHA256:
        raise _ch7_binding_error("target_path digest mismatch")


def _ch7_resolution_key(binding: Mapping[str, Any], index: int) -> tuple[str, str, str]:
    """Return and validate the exact finding key named by one binding entry."""
    document_path = binding.get("document_path")
    code = binding.get("finding_code")
    message = binding.get("finding_message")
    resolution_key = (document_path, code, message)
    if not all(isinstance(item, str) and item for item in resolution_key):
        raise _ch7_binding_error(
            f"bindings[{index}] requires document_path, finding_code, and finding_message"
        )
    if resolution_key not in CH7_PORTFOLIO_ALLOWED_BINDINGS:
        raise _ch7_binding_error(f"bindings[{index}] is not an allowed #7047 finding")
    return resolution_key


def _validate_ch7_document_digest(
    repo_root: Path,
    binding: Mapping[str, Any],
    document_path: str,
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> Path:
    """Validate that the bound package document is tracked and byte-matching."""
    document = _resolve_repo_path(repo_root, document_path)
    if document is None or not _is_tracked(
        repo_root, document[0], content_ref, content_cache, tracked_paths
    ):
        raise _ch7_binding_error(f"document_path is not tracked: {document_path}")
    document_bytes = _repository_file_bytes(repo_root, document[0], content_ref, content_cache)
    if document_bytes is None:
        raise _ch7_binding_error(f"document_path cannot be read: {document_path}")
    document_sha256 = binding.get("document_sha256")
    if (
        not isinstance(document_sha256, str)
        or not SHA256_RE.fullmatch(document_sha256)
        or hashlib.sha256(document_bytes).hexdigest() != document_sha256
    ):
        raise _ch7_binding_error(f"document digest mismatch for {document_path}")
    return document[1]


def _validate_ch7_document_pointers(
    repo_root: Path,
    document_path: str,
    document: Path,
    binding: Mapping[str, Any],
    index: int,
    expected: Mapping[str, str],
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
) -> None:
    """Validate the bound package JSON pointers and digest value."""
    pointer = binding.get("document_json_pointer")
    digest_pointer = binding.get("document_digest_pointer")
    if not isinstance(pointer, str) or not isinstance(digest_pointer, str):
        raise _ch7_binding_error(f"bindings[{index}] requires JSON pointers")
    try:
        normalized_path = document.relative_to(repo_root).as_posix()
        document_bytes = _repository_file_bytes(
            repo_root, normalized_path, content_ref, content_cache
        )
        if document_bytes is None:
            raise KeyError(document_path)
        document_value = _load_document(document, raw=document_bytes)
        actual_value = _json_pointer_get(document_value, pointer)
        actual_digest_value = _json_pointer_get(document_value, digest_pointer)
    except (KeyError, ValueError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise _ch7_binding_error(f"JSON pointer does not resolve for {document_path}") from exc
    if actual_value != binding.get("document_json_value"):
        raise _ch7_binding_error(f"pointer value mismatch for {document_path}")
    if actual_digest_value != CH7_PORTFOLIO_TARGET_SHA256:
        raise _ch7_binding_error(f"digest pointer mismatch for {document_path}")
    for field, expected_value in expected.items():
        if binding.get(field) != expected_value:
            raise _ch7_binding_error(f"unexpected {field} for {document_path}")


def _validate_ch7_binding_entry(
    repo_root: Path,
    binding: Any,
    index: int,
    resolutions: set[tuple[str, str, str]],
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> None:
    """Validate one #7047 companion binding entry."""
    if not isinstance(binding, Mapping):
        raise _ch7_binding_error(f"bindings[{index}] must be an object")
    resolution_key = _ch7_resolution_key(binding, index)
    if resolution_key in resolutions:
        document_path, code, _message = resolution_key
        raise _ch7_binding_error(f"duplicate binding for {document_path}/{code}")
    document_path = resolution_key[0]
    document = _validate_ch7_document_digest(
        repo_root,
        binding,
        document_path,
        content_ref=content_ref,
        content_cache=content_cache,
        tracked_paths=tracked_paths,
    )
    expected = CH7_PORTFOLIO_ALLOWED_BINDING_POINTERS[resolution_key]
    _validate_ch7_document_pointers(
        repo_root,
        document_path,
        document,
        binding,
        index,
        expected,
        content_ref=content_ref,
        content_cache=content_cache,
    )
    resolutions.add(resolution_key)


def _load_ch7_portfolio_companion_resolutions(
    repo_root: Path,
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> frozenset[tuple[str, str, str]]:
    """Load the canonical #7047 companion binding and return exact findings it resolves."""
    path = repo_root / CH7_PORTFOLIO_COMPANION_BINDING
    binding_bytes = _repository_file_bytes(
        repo_root,
        CH7_PORTFOLIO_COMPANION_BINDING.as_posix(),
        content_ref,
        content_cache,
    )
    if binding_bytes is None:
        return frozenset()
    value = _load_ch7_companion_payload(path, raw=binding_bytes)
    bindings = _validate_ch7_companion_header(value)
    _validate_ch7_target(
        repo_root,
        content_ref=content_ref,
        content_cache=content_cache,
        tracked_paths=tracked_paths,
    )
    resolutions: set[tuple[str, str, str]] = set()
    for index, binding in enumerate(bindings):
        _validate_ch7_binding_entry(
            repo_root,
            binding,
            index,
            resolutions,
            content_ref=content_ref,
            content_cache=content_cache,
            tracked_paths=tracked_paths,
        )
    if resolutions != CH7_PORTFOLIO_ALLOWED_BINDINGS:
        raise _ch7_binding_error("bindings do not cover the exact #7047 set")
    return frozenset(resolutions)


def _apply_companion_resolutions(
    findings: Sequence[dict[str, str]], resolutions: frozenset[tuple[str, str, str]]
) -> list[dict[str, str]]:
    """Remove only findings proven by a validated companion binding."""
    return [
        item
        for item in findings
        if (item["path"], item["code"], item["message"]) not in resolutions
    ]


def _load_strict_exclusion_policy(path: Path) -> frozenset[str]:
    """Load a strict-mode exclusion policy or fail before a gate can pass."""
    if not path.is_file():
        raise FileNotFoundError(f"Strict exclusion policy file not found: {path}")
    try:
        policy = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Could not parse strict exclusion policy {path}: {exc}") from exc
    if not isinstance(policy, Mapping):
        raise ValueError(
            f"Strict exclusion policy {path} must be a YAML mapping, got {type(policy).__name__}."
        )
    codes = policy.get("excluded_codes")
    if not isinstance(codes, list):
        raise ValueError(f"Strict exclusion policy {path} must contain an 'excluded_codes' list.")

    parsed: list[str] = []
    for item in codes:
        if isinstance(item, str) and item.strip():
            parsed.append(item)
        elif (
            isinstance(item, Mapping) and isinstance(item.get("code"), str) and item["code"].strip()
        ):
            parsed.append(item["code"])
        else:
            raise ValueError(
                f"Every strict exclusion policy entry in {path} must be a non-empty code string "
                "or mapping with a non-empty 'code' string."
            )
    return frozenset(parsed)


def _iter_mappings(
    value: Any, ancestors: tuple[Mapping[str, Any], ...] = ()
) -> Iterable[tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]]:
    """Yield mappings with their parent mappings for inherited path/location metadata."""
    if isinstance(value, Mapping):
        yield value, ancestors
        for child in value.values():
            yield from _iter_mappings(child, (*ancestors, value))
    elif isinstance(value, list):
        for child in value:
            yield from _iter_mappings(child, ancestors)


def _string_values(mapping: Mapping[str, Any], keys: set[str]) -> list[str]:
    """Return non-empty strings from case-insensitive field names."""
    return [
        value.strip()
        for key, value in mapping.items()
        if isinstance(key, str) and key.lower() in keys and isinstance(value, str) and value.strip()
    ]


def _campaign_ids(value: Any) -> list[str]:
    """Extract explicit campaign identifiers without treating README vocabulary as data."""
    if isinstance(value, str):
        return MARKDOWN_CAMPAIGN_RE.findall(value)
    values: list[str] = []
    for mapping, _ancestors in _iter_mappings(value):
        campaign_id = mapping.get("campaign_id")
        if isinstance(campaign_id, str) and campaign_id.strip():
            values.append(campaign_id.strip())
    return values


def _file_metadata(value: Any) -> tuple[list[str], list[str], list[str]]:
    """Return config paths, config checksums, and full commit references in a file."""
    if isinstance(value, str):
        return [], [], COMMIT_RE.findall(value)
    config_paths: list[str] = []
    config_hashes: list[str] = []
    commits: list[str] = []
    for mapping, _ancestors in _iter_mappings(value):
        config_paths.extend(_string_values(mapping, CONFIG_PATH_KEYS))
        for key, item in mapping.items():
            if not isinstance(key, str):
                continue
            lowered = key.lower()
            if lowered == "config_sha256" and isinstance(item, str):
                config_hashes.append(item.strip())
            if isinstance(item, str):
                commits.extend(COMMIT_RE.findall(item))
    return config_paths, config_hashes, commits


def _synthetic_commit_findings(display_path: Path, value: Any) -> list[dict[str, str]]:
    """Reject commit fields whose value is a 40-hex SHA plus extra characters.

    A synthetic commit (e.g. ``d4e17b...-reconciled-5483``) cannot be resolved
    by ``git checkout`` and should use either a real 40-char SHA or an explicit
    ``missing:<reason>`` placeholder instead. See issue #5558.
    """
    findings: list[dict[str, str]] = []
    if isinstance(value, str):
        return findings
    for mapping, _ancestors in _iter_mappings(value):
        for key, item in mapping.items():
            if not isinstance(key, str) or key.lower() not in COMMIT_KEYS:
                continue
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if FULL_SHA1_RE.fullmatch(stripped) or SHA256_RE.fullmatch(stripped):
                continue
            if SYNTHETIC_COMMIT_RE.match(stripped):
                findings.append(
                    _issue(
                        display_path,
                        "synthetic_commit",
                        f"commit field '{key}' value {stripped!r} is a 40-hex SHA "
                        "with extra characters; use a real SHA or missing:<reason>",
                    )
                )
    return findings


def _artifact_path(
    mapping: Mapping[str, Any], ancestors: tuple[Mapping[str, Any], ...]
) -> str | None:
    """Find a neighboring artifact path, including ``reports_dir`` filename manifests."""
    paths = _string_values(mapping, ARTIFACT_PATH_KEYS)
    for value in paths:
        if value and not value.startswith("configs/"):
            return value
    filename = next(
        (
            key
            for parent in reversed(ancestors)
            for key, child in parent.items()
            if child is mapping and isinstance(key, str) and Path(key).suffix
        ),
        None,
    )
    if filename is not None:
        reports_dir = next(
            (
                value
                for parent in reversed(ancestors)
                for value in _string_values(parent, {"reports_dir", "artifact_dir", "directory"})
            ),
            None,
        )
        return str(Path(reports_dir) / filename) if reports_dir else filename
    return None


def _has_location(mapping: Mapping[str, Any], ancestors: tuple[Mapping[str, Any], ...]) -> bool:
    """Return whether an uncommitted artifact has an explicit location marker."""
    return any(
        isinstance(value, str) and value.strip()
        for candidate in (*ancestors, mapping)
        for key, value in candidate.items()
        if isinstance(key, str) and (key.lower() == "location" or key.lower().endswith("_location"))
    )


def _artifact_hash_finding(  # noqa: PLR0913 - candidate-tree inputs stay explicit.
    repo_root: Path,
    display_path: Path,
    key: str,
    declared_hash: str,
    mapping: Mapping[str, Any],
    ancestors: tuple[Mapping[str, Any], ...],
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> dict[str, str] | None:
    """Check one artifact hash declaration, returning its finding when invalid."""
    if not SHA256_RE.fullmatch(declared_hash):
        return _issue(display_path, "invalid_sha256", f"{key} is not a 64-hex SHA-256")
    artifact_path = _artifact_path(mapping, ancestors)
    if artifact_path is None:
        return _issue(
            display_path, "hash_without_artifact_path", f"{key} lacks an adjacent artifact path"
        )
    resolved = _resolve_repo_path(repo_root, artifact_path)
    if resolved is None or not _is_tracked(
        repo_root, resolved[0], content_ref, content_cache, tracked_paths
    ):
        if not _has_location(mapping, ancestors):
            return _issue(
                display_path,
                "uncommitted_artifact_missing_location",
                f"{artifact_path} is not tracked and lacks an explicit location marker",
            )
        return None
    artifact_bytes = _repository_file_bytes(repo_root, resolved[0], content_ref, content_cache)
    if artifact_bytes is None:
        return _issue(
            display_path,
            "artifact_unreadable",
            f"{resolved[0]} cannot be read from the evaluated evidence tree",
        )
    actual_hash = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_hash != declared_hash.lower():
        return _issue(
            display_path,
            "artifact_hash_mismatch",
            f"{artifact_path} SHA-256 does not match the tracked file",
        )
    return None


def _artifact_findings(
    repo_root: Path,
    display_path: Path,
    value: Any,
    *,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> list[dict[str, str]]:
    """Validate artifact checksum declarations against tracked artifact paths."""
    if isinstance(value, str):
        return []
    findings: list[dict[str, str]] = []
    for mapping, ancestors in _iter_mappings(value):
        for key, declared_hash in mapping.items():
            if not isinstance(key, str) or key.lower() not in {"sha256", "source_sha256"}:
                continue
            if not isinstance(declared_hash, str):
                continue
            finding = _artifact_hash_finding(
                repo_root,
                display_path,
                key,
                declared_hash,
                mapping,
                ancestors,
                content_ref=content_ref,
                content_cache=content_cache,
                tracked_paths=tracked_paths,
            )
            if finding:
                findings.append(finding)
    return findings


def _commit_path_exists(
    repo_root: Path,
    commit: str,
    path: str,
    cache: dict[tuple[str, str], bool] | None = None,
) -> bool:
    """Return whether one path exists at one commit, with per-run memoization."""
    key = (commit, path)
    if cache is not None and key in cache:
        return cache[key]
    exists = _git_succeeds(repo_root, "cat-file", "-e", f"{commit}:{path}")
    if cache is not None:
        cache[key] = exists
    return exists


def _campaign_metadata_findings(  # noqa: C901, PLR0913 - rule-to-finding mapping is intentionally linear
    repo_root: Path,
    display_path: Path,
    campaign_ids: Sequence[str],
    config_paths: Sequence[str],
    config_hashes: Sequence[str],
    commits: Sequence[str],
    *,
    reachability_ref: str = "HEAD",
    reachability_cache: dict[tuple[str, str], bool] | None = None,
    authority_commits: set[str] | None = None,
    path_cache: dict[tuple[str, str], bool] | None = None,
    config_hash_cache: dict[tuple[str, str, tuple[str, ...]], bool] | None = None,
) -> list[dict[str, str]]:
    """Validate campaign-required identifiers and config existence at declared commits."""
    findings: list[dict[str, str]] = []
    if not campaign_ids:
        return findings
    if not config_paths:
        findings.append(
            _issue(
                display_path, "missing_config_path", "campaign_id requires a producing config path"
            )
        )
    if not config_hashes:
        findings.append(
            _issue(display_path, "missing_config_sha256", "campaign_id requires config_sha256")
        )
    elif any(not SHA256_RE.fullmatch(item) for item in config_hashes):
        findings.append(
            _issue(display_path, "invalid_config_sha256", "config_sha256 is not a 64-hex SHA-256")
        )
    if not commits:
        findings.append(
            _issue(display_path, "missing_commit", "campaign_id requires a full producing commit")
        )
    resolved_commits = [
        commit
        for commit in set(commits)
        if _commit_is_reachable(
            repo_root,
            commit,
            reachability_ref,
            reachability_cache,
            authority_commits,
        )
    ]
    declared_config_hashes = {item.lower() for item in config_hashes if SHA256_RE.fullmatch(item)}
    for config_path in sorted(set(config_paths)):
        normalized = _resolve_repo_path(repo_root, config_path)
        if normalized is None:
            findings.append(
                _issue(
                    display_path,
                    "invalid_config_path",
                    f"config path {config_path!r} is not repository-relative",
                )
            )
        elif not any(
            _commit_path_exists(repo_root, commit, normalized[0], path_cache)
            for commit in resolved_commits
        ):
            findings.append(
                _issue(
                    display_path,
                    "config_missing_at_commit",
                    f"config {normalized[0]} does not exist at a declared commit",
                )
            )
        elif declared_config_hashes:
            config_hash_matches = any(
                _config_hash_matches(
                    repo_root,
                    commit,
                    normalized[0],
                    declared_config_hashes,
                    config_hash_cache,
                )
                for commit in resolved_commits
            )
            if not config_hash_matches:
                findings.append(
                    _issue(
                        display_path,
                        "config_sha256_mismatch",
                        f"config_sha256 does not match {normalized[0]} at a declared commit",
                    )
                )
    return findings


def _lint_document(
    repo_root: Path,
    path: Path,
    *,
    raw: bytes | None = None,
    display_path: Path | None = None,
    content_ref: str | None = None,
    content_cache: Mapping[str, bytes] | None = None,
    tracked_paths: set[str] | None = None,
) -> _DocumentRecord:
    """Read one document and retain only document-local integrity findings."""
    display_path = display_path or path.relative_to(repo_root)
    try:
        value = _load_document(path, raw=raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return _DocumentRecord(
            path, [], [], [], [], [_issue(display_path, "unreadable_document", str(exc))]
        )
    campaign_ids = _campaign_ids(value)
    config_paths, config_hashes, commits = _file_metadata(value)
    local_findings = _artifact_findings(
        repo_root,
        display_path,
        value,
        content_ref=content_ref,
        content_cache=content_cache,
        tracked_paths=tracked_paths,
    )
    local_findings.extend(_synthetic_commit_findings(display_path, value))
    return _DocumentRecord(
        path,
        campaign_ids,
        config_paths,
        config_hashes,
        commits,
        local_findings,
    )


def _bundle_path(registry_root: Path, path: Path) -> Path:
    """Return an evidence bundle root, treating root-level files as independent bundles."""
    relative = path.relative_to(registry_root)
    return path if len(relative.parts) == 1 else registry_root / relative.parts[0]


def _canonical_projection_commit(repo_root: Path, value: str, label: str) -> str:
    """Resolve and return one explicitly supplied full commit identity."""
    if not isinstance(value, str) or not FULL_SHA1_RE.fullmatch(value):
        raise ProjectionValidationError(
            f"frozen-base projection {label} must be an exact 40-hex commit SHA; got {value!r}"
        )
    resolved = _git_bytes(repo_root, "rev-parse", "--verify", f"{value}^{{commit}}")
    if resolved is None:
        raise ProjectionValidationError(
            f"frozen-base projection {label} {value} is unavailable as a commit object; "
            "fetch the complete candidate/base history and retry"
        )
    try:
        canonical = resolved.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ProjectionValidationError(
            f"frozen-base projection {label} resolution was not valid ASCII"
        ) from exc
    if not FULL_SHA1_RE.fullmatch(canonical) or canonical.lower() != value.lower():
        raise ProjectionValidationError(
            f"frozen-base projection {label} identity did not bind exactly to {value}"
        )
    return canonical.lower()


def _prepare_projection(
    repo_root: Path,
    candidate_head: str | None,
    frozen_base: str | None,
) -> dict[str, str] | None:
    """Validate an optional candidate/base pair and return canonical identities."""
    if candidate_head is None and frozen_base is None:
        return None
    if not candidate_head or not frozen_base:
        raise ProjectionValidationError(
            "frozen-base projection requires both --candidate-head and --frozen-base; "
            "provide neither to retain ordinary HEAD mode"
        )
    candidate = _canonical_projection_commit(repo_root, candidate_head, "candidate head")
    base = _canonical_projection_commit(repo_root, frozen_base, "frozen base")
    current_head_bytes = _git_bytes(repo_root, "rev-parse", "--verify", "HEAD^{commit}")
    if current_head_bytes is None:
        raise ProjectionValidationError(
            "frozen-base projection could not resolve the checked-out HEAD"
        )
    current_head = current_head_bytes.decode("ascii", errors="replace").strip().lower()
    if current_head != candidate:
        raise ProjectionValidationError(
            "frozen-base projection candidate head is stale: checked-out HEAD is "
            f"{current_head or '<unavailable>'}, but candidate head is {candidate}"
        )
    if not _git_succeeds(repo_root, "merge-base", "--is-ancestor", base, candidate):
        raise ProjectionValidationError(
            f"frozen-base projection base {base} is not an ancestor of candidate head {candidate}; "
            "refresh the base/head identities"
        )
    # A non-shallow marker alone is not enough: alternates/promisor repositories
    # can still omit an object from an apparently complete checkout. Validate the
    # complete candidate/base ancestry before reading any evidence metadata.
    for label, ref in (("candidate head", candidate), ("frozen base", base)):
        if not _git_succeeds(repo_root, "rev-list", "--objects", "--missing=error", ref):
            raise ProjectionValidationError(
                f"frozen-base projection {label} history is incomplete or has missing Git objects "
                f"at {ref}; fetch a complete history before retrying"
            )
    return {"candidate_head": candidate, "frozen_base": base}


def _candidate_evidence_tree(  # noqa: C901, PLR0912, PLR0915 - immutable tree contract.
    repo_root: Path, registry_root: Path, candidate_head: str
) -> dict[str, Any]:
    """Return a complete, immutable manifest of the candidate evidence tree."""
    try:
        registry_relative = registry_root.resolve().relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise ProjectionValidationError(
            "frozen-base projection registry root must be inside the repository"
        ) from exc
    tree_bytes = _git_bytes(repo_root, "rev-parse", f"{candidate_head}:{registry_relative}")
    if tree_bytes is None:
        raise ProjectionValidationError(
            f"candidate evidence tree {registry_relative} is missing at {candidate_head}"
        )
    tree_sha = tree_bytes.decode("ascii", errors="replace").strip().lower()
    if not FULL_SHA1_RE.fullmatch(tree_sha) or not _git_succeeds(
        repo_root, "cat-file", "-t", tree_sha
    ):
        raise ProjectionValidationError(
            f"candidate evidence tree identity is unavailable at {candidate_head}"
        )
    tree_type = _git_bytes(repo_root, "cat-file", "-t", tree_sha)
    if tree_type is None or tree_type.strip() != b"tree":
        raise ProjectionValidationError(
            f"candidate evidence path {registry_relative} is not a complete Git tree"
        )
    listing = _git_bytes(
        repo_root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        candidate_head,
        "--",
        registry_relative,
    )
    if listing is None:
        raise ProjectionValidationError(
            f"candidate evidence tree listing is unavailable at {candidate_head}"
        )
    prefix = registry_relative + "/"
    files: list[str] = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, raw_path = entry.split(b"\t", 1)
            mode, object_type, _object_sha = metadata.split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ProjectionValidationError(
                "candidate evidence tree contains an invalid Git tree entry"
            ) from exc
        if mode not in {b"100644", b"100755"} or object_type != b"blob":
            raise ProjectionValidationError(
                f"candidate evidence tree contains a non-regular entry: {path!r}"
            )
        if not path.startswith(prefix) or path == prefix:
            raise ProjectionValidationError(
                f"candidate evidence tree returned an invalid path: {path!r}"
            )
        files.append(path[len(prefix) :])
    if len(files) != len(set(files)):
        raise ProjectionValidationError("candidate evidence tree contains duplicate paths")
    files.sort()
    repository_files = {f"{registry_relative}/{relative}" for relative in files}
    archive = _git_bytes(
        repo_root,
        "archive",
        "--format=tar",
        candidate_head,
        "--",
        registry_relative,
    )
    if archive is None:
        raise ProjectionValidationError(
            f"candidate evidence tree contents are unavailable at {candidate_head}"
        )
    content_cache: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as archive_file:
            for member in archive_file.getmembers():
                if not member.isfile() or member.name not in repository_files:
                    continue
                extracted = archive_file.extractfile(member)
                if extracted is None:
                    raise ProjectionValidationError(
                        f"candidate evidence file {member.name} cannot be extracted"
                    )
                content_cache[member.name] = extracted.read()
    except (OSError, tarfile.TarError) as exc:
        raise ProjectionValidationError(
            f"candidate evidence tree archive could not be read at {candidate_head}"
        ) from exc
    if set(content_cache) != repository_files:
        raise ProjectionValidationError(
            f"candidate evidence tree contents are incomplete at {candidate_head}"
        )
    tracked_listing = _git_bytes(repo_root, "ls-tree", "-r", "-z", "--name-only", candidate_head)
    if tracked_listing is None:
        raise ProjectionValidationError(
            f"candidate tracked-path inventory is unavailable at {candidate_head}"
        )
    try:
        tracked_paths = {item.decode("utf-8") for item in tracked_listing.split(b"\0") if item}
    except UnicodeDecodeError as exc:
        raise ProjectionValidationError(
            f"candidate tracked-path inventory is not valid UTF-8 at {candidate_head}"
        ) from exc
    serialized = json.dumps(files, ensure_ascii=True, separators=(",", ":"))
    files_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return {
        "tree_sha": tree_sha,
        "count": len(files),
        "files": files,
        "sha256": files_sha256,
        "files_sha256": files_sha256,
        "content_cache": content_cache,
        "tracked_paths": tracked_paths,
    }


def _candidate_artifact_cache(
    repo_root: Path,
    candidate_head: str,
    artifact_paths: set[str],
    tracked_paths: set[str],
) -> dict[str, bytes]:
    """Read referenced candidate artifacts in one immutable Git archive."""
    selected = sorted(artifact_paths & tracked_paths)
    if not selected:
        return {}
    archive = _git_bytes(repo_root, "archive", "--format=tar", candidate_head, "--", *selected)
    if archive is None:
        raise ProjectionValidationError(
            f"candidate artifact content is unavailable at {candidate_head}"
        )
    cache: dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as archive_file:
            for member in archive_file.getmembers():
                if not member.isfile() or member.name not in selected:
                    continue
                extracted = archive_file.extractfile(member)
                if extracted is not None:
                    cache[member.name] = extracted.read()
    except (OSError, tarfile.TarError) as exc:
        raise ProjectionValidationError(
            f"candidate artifact archive could not be read at {candidate_head}"
        ) from exc
    if set(cache) != set(selected):
        raise ProjectionValidationError(
            f"candidate artifact content is incomplete at {candidate_head}"
        )
    return cache


def _load_dispositions(path: Path | None, *, raw: bytes | None = None) -> dict[str, dict[str, str]]:
    """Load optional report-mode category dispositions from the versioned packet."""
    if path is None or (raw is None and not path.is_file()):
        return {}
    text = raw.decode("utf-8") if raw is not None else path.read_text(encoding="utf-8")
    value = yaml.safe_load(text)
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != "evidence_registry_disposition.v1"
    ):
        raise ValueError(f"{path} must use schema_version evidence_registry_disposition.v1")
    categories = value.get("categories")
    if not isinstance(categories, list):
        raise ValueError(f"{path} must contain a categories list")
    dispositions: dict[str, dict[str, str]] = {}
    for category in categories:
        if not isinstance(category, Mapping):
            raise ValueError(f"{path} categories must be mappings")
        code = category.get("code")
        status = category.get("status")
        if not isinstance(code, str) or not isinstance(status, str) or not code or not status:
            raise ValueError(f"{path} categories require non-empty code and status")
        dispositions[code] = {"status": status}
    return dispositions


def _disposition_summary(
    findings: Sequence[dict[str, str]], dispositions: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    """Summarize report findings by their documented disposition, never suppressing them."""
    by_status: Counter[str] = Counter()
    unclassified: Counter[str] = Counter()
    for finding in findings:
        code = finding["code"]
        disposition = dispositions.get(code)
        if disposition is None:
            unclassified[code] += 1
        else:
            by_status[disposition["status"]] += 1
    return {
        "by_status": dict(sorted(by_status.items())),
        "unclassified_by_code": dict(sorted(unclassified.items())),
    }


def lint_evidence_registry(  # noqa: C901, PLR0912, PLR0915 - ordinary/projected gate contract.
    repo_root: Path,
    registry_root: Path,
    disposition_path: Path | None = None,
    exclude_codes: frozenset[str] = frozenset(),
    *,
    candidate_head: str | None = None,
    frozen_base: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic integrity report for every supported evidence file.

    With no projection identities the linter preserves its ordinary working-tree
    and ``HEAD`` behavior. When both identities are supplied, every evidence file
    is read from the candidate commit and every producer reachability decision is
    made from the frozen base only; incidental refs never participate.
    """
    repo_root = repo_root.resolve()
    _require_full_history(repo_root)
    registry_root = registry_root.resolve()
    projection_identity = _prepare_projection(repo_root, candidate_head, frozen_base)
    projection_tree = (
        _candidate_evidence_tree(repo_root, registry_root, projection_identity["candidate_head"])
        if projection_identity is not None
        else None
    )
    candidate_content_cache: dict[str, bytes] | None = None
    candidate_tracked_paths: set[str] | None = None
    if projection_tree is not None:
        candidate_content_cache = projection_tree.pop("content_cache")
        candidate_tracked_paths = projection_tree.pop("tracked_paths")
    companion_resolutions = _load_ch7_portfolio_companion_resolutions(
        repo_root,
        content_ref=projection_identity["candidate_head"] if projection_identity else None,
        content_cache=candidate_content_cache,
        tracked_paths=candidate_tracked_paths,
    )
    campaigns: dict[str, list[_DocumentRecord]] = defaultdict(list)
    findings: list[dict[str, str]] = []
    projection_producer_records: set[tuple[str, str, str]] = set()
    reachability_cache: dict[tuple[str, str], bool] = {}
    path_cache: dict[tuple[str, str], bool] = {}
    config_hash_cache: dict[tuple[str, str, tuple[str, ...]], bool] = {}
    if projection_tree is None:
        files = sorted(
            path
            for path in registry_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".md", ".csv"}
        )
        document_inputs = [(path, None, None) for path in files]
    else:
        registry_relative = registry_root.relative_to(repo_root)
        files = sorted(
            repo_root / registry_relative / relative
            for relative in projection_tree["files"]
            if Path(relative).suffix.lower() in {".json", ".yaml", ".yml", ".md", ".csv"}
        )
        document_inputs = []
        artifact_paths: set[str] = set()
        for path in files:
            relative = path.relative_to(repo_root).as_posix()
            raw = candidate_content_cache.get(relative) if candidate_content_cache else None
            if raw is None:
                raise ProjectionValidationError(
                    f"candidate evidence file {relative} disappeared while reading "
                    f"{projection_identity['candidate_head']}"
                )
            document_inputs.append((path, raw, Path(relative)))
            try:
                value = _load_document(path, raw=raw)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError):
                continue
            if not isinstance(value, str):
                for mapping, ancestors in _iter_mappings(value):
                    for key, declared_hash in mapping.items():
                        if (
                            not isinstance(key, str)
                            or key.lower() not in {"sha256", "source_sha256"}
                            or not isinstance(declared_hash, str)
                        ):
                            continue
                        artifact_path = _artifact_path(mapping, ancestors)
                        if artifact_path is None:
                            continue
                        resolved = _resolve_repo_path(repo_root, artifact_path)
                        if resolved is not None:
                            artifact_paths.add(resolved[0])
        candidate_content_cache.update(
            _candidate_artifact_cache(
                repo_root,
                projection_identity["candidate_head"],
                artifact_paths,
                candidate_tracked_paths or set(),
            )
        )
    for path, raw, display_path in document_inputs:
        document = _lint_document(
            repo_root,
            path,
            raw=raw,
            display_path=display_path,
            content_ref=projection_identity["candidate_head"] if projection_identity else None,
            content_cache=candidate_content_cache,
            tracked_paths=candidate_tracked_paths,
        )
        findings.extend(document.findings)
        # A row-oriented CSV can repeat one campaign identifier thousands of
        # times.  The linter's bundle-level semantics are document/campaign
        # based, so retain one document membership per campaign instead of
        # multiplying every repeated row into the projection report.
        for campaign_id in set(document.campaign_ids):
            campaigns[campaign_id].append(document)
            if projection_identity is not None:
                projection_producer_records.update(
                    (
                        campaign_id,
                        document.path.relative_to(repo_root).as_posix(),
                        commit,
                    )
                    for commit in document.commits
                )
    reachability_ref = (
        projection_identity["frozen_base"] if projection_identity is not None else "HEAD"
    )
    authority_history = _git_bytes(repo_root, "rev-list", "--full-history", reachability_ref)
    if authority_history is None:
        raise RuntimeError(
            f"could not enumerate complete producer reachability from {reachability_ref}"
        )
    authority_commits = {
        line.decode("ascii").lower()
        for line in authority_history.splitlines()
        if FULL_SHA1_RE.fullmatch(line.decode("ascii"))
    }
    for campaign_id, documents in sorted(campaigns.items()):
        canonical_path = min(document.path for document in documents).relative_to(repo_root)
        config_paths = [path for document in documents for path in document.config_paths]
        config_hashes = [item for document in documents for item in document.config_hashes]
        commits = [commit for document in documents for commit in document.commits]
        for commit in sorted(set(commits)):
            if not _commit_is_reachable(
                repo_root,
                commit,
                reachability_ref,
                reachability_cache,
                authority_commits,
            ):
                authority = (
                    f" from frozen base {reachability_ref}"
                    if projection_identity is not None
                    else ""
                )
                findings.append(
                    _issue(
                        canonical_path,
                        "dangling_commit",
                        f"commit {commit} does not resolve{authority}",
                    )
                )
        findings.extend(
            _campaign_metadata_findings(
                repo_root,
                canonical_path,
                [campaign_id],
                config_paths,
                config_hashes,
                commits,
                reachability_ref=reachability_ref,
                reachability_cache=reachability_cache,
                authority_commits=authority_commits,
                path_cache=path_cache,
                config_hash_cache=config_hash_cache,
            )
        )
        bundle_paths = sorted(
            {_bundle_path(registry_root, document.path) for document in documents}
        )
        if len(bundle_paths) > 1:
            bundle_list = ", ".join(path.relative_to(repo_root).as_posix() for path in bundle_paths)
            findings.append(
                _issue(
                    canonical_path,
                    "duplicate_campaign_id",
                    f"campaign_id {campaign_id!r} appears in multiple evidence bundles: {bundle_list}",
                )
            )
    findings.sort(key=lambda item: (item["path"], item["code"], item["message"]))
    findings = _apply_companion_resolutions(findings, companion_resolutions)
    excluded = [item for item in findings if item["code"] in exclude_codes]
    active = [item for item in findings if item["code"] not in exclude_codes]
    by_code = dict(sorted(Counter(item["code"] for item in active).items()))
    excluded_by_code = dict(sorted(Counter(item["code"] for item in excluded).items()))
    disposition_present = disposition_path is not None and disposition_path.is_file()
    if projection_identity is None:
        dispositions = _load_dispositions(disposition_path)
    else:
        # The projection must not summarize candidate findings with a dirty
        # worktree companion. Read the optional disposition packet from the
        # candidate tree, and treat a packet absent from that tree as absent.
        disposition_raw = None
        if disposition_path is not None:
            try:
                disposition_relative = disposition_path.resolve().relative_to(repo_root).as_posix()
            except ValueError:
                disposition_relative = ""
            if disposition_relative:
                disposition_raw = _repository_file_bytes(
                    repo_root,
                    disposition_relative,
                    projection_identity["candidate_head"],
                    candidate_content_cache,
                )
        disposition_present = disposition_raw is not None
        dispositions = (
            _load_dispositions(disposition_path, raw=disposition_raw)
            if disposition_raw is not None
            else {}
        )
    report: dict[str, Any] = {
        "registry_root": registry_root.relative_to(repo_root).as_posix(),
        "checked_files": len(files),
        "campaign_ids": sorted(campaigns),
        "issues": active,
        "summary": {"findings": len(active), "by_code": by_code},
    }
    if projection_identity is not None and projection_tree is not None:
        campaign_records = [
            {
                "campaign_id": campaign_id,
                "receipt_paths": sorted(
                    {document.path.relative_to(repo_root).as_posix() for document in documents}
                ),
                "producer_shas": sorted(
                    {commit for document in documents for commit in document.commits}
                ),
            }
            for campaign_id, documents in sorted(campaigns.items())
        ]
        report["projection"] = {
            "schema": PROJECTION_SCHEMA,
            "mode": PROJECTION_MODE,
            "candidate_head": projection_identity["candidate_head"],
            "frozen_base": projection_identity["frozen_base"],
            "evaluated_head": projection_identity["candidate_head"],
            "evaluated_base": projection_identity["frozen_base"],
            "head_sha": projection_identity["candidate_head"],
            "base_sha": projection_identity["frozen_base"],
            "reachability_authority": projection_identity["frozen_base"],
            "evidence_tree": projection_tree,
            "checked_files": len(files),
            "complete": True,
            "campaigns": campaign_records,
            "producer_records": [
                {
                    "campaign_id": campaign_id,
                    "receipt_path": receipt_path,
                    "producer_sha": producer_sha,
                }
                for campaign_id, receipt_path, producer_sha in sorted(projection_producer_records)
            ],
            "findings_by_path": {
                path: dict(
                    sorted(Counter(item["code"] for item in active if item["path"] == path).items())
                )
                for path in sorted({item["path"] for item in active})
            },
        }
    if exclude_codes:
        report["excluded_codes"] = sorted(exclude_codes)
        report["excluded_issues"] = excluded
        report["excluded_summary"] = {
            "findings": len(excluded),
            "by_code": excluded_by_code,
        }
    if disposition_present and disposition_path is not None:
        try:
            report["disposition_path"] = disposition_path.relative_to(repo_root).as_posix()
        except ValueError:
            report["disposition_path"] = disposition_path.as_posix()
        report["disposition_summary"] = _disposition_summary(findings, dispositions)
    return report


def _repo_root_from_git() -> Path:
    """Return the current checkout root for the default command invocation."""
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    )


def main(argv: list[str] | None = None) -> int:
    """Run the report-only or strict evidence registry linter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="Repository root (defaults to Git root)."
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path("docs/context/evidence"),
        help="Evidence registry root, relative to --repo-root by default.",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero when findings are present."
    )
    parser.add_argument(
        "--disposition-file",
        type=Path,
        default=Path("docs/context/evidence/evidence_registry_dispositions.yaml"),
        help="Optional versioned category-disposition packet for report-mode output.",
    )
    parser.add_argument(
        "--exclude-codes",
        type=str,
        default="",
        help="Comma-separated finding codes to exclude from strict-mode gating.",
    )
    parser.add_argument(
        "--strict-exclusion-policy",
        type=Path,
        default=None,
        help="Optional YAML file declaring excluded codes with justification.",
    )
    parser.add_argument(
        "--candidate-head",
        type=str,
        default=None,
        help=(
            "Explicit full SHA-1 of the candidate evidence tree and checked-out HEAD; "
            "must be paired with --frozen-base."
        ),
    )
    parser.add_argument(
        "--frozen-base",
        type=str,
        default=None,
        help=(
            "Explicit full SHA-1 whose ancestry is the only producer-reachability authority; "
            "must be paired with --candidate-head."
        ),
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve() if args.repo_root else _repo_root_from_git()
    registry_root = (
        args.registry_root if args.registry_root.is_absolute() else repo_root / args.registry_root
    )
    disposition_path = (
        args.disposition_file
        if args.disposition_file.is_absolute()
        else repo_root / args.disposition_file
    )
    exclude_codes: frozenset[str] = frozenset()
    if args.strict_exclusion_policy:
        exclude_codes = _load_strict_exclusion_policy(args.strict_exclusion_policy)
    if args.exclude_codes:
        exclude_codes = exclude_codes | frozenset(
            code.strip() for code in args.exclude_codes.split(",") if code.strip()
        )
    try:
        report = lint_evidence_registry(
            repo_root,
            registry_root,
            disposition_path,
            exclude_codes,
            candidate_head=args.candidate_head,
            frozen_base=args.frozen_base,
        )
    except (CompanionBindingError, ProjectionValidationError, ShallowRepositoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["summary"]["findings"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
