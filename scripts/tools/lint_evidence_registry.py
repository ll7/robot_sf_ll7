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
import json
import re
import subprocess
import sys
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
CH7_PORTFOLIO_ALLOWED_BINDINGS = {
    (
        "docs/context/evidence/issue_6792_ch7_evidence_package_v1/manifest.json",
        "hash_without_artifact_path",
        "sha256 lacks an adjacent artifact path",
    ),
    (
        "docs/context/evidence/issue_6792_ch7_evidence_package_v1/publication/"
        "materialization_overlay.json",
        "uncommitted_artifact_missing_location",
        "ch7_worked_example_portfolio.v1.yaml is not tracked and lacks an explicit location marker",
    ),
}


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


def _commit_is_head_reachable(repo_root: Path, commit: str) -> bool:
    """Return whether ``commit`` exists and belongs to the checked-out history.

    Git object stores may also contain commits fetched through unrelated branches
    or tags. Treating raw object presence as provenance resolution makes the
    evidence baseline depend on which refs happened to be fetched. Reachability
    from ``HEAD`` gives full-history checkouts one stable authority while still
    rejecting missing commit objects.
    """
    return _git_succeeds(repo_root, "cat-file", "-e", f"{commit}^{{commit}}") and _git_succeeds(
        repo_root, "merge-base", "--is-ancestor", commit, "HEAD"
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


def _config_hash_matches(
    repo_root: Path, commit: str, path: str, declared_hashes: set[str]
) -> bool:
    """Return whether one committed config blob matches a declared SHA-256 value."""

    blob = _git_bytes(repo_root, "show", f"{commit}:{path}")
    return blob is not None and hashlib.sha256(blob).hexdigest() in declared_hashes


def _is_tracked(repo_root: Path, repo_path: str) -> bool:
    """Return whether a normalized repository path is tracked in the current index."""
    return _git_succeeds(repo_root, "ls-files", "--error-unmatch", "--", repo_path)


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


def _load_document(path: Path) -> Any:
    """Load supported structured registry files, returning text for Markdown/CSV."""
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(raw)
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(raw)
    if suffix == ".csv":
        return list(csv.DictReader(raw.splitlines()))
    return raw


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


def _load_ch7_companion_payload(path: Path) -> Mapping[str, Any]:
    """Load and parse the #7047 companion payload."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
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


def _validate_ch7_target(repo_root: Path) -> None:
    """Validate that the approved source config is tracked and byte-matching."""
    target = _resolve_repo_path(repo_root, CH7_PORTFOLIO_TARGET_PATH)
    if target is None or not _is_tracked(repo_root, target[0]):
        raise _ch7_binding_error(f"target_path is not tracked: {CH7_PORTFOLIO_TARGET_PATH}")
    actual_target_sha256 = hashlib.sha256(target[1].read_bytes()).hexdigest()
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
    repo_root: Path, binding: Mapping[str, Any], document_path: str
) -> Path:
    """Validate that the bound package document is tracked and byte-matching."""
    document = _resolve_repo_path(repo_root, document_path)
    if document is None or not _is_tracked(repo_root, document[0]):
        raise _ch7_binding_error(f"document_path is not tracked: {document_path}")
    document_sha256 = binding.get("document_sha256")
    if (
        not isinstance(document_sha256, str)
        or not SHA256_RE.fullmatch(document_sha256)
        or hashlib.sha256(document[1].read_bytes()).hexdigest() != document_sha256
    ):
        raise _ch7_binding_error(f"document digest mismatch for {document_path}")
    return document[1]


def _validate_ch7_document_pointers(
    document_path: str, document: Path, binding: Mapping[str, Any], index: int
) -> None:
    """Validate the bound package JSON pointers and digest value."""
    pointer = binding.get("document_json_pointer")
    digest_pointer = binding.get("document_digest_pointer")
    if not isinstance(pointer, str) or not isinstance(digest_pointer, str):
        raise _ch7_binding_error(f"bindings[{index}] requires JSON pointers")
    try:
        document_value = _load_document(document)
        actual_value = _json_pointer_get(document_value, pointer)
        actual_digest_value = _json_pointer_get(document_value, digest_pointer)
    except (KeyError, ValueError) as exc:
        raise _ch7_binding_error(f"JSON pointer does not resolve for {document_path}") from exc
    if actual_value != binding.get("document_json_value"):
        raise _ch7_binding_error(f"pointer value mismatch for {document_path}")
    if actual_digest_value != CH7_PORTFOLIO_TARGET_SHA256:
        raise _ch7_binding_error(f"digest pointer mismatch for {document_path}")


def _validate_ch7_binding_entry(
    repo_root: Path, binding: Any, index: int, resolutions: set[tuple[str, str, str]]
) -> None:
    """Validate one #7047 companion binding entry."""
    if not isinstance(binding, Mapping):
        raise _ch7_binding_error(f"bindings[{index}] must be an object")
    resolution_key = _ch7_resolution_key(binding, index)
    if resolution_key in resolutions:
        document_path, code, _message = resolution_key
        raise _ch7_binding_error(f"duplicate binding for {document_path}/{code}")
    document_path = resolution_key[0]
    document = _validate_ch7_document_digest(repo_root, binding, document_path)
    _validate_ch7_document_pointers(document_path, document, binding, index)
    resolutions.add(resolution_key)


def _load_ch7_portfolio_companion_resolutions(repo_root: Path) -> frozenset[tuple[str, str, str]]:
    """Load the canonical #7047 companion binding and return exact findings it resolves."""
    path = repo_root / CH7_PORTFOLIO_COMPANION_BINDING
    if not path.is_file():
        return frozenset()
    value = _load_ch7_companion_payload(path)
    bindings = _validate_ch7_companion_header(value)
    _validate_ch7_target(repo_root)
    resolutions: set[tuple[str, str, str]] = set()
    for index, binding in enumerate(bindings):
        _validate_ch7_binding_entry(repo_root, binding, index, resolutions)
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


def _artifact_hash_finding(
    repo_root: Path,
    display_path: Path,
    key: str,
    declared_hash: str,
    mapping: Mapping[str, Any],
    ancestors: tuple[Mapping[str, Any], ...],
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
    if resolved is None or not _is_tracked(repo_root, resolved[0]):
        if not _has_location(mapping, ancestors):
            return _issue(
                display_path,
                "uncommitted_artifact_missing_location",
                f"{artifact_path} is not tracked and lacks an explicit location marker",
            )
        return None
    actual_hash = hashlib.sha256(resolved[1].read_bytes()).hexdigest()
    if actual_hash != declared_hash.lower():
        return _issue(
            display_path,
            "artifact_hash_mismatch",
            f"{artifact_path} SHA-256 does not match the tracked file",
        )
    return None


def _artifact_findings(repo_root: Path, display_path: Path, value: Any) -> list[dict[str, str]]:
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
                repo_root, display_path, key, declared_hash, mapping, ancestors
            )
            if finding:
                findings.append(finding)
    return findings


def _campaign_metadata_findings(  # noqa: C901 - rule-to-finding mapping is intentionally linear
    repo_root: Path,
    display_path: Path,
    campaign_ids: Sequence[str],
    config_paths: Sequence[str],
    config_hashes: Sequence[str],
    commits: Sequence[str],
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
        commit for commit in set(commits) if _commit_is_head_reachable(repo_root, commit)
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
            _git_succeeds(repo_root, "cat-file", "-e", f"{commit}:{normalized[0]}")
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
                _config_hash_matches(repo_root, commit, normalized[0], declared_config_hashes)
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


def _lint_document(repo_root: Path, path: Path) -> _DocumentRecord:
    """Read one document and retain only document-local integrity findings."""
    display_path = path.relative_to(repo_root)
    try:
        value = _load_document(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        return _DocumentRecord(
            path, [], [], [], [], [_issue(display_path, "unreadable_document", str(exc))]
        )
    campaign_ids = _campaign_ids(value)
    config_paths, config_hashes, commits = _file_metadata(value)
    local_findings = _artifact_findings(repo_root, display_path, value)
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


def _load_dispositions(path: Path | None) -> dict[str, dict[str, str]]:
    """Load optional report-mode category dispositions from the versioned packet."""
    if path is None or not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
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


def lint_evidence_registry(
    repo_root: Path,
    registry_root: Path,
    disposition_path: Path | None = None,
    exclude_codes: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Return a deterministic integrity report for every supported evidence file."""
    repo_root = repo_root.resolve()
    _require_full_history(repo_root)
    companion_resolutions = _load_ch7_portfolio_companion_resolutions(repo_root)
    registry_root = registry_root.resolve()
    campaigns: dict[str, list[_DocumentRecord]] = defaultdict(list)
    findings: list[dict[str, str]] = []
    files = sorted(
        path
        for path in registry_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".md", ".csv"}
    )
    for path in files:
        document = _lint_document(repo_root, path)
        findings.extend(document.findings)
        for campaign_id in document.campaign_ids:
            campaigns[campaign_id].append(document)
    for campaign_id, documents in sorted(campaigns.items()):
        canonical_path = min(document.path for document in documents).relative_to(repo_root)
        config_paths = [path for document in documents for path in document.config_paths]
        config_hashes = [item for document in documents for item in document.config_hashes]
        commits = [commit for document in documents for commit in document.commits]
        for commit in sorted(set(commits)):
            if not _commit_is_head_reachable(repo_root, commit):
                findings.append(
                    _issue(canonical_path, "dangling_commit", f"commit {commit} does not resolve")
                )
        findings.extend(
            _campaign_metadata_findings(
                repo_root,
                canonical_path,
                [campaign_id],
                config_paths,
                config_hashes,
                commits,
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
    dispositions = _load_dispositions(disposition_path)
    report: dict[str, Any] = {
        "registry_root": registry_root.relative_to(repo_root).as_posix(),
        "checked_files": len(files),
        "campaign_ids": sorted(campaigns),
        "issues": active,
        "summary": {"findings": len(active), "by_code": by_code},
    }
    if exclude_codes:
        report["excluded_codes"] = sorted(exclude_codes)
        report["excluded_issues"] = excluded
        report["excluded_summary"] = {
            "findings": len(excluded),
            "by_code": excluded_by_code,
        }
    if disposition_path is not None and disposition_path.is_file():
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
        report = lint_evidence_registry(repo_root, registry_root, disposition_path, exclude_codes)
    except (CompanionBindingError, ShallowRepositoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and report["summary"]["findings"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
