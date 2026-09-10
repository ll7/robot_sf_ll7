#!/usr/bin/env python3
"""Credential-safe logical-locator snapshot for compute-window artifact custody.

Inventories logical experiment/artifact/model/dataset/checkpoint/campaign IDs into a
deterministic public JSON snapshot plus a recovery index. Exact locators may live in a
private overlay; public output and diagnostics never emit them. ``--check`` validates the
same fail-closed rules without writing outputs. Motivating issue: #8857.
Example: uv run python scripts/tools/locator_snapshot.py --registry <registry.yaml>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlparse

import yaml

from robot_sf.benchmark.artifact_catalog import sha256_file

REGISTRY_SCHEMA = "locator_registry.v1"
OVERLAY_SCHEMA = "locator_overlay.v1"
SNAPSHOT_SCHEMA = "locator_snapshot.v1"
CHECK_REPORT_SCHEMA = "locator_snapshot_check.v1"
RECOVERY_INDEX_SCHEMA = "locator_recovery_index.v1"
ARTIFACT_CLASSES = ("artifact", "campaign", "checkpoint", "dataset", "experiment", "model")
LOCATOR_CLASSES = (
    "artifact_uri",
    "dataset_root",
    "durable_mirror",
    "local_path",
    "model_alias",
    "private_overlay",
    "signed_url",
)
AVAILABILITIES = (
    "available",
    "expired",
    "missing",
    "transfer_incomplete",
    "unavailable",
    "unresolved",
)
DURABLE_DESTINATION_CLASSES = (
    "durable_mirror",
    "none",
    "registry_entry",
    "release_artifact",
    "tracked_path",
)
DEFAULT_SNAPSHOT_PATH = Path("output/locator_snapshot/locator_snapshot.json")
DEFAULT_RECOVERY_INDEX_PATH = Path("output/locator_snapshot/recovery_index.md")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MUTABLE_ALIAS_RE = re.compile(r"(?:^|[:/@])latest(?:$|[.:@/])", re.IGNORECASE)
_SIGNED_URL_QUERY_KEYS = frozenset(
    "expires se sig signature sp sv token x-amz-credential x-amz-expires "
    "x-amz-signature x-goog-signature".split()
)
_ENUM_FIELDS = (
    ("artifact_class", ARTIFACT_CLASSES),
    ("locator_class", LOCATOR_CLASSES),
    ("availability", AVAILABILITIES), ("durable_destination_class", DURABLE_DESTINATION_CLASSES),
)
_REMOTE_SCHEMES = {"http", "https", "s3", "gs", "az"}


@dataclass(frozen=True, slots=True)
class LocatorIssue:
    """One fail-closed locator snapshot issue or warning, free of private values."""

    code: str
    logical_id: str | None
    message: str


_issue = LocatorIssue


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    """Public-safe custody record for one logical ID."""

    logical_id: str
    owner: str
    artifact_class: str
    source_issue: str | None
    source_campaign: str | None
    content_digest: str
    schema_version: str
    locator_class: str
    availability: str
    durable_destination_class: str
    consumer_references: tuple[str, ...]
    locator_source: str
    verification: str
    destination_verification: str


@dataclass(frozen=True, slots=True)
class SnapshotResult:
    """Validated snapshot: public entries plus fail-closed issues."""

    entries: tuple[SnapshotEntry, ...]
    issues: tuple[LocatorIssue, ...]
    warnings: tuple[LocatorIssue, ...]
    as_of: str

    @property
    def ok(self) -> bool:
        """Return True when no fail-closed issue was found."""

        return not self.issues

    def render_snapshot_json(self) -> str:
        """Return byte-stable snapshot JSON with a trailing newline."""

        payload = {
            "schema": SNAPSHOT_SCHEMA,
            "as_of": self.as_of,
            "summary": {
                "entry_count": len(self.entries),
                "artifact_class_counts": _counts(e.artifact_class for e in self.entries),
                "availability_counts": _counts(e.availability for e in self.entries),
                "locator_class_counts": _counts(e.locator_class for e in self.entries),
                "verification_counts": _counts(e.verification for e in self.entries),
            },
            "entries": [asdict(entry) for entry in self.entries],
        }
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"

    def to_check_report_dict(self) -> dict[str, Any]:
        """Return the sanitized check report payload."""

        return {
            "schema": CHECK_REPORT_SCHEMA,
            "ok": self.ok,
            "as_of": self.as_of,
            "entry_count": len(self.entries),
            "issue_count": len(self.issues),
            "issues": [asdict(issue) for issue in self.issues],
            "warnings": [asdict(warning) for warning in self.warnings],
        }

    def render_recovery_index_markdown(self) -> str:
        """Return the concise human-readable recovery index."""

        lines = [
            "# Locator Recovery Index",
            "",
            f"- Schema: `{RECOVERY_INDEX_SCHEMA}` | As of: `{self.as_of}` | Entries: {len(self.entries)}",
            "",
            "Exact locators stay in the input registry or the private overlay and are never",
            "emitted here; resolve a logical ID through its registry or overlay key.",
            "",
            "| Logical ID | Class | Owner | Content digest | Availability | Storage class |"
            " Durable destination | Locator source | Verification |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        lines.extend(
            f"| `{e.logical_id}` | {e.artifact_class} | {e.owner} | `{e.content_digest}` |"
            f" {e.availability} | {e.locator_class} | {e.durable_destination_class} |"
            f" {e.locator_source} | {e.verification} |"
            for e in self.entries
        )
        return "\n".join(lines) + "\n"


def _counts(values: Iterable[str]) -> dict[str, int]:
    """Return sorted value counts for deterministic summaries."""

    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _optional_text(value: Any) -> str | None:
    """Return stripped text when the value is a non-empty string."""

    return value.strip() if isinstance(value, str) and value.strip() else None


def _load_mapping(
    path: Path, expected_schema: str, code: str
) -> tuple[Mapping[str, Any] | None, list[LocatorIssue]]:
    """Load a YAML/JSON mapping and verify its declared schema."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return None, [_issue(code, None, f"failed to load document ({type(exc).__name__})")]
    if not isinstance(payload, Mapping):
        return None, [_issue(code, None, "expected a mapping document")]
    if payload.get("schema") != expected_schema:
        return None, [_issue(code, None, f"declared schema must be {expected_schema}")]
    return payload, []


def _validate_entry_shape(entry: Mapping[str, Any], pointer: str) -> list[LocatorIssue]:
    """Return structural validation issues for one registry entry."""

    logical_id = _optional_text(entry.get("logical_id"))
    issues: list[LocatorIssue] = []
    if logical_id is None:
        issues.append(_issue("invalid_entry", None, f"{pointer}: bad logical_id"))
    if _optional_text(entry.get("owner")) is None:
        issues.append(_issue("missing_owner", logical_id, f"{pointer}: owner must be declared"))
    for field, allowed in _ENUM_FIELDS:
        if entry.get(field) not in allowed:
            issues.append(_issue("invalid_entry", logical_id, f"{pointer}: bad {field}"))
    digest = entry.get("content_digest")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest.strip()) is None:
        issues.append(_issue("invalid_content_digest", logical_id, f"{pointer}: bad digest"))
    if _optional_text(entry.get("schema_version")) is None:
        issues.append(_issue("invalid_entry", logical_id, f"{pointer}: schema_version required"))
    refs = entry.get("consumer_references")
    if not isinstance(refs, list) or not all(_optional_text(ref) for ref in refs):
        issues.append(_issue("invalid_consumer_references", logical_id, f"{pointer}: bad refs"))
    if not (
        _optional_text(entry.get("source_issue")) or _optional_text(entry.get("source_campaign"))
    ):
        issues.append(_issue("missing_source", logical_id, f"{pointer}: declare a source"))
    return issues


def _query_keys(text: str) -> set[str]:
    """Return lowercased HTTP(S) query-parameter keys for signature detection."""

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        return set()
    return {key.lower() for key, _ in parse_qsl(parsed.query, keep_blank_values=True)}


def _is_latest_alias(text: str | None) -> bool:
    """Return whether text names a mutable ``latest``-style alias."""

    return text is not None and _MUTABLE_ALIAS_RE.search(text) is not None


def _verify_field(
    text: str | None,
    base_dirs: Sequence[Path],
    digest: str,
    surface: str,
    logical_id: str,
    *,
    verify_bytes: bool,
    fail_on_missing: bool,
    issues: list[LocatorIssue],
) -> str:
    """Read-only verify one locator surface and append digest/missing-target issues."""

    if text is None:
        return "not_applicable"
    if not verify_bytes or "://" in text or urlparse(text).scheme in _REMOTE_SCHEMES:
        return "unresolved"
    path = Path(text)
    candidates = [path] if path.is_absolute() else [base / text for base in base_dirs]
    for candidate in candidates:
        if candidate.is_dir():
            return "unresolved"
        if candidate.is_file():
            if sha256_file(candidate) != digest:
                issues.append(_issue("digest_mismatch", logical_id, f"{surface} bytes conflict"))
            return "verified"
    if fail_on_missing:
        issues.append(_issue("missing_target", logical_id, f"{surface} target missing"))
    return "missing"


def _resolve_locators(
    entry: Mapping[str, Any],
    overlay_entry: Mapping[str, Any] | None,
    logical_id: str,
) -> tuple[str | None, str, str, list[LocatorIssue]]:
    """Resolve locator text, source, and effective class without emitting values."""

    issues: list[LocatorIssue] = []
    entry_locator = _optional_text(entry.get("locator"))
    overlay_locator = _optional_text(overlay_entry.get("locator")) if overlay_entry else None
    declared_class = str(entry["locator_class"])
    overlay_class = _optional_text(overlay_entry.get("locator_class")) if overlay_entry else None
    if overlay_class is not None and overlay_class not in LOCATOR_CLASSES:
        issues.append(_issue("invalid_overlay_entry", logical_id, "overlay class is invalid"))
        overlay_class = None
    if declared_class == "private_overlay":
        if overlay_locator is None:
            issues.append(_issue("missing_private_overlay", logical_id, "overlay locator required"))
        return overlay_locator, "private_overlay", overlay_class or "private_overlay", issues
    if entry_locator is not None:
        if overlay_class is not None and overlay_class != declared_class:
            issues.append(_issue("locator_class_conflict", logical_id, "overlay class conflicts"))
        return entry_locator, "registry_entry", declared_class, issues
    if overlay_locator is not None:
        return overlay_locator, "private_overlay", overlay_class or declared_class, issues
    issues.append(_issue("missing_locator", logical_id, "no locator in registry or overlay"))
    return None, "registry_entry", declared_class, issues


def _process_entry(
    entry: Mapping[str, Any],
    overlay_entry: Mapping[str, Any] | None,
    context: tuple[Path, Path | None, Path],
    issues: list[LocatorIssue],
    warnings: list[LocatorIssue],
) -> SnapshotEntry:
    """Validate one registry entry and build its public-safe custody record."""

    logical_id = str(entry["logical_id"]).strip()
    digest = str(entry["content_digest"]).strip().lower()
    availability = str(entry["availability"])
    destination_class = str(entry["durable_destination_class"])
    locator_text, source, effective_class, resolve_issues = _resolve_locators(
        entry, overlay_entry, logical_id
    )
    issues.extend(resolve_issues)
    destination_text = _optional_text(entry.get("durable_destination")) or (
        _optional_text(overlay_entry.get("durable_destination")) if overlay_entry else None
    )
    if availability == "expired":
        issues.append(_issue("expired_locator", logical_id, "expired locator is not durable"))
    elif availability == "missing":
        issues.append(_issue("missing_target", logical_id, "declared availability is 'missing'"))
    elif availability == "transfer_incomplete":
        issues.append(_issue("incomplete_transfer", logical_id, "durable transfer incomplete"))
    if effective_class == "signed_url" or (
        locator_text is not None
        and any(key in _SIGNED_URL_QUERY_KEYS for key in _query_keys(locator_text))
    ):
        issues.append(_issue("signed_locator", logical_id, "signed URL is not durable identity"))
    mutable = bool(entry.get("mutable_alias", False)) or (
        effective_class == "model_alias" and _is_latest_alias(locator_text)
    )
    if mutable and (destination_class == "none" or _is_latest_alias(destination_text)):
        issues.append(
            _issue("mutable_alias_not_durable", logical_id, "latest alias is not durable")
        )
    elif mutable:
        warnings.append(_issue("mutable_alias_current_locator", logical_id, "durable dest pins it"))
    registry_dir, overlay_dir, repo_root = context
    base_dirs = [registry_dir, *([overlay_dir] if overlay_dir is not None else []), repo_root]
    verification = _verify_field(
        locator_text,
        base_dirs,
        digest,
        "locator",
        logical_id,
        verify_bytes=effective_class in {"local_path", "dataset_root"},
        fail_on_missing=availability == "available",
        issues=issues,
    )
    destination_verification = _verify_field(
        destination_text,
        base_dirs,
        digest,
        "durable destination",
        logical_id,
        verify_bytes=destination_class == "tracked_path",
        fail_on_missing=destination_class == "tracked_path",
        issues=issues,
    )
    return SnapshotEntry(
        logical_id=logical_id,
        owner=str(entry["owner"]).strip(),
        artifact_class=str(entry["artifact_class"]),
        source_issue=_optional_text(entry.get("source_issue")),
        source_campaign=_optional_text(entry.get("source_campaign")),
        content_digest=digest,
        schema_version=str(entry["schema_version"]).strip(),
        locator_class=effective_class,
        availability=availability,
        durable_destination_class=destination_class,
        consumer_references=tuple(str(ref).strip() for ref in entry["consumer_references"]),
        locator_source=source,
        verification=verification,
        destination_verification=destination_verification,
    )


def _load_inputs(  # noqa: C901 - one container-validation pass per input document
    registry_paths: Sequence[Path], overlay_path: Path | None
) -> tuple[
    list[tuple[Path, Mapping[str, Any]]],
    dict[str, Mapping[str, Any]],
    Path | None,
    list[LocatorIssue],
]:
    """Load registries and the private overlay, validating their containers."""

    raw_entries: list[tuple[Path, Mapping[str, Any]]] = []
    issues: list[LocatorIssue] = []
    for path in registry_paths:
        doc, load_issues = _load_mapping(path, REGISTRY_SCHEMA, "invalid_registry_schema")
        issues.extend(load_issues)
        entries = doc.get("entries") if doc is not None else None
        if not isinstance(entries, list):
            if doc is not None:
                issues.append(_issue("invalid_registry_schema", None, "'entries' must be a list"))
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                issues.append(_issue("invalid_entry", None, f"entries[{index}]: not a mapping"))
                continue
            shape_issues = _validate_entry_shape(entry, f"entries[{index}]")
            issues.extend(shape_issues)
            if not shape_issues:
                raw_entries.append((path, entry))
    overlay_entries: dict[str, Mapping[str, Any]] = {}
    overlay_dir: Path | None = None
    if overlay_path is not None:
        path = Path(overlay_path)
        doc, overlay_issues = _load_mapping(path, OVERLAY_SCHEMA, "invalid_overlay_schema")
        issues.extend(overlay_issues)
        overlay_dir = path.resolve().parent
        raw_overlay = doc.get("entries") if doc is not None else None
        if isinstance(raw_overlay, Mapping):
            for key, value in raw_overlay.items():
                if isinstance(value, Mapping):
                    overlay_entries[str(key)] = value
                else:
                    issues.append(_issue("invalid_overlay_entry", str(key), "bad entry"))
        elif doc is not None:
            issues.append(_issue("invalid_overlay_schema", None, "'entries' must be a mapping"))
    return raw_entries, overlay_entries, overlay_dir, issues


def build_locator_snapshot(
    registry_paths: Iterable[Path],
    *,
    overlay_path: Path | None = None,
    as_of: str | None = None,
) -> SnapshotResult:
    """Build a public locator snapshot from registries and a private overlay."""

    registry_list = [Path(path) for path in registry_paths]
    raw_entries, overlay_entries, overlay_dir, issues = _load_inputs(registry_list, overlay_path)
    anchor = registry_list[0].resolve() if registry_list else Path.cwd()
    repo_root = anchor.parent
    for parent in (anchor.parent, *anchor.parents):
        if (parent / ".git").exists():
            repo_root = parent
            break
    seen: dict[str, str] = {}
    entries: list[SnapshotEntry] = []
    warnings: list[LocatorIssue] = []
    for registry_path, entry in raw_entries:
        logical_id = str(entry["logical_id"]).strip()
        digest = str(entry["content_digest"]).strip().lower()
        if logical_id in seen:
            issues.append(_issue("duplicate_logical_id", logical_id, "logical ID appears twice"))
            if seen[logical_id] != digest:
                issues.append(_issue("conflicting_bytes", logical_id, "one ID, different digests"))
            continue
        seen[logical_id] = digest
        overlay_entry = overlay_entries.get(logical_id)
        overlay_digest = (
            _optional_text(overlay_entry.get("content_digest")) if overlay_entry else None
        )
        if overlay_digest is not None and overlay_digest.lower() != digest:
            issues.append(_issue("conflicting_bytes", logical_id, "overlay digest conflicts"))
        context = (registry_path.resolve().parent, overlay_dir, repo_root)
        entries.append(_process_entry(entry, overlay_entry, context, issues, warnings))
    orphan_count = len(set(overlay_entries) - set(seen))
    if orphan_count:
        issues.append(
            _issue("orphan_overlay_entry", None, f"{orphan_count} orphan overlay entries")
        )
    return SnapshotResult(
        entries=tuple(sorted(entries, key=lambda entry: entry.logical_id)),
        issues=tuple(issues),
        warnings=tuple(warnings),
        as_of=_optional_text(as_of) or "unspecified",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the locator snapshot argument parser."""

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Example: uv run python scripts/tools/locator_snapshot.py --registry <registry.yaml>",
    )
    parser.add_argument(
        "--registry", action="append", required=True, type=Path, help="Registry fixture."
    )
    parser.add_argument("--overlay", type=Path, help="Private overlay; values are never emitted.")
    parser.add_argument("--format", choices=("json",), default="json", help="Stdout format (JSON).")
    parser.add_argument("--check", action="store_true", help="Validate only; write no files.")
    parser.add_argument(
        "--snapshot-out", type=Path, default=DEFAULT_SNAPSHOT_PATH, help="Snapshot path."
    )
    parser.add_argument("--recovery-index-out", type=Path, default=DEFAULT_RECOVERY_INDEX_PATH)
    parser.add_argument("--as-of", help="Availability normalization time.")
    return parser


def _report(result: SnapshotResult, *, check: bool) -> None:
    """Write the sanitized JSON check report or snapshot to stdout."""

    payload = result.to_check_report_dict() if check else result.render_snapshot_json()
    if isinstance(payload, str):
        sys.stdout.write(payload)
    else:
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the locator snapshot or check and return a shell-friendly exit code."""

    args = build_arg_parser().parse_args(argv)
    result = build_locator_snapshot(args.registry, overlay_path=args.overlay, as_of=args.as_of)
    if args.check:
        _report(result, check=True)
        return 0 if result.ok else 2
    if not result.ok:
        sys.stderr.write(json.dumps(result.to_check_report_dict(), indent=2, sort_keys=True) + "\n")
        return 2
    for path, text in (
        (args.snapshot_out, result.render_snapshot_json()),
        (args.recovery_index_out, result.render_recovery_index_markdown()),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _report(result, check=False)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
