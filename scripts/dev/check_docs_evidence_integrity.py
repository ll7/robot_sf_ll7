#!/usr/bin/env python3
"""Lightweight, changed-path-scoped integrity check for docs/evidence surfaces.

This guard treats documentation, compact evidence bundles, schemas, catalogues,
issue templates, and governance files as first-class research-facing state. It
is intentionally changed-path scoped: only files a change actually touches are
inspected, so it can be mandatory in CI without failing on pre-existing legacy
drift in untouched files (issue #3476).

Checks performed:

- ``.json`` files parse JSON.
- ``.yaml`` / ``.yml`` files parse YAML (multi-document allowed).
- Markdown files: every explicit repository-local relative link (a target that
  starts ``./`` or ``../``) must resolve to an existing path.
- Changed ``docs/context/evidence`` files must be registered in
  ``docs/context/catalog.yaml``.
- Changed catalog rows must point at existing files and use valid status /
  freshness vocabulary values.
- Changed checksum manifests, and changed files covered by a local checksum
  manifest, must match current file contents.
- Evidence ``README.md`` classification fields cannot disagree with adjacent
  machine-readable ``summary.json`` fields.
- Cited script/config paths in changed docs or evidence files must exist.

The check is independent from the full Python test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_CATALOG_PATH = Path("docs/context/catalog.yaml")
_EVIDENCE_DIR = Path("docs/context/evidence")
_CHECKSUM_FILENAMES = {"SHA256SUMS", "checksums.sha256", "manifest.sha256"}
_SUMMARY_FILENAME = "summary.json"
_README_FILENAME = "README.md"
_VALID_CATALOG_STATUSES = {
    "current",
    "historical",
    "superseded",
    "evidence",
    "proposal",
}
_VALID_CATALOG_FRESHNESS = {"maintained", "dated", "policy", "evidence"}
_CLASSIFICATION_KEYS = {
    "benchmark_promotion",
    "claim_boundary",
    "diagnostic_only",
    "evidence_grade",
    "evidence_tier",
    "paper_facing",
    "result_classification",
    "schema",
    "schema_version",
    "status",
}

_MD_LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_README_FIELD = re.compile(r"`(?P<key>[A-Za-z0-9_-]+)`\s*:\s*`(?P<value>[^`]+)`")
_CONFIG_FLAG = re.compile(r"--[A-Za-z0-9_-]*config(?:=|\s+)(?P<path>[A-Za-z0-9_./:-]+)")
_CITED_REPO_PATH = re.compile(
    r"(?<![\w./-])(?P<path>(?:scripts|configs)/[A-Za-z0-9_./:-]+\.(?:py|sh|ya?ml))"
)
_RELATIVE_PREFIXES = ("./", "../")


def _repo_root() -> Path:
    """Return Git repository root."""
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(out.stdout.strip())


def changed_files(base_ref: str, *, root: Path) -> list[str]:
    """Return added/copied/modified/renamed paths relative to ``base_ref``."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    return [line for line in out.stdout.splitlines() if line.strip()]


def _repo_rel(path: Path, *, root: Path) -> Path:
    """Return a repository-relative path for display and catalog matching."""
    return path.resolve().relative_to(root.resolve())


def _looks_dynamic(path: str) -> bool:
    """Return whether a cited path contains shell/template expansion."""
    return any(token in path for token in ("$", "{", "}", "<", ">", "*"))


def _clean_cited_path(path: str) -> str:
    """Remove common trailing shell and Markdown punctuation from cited paths."""
    return path.strip().rstrip(".,;:)`'\"")


def _check_json(path: Path) -> list[str]:
    """Return parse errors for a JSON file."""
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path}: invalid JSON: {exc}"]
    return []


def _check_yaml(path: Path) -> list[str]:
    """Return parse errors for a YAML file (multi-document allowed)."""
    try:
        list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
    except (OSError, yaml.YAMLError) as exc:
        return [f"{path}: invalid YAML: {exc}"]
    return []


def _check_markdown_links(path: Path, *, root: Path) -> list[str]:
    """Return broken explicit repo-local relative links in a Markdown file."""
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: unreadable Markdown: {exc}"]

    for raw_target in _MD_LINK.findall(text):
        target = raw_target.strip()
        if not target.startswith(_RELATIVE_PREFIXES):
            continue

        file_part = target.split("#", 1)[0]
        if not file_part:
            continue

        resolved = (path.parent / file_part).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            problems.append(f"{path}: relative link escapes repository: {target}")
            continue
        if not resolved.exists():
            problems.append(f"{path}: broken repo-local link: {target}")
    return problems


def _load_catalog(root: Path) -> tuple[object | None, list[str]]:
    """Load the context catalog, returning payload and parse diagnostics."""
    catalog = root / _CATALOG_PATH
    if not catalog.is_file():
        return None, [f"{_CATALOG_PATH}: missing context catalog"]
    try:
        return yaml.safe_load(catalog.read_text(encoding="utf-8")), []
    except yaml.YAMLError as exc:
        return None, [f"{_CATALOG_PATH}: invalid YAML: {exc}"]


def _catalog_entries(payload: object) -> list[dict[object, object]]:
    """Return mapping entries from a catalog payload."""
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _catalog_path_value(value: object) -> Path | None:
    """Return a normalized catalog path value when it is repository-relative."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _catalog_registered_paths(payload: object) -> set[Path]:
    """Return repository-relative paths registered in the catalog."""
    paths: set[Path] = set()
    for entry in _catalog_entries(payload):
        path = _catalog_path_value(entry.get("path"))
        if path is not None:
            paths.add(path)
    return paths


def _catalog_validation_problems(payload: object, *, root: Path) -> list[str]:  # noqa: C901, PLR0912
    """Return registration and metadata problems in docs/context/catalog.yaml."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return [f"{_CATALOG_PATH}: context catalog must be a YAML mapping"]
    if payload.get("version") != 1:
        problems.append(f"{_CATALOG_PATH}: version must be 1")

    status_values = set(_VALID_CATALOG_STATUSES)
    raw_status_values = payload.get("status_values")
    if isinstance(raw_status_values, dict):
        status_values.update(str(key) for key in raw_status_values)

    freshness_values = set(_VALID_CATALOG_FRESHNESS)
    raw_freshness_values = payload.get("freshness_values")
    if isinstance(raw_freshness_values, dict):
        freshness_values.update(str(key) for key in raw_freshness_values)

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        return [f"{_CATALOG_PATH}: entries must be a non-empty list"]

    seen: set[Path] = set()
    for index, raw_entry in enumerate(entries):
        entry_path = f"entries[{index}]"
        if not isinstance(raw_entry, dict):
            problems.append(f"{_CATALOG_PATH}: {entry_path} must be a mapping")
            continue

        path = _catalog_path_value(raw_entry.get("path"))
        if path is None:
            problems.append(f"{_CATALOG_PATH}: {entry_path}.path must be repo-relative")
        else:
            if path in seen:
                problems.append(f"{_CATALOG_PATH}: {entry_path}.path duplicates {path}")
            seen.add(path)
            # Evidence bundles may be registered either as a single file or as a
            # directory, so accept both rather than requiring a regular file.
            if not (root / path).exists():
                problems.append(f"{_CATALOG_PATH}: {entry_path}.path does not exist: {path}")

        status = raw_entry.get("status")
        if status not in status_values:
            problems.append(f"{_CATALOG_PATH}: {entry_path}.status invalid: {status!r}")

        freshness = raw_entry.get("freshness")
        if freshness not in freshness_values:
            problems.append(f"{_CATALOG_PATH}: {entry_path}.freshness invalid: {freshness!r}")

        if status == "superseded":
            replacement = _catalog_path_value(raw_entry.get("replacement"))
            if replacement is None:
                problems.append(f"{_CATALOG_PATH}: {entry_path}.replacement required")
            elif not (root / replacement).exists():
                problems.append(
                    f"{_CATALOG_PATH}: {entry_path}.replacement does not exist: {replacement}"
                )

    return problems


def _checksum_manifest_paths(path: Path, *, root: Path) -> list[Path]:
    """Return checksum manifests adjacent to a changed evidence file."""
    try:
        rel = _repo_rel(path, root=root)
    except ValueError:
        return []
    if _EVIDENCE_DIR not in rel.parents:
        return []
    parent = path.parent
    return [parent / name for name in sorted(_CHECKSUM_FILENAMES) if (parent / name).is_file()]


def _parse_checksum_line(line: str, *, manifest: Path, root: Path) -> tuple[str, Path] | None:
    """Parse a sha256sum-style manifest line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = re.match(r"^(?P<hash>[0-9a-fA-F]{64})\s+\*?(?P<path>.+)$", stripped)
    if match is None:
        return None

    raw_path = match.group("path").strip().lstrip("./")
    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None

    root_candidate = root / candidate
    if root_candidate.exists():
        return match.group("hash").lower(), root_candidate
    return match.group("hash").lower(), manifest.parent / candidate


def _sha256(path: Path) -> str:
    """Return sha256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_problems_for_manifest(manifest: Path, *, root: Path) -> list[str]:
    """Return checksum mismatches inside one manifest."""
    problems: list[str] = []
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []

    for line_no, line in enumerate(lines, start=1):
        parsed = _parse_checksum_line(line, manifest=manifest, root=root)
        if parsed is None:
            continue
        expected, target = parsed
        if not target.is_file():
            problems.append(f"{manifest}: line {line_no} target missing: {target}")
            continue
        actual = _sha256(target)
        if actual != expected:
            problems.append(f"{manifest}: line {line_no} checksum mismatch for {target}")
    return problems


def _checksum_problems_for_changed_file(path: Path, *, root: Path) -> list[str]:
    """Return checksum mismatch if a changed evidence file has an adjacent manifest."""
    if path.name in _CHECKSUM_FILENAMES:
        return _checksum_problems_for_manifest(path, root=root)

    problems: list[str] = []
    for manifest in _checksum_manifest_paths(path, root=root):
        found = False
        try:
            lines = manifest.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(lines, start=1):
            parsed = _parse_checksum_line(line, manifest=manifest, root=root)
            if parsed is None:
                continue
            expected, target = parsed
            if target.resolve() != path.resolve():
                continue
            found = True
            actual = _sha256(path)
            if actual != expected:
                problems.append(
                    f"{manifest}: line {line_no} checksum mismatch for changed file {path}"
                )
        if found:
            break
    return problems


def _evidence_registration_problems(files: Iterable[Path], *, root: Path) -> list[str]:
    """Return catalog-registration and checksum problems for changed evidence files."""
    evidence_files = []
    for path in files:
        try:
            rel = _repo_rel(path, root=root)
        except ValueError:
            continue
        if rel == _CATALOG_PATH or _EVIDENCE_DIR in rel.parents:
            evidence_files.append(path)

    if not evidence_files:
        return []

    payload, load_problems = _load_catalog(root)
    if load_problems:
        return load_problems

    problems: list[str] = []
    registered = _catalog_registered_paths(payload)
    if any(_repo_rel(path, root=root) == _CATALOG_PATH for path in evidence_files):
        problems.extend(_catalog_validation_problems(payload, root=root))

    for path in evidence_files:
        rel = _repo_rel(path, root=root)
        if rel == _CATALOG_PATH:
            continue
        # A file is registered if its exact path is listed, or if an ancestor
        # directory is registered as an evidence bundle.
        if rel not in registered and not any(parent in registered for parent in rel.parents):
            problems.append(f"{rel}: evidence file is not registered in {_CATALOG_PATH}")
        problems.extend(_checksum_problems_for_changed_file(path, root=root))

    return problems


def _read_json_object(path: Path) -> dict[str, object] | None:
    """Read a JSON object, returning None for invalid or non-object payloads."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_value(value: object) -> str:
    """Normalize values for README-vs-summary comparisons."""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip().lower().replace("_", "-")


def _readme_summary_drift_problems(path: Path, *, root: Path) -> list[str]:  # noqa: C901
    """Return classification drift between evidence README.md and summary.json."""
    if path.name not in {_README_FILENAME, _SUMMARY_FILENAME}:
        return []
    try:
        rel = _repo_rel(path, root=root)
    except ValueError:
        return []
    if _EVIDENCE_DIR not in rel.parents:
        return []

    readme = path.parent / _README_FILENAME
    summary = path.parent / _SUMMARY_FILENAME
    if not readme.is_file() or not summary.is_file():
        return []

    summary_payload = _read_json_object(summary)
    if summary_payload is None:
        return []

    try:
        declared = {
            match.group("key").replace("-", "_").lower(): match.group("value")
            for match in _README_FIELD.finditer(readme.read_text(encoding="utf-8"))
        }
    except OSError:
        return []

    problems: list[str] = []
    for key, readme_value in sorted(declared.items()):
        if key not in _CLASSIFICATION_KEYS:
            continue
        summary_key = key
        if summary_key not in summary_payload and key == "schema":
            summary_key = "schema_version"
        if summary_key not in summary_payload:
            continue
        if _normalize_value(readme_value) != _normalize_value(summary_payload[summary_key]):
            problems.append(
                f"{readme}: `{key}` disagrees with {summary.name} "
                f"({readme_value!r} != {summary_payload[summary_key]!r})"
            )
    return problems


def _extract_cited_paths(text: str) -> set[str]:
    """Return script and config paths cited by changed text."""
    paths: set[str] = set()
    for match in _CONFIG_FLAG.finditer(text):
        paths.add(_clean_cited_path(match.group("path")))
    for match in _CITED_REPO_PATH.finditer(text):
        paths.add(_clean_cited_path(match.group("path")))
    return {path for path in paths if path and not _looks_dynamic(path)}


def _cited_path_problems(path: Path, *, root: Path) -> list[str]:
    """Return missing cited command/config path diagnostics."""
    if path.suffix.lower() not in {".md", ".json", ".yaml", ".yml"}:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    problems: list[str] = []
    for cited in sorted(_extract_cited_paths(text)):
        candidate = Path(cited)
        if candidate.is_absolute() or ".." in candidate.parts:
            continue
        if not (root / candidate).exists():
            problems.append(f"{path}: cited command/config path does not exist: {cited}")
    return problems


def check_files(files: Iterable[str], *, root: Path) -> list[str]:
    """Run integrity checks over repository-relative files."""
    problems: list[str] = []
    existing_paths: list[Path] = []

    for rel in files:
        path = (root / rel).resolve()
        if not path.is_file():
            continue
        existing_paths.append(path)

        suffix = path.suffix.lower()
        if suffix == ".json":
            problems.extend(_check_json(path))
        elif suffix in {".yaml", ".yml"}:
            problems.extend(_check_yaml(path))
        elif suffix in {".md", ".markdown"}:
            problems.extend(_check_markdown_links(path, root=root))

        problems.extend(_readme_summary_drift_problems(path, root=root))
        problems.extend(_cited_path_problems(path, root=root))

    problems.extend(_evidence_registration_problems(existing_paths, root=root))
    return problems


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Changed-path docs/evidence integrity check for JSON/YAML parseability, "
            "repo-local Markdown links, evidence catalog registration/checksums, "
            "README-vs-summary drift, and cited command/config paths."
        )
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Base ref for changed-file discovery (default: origin/main).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit repository-relative files to check instead of git diff.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Emit GitHub warnings but exit 0. Prefer blocking mode for CI.",
    )
    return parser


def _emit_warnings(problems: list[str]) -> None:
    """Emit findings as GitHub Actions warning annotations (non-blocking)."""
    for problem in problems:
        sys.stdout.write(f"::warning::docs/evidence integrity: {problem}\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Run docs/evidence integrity check and return a shell exit code."""
    args = _build_parser().parse_args(argv)
    root = _repo_root()
    files = list(args.files) if args.files is not None else changed_files(args.base_ref, root=root)
    if not files:
        print("docs/evidence integrity: no changed docs/evidence files to check.")
        return 0

    problems = check_files(files, root=root)
    if not problems:
        print(f"docs/evidence integrity: {len(files)} changed file(s) passed.")
        return 0

    if args.warn_only:
        _emit_warnings(problems)
        print(
            f"docs/evidence integrity: {len(problems)} advisory finding(s) "
            "(warning-only, not blocking)."
        )
        return 0

    sys.stderr.write("docs/evidence integrity check failed:\n")
    sys.stderr.write("\n".join(f"  {problem}" for problem in problems) + "\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
