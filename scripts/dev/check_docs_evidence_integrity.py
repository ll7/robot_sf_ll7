#!/usr/bin/env python3
"""Lightweight integrity checks for docs/evidence surfaces.

This guard treats documentation, compact evidence bundles, schemas, catalogues,
issue templates, and governance files as first-class research-facing state. It
keeps evidence, catalog, checksum, and citation rules changed-path scoped so they
can be mandatory in CI without failing on pre-existing legacy drift in untouched
files (issue #3476). ``--full`` is a separate deterministic, repository-wide
Markdown-link scan.

Checks performed:

- ``.json`` files parse JSON.
- ``.yaml`` / ``.yml`` files parse YAML (multi-document allowed).
- Markdown files: inline, image, and reference-definition destinations that are
  repository-local must resolve to an existing durable path.
- Markdown links using ``file:`` absolute local paths or worktree-local
  ``output/`` targets are flagged as non-portable internal references.
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
import html
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


def _sha256(path: Path) -> str:
    """Return sha256 digest for a file.

    Kept local (no ``robot_sf`` import) so this changed-path-scoped guard runs
    in the lightweight docs-evidence-integrity CI job, which installs only
    PyYAML and not the package. See #4926/#4929 regression: importing
    ``robot_sf.benchmark.identity.hash_utils`` here broke every docs/evidence PR
    with ``ModuleNotFoundError: No module named 'robot_sf'``.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

_README_FIELD = re.compile(r"`(?P<key>[A-Za-z0-9_-]+)`\s*:\s*`(?P<value>[^`]+)`")
_CONFIG_FLAG = re.compile(r"--[A-Za-z0-9_-]*config(?:=|\s+)(?P<path>[A-Za-z0-9_./:-]+)")
# Paths handed to an output flag are *created* by the documented command, not
# pre-existing inputs, so they must not be enforced as must-exist citations.
_OUTPUT_FLAG = re.compile(
    r"(?:--[A-Za-z0-9-]*out(?:put)?[A-Za-z0-9-]*|-o)(?:=|\s+)(?P<path>[A-Za-z0-9_./:-]+)"
)
_CITED_REPO_PATH = re.compile(
    r"(?<![\w./-])(?P<path>(?:scripts|configs)/[A-Za-z0-9_./:-]+\.(?:py|sh|ya?ml))"
)
_URI_SCHEME = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*):")
_REFERENCE_DEFINITION = re.compile(r"^[ ]{0,3}\[(?:\\.|[^\]\\])+\]:[ \t]*(?P<destination>.*)$")
_FENCE_OPEN = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})")
_LIST_ITEM = re.compile(r"^(?P<indent> {0,3})(?P<marker>(?:[-+*]|\d{1,9}[.)]))(?P<spacing>[ \t]+)")
_BLOCKQUOTE_PREFIX = re.compile(r"^[ ]{0,3}>[ \t]?")
_MARKDOWN_ESCAPABLE = frozenset(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
# A self-declared artifact-presence registry (e.g. the research-package registry,
# issue #3057) enumerates artifact paths that a companion preflight probes for
# presence and surfaces as explicit gaps when missing. Such entries are meant to
# name not-yet-existing artifacts, so the must-exist cited-path check does not
# apply; all other integrity checks still run.
_ARTIFACT_REGISTRY_SCHEMA_PREFIXES = ("research-package-registry",)


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


def _is_escaped(text: str | list[str], index: int) -> bool:
    """Return whether the character at ``index`` has an odd backslash prefix."""
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _is_closing_fence(content: str, *, character: str, minimum_length: int) -> bool:
    """Return whether a line closes the active fenced-code block."""
    stripped = content.lstrip(" ")
    indentation = len(content) - len(stripped)
    closing_run = len(stripped) - len(stripped.lstrip(character))
    return indentation <= 3 and closing_run >= minimum_length and not stripped[closing_run:].strip()


def _fence_container_content(content: str, *, list_content_indent: int | None) -> str:
    """Remove list/blockquote container prefixes for fence recognition."""
    indentation = len(content) - len(content.lstrip(" "))
    if list_content_indent is not None and indentation >= list_content_indent:
        content = content[list_content_indent:]
    while blockquote := _BLOCKQUOTE_PREFIX.match(content):
        content = content[blockquote.end() :]
    return content


def _blank_markdown_block_code(text: str) -> str:  # noqa: C901
    """Blank fenced and indented code while preserving line structure."""
    visible_lines: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    indented_code = False
    list_content_indent: int | None = None

    for line in text.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        content = line[:-1] if newline else line
        fence_content = _fence_container_content(
            content,
            list_content_indent=list_content_indent,
        )
        fence = _FENCE_OPEN.match(fence_content)

        if fence_character is not None:
            if _is_closing_fence(
                fence_content,
                character=fence_character,
                minimum_length=fence_length,
            ):
                fence_character = None
                fence_length = 0
            visible_lines.append(newline)
            continue

        if indented_code:
            if not content.strip() or content.startswith(("    ", "\t")):
                visible_lines.append(newline)
                continue
            indented_code = False

        if fence:
            marker = fence.group("fence")
            fence_character = marker[0]
            fence_length = len(marker)
            visible_lines.append(newline)
            continue

        if not content.strip():
            visible_lines.append(newline)
            continue

        list_item = _LIST_ITEM.match(content)
        if list_item:
            spacing = list_item.group("spacing").expandtabs(4)
            list_content_indent = (
                len(list_item.group("indent")) + len(list_item.group("marker")) + len(spacing)
            )
            visible_lines.append(content + newline)
            continue

        indentation = len(content) - len(content.lstrip(" "))
        if content.startswith("\t"):
            indentation = 4
        if indentation >= 4 and not (
            list_content_indent is not None
            and list_content_indent <= indentation < list_content_indent + 4
        ):
            indented_code = True
            visible_lines.append(newline)
            continue

        if indentation < (list_content_indent or 0):
            list_content_indent = None
        visible_lines.append(content + newline)

    return "".join(visible_lines)


def _blank_markdown_inline_code(text: str) -> str:
    """Blank inline code spans while preserving line structure."""
    visible = list(text)
    index = 0
    while index < len(visible):
        if visible[index] != "`" or _is_escaped(visible, index):
            index += 1
            continue

        run_end = index
        while run_end < len(visible) and visible[run_end] == "`":
            run_end += 1
        run_length = run_end - index
        closing = run_end

        while closing < len(visible):
            if visible[closing] != "`":
                closing += 1
                continue
            closing_end = closing
            while closing_end < len(visible) and visible[closing_end] == "`":
                closing_end += 1
            if closing_end - closing == run_length:
                for blank_index in range(index, closing_end):
                    if visible[blank_index] != "\n":
                        visible[blank_index] = " "
                index = closing_end
                break
            closing = closing_end
        else:
            index = run_end

    return "".join(visible)


def _blank_markdown_code(text: str) -> str:
    """Blank fenced, indented, and inline code while preserving line structure."""
    return _blank_markdown_inline_code(_blank_markdown_block_code(text))


def _unescape_markdown_target(target: str) -> str:
    """Decode Markdown escapes and entities in a link target."""
    target = html.unescape(target.strip())
    decoded: list[str] = []
    index = 0
    while index < len(target):
        if (
            target[index] == "\\"
            and index + 1 < len(target)
            and target[index + 1] in _MARKDOWN_ESCAPABLE
        ):
            index += 1
        decoded.append(target[index])
        index += 1
    return "".join(decoded)


def _skip_horizontal_space(text: str, index: int) -> int:
    """Return the first index after horizontal whitespace."""
    while index < len(text) and text[index] in " \t":
        index += 1
    return index


def _parse_angle_destination(text: str, index: int) -> tuple[str, int] | None:
    """Parse a ``<destination>`` beginning at ``index``."""
    target_start = index + 1
    index = target_start
    while index < len(text):
        if text[index] == ">" and not _is_escaped(text, index):
            return text[target_start:index], index + 1
        if text[index] in "\r\n":
            return None
        index += 1
    return None


def _parse_bare_destination(text: str, index: int, *, inline: bool) -> tuple[str, int] | None:
    """Parse a bare destination, balancing parentheses for inline links."""
    target_start = index
    nested_parentheses = 0
    while index < len(text):
        character = text[index]
        if character in "\r\n\t " and not _is_escaped(text, index):
            break
        if inline and character == "(" and not _is_escaped(text, index):
            nested_parentheses += 1
        elif inline and character == ")" and not _is_escaped(text, index):
            if nested_parentheses == 0:
                break
            nested_parentheses -= 1
        index += 1
    if index == target_start or nested_parentheses:
        return None
    return text[target_start:index], index


def _parse_inline_link_end(text: str, index: int) -> int | None:
    """Parse an optional title and required closing parenthesis."""
    index = _skip_horizontal_space(text, index)
    if index < len(text) and text[index] == ")":
        return index + 1

    if index >= len(text) or text[index] not in "\"'(":
        return None
    title_close = ")" if text[index] == "(" else text[index]
    index += 1
    while index < len(text):
        if text[index] == title_close and not _is_escaped(text, index):
            index = _skip_horizontal_space(text, index + 1)
            return index + 1 if index < len(text) and text[index] == ")" else None
        if text[index] in "\r\n":
            return None
        index += 1
    return None


def _parse_destination(text: str, start: int, *, inline: bool) -> tuple[str, int] | None:
    """Parse an inline-link or reference-definition destination."""
    index = _skip_horizontal_space(text, start)
    if index >= len(text):
        return None

    parsed = (
        _parse_angle_destination(text, index)
        if text[index] == "<"
        else _parse_bare_destination(text, index, inline=inline)
    )
    if parsed is None:
        return None
    target, index = parsed
    if not inline:
        return target, index
    link_end = _parse_inline_link_end(text, index)
    return (target, link_end) if link_end is not None else None


def _matching_label_end(text: str, start: int) -> int | None:
    """Return the closing bracket for a possibly nested Markdown link label."""
    depth = 1
    index = start + 1
    while index < len(text):
        if text[index] == "[" and not _is_escaped(text, index):
            depth += 1
        elif text[index] == "]" and not _is_escaped(text, index):
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _markdown_link_targets(text: str) -> list[str]:
    """Return inline and reference-definition destinations from Markdown text."""
    visible = _blank_markdown_code(text)
    targets: list[str] = []

    for line in visible.splitlines():
        definition = _REFERENCE_DEFINITION.match(line)
        if not definition:
            continue
        parsed = _parse_destination(definition.group("destination"), 0, inline=False)
        if parsed is not None:
            targets.append(parsed[0])

    index = 0
    while index < len(visible):
        if visible[index] != "[" or _is_escaped(visible, index):
            index += 1
            continue
        label_end = _matching_label_end(visible, index)
        if label_end is None or label_end + 1 >= len(visible) or visible[label_end + 1] != "(":
            index += 1
            continue
        parsed = _parse_destination(visible, label_end + 2, inline=True)
        if parsed is None:
            index = label_end + 1
            continue
        targets.append(parsed[0])
        index = parsed[1]

    return targets


def _markdown_link_problem(path: Path, target: str, *, root: Path) -> str | None:
    """Return one portability or resolution problem for a Markdown destination."""
    target = _unescape_markdown_target(target)
    scheme_match = _URI_SCHEME.match(target)
    if scheme_match:
        if scheme_match.group("scheme").lower() == "file":
            return f"{path}: file:/// absolute local path: {target}"
        return None
    if target.startswith(("/", "#")):
        return None

    file_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not file_part:
        return None

    resolved = (path.parent / file_part).resolve()
    try:
        repo_relative = resolved.relative_to(root.resolve())
    except ValueError:
        return f"{path}: relative link escapes repository: {target}"

    if repo_relative.parts and repo_relative.parts[0] == "output":
        return f"{path}: non-durable output/ link: {target}"
    if not resolved.exists():
        return f"{path}: broken repo-local link: {target}"
    return None


def _check_markdown_links(path: Path, *, root: Path) -> list[str]:
    """Return broken repo-local relative links and file:/// references in a Markdown file."""
    problems: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"{path}: unreadable Markdown: {exc}"]

    for target in _markdown_link_targets(text):
        problem = _markdown_link_problem(path, target, root=root)
        if problem is not None:
            problems.append(problem)
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


def _resolve_checksum_target(candidate: Path, *, manifest: Path, root: Path) -> Path:
    """Resolve a manifest checksum entry to the file it should verify.

    Prefer the file adjacent to the manifest (standard ``sha256sum -c`` semantics
    run from the packet directory), so a bare entry such as ``README.md`` verifies
    the packet's own file rather than a repo-root file with the same name (issue
    #4317). Fall back to the repo-root-relative resolution only when no
    manifest-local file exists, which preserves manifests written with
    repo-root-relative paths (for example ``docs/context/evidence/.../summary.json``).
    """
    manifest_candidate = manifest.parent / candidate
    if manifest_candidate.exists():
        return manifest_candidate
    return root / candidate


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

    target = _resolve_checksum_target(candidate, manifest=manifest, root=root)
    return match.group("hash").lower(), target


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
    """Return script and config paths cited by changed text.

    Paths passed to an output flag (for example ``--output``/``--out``/``-o``)
    name files the documented command *creates*, so they are excluded from the
    must-exist citation set even when they live under ``configs/``/``scripts/``.
    """
    output_paths = {_clean_cited_path(match.group("path")) for match in _OUTPUT_FLAG.finditer(text)}
    paths: set[str] = set()
    for match in _CONFIG_FLAG.finditer(text):
        paths.add(_clean_cited_path(match.group("path")))
    for match in _CITED_REPO_PATH.finditer(text):
        paths.add(_clean_cited_path(match.group("path")))
    paths -= output_paths
    return {
        path for path in paths if path and not _looks_dynamic(path) and not path.startswith("-")
    }


def _is_artifact_registry(path: Path) -> bool:
    """Return whether a YAML file declares itself an artifact-presence registry.

    Such registries enumerate artifact paths that a companion preflight probes for
    presence and reports as explicit gaps when missing, so their entries may name
    not-yet-existing artifacts and are exempt from the must-exist cited-path check.
    """
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return False
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict):
        return False
    schema_version = str(payload.get("schema_version", "")).strip()
    return schema_version.startswith(_ARTIFACT_REGISTRY_SCHEMA_PREFIXES)


def _cited_path_problems(path: Path, *, root: Path) -> list[str]:
    """Return missing cited command/config path diagnostics."""
    if path.suffix.lower() not in {".md", ".json", ".yaml", ".yml"}:
        return []
    if _is_artifact_registry(path):
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


def check_markdown_link_files(files: Iterable[str], *, root: Path) -> list[str]:
    """Run only Markdown-link checks over repository-relative files."""
    problems: list[str] = []
    for rel in files:
        path = (root / rel).resolve()
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}:
            problems.extend(_check_markdown_links(path, root=root))
    return problems


def _all_docs_markdown_files(root: Path) -> list[str]:
    """Return deterministic repository-relative Markdown paths under docs/."""
    docs_dir = root / "docs"
    if not docs_dir.is_dir():
        return []
    return [
        str(path.relative_to(root))
        for path in sorted(docs_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}
    ]


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Docs/evidence integrity check for JSON/YAML parseability, "
            "repo-local Markdown links, evidence catalog registration/checksums, "
            "README-vs-summary drift, and cited command/config paths. "
            "By default inspects only changed files (git diff). "
            "Use --full for a link-only scan of all Markdown files under docs/."
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
        "--full",
        action="store_true",
        help=(
            "Run only Markdown-link integrity across every Markdown file under docs/. "
            "Other evidence checks remain changed-file scoped."
        ),
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

    if args.full:
        files = _all_docs_markdown_files(root)
        if not files:
            print("docs/evidence integrity: no Markdown files found under docs/.")
            return 0
        print(
            "docs/evidence integrity: full-repo Markdown link scan of "
            f"{len(files)} file(s) under docs/."
        )
    elif args.files is not None:
        files = list(args.files)
    else:
        files = changed_files(args.base_ref, root=root)

    if not files:
        print("docs/evidence integrity: no changed docs/evidence files to check.")
        return 0

    problems = (
        check_markdown_link_files(files, root=root) if args.full else check_files(files, root=root)
    )
    if not problems:
        scope = "Markdown file(s)" if args.full else "changed file(s)"
        print(f"docs/evidence integrity: {len(files)} {scope} passed.")
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
