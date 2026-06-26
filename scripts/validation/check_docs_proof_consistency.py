#!/usr/bin/env python3
"""Check branch-diff docs/proof consistency for PR handoff.

This helper is intentionally conservative. It only reports high-confidence
mechanical problems in changed files relative to a base ref (default:
``origin/main``).

Modes
-----
Default (no flags):
  Checks files in the branch diff / worktree edits against ``origin/main``.

``--path <repo-rel-path>``:
  Checks only the explicitly named file(s).

``--check-evidence-catalog``:
  Full evidence/catalog check — scans every tracked file under
  ``docs/context/evidence/`` and reports evidence bundles (immediate
  subdirectories) or standalone files that have no entry in
  ``docs/context/catalog.yaml``.  This mode does *not* use the git diff;
  it is safe to run independently as a standalone hygiene pass.
  Run with::

      uv run python scripts/validation/check_docs_proof_consistency.py --check-evidence-catalog
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_CONTEXT_README = Path("docs/context/README.md")
_CONTEXT_INDEX = Path("docs/context/INDEX.md")
_CONTEXT_CATALOG = Path("docs/context/catalog.yaml")
_TOP_LEVEL_CONTEXT_DIR = Path("docs/context")
_EVIDENCE_DIR = Path("docs/context/evidence")
_ABSOLUTE_LOCAL_PATH_RE = re.compile(r"(?<!\w)(/home/[^\s`'\"<>)\]}]+|/Users/[^\s`'\"<>)\]}]+)")
_OUTPUT_PATH_RE = re.compile(
    r"(?<!\w)(?:output/[^\s`'\"<>)\]}]+|/home/[^\s`'\"<>)\]}]*/output/[^\s`'\"<>)\]}]+|/Users/[^\s`'\"<>)\]}]*/output/[^\s`'\"<>)\]}]+)"
)
_TEXT_EVIDENCE_SUFFIXES = {".md", ".json", ".yaml", ".yml", ".txt"}
_VALIDATION_SKIP_RE = re.compile(r"\bno validation commands were run\b", re.IGNORECASE)
_COMMAND_HINT_RE = re.compile(
    r"(`[^`\n]*(?:uv run|pytest|ruff|scripts/dev/|python )[^`\n]*`|```(?:bash|sh)?[\s\S]*?(?:uv run|pytest|ruff|scripts/dev/|python )[\s\S]*?```)",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)|<((?:\./)?[^ >]+)>")
_LINE_START_ISSUE_REF_RE = re.compile(
    r"^(?:\s{0,3}|(?:\s*(?:(?:[-*+]\s+)|(?:\d+[.)]\s+)|(?:>\s*)|(?:\|\s*))))"
    r"#(?P<number>\d+)\b"
)
_FENCE_START_RE = re.compile(r"^([`~]{3,})")
_CATALOG_STATUSES = {"current", "historical", "superseded", "evidence", "proposal"}
_CATALOG_DEFAULT_FRESHNESS = {"maintained", "dated", "policy", "evidence"}


@dataclass(frozen=True)
class ChangedFile:
    """Repository-relative file path plus git diff status."""

    status: str
    path: Path


@dataclass(frozen=True)
class Diagnostic:
    """One high-confidence docs/proof consistency problem."""

    path: Path
    message: str


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run a subprocess and return stripped stdout."""
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}".rstrip()
        )
    return proc.stdout.strip()


def _path_exists_in_ref(path: Path, ref: str, repo_root: Path) -> bool:
    """Return whether a repository-relative path exists in a git ref."""
    spec = f"{ref}:{path.as_posix()}"
    proc = subprocess.run(
        ["git", "cat-file", "-e", spec],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _repo_root() -> Path:
    """Return the repository root for the current checkout."""
    return Path(_run(["git", "rev-parse", "--show-toplevel"])).resolve()


def _normalize_path(path: Path, repo_root: Path) -> Path:
    """Return a repository-relative path when possible."""
    if path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root)
        except ValueError:
            return path
    return path


def _normalize_explicit_path(raw_path: str, repo_root: Path) -> Path:
    """Return a validated repository-relative path from a user-provided --path value."""
    path = _normalize_path(Path(raw_path), repo_root)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"--path must stay within the repository root: {raw_path}")
    return path


def _is_within_dir(path: Path, root: Path) -> bool:
    """Return whether a repository-relative path is at or below a trusted root."""
    return path == root or root in path.parents


def _is_added_status(status: str) -> bool:
    """Return whether a git status represents a newly introduced file."""
    return status == "A" or status.startswith("C")


def _parse_name_status(output: str, *, default_status: str | None = None) -> list[ChangedFile]:
    """Parse git --name-status output into ChangedFile rows."""
    parsed: list[ChangedFile] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if default_status is not None:
            parsed.append(ChangedFile(status=default_status, path=Path(stripped)))
            continue
        parts = stripped.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        raw_path = parts[-1]
        parsed.append(ChangedFile(status=status, path=Path(raw_path)))
    return parsed


def _changed_files(base: str, repo_root: Path) -> list[ChangedFile]:
    """Return changed files from the branch diff plus local worktree edits."""

    combined: dict[Path, str] = {}
    commands: list[tuple[list[str], str | None]] = [
        (
            ["git", "diff", "--name-status", "--diff-filter=ACMRT", f"{base}...HEAD"],
            None,
        ),
        (["git", "diff", "--name-status", "--cached", "--diff-filter=ACMRT"], None),
        (["git", "diff", "--name-status", "--diff-filter=ACMRT"], None),
        (["git", "ls-files", "--others", "--exclude-standard"], "A"),
    ]

    for cmd, default_status in commands:
        output = _run(cmd, cwd=repo_root)
        for changed in _parse_name_status(output, default_status=default_status):
            current = combined.get(changed.path)
            if current == "A":
                continue
            combined[changed.path] = "A" if _is_added_status(changed.status) else changed.status

    return [ChangedFile(status=status, path=path) for path, status in sorted(combined.items())]


def _read_text(path: Path) -> str:
    """Read UTF-8 text from a repository path."""
    return path.read_text(encoding="utf-8")


def _strip_fenced_code_blocks(text: str) -> str:
    """Remove fenced markdown code blocks from a document."""
    return re.sub(r"```[\s\S]*?```", "", text)


def _markdown_targets(text: str) -> list[str]:
    """Extract markdown/autolink targets from a markdown string."""
    targets: list[str] = []
    for first, second in _MARKDOWN_LINK_RE.findall(text):
        target = first or second
        if target:
            targets.append(target.strip().split("#", maxsplit=1)[0])
    return targets


def _contains_link_target(targets: Iterable[str], expected: str) -> bool:
    """Return whether a markdown target list references an expected relative path."""
    normalized_expected = expected.lstrip("./")
    repo_relative_expected = f"{_TOP_LEVEL_CONTEXT_DIR.as_posix()}/{normalized_expected}"
    for target in targets:
        candidate = target.lstrip("./")
        if candidate in {normalized_expected, repo_relative_expected}:
            return True
    return False


def _context_note_is_added(changed: ChangedFile) -> bool:
    """Return whether a changed file is a new top-level context note."""
    return (
        _is_added_status(changed.status)
        and changed.path.parent == _TOP_LEVEL_CONTEXT_DIR
        and changed.path.suffix == ".md"
    )


def _context_readme_link_diagnostics(
    changed_files: Iterable[ChangedFile],
    *,
    context_readme_text: str,
) -> list[Diagnostic]:
    """Flag added top-level docs/context notes missing from the context index."""
    diagnostics: list[Diagnostic] = []
    targets = _markdown_targets(context_readme_text)
    for changed in changed_files:
        if not _context_note_is_added(changed):
            continue
        if changed.path.name == _CONTEXT_README.name:
            continue
        if _contains_link_target(targets, changed.path.name):
            continue
        diagnostics.append(
            Diagnostic(
                path=changed.path,
                message="added context note is not linked from docs/context/README.md",
            )
        )
    return diagnostics


def _context_index_link_diagnostics(
    changed_files: Iterable[ChangedFile],
    *,
    context_index_text: str,
) -> list[Diagnostic]:
    """Flag added top-level docs/context notes missing from the context index."""
    diagnostics: list[Diagnostic] = []
    targets = _markdown_targets(context_index_text)
    for changed in changed_files:
        if not _context_note_is_added(changed):
            continue
        if changed.path.name in {_CONTEXT_README.name, _CONTEXT_INDEX.name}:
            continue
        if _contains_link_target(targets, changed.path.name):
            continue
        diagnostics.append(
            Diagnostic(
                path=changed.path,
                message="added context note is not linked from docs/context/INDEX.md",
            )
        )
    return diagnostics


def _evidence_path_diagnostics(path: Path, text: str) -> list[Diagnostic]:
    """Flag durable-evidence files that contain local absolute paths or output pointers."""
    diagnostics: list[Diagnostic] = []
    if not _is_within_dir(path, _EVIDENCE_DIR):
        return diagnostics

    scan_text = _strip_fenced_code_blocks(text) if path.suffix == ".md" else text

    if _ABSOLUTE_LOCAL_PATH_RE.search(scan_text):
        diagnostics.append(
            Diagnostic(
                path=path,
                message="tracked evidence should not contain absolute local filesystem paths",
            )
        )

    if _OUTPUT_PATH_RE.search(scan_text):
        diagnostics.append(
            Diagnostic(
                path=path,
                message="tracked evidence should not point to ignored output/ artifacts",
            )
        )

    return diagnostics


def _load_yaml(path: Path) -> object:
    """Load YAML from a repository path."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _catalog_path(raw_path: object) -> Path | None:
    """Return a catalog path value when it is a non-empty string."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path.strip())
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def _catalog_value_keys(raw_values: object, *, default_values: set[str]) -> set[str]:
    """Return declared catalog vocabulary keys, falling back to default values."""
    if not isinstance(raw_values, dict):
        return default_values
    keys = {key for key in raw_values if isinstance(key, str) and key.strip()}
    return keys or default_values


def _catalog_entry_path_diagnostics(
    entry: dict[object, object],
    *,
    entry_path: str,
    catalog_path: Path,
    repo_root: Path,
    seen_paths: set[Path],
) -> tuple[Path | None, list[Diagnostic]]:
    """Validate and return a catalog row path."""
    diagnostics: list[Diagnostic] = []
    path = _catalog_path(entry.get("path"))
    if path is None:
        diagnostics.append(Diagnostic(catalog_path, f"{entry_path}.path is required"))
        return None, diagnostics
    if path in seen_paths:
        diagnostics.append(
            Diagnostic(catalog_path, f"{entry_path}.path duplicates {path.as_posix()}")
        )
    seen_paths.add(path)
    if not (repo_root / path).exists():
        diagnostics.append(
            Diagnostic(catalog_path, f"{entry_path}.path does not exist: {path.as_posix()}")
        )
    return path, diagnostics


def _catalog_entry_metadata_diagnostics(
    entry: dict[object, object],
    *,
    entry_path: str,
    catalog_path: Path,
    status_values: set[str],
    freshness_values: set[str],
) -> list[Diagnostic]:
    """Validate status and freshness metadata for a catalog row."""
    diagnostics: list[Diagnostic] = []
    status = entry.get("status")
    if status not in status_values:
        diagnostics.append(
            Diagnostic(
                catalog_path,
                f"{entry_path}.status must be one of {', '.join(sorted(status_values))}",
            )
        )

    freshness = entry.get("freshness")
    if not isinstance(freshness, str) or not freshness.strip():
        diagnostics.append(Diagnostic(catalog_path, f"{entry_path}.freshness is required"))
    elif freshness not in freshness_values:
        diagnostics.append(
            Diagnostic(
                catalog_path,
                f"{entry_path}.freshness must be one of {', '.join(sorted(freshness_values))}",
            )
        )
    return diagnostics


def _catalog_entry_replacement_diagnostics(
    entry: dict[object, object],
    *,
    entry_path: str,
    catalog_path: Path,
    repo_root: Path,
) -> list[Diagnostic]:
    """Validate superseded replacement metadata for a catalog row."""
    if entry.get("status") != "superseded":
        return []
    replacement = _catalog_path(entry.get("replacement"))
    if replacement is None:
        return [
            Diagnostic(catalog_path, f"{entry_path}.replacement is required for superseded entries")
        ]
    if not (repo_root / replacement).exists():
        return [
            Diagnostic(
                catalog_path,
                f"{entry_path}.replacement does not exist: {replacement.as_posix()}",
            )
        ]
    return []


def _catalog_entry_evidence_diagnostics(
    entry: dict[object, object],
    *,
    path: Path | None,
    entry_path: str,
    catalog_path: Path,
    repo_root: Path,
) -> list[Diagnostic]:
    """Validate durable evidence paths referenced by catalog evidence rows."""
    if entry.get("status") != "evidence" or path is None or not (repo_root / path).is_file():
        return []
    if entry.get("legacy_dirty_evidence") is True:
        return []
    if path.suffix not in _TEXT_EVIDENCE_SUFFIXES:
        return []

    try:
        text = _read_text(repo_root / path)
    except UnicodeDecodeError:
        return []
    scan_text = _strip_fenced_code_blocks(text) if path.suffix == ".md" else text
    diagnostics: list[Diagnostic] = []
    if _OUTPUT_PATH_RE.search(scan_text):
        diagnostics.append(
            Diagnostic(
                catalog_path,
                f"{entry_path}.path is evidence and points to ignored output/ artifacts",
            )
        )
    if _ABSOLUTE_LOCAL_PATH_RE.search(scan_text):
        diagnostics.append(
            Diagnostic(
                catalog_path,
                f"{entry_path}.path is evidence and contains absolute local filesystem paths",
            )
        )
    return diagnostics


def _context_catalog_diagnostics(catalog_path: Path, *, repo_root: Path) -> list[Diagnostic]:
    """Validate the curated context catalog sidecar."""
    diagnostics: list[Diagnostic] = []
    full_catalog_path = repo_root / catalog_path
    if not full_catalog_path.exists():
        return diagnostics

    try:
        payload = _load_yaml(full_catalog_path)
    except yaml.YAMLError as exc:
        return [Diagnostic(catalog_path, f"context catalog is not a valid YAML file: {exc}")]
    if not isinstance(payload, dict):
        return [Diagnostic(catalog_path, "context catalog must be a YAML mapping")]
    if payload.get("version") != 1:
        diagnostics.append(Diagnostic(catalog_path, "context catalog version must be 1"))

    status_values = _catalog_value_keys(
        payload.get("status_values"), default_values=_CATALOG_STATUSES
    )
    missing_statuses = _CATALOG_STATUSES - status_values
    if missing_statuses:
        diagnostics.append(
            Diagnostic(
                catalog_path,
                f"context catalog status_values is missing: {', '.join(sorted(missing_statuses))}",
            )
        )
    freshness_values = _catalog_value_keys(
        payload.get("freshness_values"), default_values=_CATALOG_DEFAULT_FRESHNESS
    )

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        diagnostics.append(
            Diagnostic(catalog_path, "context catalog entries must be a non-empty list")
        )
        return diagnostics

    seen_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        entry_path = f"{catalog_path.as_posix()}:entries[{index}]"
        if not isinstance(entry, dict):
            diagnostics.append(Diagnostic(catalog_path, f"{entry_path} must be a mapping"))
            continue

        path, path_diagnostics = _catalog_entry_path_diagnostics(
            entry,
            entry_path=entry_path,
            catalog_path=catalog_path,
            repo_root=repo_root,
            seen_paths=seen_paths,
        )
        diagnostics.extend(path_diagnostics)
        diagnostics.extend(
            _catalog_entry_metadata_diagnostics(
                entry,
                entry_path=entry_path,
                catalog_path=catalog_path,
                status_values=status_values,
                freshness_values=freshness_values,
            )
        )
        diagnostics.extend(
            _catalog_entry_replacement_diagnostics(
                entry,
                entry_path=entry_path,
                catalog_path=catalog_path,
                repo_root=repo_root,
            )
        )
        diagnostics.extend(
            _catalog_entry_evidence_diagnostics(
                entry,
                path=path,
                entry_path=entry_path,
                catalog_path=catalog_path,
                repo_root=repo_root,
            )
        )

    return diagnostics


def _tracked_evidence_files(repo_root: Path) -> list[Path]:
    """Return repository-relative paths for all Git-tracked files under docs/context/evidence/."""
    proc = subprocess.run(
        ["git", "ls-files", "--", _EVIDENCE_DIR.as_posix()],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    return [Path(line.strip()) for line in proc.stdout.splitlines() if line.strip()]


def _catalog_paths_from_payload(payload: object) -> set[Path]:
    """Return the set of valid path values declared in a catalog YAML payload."""
    catalog_paths: set[Path] = set()
    if not isinstance(payload, dict):
        return catalog_paths
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return catalog_paths
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        p = _catalog_path(entry.get("path"))
        if p is not None:
            catalog_paths.add(p)
    return catalog_paths


def _evidence_bundle_key(tracked_path: Path) -> Path:
    """Return the evidence-bundle group key for a tracked evidence file.

    For files directly inside ``docs/context/evidence/``, the key is the file
    itself (standalone file).  For files nested inside a subdirectory, the key
    is the immediate child directory of ``docs/context/evidence/``.
    """
    parts = tracked_path.relative_to(_EVIDENCE_DIR).parts
    return _EVIDENCE_DIR / parts[0] if parts else tracked_path


def evidence_bundle_members(repo_root: Path) -> dict[Path, list[Path]]:
    """Group tracked ``docs/context/evidence/`` files by their evidence-bundle key.

    The returned mapping uses :func:`_evidence_bundle_key` for grouping, so each
    key is either an immediate subdirectory of ``docs/context/evidence/`` (a
    bundle) or a standalone file directly under it.  Member lists are sorted for
    deterministic downstream consumption (for example, representative-file
    selection in ``scripts/tools/catalog_evidence.py``).  This is shared,
    importable discovery state, not a duplicate of the checker's scan logic.
    """
    members: dict[Path, list[Path]] = {}
    for tracked_path in _tracked_evidence_files(repo_root):
        if not _is_within_dir(tracked_path, _EVIDENCE_DIR):
            continue
        members.setdefault(_evidence_bundle_key(tracked_path), []).append(tracked_path)
    for member_list in members.values():
        member_list.sort()
    return members


def uncovered_evidence_bundles(
    repo_root: Path,
    *,
    catalog_path: Path = _CONTEXT_CATALOG,
) -> list[Path]:
    """Return sorted evidence-bundle keys that have no catalog entry covering them.

    Each immediate subdirectory under ``docs/context/evidence/`` is treated as an
    *evidence bundle*.  A bundle is *covered* when the catalog contains at least
    one entry whose path is at or below the bundle root.  Loose files directly
    under ``docs/context/evidence/`` are checked individually.  Directories with
    no tracked files are ignored.

    This is the single source of truth for "what counts as an uncataloged
    anchor"; both the ``--check-evidence-catalog`` diagnostics here and the
    ``catalog_evidence.py`` proposer consume it so they cannot diverge.
    """
    full_catalog_path = repo_root / catalog_path
    catalog_paths: set[Path] = set()
    if full_catalog_path.exists():
        try:
            payload = _load_yaml(full_catalog_path)
        except yaml.YAMLError:
            payload = None
        catalog_paths = _catalog_paths_from_payload(payload)

    bundle_keys = evidence_bundle_members(repo_root).keys()
    if not bundle_keys:
        return []

    uncovered: list[Path] = []
    for bundle_key in sorted(bundle_keys):
        covered = any(_is_within_dir(p, bundle_key) for p in catalog_paths)
        if not covered:
            uncovered.append(bundle_key)
    return uncovered


def _evidence_catalog_coverage_diagnostics(
    *,
    repo_root: Path,
    catalog_path: Path = _CONTEXT_CATALOG,
) -> list[Diagnostic]:
    """Report evidence bundles or files with no catalog entry in docs/context/catalog.yaml.

    Thin diagnostic wrapper over :func:`uncovered_evidence_bundles`.  Designed for
    explicit ``--check-evidence-catalog`` runs; it does not modify or interact
    with the diff-scoped default checks.
    """
    return [
        Diagnostic(
            path=catalog_path,
            message=(
                f"tracked evidence has no catalog entry: {bundle_key.as_posix()}"
                " — add at least one entry under this path in"
                f" {catalog_path.as_posix()}"
            ),
        )
        for bundle_key in uncovered_evidence_bundles(repo_root, catalog_path=catalog_path)
    ]


def _validation_phrase_diagnostics(path: Path, text: str) -> list[Diagnostic]:
    """Flag notes that claim no validation ran while also listing executed commands."""
    if not _is_within_dir(path, _TOP_LEVEL_CONTEXT_DIR):
        return []
    if path.suffix != ".md":
        return []
    if not _VALIDATION_SKIP_RE.search(text):
        return []
    if not _COMMAND_HINT_RE.search(text):
        return []
    return [
        Diagnostic(
            path=path,
            message=(
                "note says no validation commands were run but also includes executable validation command references"
            ),
        )
    ]


def _markdown_lines_outside_fences(text: str) -> Iterable[tuple[int, str]]:
    """Yield Markdown source lines that are not inside fenced code blocks."""
    in_fence = False
    fence_marker: str | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        marker_match = _FENCE_START_RE.match(stripped) if indent < 4 else None
        marker = marker_match.group(1) if marker_match else None
        if marker is not None:
            if in_fence and marker[0] == fence_marker:
                in_fence = False
                fence_marker = None
            elif not in_fence:
                in_fence = True
                fence_marker = marker[0]
            continue
        if not in_fence:
            yield line_number, line


def _issue_reference_style_diagnostics(path: Path, text: str) -> list[Diagnostic]:
    """Flag bare line-start issue references that render poorly in Markdown."""
    if path.suffix != ".md":
        return []

    diagnostics: list[Diagnostic] = []
    for line_number, line in _markdown_lines_outside_fences(text):
        match = _LINE_START_ISSUE_REF_RE.match(line)
        if not match:
            continue
        issue_number = match.group("number")
        diagnostics.append(
            Diagnostic(
                path=path,
                message=(
                    f"line {line_number}: bare issue reference #{issue_number} should use "
                    f"'Issue #{issue_number}' at the start of Markdown prose, lists, or tables"
                ),
            )
        )
    return diagnostics


def _file_diagnostics(path: Path, text: str) -> list[Diagnostic]:
    """Collect all diagnostics for one changed file."""
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_evidence_path_diagnostics(path, text))
    diagnostics.extend(_validation_phrase_diagnostics(path, text))
    diagnostics.extend(_issue_reference_style_diagnostics(path, text))
    return diagnostics


def _collect_diagnostics(
    changed_files: Iterable[ChangedFile],
    *,
    repo_root: Path,
) -> list[Diagnostic]:
    """Collect all docs/proof consistency diagnostics for the selected file set."""
    diagnostics: list[Diagnostic] = []
    changed_list = list(changed_files)
    context_readme = repo_root / _CONTEXT_README
    if context_readme.exists():
        diagnostics.extend(
            _context_readme_link_diagnostics(
                changed_list,
                context_readme_text=_read_text(context_readme),
            )
        )
    context_index = repo_root / _CONTEXT_INDEX
    if context_index.exists():
        diagnostics.extend(
            _context_index_link_diagnostics(
                changed_list,
                context_index_text=_read_text(context_index),
            )
        )
    diagnostics.extend(_context_catalog_diagnostics(_CONTEXT_CATALOG, repo_root=repo_root))

    for changed in changed_list:
        full_path = repo_root / changed.path
        if not full_path.exists() or not full_path.is_file():
            continue
        if full_path.suffix not in {".md", ".json", ".yaml", ".yml", ".txt"}:
            continue
        diagnostics.extend(_file_diagnostics(changed.path, _read_text(full_path)))
    return diagnostics


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the docs/proof consistency checker."""
    parser = argparse.ArgumentParser(
        description="Check changed docs/proof surfaces for high-confidence consistency issues.",
    )
    parser.add_argument("--base", default="origin/main", help="Base ref to diff against.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Optional repository-relative path(s) to check instead of the git diff.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit diagnostics as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--check-evidence-catalog",
        action="store_true",
        dest="check_evidence_catalog",
        help=(
            "Run a full evidence/catalog coverage check: scan all tracked files under"
            f" {_EVIDENCE_DIR.as_posix()} and report evidence bundles with no entry in"
            f" {_CONTEXT_CATALOG.as_posix()}. Does not use the git diff."
        ),
    )
    return parser.parse_args()


def _selected_files(args: argparse.Namespace, repo_root: Path) -> list[ChangedFile]:
    """Resolve the changed file set from explicit paths or the branch diff."""
    if args.path:
        selected: list[ChangedFile] = []
        for raw_path in args.path:
            path = _normalize_explicit_path(raw_path, repo_root)
            status = "M" if _path_exists_in_ref(path, str(args.base), repo_root) else "A"
            selected.append(ChangedFile(status=status, path=path))
        return selected
    return _changed_files(str(args.base), repo_root)


def main() -> int:
    """Run the docs/proof consistency checker."""
    args = _parse_args()
    repo_root = _repo_root()

    if args.check_evidence_catalog:
        diagnostics = _evidence_catalog_coverage_diagnostics(repo_root=repo_root)
        if args.json:
            payload = [
                {"path": diagnostic.path.as_posix(), "message": diagnostic.message}
                for diagnostic in diagnostics
            ]
            print(json.dumps(payload, indent=2))
        elif diagnostics:
            for diagnostic in diagnostics:
                print(f"ERROR {diagnostic.path.as_posix()}: {diagnostic.message}")
        else:
            print(
                "OK evidence/catalog coverage check passed:"
                f" all tracked {_EVIDENCE_DIR.as_posix()} bundles have catalog entries."
            )
        return 1 if diagnostics else 0

    try:
        changed_files = _selected_files(args, repo_root)
    except ValueError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    diagnostics = _collect_diagnostics(changed_files, repo_root=repo_root)

    if args.json:
        payload = [
            {"path": diagnostic.path.as_posix(), "message": diagnostic.message}
            for diagnostic in diagnostics
        ]
        print(json.dumps(payload, indent=2))
    elif diagnostics:
        for diagnostic in diagnostics:
            print(f"ERROR {diagnostic.path.as_posix()}: {diagnostic.message}")
    else:
        checked = len(changed_files)
        print(f"OK docs/proof consistency check passed for {checked} changed file(s).")

    return 1 if diagnostics else 0


if __name__ == "__main__":
    raise SystemExit(main())
