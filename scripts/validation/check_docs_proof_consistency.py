#!/usr/bin/env python3
"""Check branch-diff docs/proof consistency for PR handoff.

This helper is intentionally conservative. It only reports high-confidence
mechanical problems in changed files relative to a base ref (default:
``origin/main``).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_CONTEXT_README = Path("docs/context/README.md")
_TOP_LEVEL_CONTEXT_DIR = Path("docs/context")
_EVIDENCE_DIR = Path("docs/context/evidence")
_ABSOLUTE_LOCAL_PATH_RE = re.compile(r"(?<!\w)(/home/[^\s`'\"<>)\]}]+|/Users/[^\s`'\"<>)\]}]+)")
_OUTPUT_PATH_RE = re.compile(
    r"(?<!\w)(?:output/[^\s`'\"<>)\]}]+|/home/[^\s`'\"<>)\]}]*/output/[^\s`'\"<>)\]}]+|/Users/[^\s`'\"<>)\]}]*/output/[^\s`'\"<>)\]}]+)"
)
_VALIDATION_SKIP_RE = re.compile(r"\bno validation commands were run\b", re.IGNORECASE)
_COMMAND_HINT_RE = re.compile(
    r"(`[^`\n]*(?:uv run|pytest|ruff|scripts/dev/|python )[^`\n]*`|```(?:bash|sh)?[\s\S]*?(?:uv run|pytest|ruff|scripts/dev/|python )[\s\S]*?```)",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)|<((?:\./)?[^ >]+)>")


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


def _is_within_dir(path: Path, root: Path) -> bool:
    """Return whether a repository-relative path is at or below a trusted root."""
    return path == root or root in path.parents


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
            combined[changed.path] = "A" if changed.status == "A" else changed.status

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
    expected_name = Path(normalized_expected).name
    for target in targets:
        candidate = target.lstrip("./")
        if candidate == normalized_expected or candidate.endswith(f"/{normalized_expected}"):
            return True
        if Path(candidate).name == expected_name:
            return True
    return False


def _context_readme_link_diagnostics(
    changed_files: Iterable[ChangedFile],
    *,
    context_readme_text: str,
) -> list[Diagnostic]:
    """Flag added top-level docs/context notes missing from the context index."""
    diagnostics: list[Diagnostic] = []
    targets = _markdown_targets(context_readme_text)
    for changed in changed_files:
        if changed.status != "A":
            continue
        if changed.path.parent != _TOP_LEVEL_CONTEXT_DIR:
            continue
        if changed.path.suffix != ".md" or changed.path.name == "README.md":
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

    if path.suffix == ".md":
        link_targets = _markdown_targets(text)
        if any(_OUTPUT_PATH_RE.search(target) for target in link_targets):
            diagnostics.append(
                Diagnostic(
                    path=path,
                    message="tracked evidence should not link to ignored output/ artifacts",
                )
            )
    elif _OUTPUT_PATH_RE.search(scan_text):
        diagnostics.append(
            Diagnostic(
                path=path,
                message="tracked evidence should not point to ignored output/ artifacts",
            )
        )

    return diagnostics


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


def _file_diagnostics(path: Path, text: str) -> list[Diagnostic]:
    """Collect all diagnostics for one changed file."""
    diagnostics: list[Diagnostic] = []
    diagnostics.extend(_evidence_path_diagnostics(path, text))
    diagnostics.extend(_validation_phrase_diagnostics(path, text))
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
    return parser.parse_args()


def _selected_files(args: argparse.Namespace, repo_root: Path) -> list[ChangedFile]:
    """Resolve the changed file set from explicit paths or the branch diff."""
    if args.path:
        return [
            ChangedFile(
                status=(
                    "M"
                    if _path_exists_in_ref(
                        _normalize_path(Path(raw_path), repo_root), str(args.base), repo_root
                    )
                    else "A"
                ),
                path=_normalize_path(Path(raw_path), repo_root),
            )
            for raw_path in args.path
        ]
    return _changed_files(str(args.base), repo_root)


def main() -> int:
    """Run the docs/proof consistency checker."""
    args = _parse_args()
    repo_root = _repo_root()
    changed_files = _selected_files(args, repo_root)
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
