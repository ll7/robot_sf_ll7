#!/usr/bin/env python3
"""Warn when touched definitions still use TODO docstrings.

This checker scans git diff hunks (relative to a base ref) and warns if a
function/class touched in the diff still has a "TODO docstring" placeholder.
It is intentionally non-blocking unless --fail-on-warning is provided.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class DefInfo:
    """Metadata for a definition with a docstring."""

    name: str
    lineno: int
    end_lineno: int
    docstring: str


def _run(cmd: list[str], *, cwd: Path | None = None) -> str:
    """Run a command and return stdout.

    Returns:
        Captured standard output.
    """
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr.strip()}"
        )
    return proc.stdout


def _repo_root() -> Path:
    """Resolve the current git repository root.

    Returns:
        Absolute repository root path.
    """
    return Path(_run(["git", "rev-parse", "--show-toplevel"]).strip())


def _diff_text(base: str, repo_root: Path) -> str:
    """Read zero-context Python diff text against a base ref.

    Returns:
        Git diff text for changed Python files.
    """
    return _run(["git", "diff", "--unified=0", f"{base}...HEAD", "--", "*.py"], cwd=repo_root)


def _parse_changed_line_ranges(diff_text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse changed line ranges from unified diff text.

    Returns:
        Mapping from repository-relative path to changed new-file line ranges.
    """
    current_file: str | None = None
    ranges: dict[str, list[tuple[int, int]]] = {}
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[len("+++ b/") :].strip()
            ranges.setdefault(current_file, [])
            continue
        if not current_file:
            continue
        if line.startswith("@@"):
            # @@ -a,b +c,d @@
            try:
                hunk = line.split("+")[1]
                new_part = hunk.split("@@")[0].strip()
                if "," in new_part:
                    start_str, count_str = new_part.split(",", 1)
                    start = int(start_str)
                    count = int(count_str)
                else:
                    start = int(new_part)
                    count = 1
                if count <= 0:
                    continue
                end = start + count - 1
                ranges[current_file].append((start, end))
            except Exception:
                continue
    return {k: v for k, v in ranges.items() if v}


def _matches_any(path_str: str, patterns: Iterable[str]) -> bool:
    """Check whether a path matches any glob pattern.

    Returns:
        True when at least one pattern matches.
    """
    return any(fnmatch(path_str, pattern) for pattern in patterns)


def _parse_source(source: str, path: Path) -> ast.AST | None:
    """Parse Python source, reporting syntax errors without aborting the scan.

    Returns:
        Parsed AST, or ``None`` when parsing fails.
    """
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        print(f"Skipping {path} due to SyntaxError: {exc}", file=sys.stderr)
        return None


def _collect_defs(tree: ast.AST) -> list[DefInfo]:
    """Collect definitions that have docstrings from a parsed AST.

    Returns:
        Definition metadata with qualified names and source spans.
    """
    defs: list[DefInfo] = []

    class Visitor(ast.NodeVisitor):
        """AST visitor that records docstring-bearing definitions."""

        def __init__(self) -> None:
            self._stack: list[str] = []

        def _qual(self, name: str) -> str:
            """Build a dotted name from the current class stack.

            Returns:
                Qualified definition name.
            """
            return ".".join(self._stack + [name]) if self._stack else name

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            """Record a class definition and visit nested members."""
            doc = ast.get_docstring(node) or ""
            if doc:
                defs.append(
                    DefInfo(
                        name=self._qual(node.name),
                        lineno=node.lineno,
                        end_lineno=node.end_lineno or node.lineno,
                        docstring=doc,
                    )
                )
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            """Record a function definition and visit nested definitions."""
            doc = ast.get_docstring(node) or ""
            if doc:
                defs.append(
                    DefInfo(
                        name=self._qual(node.name),
                        lineno=node.lineno,
                        end_lineno=node.end_lineno or node.lineno,
                        docstring=doc,
                    )
                )
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            """Record an async function definition and visit nested definitions."""
            doc = ast.get_docstring(node) or ""
            if doc:
                defs.append(
                    DefInfo(
                        name=self._qual(node.name),
                        lineno=node.lineno,
                        end_lineno=node.end_lineno or node.lineno,
                        docstring=doc,
                    )
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return defs


def _read_defs(path: Path) -> list[DefInfo]:
    """Read and parse definitions from one source file.

    Returns:
        Definition metadata, or an empty list for missing/unparseable files.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    tree = _parse_source(source, path)
    if tree is None:
        return []
    return _collect_defs(tree)


def _overlaps(ranges: list[tuple[int, int]], start: int, end: int) -> bool:
    """Check whether a definition span overlaps any changed line range.

    Returns:
        True when any changed range intersects ``start`` through ``end``.
    """
    for r_start, r_end in ranges:
        if r_start <= end and r_end >= start:
            return True
    return False


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for the TODO-docstring checker.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        description="Warn on TODO docstrings in touched definitions (diff-only).",
    )
    parser.add_argument("--base", default="origin/main", help="Base ref to diff against")
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with code 1 if warnings are found",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern(s) to include (can be repeated)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern(s) to exclude (can be repeated)",
    )
    return parser.parse_args()


def main() -> int:
    """Run the diff-only TODO-docstring warning check."""
    args = _parse_args()
    repo_root = _repo_root()

    include_patterns = args.include or ["**/*.py"]
    exclude_patterns = args.exclude or []

    diff_text = _diff_text(args.base, repo_root)
    changed = _parse_changed_line_ranges(diff_text)
    if not changed:
        print("No changed Python files detected.")
        return 0

    warnings: list[str] = []
    for rel_path, ranges in sorted(changed.items()):
        if not _matches_any(rel_path, include_patterns) or _matches_any(rel_path, exclude_patterns):
            continue
        path = repo_root / rel_path
        defs = _read_defs(path)
        for info in defs:
            if "TODO docstring" not in info.docstring:
                continue
            if _overlaps(ranges, info.lineno, info.end_lineno):
                warnings.append(f"{rel_path}:{info.lineno} {info.name} has TODO docstring")

    if warnings:
        print("Docstring TODO warnings (touched definitions), please update the docstrings:")
        for line in warnings:
            print(f"- {line}")
        if args.fail_on_warning:
            return 1
    else:
        print("No TODO docstrings found in touched definitions.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
