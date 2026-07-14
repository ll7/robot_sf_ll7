"""Guard changed Python files against unmarked evidence-tree writers."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

EVIDENCE_PATH_FRAGMENT = "docs/context/evidence"
EXEMPTION_PATTERN = re.compile(r"#\s*evidence-writer-exempt:\s*(.*)$", re.IGNORECASE)
WRITE_METHODS = frozenset({"write_bytes", "write_text"})


def _has_write_mode(call: ast.Call) -> bool:
    """Return whether an ``open`` call requests a write-capable mode."""
    mode: ast.expr | None = None
    if len(call.args) >= 2:
        mode = call.args[1]
    for keyword in call.keywords:
        if keyword.arg == "mode":
            mode = keyword.value
            break
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(flag in mode.value for flag in ("w", "a", "x", "+"))
    return False


class _DirectWriterVisitor(ast.NodeVisitor):
    """Find calls that can write generated output without the shared module."""

    def __init__(self) -> None:
        self.violations: list[tuple[int, str]] = []

    def _record(self, node: ast.Call, operation: str) -> None:
        self.violations.append((node.lineno, operation))

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        if isinstance(function, ast.Attribute):
            if function.attr in WRITE_METHODS:
                self._record(node, f".{function.attr}()")
            elif function.attr == "open" and _has_write_mode(node):
                self._record(node, ".open(..., write mode)")
            elif function.attr == "DictWriter":
                self._record(node, "csv.DictWriter()")
            elif function.attr == "dump" and isinstance(function.value, ast.Name):
                if function.value.id == "json":
                    self._record(node, "json.dump()")
            elif function.attr in {"_write_sha256sums", "write_sha256sums"}:
                if not (isinstance(function.value, ast.Name) and function.value.id == "writers"):
                    self._record(node, f"{function.attr}()")
        elif isinstance(function, ast.Name) and function.id == "open" and _has_write_mode(node):
            self._record(node, "open(..., write mode)")
        self.generic_visit(node)


def _exemption(source: str) -> tuple[bool, str | None]:
    """Return whether source has a valid file-level exemption."""
    for line_number, line in enumerate(source.splitlines(), start=1):
        match = EXEMPTION_PATTERN.search(line)
        if match is None:
            continue
        reason = match.group(1).strip()
        if not reason:
            return True, f"line {line_number} has an empty evidence-writer exemption reason"
        return True, None
    return False, None


def check_file(path: str | Path) -> list[str]:
    """Return fail-closed guard messages for one changed Python file."""
    source_path = Path(path)
    if source_path.suffix != ".py" or not source_path.is_file():
        return []
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"BLOCKER: could not read changed Python file '{source_path}': {exc}"]

    has_exemption, exemption_error = _exemption(source)
    if exemption_error is not None:
        return [f"BLOCKER: evidence-writer exemption in '{source_path}' {exemption_error}"]
    if has_exemption:
        return []
    if EVIDENCE_PATH_FRAGMENT not in source:
        return []

    try:
        tree = ast.parse(source, filename=str(source_path))
    except SyntaxError as exc:
        return [f"BLOCKER: cannot parse changed Python file '{source_path}': {exc}"]

    visitor = _DirectWriterVisitor()
    visitor.visit(tree)
    return [
        f"BLOCKER: '{source_path}:{line}' directly uses {operation} in a file that writes "
        "generated evidence. Use robot_sf.evidence.writers.write_json/write_csv/write_text/"
        "write_sha256sums, or add a justified '# evidence-writer-exempt: <reason>' comment."
        for line, operation in visitor.violations
    ]


def _is_changed_from_base(path: str | Path, base_ref: str) -> bool:
    """Return whether a repository file differs from ``base_ref``.

    Historical PR regression checks replay old file lists against the current
    checkout. Skipping files that are no longer changed avoids re-linting those
    old paths; an unavailable base is treated as changed so the guard fails
    closed in CI.
    """
    source_path = Path(path)
    try:
        repo_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        relative_path = source_path.resolve().relative_to(repo_root.resolve())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return True
    result = subprocess.run(
        ["git", "diff", "--quiet", base_ref, "--", relative_path.as_posix()],
        capture_output=True,
        check=False,
    )
    return result.returncode != 0


def check_changed_files(changed_files: list[str], base_ref: str = "origin/main") -> list[str]:
    """Check only changed Python files, preserving the PR contract boundary."""
    blockers: list[str] = []
    for path in changed_files:
        if not _is_changed_from_base(path, base_ref):
            continue
        blockers.extend(check_file(path))
    return blockers


def main() -> int:
    """Run the changed-file evidence-writer guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--changed-files-file",
        type=Path,
        required=True,
        help="newline-delimited changed-file paths",
    )
    parser.add_argument("--base-ref", default="origin/main", help="git base ref")
    args = parser.parse_args()
    changed_files = [
        line.strip()
        for line in args.changed_files_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blockers = check_changed_files(changed_files, args.base_ref)
    for blocker in blockers:
        print(blocker)
    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
