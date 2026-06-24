#!/usr/bin/env python3
"""Inventory and ratchet broad exception handlers in benchmark/script surfaces."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_BASELINE = Path("scripts/validation/broad_exception_baseline.json")
DEFAULT_PATTERNS = ("robot_sf/benchmark/**/*.py", "scripts/**/*.py")
BROAD_NAMES = {"Exception", "BaseException"}


@dataclass(frozen=True)
class BroadExceptionEntry:
    """A reviewable broad exception handler occurrence."""

    path: str
    lineno: int
    col_offset: int
    handler: str
    context: str
    source: str
    fingerprint: str


def _repo_root() -> Path:
    """Return the current Git repository root."""
    return Path(_run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).strip())


def _run(cmd: list[str], *, cwd: Path) -> str:
    """Run a command and return stdout, raising on failure."""
    proc = __import__("subprocess").run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return proc.stdout


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    """Return whether ``path`` matches one of the ratcheted surface globs."""
    return any(
        fnmatch.fnmatch(path, pattern)
        or ("/**/" in pattern and fnmatch.fnmatch(path, pattern.replace("/**/", "/")))
        for pattern in patterns
    )


def _python_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Collect tracked Python files under the configured surface globs."""
    files = _run(["git", "ls-files", "*.py"], cwd=root).splitlines()
    return [root / path for path in sorted(files) if _matches_any(path, patterns)]


def _handler_name(node: ast.expr | None) -> str | None:
    """Return broad handler spelling when an except handler catches too broadly."""
    if node is None:
        return "bare"
    if isinstance(node, ast.Name) and node.id in BROAD_NAMES:
        return node.id
    if isinstance(node, ast.Attribute) and node.attr in BROAD_NAMES:
        return node.attr
    if isinstance(node, ast.Tuple):
        names = [_handler_name(element) for element in node.elts]
        if any(name in BROAD_NAMES or name == "bare" for name in names):
            return "(" + ", ".join(_expr_label(element) for element in node.elts) + ")"
    return None


def _expr_label(node: ast.expr) -> str:
    """Render a compact exception expression label."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return type(node).__name__


class _Visitor(ast.NodeVisitor):
    """AST visitor that records broad exception handlers with enclosing context."""

    def __init__(self, path: str, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.context: list[str] = []
        self.entries: list[BroadExceptionEntry] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.context.append(node.name)
        self.generic_visit(node)
        self.context.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> Any:
        handler = _handler_name(node.type)
        if handler is not None:
            source = self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else ""
            context = ".".join(self.context) or "<module>"
            fingerprint_payload = {
                "path": self.path,
                "handler": handler,
                "context": context,
                "source": source,
            }
            fingerprint = hashlib.sha256(
                json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16]
            self.entries.append(
                BroadExceptionEntry(
                    path=self.path,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    handler=handler,
                    context=context,
                    source=source,
                    fingerprint=fingerprint,
                )
            )
        self.generic_visit(node)


def inventory(root: Path, patterns: tuple[str, ...]) -> list[BroadExceptionEntry]:
    """Return deterministic broad exception inventory entries."""
    entries: list[BroadExceptionEntry] = []
    for path in _python_files(root, patterns):
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=rel)
        visitor = _Visitor(rel, source.splitlines())
        visitor.visit(tree)
        entries.extend(visitor.entries)
    return sorted(entries, key=lambda entry: (entry.path, entry.lineno, entry.col_offset))


def _counts(entries: list[BroadExceptionEntry]) -> dict[str, Any]:
    """Build stable summary counts for baseline review."""
    by_path: dict[str, int] = {}
    for entry in entries:
        by_path[entry.path] = by_path.get(entry.path, 0) + 1
    return {"total": len(entries), "by_path": dict(sorted(by_path.items()))}


def _baseline_payload(
    entries: list[BroadExceptionEntry], patterns: tuple[str, ...]
) -> dict[str, Any]:
    """Build the versioned baseline JSON payload."""
    return {
        "schema_version": 1,
        "description": (
            "Broad exception handler baseline for benchmark/script surfaces. "
            "Run scripts/validation/check_broad_exceptions.py --write-baseline to refresh "
            "after intentionally removing or approving entries."
        ),
        "patterns": list(patterns),
        "counts": _counts(entries),
        "entries": [asdict(entry) for entry in entries],
    }


def _load_baseline(path: Path) -> dict[str, Any]:
    """Load and minimally validate a baseline file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported baseline schema_version in {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, reviewable JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_against_baseline(
    current: list[BroadExceptionEntry], baseline: dict[str, Any]
) -> list[str]:
    """Return ratchet failure messages for new or increased broad catches."""
    failures: list[str] = []
    baseline_entries = baseline.get("entries", [])
    baseline_fingerprints = {entry["fingerprint"] for entry in baseline_entries}
    current_fingerprints = {entry.fingerprint for entry in current}
    if len(current) > len(baseline_entries):
        failures.append(
            f"Broad exception count increased from {len(baseline_entries)} to {len(current)}."
        )
    new_entries = [entry for entry in current if entry.fingerprint not in baseline_fingerprints]
    if new_entries:
        failures.append("Unapproved broad exception handlers were added:")
        failures.extend(
            f"  {entry.path}:{entry.lineno}: {entry.source}" for entry in new_entries[:20]
        )
        if len(new_entries) > 20:
            failures.append(f"  ... {len(new_entries) - 20} more")
    removed = baseline_fingerprints - current_fingerprints
    if removed and not failures:
        failures.append(
            "Broad exception handlers were removed; refresh the baseline to ratchet downward."
        )
    return failures


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--pattern", action="append", dest="patterns")
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="Optional inventory output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run inventory or ratchet check."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve() if args.root is not None else _repo_root()
    patterns = tuple(args.patterns or DEFAULT_PATTERNS)
    entries = inventory(root, patterns)
    payload = _baseline_payload(entries, patterns)
    if args.json is not None:
        _write_json(args.json, payload)
    if args.write_baseline:
        _write_json(root / args.baseline, payload)
        print(f"Wrote {len(entries)} broad exception entries to {args.baseline}")
        return 0

    baseline_path = root / args.baseline
    baseline = _load_baseline(baseline_path)
    failures = check_against_baseline(entries, baseline)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Broad exception ratchet passed with {len(entries)} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
