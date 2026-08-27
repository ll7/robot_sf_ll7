#!/usr/bin/env python3
"""AST-based function-length audit for first-party Python (issue #7899).

Parses Python with ``ast`` (never line-oriented regex), records every
function/method with its inclusive source-line count, and reports functions
over a declared threshold (default 200 lines).  Supports a fixture-root test
mode, deterministic JSON/Markdown output, and an explicit ``--check``
allowlist format without introducing a broad repository CI gate.

No production code is modified; the audit is read-only over the declared root.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = "function_length_audit.v1"
DEFAULT_THRESHOLD = 200
DEFAULT_ROOT = "robot_sf"

#: Versioned exclusion list with rationale.  Empty implicit exclusions are
#: preferred; every entry here is explicit and justified.  ``<root>/__init__.py``
#: matches the package-namespace init at the scan root.
EXCLUSIONS: dict[str, str] = {
    "<root>/__init__.py": "package namespace re-exports only",
}

#: Vendored/generated/stub path fragments that are never scanned.
VENDORED_FRAGMENTS = ("third_party", "vendor", "generated", "protobuf", "grpc")


@dataclass(frozen=True)
class FunctionFinding:
    """One audited function/method record."""

    module: str
    qualified_name: str
    start_line: int
    end_line: int
    inclusive_lines: int
    kind: str
    nesting: str
    file_digest: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return asdict(self)


def _file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_excluded(relative: str, exclusions: dict[str, str]) -> str | None:
    """Return the exclusion rationale when a path is excluded, else ``None``.

    Exact relative paths match directly; a path whose basename is ``__init__.py``
    at the scan root matches the ``<root>/__init__.py`` key so versioned package
    namespace exclusions survive root-relative scanning.
    """
    if relative in exclusions:
        return exclusions[relative]
    if relative == "__init__.py" and "<root>/__init__.py" in exclusions:
        return exclusions["<root>/__init__.py"]
    for fragment in VENDORED_FRAGMENTS:
        if fragment in relative:
            return f"vendored/generated fragment {fragment!r}"
    return None


def _python_files(root: Path) -> list[Path]:
    """Return all first-party ``.py`` files under ``root`` (sorted)."""
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _node_lines(node: ast.AST) -> tuple[int, int]:
    """Return the inclusive ``(start, end)`` source lines of a node."""
    return (int(node.lineno), int(node.end_lineno or node.lineno))


def _iter_functions(tree: ast.AST, module: str) -> list[FunctionFinding]:
    """Walk the AST and record every function/method with its nesting context."""
    findings: list[FunctionFinding] = []

    def visit_body(
        body: list[ast.stmt],
        *,
        prefix: str,
        nesting: str,
        class_stack: list[str],
    ) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
                if class_stack:
                    kind = "async_method" if kind == "async_function" else "method"
                start, end = _node_lines(node)
                name = ".".join([*class_stack, node.name]) if class_stack else node.name
                qualified = f"{prefix}.{name}" if prefix else name
                findings.append(
                    FunctionFinding(
                        module=module,
                        qualified_name=qualified,
                        start_line=start,
                        end_line=end,
                        inclusive_lines=end - start + 1,
                        kind=kind,
                        nesting=nesting,
                        file_digest="",
                    )
                )
                visit_body(
                    node.body,
                    prefix=qualified,
                    nesting=nesting + 1,
                    class_stack=class_stack,
                )
            elif isinstance(node, ast.ClassDef):
                start, end = _node_lines(node)
                visit_body(
                    node.body,
                    prefix=prefix,
                    nesting=nesting + 1,
                    class_stack=[*class_stack, node.name],
                )
                del start, end

    visit_body(tree.body, prefix="", nesting=0, class_stack=[])
    return findings


def scan_file(path: Path, *, threshold: int, root: Path) -> list[FunctionFinding]:
    """Scan one Python file and return functions over the threshold.

    Raises:
        SyntaxError: When the file cannot be parsed (fail closed).
        ValueError: When the file cannot be read.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unreadable file {path}: {exc}") from exc
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise SyntaxError(f"syntax error in {path}: {exc.msg} (line {exc.lineno})") from exc

    module = str(path.relative_to(root)).replace("\\", "/").removesuffix(".py")
    digest = _file_digest(path)
    findings = _iter_functions(tree, module)
    return [
        FunctionFinding(
            module=finding.module,
            qualified_name=finding.qualified_name,
            start_line=finding.start_line,
            end_line=finding.end_line,
            inclusive_lines=finding.inclusive_lines,
            kind=finding.kind,
            nesting=finding.nesting,
            file_digest=digest,
        )
        for finding in findings
        if finding.inclusive_lines > threshold
    ]


def run_audit(
    root: Path,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    exclusions: dict[str, str] | None = None,
    include_all: bool = False,
) -> dict[str, Any]:
    """Scan ``root`` and return the versioned audit report."""
    active_exclusions = dict(exclusions or EXCLUSIONS)
    files = _python_files(root)
    scanned: list[FunctionFinding] = []
    excluded_paths: list[dict[str, str]] = []
    for path in files:
        relative = str(path.relative_to(root)).replace("\\", "/")
        reason = _is_excluded(relative, active_exclusions)
        if reason is not None and not include_all:
            excluded_paths.append({"path": relative, "rationale": reason})
            continue
        try:
            scanned.extend(scan_file(path, threshold=threshold, root=root))
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc

    over = sorted(scanned, key=lambda finding: (-finding.inclusive_lines, finding.qualified_name))
    return {
        "schema": SCHEMA,
        "threshold": threshold,
        "root": str(root),
        "scan": {
            "file_count": len(files),
            "scanned_file_count": len(files) - len(excluded_paths),
            "excluded_count": len(excluded_paths),
            "exclusions": active_exclusions,
        },
        "findings_count": len(over),
        "findings": [finding.as_dict() for finding in over],
        "excluded_paths": excluded_paths,
    }


def _allowlist_load(path: Path) -> dict[str, int]:
    """Load an allowlist of ``qualified_name -> inclusive_lines``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("allowlist must be a JSON object mapping names to line counts")
    return {str(key): int(value) for key, value in payload.items()}


def run_check(report: dict[str, Any], allowlist: dict[str, int]) -> tuple[int, list[str]]:
    """Return ``(exit_code, problems)`` for the audit against an allowlist."""
    problems: list[str] = []
    allowlist_names = set(allowlist)
    for finding in report["findings"]:
        name = finding["qualified_name"]
        if name not in allowlist_names:
            problems.append(f"function over threshold without allowlist entry: {name}")
            continue
        if allowlist[name] < finding["inclusive_lines"]:
            problems.append(
                f"function {name} grew from {allowlist[name]} to {finding['inclusive_lines']} lines"
            )
    return (1 if problems else 0, problems)


def main(argv: list[str] | None = None) -> int:
    """Run the audit CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--check", type=Path, default=None, help="allowlist JSON path")
    parser.add_argument("--markdown", action="store_true", help="emit a Markdown summary")
    parser.add_argument(
        "--include-all", action="store_true", help="scan even excluded paths (for fixtures)"
    )
    parser.add_argument(
        "--report", type=Path, default=None, help="write the JSON report to this path"
    )
    args = parser.parse_args(argv)

    try:
        report = run_audit(args.root, threshold=args.threshold, include_all=args.include_all)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, sort_keys=True))
        return 2

    if args.check is not None:
        try:
            allowlist = _allowlist_load(args.check)
        except (OSError, ValueError) as exc:
            print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, sort_keys=True))
            return 2
        exit_code, problems = run_check(report, allowlist)
        if problems:
            for problem in problems:
                print(f"check failed: {problem}")
            return exit_code

    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown:
        print(_markdown_summary(report))
    elif args.report is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _markdown_summary(report: dict[str, Any]) -> str:
    """Render a concise Markdown summary of the audit."""
    lines = [
        f"# Function-length audit (`{report['schema']}`)",
        "",
        f"- Root: `{report['root']}` | threshold: `{report['threshold']}` lines",
        f"- Files: {report['scan']['scanned_file_count']} scanned / "
        f"{report['scan']['file_count']} total ({report['scan']['excluded_count']} excluded)",
        f"- Functions over threshold: **{report['findings_count']}**",
        "",
        "| Module | Function | Lines | Kind |",
        "| --- | --- | --- | --- |",
    ]
    for finding in report["findings"]:
        lines.append(
            f"| `{finding['module']}` | `{finding['qualified_name']}` | "
            f"{finding['inclusive_lines']} | {finding['kind']} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
