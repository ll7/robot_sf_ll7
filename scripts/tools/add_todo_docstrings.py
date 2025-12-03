"""Add TODO placeholder docstrings where they are missing.

This utility inserts lightweight, Google-style docstrings so Ruff's docstring
rules pass. It only touches modules, classes, and functions that lack a
non-empty docstring. Placeholders include a `TODO docstring` marker to make them
simple to find and replace later.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "output",
    "results",
}

THIS_FILE = Path(__file__).resolve()

DocstringableNode = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@dataclass
class DocTarget:
    """Representation of a node needing a docstring."""

    node: DocstringableNode
    kind: str
    name: str
    lineno: int
    col_offset: int
    doc_expr: ast.Expr | None
    doc_value: str | None


def parse_args() -> argparse.Namespace:
    """Parse CLI options.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Insert TODO docstrings for modules, classes, and functions that are "
            "missing them. Existing empty docstrings are replaced."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Files or directories to scan (default: repository root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which files would change without writing them.",
    )
    parser.add_argument(
        "--include-notebooks",
        action="store_true",
        help="Include *.ipynb files (not modified; listed for awareness).",
    )
    return parser.parse_args()


def iter_python_files(paths: Sequence[str], include_notebooks: bool) -> Iterator[Path]:
    """Yield Python files under the provided paths, honoring exclusions."""

    for raw_path in paths:
        yield from _iter_path(Path(raw_path), include_notebooks)


def _is_excluded(path: Path) -> bool:
    """Check whether a path is in an excluded directory.

    Returns:
        bool: True if the path should be skipped.
    """

    return any(part in EXCLUDED_PARTS for part in path.parts)


def _iter_path(path: Path, include_notebooks: bool) -> Iterator[Path]:
    """Yield Python (and optionally notebook) files for a single path."""

    if path.resolve() == THIS_FILE or _is_excluded(path):
        return

    if path.is_dir():
        yield from _iter_directory(path, include_notebooks)
        return

    if path.suffix == ".py":
        yield path
    elif include_notebooks and path.suffix == ".ipynb":
        yield path


def _iter_directory(root: Path, include_notebooks: bool) -> Iterator[Path]:
    """Iterate over files in a directory respecting exclusions."""

    for child in root.rglob("*.py"):
        if _is_excluded(child) or child.resolve() == THIS_FILE:
            continue
        yield child

    if include_notebooks:
        for child in root.rglob("*.ipynb"):
            if _is_excluded(child):
                continue
            yield child


def collect_targets(tree: ast.Module) -> list[DocTarget]:
    """Collect all module/class/function nodes with their docstring metadata.

    Returns:
        list[DocTarget]: Targets that could need docstrings.
    """
    targets: list[DocTarget] = []

    module_doc = _extract_doc_expr(tree)
    targets.append(
        DocTarget(
            node=tree,
            kind="module",
            name="module",
            lineno=1,
            col_offset=0,
            doc_expr=module_doc,
            doc_value=ast.get_docstring(tree, clean=False),
        ),
    )

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            targets.append(
                DocTarget(
                    node=node,
                    kind="function",
                    name=node.name,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    doc_expr=_extract_doc_expr(node),
                    doc_value=ast.get_docstring(node, clean=False),
                ),
            )
        elif isinstance(node, ast.ClassDef):
            targets.append(
                DocTarget(
                    node=node,
                    kind="class",
                    name=node.name,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    doc_expr=_extract_doc_expr(node),
                    doc_value=ast.get_docstring(node, clean=False),
                ),
            )

    return targets


def _extract_doc_expr(node: DocstringableNode) -> ast.Expr | None:
    """Return the AST expression for the node's docstring if present.

    Returns:
        ast.Expr | None: Docstring expression or None.
    """
    if not node.body:
        return None
    first_stmt = node.body[0]
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        if isinstance(first_stmt.value.value, str):
            return first_stmt
    return None


def _should_add_doc(target: DocTarget) -> bool:
    """Decide whether a placeholder docstring is needed.

    Returns:
        bool: True when a placeholder should be added.
    """
    if target.doc_value is None:
        return True
    cleaned = target.doc_value.strip()
    if not cleaned:
        return True
    return "TODO docstring" in cleaned


def _function_returns_value(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Check whether the function yields or returns a non-None value.

    Returns:
        bool: True if a non-None value is returned or yielded.
    """

    class ReturnVisitor(ast.NodeVisitor):
        """Visitor that notes returns while ignoring nested defs."""

        def __init__(self) -> None:
            self.returns_value = False

        def visit_Return(self, return_node: ast.Return) -> None:
            if return_node.value is not None:
                if not (
                    isinstance(return_node.value, ast.Constant) and return_node.value.value is None
                ):
                    self.returns_value = True
            self.generic_visit(return_node)

        def visit_Yield(self, node: ast.Yield) -> None:
            self.returns_value = True

        def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
            self.returns_value = True

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return

    visitor = ReturnVisitor()
    visitor.visit(node)
    return visitor.returns_value


def _arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return argument names excluding self/cls.

    Returns:
        list[str]: Argument names to document.
    """
    names: list[str] = []
    args = node.args
    for arg in (*args.posonlyargs, *args.args):
        if arg.arg in {"self", "cls"}:
            continue
        names.append(arg.arg)
    if args.vararg:
        names.append(args.vararg.arg)
    for arg in args.kwonlyargs:
        names.append(arg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


def _build_docstring_lines(target: DocTarget, body_indent: str) -> list[str]:
    """Construct placeholder docstring lines for a node.

    Returns:
        list[str]: Docstring lines ready to inject.
    """
    summary_subject = {
        "module": "module",
        "class": "class",
        "function": "function",
    }.get(target.kind, "object")
    summary = f"TODO docstring. Document this {summary_subject}."

    include_args = isinstance(target.node, (ast.FunctionDef, ast.AsyncFunctionDef))
    arg_names = _arg_names(target.node) if include_args else []
    returns_value = (
        isinstance(target.node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and target.name != "__init__"
        and _function_returns_value(target.node)
    )

    if not arg_names and not returns_value:
        return [f'{body_indent}"""{summary}"""']

    lines = [f'{body_indent}"""{summary}']
    lines.append("")

    if arg_names:
        lines.append(f"{body_indent}Args:")
        for name in arg_names:
            lines.append(f"{body_indent}    {name}: TODO docstring.")

    if returns_value:
        lines.append("")
        lines.append(f"{body_indent}Returns:")
        lines.append(f"{body_indent}    TODO docstring.")

    lines.append(f'{body_indent}"""')
    return lines


def _header_indent(line: str) -> str:
    """Return the leading whitespace for a line.

    Returns:
        str: Leading whitespace characters.
    """
    stripped = line.lstrip()
    return line[: len(line) - len(stripped)]


def _compute_body_indent(lines: list[str], target: DocTarget) -> str:
    """Infer the indentation to use for a new docstring.

    Returns:
        str: Indentation to use.
    """
    node = target.node
    if isinstance(node, ast.Module):
        return ""

    header_idx = target.lineno - 1
    while header_idx < len(lines) and lines[header_idx].lstrip().startswith("@"):
        header_idx += 1

    existing_body_lines = [
        child.lineno - 1
        for child in getattr(node, "body", [])
        if not isinstance(child, ast.Expr)
        or not isinstance(child.value, ast.Constant)
        or not isinstance(child.value.value, str)
    ]
    if existing_body_lines:
        first_line = lines[min(existing_body_lines)]
        indent = _header_indent(first_line)
        if indent:
            return indent

    header_line = lines[header_idx]
    return f"{_header_indent(header_line)}    "


def _split_inline_body(header_line: str, body_indent: str) -> tuple[str, list[str]]:
    """Split a one-line body (e.g., `def foo(): pass`) from its header.

    Returns:
        tuple[str, list[str]]: Updated header and any trailing body lines.
    """
    if ":" not in header_line:
        return header_line, []

    prefix, suffix = header_line.rsplit(":", 1)
    if not suffix.strip():
        return header_line, []

    return f"{prefix.rstrip()}:", [f"{body_indent}{suffix.lstrip()}"]


def _module_insert_index(lines: list[str]) -> int:
    """Return the index where a module docstring should be inserted.

    Returns:
        int: Line index for module docstring insertion.
    """
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    if idx < len(lines) and lines[idx].lstrip().startswith("#") and "coding" in lines[idx]:
        idx += 1
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    return idx


def _apply_docstring(lines: list[str], target: DocTarget) -> bool:
    """Insert or replace a docstring for a target node.

    Returns:
        bool: True if modifications were applied.
    """
    if not _should_add_doc(target):
        return False

    body_indent = _compute_body_indent(lines, target)
    doc_lines = _build_docstring_lines(target, body_indent)

    if target.doc_expr is not None:
        start = target.doc_expr.lineno - 1
        end = target.doc_expr.end_lineno or start + 1
        lines[start:end] = doc_lines
        return True

    if isinstance(target.node, ast.Module):
        insert_at = _module_insert_index(lines)
        lines[insert_at:insert_at] = doc_lines
        return True

    header_idx = target.lineno - 1
    while header_idx < len(lines) and lines[header_idx].lstrip().startswith("@"):
        header_idx += 1

    header_line = lines[header_idx]
    if target.kind == "class":
        lines[header_idx] = header_line
        insert_at = header_idx + 1
        while insert_at < len(lines) and lines[insert_at].strip() == "":
            lines.pop(insert_at)
        lines[insert_at:insert_at] = doc_lines
        return True

    body_start_lineno = target.node.body[0].lineno if target.node.body else target.lineno + 1
    has_inline_body = body_start_lineno == target.lineno

    if has_inline_body:
        new_header, trailing_lines = _split_inline_body(header_line, body_indent)
        insert_at = header_idx + 1
    else:
        new_header, trailing_lines = header_line, []
        insert_at = max(header_idx + 1, body_start_lineno - 1)

    lines[header_idx] = new_header
    while insert_at < len(lines) and lines[insert_at].strip() == "":
        lines.pop(insert_at)

    lines[insert_at:insert_at] = doc_lines + trailing_lines
    return True


def apply_docstrings(path: Path, write: bool = True) -> bool:
    """Add placeholder docstrings to a file; return True if modified.

    Returns:
        bool: True when the file was changed.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    targets = collect_targets(tree)
    candidates = [t for t in targets if _should_add_doc(t)]
    if not candidates:
        return False

    module_targets = [t for t in candidates if t.kind == "module"]
    other_targets = [t for t in candidates if t.kind != "module"]

    lines = source.splitlines()
    changed = False
    for target in sorted(other_targets, key=lambda t: (t.lineno, t.col_offset), reverse=True):
        changed |= _apply_docstring(lines, target)
    for target in module_targets:
        changed |= _apply_docstring(lines, target)

    if changed and write:
        path.write_text(
            "\n".join(lines) + ("\n" if source.endswith("\n") else ""),
            encoding="utf-8",
        )
    return changed


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    python_files = list(iter_python_files(args.paths, include_notebooks=args.include_notebooks))

    changed_files: list[Path] = []
    notebook_hits: list[Path] = []
    for file_path in python_files:
        if file_path.suffix == ".ipynb":
            notebook_hits.append(file_path)
            continue
        if apply_docstrings(file_path, write=not args.dry_run):
            changed_files.append(file_path)

    if args.dry_run:
        for path in changed_files:
            print(f"[DRY RUN] Would update docstrings: {path}")
    else:
        for path in changed_files:
            print(f"Updated docstrings: {path}")

    if notebook_hits:
        print("Notebooks detected (not modified):")
        for path in notebook_hits:
            print(f" - {path}")

    if not changed_files:
        print("No docstring placeholders added.")


if __name__ == "__main__":
    main()
