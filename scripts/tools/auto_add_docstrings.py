"""Utility to inject placeholder docstrings across the repository.

The script parses Python files with LibCST so that indentation, comments, and
formatting are preserved. It emits docstrings for every module, class, and
function lacking one, ensuring Ruff docstring rules (D100–D107, D417, D419)
have baseline coverage. Generated docstrings include Args/Returns sections so
contributors can fill them in later with richer explanations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst import matchers as m

SKIP_PARTS = {
    ".git",
    ".venv",
    ".ruff_cache",
    "__pycache__",
    "dist",
    "build",
    "output",
    ".uv-cache",
    "maps/svg_maps/cache",
}


def should_skip(path: Path) -> bool:
    """Return True when a given file path should be skipped."""

    return any(part in SKIP_PARTS for part in path.parts)


def titleize(name: str) -> str:
    """Convert snake_case or dunder identifiers to a human-friendly phrase."""

    cleaned = name.strip("_").replace("_", " ").strip()
    if not cleaned:
        cleaned = name.strip() or "Item"
    cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else "Item"
    return cleaned


def format_docstring(summary: str, args: list[str] | None, returns: str | None) -> str:
    """Build a docstring literal with Args/Returns sections when necessary."""

    lines: list[str] = [summary]
    if args:
        lines.append("")
        lines.append("Args:")
        for arg in args:
            lines.append(f"    {arg}: Auto-generated placeholder description.")
    if returns:
        lines.append("")
        lines.append("Returns:")
        lines.append(f"    {returns}: Auto-generated placeholder description.")

    body = "\n".join(lines)
    return f'"""{body}\n"""'


MODULE_RENDERER = cst.Module(body=())


def annotation_name(annotation: cst.Annotation | None) -> str:
    """Return the source text for an annotation, defaulting to ``Any``."""

    if annotation is None:
        return "Any"
    try:
        return MODULE_RENDERER.code_for_node(annotation.annotation)
    except Exception:
        return "Any"


def function_params(node: cst.FunctionDef | cst.AsyncFunctionDef) -> list[str]:
    """Return the ordered parameter names, excluding self/cls."""

    params: list[str] = []
    for param in node.params.posonly_params + node.params.params + node.params.kwonly_params:
        name = param.name.value
        if name in {"self", "cls"}:
            continue
        params.append(name)
    star_arg = node.params.star_arg
    if isinstance(star_arg, cst.Param):
        name = star_arg.name.value
        if name not in {"self", "cls"}:
            params.append(name)
    star_kwarg = node.params.star_kwarg
    if isinstance(star_kwarg, cst.Param):
        name = star_kwarg.name.value
        if name not in {"self", "cls"}:
            params.append(name)
    return params


def make_docstring_line(doc_text: str) -> cst.SimpleStatementLine:
    """Return a SimpleStatementLine containing the given docstring literal."""

    return cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString(doc_text))])


def prepend_docstring(block: cst.BaseSuite, doc_text: str) -> cst.BaseSuite:
    """Return a suite with the docstring injected as the first statement."""

    if isinstance(block, cst.IndentedBlock):
        doc_stmt = make_docstring_line(doc_text)
        return block.with_changes(body=[doc_stmt, *block.body])
    if isinstance(block, cst.SimpleStatementSuite):
        doc_expr = cst.Expr(cst.SimpleString(doc_text))
        return block.with_changes(body=[doc_expr, *block.body])
    raise TypeError(f"Unsupported suite type: {type(block)}")


DOCSTRING_MATCHER = m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())])


def module_has_docstring(node: cst.Module) -> bool:
    """Return True when the module already includes a top-level docstring."""

    if not node.body:
        return False
    return m.matches(node.body[0], DOCSTRING_MATCHER)


def suite_has_docstring(block: cst.BaseSuite) -> bool:
    """Return True when an indented block starts with a docstring."""

    if isinstance(block, cst.IndentedBlock):
        body = block.body
    elif isinstance(block, cst.SimpleStatementSuite):
        body = block.body
    else:
        body = []
    if not body:
        return False
    return m.matches(body[0], DOCSTRING_MATCHER)


@dataclass
class DocstringTransformer(cst.CSTTransformer):
    """CST transformer that injects docstrings where missing."""

    file_path: Path
    changed: bool = False

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        """Leave Module.

        Args:
            original_node: Auto-generated placeholder description.
            updated_node: Auto-generated placeholder description.

        Returns:
            cst.Module: Auto-generated placeholder description.
        """
        if module_has_docstring(original_node):
            return updated_node
        summary = f"Module {self.file_path.stem} auto-generated docstring."
        doc_text = format_docstring(summary, args=None, returns=None)
        doc_stmt = make_docstring_line(doc_text)
        new_body = [doc_stmt, *updated_node.body]
        self.changed = True
        return updated_node.with_changes(body=new_body)

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.CSTNode:
        """Leave ClassDef.

        Args:
            original_node: Auto-generated placeholder description.
            updated_node: Auto-generated placeholder description.

        Returns:
            cst.CSTNode: Auto-generated placeholder description.
        """
        if suite_has_docstring(original_node.body):
            return updated_node
        summary = f"{titleize(original_node.name.value)} class."
        doc_text = format_docstring(summary, args=None, returns=None)
        new_body = prepend_docstring(updated_node.body, doc_text)
        self.changed = True
        return updated_node.with_changes(body=new_body)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.CSTNode:
        """Leave FunctionDef.

        Args:
            original_node: Auto-generated placeholder description.
            updated_node: Auto-generated placeholder description.

        Returns:
            cst.CSTNode: Auto-generated placeholder description.
        """
        if suite_has_docstring(original_node.body):
            return updated_node
        summary = f"{titleize(original_node.name.value)}."
        params = function_params(original_node)
        returns = annotation_name(original_node.returns)
        doc_text = format_docstring(summary, args=params, returns=returns)
        new_body = prepend_docstring(updated_node.body, doc_text)
        self.changed = True
        return updated_node.with_changes(body=new_body)

    def leave_AsyncFunctionDef(
        self, original_node: cst.AsyncFunctionDef, updated_node: cst.AsyncFunctionDef
    ) -> cst.CSTNode:
        """Leave AsyncFunctionDef.

        Args:
            original_node: Auto-generated placeholder description.
            updated_node: Auto-generated placeholder description.

        Returns:
            cst.CSTNode: Auto-generated placeholder description.
        """
        if suite_has_docstring(original_node.body):
            return updated_node
        summary = f"{titleize(original_node.name.value)}."
        params = function_params(original_node)
        returns = annotation_name(original_node.returns)
        doc_text = format_docstring(summary, args=params, returns=returns)
        new_body = prepend_docstring(updated_node.body, doc_text)
        self.changed = True
        return updated_node.with_changes(body=new_body)


def transform_file(path: Path) -> bool:
    """Parse, transform, and rewrite a file. Returns True if modified."""

    try:
        module = cst.parse_module(path.read_text())
    except Exception as exc:  # pragma: no cover - defensive parse guard
        raise RuntimeError(f"Failed to parse {path}") from exc

    transformer = DocstringTransformer(file_path=path)
    updated = module.visit(transformer)
    if transformer.changed:
        path.write_text(updated.code)
    return transformer.changed


def iter_python_files(paths: list[Path]) -> list[Path]:
    """Yield Python files residing under provided directories/files."""

    files: list[Path] = []
    for target in paths:
        if target.is_file() and target.suffix == ".py":
            if not should_skip(target):
                files.append(target)
            continue
        if target.is_dir():
            for file_path in target.rglob("*.py"):
                if should_skip(file_path):
                    continue
                files.append(file_path)
    return files


def main() -> None:
    """Main.

    Returns:
        None: Auto-generated placeholder description.
    """
    parser = argparse.ArgumentParser(description="Auto-add placeholder docstrings.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=[
            "robot_sf",
            "fast-pysf",
            "scripts",
            "examples",
            "tests",
            "test_pygame",
            "docs",
            "utilities",
            "SLURM",
            "svg_conv",
            "hooks",
            "setup.py",
        ],
        help="Directories or files to process.",
    )
    args = parser.parse_args()
    targets = [Path(p) for p in args.paths]
    files = iter_python_files(targets)
    modified = 0
    for file_path in files:
        if transform_file(file_path):
            modified += 1
    print(f"Updated {modified} files.")


if __name__ == "__main__":
    main()
