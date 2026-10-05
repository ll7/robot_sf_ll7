"""Conservatively select existing tests affected by a committed Git diff.

Imports include transitive local dependencies; literal paths include directory
pins and joined Path expressions. Dynamic dependency construction remains a
reason to run the complete suite on the combined train head.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


def changed_paths(root: Path, base: str, head: str = "HEAD") -> set[str]:
    """Include deleted paths and both rename endpoints without executing them."""
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=ACDMRT",
            f"{base}...{head}",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    fields = iter(result.stdout.decode().split("\0")[:-1])
    paths = set()
    for status in fields:
        paths.add(next(fields))
        if status.startswith(("R", "C")):
            paths.add(next(fields))
    return paths


def _module(path: str) -> str:
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _joined_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        left, right = _joined_literal(node.left), _joined_literal(node.right)
        if isinstance(node.op, ast.Div) and left is None:
            return right
        if left is not None and right is not None:
            return left + ("/" if isinstance(node.op, ast.Div) else "") + right
    if isinstance(node, ast.Call) and node.args:
        if isinstance(node.func, ast.Name) and node.func.id in {"Path", "PurePath"}:
            return _joined_literal(node.args[0])
    return None


def _references(path: Path, relative: str) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    imports, literals = set(), set()
    package = _module(relative).split(".")[:-1]
    if path.name == "__init__.py":
        package = _module(relative).split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            if node.level:
                prefix = ".".join(
                    package[: len(package) - node.level + 1] + ([prefix] if prefix else [])
                )
            imports.add(prefix)
            imports.update(f"{prefix}.{alias.name}" for alias in node.names)
        literal = _joined_literal(node)
        if literal:
            literals.add(literal.strip("/"))
    return imports, literals


def affected_tests(root: Path, paths: set[str]) -> list[str]:
    """Return existing tests importing changed code or naming changed inputs."""
    sources = {}
    for directory in ("robot_sf", "scripts", "tests", "fast-pysf"):
        for path in (root / directory).rglob("*.py"):
            relative = path.relative_to(root).as_posix()
            sources[relative] = _references(path, relative)
    affected = set(paths)
    while True:
        modules = {_module(path) for path in affected if path.endswith(".py")}
        additions = set()
        for path, (imports, literals) in sources.items():
            imported = any(
                name == module or name.startswith(module + ".")
                for name in imports
                for module in modules
            )
            pinned = any(
                changed == literal or changed.startswith(literal.rstrip("/") + "/")
                for literal in literals
                if "/" in literal
                for changed in affected
            )
            if imported or pinned:
                additions.add(path)
        if additions <= affected:
            break
        affected.update(additions)
    # CI support tests cover shell and workflow inputs that cannot be imported.
    if any(path.startswith(("scripts/ci/", ".github/workflows/")) for path in paths):
        affected.update(path for path in sources if path.startswith("tests/ci/"))
    return sorted(
        path
        for path in affected
        if path in sources
        and (path.startswith("tests/") or path.startswith("fast-pysf/tests/"))
        and (Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py"))
    )


def main() -> None:
    """Print newline-delimited existing affected test files, failing on bad refs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    print("\n".join(affected_tests(root, changed_paths(root, args.base, args.head))))


if __name__ == "__main__":
    main()
