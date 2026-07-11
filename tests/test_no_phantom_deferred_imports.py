"""Ratchet deferred first-party imports that collection-time tests cannot see.

The resource-lifecycle subprocess worker once deferred an import of the nonexistent
``robot_sf.benchmark.scenario_matrix`` module.  Import-time collection did not execute
that function, and an earlier serialization failure hid the problem in Slurm jobs 13344
and 13364.  The original resource-lifecycle-only check from issue #4826 is generalized
here for issue #5242: every deferred ``robot_sf`` import must resolve its module and
requested symbols, unless its nearest enclosing ``try`` explicitly catches an optional
dependency's ``ImportError`` or ``ModuleNotFoundError``.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ROBOT_SF_ROOT = _REPO_ROOT / "robot_sf"
_OPTIONAL_IMPORT_EXCEPTIONS = {"ImportError", "ModuleNotFoundError"}


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    """Return each AST node's direct parent for structural scope checks."""
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _is_deferred(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether ``node`` is nested under a function or method definition."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        current = parents.get(current)
    return False


def _exception_names(node: ast.expr | None) -> Iterable[str]:
    """Yield exception names from a direct exception or its tuple form."""
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Tuple):
        for element in node.elts:
            yield from _exception_names(element)


def _is_optional_dependency_guard(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether the nearest enclosing try explicitly guards an optional import."""
    current = parents.get(node)
    while current is not None:
        if isinstance(current, ast.Try):
            return any(
                _OPTIONAL_IMPORT_EXCEPTIONS.intersection(_exception_names(handler.type))
                for handler in current.handlers
            )
        current = parents.get(current)
    return False


def _deferred_robot_sf_imports(path: Path) -> list[tuple[int, str, tuple[str, ...]]]:
    """Collect unguarded first-party imports nested inside functions in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    parents = _parent_map(tree)
    imports: list[tuple[int, str, tuple[str, ...]]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if not _is_deferred(node, parents) or _is_optional_dependency_guard(node, parents):
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.startswith("robot_sf"):
                imports.append(
                    (node.lineno, node.module, tuple(alias.name for alias in node.names))
                )
        else:
            imports.extend(
                (node.lineno, alias.name, ())
                for alias in node.names
                if alias.name.startswith("robot_sf")
            )

    return imports


def _phantom_imports(root: Path) -> list[str]:
    """Return actionable diagnostics for deferred imports that cannot resolve."""
    problems: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line, module_name, imported_names in _deferred_robot_sf_imports(path):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # A deferred module that raises cannot satisfy the contract.
                problems.append(f"{path}:{line}: {module_name}: module does not resolve ({exc})")
                continue
            for imported_name in imported_names:
                if imported_name != "*" and not hasattr(module, imported_name):
                    problems.append(
                        f"{path}:{line}: {module_name}.{imported_name}: name missing from module"
                    )
    return problems


def test_no_phantom_deferred_robot_sf_imports() -> None:
    """Every unguarded deferred first-party import resolves its module and symbols."""
    problems = _phantom_imports(_ROBOT_SF_ROOT)
    assert not problems, "phantom deferred robot_sf imports:\n  " + "\n  ".join(problems)


def test_phantom_deferred_import_reports_file_line_and_symbol(tmp_path: Path) -> None:
    """A nonexistent deferred import gives a human enough information to repair it."""
    source = tmp_path / "robot_sf" / "probe.py"
    source.parent.mkdir()
    source.write_text("def load():\n    from robot_sf.does_not_exist import missing\n")

    assert _phantom_imports(source.parent) == [
        f"{source}:2: robot_sf.does_not_exist: module does not resolve "
        "(No module named 'robot_sf.does_not_exist')"
    ]


def test_optional_import_guard_is_excluded(tmp_path: Path) -> None:
    """An explicit ImportError guard remains available for optional dependencies."""
    source = tmp_path / "robot_sf" / "optional.py"
    source.parent.mkdir()
    source.write_text(
        "def load():\n"
        "    try:\n"
        "        from robot_sf.does_not_exist import missing\n"
        "    except (ImportError, ModuleNotFoundError):\n"
        "        return None\n"
    )

    assert _phantom_imports(source.parent) == []
