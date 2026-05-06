"""Tests for pedestrian NPC documentation contracts."""

import ast
from pathlib import Path

PED_NPC_ROOT = Path(__file__).resolve().parents[1] / "robot_sf" / "ped_npc"
TARGET_MODULES = (
    "__init__.py",
    "ped_behavior.py",
    "ped_robot_force.py",
)
PLACEHOLDER_MARKERS = ("TODO docstring", "TODO:")


def _iter_docstrings(tree: ast.Module) -> list[tuple[str, str]]:
    """Collect module, class, and function docstrings from an AST."""
    docstrings: list[tuple[str, str]] = []
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        docstrings.append(("<module>", module_docstring))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append((node.name, docstring))
    return docstrings


def test_targeted_ped_npc_docstrings_do_not_contain_todo_placeholders() -> None:
    """Keep pedestrian NPC force and behavior docs free of TODO placeholders."""
    offenders: list[str] = []
    for module_name in TARGET_MODULES:
        path = PED_NPC_ROOT / module_name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for qualified_name, docstring in _iter_docstrings(tree):
            if any(marker in docstring for marker in PLACEHOLDER_MARKERS):
                first_line = docstring.splitlines()[0]
                offenders.append(f"{module_name}:{qualified_name}: {first_line}")

    assert offenders == []
