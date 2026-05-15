"""Dependency policy checks for visual schema validation tests."""

from __future__ import annotations

import ast
from pathlib import Path

VISUAL_TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = VISUAL_TEST_DIR.parents[1]
SCHEMA_TEST_PATHS = tuple(sorted(VISUAL_TEST_DIR.glob("test_*_schema_validation.py")))


def test_visual_schema_tests_require_declared_jsonschema_dependency():
    """Visual schema tests should fail clearly if the declared dependency is missing."""
    offenders: list[str] = []

    for path in SCHEMA_TEST_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative_path = path.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if _is_jsonschema_find_spec_call(node):
                offenders.append(
                    f"{relative_path}:{node.lineno}: importlib.util.find_spec('jsonschema')"
                )
            if _is_pytest_skipif_call(node):
                offenders.append(f"{relative_path}:{node.lineno}: pytest.mark.skipif")

    assert not offenders, "Found prohibited dependency guards:\n" + "\n".join(offenders)


def _is_jsonschema_find_spec_call(node: ast.AST) -> bool:
    """Return whether node probes jsonschema availability with importlib."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "find_spec"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "jsonschema"
    )


def _is_pytest_skipif_call(node: ast.AST) -> bool:
    """Return whether node applies a pytest skip-if marker."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skipif"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "mark"
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "pytest"
    )
