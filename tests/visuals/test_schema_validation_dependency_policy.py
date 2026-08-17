"""Dependency policy checks for visual schema validation tests."""

from __future__ import annotations

import ast
from pathlib import Path

VISUAL_TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = VISUAL_TEST_DIR.parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
SCHEMA_TEST_PATHS = tuple(
    sorted(
        path
        for path in TESTS_ROOT.rglob("test_*.py")
        if "schema_validation" in path.name
    )
)


def test_visual_schema_tests_require_declared_jsonschema_dependency():
    """All schema-validation tests should use the declared dependency policy."""
    offenders: list[str] = []

    relative_paths = {path.relative_to(REPO_ROOT).as_posix() for path in SCHEMA_TEST_PATHS}
    assert {
        "tests/visuals/test_schema_validation_dependency_policy.py",
        "tests/test_snqi/test_jsonschema_validation.py",
        "tests/unit/test_schema_validation.py",
    }.issubset(relative_paths)

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
