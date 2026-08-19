"""Repository-wide dependency policy checks for schema-validation tests."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_CONFIG = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
TEST_ROOTS = tuple(
    REPO_ROOT / relative_path
    for relative_path in PROJECT_CONFIG["tool"]["pytest"]["ini_options"]["testpaths"]
)
JSONSCHEMA_CORE_REQUIREMENT = "jsonschema>=4.23.0"


def test_jsonschema_is_declared_as_a_core_dependency() -> None:
    """The schema-validation dependency must not be made optional accidentally."""
    dependencies = PROJECT_CONFIG["project"]["dependencies"]
    assert JSONSCHEMA_CORE_REQUIREMENT in dependencies


def test_schema_validation_policy_covers_both_configured_test_roots() -> None:
    """Discover schema-policy surfaces from every configured pytest test root."""
    assert REPO_ROOT / "tests" in TEST_ROOTS
    assert REPO_ROOT / "fast-pysf" / "tests" in TEST_ROOTS

    relative_paths = {path.relative_to(REPO_ROOT).as_posix() for path in _schema_test_paths()}
    assert {
        "tests/test_schema_validation_dependency_policy.py",
        "tests/test_metrics.py",
        "tests/test_snqi/test_jsonschema_validation.py",
        "tests/unit/test_schema_validation.py",
    }.issubset(relative_paths)


def test_schema_validation_tests_do_not_mask_missing_jsonschema() -> None:
    """Core dependency failures must remain visible instead of becoming skips."""
    offenders: list[str] = []
    for path in _schema_test_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative_path = path.relative_to(REPO_ROOT)
        offenders.extend(
            f"{relative_path}:{line}: {kind}" for line, kind in _find_jsonschema_guards(tree)
        )

    assert not offenders, "Found prohibited jsonschema dependency guards:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    ("source", "expected_kind"),
    [
        (
            """
import importlib.util
import pytest
if importlib.util.find_spec("jsonschema") is None:
    pytest.skip("missing")
""",
            "importlib.util.find_spec('jsonschema')",
        ),
        (
            'import pytest\npytest.importorskip("jsonschema")',
            "pytest.importorskip('jsonschema')",
        ),
        (
            """
import pytest
jsonschema_available = False
pytest.mark.skipif(not jsonschema_available, reason="missing")
""",
            "pytest.mark.skipif(jsonschema condition)",
        ),
        (
            """
import pytest
try:
    import jsonschema
except ImportError:
    pytest.skip("missing")
""",
            "try/import jsonschema followed by skip",
        ),
    ],
)
def test_jsonschema_guard_detector_finds_prohibited_forms(source: str, expected_kind: str) -> None:
    """Each dependency-specific availability guard has an actionable detector result."""
    guards = _find_jsonschema_guards(ast.parse(source))
    assert expected_kind in {kind for _line, kind in guards}


def test_jsonschema_guard_detector_allows_unrelated_skip_conditions() -> None:
    """Platform and optional-dependency skips remain legal under the focused policy."""
    source = """
import importlib.util
import pytest
import sys

pytest.mark.skipif(sys.platform == "win32", reason="platform-specific")
pytest.mark.skipif(
    importlib.util.find_spec("moviepy") is None,
    reason="optional dependency",
)
pytest.importorskip("moviepy")
"""
    assert _find_jsonschema_guards(ast.parse(source)) == []


def _test_paths() -> tuple[Path, ...]:
    """Return test files from the roots configured for pytest execution."""
    return tuple(
        sorted(
            path
            for tests_root in TEST_ROOTS
            if tests_root.is_dir()
            for path in tests_root.rglob("test_*.py")
        )
    )


def _schema_test_paths() -> tuple[Path, ...]:
    """Return every test file eligible for the core dependency policy."""
    return _test_paths()


def _find_jsonschema_guards(tree: ast.AST) -> list[tuple[int, str]]:
    """Return sorted, de-duplicated line and kind pairs for prohibited guards."""
    guards: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if _is_jsonschema_find_spec_call(node):
            guards.add((node.lineno, "importlib.util.find_spec('jsonschema')"))  # type: ignore[attr-defined]
        if _is_jsonschema_importorskip_call(node):
            guards.add((node.lineno, "pytest.importorskip('jsonschema')"))  # type: ignore[attr-defined]
        if _is_jsonschema_skipif_call(node):
            guards.add((node.lineno, "pytest.mark.skipif(jsonschema condition)"))  # type: ignore[attr-defined]
        if isinstance(node, ast.Try) and _is_jsonschema_import_skip_try(node):
            guards.add((node.lineno, "try/import jsonschema followed by skip"))
    return sorted(guards)


def _is_jsonschema_import(node: ast.AST) -> bool:
    """Return whether node imports jsonschema or one of its submodules."""
    if isinstance(node, ast.Import):
        return any(_is_jsonschema_module(alias.name) for alias in node.names)
    return isinstance(node, ast.ImportFrom) and _is_jsonschema_module(node.module or "")


def _is_jsonschema_find_spec_call(node: ast.AST) -> bool:
    """Return whether node probes jsonschema availability with importlib."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "find_spec" or _dotted_name(node.func.value) != "importlib.util":
        return False
    return _string_argument(node, positional_index=0, keyword_names=("name",)) == "jsonschema"


def _is_jsonschema_importorskip_call(node: ast.AST) -> bool:
    """Return whether node skips when the required jsonschema dependency is absent."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if _dotted_name(node.func.value) != "pytest" or node.func.attr != "importorskip":
        return False
    return _string_argument(node, positional_index=0, keyword_names=("modname",)) == "jsonschema"


def _is_jsonschema_skipif_call(node: ast.AST) -> bool:
    """Return whether a skipif condition references jsonschema availability."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "skipif" or _dotted_name(node.func.value) != "pytest.mark":
        return False
    condition = (
        node.args[0]
        if node.args
        else next((keyword.value for keyword in node.keywords if keyword.arg == "condition"), None)
    )
    return condition is not None and _references_jsonschema(condition)


def _is_jsonschema_import_skip_try(node: ast.Try) -> bool:
    """Return whether a try/import jsonschema path converts import failure into a skip."""
    imports_jsonschema = any(
        _is_jsonschema_import(nested) for statement in node.body for nested in ast.walk(statement)
    )
    return imports_jsonschema and any(
        _is_import_failure_handler(handler) and _contains_skip_call(handler.body)
        for handler in node.handlers
    )


def _is_import_failure_handler(handler: ast.ExceptHandler) -> bool:
    """Return whether handler can catch a missing jsonschema import."""
    if handler.type is None:
        return True
    names = {nested.id for nested in ast.walk(handler.type) if isinstance(nested, ast.Name)}
    return bool(names & {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})


def _contains_skip_call(body: list[ast.stmt]) -> bool:
    """Return whether a handler body contains a skip or xfail mechanism."""
    return any(_is_skip_call(node) for statement in body for node in ast.walk(statement))


def _is_skip_call(node: ast.AST) -> bool:
    """Return whether node raises or calls a test skip mechanism."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id in {"SkipTest", "skip"}
    if not isinstance(node.func, ast.Attribute):
        return False
    dotted_name = _dotted_name(node.func)
    return dotted_name in {
        "pytest.skip",
        "pytest.xfail",
        "pytest.importorskip",
        "unittest.SkipTest",
    }


def _references_jsonschema(node: ast.AST) -> bool:
    """Return whether an expression references the jsonschema availability surface."""
    for nested in ast.walk(node):
        if isinstance(nested, ast.Name) and "jsonschema" in nested.id.lower():
            return True
        if isinstance(nested, ast.Attribute) and "jsonschema" in _dotted_name(nested).lower():
            return True
        if isinstance(nested, ast.Constant) and nested.value == "jsonschema":
            return True
    return False


def _is_jsonschema_module(module_name: str) -> bool:
    """Return whether module_name names jsonschema or one of its submodules."""
    return module_name == "jsonschema" or module_name.startswith("jsonschema.")


def _dotted_name(node: ast.AST) -> str:
    """Return a dotted name for simple Name/Attribute AST nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _string_argument(
    node: ast.Call, positional_index: int, keyword_names: tuple[str, ...]
) -> str | None:
    """Return a literal string argument by position or accepted keyword name."""
    value: ast.AST | None = None
    if len(node.args) > positional_index:
        value = node.args[positional_index]
    else:
        value = next(
            (keyword.value for keyword in node.keywords if keyword.arg in keyword_names),
            None,
        )
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
