"""Test-health policy: forbid `pytest.skip(...)` inside a broad-except handler.

A `try/except Exception: pytest.skip(...)` (or `except BaseException`, or a bare
`except:`) converts *any* failure — including a real regression in the code under
test — into a green skip. The test then asserts nothing and can never fail, which
is the textbook "skip mask" coverage hole (see issue #3382).

This check generalizes the AST-policy approach already used for visual schema
dependency guards in
``tests/visuals/test_schema_validation_dependency_policy.py`` and applies it
across the whole test tree so new offenders are rejected at CI time.

Legitimate conditional skips should use a *narrow* guard instead:
``pytest.importorskip(...)``, a specific exception type
(e.g. ``except ImportError``), or ``pytest.mark.skipif(...)``.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

_BROAD_EXCEPTION_NAMES = frozenset({"Exception", "BaseException"})


def test_no_pytest_skip_inside_broad_except_handler():
    """No test may call ``pytest.skip(...)`` from a broad ``except`` handler."""
    offenders: list[str] = []

    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative_path = path.relative_to(REPO_ROOT)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ExceptHandler) and _is_broad_handler(node)):
                continue
            skip_line = _find_pytest_skip(node.body)
            if skip_line is not None:
                offenders.append(
                    f"{relative_path}:{skip_line}: pytest.skip(...) reached from a broad "
                    "except handler (use pytest.importorskip / a specific exception / "
                    "pytest.mark.skipif instead)"
                )

    assert not offenders, "Found prohibited except -> pytest.skip patterns:\n" + "\n".join(
        offenders
    )


def _is_broad_handler(handler: ast.ExceptHandler) -> bool:
    """Return whether the handler catches everything (bare/Exception/BaseException)."""
    exc_type = handler.type
    if exc_type is None:  # bare `except:`
        return True
    names: list[str] = []
    if isinstance(exc_type, ast.Name):
        names = [exc_type.id]
    elif isinstance(exc_type, ast.Tuple):
        names = [elt.id for elt in exc_type.elts if isinstance(elt, ast.Name)]
    return any(name in _BROAD_EXCEPTION_NAMES for name in names)


def _find_pytest_skip(body: list[ast.stmt]) -> int | None:
    """Return the line of the first ``pytest.skip(...)`` call in ``body``, if any."""
    for stmt in body:
        for node in ast.walk(stmt):
            if _is_pytest_skip_call(node):
                return node.lineno
    return None


def _is_pytest_skip_call(node: ast.AST) -> bool:
    """Return whether node is a ``pytest.skip(...)`` call."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "skip"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
    )
