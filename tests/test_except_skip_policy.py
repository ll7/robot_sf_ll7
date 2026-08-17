"""Test-health policy: forbid skip masks inside broad-except handlers.

A `try/except Exception: pytest.skip(...)` (or `pytest.xfail(...)`,
`unittest.SkipTest`, `except BaseException`, or a bare `except:`) converts *any*
failure — including a real regression in the code under test — into a green skip.
The test then asserts nothing and can never fail, which is the textbook "skip
mask" coverage hole (see issue #3382).

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

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
TEST_ROOTS = (REPO_ROOT / "tests", REPO_ROOT / "fast-pysf" / "tests")

_BROAD_EXCEPTION_NAMES = frozenset({"Exception", "BaseException"})


def test_no_pytest_skip_inside_broad_except_handler():
    """No test tree may hide a broad-except failure behind a skip mechanism."""
    offenders: list[str] = []

    for tests_root in TEST_ROOTS:
        for path in sorted(tests_root.rglob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            relative_path = path.relative_to(REPO_ROOT)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.ExceptHandler) and _is_broad_handler(node)):
                    continue
                skip_mask = _find_skip_mask(node.body)
                if skip_mask is not None:
                    skip_line, skip_kind = skip_mask
                    offenders.append(
                        f"{relative_path}:{skip_line}: {skip_kind} reached from a broad "
                        "except handler (use a narrow exception guard or an explicit "
                        "test-level skip policy instead)"
                    )

    assert not offenders, "Found prohibited except -> pytest.skip patterns:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize(
    ("source", "expected_kind"),
    [
        ("try:\n    pass\nexcept Exception:\n    pytest.xfail('optional')", "pytest.xfail"),
        (
            "try:\n    pass\nexcept Exception:\n    raise unittest.SkipTest('optional')",
            "unittest.SkipTest",
        ),
        (
            "try:\n    pass\nexcept Exception:\n    raise SkipTest('optional')",
            "SkipTest",
        ),
    ],
)
def test_skip_mask_detector_covers_equivalent_mechanisms(source: str, expected_kind: str):
    """The policy must recognize every broad-except skip mechanism it governs."""
    tree = ast.parse(source)
    handler = next(node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler))

    assert _is_broad_handler(handler)
    skip_mask = _find_skip_mask(handler.body)
    assert skip_mask is not None
    assert skip_mask[1] == expected_kind


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


def _find_skip_mask(body: list[ast.stmt]) -> tuple[int, str] | None:
    """Return the first governed skip call and its source-level kind."""
    for stmt in body:
        for node in ast.walk(stmt):
            skip_kind = _skip_mask_kind(node)
            if skip_kind is not None:
                return node.lineno, skip_kind
    return None


def _skip_mask_kind(node: ast.AST) -> str | None:
    """Return the governed skip mechanism represented by an AST call."""
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name) and node.func.id == "SkipTest":
        return "SkipTest"
    if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
        return None
    if node.func.value.id == "pytest" and node.func.attr in {"skip", "xfail"}:
        return f"pytest.{node.func.attr}"
    if node.func.value.id == "unittest" and node.func.attr == "SkipTest":
        return "unittest.SkipTest"
    return None
