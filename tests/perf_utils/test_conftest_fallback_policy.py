"""Regression test for the conftest ``_Fallback`` perf policy docstrings (issue #5854).

PR #5847 introduced three placeholder docstrings on the ``_Fallback`` policy
inside the ``perf_policy`` session fixture in ``tests/conftest.py``
(``is_under_xdist``, ``effective_soft_threshold``, and ``classify``), pushing
the tests/conftest.py placeholder backlog from 15 to 18 and breaking the
docstring-todo ratchet. This test pins the documented soft-threshold envelope
so the fallback policy stays correct and its docstrings stay real, not
placeholders.
"""

from __future__ import annotations

import ast
from pathlib import Path

CONFTEST = Path(__file__).resolve().parents[1] / "conftest.py"

_PLACEHOLDER = "TODO docstring"
# Methods whose docstrings PR #5847 left as placeholders and this issue restores.
_TARGET_METHODS = ("is_under_xdist", "effective_soft_threshold", "classify")


def _fallback_source() -> str:
    return CONFTEST.read_text(encoding="utf-8")


def _fallback_class() -> ast.ClassDef:
    classes = [
        node
        for node in ast.walk(ast.parse(_fallback_source()))
        if isinstance(node, ast.ClassDef) and node.name == "_Fallback"
    ]
    assert len(classes) == 1, "expected exactly one _Fallback class in tests/conftest.py"
    return classes[0]


def _method_block(method: str) -> str:
    methods = [
        node
        for node in _fallback_class().body
        if isinstance(node, ast.FunctionDef) and node.name == method
    ]
    assert len(methods) == 1, f"expected exactly one {method} method on _Fallback"
    source = ast.get_source_segment(_fallback_source(), methods[0])
    assert source is not None
    return source


def test_conftest_fallback_target_methods_have_no_placeholder_docstrings():
    """The regression point: no placeholder docstring remains on the three #5847 methods."""
    for method in _TARGET_METHODS:
        assert _PLACEHOLDER not in _method_block(method), (
            f"{method} still has a placeholder docstring"
        )


def test_fallback_effective_soft_threshold_documents_ci_and_xdist():
    """Documented contract: CI uses full soft threshold; xdist widens below hard."""
    block = _method_block("effective_soft_threshold")
    assert "effective_soft_threshold" in block
    assert "is_under_xdist" in block
    # The fix documents both the CI branch and the xdist contention multiplier.
    assert "xdist_contention_multiplier" in block
    assert "ci" in block
