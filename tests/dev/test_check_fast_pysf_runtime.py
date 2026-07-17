"""Tests for the check_fast_pysf_runtime.py script."""

from __future__ import annotations

import sys

from scripts.dev.check_fast_pysf_runtime import main


def test_check_fast_pysf_runtime_passes(monkeypatch) -> None:
    """When the GIL-releasing symbol is present in the environment, check should pass."""

    class FakeForces:
        @staticmethod
        def social_force_gil_releasing_context():
            pass

    monkeypatch.setitem(sys.modules, "pysocialforce.forces", FakeForces)
    assert main() == 0


def test_check_fast_pysf_runtime_fails_on_missing_symbol(monkeypatch, capsys) -> None:
    """When the module exists but lacks the symbol, check should fail with instructions."""

    class FakeForcesWithoutSymbol:
        pass

    monkeypatch.setitem(sys.modules, "pysocialforce.forces", FakeForcesWithoutSymbol)
    assert main() == 1
    captured = capsys.readouterr()
    assert "Stale PySocialForce installation detected" in captured.err
    assert "uv sync --all-extras --reinstall-package robot-sf" in captured.err


def test_check_fast_pysf_runtime_fails_on_import_error(monkeypatch, capsys) -> None:
    """When the package is not installed or importable, check should fail with instructions."""
    # Putting None in sys.modules causes Python to raise ImportError when importing.
    monkeypatch.setitem(sys.modules, "pysocialforce.forces", None)
    assert main() == 1
    captured = capsys.readouterr()
    assert "ImportError" in captured.err
    assert "uv sync --all-extras --reinstall-package robot-sf" in captured.err
