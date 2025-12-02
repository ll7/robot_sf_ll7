"""Tests ensuring docstring rule configuration remains authoritative."""

from __future__ import annotations

import tomllib
from pathlib import Path

DOCSTRING_RULES = {
    "D100",
    "D101",
    "D102",
    "D103",
    "D104",
    "D105",
    "D106",
    "D107",
    "D201",
    "D417",
    "D419",
}


def test_docstring_rules_in_pyproject() -> None:
    """Ensure Ruff extend-select contains the required docstring rules."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    lint_cfg = pyproject["tool"]["ruff"]["lint"]
    extend_select = set(lint_cfg.get("extend-select", []))
    missing = DOCSTRING_RULES - extend_select
    assert not missing, f"Docstring rules missing from extend-select: {sorted(missing)}"
