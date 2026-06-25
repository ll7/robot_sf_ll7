"""Guard that the Ruff security (Bandit `S`) suppression stays explicit (issue #3477)."""

from __future__ import annotations

import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Codes that are explicitly baselined (suppressed) pending gradual ratchet-down.
_EXPECTED_BASELINE_CODES = {
    "S101",
    "S105",
    "S108",
    "S110",
    "S112",
    "S202",
    "S301",
    "S310",
    "S311",
    "S314",
    "S324",
    "S602",
    "S603",
    "S607",
    "S608",
}


def _lint_ignore() -> list[str]:
    """Return the `[tool.ruff.lint] ignore` list from pyproject.toml."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return list(data["tool"]["ruff"]["lint"]["ignore"])


def test_blanket_S_category_is_not_suppressed() -> None:
    """The whole `S` category must not be blanket-ignored (only specific codes)."""
    assert "S" not in _lint_ignore(), (
        "Ruff must not blanket-ignore the S category; list specific S codes instead (issue #3477)"
    )


def test_explicit_S_baseline_codes_are_present() -> None:
    """The explicit S-baseline codes must remain listed until ratcheted down."""
    ignore = set(_lint_ignore())
    assert _EXPECTED_BASELINE_CODES <= ignore
