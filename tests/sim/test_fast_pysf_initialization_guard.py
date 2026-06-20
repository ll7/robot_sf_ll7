"""Tests for fast-pysf source checkout and wheel-install guards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf import sim

if TYPE_CHECKING:
    from pathlib import Path


def test_fast_pysf_guard_accepts_initialized_source_checkout(tmp_path: Path) -> None:
    """Source checkouts pass when the vendored fast-pysf package is present."""
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'robot_sf'\n", encoding="utf-8")
    (repo_root / "fast-pysf" / "pysocialforce").mkdir(parents=True)

    sim._assert_fast_pysf_initialized(repo_root)


def test_fast_pysf_guard_rejects_incomplete_source_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source checkouts should still give submodule initialization guidance."""
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname = 'robot_sf'\n", encoding="utf-8")
    monkeypatch.setattr(sim, "_has_installed_pysocialforce", lambda: True)

    with pytest.raises(RuntimeError, match="git submodule update --init --recursive"):
        sim._assert_fast_pysf_initialized(repo_root)


def test_fast_pysf_guard_accepts_wheel_install_with_declared_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed wheels do not carry the source fast-pysf tree."""
    monkeypatch.setattr(sim, "_has_installed_pysocialforce", lambda: True)

    sim._assert_fast_pysf_initialized(tmp_path)
