"""Tests for Optuna sqlite storage path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training.optuna_expert_ppo import (
    _ensure_sqlite_storage_parent,
    _sanitize_storage_filename,
)


def test_sanitize_storage_filename_strips_path_tokens() -> None:
    """Ensure study-derived sqlite filenames stay path-safe and deterministic."""
    assert _sanitize_storage_filename("../unsafe/name") == "unsafe_name"


def test_ensure_sqlite_storage_parent_allows_output_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure sqlite parents are created when storage stays within the output root."""
    monkeypatch.chdir(tmp_path)
    allowed_root = Path("output").resolve()
    _ensure_sqlite_storage_parent(
        "sqlite:///output/benchmarks/ppo_imitation/hparam_opt/study.db",
        allowed_root=allowed_root,
    )
    assert (tmp_path / "output/benchmarks/ppo_imitation/hparam_opt").exists()


def test_ensure_sqlite_storage_parent_rejects_paths_outside_allowed_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure traversal-like sqlite paths cannot escape the configured output root."""
    monkeypatch.chdir(tmp_path)
    allowed_root = Path("output").resolve()
    with pytest.raises(ValueError, match="outside allowed root"):
        _ensure_sqlite_storage_parent("sqlite:///../escape.db", allowed_root=allowed_root)
