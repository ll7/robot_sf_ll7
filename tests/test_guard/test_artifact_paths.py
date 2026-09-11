"""Tests for canonical artifact root helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.common.artifact_paths import (
    ensure_canonical_tree,
    find_legacy_artifact_paths,
    get_artifact_category,
    get_artifact_category_path,
    get_artifact_override_root,
    get_artifact_root,
    get_legacy_migration_plan,
    get_repository_root,
    iter_artifact_categories,
    resolve_artifact_path,
)


def test_get_artifact_root_defaults_to_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the default ``output/`` root when no override is set.

    Args:
        monkeypatch: Fixture used to remove ``ROBOT_SF_ARTIFACT_ROOT``.
    """
    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)
    root = get_artifact_root()
    repo_root = get_repository_root()
    assert root == (repo_root / "output")


def test_get_artifact_root_honors_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Honor ``ROBOT_SF_ARTIFACT_ROOT`` for the override and resolved root.

    Args:
        monkeypatch: Fixture used to set the override environment variable.
        tmp_path: Temporary directory used as the override root.
    """
    override = (tmp_path / "custom").resolve()
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(override))
    assert get_artifact_override_root() == override
    assert get_artifact_root() == override


def test_get_artifact_category_path_uses_active_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolve category paths under the active override root.

    Args:
        monkeypatch: Fixture used to set the override environment variable.
        tmp_path: Temporary directory used as the override root.
    """
    override = (tmp_path / "override").resolve()
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(override))
    ensure_canonical_tree(override)
    coverage_path = get_artifact_category_path("coverage")
    assert coverage_path == (override / "coverage").resolve()


def test_ensure_canonical_tree_creates_category_directories(tmp_path: Path) -> None:
    """Create every default category directory beneath a new root.

    Args:
        tmp_path: Temporary directory used as the artifact root parent.
    """
    root = ensure_canonical_tree(tmp_path / "artifacts")
    expected_relative = {category.relative_path for category in iter_artifact_categories()}
    for relative_path in expected_relative:
        assert (root / relative_path).is_dir()


def test_find_legacy_artifact_paths_detects_known_locations(tmp_path: Path) -> None:
    """Detect existing legacy paths and ignore unrelated directories.

    Args:
        tmp_path: Temporary directory holding the synthetic legacy layout.
    """
    (tmp_path / "results").mkdir()
    (tmp_path / "coverage.json").write_text("{}", encoding="utf-8")
    (tmp_path / "output").mkdir()
    legacy = find_legacy_artifact_paths(tmp_path)
    assert {path.relative_to(tmp_path) for path in legacy} == {
        Path("results"),
        Path("coverage.json"),
    }


def test_get_repository_root_matches_expected_path() -> None:
    """Derive the repository root from the test file location."""
    repo_root = get_repository_root()
    assert repo_root == Path(__file__).resolve().parents[2]


def test_get_legacy_migration_plan_contains_expected_keys() -> None:
    """Return a migration map containing the known legacy entries."""
    plan = get_legacy_migration_plan()
    assert Path("results") in plan
    assert Path("coverage.json") in plan


def test_get_artifact_category_unknown_raises() -> None:
    """Raise ``KeyError`` for an unknown artifact category."""
    with pytest.raises(KeyError):
        get_artifact_category("non-existent")


def test_resolve_artifact_path_uses_canonical_root_when_no_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve category-relative paths under the canonical root.

    Args:
        monkeypatch: Fixture used to remove ``ROBOT_SF_ARTIFACT_ROOT``.
    """
    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)
    expected_base = get_artifact_category_path("benchmarks")
    resolved = resolve_artifact_path("benchmarks/output.json")
    assert resolved == (expected_base / "output.json").resolve()


def test_resolve_artifact_path_preserves_repo_relative_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve non-category paths under the repository root.

    Args:
        monkeypatch: Fixture used to remove ``ROBOT_SF_ARTIFACT_ROOT``.
    """
    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)
    resolved = resolve_artifact_path(Path("docs/example.md"))
    assert resolved == (get_repository_root() / "docs/example.md").resolve()


def test_resolve_artifact_path_migrates_legacy_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map a legacy file name to its canonical destination.

    Args:
        monkeypatch: Fixture used to remove ``ROBOT_SF_ARTIFACT_ROOT``.
    """
    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)
    expected = get_artifact_category_path("coverage") / "coverage.json"
    resolved = resolve_artifact_path("coverage.json")
    assert resolved == expected.resolve()


def test_resolve_artifact_path_honors_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve category-relative paths under the override root.

    Args:
        monkeypatch: Fixture used to set the override environment variable.
        tmp_path: Temporary directory used as the override root.
    """
    override = (tmp_path / "override").resolve()
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(override))
    ensure_canonical_tree(override)
    resolved = resolve_artifact_path("benchmarks/output.json")
    assert resolved == (override / "benchmarks/output.json").resolve()
