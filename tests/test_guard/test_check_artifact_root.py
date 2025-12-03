"""Tests for the artifact root guard script."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.common.artifact_paths import (
    ensure_canonical_tree,
    get_legacy_migration_plan,
)
from scripts.tools.check_artifact_root import GuardResult, check_artifact_root


@pytest.fixture(name="repo_root")
def _repo_root_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        monkeypatch: TODO docstring.

    Returns:
        TODO docstring.
    """
    root = tmp_path / "repo"
    root.mkdir()

    artifact_root = root / "output"
    ensure_canonical_tree(root=artifact_root)

    monkeypatch.delenv("ROBOT_SF_ARTIFACT_ROOT", raising=False)

    return root


def test_check_artifact_root_passes_when_clean(repo_root: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        repo_root: TODO docstring.
    """
    artifact_root = repo_root / "output"
    result = check_artifact_root(source_root=repo_root, artifact_root=artifact_root)

    assert isinstance(result, GuardResult)
    assert result.violations == []
    assert result.exit_code == 0


def test_check_artifact_root_detects_legacy_paths(repo_root: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        repo_root: TODO docstring.
    """
    legacy_path = repo_root / "results"
    legacy_path.mkdir(parents=True)

    artifact_root = repo_root / "output"
    plan = get_legacy_migration_plan()
    expected_destination = artifact_root / plan[Path("results")]

    result = check_artifact_root(source_root=repo_root, artifact_root=artifact_root)

    assert result.exit_code == 1
    assert len(result.violations) == 1
    violation = result.violations[0]
    assert violation.path == str(legacy_path)
    assert str(expected_destination) in violation.remediation


def test_check_artifact_root_respects_allowlist(repo_root: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        repo_root: TODO docstring.
    """
    legacy_file = repo_root / "coverage.json"
    legacy_file.write_text("{}", encoding="utf-8")

    artifact_root = repo_root / "output"

    result = check_artifact_root(
        source_root=repo_root,
        artifact_root=artifact_root,
        allowlist=("coverage.json",),
    )

    assert result.exit_code == 0
    assert result.violations == []
