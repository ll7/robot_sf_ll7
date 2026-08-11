"""Contract tests for release tag-side and runtime environment capture."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts.repro.capture_release_environment import (
    _lock_resolved_packages,
    _project_constraints,
    _resolved_tag_commit,
    _runtime_record,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_project_constraints_preserve_declared_dependency_surfaces(tmp_path: Path) -> None:
    """The packet must preserve project constraints instead of treating the lock as proof."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[project]
requires-python = ">=3.11"
license = "GPL-3.0-only"
dependencies = ["zeta>=2", "alpha==1"]

[project.optional-dependencies]
viz = ["plot>=1"]

[dependency-groups]
dev = ["pytest>=9"]
""".lstrip(),
        encoding="utf-8",
    )

    assert _project_constraints(path) == {
        "requires_python": ">=3.11",
        "license": "GPL-3.0-only",
        "dependencies": ["alpha==1", "zeta>=2"],
        "optional_dependencies": {"viz": ["plot>=1"]},
        "dependency_groups": {"dev": ["pytest>=9"]},
    }


def test_lock_resolved_packages_are_sorted_and_versioned(tmp_path: Path) -> None:
    """The packet must expose the exact lock resolution used by the tag-side tree."""
    path = tmp_path / "uv.lock"
    path.write_text(
        """
version = 1
revision = 3

[[package]]
name = "zeta"
version = "2.0.0"

[[package]]
name = "alpha"
version = "1.0.0"
""".lstrip(),
        encoding="utf-8",
    )

    assert _lock_resolved_packages(path) == [
        {"name": "alpha", "version": "1.0.0"},
        {"name": "zeta", "version": "2.0.0"},
    ]


def test_lock_resolved_packages_preserve_editable_root_source(tmp_path: Path) -> None:
    """The local project entry is versionless in uv.lock and must remain explicit."""
    path = tmp_path / "uv.lock"
    path.write_text(
        """
version = 1
revision = 3

[[package]]
name = "robot-sf"
source = { editable = "." }
""".lstrip(),
        encoding="utf-8",
    )

    assert _lock_resolved_packages(path) == [
        {"name": "robot-sf", "version": None, "source": {"editable": "."}}
    ]


def test_runtime_record_fails_closed_without_package_inventory(tmp_path: Path) -> None:
    """A Python-only campaign record must not satisfy the historical runtime gate."""
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps({"run": {"python_version": "3.13.1"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="package inventory"):
        _runtime_record(path)


def test_resolved_tag_commit_peels_annotated_tag(tmp_path: Path) -> None:
    """Tag-side packets must bind annotated and lightweight tags to a commit SHA."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.email", "agent@example.invalid")
    git("config", "user.name", "Robot SF test")
    (repo / "payload.txt").write_text("release\n", encoding="utf-8")
    git("add", "payload.txt")
    git("commit", "-qm", "release")
    commit_sha = git("rev-parse", "HEAD")
    git("tag", "-a", "0.0.5", "-m", "0.0.5")

    assert _resolved_tag_commit(repo, "0.0.5") == commit_sha
