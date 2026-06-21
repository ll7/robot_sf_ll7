"""Tests for the external repository staging assistant."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts.tools import manage_external_repos

if TYPE_CHECKING:
    from pathlib import Path


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a git command for local staging fixtures."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _init_git_repo(path: Path, *, gitignore: str = "") -> None:
    """Create a small git repo for git-ignore staging checks."""
    _run_git(["init"], cwd=path)
    _run_git(["config", "user.email", "tests@example.invalid"], cwd=path)
    _run_git(["config", "user.name", "Robot SF Tests"], cwd=path)
    if gitignore:
        (path / ".gitignore").write_text(gitignore, encoding="utf-8")
        _run_git(["add", ".gitignore"], cwd=path)
        _run_git(["commit", "-m", "add ignore rules"], cwd=path)


def _init_upstream_repo(path: Path) -> str:
    """Create an upstream fixture repository and return its pinned commit."""
    _init_git_repo(path)
    (path / "README.md").write_text("# fixture repo\n", encoding="utf-8")
    (path / "planner.py").write_text("print('planner fixture')\n", encoding="utf-8")
    _run_git(["add", "README.md", "planner.py"], cwd=path)
    _run_git(["commit", "-m", "initial fixture"], cwd=path)
    return _run_git(["rev-parse", "HEAD"], cwd=path).stdout.strip()


def _repo_spec(
    *,
    upstream: Path,
    pinned_sha: str,
    stage_path: Path,
    validation_command: str = "python planner.py",
) -> manage_external_repos.RepoSpec:
    """Build a local fixture RepoSpec."""
    return manage_external_repos.RepoSpec(
        name="fixture-planner",
        title="Fixture planner",
        upstream_url=str(upstream),
        fork_url=None,
        pinned_sha=pinned_sha,
        stage_path=stage_path,
        source_access_date="2026-06-21",
        license_note="MIT fixture license.",
        license_compatibility_decision="compatible for local test staging",
        redistribution_decision="do not redistribute fixture clone",
        intended_use="exercise external repository staging",
        validation_command=validation_command,
        related_issues=(3347,),
    )


def test_registry_covers_first_external_repo_entry() -> None:
    """The first production slice should include at least one real stageable repo."""
    repo_names = {repo.name for repo in manage_external_repos.list_repos()}

    assert "sicnav" in repo_names


def test_check_all_reports_missing_unstaged_registered_repos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registered repos should be visible even before their local clones are staged."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    pinned_sha = _init_upstream_repo(upstream)
    spec = _repo_spec(
        upstream=upstream,
        pinned_sha=pinned_sha,
        stage_path=tmp_path / "third_party" / "external_repos" / "fixture-planner",
    )
    monkeypatch.setattr(manage_external_repos, "REPOS", (spec,))

    reports = manage_external_repos.check_all(repo_root=tmp_path)

    assert reports
    assert {report["name"] for report in reports} == {"fixture-planner"}
    assert all(report["status"] == "missing" for report in reports)


def test_check_reports_missing_unstaged_repo(tmp_path: Path) -> None:
    """Registered repos should fail closed until the pinned clone is staged."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    pinned_sha = _init_upstream_repo(upstream)
    spec = _repo_spec(
        upstream=upstream,
        pinned_sha=pinned_sha,
        stage_path=tmp_path / "third_party" / "external_repos" / "fixture-planner",
    )

    report = manage_external_repos.check_repo(spec=spec, repo_root=tmp_path)

    assert report["ok"] is False
    assert report["status"] == "missing"
    assert report["pinned_sha"] == pinned_sha
    assert "stage fixture-planner" in report["action"]


def test_stage_rejects_destination_not_covered_by_gitignore(tmp_path: Path) -> None:
    """Repo-local external clones must be gitignored before staging."""
    _init_git_repo(tmp_path)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    pinned_sha = _init_upstream_repo(upstream)
    spec = _repo_spec(
        upstream=upstream,
        pinned_sha=pinned_sha,
        stage_path=tmp_path / "third_party" / "external_repos" / "fixture-planner",
    )

    with pytest.raises(manage_external_repos.ExternalRepoError, match="not covered by gitignore"):
        manage_external_repos.stage_repo(
            spec=spec,
            manifest_out=tmp_path / "manifest.json",
            repo_root=tmp_path,
        )


def test_stage_clones_pinned_sha_and_writes_manifest(tmp_path: Path) -> None:
    """Staging should clone, pin, checksum, and write compact provenance."""
    _init_git_repo(tmp_path, gitignore="third_party/external_repos/\n")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    pinned_sha = _init_upstream_repo(upstream)
    stage_path = tmp_path / "third_party" / "external_repos" / "fixture-planner"
    manifest_path = tmp_path / "manifests" / "fixture-planner.provenance.json"
    spec = _repo_spec(upstream=upstream, pinned_sha=pinned_sha, stage_path=stage_path)

    manifest = manage_external_repos.stage_repo(
        spec=spec,
        manifest_out=manifest_path,
        repo_root=tmp_path,
    )

    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == manifest
    assert payload["schema"] == "robot_sf_external_repo_manifest.v1"
    assert payload["name"] == "fixture-planner"
    assert payload["upstream_url"] == str(upstream)
    assert payload["fork_url"] is None
    assert payload["pinned_sha"] == pinned_sha
    assert payload["staged_commit"] == pinned_sha
    assert payload["source_access_date"] == "2026-06-21"
    assert payload["license_compatibility_decision"] == "compatible for local test staging"
    assert payload["redistribution_decision"] == "do not redistribute fixture clone"
    assert payload["intended_use"] == "exercise external repository staging"
    assert payload["validation_command"] == "python planner.py"
    assert payload["tree_sha256"]
    assert payload["file_count"] >= 2
    assert payload["sample_files"][0]["path"] in {"README.md", "planner.py"}
    assert (stage_path / ".git").exists()
    assert _run_git(["rev-parse", "HEAD"], cwd=stage_path).stdout.strip() == pinned_sha


def test_stage_rejects_unreachable_pinned_sha(tmp_path: Path) -> None:
    """Staging must fail before writing a manifest when the pinned SHA cannot be fetched."""
    _init_git_repo(tmp_path, gitignore="third_party/external_repos/\n")
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _init_upstream_repo(upstream)
    unreachable_sha = "f" * 40
    spec = _repo_spec(
        upstream=upstream,
        pinned_sha=unreachable_sha,
        stage_path=tmp_path / "third_party" / "external_repos" / "fixture-planner",
    )
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(manage_external_repos.ExternalRepoError, match="Pinned SHA is unreachable"):
        manage_external_repos.stage_repo(
            spec=spec,
            manifest_out=manifest_path,
            repo_root=tmp_path,
        )

    assert not manifest_path.exists()
    assert not spec.stage_path.exists()
