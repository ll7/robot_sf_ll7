"""Tests for host-specific test guard predicates."""

from __future__ import annotations

import os
import subprocess

import pytest

from tests.support.environment_guards import (
    GIT_TEST_IDENTITY_EMAIL,
    GIT_TEST_IDENTITY_NAME,
    configure_git_identity,
    git_identity_environment,
    is_github_actions,
    is_licca_or_shared_hpc,
    should_enforce_wallclock_budget,
    skip_on_licca_shared_hpc,
)


def test_simulated_licca_env_is_shared_hpc_without_github_actions() -> None:
    """Explicit LiCCA marker should trigger shared-HPC guard behavior."""
    assert is_licca_or_shared_hpc({"ROBOT_SF_TEST_ENV": "licca"})
    assert is_licca_or_shared_hpc({"SLURM_JOB_ID": "12345"})
    assert is_licca_or_shared_hpc({"SLURM_CLUSTER_NAME": "licca"})


def test_github_actions_takes_precedence_over_licca_markers() -> None:
    """GitHub CI must not lose coverage when a LiCCA marker is also present."""
    env = {"GITHUB_ACTIONS": "true", "ROBOT_SF_TEST_ENV": "licca", "SLURM_JOB_ID": "12345"}
    assert is_github_actions(env)
    assert not is_licca_or_shared_hpc(env)
    assert should_enforce_wallclock_budget(env)


def test_unknown_local_env_does_not_skip_or_enforce_perf_budget() -> None:
    """Plain developer shells should keep functional tests running without perf enforcement."""
    assert not is_github_actions({})
    assert not is_licca_or_shared_hpc({})
    assert not should_enforce_wallclock_budget({})


def test_perf_enforce_override_enables_wallclock_budget_on_any_host() -> None:
    """Maintainers can explicitly opt into hard performance assertions off GitHub."""
    assert should_enforce_wallclock_budget({"ROBOT_SF_PERF_ENFORCE": "1"})


def test_skip_helper_uses_explicit_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip helper should require a reason and skip only on shared-HPC markers."""
    with pytest.raises(ValueError, match="skip reason"):
        skip_on_licca_shared_hpc("")

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("ROBOT_SF_TEST_ENV", "licca")
    with pytest.raises(pytest.skip.Exception, match="requires stable wall-clock budget"):
        skip_on_licca_shared_hpc("requires stable wall-clock budget")


def test_git_identity_environment_sets_deterministic_identity() -> None:
    """The identity environment carries deterministic author/committer values."""
    env = git_identity_environment({})
    assert env["GIT_AUTHOR_NAME"] == GIT_TEST_IDENTITY_NAME
    assert env["GIT_AUTHOR_EMAIL"] == GIT_TEST_IDENTITY_EMAIL
    assert env["GIT_COMMITTER_NAME"] == GIT_TEST_IDENTITY_NAME
    assert env["GIT_COMMITTER_EMAIL"] == GIT_TEST_IDENTITY_EMAIL
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"


def test_git_identity_environment_extends_base_env() -> None:
    """Passing a base environment preserves unrelated variables."""
    env = git_identity_environment({"FOO": "bar"})
    assert env["FOO"] == "bar"
    assert env["GIT_AUTHOR_NAME"] == GIT_TEST_IDENTITY_NAME


def test_git_identity_environment_allows_overrides() -> None:
    """Callers can override the default name and email."""
    env = git_identity_environment({}, name="Agent", email="agent@example.invalid")
    assert env["GIT_AUTHOR_NAME"] == "Agent"
    assert env["GIT_AUTHOR_EMAIL"] == "agent@example.invalid"


def test_configure_git_identity_enables_clean_commit(tmp_path) -> None:
    """Configuring repo-local identity lets a commit run without ambient config."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True, text=True)
    configure_git_identity(repo)
    (repo / "file.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    commit = subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=git_identity_environment(),
    )
    assert commit.returncode == 0
    author = subprocess.run(
        ["git", "log", "-1", "--format=%an <%ae>"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    assert author.stdout.strip() == f"{GIT_TEST_IDENTITY_NAME} <{GIT_TEST_IDENTITY_EMAIL}>"


def test_commit_fails_closed_without_local_identity(tmp_path) -> None:
    """A hermetic commit without local identity fails with an actionable error."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True, text=True)
    (repo / "file.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    # Simulate the hermetic lane: global/system config disabled and no ambient
    # author/committer identity and no repo-local config.
    hermetic_env = {
        "PATH": os.environ.get("PATH", ""),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    commit = subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=hermetic_env,
    )
    assert commit.returncode != 0
    assert "Author identity unknown" in commit.stderr


def test_commit_succeeds_when_identity_configured_even_with_ambient_unset(tmp_path) -> None:
    """configure_git_identity + hermetic env commits without ambient vars."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True, text=True)
    configure_git_identity(repo)
    (repo / "file.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    env = git_identity_environment()
    env.pop("GIT_AUTHOR_NAME")
    env.pop("GIT_COMMITTER_NAME")
    commit = subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert commit.returncode == 0, commit.stderr
