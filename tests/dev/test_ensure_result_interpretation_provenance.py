"""Tests for CI hydration of result-interpretation fixture provenance."""

from __future__ import annotations

import subprocess

import pytest

from scripts.dev import ensure_result_interpretation_provenance as hydration


def test_required_commits_match_source_and_catalog_provenance() -> None:
    """Only validator-required source and catalog commits are hydrated."""
    commits = set(hydration.collect_required_commits())

    assert "2fc4498cc5499bd3569eb1ac941a3029e0f51040" in commits
    assert "8f9438632e794f084db72bb016a14b539bbca648" in commits
    assert "4e513ebbbc3b11ef580cea76888fd6de43836c66" in commits
    assert "54ed835669192dd22974ff4a68acbc83ddfe5148" not in commits
    assert all(len(commit) == 40 for commit in commits)


@pytest.mark.parametrize("use_token", [False, True])
def test_fetch_uses_ephemeral_gh_credential_helper_only_with_token(
    monkeypatch: pytest.MonkeyPatch, use_token: bool
) -> None:
    """A CI token enables GitHub auth without entering the Git command line."""
    token = "fixture-token-not-a-secret"
    if use_token:
        monkeypatch.setenv("GH_TOKEN", token)
    else:
        monkeypatch.delenv("GH_TOKEN", raising=False)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(hydration.subprocess, "run", fake_run)
    commit = "a" * 40
    assert hydration._fetch_commits([commit]).returncode == 0
    command = commands[0]
    assert command[:3] == ["git", "-C", str(hydration.ROOT)]
    assert command[-4:] == ["fetch", "--no-tags", "origin", commit]
    assert ("credential.helper=!gh auth git-credential" in command) is use_token
    assert token not in " ".join(command)
