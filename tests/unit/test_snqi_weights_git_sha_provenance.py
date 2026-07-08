"""Regression tests for SNQI weights ``git_sha`` provenance (issue #4895).

``recompute_snqi_weights`` looked up the repo SHA via ``subprocess.run`` with
BOTH ``capture_output=True`` AND ``stderr=DEVNULL`` -- an invalid combination
that always raises ``ValueError``. The handler swallowed it, so ``git_sha`` was
silently degraded to ``"unknown"`` for *every* recorded weight config, even in a
real git checkout. These tests pin the output contract:

* the real short SHA is recorded when git is available inside a repo, and
* the field falls back cleanly to ``"unknown"`` outside any repo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from robot_sf.benchmark.snqi.compute import recompute_snqi_weights

_BASELINE_STATS = {"collisions": {"median": 0.5, "p95": 2.0}}


def _git_available() -> bool:
    """Whether a usable ``git`` executable is on PATH."""
    return subprocess.run(["git", "--version"], capture_output=True, check=False).returncode == 0


def test_git_sha_resolves_in_a_git_checkout(monkeypatch, tmp_path):
    """Inside a real git repo, ``git_sha`` must be the actual short SHA.

    Previously this returned ``"unknown"`` because the subprocess call raised
    ``ValueError`` (capture_output + stderr=DEVNULL) and the handler swallowed it.
    """
    if not _git_available():
        pytest.skip("git executable not available")

    # Build a throwaway repo with at least one commit so rev-parse resolves.
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.example"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repo, check=True, capture_output=True
    )
    (repo / "README").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()[:8]

    # Run the lookup from inside the repo so ``git rev-parse`` resolves to it.
    monkeypatch.chdir(repo)
    weights = recompute_snqi_weights(baseline_stats=_BASELINE_STATS, method="canonical", seed=42)

    assert weights.git_sha == expected
    assert weights.git_sha != "unknown"
    assert len(weights.git_sha) == 8  # short SHA width


def test_git_sha_falls_back_to_unknown_outside_a_repo(monkeypatch, tmp_path):
    """In a directory that is not a git repo, fall back cleanly to ``"unknown"``.

    This must not raise: genuine failure (no repo) yields ``"unknown"`` per the
    output contract.
    """
    if not _git_available():
        pytest.skip("git executable not available")

    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()
    # Sanity check: this tmpdir is genuinely outside any git repo.
    assert (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=non_repo,
            capture_output=True,
            check=False,
        ).returncode
        != 0
    )

    monkeypatch.chdir(non_repo)
    weights = recompute_snqi_weights(baseline_stats=_BASELINE_STATS, method="canonical", seed=42)

    assert weights.git_sha == "unknown"


def test_git_sha_lookup_never_uses_invalid_capture_stderr_combo():
    """Guard against regressing the invalid subprocess argument combination.

    ``capture_output=True`` sets ``stderr=PIPE``; passing ``stderr=DEVNULL`` (or
    any explicit ``stderr``) at the same time raises ``ValueError`` at call time.
    This static check ensures the source no longer combines them.
    """
    source = (
        Path(__file__).resolve().parents[2] / "robot_sf" / "benchmark" / "snqi" / "compute.py"
    ).read_text(encoding="utf-8")

    # Find the git rev-parse lookup block and assert it does not pass stderr=
    # alongside capture_output=True. We scope to the provenance block so an
    # unrelated stderr= elsewhere cannot mask the regression.
    marker = '["git", "rev-parse", "HEAD"]'
    assert marker in source, "git rev-parse provenance lookup moved or removed"
    start = source.index(marker)
    # The call spans from the argv list to the closing paren of run(...).
    end = source.index(")", start)
    block = source[start:end]
    assert "capture_output=True" in block
    assert "stderr=" not in block, (
        "git_sha provenance lookup must not pass stderr= together with "
        "capture_output=True (issue #4895)"
    )
