"""Tests for xdist race artifact scanning."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.dev.check_xdist_race_artifacts import scan_xdist_race_artifacts


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in a temporary repository."""
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _init_repo(repo: Path) -> None:
    """Create a minimal git repository for scanner tests."""
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _git(repo, "config", "user.email", "agent@example.invalid")
    _git(repo, "config", "user.name", "Agent")
    (repo / ".gitignore").write_text("output/\n", encoding="utf-8")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "seed")


def test_scan_writes_baseline_and_accepts_isolated_artifact(tmp_path: Path) -> None:
    """New artifacts under the run-specific xdist directory are allowed."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    artifact_dir = repo / "output" / "tmp" / "xdist-race" / "run-1"
    artifact_dir.mkdir(parents=True)
    baseline = repo / "baseline.json"
    post_run = repo / "post-run.json"

    baseline_result = scan_xdist_race_artifacts(
        paths=[Path("output/tmp"), Path("output/coverage")],
        run_id="run-1",
        output_manifest=baseline,
        cwd=repo,
    )
    (artifact_dir / "pytest.log").write_text("ok\n", encoding="utf-8")
    result = scan_xdist_race_artifacts(
        paths=[Path("output/tmp"), Path("output/coverage")],
        run_id="run-1",
        baseline_path=baseline,
        output_manifest=post_run,
        cwd=repo,
    )

    assert baseline_result["status"] == "passed"
    assert baseline.exists()
    assert post_run.exists()
    assert result["status"] == "passed"
    assert result["new_file_count"] == 1
    assert result["violations"] == []


def test_scan_rejects_truncated_temp_and_cross_worker_artifacts(tmp_path: Path) -> None:
    """Suspicious new shared-output files should fail closed."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    output_tmp = repo / "output" / "tmp"
    output_tmp.mkdir(parents=True)
    baseline = repo / "baseline.json"
    scan_xdist_race_artifacts(
        paths=[Path("output/tmp"), Path("output/coverage")],
        run_id="run-2",
        output_manifest=baseline,
        cwd=repo,
    )
    (output_tmp / "gw0.partial").write_text("", encoding="utf-8")

    result = scan_xdist_race_artifacts(
        paths=[Path("output/tmp"), Path("output/coverage")],
        run_id="run-2",
        baseline_path=baseline,
        cwd=repo,
    )

    codes = {violation["code"] for violation in result["violations"]}
    assert result["status"] == "failed"
    assert {
        "truncated_zero_byte",
        "orphaned_temp_file",
        "cross_worker_artifact",
    } <= codes


def test_scan_allows_non_suspicious_shared_output_file(tmp_path: Path) -> None:
    """Legitimate non-empty files in scanned shared roots should not fail by location alone."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    output_tmp = repo / "output" / "tmp"
    output_tmp.mkdir(parents=True)
    baseline = repo / "baseline.json"
    scan_xdist_race_artifacts(
        paths=[Path("output/tmp"), Path("output/coverage")],
        run_id="run-3",
        output_manifest=baseline,
        cwd=repo,
    )
    (output_tmp / "diagnostic.json").write_text("{}\n", encoding="utf-8")

    result = scan_xdist_race_artifacts(
        paths=[Path("output/tmp"), Path("output/coverage")],
        run_id="run-3",
        baseline_path=baseline,
        cwd=repo,
    )

    assert result["status"] == "passed"
    assert result["new_file_count"] == 1
    assert result["violations"] == []
