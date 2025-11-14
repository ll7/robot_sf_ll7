"""Tests for legacy artifact migration helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from scripts.tools.migrate_artifacts import migrate_artifacts


def _create_file(path: Path, content: str = "test") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_migrate_artifacts_moves_known_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "output"
    repo_root.mkdir()

    # Directories
    (repo_root / "results").mkdir(parents=True)
    (repo_root / "recordings").mkdir()
    (repo_root / "wandb").mkdir()
    (repo_root / "htmlcov").mkdir()
    (repo_root / "tmp").mkdir()

    # Files
    _create_file(repo_root / "benchmark_results.json", "{}")
    _create_file(repo_root / "coverage.json", "{}")

    report = migrate_artifacts(
        source_root=repo_root,
        artifact_root=artifact_root,
        dry_run=False,
        report_path=artifact_root / "migration-report.json",
    )

    assert not (repo_root / "results").exists()
    assert (artifact_root / "benchmarks/results").is_dir()
    assert (artifact_root / "recordings").is_dir()
    assert (artifact_root / "wandb").is_dir()
    assert (artifact_root / "coverage/htmlcov").is_dir()
    assert (artifact_root / "tmp/legacy").is_dir()
    assert (artifact_root / "benchmarks/benchmark_results.json").is_file()
    assert (artifact_root / "coverage/coverage.json").is_file()

    assert not report.skipped
    assert len(report.relocated) == 7

    report_path = artifact_root / "migration-report.json"
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["artifact_root"].endswith("output")


def test_migrate_artifacts_dry_run_does_not_modify(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    artifact_root = repo_root / "output"
    repo_root.mkdir()
    (repo_root / "results").mkdir(parents=True)

    report = migrate_artifacts(
        source_root=repo_root,
        artifact_root=artifact_root,
        dry_run=True,
    )

    assert (repo_root / "results").exists()
    assert not (artifact_root / "benchmarks/results").exists()
    assert report.relocated
    assert not report.skipped
