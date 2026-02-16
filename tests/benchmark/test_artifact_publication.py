"""Tests for benchmark publication bundle helpers."""

from __future__ import annotations

import json
import os
import tarfile
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.artifact_publication import (
    PUBLICATION_BUNDLE_SCHEMA_VERSION,
    SIZE_REPORT_SCHEMA_VERSION,
    discover_run_directories,
    export_publication_bundle,
    list_publication_files,
    measure_artifact_size_ranges,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, payload: str) -> None:
    """Write UTF-8 text payload to a file, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_run(run_dir: Path, *, with_video: bool = True) -> None:
    """Create a minimal synthetic benchmark run directory for publication tests."""
    _write(
        run_dir / "manifest.json", json.dumps({"git_hash": "abc123", "scenario_matrix_hash": "m1"})
    )
    _write(
        run_dir / "run_meta.json",
        json.dumps({"repo": {"remote": "git@github.com:ll7/robot_sf_ll7.git", "commit": "abc123"}}),
    )
    _write(run_dir / "episodes" / "episodes.jsonl", '{"episode_id":"ep-1"}\n')
    _write(run_dir / "aggregates" / "summary.json", '{"success_rate":1.0}\n')
    _write(run_dir / "reports" / "report.md", "# Report\n")
    _write(run_dir / "plots" / "path_efficiency.pdf", "fake-pdf")
    if with_video:
        _write(run_dir / "videos" / "episode_001.mp4", "fake-video")


def test_list_publication_files_respects_video_toggle(tmp_path: Path) -> None:
    """Video toggle should include or exclude video paths from selection."""
    run_dir = tmp_path / "run_a"
    _make_run(run_dir, with_video=True)
    with_videos = list_publication_files(run_dir, include_videos=True)
    without_videos = list_publication_files(run_dir, include_videos=False)

    assert any(path.as_posix().startswith("videos/") for path in with_videos)
    assert not any(path.as_posix().startswith("videos/") for path in without_videos)


def test_discover_run_directories_returns_leaf_runs(tmp_path: Path) -> None:
    """Discovery should return leaf marker directories and avoid parent duplicates."""
    root = tmp_path / "benchmarks"
    _make_run(root / "seed_holdout" / "ppo")
    _make_run(root / "seed_holdout" / "orca")

    runs = discover_run_directories(root)
    run_names = {path.name for path in runs}
    assert run_names == {"ppo", "orca"}


def test_measure_artifact_size_ranges_reports_schema_and_counts(tmp_path: Path) -> None:
    """Size report should expose schema metadata and per-run totals."""
    root = tmp_path / "benchmarks"
    _make_run(root / "run_small", with_video=False)
    _make_run(root / "run_large", with_video=True)

    report = measure_artifact_size_ranges(root, include_videos=False)
    assert report["schema_version"] == SIZE_REPORT_SCHEMA_VERSION
    assert report["run_count"] == 2
    assert report["distributions"]["total_bytes"]["count"] == 2
    assert not str(report["benchmarks_root"]).startswith("/")
    assert all(not str(run["run_dir"]).startswith("/") for run in report["runs"])


def test_export_publication_bundle_writes_manifest_checksums_and_archive(tmp_path: Path) -> None:
    """Export should emit a DOI-ready bundle directory and compressed archive."""
    run_dir = tmp_path / "benchmarks" / "run_export"
    _make_run(run_dir, with_video=False)
    out_dir = tmp_path / "publication"

    result = export_publication_bundle(
        run_dir,
        out_dir,
        bundle_name="run_export_bundle",
        include_videos=False,
    )

    assert result.bundle_dir.exists()
    assert result.archive_path.exists()
    assert result.manifest_path.exists()
    assert result.checksums_path.exists()
    assert result.file_count > 0
    assert result.total_bytes > 0

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == PUBLICATION_BUNDLE_SCHEMA_VERSION
    assert manifest["totals"]["file_count"] == result.file_count
    assert len(manifest["files"]) == result.file_count
    assert all("sha256" in entry for entry in manifest["files"])

    checksum_lines = [
        line
        for line in result.checksums_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(checksum_lines) == result.file_count

    with tarfile.open(result.archive_path, "r:gz") as handle:
        names = handle.getnames()
    assert any(name.endswith("publication_manifest.json") for name in names)
    assert any(name.endswith("checksums.sha256") for name in names)
    assert any("/payload/" in name for name in names)


def test_list_publication_files_skips_symlink_targets(tmp_path: Path) -> None:
    """Symlinked files inside a run must be excluded from publication payloads."""
    run_dir = tmp_path / "run_symlink"
    _make_run(run_dir, with_video=False)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = run_dir / "reports" / "leak_link.txt"
    try:
        os.symlink(secret, link)
    except (NotImplementedError, OSError):  # pragma: no cover
        pytest.skip("Symlink creation unavailable on this platform")

    files = list_publication_files(run_dir, include_videos=False)
    assert not any(path.as_posix() == "reports/leak_link.txt" for path in files)


def test_export_publication_bundle_rejects_unsafe_bundle_names(tmp_path: Path) -> None:
    """Absolute and parent-traversal bundle names should be rejected."""
    run_dir = tmp_path / "benchmarks" / "run_invalid_name"
    _make_run(run_dir, with_video=False)
    out_dir = tmp_path / "publication"

    with pytest.raises(ValueError, match="Invalid bundle_name"):
        export_publication_bundle(run_dir, out_dir, bundle_name="/tmp/evil", overwrite=True)

    with pytest.raises(ValueError, match="Invalid bundle_name"):
        export_publication_bundle(run_dir, out_dir, bundle_name="../escape", overwrite=True)
