"""Tests for guided camera-ready release publication helper."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools import publish_camera_ready_release

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, payload: str) -> None:
    """Write UTF-8 test fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_campaign_tree(tmp_path: Path) -> Path:
    """Create minimal campaign output tree with publication bundle metadata."""
    campaign_root = tmp_path / "output" / "benchmarks" / "camera_ready" / "campaign_1"
    reports_dir = campaign_root / "reports"
    publication_dir = tmp_path / "output" / "benchmarks" / "publication" / "bundle"
    _write(tmp_path / "output" / "benchmarks" / "publication" / "bundle.tar.gz", "archive")
    _write(publication_dir / "checksums.sha256", "abc  payload/file.txt\n")
    _write(publication_dir / "publication_manifest.json", "{}")
    _write(
        reports_dir / "campaign_summary.json",
        json.dumps(
            {
                "campaign": {
                    "repository_url": "https://github.com/ll7/robot_sf_ll7",
                    "doi": "10.5281/zenodo.1234567",
                },
                "publication_bundle": {
                    "archive_path": "output/benchmarks/publication/bundle.tar.gz",
                    "checksums_path": "output/benchmarks/publication/bundle/checksums.sha256",
                    "manifest_path": "output/benchmarks/publication/bundle/publication_manifest.json",
                },
            }
        ),
    )
    return campaign_root


def test_publish_camera_ready_release_dry_run_outputs_plan(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Dry-run should validate files and emit deterministic upload plan."""
    campaign_root = _make_campaign_tree(tmp_path)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    exit_code = publish_camera_ready_release.main(
        ["--campaign-root", str(campaign_root), "--tag", "v1.0.0"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["release_url"].endswith("/releases/tag/v1.0.0")
    assert payload["doi_url"] == "https://doi.org/10.5281/zenodo.1234567"
    assert payload["upload_command"][0:3] == ["gh", "release", "upload"]


def test_publish_camera_ready_release_executes_upload(tmp_path: Path, monkeypatch) -> None:
    """Execute mode should call subprocess upload command."""
    campaign_root = _make_campaign_tree(tmp_path)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)
    calls: list[list[str]] = []

    def _fake_run(cmd, check):
        assert check is True
        calls.append(list(cmd))

    monkeypatch.setattr(publish_camera_ready_release.subprocess, "run", _fake_run)

    exit_code = publish_camera_ready_release.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--tag",
            "v1.0.1",
            "--execute-upload",
        ]
    )
    assert exit_code == 0
    assert calls
    assert calls[0][0:4] == ["gh", "release", "upload", "v1.0.1"]


def test_publish_camera_ready_release_rejects_missing_checksums(
    tmp_path: Path, monkeypatch
) -> None:
    """Validation should fail when checksums file is absent."""
    campaign_root = tmp_path / "output" / "benchmarks" / "camera_ready" / "campaign_1"
    _write(
        campaign_root / "reports" / "campaign_summary.json",
        json.dumps(
            {
                "publication_bundle": {
                    "archive_path": "output/benchmarks/publication/bundle.tar.gz",
                    "checksums_path": "output/benchmarks/publication/bundle/checksums.sha256",
                    "manifest_path": "output/benchmarks/publication/bundle/publication_manifest.json",
                }
            }
        ),
    )
    _write(tmp_path / "output" / "benchmarks" / "publication" / "bundle.tar.gz", "archive")
    _write(
        tmp_path / "output" / "benchmarks" / "publication" / "bundle" / "publication_manifest.json",
        "{}",
    )
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="checksums"):
        publish_camera_ready_release.main(
            ["--campaign-root", str(campaign_root), "--tag", "v1.0.2"]
        )
