"""Tests for guided camera-ready release publication helper."""

from __future__ import annotations

import hashlib
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


def _write_json(path: Path, payload: object) -> None:
    """Write a compact JSON fixture."""
    _write(path, json.dumps(payload))


def _make_campaign_tree(tmp_path: Path, *, tag: str) -> Path:
    """Create minimal campaign output tree with publication bundle metadata."""
    campaign_root = tmp_path / "output" / "benchmarks" / "camera_ready" / "campaign_1"
    reports_dir = campaign_root / "reports"
    publication_dir = tmp_path / "output" / "benchmarks" / "publication" / "bundle"
    _write(tmp_path / "output" / "benchmarks" / "publication" / "bundle.tar.gz", "archive")
    commit = "a" * 40
    campaign = {
        "campaign_id": "campaign_1",
        "status": "benchmark_success",
        "evidence_status": "valid",
        "benchmark_success": True,
        "campaign_execution_status": "completed",
        "total_episodes": 1,
        "total_runs": 1,
        "successful_runs": 1,
        "non_success_runs": 0,
        "accepted_unavailable_runs": 0,
        "unexpected_failed_runs": 0,
        "core_total_runs": 1,
        "core_successful_runs": 1,
        "row_status_summary": {
            "successful_evidence_rows": 1,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        },
        "release_tag": tag,
        "doi": "10.5281/zenodo.1234567",
        "doi_url": "https://doi.org/10.5281/zenodo.1234567",
        "release_url": f"https://github.com/ll7/robot_sf_ll7/releases/tag/{tag}",
        "git_hash": commit,
    }
    _write_json(reports_dir / "campaign_summary.json", {"campaign": campaign})
    _write_json(campaign_root / "release" / "release_result.json", campaign)
    _write_json(
        campaign_root / "runs" / "planner__differential_drive" / "episodes.jsonl",
        {
            "git_hash": commit,
            "event_ledger": {"exact_events": {"goal_reached": False, "timeout": False}},
        },
    )
    payloads = {
        "release/release_manifest.resolved.json": {
            "release_tag": tag,
            "provenance": {"doi": "10.5281/zenodo.1234567"},
        },
        "release/release_result.json": campaign,
        "reports/campaign_summary.json": {"campaign": campaign},
        "runs/planner__differential_drive/episodes.jsonl": {
            "git_hash": commit,
            "event_ledger": {"exact_events": {"goal_reached": False, "timeout": False}},
        },
    }
    for relative, payload in payloads.items():
        _write_json(publication_dir / "payload" / relative, payload)
    checksum_entries = []
    for relative in payloads:
        path = publication_dir / "payload" / relative
        checksum_entries.append(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  payload/{relative}"
        )
    _write(publication_dir / "checksums.sha256", "\n".join(checksum_entries) + "\n")
    _write_json(
        publication_dir / "publication_manifest.json",
        {
            "publication_channels": {
                "release_tag": tag,
                "doi": "10.5281/zenodo.1234567",
                "release_url": f"https://github.com/ll7/robot_sf_ll7/releases/tag/{tag}",
            },
            "provenance": {"repository": {"commit": commit}},
            "files": [
                {
                    "path": relative,
                    "sha256": hashlib.sha256(
                        (publication_dir / "payload" / relative).read_bytes()
                    ).hexdigest(),
                }
                for relative in payloads
            ],
        },
    )
    _write_json(
        reports_dir / "campaign_summary.json",
        {
            "campaign": campaign,
            "publication_bundle": {
                "archive_path": "output/benchmarks/publication/bundle.tar.gz",
                "checksums_path": "output/benchmarks/publication/bundle/checksums.sha256",
                "manifest_path": "output/benchmarks/publication/bundle/publication_manifest.json",
            },
        },
    )
    return campaign_root


def test_publish_camera_ready_release_dry_run_outputs_plan(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Dry-run should validate files and emit deterministic upload plan."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.0")
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    exit_code = publish_camera_ready_release.main(
        ["--campaign-root", str(campaign_root), "--tag", "v1.0.0"]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["release_url"].endswith("/releases/tag/v1.0.0")
    assert payload["release_asset_url"].endswith("/releases/download/v1.0.0/bundle.tar.gz")
    assert payload["doi_url"] == "https://doi.org/10.5281/zenodo.1234567"
    assert payload["upload_command"][0:3] == ["gh", "release", "upload"]


def test_publish_camera_ready_release_accepts_campaign_relative_derived_bundle(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Recovered releases may keep their publication bundle beside the campaign."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.0-derived")
    repository_publication = tmp_path / "output" / "benchmarks" / "publication"
    campaign_publication = campaign_root / "publication"
    repository_publication.rename(campaign_publication)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["publication_bundle"] = {
        "archive_path": "publication/bundle.tar.gz",
        "checksums_path": "publication/bundle/checksums.sha256",
        "manifest_path": "publication/bundle/publication_manifest.json",
    }
    _write_json(summary_path, summary)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    exit_code = publish_camera_ready_release.main(
        ["--campaign-root", str(campaign_root), "--tag", "v1.0.0-derived"]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["archive_path"].startswith(str(campaign_root))
    assert payload["manifest_path"].startswith(str(campaign_root))


def test_publish_camera_ready_release_executes_upload(tmp_path: Path, monkeypatch) -> None:
    """Execute mode should call subprocess upload command."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.1")
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)
    calls: list[list[str]] = []

    def _fake_run(cmd, check):
        """Capture upload commands without invoking subprocesses."""
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


@pytest.mark.parametrize("field", ("archive_path", "checksums_path", "manifest_path"))
def test_publish_camera_ready_release_rejects_absolute_bundle_descriptor_path(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    """Every publication descriptor must remain a repository-relative path."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.2")
    absolute_path = tmp_path / f"absolute-{field}.dat"
    absolute_path.write_text("archive\n", encoding="utf-8")
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["publication_bundle"][field] = str(absolute_path)
    _write_json(summary_path, summary)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="relative"):
        publish_camera_ready_release._validate_prerequisites(
            campaign_root, expected_release_tag="v1.0.2"
        )


def test_publish_camera_ready_release_rejects_parent_traversal_bundle_path(
    tmp_path: Path, monkeypatch
) -> None:
    """Publication descriptors cannot escape the repository with parent traversal."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.3")
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["publication_bundle"]["checksums_path"] = "../outside/checksums.sha256"
    _write_json(summary_path, summary)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="parent traversal"):
        publish_camera_ready_release._validate_prerequisites(
            campaign_root, expected_release_tag="v1.0.3"
        )


@pytest.mark.parametrize("field", ("archive_path", "checksums_path", "manifest_path"))
def test_publish_camera_ready_release_rejects_symlinked_bundle_descriptor_path(
    tmp_path: Path, monkeypatch, field: str
) -> None:
    """Publication descriptors must not follow symlink components."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.4")
    target = tmp_path / f"target-{field}.dat"
    target.write_text("payload\n", encoding="utf-8")
    link = tmp_path / f"link-{field}.dat"
    link.symlink_to(target)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["publication_bundle"][field] = link.name
    _write_json(summary_path, summary)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        publish_camera_ready_release._validate_prerequisites(
            campaign_root, expected_release_tag="v1.0.4"
        )


def test_publish_camera_ready_release_rejects_incoherent_bundle_descriptor_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """Checksums and archive paths must belong to the declared publication bundle."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.5")
    other_bundle = tmp_path / "output" / "benchmarks" / "other_bundle"
    other_bundle.mkdir(parents=True)
    (other_bundle / "checksums.sha256").write_text("placeholder\n", encoding="utf-8")
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["publication_bundle"]["checksums_path"] = (
        "output/benchmarks/other_bundle/checksums.sha256"
    )
    _write_json(summary_path, summary)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="bundle directory"):
        publish_camera_ready_release._validate_prerequisites(
            campaign_root, expected_release_tag="v1.0.5"
        )


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


def test_publish_camera_ready_release_rejects_symlink_campaign_summary(
    tmp_path: Path, monkeypatch
) -> None:
    """Publication validation rejects a summary link before reading it."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.4")
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    outside = tmp_path / "outside-summary.json"
    outside.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    summary_path.unlink()
    summary_path.symlink_to(outside)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        publish_camera_ready_release._validate_prerequisites(
            campaign_root, expected_release_tag="v1.0.4"
        )


def test_publish_camera_ready_release_rejects_symlink_campaign_root(
    tmp_path: Path, monkeypatch
) -> None:
    """CLI validation must inspect the lexical campaign root before resolving it."""
    campaign_root = _make_campaign_tree(tmp_path, tag="v1.0.6")
    linked_campaign_root = tmp_path / "linked-campaign"
    linked_campaign_root.symlink_to(campaign_root, target_is_directory=True)
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        publish_camera_ready_release.main(
            ["--campaign-root", str(linked_campaign_root), "--tag", "v1.0.6"]
        )


def test_publish_camera_ready_release_rejects_empty_publication_path(
    tmp_path: Path, monkeypatch
) -> None:
    """Validation should reject empty publication artifact path fields."""
    campaign_root = tmp_path / "output" / "benchmarks" / "camera_ready" / "campaign_1"
    _write(
        campaign_root / "reports" / "campaign_summary.json",
        json.dumps(
            {
                "publication_bundle": {
                    "archive_path": "",
                    "checksums_path": "output/benchmarks/publication/bundle/checksums.sha256",
                    "manifest_path": "output/benchmarks/publication/bundle/publication_manifest.json",
                }
            }
        ),
    )
    monkeypatch.setattr(publish_camera_ready_release, "get_repository_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="archive_path"):
        publish_camera_ready_release.main(
            ["--campaign-root", str(campaign_root), "--tag", "v1.0.3"]
        )
