"""Tests for the fail-closed release publication contract."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from robot_sf.benchmark.release_publication_contract import (
    validate_release_publication_contract,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(
    tmp_path: Path, *, mismatch: bool = False, boundary: bool = False
) -> tuple[Path, Path]:
    campaign_root = tmp_path / "campaign"
    bundle_dir = tmp_path / "publication" / "bundle"
    runtime_commit = "r" * 40
    publication_commit = "p" * 40
    campaign = {
        "campaign_id": "campaign-1",
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
        "release_tag": "v1",
        "doi": "10.5281/zenodo.1234567",
        "doi_url": "https://doi.org/10.5281/zenodo.1234567",
        "release_url": "https://github.com/example/repo/releases/tag/v1",
        "git_hash": runtime_commit,
    }
    release_result = dict(campaign)
    if mismatch:
        release_result["total_episodes"] = 0
    _write_json(campaign_root / "reports" / "campaign_summary.json", {"campaign": campaign})
    _write_json(campaign_root / "release" / "release_result.json", release_result)

    exact = {
        "collision": False,
        "goal_reached": boundary,
        "timeout": boundary,
        "invalid_run": False,
    }
    episode = {
        "episode_id": "episode-1",
        "git_hash": runtime_commit,
        "event_ledger": {"exact_events": exact},
    }
    _write_json(campaign_root / "runs" / "planner__differential_drive" / "episodes.jsonl", episode)
    episode_path = campaign_root / "runs" / "planner__differential_drive" / "episodes.jsonl"
    payload_files = {
        "release/release_manifest.resolved.json": {
            "release_tag": "v1",
            "provenance": {"doi": "10.5281/zenodo.1234567"},
        },
        "release/release_result.json": release_result,
        "reports/campaign_summary.json": {"campaign": campaign},
    }
    for relative, payload in payload_files.items():
        _write_json(bundle_dir / "payload" / relative, payload)
    _write_json(
        bundle_dir / "payload" / "runs" / "planner__differential_drive" / "episodes.jsonl",
        episode,
    )
    provenance = {"repository": {"commit": publication_commit}}
    if mismatch:
        # The mismatch test should reach metadata reconciliation, not fail on
        # the unrelated runtime/publication difference.
        provenance["commit_reconciliation"] = {
            "status": "explained",
            "runtime_commits": [runtime_commit],
            "publication_commit": publication_commit,
            "explanation": "Rows were imported from a separately pinned runtime snapshot.",
        }
    if boundary:
        provenance["goal_timeout_boundary"] = {
            "status": "excluded",
            "note": "Boundary row is excluded until reached-goal timing is available.",
        }
    _write_json(
        bundle_dir / "publication_manifest.json",
        {
            "publication_channels": {
                "release_tag": "v1",
                "doi": "10.5281/zenodo.1234567",
                "release_url": "https://github.com/example/repo/releases/tag/v1",
            },
            "provenance": provenance,
            "files": [],
        },
    )
    entries: list[str] = []
    for relative in payload_files:
        path = bundle_dir / "payload" / relative
        entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  payload/{relative}")
    entries.append(
        f"{hashlib.sha256(episode_path.read_bytes()).hexdigest()}  payload/runs/planner__differential_drive/episodes.jsonl"
    )
    (bundle_dir / "checksums.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")
    publication = json.loads((bundle_dir / "publication_manifest.json").read_text(encoding="utf-8"))
    publication["files"] = [
        {
            "path": entry.split("  ", 1)[1].removeprefix("payload/"),
            "sha256": entry.split("  ", 1)[0],
        }
        for entry in entries
    ]
    _write_json(bundle_dir / "publication_manifest.json", publication)
    return campaign_root, bundle_dir


def test_consistency_mismatch_blocks_publication(tmp_path: Path) -> None:
    """A stale release result cannot accompany a rebuilt successful summary."""
    campaign_root, bundle_dir = _build_fixture(tmp_path, mismatch=True)

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "blocked"
    assert any("total_episodes" in blocker for blocker in report["blockers"])


def test_explained_commit_drift_and_excluded_boundary_can_pass(tmp_path: Path) -> None:
    """Explicit provenance metadata resolves otherwise blocking release differences."""
    campaign_root, bundle_dir = _build_fixture(tmp_path, mismatch=False, boundary=True)
    publication_path = bundle_dir / "publication_manifest.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["provenance"]["commit_reconciliation"] = {
        "status": "explained",
        "runtime_commits": ["r" * 40],
        "publication_commit": "p" * 40,
        "explanation": "Rows were imported from a separately pinned runtime snapshot.",
    }
    _write_json(publication_path, publication)

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "pass"


def test_missing_checksum_manifest_blocks_publication(tmp_path: Path) -> None:
    """A missing checksum manifest is a publication blocker."""
    campaign_root, bundle_dir = _build_fixture(tmp_path)
    (bundle_dir / "checksums.sha256").unlink()

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "blocked"
    assert any("checksums.sha256" in blocker for blocker in report["blockers"])


def test_placeholder_and_manifest_shape_blocks_publication(tmp_path: Path) -> None:
    """Unresolved publication metadata and malformed file lists fail closed."""
    campaign_root, bundle_dir = _build_fixture(tmp_path)
    publication_path = bundle_dir / "publication_manifest.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["publication_channels"]["doi"] = "10.5281/zenodo.<record-id>"
    publication["files"] = {}
    _write_json(publication_path, publication)

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "blocked"
    assert any("placeholder" in blocker for blocker in report["blockers"])
    assert any("files must be a list" in blocker for blocker in report["blockers"])


def test_empty_publication_metadata_blocks(tmp_path: Path) -> None:
    """Empty required release fields cannot masquerade as resolved metadata."""
    campaign_root, bundle_dir = _build_fixture(tmp_path)
    publication_path = bundle_dir / "publication_manifest.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["publication_channels"]["doi"] = "   "
    publication["publication_channels"]["release_url"] = ""
    _write_json(publication_path, publication)

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "blocked"
    assert sum("must be a non-empty string" in blocker for blocker in report["blockers"]) == 2


def test_checksum_and_commit_provenance_mismatches_block(tmp_path: Path) -> None:
    """Changed payloads and missing publication commits cannot pass the gate."""
    campaign_root, bundle_dir = _build_fixture(tmp_path)
    payload_path = bundle_dir / "payload" / "reports" / "campaign_summary.json"
    payload_path.write_text("{}\n", encoding="utf-8")
    publication_path = bundle_dir / "publication_manifest.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["provenance"]["repository"].pop("commit")
    _write_json(publication_path, publication)

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "blocked"
    assert any("checksum mismatch" in blocker for blocker in report["blockers"])
    assert any("missing provenance.repository.commit" in blocker for blocker in report["blockers"])


def test_resolved_goal_timeout_requires_timing_evidence(tmp_path: Path) -> None:
    """A resolved goal+timeout boundary must carry timing evidence."""
    campaign_root, bundle_dir = _build_fixture(tmp_path, mismatch=True, boundary=True)
    publication_path = bundle_dir / "publication_manifest.json"
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    publication["provenance"]["goal_timeout_boundary"]["status"] = "resolved"
    _write_json(publication_path, publication)

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "blocked"
    assert any("timing_evidence" in blocker for blocker in report["blockers"])


def test_malformed_checksums_and_release_tag_mismatch_block(tmp_path: Path) -> None:
    """Malformed checksum syntax and an unexpected requested tag fail closed."""
    campaign_root, bundle_dir = _build_fixture(tmp_path)
    (bundle_dir / "checksums.sha256").write_text("not a checksum\n", encoding="utf-8")

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v2"
    )

    assert report["status"] == "blocked"
    assert any("malformed checksum" in blocker for blocker in report["blockers"])
    assert any("does not match requested tag" in blocker for blocker in report["blockers"])


def test_missing_campaign_object_blocks_publication(tmp_path: Path) -> None:
    """A summary without its campaign object is not a valid publication input."""
    campaign_root, bundle_dir = _build_fixture(tmp_path)
    _write_json(campaign_root / "reports" / "campaign_summary.json", {})

    report = validate_release_publication_contract(
        campaign_root, bundle_dir, expected_release_tag="v1"
    )

    assert report["status"] == "blocked"
    assert any("missing object field 'campaign'" in blocker for blocker in report["blockers"])
