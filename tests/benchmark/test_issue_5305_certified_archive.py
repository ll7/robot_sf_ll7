"""Contract tests for the issue #5305 certified archive registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from robot_sf.adversarial.disjoint_evaluation import archive_sha256, compute_overlap_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "docs/context/evidence/issue_5305_certified_archive"
ARCHIVE = BUNDLE / "archive.json"
REGISTRATION = BUNDLE / "registration.json"
ACCEPTANCE_REPORT = BUNDLE / "acceptance_report.json"
REPOSITORY_READINESS = BUNDLE / "repository_readiness.json"
REFERENCE_ARCHIVES = (
    REPO_ROOT / "docs/context/evidence/issue_1501_adversarial_smoke_2026-05-28/archive.json",
    REPO_ROOT / "docs/context/evidence/issue_1502_adversarial_two_family_2026-05-31/archive.json",
)


def _load(path: Path) -> dict:
    """Load one JSON object from ``path``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_registered_archive_preserves_accepted_source_provenance() -> None:
    """The public projection records both source and registered byte hashes."""
    registration = _load(REGISTRATION)
    assert registration["status"] == "issue5305_archive_accepted"
    assert registration["job_id"] == 13518
    assert registration["entry_count"] == 17
    assert registration["source_archive_sha256"] == (
        "1318e210bc4771fb0ab4b30d5bc6739f9ecd416e039f61e7da4cd7411f3baa6d"
    )
    assert (
        registration["registered_archive_sha256"]
        == hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    )
    assert (
        registration["registration_transformation"]["candidate_or_metric_values_changed"] is False
    )


def test_readiness_reports_bind_the_registered_archive_content() -> None:
    """Standalone and embedded readiness hashes must match the tracked archive projection."""
    expected_hash = archive_sha256(_load(ARCHIVE))
    acceptance_report = _load(ACCEPTANCE_REPORT)
    repository_readiness = _load(REPOSITORY_READINESS)

    assert acceptance_report["repository_readiness"]["archive_sha256"] == expected_hash
    assert repository_readiness["archive_sha256"] == expected_hash


def test_archive_entries_are_certified_and_partition_disjoint() -> None:
    """Every entry is certified and the explicit train/eval split shares no key."""
    archive = _load(ARCHIVE)
    entries = archive["entries"]
    assert len(entries) == 17
    assert all(entry["candidate_certification"]["status"] == "passed" for entry in entries)
    assert all(entry["certification_status"]["status"] == "passed" for entry in entries)
    assert all(entry["failure_attribution"]["status"] == "attributed" for entry in entries)

    train = archive["partitions"]["train"]
    evaluation = archive["partitions"]["eval"]
    assert set(train["scenario_families"]).isdisjoint(evaluation["scenario_families"])
    assert set(train["scenario_seeds"]).isdisjoint(evaluation["scenario_seeds"])
    assert set(train["archive_ids"]).isdisjoint(evaluation["archive_ids"])
    assert archive["partitions"]["overlap_provenance"]["disjointness_checks_passed"] is True


def test_archive_is_family_seed_and_id_disjoint_from_1501_and_1502() -> None:
    """The new input does not reuse leakage keys from either retained archive."""
    new_entries = _load(ARCHIVE)["entries"]
    combined_reference_entries: list[dict] = []
    for reference_path in REFERENCE_ARCHIVES:
        reference_entries = _load(reference_path)["entries"]
        combined_reference_entries.extend(reference_entries)
        overlap = compute_overlap_provenance(reference_entries, new_entries)
        assert overlap["scenario_family_overlap"] == []
        assert overlap["seed_overlap"] == []
        assert overlap["archive_id_overlap"] == []
        assert overlap["disjointness_checks_passed"] is True

    union_overlap = compute_overlap_provenance(combined_reference_entries, new_entries)
    assert union_overlap["scenario_family_overlap"] == []
    assert union_overlap["seed_overlap"] == []
    assert union_overlap["archive_id_overlap"] == []
    assert union_overlap["disjointness_checks_passed"] is True


def test_registered_archive_contains_no_private_absolute_path() -> None:
    """Tracked evidence uses public-safe private artifact URIs, not host paths."""
    text = ARCHIVE.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "worktrees/" not in text
    assert "private-campaign://job-13518/" in text
