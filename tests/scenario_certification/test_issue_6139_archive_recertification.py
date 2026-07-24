"""Tests for issue #6139 transparent re-certification of the tracked #5305 archive.

Issue #6139 repairs the certifier to reject A* paths whose continuous swept envelope
clips obstacle corners. The 17 records registered by PR #5905 were certified under the
old occupancy-only semantics, so this test re-derives their eligibility under the
corrected certifier (including runtime collision verdicts) and asserts the committed re-certification evidence is
reproducible, faithful, and never silently rewrites the accepted archive.

These tests do not make a benchmark, transfer, or minimax claim.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.scenario_certification.recertification import (
    RECERT_SCHEMA_VERSION,
    recertification_report_to_dict,
    recertify_tracked_archive,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_DIR = _REPO_ROOT / "docs/context/evidence/issue_5305_certified_archive"
_ARCHIVE_PATH = _EVIDENCE_DIR / "archive.json"
_REGISTRATION_PATH = _EVIDENCE_DIR / "registration.json"
_RECERT_PATH = _EVIDENCE_DIR / "recertification_issue_6139.json"


@pytest.fixture(scope="module")
def committed_recert() -> dict:
    """Load the committed re-certification evidence JSON."""
    return json.loads(_RECERT_PATH.read_text(encoding="utf-8"))


def test_recertification_schema_and_provenance(committed_recert: dict) -> None:
    """The committed evidence carries the versioned schema and provenance."""
    assert committed_recert["schema_version"] == RECERT_SCHEMA_VERSION
    assert committed_recert["issue"] == "6139"
    assert committed_recert["claim_boundary"] == "corrected_recertification_not_benchmark_evidence"
    assert committed_recert["correction"]["accepted_archive_modified"] is False


def test_recertification_does_not_modify_accepted_archive(committed_recert: dict) -> None:
    """The accepted archive bytes are unchanged and match the registered projection."""
    archive_sha256 = committed_recert["archive_sha256"]
    assert archive_sha256 == _sha256(_ARCHIVE_PATH.read_bytes())
    registration = json.loads(_REGISTRATION_PATH.read_text(encoding="utf-8"))
    # The registered (public-safe) archive hash must match the unchanged accepted archive.
    assert registration["registered_archive_sha256"] == archive_sha256


def test_recertification_covers_all_17_records(committed_recert: dict) -> None:
    """Every tracked record is re-certified with before/after status."""
    records = committed_recert["records"]
    assert len(records) == 17
    assert committed_recert["counts"]["record_count"] == 17
    for record in records:
        assert record["status"] in {"unchanged", "changed"}
        assert "before" in record and "after" in record
        swept = record["after"]["swept_envelope"]
        assert "validated" in swept
        assert "clips_obstacle" in swept
        assert "clearance_m" in swept
        simulator = record["after"]["simulator_obstacle_collision"]
        assert simulator["validated"] is True
        assert simulator["collides_obstacle"] is False
        assert simulator["runtime_component"] == "ContinuousOccupancy.is_obstacle_collision"


def test_recertification_reconstruction_is_bit_for_bit_faithful(committed_recert: dict) -> None:
    """Reconstruction reproduces the embedded certification geometry exactly."""
    fidelity = committed_recert["reconstruction_fidelity"]
    assert fidelity["fidelity_mismatch_count"] == 0
    assert fidelity["max_shortest_path_length_abs_error_m"] == 0.0
    assert fidelity["max_minimum_static_clearance_abs_error_m"] == 0.0


def test_recertification_is_reproducible() -> None:
    """Re-running the corrected certifier reproduces the committed payload and hash."""
    report = recertify_tracked_archive(_ARCHIVE_PATH)
    payload = recertification_report_to_dict(report, archive_path=_ARCHIVE_PATH)
    committed = json.loads(_RECERT_PATH.read_text(encoding="utf-8"))
    assert payload["recertification_sha256"] == committed["recertification_sha256"]
    assert payload["counts"] == committed["counts"]


def test_recertification_records_acceptance_counts(committed_recert: dict) -> None:
    """The output states how many records remain eligible, stress_only, or excluded."""
    after = committed_recert["counts"]["after_benchmark_eligibility"]
    # The corrected certifier must not silently promote or exclude records; the
    # pre-correction accepted inputs keep their eligibility unless a planned path now
    # clips a corner, in which case the record is excluded with a recorded reason.
    assert sum(after.values()) == 17
    assert after.get("excluded", 0) == sum(
        1
        for record in committed_recert["records"]
        if record["after"]["swept_envelope"]["clips_obstacle"] is True
    )


def test_every_accepted_path_has_finite_nonnegative_clearance(committed_recert: dict) -> None:
    """Every non-excluded accepted path keeps finite non-negative swept-envelope clearance."""
    for record in committed_recert["records"]:
        swept = record["after"]["swept_envelope"]
        eligibility = record["after"]["benchmark_eligibility"]
        if eligibility == "excluded":
            continue
        assert swept["validated"] is True
        assert swept["clips_obstacle"] is False
        assert swept["clearance_m"] is not None
        assert swept["clearance_m"] >= 0.0


def _sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    import hashlib

    return hashlib.sha256(data).hexdigest()
