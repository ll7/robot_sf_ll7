"""Tests for the immutable deterministic diagnosis fixture contract (#7197)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.failure_diagnosis_comparison import FixtureReviewPendingError
from robot_sf.benchmark.failure_diagnosis_fixture import (
    FailureDiagnosisFixtureManifestError,
    FixtureLeakageError,
    build_deterministic_failure_diagnosis_records,
    canonical_source_sha256,
    evaluate_deterministic_failure_diagnosis_fixture,
    load_failure_diagnosis_fixture_manifest,
    validate_failure_diagnosis_fixture_manifest,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE_ROOT = _REPO_ROOT / "tests/benchmark/fixtures/issue_7197_failure_diagnosis"
_MANIFEST_PATH = _FIXTURE_ROOT / "manifest.json"
_SOURCE_PATH = _FIXTURE_ROOT / "source_predicates.json"
_REFERENCE_PATH = _REPO_ROOT / (
    "docs/context/evidence/issue_6646_failure_diagnosis_reference_fixture.v1.json"
)
_CASE_IDS = ("collision_case", "near_miss_case", "low_progress_case", "oscillation_case")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reviewed_reference() -> dict[str, Any]:
    """Use the committed fixture shape without treating its production marker as reviewed."""
    fixture = copy.deepcopy(_load(_REFERENCE_PATH))
    fixture["review_marker"] = "reviewed"
    fixture["records"] = [record for record in fixture["records"] if record["case_id"] in _CASE_IDS]
    return fixture


def test_manifest_and_source_digests_are_versioned_and_admitted() -> None:
    """The test-only source bundle binds every case to an immutable digest and review record."""
    manifest = load_failure_diagnosis_fixture_manifest(_MANIFEST_PATH)
    source = _load(_SOURCE_PATH)

    assert manifest["manifest_version"] == 1
    assert {entry["case_id"] for entry in manifest["fixtures"]} == set(_CASE_IDS)
    for entry in manifest["fixtures"]:
        assert entry["review"]["adjudication_status"] == "adjudicated"
        assert entry["provenance"]["excluded_from_training"] is True
        assert entry["provenance"]["excluded_from_prompt_development"] is True
        assert canonical_source_sha256(source[entry["case_id"]]) == entry["source_trace_sha256"]


def test_runner_uses_existing_adapter_and_reports_metric_integrity() -> None:
    """Admitted source predicates flow through the existing adapter and quality evaluator."""
    manifest = load_failure_diagnosis_fixture_manifest(_MANIFEST_PATH)
    source = _load(_SOURCE_PATH)

    records = build_deterministic_failure_diagnosis_records(manifest, source)
    assert records["collision_case"]["diagnosis_source"] == ("failure_diagnosis.deterministic.v1")
    assert records["oscillation_case"]["failure_type"] == "unknown"

    report = evaluate_deterministic_failure_diagnosis_fixture(
        manifest,
        source,
        _reviewed_reference(),
    )
    assert report["fixture_manifest"]["fixture_count"] == 4
    assert report["detection"]["denominator"] == 4
    assert report["detection"]["agreement"] == 1.0
    assert report["onset"]["denominator"] == 4
    assert report["failure_type"]["denominator"] == 3
    assert report["severity"]["denominator"] == 4
    assert report["claim_boundary"]["no_general_diagnostic_accuracy_claim"] is True


def test_source_digest_drift_fails_before_adapter_execution() -> None:
    """Changing one source value cannot silently produce a new diagnosis row."""
    manifest = load_failure_diagnosis_fixture_manifest(_MANIFEST_PATH)
    source = _load(_SOURCE_PATH)
    source["collision_case"]["evidence_fields"]["event"] = "mutated"

    with pytest.raises(FailureDiagnosisFixtureManifestError, match="source digest mismatch"):
        build_deterministic_failure_diagnosis_records(manifest, source)


def test_reference_label_leakage_fails_closed() -> None:
    """Source predicates cannot carry reference labels or review metadata."""
    manifest = load_failure_diagnosis_fixture_manifest(_MANIFEST_PATH)
    source = _load(_SOURCE_PATH)
    source["collision_case"]["evidence_fields"]["reference_labels"] = {"failure_type": "collision"}

    with pytest.raises(FixtureLeakageError, match="forbidden reference metadata"):
        build_deterministic_failure_diagnosis_records(manifest, source)


def test_pending_manifest_review_fails_closed() -> None:
    """A pending source review cannot be upgraded by a caller-provided label."""
    manifest = _load(_MANIFEST_PATH)
    manifest["fixtures"][0]["review"]["status"] = "pending"

    with pytest.raises(FailureDiagnosisFixtureManifestError, match="must be 'reviewed'"):
        validate_failure_diagnosis_fixture_manifest(manifest)


def test_pending_reference_marker_fails_before_metrics() -> None:
    """The production pending marker remains an unavailable admission state."""
    manifest = load_failure_diagnosis_fixture_manifest(_MANIFEST_PATH)
    source = _load(_SOURCE_PATH)

    with pytest.raises(FixtureReviewPendingError, match="pending review marker"):
        evaluate_deterministic_failure_diagnosis_fixture(
            manifest,
            source,
            _load(_REFERENCE_PATH),
        )


def test_unadjudicated_reference_metadata_fails_before_metrics() -> None:
    """A marker-free but unadjudicated reference is still not admissible."""
    manifest = load_failure_diagnosis_fixture_manifest(_MANIFEST_PATH)
    source = _load(_SOURCE_PATH)
    reference = _reviewed_reference()
    reference["review"]["status"] = "unreviewed"

    with pytest.raises(FailureDiagnosisFixtureManifestError, match="review must be reviewed"):
        evaluate_deterministic_failure_diagnosis_fixture(manifest, source, reference)


def test_source_case_set_must_match_manifest() -> None:
    """Missing or extra source cases are structural blockers, not metric exclusions."""
    manifest = load_failure_diagnosis_fixture_manifest(_MANIFEST_PATH)
    source = _load(_SOURCE_PATH)
    del source["oscillation_case"]

    with pytest.raises(FailureDiagnosisFixtureManifestError, match="case ids do not match"):
        build_deterministic_failure_diagnosis_records(manifest, source)
