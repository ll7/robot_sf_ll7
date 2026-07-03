"""Tests for failure-mechanism taxonomy helpers."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    FailureMechanismTaxonomyError,
    unknown_failure_mechanism_record,
    validate_failure_mechanism_record,
)


def test_unknown_failure_mechanism_record_is_valid() -> None:
    """Unknown records remain explicit and valid when traces are absent."""

    record = unknown_failure_mechanism_record("not_derivable_missing_trace")

    assert validate_failure_mechanism_record(record)["mechanism_label"] == "unknown"
    assert record["mechanism_caveat"] == "not_derivable_missing_trace"


def test_observed_mechanism_requires_trace_verified_evidence_mode() -> None:
    """Observed labels cannot be backed only by aggregate summaries."""

    record = {
        "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
        "mechanism_label": "static_deadlock_or_local_minimum",
        "mechanism_confidence": "observed_mechanism",
        "mechanism_evidence_mode": "aggregate_summary",
        "mechanism_evidence_uri": "docs/context/evidence/example.json",
        "mechanism_case_id": "case-1",
        "mechanism_caveat": "",
    }

    with pytest.raises(FailureMechanismTaxonomyError, match="observed_mechanism requires"):
        validate_failure_mechanism_record(record)


def test_geometry_label_cannot_substitute_for_mechanism_label() -> None:
    """Scenario geometry buckets cannot masquerade as mechanism evidence."""

    record = {
        "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
        "mechanism_label": "static_deadlock_or_local_minimum",
        "mechanism_confidence": "observed_mechanism",
        "mechanism_evidence_mode": "paired_trace",
        "mechanism_evidence_uri": "docs/context/evidence/example.json",
        "mechanism_case_id": "case-1",
        "mechanism_caveat": "",
        "geometry_label": "static_deadlock_or_local_minimum",
    }

    with pytest.raises(FailureMechanismTaxonomyError, match="cannot substitute"):
        validate_failure_mechanism_record(record)
