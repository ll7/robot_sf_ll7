"""Tests for the issue #6969 lane-metric reference diagnostic."""

from __future__ import annotations

import pytest

from robot_sf.research.emergent_phenomena import RELEASED_DEFAULT_CALIBRATION
from robot_sf.research.lane_formation_reference import (
    ReferenceProtocol,
    metric_reference_audit,
    run_native_reference,
    run_reference_campaign,
    summarize_reference_rows,
)


def test_reference_protocol_rejects_invalid_lane_offset_and_population():
    with pytest.raises(ValueError, match="lane_offset_m"):
        ReferenceProtocol(half_width_m=1.0, lane_offset_m=1.0).validate()
    with pytest.raises(ValueError, match="even integer"):
        ReferenceProtocol(n_pedestrians=5).validate()


def test_metric_reference_audit_separates_controls_and_is_sampling_stable():
    audit = metric_reference_audit(sampling_strides=[1, 2, 4], steps=40)

    assert audit["passed"] is True
    assert audit["checks"]["separated_lane_control_clears_reference_floor"] is True
    assert audit["checks"]["mixed_flow_stays_below_clear_threshold"] is True
    assert len(audit["records"]) == 6


def test_native_reference_records_warmup_recycling_and_positive_control():
    protocol = ReferenceProtocol(
        length_m=10.0,
        half_width_m=2.0,
        n_pedestrians=6,
        warmup_steps=5,
        observation_steps=12,
    )
    row = run_native_reference(
        protocol=protocol,
        condition="separated_lane_control",
        seed=7,
        calibration=RELEASED_DEFAULT_CALIBRATION,
        sampling_strides=[1, 2],
    )

    assert row["execution"]["execution_mode"] == "native"
    assert row["execution"]["status"] == "computed"
    assert row["execution"]["warmup_steps_discarded"] == 5
    assert row["execution"]["observation_steps_recorded"] == 12
    assert row["recycled_agents"] >= 0
    assert set(row["sampling_metrics"]) == {"1", "2"}
    assert row["positive_control_is_not_emergence_claim"] is True


def test_reference_campaign_is_native_only_and_summary_keeps_conditions_distinct():
    protocol = ReferenceProtocol(
        length_m=10.0,
        half_width_m=2.0,
        n_pedestrians=6,
        warmup_steps=3,
        observation_steps=8,
    )
    payload = run_reference_campaign(
        protocol=protocol,
        seeds=[7],
        conditions=["mixed_sustained_flow", "separated_lane_control"],
        calibrations=[RELEASED_DEFAULT_CALIBRATION],
        sampling_strides=[1],
    )

    assert payload["schema_version"] == "lane_formation_reference_diagnostic.v1"
    assert payload["manifest"]["claim_boundary"] == (
        "diagnostic_only_not_benchmark_or_paper_evidence"
    )
    assert payload["manifest"]["released_defaults_changed"] is False
    assert payload["manifest"]["metric_semantics_changed"] is False
    assert {row["execution"]["execution_mode"] for row in payload["rows"]} == {"native"}
    assert len(summarize_reference_rows(payload["rows"])) == 2


def test_sampling_stride_validation_fails_closed():
    with pytest.raises(ValueError, match="unique"):
        metric_reference_audit(sampling_strides=[1, 1])
