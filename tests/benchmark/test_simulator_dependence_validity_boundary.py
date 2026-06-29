"""Tests simulator-dependence validity-boundary decision checks for issue #3207."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.simulator_dependence_validity_boundary import (
    DECISION_BLOCKED,
    DECISION_NO_CLAIM,
    DECISION_SUPPORTED,
    SIMULATOR_DEPENDENCE_DECISION_SCHEMA,
    build_simulator_dependence_decision,
    load_json_mapping,
    write_simulator_dependence_decision,
)


def _full_scope_summary() -> dict[str, object]:
    return {
        "status": "actual_campaign_slice",
        "scope": {"classification": "full_fixed_scope"},
        "claim_boundary": (
            "not benchmark evidence; not simulator-realism evidence; "
            "not sim-to-real evidence; not paper-facing evidence"
        ),
        "rank_stability": {
            "rank_identifiable": True,
            "rank_stable": True,
            "rank_identifiability_reason": None,
            "flipping_axes": [],
            "non_identifiable_axes": [],
            "axes": [{"axis": "integration_timestep"}, {"axis": "observation_noise"}],
        },
    }


def test_current_issue_3207_actual_slice_is_no_claim() -> None:
    """Merged actual slice remains diagnostic because rank evidence is non-identifiable."""

    summary_path = Path(
        "docs/context/evidence/issue_3207_fidelity_sensitivity_actual_slice_2026-06-23/summary.json"
    )
    packet = build_simulator_dependence_decision(load_json_mapping(summary_path))

    assert packet["schema_version"] == SIMULATOR_DEPENDENCE_DECISION_SCHEMA
    assert packet["decision"] == DECISION_NO_CLAIM
    assert packet["claim_ready"] is False
    assert "study_scope_not_full_fixed_scope" in packet["no_claim_reasons"]
    assert "rank_non_identifiable:primary_metric_zero_variance" in packet["no_claim_reasons"]


def test_missing_summary_fails_closed() -> None:
    """Missing study input blocks rather than silently approving a claim."""

    packet = build_simulator_dependence_decision(None)

    assert packet["decision"] == DECISION_BLOCKED
    assert packet["claim_ready"] is False
    assert "study_summary" in packet["missing_inputs"]


def test_rank_stable_full_scope_packet_can_be_claim_ready() -> None:
    """Future complete study packet may pass when all boundary checks are satisfied."""

    packet = build_simulator_dependence_decision(
        _full_scope_summary(),
        manifest_check={"passes": True, "axis_count": 2},
        expected_axes=["integration_timestep", "observation_noise"],
    )

    assert packet["decision"] == DECISION_SUPPORTED
    assert packet["claim_ready"] is True
    assert packet["missing_inputs"] == []
    assert packet["no_claim_reasons"] == []
    assert packet["boundary_violations"] == []


def test_missing_axis_prevents_claim_ready() -> None:
    """Expected axis coverage is part of the validity-boundary decision."""

    packet = build_simulator_dependence_decision(
        _full_scope_summary(),
        expected_axes=["integration_timestep", "clearance_radius"],
    )

    assert packet["decision"] == DECISION_NO_CLAIM
    assert "missing_expected_axes:clearance_radius" in packet["no_claim_reasons"]


def test_decision_packet_json_output_is_deterministic(tmp_path: Path) -> None:
    """Decision packets write sorted JSON for reviewable evidence bundles."""

    packet = build_simulator_dependence_decision(_full_scope_summary())
    output_path = write_simulator_dependence_decision(packet, tmp_path / "packet.json")

    assert json.loads(output_path.read_text(encoding="utf-8")) == packet
    assert (
        output_path.read_text(encoding="utf-8")
        == json.dumps(packet, indent=2, sort_keys=True) + "\n"
    )


def test_manifest_check_failure_prevents_claim_ready() -> None:
    """Failed manifest checker input keeps the decision conservative."""

    packet = build_simulator_dependence_decision(
        _full_scope_summary(),
        manifest_check={"passes": False, "axis_count": 2},
    )

    assert packet["decision"] == DECISION_NO_CLAIM
    assert "manifest_check_not_passing" in packet["no_claim_reasons"]


def test_none_claim_boundary_is_normalized_not_stringified() -> None:
    """An explicit None claim_boundary must not coerce to the literal 'None' string."""

    summary = _full_scope_summary()
    summary["claim_boundary"] = None
    packet = build_simulator_dependence_decision(
        summary,
        manifest_check={"passes": True, "axis_count": 2},
        expected_axes=["integration_timestep", "observation_noise"],
    )

    assert packet["decision"] == DECISION_NO_CLAIM
    assert packet["boundary_violations"] == [
        "claim_boundary_missing:not benchmark evidence",
        "claim_boundary_missing:not simulator-realism evidence",
        "claim_boundary_missing:not sim-to-real evidence",
        "claim_boundary_missing:not paper-facing evidence",
    ]


def test_none_axis_name_does_not_satisfy_expected_axis() -> None:
    """A null axis name must not coerce to 'None' and mask a missing expected axis."""

    summary = _full_scope_summary()
    summary["rank_stability"]["axes"] = [{"axis": None}, {"axis": "integration_timestep"}]
    packet = build_simulator_dependence_decision(
        summary,
        expected_axes=["integration_timestep", "None"],
    )

    assert packet["decision"] == DECISION_NO_CLAIM
    assert "missing_expected_axes:None" in packet["no_claim_reasons"]


def test_non_mapping_json_loader_rejects_payload(tmp_path: Path) -> None:
    """Loader raises a clear error when JSON is not an object."""

    path = tmp_path / "list.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        load_json_mapping(path)
