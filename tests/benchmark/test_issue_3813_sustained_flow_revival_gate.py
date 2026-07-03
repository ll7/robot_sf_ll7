"""Tests for issue #3813 sustained-flow revival gate."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.sustained_flow_revival_gate import (
    DECISION_DEFER,
    DECISION_REVIVE,
    DECISION_STOP,
    DEFAULT_H600_CLAIM_IMPACT_EVIDENCE,
    DEFAULT_H600_INTERACTION_EXPOSURE_EVIDENCE,
    REQUIRED_INTERACTION_EXPOSURE_FIELDS,
    build_sustained_flow_revival_gate_report,
)
from scripts.validation.report_issue_3813_sustained_flow_revival_gate import main as gate_main

REPO_ROOT = Path(__file__).resolve().parents[2]


def _complete_exposure() -> dict:
    return {
        "status": "computed",
        "computed_fields": list(REQUIRED_INTERACTION_EXPOSURE_FIELDS),
        "runs": [
            {
                "job_id": "13268",
                "run_label": "confirm",
                "status": "computed",
                "computed_fields": list(REQUIRED_INTERACTION_EXPOSURE_FIELDS),
            }
        ],
    }


def _claim_impact(*, changed: bool) -> dict:
    return {
        "claim_decisions_changed": changed,
        "affected_rows": [
            {
                "planner_key": "goal",
                "scenario_id": "warehouse_static_obstacles",
                "reason": "low exposure success row would be caveated",
            }
        ],
    }


def test_tracked_h600_evidence_defers_until_exposure_fields_exist() -> None:
    """Current tracked #3810 evidence is incomplete, so the gate fails closed."""

    exposure_path = REPO_ROOT / DEFAULT_H600_INTERACTION_EXPOSURE_EVIDENCE
    exposure = json.loads(exposure_path.read_text(encoding="utf-8"))
    claim_impact_path = REPO_ROOT / DEFAULT_H600_CLAIM_IMPACT_EVIDENCE
    claim_impact = json.loads(claim_impact_path.read_text(encoding="utf-8"))
    report = build_sustained_flow_revival_gate_report(
        exposure,
        interaction_exposure_evidence_path=DEFAULT_H600_INTERACTION_EXPOSURE_EVIDENCE,
        claim_impact=claim_impact,
        claim_impact_evidence_path=DEFAULT_H600_CLAIM_IMPACT_EVIDENCE,
    )
    payload = report.to_payload()
    assert payload["decision"] == DECISION_DEFER
    assert payload["ready_for_revived_implementation"] is False
    assert (
        "interaction-exposure row fields retained but not derivable from h600 episode records"
        in payload["blocking_reasons"]
    )
    assert (
        "claim-decision impact not computable from supplied h600 evidence"
        in payload["blocking_reasons"]
    )
    assert payload["claim_impact_evidence_path"].endswith("sustained_flow_claim_impact_input.json")
    assert payload["affected_rows"]
    assert payload["interaction_exposure_evidence_path"].endswith(
        "interaction_exposure_diagnostics.json"
    )


def test_thin_status_without_declared_fields_fails_closed() -> None:
    """A bare 'computed' status with no declared exposure fields must DEFER.

    Guards against a degraded upstream artifact that only labels itself complete
    (without positively declaring the required interaction-exposure fields)
    silently passing the gate to revive/stop.
    """

    for thin in ({"status": "computed"}, {"runs": [{"status": "computed"}]}):
        report = build_sustained_flow_revival_gate_report(
            thin,
            claim_impact=_claim_impact(changed=True),
            claim_impact_evidence_path="docs/context/evidence/claim_impact.json",
        )
        payload = report.to_payload()
        assert payload["decision"] == DECISION_DEFER
        assert payload["ready_for_revived_implementation"] is False
        assert (
            "interaction-exposure diagnostics missing computed required fields"
            in payload["blocking_reasons"]
        )


def test_empty_claim_impact_still_counts_as_not_supplied() -> None:
    """An empty claim-impact object does not satisfy the gate input contract."""

    report = build_sustained_flow_revival_gate_report(
        _complete_exposure(),
        claim_impact={},
        claim_impact_evidence_path="docs/context/evidence/claim_impact.json",
    )
    payload = report.to_payload()
    assert payload["decision"] == DECISION_DEFER
    assert "claim-decision impact not supplied" in payload["blocking_reasons"]


def test_retained_but_not_derivable_exposure_rows_fail_closed() -> None:
    """Retained row fields do not satisfy the gate unless values are derivable."""

    exposure = {
        "status": "blocked_no_derivable_episode_rows",
        "missing_required_fields": [],
        "available_columns": list(REQUIRED_INTERACTION_EXPOSURE_FIELDS),
        "runs": [
            {
                "job_id": "13268",
                "run_label": "confirm",
                "status": "blocked_no_derivable_episode_rows",
                "missing_required_fields": [],
                "available_columns": list(REQUIRED_INTERACTION_EXPOSURE_FIELDS),
                "derivable_episode_rows": 0,
                "not_derivable_episode_rows": 1008,
            }
        ],
    }

    report = build_sustained_flow_revival_gate_report(
        exposure,
        claim_impact=_claim_impact(changed=True),
    )

    assert report.decision == DECISION_DEFER
    assert report.blocking_reasons == (
        "interaction-exposure row fields retained but not derivable from h600 episode records",
    )


def test_complete_load_bearing_evidence_revives_sustained_flow() -> None:
    """Complete exposure plus changed claim decisions revives the opt-in design lane."""

    report = build_sustained_flow_revival_gate_report(
        _complete_exposure(),
        claim_impact=_claim_impact(changed=True),
        claim_impact_evidence_path="docs/context/evidence/claim_impact.json",
    )
    assert report.decision == DECISION_REVIVE
    assert report.ready_for_revived_implementation is True
    assert report.blocking_reasons == ()
    assert report.affected_rows[0]["planner_key"] == "goal"


def test_complete_non_load_bearing_evidence_stops_sustained_flow() -> None:
    """Complete exposure with unchanged claim decisions stops speculative scenario work."""

    report = build_sustained_flow_revival_gate_report(
        _complete_exposure(),
        claim_impact=_claim_impact(changed=False),
    )
    assert report.decision == DECISION_STOP
    assert report.ready_for_revived_implementation is False
    assert report.blocking_reasons == ()


def test_cli_reports_default_tracked_gate_decision(capsys) -> None:
    """CLI emits the checked-in default gate report without running benchmarks."""

    exit_code = gate_main(["--json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["schema_version"] == "issue_3813.sustained_flow_revival_gate.v1"
    assert payload["decision"] == DECISION_DEFER
    assert payload["claim_impact_evidence_path"].endswith("sustained_flow_claim_impact_input.json")
    assert payload["blocking_reasons"] == [
        "interaction-exposure row fields retained but not derivable from h600 episode records",
        "claim-decision impact not computable from supplied h600 evidence",
    ]
