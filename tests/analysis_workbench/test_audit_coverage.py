"""BA-04 protocol, denominator, identity, and health-report contracts."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.analysis_workbench import (
    AuditIdentity,
    ReviewRecord,
    evaluate_coverage,
    is_completion_current,
    receipt_matches_identity,
    validate_audit_coverage,
)
from robot_sf.analysis_workbench.audit_coverage import (
    STATUS_COMPLETE_UNDER_PROTOCOL,
    STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS,
    STATUS_INCOMPLETE,
    AuditProtocol,
    default_audit_protocol,
    load_audit_protocol,
)
from robot_sf.analysis_workbench.audit_scan import scan_campaign

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "analysis_workbench" / "audit_coverage_v1"


def _identity(release: str = "release-1") -> AuditIdentity:
    return AuditIdentity(
        campaign_digest="campaign-1",
        source_digest="source-1",
        release_digest=release,
        protocol_version="audit-protocol.v1",
        protocol_digest="protocol-1",
        detector_registry_digest="registry-1",
        detector_config_digest="detector-config-1",
    )


def _review(
    review_id: str, episode_id: str, *, author: str = "human", scope: str = "full_episode"
) -> ReviewRecord:
    return ReviewRecord(
        review_id=review_id,
        episode_id=episode_id,
        scope=scope,
        author_kind=author,
    )


def _complete_inputs() -> tuple[list[dict[str, object]], list[dict[str, str]], list[ReviewRecord]]:
    rows = [
        {
            "episode_id": "episode-success",
            "planner_id": "ppo",
            "scenario_group": "corridor",
            "outcome": "success",
            "ordinary_control": True,
        },
        {
            "episode_id": "episode-collision",
            "planner_id": "ppo",
            "scenario_group": "corridor",
            "outcome": "collision",
        },
    ]
    attempts = [
        {"episode_id": "episode-success", "detector_id": "telemetry", "status": "clear"},
        {"episode_id": "episode-collision", "detector_id": "telemetry", "status": "flagged"},
    ]
    reviews = [
        _review("review-success", "episode-success"),
        _review("review-collision", "episode-collision"),
    ]
    return rows, attempts, reviews


def test_default_protocol_is_versioned_and_non_statistical() -> None:
    protocol = default_audit_protocol()
    assert protocol.version == "audit-protocol.v1"
    assert protocol.full_human_review_target == 1
    assert protocol.digest == default_audit_protocol().digest
    assert load_audit_protocol({"schema_version": "audit-protocol.v1"}).digest == protocol.digest
    assert "confidence" not in json.dumps(protocol.to_dict()).lower()


def test_versioned_coverage_input_fixtures_drive_complete_and_incomplete_reports() -> None:
    complete = json.loads((FIXTURE_ROOT / "complete.json").read_text(encoding="utf-8"))
    complete_report = evaluate_coverage(
        **complete,
        review_records=[
            _review("review-success", "episode-success"),
            _review("review-collision", "episode-collision"),
        ],
        identity=_identity(),
    )
    incomplete = json.loads((FIXTURE_ROOT / "incomplete.json").read_text(encoding="utf-8"))
    incomplete_report = evaluate_coverage(**incomplete, identity=_identity())
    assert complete_report.status == STATUS_COMPLETE_UNDER_PROTOCOL
    assert incomplete_report.status == STATUS_INCOMPLETE
    assert any(item.reason == "detector_attempt_unavailable" for item in incomplete_report.deficits)


def test_complete_report_has_hand_calculated_strata_and_deterministic_digest() -> None:
    rows, attempts, reviews = _complete_inputs()
    first = evaluate_coverage(
        rows, detector_attempts=attempts, review_records=reviews, identity=_identity()
    )
    second = evaluate_coverage(
        rows, detector_attempts=attempts, review_records=reviews, identity=_identity()
    )
    assert first.status == STATUS_COMPLETE_UNDER_PROTOCOL
    assert first.counts["coverage"]["expected"] == 2
    assert first.counts["coverage"]["readable"] == 2
    assert first.counts["detectors"] == {
        "scheduled": 2,
        "evaluable": 2,
        "flagged": 1,
        "clear": 1,
        "unavailable": 0,
        "error": 0,
        "by_detector": {
            "telemetry": {
                "scheduled": 2,
                "evaluable": 2,
                "flagged": 1,
                "clear": 1,
                "unavailable": 0,
                "error": 0,
            }
        },
    }
    assert first.report_digest == second.report_digest
    assert first.to_json() == second.to_json()
    validate_audit_coverage(first.to_dict())


def test_agent_interval_duplicate_and_replay_evidence_cannot_inflate_credit() -> None:
    rows = [
        {"episode_id": "e1", "planner_id": "p", "scenario_group": "g", "outcome": "ok"},
        {"episode_id": "e1", "planner_id": "p", "scenario_group": "g", "outcome": "ok"},
        {"episode_id": "e2", "planner_id": "p", "scenario_group": "g", "outcome": "timeout"},
    ]
    attempts = [
        {"episode_id": "e1", "detector_id": "d", "status": "clear"},
        {"episode_id": "e1", "detector_id": "d", "status": "clear"},
        {"episode_id": "e2", "detector_id": "d", "status": "clear"},
    ]
    reviews = [
        _review("full-e1", "e1"),
        _review("interval-e1", "e1", scope="interval"),
        _review("agent-e2", "e2", author="agent", scope="full_episode"),
        _review("full-e1", "e1"),
    ]
    report = evaluate_coverage(
        rows, detector_attempts=attempts, review_records=reviews, identity=_identity()
    )
    assert report.status == STATUS_INCOMPLETE
    assert report.counts["coverage"]["expected"] == 3
    assert report.counts["coverage"]["duplicate"] == 1
    assert report.counts["coverage"]["readable"] == 2
    assert report.counts["detectors"]["scheduled"] == 2
    assert report.counts["reviews"]["full_episode_human"] == 1
    assert report.counts["reviews"]["duplicate_or_replayed"] == 1
    assert report.counts["reviews"]["agent_reviewed"] == 1
    assert any(item.reason == "missing_full_human_review" for item in report.deficits)
    assert not is_completion_current(report.completion_receipt, _identity())


def test_missing_detector_result_fails_closed_and_is_typed() -> None:
    report = evaluate_coverage(
        [{"episode_id": "e1", "planner_id": "p", "scenario_group": "g", "outcome": "ok"}],
        detector_registry=["telemetry"],
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    assert report.counts["detectors"]["scheduled"] == 1
    assert report.counts["detectors"]["error"] == 1
    assert any(
        item.reason == "detector_attempt_error" and item.status == "error"
        for item in report.deficits
    )


def test_declared_exception_is_visible_but_can_complete_with_exception() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        identity=_identity(),
        materialization=[{"episode_id": "episode-success", "status": "diverged"}],
        exceptions={
            "materialization_mismatch": "release asset is unavailable for this offline audit"
        },
    )
    assert report.status == STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS
    mismatch = [item for item in report.deficits if item.reason == "materialization_mismatch"]
    assert len(mismatch) == 1
    assert mismatch[0].status == "waived"
    assert mismatch[0].to_ba02_dict()["status"] == "met"
    assert set(mismatch[0].to_ba02_dict()) == {
        "stratum_id",
        "observed",
        "target",
        "source_revision",
        "protocol_revision",
        "reason",
        "status",
        "source_id",
        "protocol_id",
    }
    assert report.exceptions[0]["reason"]


def test_high_priority_finding_and_unresolved_integrity_remain_visible() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews[:1],
        identity=_identity(),
        findings=[
            {
                "finding_id": "cluster-1",
                "priority": "high",
                "candidate_members": ["episode-collision"],
                "different_context_episode_id": "episode-success",
            },
            {
                "finding_id": "integrity-1",
                "tags": ["integrity", "critical"],
                "status": "proposed",
                "candidate_members": ["episode-success"],
            },
        ],
    )
    assert report.status == STATUS_INCOMPLETE
    assert report.findings["unresolved_critical_integrity"] == 1
    assert any(item.reason == "missing_finding_representative" for item in report.deficits)
    assert any(item.reason == "unresolved_critical_integrity_finding" for item in report.deficits)


def test_changed_release_invalidates_previous_completion_receipt() -> None:
    rows, attempts, reviews = _complete_inputs()
    original = evaluate_coverage(
        rows, detector_attempts=attempts, review_records=reviews, identity=_identity("release-1")
    )
    changed_identity = _identity("release-2")
    assert not receipt_matches_identity(original.completion_receipt, changed_identity)
    assert not is_completion_current(original.completion_receipt, changed_identity)
    changed = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        identity=changed_identity,
        prior_receipt=original.completion_receipt,
    )
    assert changed.status == STATUS_INCOMPLETE
    assert any(item.reason == "stale_completion_receipt" for item in changed.deficits)


def test_ba01_scan_report_uses_typed_inventory_and_detector_identity() -> None:
    source = FIXTURE_ROOT.parent / "audit_campaign_v1" / "campaign.json"
    scan = scan_campaign(source)
    report = evaluate_coverage(scan, identity={"release_digest": "release-1"})
    assert report.counts["coverage"]["expected"] == 5
    assert report.counts["coverage"]["duplicate"] == 1
    assert report.identity.source_digest == scan.audit.source_digest
    assert report.identity.detector_registry_digest == scan.detector_registry.digest
    assert report.counts["detectors"]["scheduled"] == 70
    assert report.status == STATUS_INCOMPLETE


def test_renderers_and_round_trip_are_deterministic_and_disclose_boundary() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows, detector_attempts=attempts, review_records=reviews, identity=_identity()
    )
    restored = report.from_dict(report.to_dict())
    assert restored.to_json() == report.to_json()
    assert report.from_json(report.to_json()).report_digest == report.report_digest
    assert "not benchmark validity" in report.to_markdown().lower()
    assert "Operational coverage report" in report.to_html()


def test_partial_detector_schedule_and_configured_dimension_aliases_fail_closed() -> None:
    protocol = AuditProtocol(
        scenario_group_fields=("group",),
        ordinary_candidate_fields=("ordinary_candidate",),
    )
    report = evaluate_coverage(
        [
            {
                "episode_id": "e1",
                "planner_id": "planner",
                "group": "corridor",
                "outcome": "success",
                "ordinary_candidate": True,
            },
            {
                "episode_id": "e2",
                "planner_id": "planner",
                "group": "corridor",
                "outcome": "collision",
            },
        ],
        protocol=protocol,
        detector_registry=["telemetry"],
        detector_attempts=[{"episode_id": "e1", "detector_id": "telemetry", "status": "clear"}],
        review_records=[_review("review-e1", "e1"), _review("review-e2", "e2")],
        identity=_identity(),
    )
    assert report.identity.protocol_digest == protocol.digest
    assert report.counts["detectors"]["scheduled"] == 2
    assert report.counts["detectors"]["error"] == 1
    assert report.counts["reviews"]["ordinary_control_candidates"] == 1
    assert report.strata[0]["scenario_group"] == "corridor"
    assert any(item.reason == "detector_attempt_error" for item in report.deficits)


def test_replayed_similarity_review_mapping_cannot_earn_human_credit() -> None:
    rows, attempts, _ = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=[
            {
                "review_id": "similarity-replay",
                "episode_id": "episode-success",
                "author_kind": "human",
                "scope": "full_episode",
                "evidence_kind": "similarity",
            },
            {
                "review_id": "regenerated-replay",
                "episode_id": "episode-collision",
                "author_kind": "human",
                "scope": "full_episode",
                "regenerated": True,
            },
        ],
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    assert report.counts["reviews"]["full_episode_human"] == 0
    assert report.counts["reviews"]["duplicate_or_replayed"] == 2
    assert sum(item.reason == "missing_full_human_review" for item in report.deficits) == 2
    assert any(item.reason == "missing_ordinary_control_review" for item in report.deficits)


def test_explicit_protocol_change_invalidates_completion_receipt() -> None:
    rows, attempts, reviews = _complete_inputs()
    original = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        protocol=default_audit_protocol(),
        identity=_identity(),
    )
    changed_protocol = AuditProtocol(full_human_review_target=2)
    changed = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        protocol=changed_protocol,
        identity=_identity(),
        prior_receipt=original.completion_receipt,
    )
    assert changed.status == STATUS_INCOMPLETE
    assert changed.identity.protocol_digest == changed_protocol.digest
    assert any(item.reason == "stale_completion_receipt" for item in changed.deficits)
