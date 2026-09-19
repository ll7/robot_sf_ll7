"""BA-04 protocol, denominator, identity, and health-report contracts."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.analysis_workbench import (
    AuditCoverageError,
    AuditIdentity,
    DetectorRegistry,
    DetectorSpec,
    ReviewRecord,
    evaluate_coverage,
    is_completion_current,
    receipt_matches_identity,
    record_to_dict,
    validate_audit_coverage,
)
from robot_sf.analysis_workbench.audit_coverage import (
    STATUS_COMPLETE_UNDER_PROTOCOL,
    STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS,
    STATUS_INCOMPLETE,
    AuditProtocol,
    _digest,
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


def _registry(detector_id: str = "telemetry") -> DetectorRegistry:
    return DetectorRegistry(detectors=(DetectorSpec(detector_id, "test", "typed test detector"),))


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


def _redigest(payload: dict[str, object]) -> dict[str, object]:
    """Recompute only the outer report digest after a deliberate mutation."""

    payload["report_digest"] = _digest(
        {key: value for key, value in payload.items() if key != "report_digest"}
    )
    return payload


def _redigest_identity(payload: dict[str, object]) -> dict[str, object]:
    """Recompute the nested identity and outer report digests after a mutation."""

    identity = AuditIdentity.from_dict(payload["identity"])
    payload["identity"]["identity_digest"] = identity.digest
    return _redigest(payload)


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
        detector_registry=_registry(),
        identity=_identity(),
    )
    incomplete = json.loads((FIXTURE_ROOT / "incomplete.json").read_text(encoding="utf-8"))
    incomplete_report = evaluate_coverage(
        **incomplete, detector_registry=_registry(), identity=_identity()
    )
    assert complete_report.status == STATUS_COMPLETE_UNDER_PROTOCOL
    assert incomplete_report.status == STATUS_INCOMPLETE
    assert any(item.reason == "detector_attempt_unavailable" for item in incomplete_report.deficits)


def test_complete_report_has_hand_calculated_strata_and_deterministic_digest() -> None:
    rows, attempts, reviews = _complete_inputs()
    first = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        detector_registry=_registry(),
        identity=_identity(),
    )
    second = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        detector_registry=_registry(),
        identity=_identity(),
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
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        detector_registry=_registry("d"),
        identity=_identity(),
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
        detector_registry=_registry(),
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
        detector_registry=_registry(),
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


@pytest.mark.parametrize(
    ("materialization", "reason", "episode_id", "status"),
    (
        (
            [{"episode_id": "foreign", "status": "verified"}],
            "materialization_foreign_episode",
            "foreign",
            "unavailable",
        ),
        ([{"status": "verified"}], "materialization_unbound", "", "unavailable"),
    ),
)
def test_materialization_must_bind_to_readable_episode_ids(
    materialization, reason, episode_id, status
) -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        materialization=materialization,
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    assert report.evidence["materialization_rows"] == (
        {"episode_id": episode_id, "status": status},
    )
    assert any(item.reason == reason for item in report.deficits)
    validate_audit_coverage(report.to_dict())
    assert report.from_dict(report.to_dict()).status == STATUS_INCOMPLETE


def test_source_scan_materialization_also_binds_to_readable_episode_ids() -> None:
    rows, attempts, reviews = _complete_inputs()
    source_scan = {
        "inventory": rows,
        "signals": attempts,
        "detector_registry": _registry(),
        "materialization": [{"episode_id": "foreign", "status": "verified"}],
    }
    report = evaluate_coverage(source_scan, review_records=reviews, identity=_identity())
    assert report.status == STATUS_INCOMPLETE
    assert any(item.reason == "materialization_foreign_episode" for item in report.deficits)
    validate_audit_coverage(report.to_dict())
    assert report.from_dict(report.to_dict()).status == STATUS_INCOMPLETE


def test_complete_report_rejects_forged_foreign_materialization_in_both_validators() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        materialization=[{"episode_id": "episode-success", "status": "verified"}],
        identity=_identity(),
    )
    assert report.status == STATUS_COMPLETE_UNDER_PROTOCOL
    payload = report.to_dict()
    payload["evidence"]["materialization_rows"][0]["episode_id"] = "foreign"
    candidate = _redigest(payload)
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(candidate)
    with pytest.raises(AuditCoverageError):
        report.from_dict(candidate)


def test_materialization_permutations_have_canonical_report_identity_and_renderers() -> None:
    rows, attempts, reviews = _complete_inputs()
    first = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        materialization=[
            {"episode_id": "episode-success", "status": "verified"},
            {"episode_id": "episode-collision", "status": "diverged"},
        ],
        identity=_identity(),
    )
    second = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        materialization=[
            {"episode_id": "episode-collision", "status": "diverged"},
            {"episode_id": "episode-success", "status": "verified"},
        ],
        identity=_identity(),
    )
    assert first.status == STATUS_INCOMPLETE
    assert second.status == STATUS_INCOMPLETE
    assert first.report_digest == second.report_digest
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert first.to_html() == second.to_html()
    assert list(first.evidence["materialization_rows"]) == [
        {"episode_id": "episode-collision", "status": "diverged"},
        {"episode_id": "episode-success", "status": "verified"},
    ]


def test_high_priority_finding_and_unresolved_integrity_remain_visible() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews[:1],
        detector_registry=_registry(),
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
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        detector_registry=_registry(),
        identity=_identity("release-1"),
    )
    changed_identity = _identity("release-2")
    assert not receipt_matches_identity(original.completion_receipt, changed_identity)
    assert not is_completion_current(original.completion_receipt, changed_identity)
    changed = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        detector_registry=_registry(),
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
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        detector_registry=_registry(),
        identity=_identity(),
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
        detector_registry=_registry(),
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
        detector_registry=_registry(),
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
        detector_registry=_registry(),
        protocol=default_audit_protocol(),
        identity=_identity(),
    )
    changed_protocol = AuditProtocol(full_human_review_target=2)
    changed = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        review_records=reviews,
        detector_registry=_registry(),
        protocol=changed_protocol,
        identity=_identity(),
        prior_receipt=original.completion_receipt,
    )
    assert changed.status == STATUS_INCOMPLETE
    assert changed.identity.protocol_digest == changed_protocol.digest
    assert any(item.reason == "stale_completion_receipt" for item in changed.deficits)


def test_only_typed_ba03_review_receipts_can_earn_human_credit() -> None:
    rows, attempts, reviews = _complete_inputs()
    forged = {
        "review_id": "forged",
        "episode_id": "episode-success",
        "author_kind": "human",
        "scope": "full_episode",
    }
    rejected = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=[forged, reviews[1]],
        identity=_identity(),
    )
    assert rejected.status == STATUS_INCOMPLETE
    assert rejected.counts["reviews"]["full_episode_human"] == 1
    assert rejected.counts["reviews"]["untyped_or_invalid"] == 1

    serialized = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=[record_to_dict(item) for item in reviews],
        identity=_identity(),
    )
    assert serialized.status == STATUS_COMPLETE_UNDER_PROTOCOL


def test_ba02_deficit_adapter_keeps_revision_and_digest_directions() -> None:
    identity = AuditIdentity(
        campaign_digest="campaign-1",
        source_digest="source-1",
        release_digest="release-1",
        source_revision="source-commit-1",
        scan_revision="scan-report-1",
        detector_registry_digest="registry-1",
        detector_config_digest="detector-config-1",
    )
    report = evaluate_coverage(
        [{"episode_id": "e1", "planner_id": "p", "scenario_group": "g"}],
        detector_attempts=[{"episode_id": "e1", "detector_id": "telemetry", "status": "clear"}],
        detector_registry=_registry(),
        identity=identity,
    )
    deficit = next(item for item in report.deficits if item.reason == "missing_full_human_review")
    handoff = deficit.to_ba02_dict()
    assert handoff["source_revision"] == "scan-report-1"
    assert handoff["protocol_revision"] == report.protocol.version
    assert handoff["source_id"] == report.identity.source_digest
    assert handoff["protocol_id"] == report.protocol.digest
    assert handoff["source_revision"] != report.identity.source_digest


def test_report_deserialization_rejects_identity_deficit_nan_and_derived_tampering() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
    )
    payload = report.to_dict()
    missing_digest = deepcopy(payload)
    missing_digest.pop("report_digest")
    with pytest.raises(AuditCoverageError):
        report.from_dict(missing_digest)

    empty_identity = deepcopy(payload)
    empty_identity["identity"] = {}
    with pytest.raises(AuditCoverageError):
        report.from_dict(empty_identity)

    unwaived = deepcopy(payload)
    unwaived["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    unwaived["deficits"] = [
        {
            **evaluate_coverage(
                rows,
                detector_attempts=attempts,
                detector_registry=_registry(),
                identity=_identity(),
                review_records=reviews[:1],
            )
            .deficits[0]
            .to_dict()
        }
    ]
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(unwaived)

    nonfinite = deepcopy(payload)
    nonfinite["counts"]["coverage"]["expected"] = math.nan
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(nonfinite)

    unknown = deepcopy(payload)
    unknown["identity"]["unexpected"] = "forged"
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(unknown)

    inconsistent = deepcopy(payload)
    inconsistent["counts"]["status"] = STATUS_INCOMPLETE
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(inconsistent)


def test_report_semantics_reject_all_derived_and_status_tampering() -> None:
    rows, attempts, reviews = _complete_inputs()
    complete = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
    )
    complete_payload = complete.to_dict()

    def reject_complete(mutator, *, refresh_identity: bool = False) -> None:
        candidate = deepcopy(complete_payload)
        mutator(candidate)
        candidate = _redigest_identity(candidate) if refresh_identity else _redigest(candidate)
        with pytest.raises(AuditCoverageError):
            validate_audit_coverage(candidate)

    reject_complete(lambda item: item["identity"].update({"identity_digest": "0" * 64}))
    reject_complete(lambda item: item["protocol"].update({"protocol_digest": "0" * 64}))
    reject_complete(
        lambda item: item["identity"].update({"protocol_version": "different"}),
        refresh_identity=True,
    )
    reject_complete(
        lambda item: item["identity"].update({"protocol_digest": "1" * 64}),
        refresh_identity=True,
    )
    reject_complete(lambda item: item["counts"].update({"status": STATUS_INCOMPLETE}))
    reject_complete(lambda item: item["counts"]["coverage"].update({"expected": 3}))
    reject_complete(lambda item: item["counts"]["reviews"].update({"human_reviewed": 3}))
    reject_complete(lambda item: item["counts"]["materialization"].update({"total": 1}))
    reject_complete(lambda item: item["counts"]["diagnostics"].update({"tampered": 1}))
    reject_complete(lambda item: item["findings"].update({"candidate_clusters": 1}))
    reject_complete(lambda item: item["counts"]["reviews"].update({"observed_strata": 3}))
    reject_complete(lambda item: item["counts"]["reviews"].update({"full_human_strata": 3}))
    reject_complete(lambda item: item["counts"]["detectors"].update({"scheduled": 3}))

    def mutate_evaluable(item):
        item["counts"]["detectors"]["evaluable"] = 3
        item["counts"]["detectors"]["by_detector"]["telemetry"]["evaluable"] = 3

    reject_complete(mutate_evaluable)

    incomplete = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews[:1],
        identity=_identity(),
    )
    incomplete_payload = incomplete.to_dict()

    def reject_incomplete(mutator) -> None:
        candidate = deepcopy(incomplete_payload)
        mutator(candidate)
        with pytest.raises(AuditCoverageError):
            validate_audit_coverage(_redigest(candidate))

    reject_incomplete(lambda item: item.update({"deficits": []}))
    reject_incomplete(
        lambda item: item.update(
            {
                "status": STATUS_COMPLETE_UNDER_PROTOCOL,
                "counts": {**item["counts"], "status": STATUS_COMPLETE_UNDER_PROTOCOL},
            }
        )
    )

    def waive_all(item):
        item["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
        item["counts"]["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
        for deficit in item["deficits"]:
            deficit["status"] = "waived"

    reject_incomplete(waive_all)
    reject_incomplete(
        lambda item: (
            item.update({"status": STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS}),
            item["counts"].update({"status": STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS}),
            item.update({"deficits": []}),
        )
    )
    reject_complete(
        lambda item: item["identity"].update({"campaign_digest": ""}),
        refresh_identity=True,
    )

    for field in (
        "campaign_digest",
        "source_digest",
        "release_digest",
        "detector_registry_digest",
        "detector_config_digest",
    ):
        reject_incomplete(
            lambda item, field=field: item["deficits"][0].update({field: "different"})
        )
    reject_incomplete(lambda item: item["deficits"][0].update({"protocol_revision": "different"}))
    reject_incomplete(lambda item: item["deficits"][0].update({"protocol_id": "0" * 64}))
    reject_incomplete(lambda item: item["deficits"][0].update({"source_id": "different"}))


def test_recomputed_complete_report_cannot_hide_unmet_strata_or_controls() -> None:
    incomplete = evaluate_coverage(
        [
            {
                "episode_id": "unreviewed",
                "planner_id": "planner",
                "scenario_group": "corridor",
                "outcome": "success",
                "ordinary_control": True,
            }
        ],
        detector_attempts=[
            {"episode_id": "unreviewed", "detector_id": "telemetry", "status": "clear"}
        ],
        detector_registry=_registry(),
        identity=_identity(),
    )
    assert incomplete.status == STATUS_INCOMPLETE
    payload = incomplete.to_dict()
    payload["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["counts"]["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["deficits"] = []
    payload["exceptions"] = []
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(_redigest(payload))
    with pytest.raises(AuditCoverageError):
        incomplete.from_dict(payload)


def test_recomputed_complete_report_rejects_impossible_accounting_equations() -> None:
    rows, attempts, reviews = _complete_inputs()
    complete = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
    )
    mutators = (
        lambda item: item["strata"][0].update({"denominator": 0}),
        lambda item: item["strata"][0].update({"full_human_reviewed": 2}),
        lambda item: item["strata"][0].update(
            {"full_human_reviewed": 1, "reviewed_episode_ids": []}
        ),
        lambda item: (
            item["counts"]["coverage"].update({"indexed": 999, "readable": 0}),
            item["counts"]["inventory"].update({"indexed": 999, "readable": 0}),
        ),
    )
    for mutate in mutators:
        payload = complete.to_dict()
        mutate(payload)
        candidate = _redigest(payload)
        with pytest.raises(AuditCoverageError):
            validate_audit_coverage(candidate)
        with pytest.raises(AuditCoverageError):
            complete.from_dict(candidate)


def test_complete_report_requires_typed_detector_schedule_and_attempts() -> None:
    report = evaluate_coverage(
        [{"episode_id": "e", "planner_id": "p", "scenario_group": "g"}],
        review_records=[_review("review-e", "e")],
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    payload = report.to_dict()
    payload["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["counts"]["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["counts"]["detectors"] = {
        "scheduled": 1,
        "evaluable": 1,
        "flagged": 0,
        "clear": 1,
        "unavailable": 0,
        "error": 0,
        "by_detector": {
            "telemetry": {
                "scheduled": 1,
                "evaluable": 1,
                "flagged": 0,
                "clear": 1,
                "unavailable": 0,
                "error": 0,
            }
        },
    }
    payload["deficits"] = []
    payload["exceptions"] = []
    candidate = _redigest(payload)
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(candidate)
    with pytest.raises(AuditCoverageError):
        report.from_dict(candidate)


def test_complete_report_requires_reviewed_ordinary_control_candidates() -> None:
    rows = [
        {
            "episode_id": "control",
            "planner_id": "p",
            "scenario_group": "g",
            "ordinary_control": True,
        },
        {"episode_id": "reviewed", "planner_id": "p", "scenario_group": "g"},
    ]
    report = evaluate_coverage(
        rows,
        detector_attempts=[
            {"episode_id": "control", "detector_id": "telemetry", "status": "clear"},
            {"episode_id": "reviewed", "detector_id": "telemetry", "status": "clear"},
        ],
        review_records=[_review("reviewed-receipt", "reviewed")],
        detector_registry=_registry(),
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    payload = report.to_dict()
    payload["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["counts"]["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    for review_counts in (payload["counts"]["reviews"], payload["counts"]["review"]):
        review_counts["ordinary_control_reviewed"] = 1
        review_counts["ordinary_control_gap_groups"] = []
    payload["deficits"] = []
    payload["exceptions"] = []
    candidate = _redigest(payload)
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(candidate)
    with pytest.raises(AuditCoverageError):
        report.from_dict(candidate)


def test_complete_report_requires_materialization_rows_for_aggregate_counts() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        materialization=[{"episode_id": "episode-success", "status": "diverged"}],
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    payload = report.to_dict()
    verified = {"total": 1, "verified": 1, "diverged": 0, "unverifiable": 0, "unavailable": 0}
    payload["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["counts"]["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["counts"]["materialization"] = verified
    payload["materialization"] = verified
    payload["deficits"] = []
    payload["exceptions"] = []
    candidate = _redigest(payload)
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(candidate)
    with pytest.raises(AuditCoverageError):
        report.from_dict(candidate)


def test_complete_report_requires_finding_representative_typed_review() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews[:1],
        findings=[
            {
                "finding_id": "high-finding",
                "priority": "high",
                "candidate_members": ["episode-collision"],
                "representative_episode_id": "episode-collision",
            }
        ],
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    payload = report.to_dict()
    payload["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["counts"]["status"] = STATUS_COMPLETE_UNDER_PROTOCOL
    payload["findings"]["rows"][0]["representative_reviewed"] = True
    payload["counts"]["findings"]["representatives_reviewed"] = 1
    payload["deficits"] = []
    payload["exceptions"] = []
    candidate = _redigest(payload)
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(candidate)
    with pytest.raises(AuditCoverageError):
        report.from_dict(candidate)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda item: item["evidence"]["readable_episode_ids"].append("duplicate"),
        lambda item: item["evidence"]["detector_schedule"].append(
            dict(item["evidence"]["detector_schedule"][0])
        ),
        lambda item: item["evidence"]["detector_schedule"].append(
            {"episode_id": "unreadable", "detector_id": "telemetry"}
        ),
        lambda item: item["evidence"]["detector_attempts"].pop(),
        lambda item: item["evidence"]["detector_attempts"][0].update(
            {"status": "error", "reason_code": "", "message": ""}
        ),
        lambda item: item["evidence"]["detector_attempts"].append(
            dict(item["evidence"]["detector_attempts"][0])
        ),
        lambda item: item["evidence"]["detector_attempts"].append(
            {
                "episode_id": "episode-success",
                "detector_id": "other",
                "status": "clear",
                "detector_version": "",
                "reason_code": "",
                "message": "",
                "attempt_id": "other-attempt",
            }
        ),
        lambda item: item["counts"]["detectors"].update({"error": 1}),
        lambda item: item["counts"]["detectors"]["by_detector"].pop("telemetry"),
        lambda item: item["counts"]["detectors"]["by_detector"]["telemetry"].update({"clear": 0}),
        lambda item: item["evidence"]["ordinary_control_groups"][0]["candidate_episode_ids"].append(
            "episode-success"
        ),
        lambda item: item["evidence"]["ordinary_control_groups"].append(
            deepcopy(item["evidence"]["ordinary_control_groups"][0])
        ),
        lambda item: item["evidence"]["ordinary_control_groups"][0].update(
            {"reviewed_episode_ids": ["episode-collision"]}
        ),
    ),
)
def test_episode_evidence_rejects_detector_and_control_tampering(mutate) -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
    )
    payload = report.to_dict()
    mutate(payload)
    candidate = _redigest(payload)
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(candidate)
    with pytest.raises(AuditCoverageError):
        report.from_dict(candidate)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda item: item["evidence"].update(
            {
                "materialization_rows": [
                    {"episode_id": "same", "status": "verified"},
                    {"episode_id": "same", "status": "verified"},
                ]
            }
        ),
        lambda item: (
            item["findings"]["rows"][0].update({"representative_episode_id": "other"}),
            item["evidence"]["finding_reviews"][0].update({"representative_episode_id": "other"}),
        ),
        lambda item: (
            item["findings"]["rows"][0].update({"representative_reviewed": False}),
            item["evidence"]["finding_reviews"][0].update({"representative_reviewed": False}),
        ),
        lambda item: (
            item["findings"]["rows"][0].update({"control_reviewed": False}),
            item["evidence"]["finding_reviews"][0].update({"control_reviewed": False}),
        ),
    ),
)
def test_episode_evidence_rejects_materialization_and_finding_tampering(mutate) -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        findings=[
            {
                "finding_id": "high-finding",
                "priority": "high",
                "candidate_members": ["episode-success"],
                "representative_episode_id": "episode-success",
                "different_context_episode_id": "episode-collision",
            }
        ],
        identity=_identity(),
    )
    assert report.status == STATUS_COMPLETE_UNDER_PROTOCOL
    payload = report.to_dict()
    mutate(payload)
    candidate = _redigest(payload)
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(candidate)
    with pytest.raises(AuditCoverageError):
        report.from_dict(candidate)


def test_exceptions_must_be_nonempty_and_match_one_waived_deficit() -> None:
    rows, attempts, reviews = _complete_inputs()
    complete = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
    )
    unmatched = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
        exceptions={"bogus_requirement": "nothing to waive"},
    )
    assert unmatched.status == STATUS_COMPLETE_UNDER_PROTOCOL
    assert unmatched.exceptions == ()

    for exceptions in (
        [{}],
        [{"requirement": "bogus", "reason": "nothing to waive", "status": "waived"}],
        [{"requirement": "bogus", "reason": "", "status": "waived"}],
    ):
        payload = complete.to_dict()
        payload["status"] = STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS
        payload["counts"]["status"] = STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS
        payload["deficits"] = []
        payload["exceptions"] = exceptions
        with pytest.raises(AuditCoverageError):
            validate_audit_coverage(_redigest(payload))

    with pytest.raises(AuditCoverageError):
        evaluate_coverage(
            rows,
            detector_attempts=attempts,
            detector_registry=_registry(),
            review_records=reviews,
            identity=_identity(),
            exceptions={"bogus_requirement": ""},
        )

    waived = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
        materialization=[{"episode_id": "episode-success", "status": "diverged"}],
        exceptions={"materialization_mismatch": "offline"},
    )
    duplicate = waived.to_dict()
    duplicate["exceptions"].append(dict(duplicate["exceptions"][0]))
    with pytest.raises(AuditCoverageError):
        validate_audit_coverage(_redigest(duplicate))


def test_completion_helpers_reject_minimal_unkeyed_receipts() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
    )
    forged = {"status": STATUS_COMPLETE_UNDER_PROTOCOL, "identity_digest": report.identity.digest}
    assert not receipt_matches_identity(forged, report.identity)
    assert not is_completion_current(forged, report.identity)
    assert receipt_matches_identity(report.completion_receipt, report.identity)
    assert is_completion_current(report.completion_receipt, report.identity)


def test_report_exports_do_not_alias_nested_state_or_stale_digest() -> None:
    rows, attempts, reviews = _complete_inputs()
    report = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        identity=_identity(),
    )
    before = report.to_json()
    exported = report.to_dict()
    exported["counts"]["coverage"]["expected"] = 999
    exported["findings"]["rows"].append(
        {
            "finding_id": "forged",
            "candidate_members": [],
            "confirmed_members": [],
            "high_priority": False,
            "integrity": False,
            "status": "proposed",
            "representative_episode_id": "",
            "representative_reviewed": False,
            "control_episode_ids": [],
            "control_reviewed": False,
        }
    )
    exported["protocol"]["metadata"]["forged"] = True
    assert report.to_json() == before
    validate_audit_coverage(report.to_dict())


def test_deficit_dimensions_are_frozen_and_exported_without_aliases() -> None:
    report = evaluate_coverage(
        [{"episode_id": "e", "planner_id": "p", "scenario_group": "g"}],
        detector_attempts=[{"episode_id": "e", "detector_id": "d", "status": "clear"}],
        detector_registry=_registry("d"),
        identity=_identity(),
    )
    deficit = report.deficits[0]
    before = report.to_json()
    with pytest.raises(TypeError):
        deficit.dimensions["forged"] = "x"
    exported = deficit.to_dict()
    exported["dimensions"]["forged"] = "x"
    assert report.to_json() == before
    validate_audit_coverage(report.to_dict())


def test_unexpected_detector_attempt_cannot_replace_missing_registry_accounting() -> None:
    report = evaluate_coverage(
        [{"episode_id": "expected", "planner_id": "p", "scenario_group": "g"}],
        detector_attempts=[
            {"episode_id": "not-expected", "detector_id": "unregistered", "status": "clear"}
        ],
        review_records=[_review("review-expected", "expected")],
        identity=_identity(),
    )
    assert report.status == STATUS_INCOMPLETE
    assert report.counts["reviews"]["full_episode_human"] == 1
    assert report.counts["detectors"]["clear"] == 0
    assert report.counts["detectors"]["error"] == 1
    assert any(item.reason == "missing_detector_registry_or_attempts" for item in report.deficits)
    assert any(item.reason == "detector_attempt_error" for item in report.deficits)


def test_typed_review_receipt_must_match_active_source_and_scan_revision() -> None:
    identity = AuditIdentity(
        campaign_digest="campaign-1",
        source_digest="source-1",
        release_digest="release-1",
        source_revision="source-current",
        scan_revision="scan-current",
        detector_registry_digest="registry-1",
        detector_config_digest="detector-config-1",
    )
    report = evaluate_coverage(
        [{"episode_id": "e1", "planner_id": "p", "scenario_group": "g"}],
        detector_attempts=[{"episode_id": "e1", "detector_id": "telemetry", "status": "clear"}],
        detector_registry=_registry(),
        review_records=[ReviewRecord("review-e1", "e1", "full_episode", source_revision=7)],
        identity=identity,
    )
    assert report.status == STATUS_INCOMPLETE
    assert report.counts["reviews"]["full_episode_human"] == 0
    assert report.counts["reviews"]["stale_or_mismatched"] == 1
    assert any(item.reason == "missing_full_human_review" for item in report.deficits)


def test_finding_order_and_renderer_boundary_are_deterministic() -> None:
    rows, attempts, reviews = _complete_inputs()
    first = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        findings=[
            {"finding_id": "b", "candidate_members": ["episode-success"]},
            {"finding_id": "a", "candidate_members": ["episode-collision"]},
        ],
        identity=_identity(),
    )
    second = evaluate_coverage(
        rows,
        detector_attempts=attempts,
        detector_registry=_registry(),
        review_records=reviews,
        findings=[
            {"finding_id": "a", "candidate_members": ["episode-collision"]},
            {"finding_id": "b", "candidate_members": ["episode-success"]},
        ],
        identity=_identity(),
    )
    assert first.report_digest == second.report_digest
    assert first.to_json() == second.to_json()
    assert first.to_markdown() == second.to_markdown()
    assert first.to_html() == second.to_html()
    assert "full episode displayed" not in first.to_markdown().lower()
    assert "all required measurements were present" not in first.to_markdown().lower()
