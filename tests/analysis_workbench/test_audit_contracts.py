"""Versioned BA-03 record and identity contracts."""

from __future__ import annotations

import math

import pytest
from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.audit_contracts import (
    ActionRecord,
    Annotation,
    AuditContractError,
    AuditIdentityError,
    CampaignAudit,
    DetectorRuleProposal,
    EpisodeRef,
    ImageDisplayTransform,
    Reference,
    Signal,
    TimeInterval,
    deserialize_record,
    load_audit_record_schema,
    record_to_dict,
    serialize_record,
    validate_record,
    write_ndjson,
)
from robot_sf.analysis_workbench.review_contracts import SourceRef


def _episode(**changes) -> EpisodeRef:
    values = {
        "campaign_digest": "a" * 64,
        "source_digest": "b" * 64,
        "execution_id": "exec-1",
        "planner_id": "ppo",
        "scenario_id": "corridor",
        "seed": 5,
        "config_digest": "c" * 64,
    }
    values.update(changes)
    return EpisodeRef(**values)


def _proposal(**changes) -> DetectorRuleProposal:
    values = {
        "proposal_id": "proposal-1",
        "proposal_kind": "threshold_change",
        "target_detector_id": "goal_adjacent_timeout",
        "candidate_rule": {
            "predicate": "goal_adjacent_timeout.v1",
            "parameters": {"tail_steps": 120, "wall_margin_m": 0.5},
        },
        "detector_registry_version": "audit-detector-registry.v1",
        "detector_registry_digest": "d" * 64,
        "campaign_digest": "a" * 64,
        "source_identity": "source-1",
        "source_revision": "commit-1",
        "annotation_ids": ("annotation-1",),
        "finding_ids": ("finding-1",),
        "episode_ids": ("episode-1",),
        "rationale": "The observed tail behavior warrants a bounded threshold probe.",
        "metadata": {"evidence_boundary": "diagnostic_only"},
        "proposer_kind": "agent",
        "proposer_id": "agent-1",
        "author_kind": "agent",
        "author_id": "agent-1",
    }
    values.update(changes)
    return DetectorRuleProposal(**values)


def test_episode_identity_does_not_collapse_reruns_with_same_lookup_fields() -> None:
    first = _episode(execution_id="run-1")
    second = _episode(execution_id="run-2")
    changed_config = _episode(execution_id="run-1", config_digest="d" * 64)
    assert first.episode_id != second.episode_id
    assert first.episode_id != changed_config.episode_id
    assert first.identity_key != second.identity_key


def test_episode_identity_includes_campaign_for_second_campaign_probe() -> None:
    first = _episode(campaign_digest="a" * 64)
    second_campaign = _episode(campaign_digest="d" * 64)
    assert first.episode_id != second_campaign.episode_id
    assert first.identity_key != second_campaign.identity_key


def test_episode_id_cannot_override_collision_resistant_identity() -> None:
    with pytest.raises(AuditIdentityError, match="episode_id"):
        _episode(episode_id="caller-chosen")


def test_episode_source_digest_must_bind_source_sha256() -> None:
    source = SourceRef(
        artifact_id="trace",
        uri="trace.json",
        format="trace",
        sha256="d" * 64,
    )
    with pytest.raises(AuditIdentityError, match="source_digest"):
        _episode(source=source)
    bound = _episode(source=source, source_digest="d" * 64)
    assert bound.source == source


def test_quick_annotation_does_not_require_cause_confidence_or_full_review() -> None:
    annotation = Annotation(
        annotation_id="a-1",
        episode_id=_episode().episode_id,
        classification="interesting_valid",
        mode="quick",
        interval=TimeInterval(0.5),
    )
    assert annotation.suspected_cause == ""
    assert annotation.confidence is None
    assert not annotation.is_full_episode_review


def test_reference_rejects_world_coordinates_from_uncalibrated_video() -> None:
    for format_name in ("video", "video/mp4", "video-mp4.v1", "image/jpeg"):
        source = SourceRef(artifact_id="video", uri="video.mp4", format=format_name)
        with pytest.raises(AuditContractError, match="uncalibrated"):
            Reference(
                reference_id=f"r-{format_name}",
                coordinate_frame="world",
                point=(1.0, 2.0),
                source=source,
            )
    source = SourceRef(artifact_id="video", uri="video.mp4", format="video/mp4")
    image = Reference(
        reference_id="r-2", coordinate_frame="image", point=(10.0, 20.0), source=source
    )
    assert image.coordinate_frame == "image"


def test_image_display_transform_round_trips_crop_resize_and_preserves_seek_identity() -> None:
    transform = ImageDisplayTransform(
        source_width=1000,
        source_height=800,
        crop_x=100,
        crop_y=50,
        crop_width=400,
        crop_height=200,
        display_width=800,
        display_height=400,
    )
    display_point = transform.source_to_display((300.0, 100.0))
    assert display_point == (400.0, 100.0)
    assert transform.display_to_source(display_point) == (300.0, 100.0)
    source = SourceRef(artifact_id="frame", uri="video.mp4", format="image/jpeg")
    reference = Reference(
        reference_id="r-image",
        coordinate_frame="image",
        point=(300.0, 100.0),
        source=source,
        timestamp_s=12.5,
        source_revision="commit-1",
        calibration=transform,
        seek_identity="pts:375000",
    )
    assert reference.source_point == (300.0, 100.0)
    assert reference.timestamp_s == 12.5
    assert reference.seek_identity == "pts:375000"
    restored = deserialize_record(serialize_record(reference))
    assert restored == reference


def test_image_display_transform_rejects_invalid_bounds_and_opaque_calibration() -> None:
    with pytest.raises(AuditContractError, match="source_width"):
        ImageDisplayTransform(
            source_width=0,
            source_height=100,
            display_width=100,
            display_height=100,
        )
    with pytest.raises(AuditContractError, match="finite"):
        ImageDisplayTransform(
            source_width=100,
            source_height=100,
            display_width=100,
            display_height=100,
            crop_x=math.nan,
        )
    with pytest.raises(AuditContractError, match="crop"):
        ImageDisplayTransform(
            source_width=100,
            source_height=100,
            display_width=100,
            display_height=100,
            crop_x=90,
            crop_width=20,
        )
    source = SourceRef(artifact_id="frame", uri="frame.jpg", format="image/jpeg")
    with pytest.raises(AuditContractError, match="transform"):
        Reference(
            reference_id="r-opaque",
            coordinate_frame="image",
            point=(1.0, 2.0),
            source=source,
            calibration=1.0,
        )
    with pytest.raises(AuditContractError, match="calibration"):
        Reference(
            reference_id="r-world-bad",
            coordinate_frame="world",
            point=(1.0, 2.0),
            source=SourceRef(artifact_id="video", uri="video.mp4", format="video/mp4"),
            calibration={"scale": 2.0},
        )
    valid_world_calibration = {
        "kind": "world_from_image.v1",
        "version": 1,
        "parameters": {"matrix": [[1.0, 0.0], [0.0, 1.0]]},
    }
    world = Reference(
        reference_id="r-world-good",
        coordinate_frame="world",
        point=(1.0, 2.0),
        source=SourceRef(artifact_id="video", uri="video.mp4", format="video/mp4"),
        calibration=valid_world_calibration,
    )
    assert world.calibration == valid_world_calibration


def test_reference_source_revision_must_bind_source_commit() -> None:
    source = SourceRef(
        artifact_id="frame",
        uri="frame.jpg",
        format="image/jpeg",
        source_commit="commit-1",
    )
    with pytest.raises(AuditIdentityError, match="source_revision"):
        Reference(
            reference_id="r-mismatch",
            coordinate_frame="image",
            point=(1.0, 2.0),
            source=source,
            source_revision="commit-2",
        )
    assert (
        Reference(
            reference_id="r-match",
            coordinate_frame="image",
            point=(1.0, 2.0),
            source=source,
            source_revision="commit-1",
        ).source_revision
        == "commit-1"
    )


def test_campaign_source_digest_must_bind_source_sha256() -> None:
    source = SourceRef(
        artifact_id="campaign",
        uri="campaign.json",
        format="json",
        sha256="d" * 64,
    )
    with pytest.raises(AuditIdentityError, match="source_digest"):
        CampaignAudit(
            audit_id="audit-1",
            campaign_digest="a" * 64,
            source_digest="b" * 64,
            source=source,
        )


def test_annotation_source_identity_mismatch_is_rejected_and_unavailable_is_explicit() -> None:
    source = SourceRef(
        artifact_id="trace",
        uri="trace.json",
        format="trace",
        sha256="d" * 64,
    )
    with pytest.raises(AuditIdentityError, match="source_identity"):
        Annotation(
            annotation_id="a-1",
            episode_id=_episode(source=source, source_digest="d" * 64).episode_id,
            classification="interesting_valid",
            source_ref=source,
            source_identity="e" * 64,
        )
    unavailable = Annotation(
        annotation_id="a-2",
        episode_id=_episode().episode_id,
        classification="interesting_valid",
        provenance_status="unavailable",
    )
    assert unavailable.provenance_status == "unavailable"
    stale = Annotation(
        annotation_id="a-3",
        episode_id=_episode().episode_id,
        classification="interesting_valid",
        source_ref=source,
        source_identity="e" * 64,
        provenance_status="stale",
    )
    assert stale.provenance_status == "stale"
    committed = SourceRef(
        artifact_id="trace",
        uri="trace.json",
        format="trace",
        sha256="d" * 64,
        source_commit="rev-2",
    )
    unavailable_revision = Annotation(
        annotation_id="a-4",
        episode_id=_episode(source=committed, source_digest="d" * 64).episode_id,
        classification="interesting_valid",
        source_ref=committed,
        source_identity="d" * 64,
    )
    assert unavailable_revision.provenance_status == "unavailable"
    with pytest.raises(AuditIdentityError, match="source_revision"):
        Annotation(
            annotation_id="a-5",
            episode_id=_episode(source=committed, source_digest="d" * 64).episode_id,
            classification="interesting_valid",
            source_ref=committed,
            source_identity="d" * 64,
            source_revision="rev-1",
        )


def test_missing_signal_is_explicit_and_records_round_trip() -> None:
    signal = Signal(
        signal_id="signal-1",
        detector_id="telemetry",
        status="unavailable",
        missingness=("video_stream",),
        message="optional video was not recorded",
    )
    payload = record_to_dict(signal)
    validate_record(payload)
    assert deserialize_record(serialize_record(signal)) == signal


def test_detector_rule_proposal_is_typed_strict_and_round_trips() -> None:
    proposal = _proposal()
    payload = record_to_dict(proposal)

    validate_record(payload)
    assert payload["record_type"] == "detector_rule_proposal"
    assert payload["activation_status"] == "inactive"
    assert deserialize_record(serialize_record(proposal)) == proposal


@pytest.mark.parametrize(
    "executable_key",
    (
        "callable",
        "module",
        "shell_command",
        "python_expression",
        "command",
        "CALL-BACK",
        "shell.command",
        "pythonexpression",
        "__class__",
        "__code__",
        "call_back",
        "call__back",
        "call_back_handler",
        "class_name",
        "code_path",
        "c_l_a_s_s_name",
        "c_o_d_e_path",
        "e_xec",
        "c_a_l_l_a_b_l_e",
        "r_un_time",
        "p_ython",
        "c_ommand",
        "e_ntrypoint",
    ),
)
def test_detector_rule_proposal_rejects_executable_candidate_keys(executable_key: str) -> None:
    with pytest.raises(AuditContractError, match="declarative|executable"):
        _proposal(candidate_rule={"predicate": "candidate", executable_key: "run-me"})


def test_detector_rule_proposal_schema_rejects_nested_executable_keys() -> None:
    schema = load_audit_record_schema()
    validator = Draft202012Validator(schema)
    for executable_key in (
        "CALL-BACK",
        "shell.command",
        "pythonexpression",
        "call_back",
        "call__back",
        "call_back_handler",
        "c_l_a_s_s_name",
        "c_o_d_e_path",
        "e_xec",
        "c_a_l_l_a_b_l_e",
        "r_un_time",
        "p_ython",
        "c_ommand",
        "e_ntrypoint",
    ):
        payload = record_to_dict(_proposal())
        payload["candidate_rule"] = {"outer": {"inner": {executable_key: "run-me"}}}
        assert list(validator.iter_errors(payload))
        with pytest.raises(AuditContractError):
            deserialize_record(payload)


def test_detector_rule_proposal_schema_enforces_human_decision_gate() -> None:
    validator = Draft202012Validator(load_audit_record_schema())
    approved = _proposal(
        lifecycle_status="approved",
        author_kind="human",
        author_id="reviewer-1",
        decided_by_kind="human",
        decided_by_id="reviewer-1",
        decided_at="2026-09-20T10:00:00Z",
        decision_reason="Human review accepted this diagnostic candidate.",
    )
    valid = record_to_dict(approved)
    assert list(validator.iter_errors(valid)) == []

    invalid_payloads = {}
    payload = record_to_dict(_proposal())
    payload["lifecycle_status"] = "approved"
    invalid_payloads["approved_missing_decision"] = payload
    payload = record_to_dict(_proposal())
    payload["lifecycle_status"] = "approved"
    invalid_payloads["approved_agent_author"] = payload
    payload = record_to_dict(_proposal())
    payload["lifecycle_status"] = "approved"
    payload["author_kind"] = "human"
    invalid_payloads["approved_human_missing_fields"] = payload
    payload = record_to_dict(_proposal())
    payload.update(
        {
            "lifecycle_status": "approved",
            "author_kind": "human",
            "decided_by_kind": "human",
            "decided_by_id": "   ",
            "decided_at": "2026-09-20T10:00:00Z",
            "decision_reason": "Human review accepted this diagnostic candidate.",
        }
    )
    invalid_payloads["approved_whitespace_decider"] = payload
    payload = record_to_dict(_proposal())
    payload.update(
        {
            "lifecycle_status": "approved",
            "author_kind": "human",
            "decided_by_kind": "human",
            "decided_by_id": "reviewer-1",
            "decided_at": "   ",
            "decision_reason": "Human review accepted this diagnostic candidate.",
        }
    )
    invalid_payloads["approved_whitespace_decided_at"] = payload
    payload = record_to_dict(_proposal())
    payload.update(
        {
            "lifecycle_status": "approved",
            "author_kind": "human",
            "decided_by_kind": "human",
            "decided_by_id": "reviewer-1",
            "decided_at": "2026-09-20T10:00:00Z",
            "decision_reason": "   ",
        }
    )
    invalid_payloads["approved_whitespace_reason"] = payload
    payload = record_to_dict(_proposal())
    payload["rationale"] = "   "
    invalid_payloads["whitespace_rationale"] = payload
    payload = record_to_dict(_proposal())
    payload["source_revision"] = "   "
    invalid_payloads["whitespace_source_revision"] = payload
    payload = record_to_dict(_proposal())
    payload["metadata"] = {"   ": True}
    invalid_payloads["whitespace_metadata_key"] = payload
    invalid_payloads["proposed_with_decision"] = record_to_dict(_proposal())
    payload = record_to_dict(_proposal())
    payload.update(
        {
            "lifecycle_status": "rejected",
            "author_kind": "human",
            "author_id": "reviewer-1",
            "decided_by_kind": "agent",
            "decided_by_id": "agent-1",
            "decided_at": "2026-09-20T10:00:00Z",
            "decision_reason": "Agent cannot decide.",
        }
    )
    invalid_payloads["rejected_agent_decider"] = payload
    invalid_payloads["proposed_with_decision"]["decided_by_kind"] = "human"
    invalid_payloads["proposed_with_decision"]["decided_by_id"] = "reviewer-1"
    invalid_payloads["proposed_with_decision"]["decided_at"] = "2026-09-20T10:00:00Z"
    invalid_payloads["proposed_with_decision"]["decision_reason"] = "Human decision"
    for name, payload in invalid_payloads.items():
        assert list(validator.iter_errors(payload)), name


def test_detector_rule_proposal_nested_candidate_is_immutable_after_construction() -> None:
    proposal = _proposal()
    with pytest.raises(TypeError):
        proposal.candidate_rule["parameters"]["tail_steps"] = 121
    with pytest.raises(TypeError):
        proposal.candidate_rule["parameters"]["shell.command"] = "run-me"
    assert deserialize_record(serialize_record(proposal)) == proposal


def test_detector_rule_proposal_requires_human_decision_and_stays_inactive() -> None:
    with pytest.raises(AuditContractError, match="human|decision"):
        _proposal(lifecycle_status="approved")
    with pytest.raises(AuditContractError, match="inactive"):
        _proposal(activation_status="active")
    with pytest.raises(AuditContractError, match="decision fields"):
        _proposal(decided_by_kind="human")
    with pytest.raises(AuditContractError, match="human"):
        _proposal(
            lifecycle_status="rejected",
            decided_by_kind="agent",
            decided_by_id="agent-1",
            decided_at="2026-09-20T10:00:00Z",
            decision_reason="agent cannot decide",
        )
    with pytest.raises(AuditContractError, match="decision"):
        _proposal(
            lifecycle_status="approved",
            author_kind="human",
            decided_by_kind="human",
            decided_by_id="reviewer-1",
            decided_at="2026-09-20T10:00:00Z",
            decision_reason="   ",
        )


def test_detector_rule_proposal_payload_rejects_unknown_root_fields() -> None:
    payload = record_to_dict(_proposal())
    payload["unexpected"] = True
    with pytest.raises(AuditContractError):
        deserialize_record(payload)


def test_action_record_payload_validates_with_closed_root_properties() -> None:
    action = ActionRecord(action_id="action-1", action_type="review", actor_kind="agent")
    payload = record_to_dict(action)
    validate_record(payload)
    assert deserialize_record(serialize_record(action)) == action


def test_deserialization_requires_version_and_type_specific_identity() -> None:
    signal = Signal(signal_id="signal-1", detector_id="telemetry")
    payload = record_to_dict(signal)
    without_version = dict(payload)
    without_version.pop("schema_version")
    with pytest.raises(AuditContractError, match="schema_version"):
        deserialize_record(without_version)
    without_id = dict(payload)
    without_id.pop("record_id")
    with pytest.raises(AuditContractError, match="record_id"):
        validate_record(without_id)
    malformed = dict(payload)
    malformed["record_type"] = "annotation"
    with pytest.raises(AuditContractError):
        deserialize_record(malformed)
    with pytest.raises(AuditContractError, match="invalid audit JSON"):
        deserialize_record("{not-json")


def test_deserialization_rejects_nonfinite_numbers_and_unknown_fields() -> None:
    signal = Signal(signal_id="signal-strict", detector_id="telemetry")
    unknown = record_to_dict(signal)
    unknown["unexpected"] = True
    with pytest.raises(AuditContractError):
        deserialize_record(unknown)
    nonfinite = record_to_dict(signal)
    nonfinite["measured"] = {"score": float("nan")}
    with pytest.raises(AuditContractError):
        deserialize_record(nonfinite)
    with pytest.raises(AuditContractError):
        deserialize_record(
            '{"schema_version":"audit-record.v1","record_type":"signal",'
            '"record_id":"signal-strict","signal_id":"signal-strict",'
            '"detector_id":"telemetry","measured":{"score":NaN}}'
        )


def test_deserialization_rejects_non_string_closed_text_fields() -> None:
    annotation = Annotation(
        annotation_id="strict-text",
        episode_id=_episode().episode_id,
        classification="unclear",
    )
    payload = record_to_dict(annotation)
    for field_name in ("observed_behavior", "author_id", "created_at", "provenance_reason"):
        malformed = dict(payload)
        malformed[field_name] = 1
        with pytest.raises(AuditContractError, match=field_name):
            deserialize_record(malformed)


def test_write_ndjson_cannot_create_a_record_only_journal(tmp_path) -> None:
    signal = Signal(signal_id="signal-1", detector_id="telemetry")
    with pytest.raises(AuditContractError, match="canonical audit journal"):
        write_ndjson([signal], tmp_path / "records.ndjson")
    write_ndjson([signal], tmp_path / "audit.ndjson")
    from robot_sf.analysis_workbench.audit_store import AuditStore

    with AuditStore(tmp_path) as store:
        assert store.get("signal-1") is not None


def test_write_ndjson_round_trips_store_journal_and_repeats_are_noops(tmp_path) -> None:
    from robot_sf.analysis_workbench.audit_contracts import read_ndjson

    signal = Signal(signal_id="z-signal-journal", detector_id="telemetry")
    first = Signal(signal_id="a-signal-journal", detector_id="telemetry")
    journal = tmp_path / "audit.ndjson"
    write_ndjson([signal, first], journal)
    before = journal.read_bytes()
    write_ndjson([signal, first], journal)
    assert journal.read_bytes() == before
    assert read_ndjson(journal) == [signal, first]
    changed = Signal(signal_id="z-signal-journal", detector_id="telemetry", reason_code="new")
    write_ndjson([changed, first], journal)
    assert read_ndjson(journal) == [changed, first]
