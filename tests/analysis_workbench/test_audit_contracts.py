"""Versioned BA-03 record and identity contracts."""

from __future__ import annotations

import math

import pytest

from robot_sf.analysis_workbench.audit_contracts import (
    ActionRecord,
    Annotation,
    AuditContractError,
    AuditIdentityError,
    CampaignAudit,
    EpisodeRef,
    ImageDisplayTransform,
    Reference,
    Signal,
    TimeInterval,
    deserialize_record,
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
