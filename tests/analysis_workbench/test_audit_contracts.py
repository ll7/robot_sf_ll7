"""Versioned BA-03 record and identity contracts."""

from __future__ import annotations

import pytest

from robot_sf.analysis_workbench.audit_contracts import (
    Annotation,
    AuditContractError,
    AuditIdentityError,
    EpisodeRef,
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


def test_write_ndjson_cannot_create_a_record_only_journal(tmp_path) -> None:
    signal = Signal(signal_id="signal-1", detector_id="telemetry")
    with pytest.raises(AuditContractError, match="canonical audit journal"):
        write_ndjson([signal], tmp_path / "records.ndjson")
    write_ndjson([signal], tmp_path / "audit.ndjson")
    from robot_sf.analysis_workbench.audit_store import AuditStore

    with AuditStore(tmp_path) as store:
        assert store.get("signal-1") is not None
