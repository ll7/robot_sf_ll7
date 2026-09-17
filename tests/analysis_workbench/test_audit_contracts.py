"""Versioned BA-03 record and identity contracts."""

from __future__ import annotations

import pytest

from robot_sf.analysis_workbench.audit_contracts import (
    Annotation,
    AuditContractError,
    EpisodeRef,
    Reference,
    Signal,
    TimeInterval,
    deserialize_record,
    record_to_dict,
    serialize_record,
    validate_record,
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
    source = SourceRef(artifact_id="video", uri="video.mp4", format="video")
    with pytest.raises(AuditContractError, match="uncalibrated"):
        Reference(reference_id="r-1", coordinate_frame="world", point=(1.0, 2.0), source=source)
    image = Reference(
        reference_id="r-2", coordinate_frame="image", point=(10.0, 20.0), source=source
    )
    assert image.coordinate_frame == "image"


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
