"""Focused tests for the SREV-03 video-sync component (issue #9272)."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_result_from_dict,
)
from robot_sf.render import video_sync
from robot_sf.render.video_sync import (
    COMPONENT_ID,
    _result_document,
    descriptor,
    run,
)

FIXTURE_DIR = "tests/fixtures/scenario_review/video_sync"

CAPTURE = {
    "fps_nominal": 10.0,
    "camera": {"id": "cam-t"},
    "frames": [
        {"frame_index": 0, "pts_s": 0.0},
        {"frame_index": 1, "pts_s": 0.1},
        {"frame_index": 2, "pts_s": 0.2},
    ],
}

STAMPS = {
    "dt_s": 0.1,
    "steps": [
        {"step": s, "time_s": round(s * 0.1, 3), "episode_id": "ep-t", "reset_id": "r-t"}
        for s in range(5)
    ],
}


def _stage(tmp_path: Path) -> None:
    (tmp_path / "capture.json").write_text(json.dumps(CAPTURE), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")


def _request(
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict | None = None,
    output: str = "out",
    sources: list[dict] | None = None,
) -> dict:
    from robot_sf.render.video_sync import (
        component_request_from_dict,
    )

    config: dict = {
        "presentation": {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0},
        "time_origin_s": 0.0,
    }
    config.update(config_extra or {})
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": component_id,
        "sources": (
            sources
            if sources is not None
            else [
                {"artifact_id": "capture", "uri": "capture.json", "format": "capture-frames"},
                {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
            ]
        ),
        "output_directory": output,
        "config": config,
        "required_capabilities": list(required),
    }
    return component_request_from_dict(payload)


def test_success_maps_frames_to_steps_without_touching_sources(tmp_path: Path) -> None:
    _stage(tmp_path)
    source_before = (tmp_path / "capture.json").read_bytes()
    result = run(_request(), base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    assert (tmp_path / "capture.json").read_bytes() == source_before
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert [e["frame_index"] for e in mapping["entries"]] == [0, 1, 2]
    assert [e["sim_step"] for e in mapping["entries"]] == [0, 1, 2]
    assert [e["sim_stamp_time_s"] for e in mapping["entries"]] == [0.0, 0.1, 0.2]
    assert all(e["temporal_error_s"] == 0.0 for e in mapping["entries"])
    assert mapping["reset_ids"] == ["r-t"]
    assert mapping["first_frame"]["frame_index"] == 0
    assert mapping["last_frame"]["frame_index"] == 2
    assert len(result.artifacts) == 1
    artifact = dict(result.artifacts[0])
    raw = (tmp_path / "out" / "media-mapping.json").read_bytes()
    import hashlib

    assert artifact["sha256"] == hashlib.sha256(raw).hexdigest()


def test_deterministic_repeat_runs_match_digest(tmp_path: Path) -> None:
    _stage(tmp_path)
    first = run(_request(output="out-a"), base=tmp_path)
    second = run(_request(output="out-b"), base=tmp_path)
    assert first.status == second.status == "complete"
    a = (tmp_path / "out-a" / "media-mapping.json").read_bytes()
    b = (tmp_path / "out-b" / "media-mapping.json").read_bytes()
    assert a == b


def test_skipped_frames_are_explicit_and_partial(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [f for f in CAPTURE["frames"] if f["frame_index"] != 1]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status == "partial"
    assert "skipped_frames:1" in result.reason
    assert result.artifacts == ()
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert mapping["skipped_frame_indexes"] == [1]


def test_frames_outside_stamp_range_are_partial(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [{"frame_index": 0, "pts_s": 99.0}]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status == "partial"
    assert "frames_outside_stamp_range" in result.reason


def test_after_last_frames_are_unavailable_without_a_stale_anchor(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [
        {"frame_index": 0, "pts_s": 0.0},
        {
            "frame_index": 6,
            "pts_s": 0.6,
            "episode_id": "capture-ep",
            "reset_id": "capture-reset",
        },
    ]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "partial"
    assert "frame_6_after_last_sim_stamp" in result.reason
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert [entry["frame_index"] for entry in mapping["entries"]] == [0]
    assert mapping["unavailable_frame_indexes"] == [6]
    unavailable = mapping["unavailable_frames"][0]
    assert unavailable["status"] == "unavailable"
    assert unavailable["reason"] == "after_last_sim_stamp"
    assert "sim_step" not in unavailable
    assert "episode_id" not in unavailable
    assert "reset_id" not in unavailable
    assert unavailable["capture_episode_id"] == "capture-ep"
    assert unavailable["capture_reset_id"] == "capture-reset"
    assert mapping["last_frame"]["status"] == "unavailable"


def test_before_first_frames_are_unavailable_with_temporal_error(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [
        {"frame_index": 0, "pts_s": -0.1, "episode_id": "capture-ep", "reset_id": "capture-reset"}
    ]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "partial"
    assert "frame_0_before_first_sim_stamp" in result.reason
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    unavailable = mapping["unavailable_frames"][0]
    assert unavailable["reason"] == "before_first_sim_stamp"
    assert unavailable["temporal_error_s"] == pytest.approx(0.1)
    assert unavailable["capture_episode_id"] == "capture-ep"
    assert unavailable["capture_reset_id"] == "capture-reset"


def test_matching_capture_identity_is_retained_on_mapped_entry(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [{"frame_index": 0, "pts_s": 0.0, "episode_id": "ep-t", "reset_id": "r-t"}]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "complete"
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert mapping["entries"][0]["capture_episode_id"] == "ep-t"
    assert mapping["entries"][0]["capture_reset_id"] == "r-t"


def test_nonexact_timestamp_is_unavailable_without_silent_floor_anchor(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [{"frame_index": 0, "pts_s": 0.15}]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "partial"
    assert "frame_0_no_sim_stamp_within_tolerance" in result.reason
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert mapping["entries"] == []
    assert mapping["timestamp_alignment"] == {
        "policy": "unique_sim_stamp_within_tolerance",
        "tolerance_s": 0.0,
    }
    unavailable = mapping["unavailable_frames"][0]
    assert unavailable["reason"] == "no_sim_stamp_within_tolerance"
    assert unavailable["temporal_error_s"] == pytest.approx(0.05)


def test_timestamp_tolerance_requires_a_unique_anchor_and_records_error(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [{"frame_index": 0, "pts_s": 0.14}]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(config_extra={"timestamp_tolerance_s": 0.05}), base=tmp_path)

    assert result.status == "complete"
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    entry = mapping["entries"][0]
    assert entry["sim_step"] == 1
    assert entry["sim_stamp_time_s"] == pytest.approx(0.1)
    assert entry["temporal_error_s"] == pytest.approx(0.04)
    assert mapping["timestamp_alignment"]["tolerance_s"] == pytest.approx(0.05)


def test_timestamp_tolerance_rejects_ambiguous_anchor(tmp_path: Path) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [{"frame_index": 0, "pts_s": 0.15}]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(config_extra={"timestamp_tolerance_s": 0.05}), base=tmp_path)

    assert result.status == "partial"
    assert "frame_0_ambiguous_sim_stamp_match" in result.reason
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert mapping["entries"] == []
    assert mapping["unavailable_frames"][0]["temporal_error_s"] == pytest.approx(0.05)


def test_invalid_timestamp_tolerance_fails_closed(tmp_path: Path) -> None:
    _stage(tmp_path)

    result = run(_request(config_extra={"timestamp_tolerance_s": -0.01}), base=tmp_path)

    assert result.status == "failed"
    assert "timestamp_tolerance_invalid" in result.reason
    assert not (tmp_path / "out").exists()


def test_run_accepts_a_string_base_without_changing_resolution(tmp_path: Path) -> None:
    """The public adapter accepts path-like CLI bases and keeps them contained."""
    _stage(tmp_path)

    result = run(_request(), base=str(tmp_path))

    assert result.status == "complete"
    assert (tmp_path / "out" / "media-mapping.json").is_file()


def test_duplicate_source_json_keys_fail_closed(tmp_path: Path) -> None:
    """Duplicate source keys cannot silently replace provenance-relevant values."""
    (tmp_path / "capture.json").write_text(
        '{"fps_nominal":10,"fps_nominal":11,"frames":[]}', encoding="utf-8"
    )
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "capture: source_not_strict_json" in result.reason
    assert not (tmp_path / "out").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX mkfifo")
def test_fifo_source_is_rejected_without_blocking(tmp_path: Path) -> None:
    """A special source file is rejected before the mapper attempts to read it."""
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")
    fifo = tmp_path / "capture.fifo"
    os.mkfifo(fifo)
    sources = [
        {"artifact_id": "capture", "uri": fifo.name, "format": "capture-frames"},
        {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
    ]

    result = run(_request(sources=sources), base=tmp_path)

    assert result.status == "failed"
    assert "capture: source_not_regular_file" in result.reason
    assert not (tmp_path / "out").exists()


def test_source_byte_limit_is_reported_before_json_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversized diagnostic sources fail at the bounded file-read boundary."""
    _stage(tmp_path)
    monkeypatch.setattr(video_sync, "MAX_JSON_BYTES", 256)
    (tmp_path / "capture.json").write_bytes(b"{" + b"x" * 512)

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "capture: resource_limit:source_bytes" in result.reason
    assert not (tmp_path / "out").exists()


def test_frame_row_limit_fails_before_materializing_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frame cardinality is bounded before mapping or output publication."""
    _stage(tmp_path)
    monkeypatch.setattr(video_sync, "MAX_FRAME_ROWS", 2)
    capture = dict(CAPTURE)
    capture["frames"] = [{"frame_index": index, "pts_s": index / 10.0} for index in range(3)]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "resource_limit:capture_frame_rows:2" in result.reason
    assert not (tmp_path / "out").exists()


def test_frame_index_span_limit_fails_before_gap_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sparse index span cannot force an unbounded skipped-frame list."""
    _stage(tmp_path)
    monkeypatch.setattr(video_sync, "MAX_FRAME_INDEX_SPAN", 1)
    capture = dict(CAPTURE)
    capture["frames"] = [
        {"frame_index": 0, "pts_s": 0.0},
        {"frame_index": 2, "pts_s": 0.2},
    ]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "resource_limit:frame_index_span:1" in result.reason
    assert not (tmp_path / "out").exists()


def test_episode_and_reset_boundaries_are_preserved_in_source_order(tmp_path: Path) -> None:
    _stage(tmp_path)
    stamps = {
        "steps": [
            {"step": 0, "time_s": 0.0, "episode_id": "ep-a", "reset_id": "reset-a"},
            {"step": 1, "time_s": 0.1, "episode_id": "ep-a", "reset_id": "reset-a"},
            {"step": 0, "time_s": 0.2, "episode_id": "ep-b", "reset_id": "reset-b"},
            {"step": 1, "time_s": 0.3, "episode_id": "ep-b", "reset_id": "reset-b"},
        ]
    }
    (tmp_path / "stamps.json").write_text(json.dumps(stamps), encoding="utf-8")
    capture = dict(CAPTURE)
    capture["frames"] = [
        {"frame_index": 0, "pts_s": 0.0},
        {"frame_index": 1, "pts_s": 0.2},
        {"frame_index": 2, "pts_s": 0.3},
    ]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "complete"
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert [entry["episode_id"] for entry in mapping["entries"]] == ["ep-a", "ep-b", "ep-b"]
    assert mapping["episode_ids"] == ["ep-a", "ep-b"]
    assert mapping["reset_ids"] == ["reset-a", "reset-b"]
    assert mapping["episode_reset_boundaries"] == [
        {
            "source_index": 0,
            "time_s": 0.0,
            "step": 0,
            "episode_id": "ep-a",
            "reset_id": "reset-a",
            "kind": "initial",
        },
        {
            "source_index": 2,
            "time_s": 0.2,
            "step": 0,
            "episode_id": "ep-b",
            "reset_id": "reset-b",
            "kind": "episode_or_reset_change",
        },
    ]


def test_repeated_simulation_times_are_explicitly_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    stamps = {
        "steps": [
            {"step": 0, "time_s": 0.0, "episode_id": "ep-a", "reset_id": "reset-a"},
            {"step": 1, "time_s": 0.1, "episode_id": "ep-a", "reset_id": "reset-a"},
            {"step": 0, "time_s": 0.1, "episode_id": "ep-b", "reset_id": "reset-b"},
            {"step": 1, "time_s": 0.2, "episode_id": "ep-b", "reset_id": "reset-b"},
        ]
    }
    (tmp_path / "stamps.json").write_text(json.dumps(stamps), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "partial"
    assert "ambiguous_repeated_sim_time:0.1" in result.reason
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert mapping["entries"] == []
    assert {item["reason"] for item in mapping["unavailable_frames"]} == {
        "ambiguous_sim_stamp_timeline"
    }
    assert [item["episode_id"] for item in mapping["episode_reset_boundaries"]] == [
        "ep-a",
        "ep-b",
    ]


@pytest.mark.parametrize(
    ("presentation", "reason"),
    [
        (None, "presentation_invalid"),
        ({"width": 320.0}, "presentation_width_invalid"),
        ({"height": False}, "presentation_height_invalid"),
        ({"fps": "10"}, "presentation_fps_invalid"),
        ({"fps": 0.0}, "presentation_fps_invalid"),
        ({"speed": "fast"}, "presentation_speed_invalid"),
        ({"speed": 0.0}, "presentation_speed_invalid"),
    ],
)
def test_invalid_presentation_values_fail_closed(
    tmp_path: Path, presentation: object, reason: str
) -> None:
    _stage(tmp_path)
    result = run(_request(config_extra={"presentation": presentation}), base=tmp_path)

    assert result.status == "failed"
    assert reason in result.reason
    assert not (tmp_path / "out").exists()


def test_nonfinite_presentation_value_is_rejected_as_non_strict_request(
    tmp_path: Path,
) -> None:
    _stage(tmp_path)
    result = run(_request(config_extra={"presentation": {"fps": float("nan")}}), base=tmp_path)

    assert result.status == "failed"
    assert result.reason == "invalid_request: config must be strict-JSON safe"
    assert not (tmp_path / "out").exists()


def test_corrupt_source_yields_partial_or_failed(tmp_path: Path) -> None:
    (tmp_path / "capture.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status in ("partial", "failed")
    assert result.artifacts == ()


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(required=("rvo2-binary",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_known_required_camera_calibration_is_unavailable_when_missing(tmp_path: Path) -> None:
    _stage(tmp_path)

    result = run(_request(required=("camera-calibration",)), base=tmp_path)

    assert result.status == "unavailable"
    assert result.reason == "missing_required_capabilities: camera-calibration"
    assert "missing_required_capability:camera-calibration" in {
        item["code"] for item in result.diagnostics
    }
    assert not (tmp_path / "out").exists()


def test_known_required_camera_calibration_with_unreadable_source_is_unavailable(
    tmp_path: Path,
) -> None:
    _stage(tmp_path)
    sources = [
        {"artifact_id": "capture", "uri": "capture.json", "format": "capture-frames"},
        {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
        {
            "artifact_id": "camera",
            "uri": "missing-camera.json",
            "format": "camera-calibration",
        },
    ]

    result = run(_request(required=("camera-calibration",), sources=sources), base=tmp_path)

    assert result.status == "unavailable"
    assert result.reason == "missing_required_capabilities: camera-calibration"
    camera = next(
        source for source in result.provenance["sources"] if source["artifact_id"] == "camera"
    )
    assert camera["availability"] == "source_unreadable"


def test_required_camera_calibration_is_admitted_when_source_is_usable(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "camera.json").write_text(json.dumps({"camera_id": "cam-t"}), encoding="utf-8")
    sources = [
        {"artifact_id": "capture", "uri": "capture.json", "format": "capture-frames"},
        {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
        {"artifact_id": "camera", "uri": "camera.json", "format": "camera-calibration"},
    ]

    result = run(_request(required=("camera-calibration",), sources=sources), base=tmp_path)

    assert result.status == "complete"
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    camera = next(source for source in mapping["sources"] if source["artifact_id"] == "camera")
    assert camera["availability"] == "ok"
    assert camera["selected"] is True


def test_capture_simulation_identity_mismatch_is_unavailable_and_explicit(
    tmp_path: Path,
) -> None:
    capture = dict(CAPTURE)
    capture["frames"] = [
        {"frame_index": 0, "pts_s": 0.0, "episode_id": "capture-ep", "reset_id": "capture-reset"}
    ]
    (tmp_path / "capture.json").write_text(json.dumps(capture), encoding="utf-8")
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")

    result = run(_request(), base=tmp_path)

    assert result.status == "partial"
    assert "frame_0_capture_sim_identity_mismatch" in result.reason
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    assert mapping["entries"] == []
    assert mapping["identity_namespace"]["capture"] == "shared-episode-reset-v1"
    assert mapping["identity_namespace"]["simulation"] == "shared-episode-reset-v1"
    unavailable = mapping["unavailable_frames"][0]
    assert unavailable["reason"] == "capture_sim_identity_mismatch"
    assert unavailable["capture_episode_id"] == "capture-ep"
    assert unavailable["capture_reset_id"] == "capture-reset"
    assert "episode_id" not in unavailable
    assert "reset_id" not in unavailable


def test_source_symlink_escape_is_rejected(tmp_path: Path) -> None:
    import shutil

    _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-source-outside"
    outside.mkdir()
    try:
        (outside / "capture.json").write_text(json.dumps(CAPTURE), encoding="utf-8")
        (tmp_path / "capture-link.json").symlink_to(outside / "capture.json")
        sources = [
            {
                "artifact_id": "capture",
                "uri": "capture-link.json",
                "format": "capture-frames",
            },
            {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
        ]

        result = run(_request(sources=sources), base=tmp_path)

        assert result.status == "failed"
        assert "capture: source_uri_rejected" in result.reason
        assert not (tmp_path / "out").exists()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_missing_required_source_family_is_failed_closed(tmp_path: Path) -> None:
    (tmp_path / "stamps.json").write_text(json.dumps(STAMPS), encoding="utf-8")
    sources = [
        {"artifact_id": "capture", "uri": "missing-capture.json", "format": "capture-frames"},
        {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
    ]

    result = run(_request(sources=sources), base=tmp_path)

    assert result.status == "failed"
    assert "required_source_family_missing: capture-frames" in result.reason
    assert "capture: source_unreadable" in result.reason
    assert not (tmp_path / "out").exists()


def test_output_symlink_escape_is_rejected(tmp_path: Path) -> None:
    import shutil

    _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-output-outside"
    outside.mkdir()
    try:
        (tmp_path / "output-link").symlink_to(outside, target_is_directory=True)

        result = run(_request(output="output-link/escaped"), base=tmp_path)

        assert result.status == "failed"
        assert "unsafe_output_path" in result.reason
        assert not (outside / "escaped").exists()
    finally:
        shutil.rmtree(outside, ignore_errors=True)


def test_absolute_paths_are_rejected_by_shared_request_contract(tmp_path: Path) -> None:
    with pytest.raises(ReviewContractsValidationError, match="traversal"):
        _request(output=str(tmp_path / "out"))
    with pytest.raises(ReviewContractsValidationError, match="traversal"):
        _request(
            sources=[
                {
                    "artifact_id": "capture",
                    "uri": str(tmp_path / "capture.json"),
                    "format": "capture-frames",
                },
                {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
            ]
        )


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(config_extra={"min_component_version": "2.0.0"}), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_dangling_output_symlink_is_a_collision(tmp_path: Path) -> None:
    """A pre-existing output entry, even dangling, cannot be replaced."""
    _stage(tmp_path)
    (tmp_path / "out").symlink_to(tmp_path / "not-created")

    result = run(_request(), base=tmp_path)

    assert result.status == "failed"
    assert "output_collision" in result.reason
    assert (tmp_path / "out").is_symlink()


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_declares_capabilities() -> None:
    info = descriptor()
    assert info["schema_version"] == "component-descriptor.v1"
    assert info["component_id"] == COMPONENT_ID
    assert set(info["required_capabilities"]) == {"capture-frames", "sim-stamps"}
    assert "camera-calibration" in info["optional_capabilities"]
    assert component_descriptor_from_dict(info).component_id == COMPONENT_ID


def test_result_document_is_json_safe_and_shared_contract_valid() -> None:
    payload = _result_document(
        ComponentResult(
            request_id="schema-check",
            component_id=COMPONENT_ID,
            status="failed",
            diagnostics=({"code": "diagnostic"},),
        )
    )

    assert isinstance(payload["diagnostics"], list)
    assert component_result_from_dict(payload).status == "failed"


def test_source_identity_digest_integrity_and_admission_are_retained(tmp_path: Path) -> None:
    _stage(tmp_path)
    metadata = {
        "capture": {
            "schema": "capture-frames.fixture.v1",
            "source_commit": "a" * 40,
            "config_identity": "capture-config-1",
            "sha256": hashlib.sha256((tmp_path / "capture.json").read_bytes()).hexdigest(),
        },
        "stamps": {
            "schema": "sim-stamps.fixture.v1",
            "source_commit": "b" * 40,
            "config_identity": "stamps-config-1",
            "sha256": hashlib.sha256((tmp_path / "stamps.json").read_bytes()).hexdigest(),
        },
    }
    result = run(_request(config_extra={"source_metadata": metadata}), base=tmp_path)

    assert result.status == "complete"
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    capture = next(source for source in mapping["sources"] if source["artifact_id"] == "capture")
    assert capture["uri"] == "capture.json"
    assert capture["format"] == "capture-frames"
    assert capture["schema"] == "capture-frames.fixture.v1"
    assert capture["source_commit"] == "a" * 40
    assert capture["config_identity"] == "capture-config-1"
    assert capture["source_sha256"] == metadata["capture"]["sha256"]
    assert capture["integrity"] == capture["integrity_status"] == "match"
    assert capture["admission"] == "not_evaluated"
    assert mapping["provenance"]["admission"] == "not_evaluated"
    assert result.provenance["source_integrity"] == {"capture": "match", "stamps": "match"}
    assert result.provenance["admission"] == "not_evaluated"


def test_source_metadata_conflict_with_source_ref_fails_closed(tmp_path: Path) -> None:
    """Config provenance cannot override a declaration carried by the source ref."""
    _stage(tmp_path)
    sources = [
        {
            "artifact_id": "capture",
            "uri": "capture.json",
            "format": "capture-frames",
            "schema": "ref-schema",
        },
        {"artifact_id": "stamps", "uri": "stamps.json", "format": "sim-stamps"},
    ]

    result = run(
        _request(
            sources=sources,
            config_extra={"source_metadata": {"capture": {"schema": "config-schema"}}},
        ),
        base=tmp_path,
    )

    assert result.status == "failed"
    assert "source_metadata.capture.schema conflicts" in result.reason
    assert not (tmp_path / "out").exists()


def test_required_component_version_alias_is_admitted_or_rejected_deterministically(
    tmp_path: Path,
) -> None:
    """The version admission alias uses the same strict comparison as the legacy field."""
    _stage(tmp_path)

    result = run(_request(config_extra={"required_component_version": "2.0.0"}), base=tmp_path)

    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason
    assert not (tmp_path / "out").exists()


def test_source_digest_mismatch_is_diagnostic_partial(tmp_path: Path) -> None:
    _stage(tmp_path)
    metadata = {
        "capture": {"sha256": "0" * 64},
        "stamps": {"sha256": hashlib.sha256((tmp_path / "stamps.json").read_bytes()).hexdigest()},
    }

    result = run(_request(config_extra={"source_metadata": metadata}), base=tmp_path)

    assert result.status == "partial"
    assert "capture: stale_digest" in result.reason
    mapping = json.loads((tmp_path / "out" / "media-mapping.json").read_text())
    capture = next(source for source in mapping["sources"] if source["artifact_id"] == "capture")
    assert capture["integrity"] == "mismatch"
    assert result.artifacts == ()


def test_module_touches_no_simulator_paths() -> None:
    """The adapter must stay observational: no simulator/planner imports."""
    tree = ast.parse(
        (Path(__file__).resolve().parents[2] / "robot_sf/render/video_sync.py").read_bytes()
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    imported -= {"robot_sf", "__future__", "typing"}
    assert not any(name in {"sim", "planner", "training"} for name in imported), imported
    assert "torch" not in imported


def test_cli_produces_mapping_from_fixture_request(tmp_path: Path) -> None:
    import shutil

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / FIXTURE_DIR
    assert (fixture / "request.json").exists()
    output_rel = "output/scenario_review/srev-03-cli-test"
    shutil.rmtree(repo / output_rel, ignore_errors=True)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_sf.render.video_sync",
                "--input",
                str(fixture / "request.json"),
                "--config",
                str(fixture / "config.json"),
                "--output",
                output_rel,
            ],
            capture_output=True,
            text=True,
            cwd=repo,
            check=False,
        )
        # The gap-exercising fixture yields partial with explicit codes.
        assert completed.returncode == 1, completed.stderr[-2000:]
        payload = json.loads(completed.stdout)
        assert payload["schema_version"] == "component-result.v1"
        assert component_result_from_dict(payload).status == "partial"
        assert payload["status"] == "partial"
        assert "skipped_frames:3" in payload["reason"]
        assert "nonuniform_sampling" in payload["reason"]
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)


def test_cli_descriptor_is_shared_contract_valid() -> None:
    repo = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-m", "robot_sf.render.video_sync", "--descriptor"],
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert component_descriptor_from_dict(json.loads(completed.stdout)).component_id == COMPONENT_ID


def test_cli_malformed_strict_json_returns_failed_result(tmp_path: Path) -> None:
    input_path = tmp_path / "request.json"
    input_path.write_text(
        '{"request_id":"bad", "component_id":"srev03-video-sync", NaN}', encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.render.video_sync",
            "--input",
            str(input_path),
            "--output",
            "out",
            "--base",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert component_result_from_dict(result).status == "failed"
    assert result["reason"] == "invalid_input: request JSON cannot be parsed safely"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX mkfifo")
def test_cli_special_input_file_returns_failed_result_without_blocking(tmp_path: Path) -> None:
    """The CLI rejects a FIFO before attempting a potentially blocking read."""
    input_path = tmp_path / "request.fifo"
    os.mkfifo(input_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.render.video_sync",
            "--input",
            str(input_path),
            "--output",
            "out",
            "--base",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        timeout=5,
    )

    assert completed.returncode == 1
    result = json.loads(completed.stdout)
    assert component_result_from_dict(result).status == "failed"
    assert result["reason"] == "invalid_input: request JSON cannot be parsed safely"


def test_run_malformed_request_returns_failed_result() -> None:
    result = run({"request_id": "bad"})  # type: ignore[arg-type]

    assert result.status == "failed"
    assert result.reason == "invalid_request: expected ComponentRequest"
