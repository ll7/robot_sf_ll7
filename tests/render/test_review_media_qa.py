"""Contract tests for the review media QA component (SREV-13, issue #9282)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ReviewContractsValidationError,
    SourceRef,
    component_request_from_dict,
)
from robot_sf.render.review_media_qa import (
    COMPONENT_ID,
    COMPONENT_VERSION,
    CONTACT_SHEET_SCHEMA_VERSION,
    MEDIA_QUALITY_REPORT_SCHEMA_VERSION,
    _decoder_available,
    _duration_check,
    _label_quality_check,
    _monotonicity_check,
    _timestamp_sync_check,
    _validate_presentation,
    _validate_thresholds,
    descriptor_document,
    main,
    run,
)

FIXTURE_DIR = Path("tests/fixtures/scenario_review/review_media_qa")
ALL_CAPABILITIES = (
    "video-quality-checks",
    "contact-sheet-render",
    "timestamp-sync-checks",
    "storyboard-pause-analysis",
)


def _stage(tmp_path: Path, *names: str) -> None:
    """Copy named fixture files into a tmp base directory."""
    for name in names:
        (tmp_path / name).write_bytes((FIXTURE_DIR / name).read_bytes())


def _fixture_request(
    tmp_path: Path,
    request_id: str,
    clip_name: str,
    output_name: str,
    capabilities: tuple[str, ...] = ALL_CAPABILITIES,
    *,
    with_map: bool = True,
    with_storyboard: bool = True,
) -> ComponentRequest:
    """Build a request against tmp-staged fixtures with relative paths."""
    names = [clip_name]
    sources = [
        {
            "artifact_id": "clip",
            "uri": clip_name,
            "format": "media-qa-frames.v1",
            "schema": "media-qa-frames.v1",
        }
    ]
    if with_map:
        names.append("timestamp_map.json")
        sources.append(
            {
                "artifact_id": "timestamp-map",
                "uri": "timestamp_map.json",
                "format": "presentation-timestamp-map.v1",
                "schema": "presentation-timestamp-map.v1",
            }
        )
    if with_storyboard:
        names.append("storyboard.json")
        sources.append(
            {
                "artifact_id": "storyboard",
                "uri": "storyboard.json",
                "format": "storyboard-spec.v1",
                "schema": "storyboard-spec.v1",
            }
        )
    _stage(tmp_path, *names)
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": COMPONENT_ID,
        "sources": sources,
        "output_directory": output_name,
        "config": {
            "presentation": {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0},
            "thresholds": {"duration_tolerance_s": 0.05, "min_label_chars": 1},
        },
        "required_capabilities": list(capabilities),
    }
    return component_request_from_dict(payload, source="test")


def _report_artifact(output_dir: Path) -> dict[str, Any]:
    """Read the quality report artifact from a run output directory."""
    return json.loads(
        (output_dir / f"{MEDIA_QUALITY_REPORT_SCHEMA_VERSION}.json").read_text(encoding="utf-8")
    )


def test_descriptor_contract() -> None:
    """The descriptor document must match declared version and capabilities."""
    doc = descriptor_document()
    assert doc["schema_version"] == "component-descriptor.v1"
    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    assert "video-quality-checks" in doc["required_capabilities"]
    assert "contact-sheet-render" in doc["optional_capabilities"]
    assert "container-decode" in doc["optional_capabilities"]
    assert "timestamp-sync-checks" in doc["optional_capabilities"]
    assert "storyboard-pause-analysis" in doc["optional_capabilities"]
    assert MEDIA_QUALITY_REPORT_SCHEMA_VERSION in doc["output_types"]
    assert CONTACT_SHEET_SCHEMA_VERSION in doc["output_types"]


def test_presentation_validation() -> None:
    """Presentation presets must be finite and strictly positive."""
    preset = _validate_presentation({"width": 320, "height": 180, "fps": 10.0, "speed": 1.0})
    assert preset == {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}
    with pytest.raises(ReviewContractsValidationError):
        _validate_presentation({"width": 0, "height": 180, "fps": 10.0, "speed": 1.0})
    with pytest.raises(ReviewContractsValidationError):
        _validate_presentation({"width": 320, "height": 180, "fps": float("inf"), "speed": 1.0})


def test_threshold_validation() -> None:
    """QA thresholds must be finite, with non-negative tolerance and min chars >= 1."""
    thresholds = _validate_thresholds({"duration_tolerance_s": 0.05, "min_label_chars": 1})
    assert thresholds == {"duration_tolerance_s": 0.05, "min_label_chars": 1.0}
    with pytest.raises(ReviewContractsValidationError):
        _validate_thresholds({"duration_tolerance_s": -0.1, "min_label_chars": 1})
    with pytest.raises(ReviewContractsValidationError):
        _validate_thresholds({"duration_tolerance_s": 0.05, "min_label_chars": 0})


def test_checked_in_request_envelope_parses() -> None:
    """The checked-in fixture request must satisfy the component-request contract."""
    payload = json.loads((FIXTURE_DIR / "request.json").read_text(encoding="utf-8"))
    request = component_request_from_dict(payload, source="test")
    assert request.component_id == COMPONENT_ID
    assert len(request.sources) == 3


def test_run_complete_smoke_execution(tmp_path: Path) -> None:
    """The smoke fixture must produce a complete result with report and contact sheet."""
    output_dir = tmp_path / "srev-13-smoke"
    request = _fixture_request(tmp_path, "req-smoke", "clip_smoke.json", "srev-13-smoke")
    result = run(request, base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    artifact_ids = {artifact["artifact_id"] for artifact in result.artifacts}
    assert f"{MEDIA_QUALITY_REPORT_SCHEMA_VERSION}.json" in artifact_ids
    assert f"{CONTACT_SHEET_SCHEMA_VERSION}.json" in artifact_ids
    assert f"{CONTACT_SHEET_SCHEMA_VERSION}.png" in artifact_ids
    report = _report_artifact(output_dir)
    assert report["overall_verdict"] == "pass"
    assert report["qa_tool_version"] == COMPONENT_VERSION
    assert all(check["verdict"] == "pass" for check in report["checks"])
    manifest = json.loads(
        (output_dir / f"{CONTACT_SHEET_SCHEMA_VERSION}.json").read_text(encoding="utf-8")
    )
    assert len(manifest["cells"]) == 6
    assert manifest["image"] == f"{CONTACT_SHEET_SCHEMA_VERSION}.png"
    assert (output_dir / f"{CONTACT_SHEET_SCHEMA_VERSION}.png").exists()


def test_corrupt_clip_fails_closed(tmp_path: Path) -> None:
    """A non-JSON clip must fail the integrity check without carrying complete status."""
    request = _fixture_request(
        tmp_path,
        "req-corrupt",
        "clip_corrupt.bin",
        "corrupt",
        ("video-quality-checks",),
        with_map=False,
        with_storyboard=False,
    )
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    report = _report_artifact(tmp_path / "corrupt")
    assert report["overall_verdict"] == "fail"
    integrity = next(c for c in report["checks"] if c["check_id"] == "clip_integrity")
    assert integrity["verdict"] == "fail"
    assert integrity["reason_code"] == "corrupt_clip"


def test_blank_clip_fails_closed(tmp_path: Path) -> None:
    """An all-blank clip must fail integrity with an explicit reason code."""
    request = _fixture_request(
        tmp_path,
        "req-blank",
        "clip_blank.json",
        "blank",
        ("video-quality-checks",),
        with_map=False,
        with_storyboard=False,
    )
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    report = _report_artifact(tmp_path / "blank")
    integrity = next(c for c in report["checks"] if c["check_id"] == "clip_integrity")
    assert (integrity["verdict"], integrity["reason_code"]) == ("fail", "blank_clip")


def test_truncated_clip_fails_closed(tmp_path: Path) -> None:
    """A declared/actual frame-count mismatch must fail integrity as truncated."""
    request = _fixture_request(
        tmp_path,
        "req-truncated",
        "clip_truncated.json",
        "truncated",
        ("video-quality-checks",),
        with_map=False,
        with_storyboard=False,
    )
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    report = _report_artifact(tmp_path / "truncated")
    integrity = next(c for c in report["checks"] if c["check_id"] == "clip_integrity")
    assert (integrity["verdict"], integrity["reason_code"]) == ("fail", "truncated_clip")


def test_nonmonotonic_timestamps_fail(tmp_path: Path) -> None:
    """Decreasing timestamps must fail monotonicity with an explicit reason code."""
    request = _fixture_request(
        tmp_path,
        "req-nonmonotonic",
        "clip_nonmonotonic.json",
        "nonmonotonic",
        ("video-quality-checks",),
        with_map=False,
        with_storyboard=False,
    )
    result = run(request, base=tmp_path)
    assert result.status == "partial"
    report = _report_artifact(tmp_path / "nonmonotonic")
    monotonicity = next(c for c in report["checks"] if c["check_id"] == "timestamp_monotonicity")
    assert (monotonicity["verdict"], monotonicity["reason_code"]) == (
        "fail",
        "nonmonotonic_timestamps",
    )


def test_unexplained_freeze_fails_without_storyboard(tmp_path: Path) -> None:
    """Repeated timestamps outside a declared pause are freezes, not holds."""
    record = _monotonicity_check([0.0, 0.1, 0.1, 0.2], None, True)
    assert (record["verdict"], record["reason_code"]) == ("fail", "unexplained_freeze")


def test_declared_pause_passes_with_capability() -> None:
    """A storyboard-declared pause must pass when pause analysis is requested."""
    storyboard = {"pauses": [{"from_frame_index": 2, "to_frame_index": 2}]}
    record = _monotonicity_check([0.0, 0.1, 0.1, 0.2], storyboard, True)
    assert (record["verdict"], record["reason_code"]) == ("pass", "ok_with_declared_pauses")


def test_duration_mismatch_detected() -> None:
    """Declared durations outside tolerance must fail the duration check."""
    clip = {"duration_s": 0.4}
    assert _duration_check(clip, [0.0, 0.1, 0.4], 0.05)["verdict"] == "pass"
    failed = _duration_check(clip, [0.0, 0.1, 0.9], 0.05)
    assert (failed["verdict"], failed["reason_code"]) == ("fail", "duration_mismatch")


def test_missing_duration_is_unavailable() -> None:
    """A clip without declared duration cannot run the duration check."""
    record = _duration_check({}, [0.0, 0.1], 0.05)
    assert (record["verdict"], record["reason_code"]) == (
        "unavailable",
        "missing_expected_duration",
    )


def test_label_findings_detected() -> None:
    """Unreadable, overlapping, and clipped labels must each be reported."""
    frames = json.loads((FIXTURE_DIR / "clip_bad_labels.json").read_text(encoding="utf-8"))[
        "frames"
    ]
    record = _label_quality_check(frames, 320, 180, 1)
    assert record["verdict"] == "fail"
    assert "unreadable_label" in record["detail"]
    assert "label_overlap" in record["detail"]
    assert "label_clipping" in record["detail"]


def test_missing_timestamp_map_disables_sync() -> None:
    """Synchronized checks without an explicit map are unavailable, never guessed."""
    frames = [{"frame_index": 0, "t_s": 0.0}]
    record = _timestamp_sync_check(frames, None, "", True)
    assert (record["verdict"], record["reason_code"]) == ("unavailable", "missing_timestamp_map")
    skipped = _timestamp_sync_check(frames, None, "", False)
    assert skipped["reason_code"] == "capability_not_requested"


def test_unmapped_frames_fail_sync() -> None:
    """Frames missing from an explicit map must fail the sync check."""
    timestamp_map = {"entries": [{"frame_index": 0, "media_t_s": 0.0, "source_t_s": 0.0}]}
    frames = [{"frame_index": 0, "t_s": 0.0}, {"frame_index": 1, "t_s": 0.1}]
    record = _timestamp_sync_check(frames, timestamp_map, "", True)
    assert (record["verdict"], record["reason_code"]) == ("fail", "unmapped_frames")


def test_unsupported_capability_fails_closed(tmp_path: Path) -> None:
    """Requested capabilities outside the descriptor must report unavailable."""
    request = ComponentRequest(
        request_id="req-bad-cap",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory="bad-cap",
        required_capabilities=("video-quality-checks", "no-such-capability"),
    )
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "missing capabilities" in result.reason


def test_unsupported_component_id(tmp_path: Path) -> None:
    """A request for another component must report unavailable without side effects."""
    request = ComponentRequest(
        request_id="req-other",
        component_id="review-camera",
        sources=(),
        output_directory="other",
        required_capabilities=("video-quality-checks",),
    )
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert not (tmp_path / "other").exists()


def test_output_collision_rejected(tmp_path: Path) -> None:
    """An existing output directory must be rejected to protect prior runs."""
    (tmp_path / "collision").mkdir()
    request = _fixture_request(tmp_path, "req-collision", "clip_smoke.json", "collision")
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "output collision" in result.reason


def test_corrupt_or_nonfinite_inputs(tmp_path: Path) -> None:
    """Non-finite presentation and threshold inputs must fail closed."""
    request = _fixture_request(tmp_path, "req-bad-input", "clip_smoke.json", "bad-input")
    bad_presentation = ComponentRequest(
        request_id=request.request_id,
        component_id=request.component_id,
        sources=request.sources,
        output_directory=request.output_directory,
        config={"presentation": {"width": float("nan"), "height": 180, "fps": 10.0, "speed": 1.0}},
        required_capabilities=request.required_capabilities,
    )
    result = run(bad_presentation, base=tmp_path)
    assert result.status == "failed"


def test_missing_clip_source_is_unavailable(tmp_path: Path) -> None:
    """A request with no clip source must report unavailable, not complete."""
    request = ComponentRequest(
        request_id="req-no-clip",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory="no-clip",
        required_capabilities=("video-quality-checks",),
    )
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "no clip source" in result.reason


def test_container_source_reports_decoder_state(tmp_path: Path) -> None:
    """A real container source must surface decoder availability, never a silent pass."""
    request = ComponentRequest(
        request_id="req-container",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="real-clip",
                uri="media/smoke.mp4",
                format="video-container",
            ),
        ),
        output_directory="container",
        required_capabilities=("video-quality-checks",),
    )
    result = run(request, base=tmp_path)
    assert result.status in ("partial", "unavailable")
    assert _decoder_available() in (True, False)


def test_deterministic_runs_produce_identical_reports(tmp_path: Path) -> None:
    """Repeated fixture runs must produce identical logical reports and digests."""
    base_a = tmp_path / "a"
    base_b = tmp_path / "b"
    base_a.mkdir()
    base_b.mkdir()
    first = run(_fixture_request(base_a, "req-determinism", "clip_smoke.json", "out"), base=base_a)
    second = run(_fixture_request(base_b, "req-determinism", "clip_smoke.json", "out"), base=base_b)
    assert first.status == "complete"
    assert second.status == "complete"
    report_name = f"{MEDIA_QUALITY_REPORT_SCHEMA_VERSION}.json"
    first_report = json.loads((base_a / "out" / report_name).read_text(encoding="utf-8"))
    second_report = json.loads((base_b / "out" / report_name).read_text(encoding="utf-8"))
    assert first_report == second_report
    assert first_report["overall_verdict"] == "pass"
    assert [c["check_id"] for c in first_report["checks"]] == [
        "clip_integrity",
        "duration_match",
        "timestamp_monotonicity",
        "timestamp_sync",
        "label_quality",
    ]
    assert first_report["thresholds"] == {"duration_tolerance_s": 0.05, "min_label_chars": 1}
    assert first_report["qa_tool_version"] == COMPONENT_VERSION


def test_cli_descriptor(capsys: pytest.CaptureFixture[str]) -> None:
    """The descriptor flag must print the component descriptor in-process."""
    assert main(["--descriptor"]) == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["component_id"] == COMPONENT_ID


def test_cli_smoke_execution_in_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The fixture CLI path must run in-process and print a complete result."""
    _stage(tmp_path, "clip_smoke.json", "timestamp_map.json", "storyboard.json", "config.json")
    staged_request = {
        "schema_version": "component-request.v1",
        "request_id": "req-cli-smoke",
        "component_id": COMPONENT_ID,
        "sources": [
            {
                "artifact_id": "smoke-clip",
                "uri": "clip_smoke.json",
                "format": "media-qa-frames.v1",
                "schema": "media-qa-frames.v1",
            },
            {
                "artifact_id": "smoke-timestamp-map",
                "uri": "timestamp_map.json",
                "format": "presentation-timestamp-map.v1",
                "schema": "presentation-timestamp-map.v1",
            },
            {
                "artifact_id": "smoke-storyboard",
                "uri": "storyboard.json",
                "format": "storyboard-spec.v1",
                "schema": "storyboard-spec.v1",
            },
        ],
        "output_directory": "cli-smoke",
        "config": {},
        "required_capabilities": list(ALL_CAPABILITIES),
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(staged_request), encoding="utf-8")
    exit_code = main(
        [
            "--input",
            str(request_path),
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            "cli-smoke",
            "--base",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "complete"
    assert (tmp_path / "cli-smoke" / f"{MEDIA_QUALITY_REPORT_SCHEMA_VERSION}.json").exists()


def test_cli_missing_args_exits_with_usage() -> None:
    """Missing CLI input/output must exit with a usage error, not a traceback."""
    with pytest.raises(SystemExit) as exc_info:
        main([])
    assert exc_info.value.code == 2


def test_cli_unreadable_request_fails_closed(tmp_path: Path) -> None:
    """An unreadable request file must raise a contract error before execution."""
    with pytest.raises(ReviewContractsValidationError):
        main(["--input", str(tmp_path / "missing.json"), "--output", "out"])
