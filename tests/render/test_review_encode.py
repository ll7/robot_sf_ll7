"""Focused tests for the SREV-10 review-encode leaf (issue #9279)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import component_request_from_dict
from robot_sf.render import review_encode
from robot_sf.render.review_encode import _render_frame, descriptor, run

FIXTURES = Path("tests/fixtures/scenario_review/review_encode")

EXPECTED_ORDER = [0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 20, 22, 24, 26, 28]


def _request(tmp_path: Path, config_extra: dict | None = None, caps: list[str] | None = None):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source_fps": 10, "source_frames": 30}), encoding="utf-8")
    payload = {
        "schema_version": "component-request.v1",
        "request_id": "t10",
        "component_id": "srev10-review-encode",
        "sources": [
            {
                "artifact_id": "frames",
                "uri": "manifest.json",
                "format": "frame-sequence-manifest.v1",
            }
        ],
        "output_directory": "out",
        "config": config_extra or {},
        "required_capabilities": caps if caps is not None else ["frame-sequence"],
    }
    return component_request_from_dict(payload)


def _time_map(out: Path) -> dict:
    return json.loads((out / "time-map.json").read_text(encoding="utf-8"))


def test_descriptor_declares_capabilities() -> None:
    desc = descriptor()
    assert desc["component_id"] == "srev10-review-encode"
    assert "frame-sequence" in desc["required_capabilities"]
    assert "review-edit-mp4.v1" in desc["output_types"]


def test_fixture_plan_produces_expected_frame_order(tmp_path: Path) -> None:
    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete", result.reason
    assert _time_map(tmp_path / "out")["frame_order_source_indices"] == EXPECTED_ORDER


def test_interval_endpoints_and_first_terminal_frames(tmp_path: Path) -> None:
    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete"
    mapping = _time_map(tmp_path / "out")
    assert mapping["first_source_frame"] == 0
    assert mapping["terminal_source_frame"] == 28
    assert mapping["segments"][0]["presentation_start_s"] == 0.0
    assert mapping["segments"][-1]["presentation_end_s"] == pytest.approx(2.0)


def test_encoded_video_decodes_to_planned_pixels(tmp_path: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np

    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete"
    with imageio.get_reader(str(tmp_path / "out" / "edit.mp4")) as reader:
        decoded = [np.asarray(frame) for frame in reader]
    assert len(decoded) == len(EXPECTED_ORDER)
    for position in (0, 10, 15, 19):
        expected = np.asarray(_render_frame(EXPECTED_ORDER[position], 160, 120))
        assert decoded[position].shape == expected.shape
        assert abs(decoded[position].astype(int) - expected.astype(int)).mean() < 12


def test_pause_repeats_anchor_frame(tmp_path: Path) -> None:
    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete"
    order = _time_map(tmp_path / "out")["frame_order_source_indices"]
    assert order.count(5) == 6


def test_speed_resamples_kept_span(tmp_path: Path) -> None:
    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    result = run(_request(tmp_path, config_extra=config), base=tmp_path)
    assert result.status == "complete"
    order = _time_map(tmp_path / "out")["frame_order_source_indices"]
    assert order[-5:] == [20, 22, 24, 26, 28]


def test_crop_clamp_is_partial(tmp_path: Path) -> None:
    edits = [{"op": "crop", "x": 100, "y": 100, "width": 200, "height": 200}]
    result = run(
        _request(
            tmp_path, config_extra={"edits": edits, "preset": dict(review_encode.TINY_PRESET)}
        ),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "crop_clamped" in result.reason
    assert result.artifacts == ()


def test_conflicting_speeds_fail(tmp_path: Path) -> None:
    edits = [
        {"op": "speed", "start_s": 0.0, "end_s": 2.0, "factor": 2.0},
        {"op": "speed", "start_s": 1.0, "end_s": 3.0, "factor": 0.5},
    ]
    result = run(_request(tmp_path, config_extra={"edits": edits}), base=tmp_path)
    assert result.status == "failed"
    assert "conflicting_time_map" in result.reason


def test_cut_everything_fails(tmp_path: Path) -> None:
    edits = [{"op": "cut", "start_s": 0.0, "end_s": 3.0}]
    result = run(_request(tmp_path, config_extra={"edits": edits}), base=tmp_path)
    assert result.status == "failed"
    assert "empty_timeline" in result.reason


def test_corrupt_manifest_fails(tmp_path: Path) -> None:
    request = _request(tmp_path)
    (tmp_path / "manifest.json").write_text("{nope", encoding="utf-8")
    result = run(request, base=tmp_path)
    assert result.status == "failed"


def test_missing_family_is_unavailable(tmp_path: Path) -> None:
    payload = {
        "schema_version": "component-request.v1",
        "request_id": "t10",
        "component_id": "srev10-review-encode",
        "sources": [
            {
                "artifact_id": "scores",
                "uri": "manifest.json",
                "format": "exemplar-scores",
            }
        ],
        "output_directory": "out",
        "config": {},
        "required_capabilities": ["frame-sequence"],
    }
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "unavailable"


def test_unsupported_capability_is_unavailable(tmp_path: Path) -> None:
    result = run(_request(tmp_path, caps=["telemetry-xyz"]), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_required_capability" in result.reason


def test_incompatible_version_fails(tmp_path: Path) -> None:
    result = run(_request(tmp_path, config_extra={"min_component_version": "9.0.0"}), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "sentinel.txt").write_text("x", encoding="utf-8")
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_non_silent_audio_fails(tmp_path: Path) -> None:
    result = run(_request(tmp_path, config_extra={"audio_policy": "passthrough"}), base=tmp_path)
    assert result.status == "failed"
    assert "unsupported_audio_policy" in result.reason


def test_deterministic_repeat_runs_match_bytes(tmp_path: Path) -> None:
    import imageio.v2 as imageio
    import numpy as np

    first = tmp_path / "a"
    second = tmp_path / "b"
    for target in (first, second):
        manifest = target / "manifest.json"
        target.mkdir(parents=True)
        manifest.write_text(json.dumps({"source_fps": 10, "source_frames": 30}), encoding="utf-8")
        payload = {
            "schema_version": "component-request.v1",
            "request_id": "t10",
            "component_id": "srev10-review-encode",
            "sources": [
                {
                    "artifact_id": "frames",
                    "uri": "manifest.json",
                    "format": "frame-sequence-manifest.v1",
                }
            ],
            "output_directory": "out",
            "config": json.loads((FIXTURES / "config.json").read_text(encoding="utf-8")),
            "required_capabilities": ["frame-sequence"],
        }
        result = run(component_request_from_dict(payload), base=target)
        assert result.status == "complete"
    assert (first / "out" / "time-map.json").read_bytes() == (
        second / "out" / "time-map.json"
    ).read_bytes()
    assert (first / "out" / "encode-receipt.json").read_bytes() == (
        second / "out" / "encode-receipt.json"
    ).read_bytes()
    with imageio.get_reader(str(first / "out" / "edit.mp4")) as reader_a:
        pixels_a = [np.asarray(frame) for frame in reader_a]
    with imageio.get_reader(str(second / "out" / "edit.mp4")) as reader_b:
        pixels_b = [np.asarray(frame) for frame in reader_b]
    assert len(pixels_a) == len(pixels_b) == len(EXPECTED_ORDER)
    for left, right in zip(pixels_a, pixels_b, strict=True):
        assert (left == right).all()


def test_cli_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "cli-out"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.render.review_encode",
            "--input",
            str((FIXTURES / "request.json").resolve()),
            "--config",
            str((FIXTURES / "config.json").resolve()),
            "--output",
            str(out),
            "--base",
            ".",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    envelope = json.loads(completed.stdout)
    assert envelope["status"] == "complete"
    assert (out / "edit.mp4").is_file()
    assert (out / "encode-receipt.json").is_file()


def test_module_has_no_simulation_imports() -> None:
    tree = ast.parse(
        (Path(__file__).resolve().parents[2] / "robot_sf/render/review_encode.py").read_bytes()
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(part.name for part in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not {name for name in imported if "robot_sf.sim" in name}
