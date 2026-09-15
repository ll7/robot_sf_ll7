"""Focused tests for the SREV-03 video-sync component (issue #9272)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from robot_sf.render.video_sync import (
    COMPONENT_ID,
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


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_declares_capabilities() -> None:
    info = descriptor()
    assert info["component_id"] == COMPONENT_ID
    assert set(info["required_capabilities"]) == {"capture-frames", "sim-stamps"}
    assert "camera-calibration" in info["optional_capabilities"]


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
        assert payload["status"] == "partial"
        assert "skipped_frames:3" in payload["reason"]
        assert "nonuniform_sampling" in payload["reason"]
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)
