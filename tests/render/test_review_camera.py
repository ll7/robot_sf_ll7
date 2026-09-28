"""Contract tests for the review camera and presentation layout component (SREV-11, issue #9280)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ReviewContractsValidationError,
    SourceRef,
)
from robot_sf.render.review_camera import (
    CAMERA_LAYOUT_SPEC_SCHEMA_VERSION,
    COMPONENT_ID,
    COMPONENT_VERSION,
    VISUALIZATION_SPEC_SCHEMA_VERSION,
    _validate_aspect_ratio,
    compute_event_camera,
    compute_follow_camera_track,
    compute_full_scene_camera,
    descriptor_document,
    main,
    run,
)

FIXTURE_DIR = Path("tests/fixtures/scenario_review/review_camera")


def _sample_source_scene() -> dict:
    return {
        "schema_version": "threejs-viewer.v1",
        "map": {
            "width": 10.0,
            "height": 8.0,
            "obstacles": [{"vertices": [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]}],
        },
        "frames": [
            {"timestep": 0, "t_s": 0.0, "robot": {"position": [1.0, 1.5]}},
            {"timestep": 1, "t_s": 0.1, "robot": {"position": [2.0, 2.0]}},
            {"timestep": 2, "t_s": 0.2, "robot": {"position": [3.5, 3.0]}},
            {"timestep": 3, "t_s": 0.3, "robot": {"position": [5.0, 4.0]}},
            {"timestep": 4, "t_s": 0.4, "robot": {"position": [7.0, 5.5]}},
        ],
        "events": [
            {"event_id": "evt-001", "type": "checkpoint", "t_s": 0.2, "position": [3.5, 3.0]}
        ],
    }


def test_descriptor_contract() -> None:
    """The descriptor document must match declared version and capabilities."""
    doc = descriptor_document()
    assert doc["schema_version"] == "component-descriptor.v1"
    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    assert "full-scene-camera" in doc["required_capabilities"]
    assert "camera-interpolation" in doc["required_capabilities"]
    assert "follow-camera" in doc["optional_capabilities"]
    assert "event-camera" in doc["optional_capabilities"]
    assert "inset-view" in doc["optional_capabilities"]
    assert CAMERA_LAYOUT_SPEC_SCHEMA_VERSION in doc["output_types"]
    assert VISUALIZATION_SPEC_SCHEMA_VERSION in doc["output_types"]


def test_aspect_ratio_validation() -> None:
    """Aspect ratio must be strictly positive and finite."""
    assert pytest.approx(_validate_aspect_ratio(1920, 1080), rel=1e-4) == 16.0 / 9.0
    assert pytest.approx(_validate_aspect_ratio(320, 180), rel=1e-4) == 16.0 / 9.0
    assert pytest.approx(_validate_aspect_ratio(100, 100), rel=1e-4) == 1.0

    with pytest.raises(ReviewContractsValidationError, match="finite and > 0"):
        _validate_aspect_ratio(0, 100)

    with pytest.raises(ReviewContractsValidationError, match="finite and > 0"):
        _validate_aspect_ratio(-1920, 1080)

    with pytest.raises(ReviewContractsValidationError, match="finite and > 0"):
        _validate_aspect_ratio(float("nan"), 100)

    with pytest.raises(ReviewContractsValidationError, match="finite and > 0"):
        _validate_aspect_ratio(100, float("inf"))


def test_compute_full_scene_camera() -> None:
    """Full scene camera centers on geometry and fits aspect ratio with margins."""
    bounds = (0.0, 10.0, 0.0, 8.0)
    aspect_ratio = 16.0 / 9.0
    cam = compute_full_scene_camera(bounds, aspect_ratio, margin_m=1.0)

    assert cam["kind"] == "orthographic"
    assert cam["center"] == [5.0, 4.0]
    assert cam["is_cropped"] is False

    ext_w, ext_h = cam["extents_m"]
    assert ext_w / ext_h == pytest.approx(aspect_ratio, rel=1e-4)
    # With 1m margin, content is 12m wide, 10m high.
    # To fit 16:9 aspect ratio, width must be at least 10 * (16/9) = 17.777...
    assert ext_h == 10.0
    assert pytest.approx(ext_w, rel=1e-4) == 17.7778
    assert cam["view_bounds"]["min_x"] <= 0.0
    assert cam["view_bounds"]["max_x"] >= 10.0
    assert cam["view_bounds"]["min_y"] <= 0.0
    assert cam["view_bounds"]["max_y"] >= 8.0


def test_compute_follow_camera_track() -> None:
    """Follow camera track centers on actor and handles cropping and clamping."""
    geometry_bounds = (0.0, 10.0, 0.0, 8.0)
    trajectory = [
        {"timestep": 0, "t_s": 0.0, "position": [1.0, 2.0]},
        {"timestep": 1, "t_s": 0.1, "position": [5.0, 4.0]},
        {"timestep": 2, "t_s": 0.2, "position": [9.0, 6.0]},
    ]
    aspect_ratio = 16.0 / 9.0

    # With small context retention (2m), camera will be cropped relative to 10x8 scene.
    spec, track, is_cropped = compute_follow_camera_track(
        trajectory, geometry_bounds, aspect_ratio, context_retention_m=2.0
    )
    assert spec["mode"] == "follow"
    assert spec["sample_count"] == 3
    assert is_cropped is True
    assert len(track) == 3

    # Clamping: at t=0, actor is at x=1.0, but view width is 4m, so min center x is 2.0
    assert track[0]["center"][0] >= 2.0 - 1e-4
    assert track[2]["center"][0] <= 8.0 + 1e-4


def test_compute_event_camera() -> None:
    """Event camera focuses on scenario event locations."""
    geometry_bounds = (0.0, 20.0, 0.0, 20.0)
    events = [
        {"event_id": "e1", "type": "collision", "t_s": 1.5, "position": [10.0, 12.0]},
        {"event_id": "e2", "type": "near_miss", "t_s": 3.0, "position": [15.0, 8.0]},
    ]
    aspect_ratio = 16.0 / 9.0

    spec, keyframes, is_cropped = compute_event_camera(
        events, geometry_bounds, aspect_ratio, context_retention_m=4.0
    )
    assert spec["mode"] == "event"
    assert spec["event_count"] == 2
    assert is_cropped is True
    assert keyframes[0]["event_id"] == "e1"
    assert keyframes[0]["center"] == [10.0, 12.0]
    assert keyframes[1]["event_id"] == "e2"
    assert keyframes[1]["center"] == [15.0, 8.0]


def test_run_complete_smoke_execution(tmp_path: Path) -> None:
    """Execute smoke test request with source file and verify generated artifacts."""
    source_data = _sample_source_scene()
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source_data), encoding="utf-8")

    out_dir = tmp_path / "out_smoke"
    request = ComponentRequest(
        request_id="test-req-001",
        component_id="review-camera",
        sources=(
            SourceRef(
                artifact_id="source-scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=str(out_dir.relative_to(tmp_path)),
        config={
            "camera_mode": "full_scene",
            "presentation": {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0},
        },
    )

    result = run(request, base=tmp_path)
    assert result.status == "complete"
    assert result.component_id == COMPONENT_ID
    assert result.request_id == "test-req-001"
    assert result.provenance["pixel_metric_free"] is True
    assert len(result.artifacts) == 2

    # Verify camera layout spec file
    spec_path = out_dir / f"{CAMERA_LAYOUT_SPEC_SCHEMA_VERSION}.json"
    assert spec_path.exists()
    spec_data = json.loads(spec_path.read_text(encoding="utf-8"))
    assert spec_data["schema_version"] == CAMERA_LAYOUT_SPEC_SCHEMA_VERSION
    assert spec_data["active_camera_mode"] == "full_scene"
    assert "full_scene" in spec_data["cameras"]
    assert "follow" in spec_data["cameras"]
    assert "event" in spec_data["cameras"]
    assert any("world units" in d for d in spec_data["disclosures"])

    # Verify visualization spec file
    vis_path = out_dir / f"{VISUALIZATION_SPEC_SCHEMA_VERSION}.json"
    assert vis_path.exists()
    vis_data = json.loads(vis_path.read_text(encoding="utf-8"))
    assert vis_data["schema_version"] == VISUALIZATION_SPEC_SCHEMA_VERSION
    assert vis_data["presentation"]["width"] == 320
    assert vis_data["presentation"]["height"] == 180


def test_run_follow_mode_with_inset_disclosure(tmp_path: Path) -> None:
    """Follow mode with small context retention discloses cropped view and activates inset."""
    out_dir = tmp_path / "out_follow"
    request = ComponentRequest(
        request_id="test-follow-001",
        component_id="review-camera",
        sources=(),
        output_directory=str(out_dir.relative_to(tmp_path)),
        config={
            "camera_mode": "follow",
            "geometry_bounds": [0.0, 50.0, 0.0, 40.0],
            "context_retention_m": 3.0,
            "trajectory": [
                {"timestep": 0, "t_s": 0.0, "pos": [10.0, 10.0]},
                {"timestep": 1, "t_s": 0.1, "pos": [20.0, 15.0]},
            ],
            "presentation": {"width": 320, "height": 180},
        },
    )

    result = run(request, base=tmp_path)
    assert result.status == "complete"
    spec_path = out_dir / f"{CAMERA_LAYOUT_SPEC_SCHEMA_VERSION}.json"
    spec_data = json.loads(spec_path.read_text(encoding="utf-8"))

    assert spec_data["active_camera_mode"] == "follow"
    assert spec_data["layout"]["inset"]["enabled"] is True
    assert spec_data["layout"]["inset"]["reference_bounds"]["max_x"] == 50.0
    assert any("Cropped content is disclosed" in d for d in spec_data["disclosures"])


def test_follow_mode_fails_closed_when_trajectory_missing(tmp_path: Path) -> None:
    """Follow mode requested without trajectory data must report unavailable status."""
    out_dir = tmp_path / "out_no_traj"
    request = ComponentRequest(
        request_id="test-no-traj",
        component_id="review-camera",
        sources=(),
        output_directory=str(out_dir.relative_to(tmp_path)),
        config={
            "camera_mode": "follow",
            "geometry_bounds": [0.0, 10.0, 0.0, 10.0],
        },
    )

    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "follow mode requires stated trajectory data" in result.reason


def test_unsupported_capability_fails_closed(tmp_path: Path) -> None:
    """Requests demanding unsupported capabilities must report unavailable."""
    out_dir = tmp_path / "out_cap"
    request = ComponentRequest(
        request_id="test-cap",
        component_id="review-camera",
        sources=(),
        output_directory=str(out_dir.relative_to(tmp_path)),
        config={"geometry_bounds": [0.0, 10.0, 0.0, 10.0]},
        required_capabilities=("unsupported-3d-raymarching",),
    )

    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "missing capabilities: unsupported-3d-raymarching" in result.reason


def test_unsupported_component_id(tmp_path: Path) -> None:
    """Request with mismatching component_id must report unavailable."""
    request = ComponentRequest(
        request_id="test-comp",
        component_id="wrong-component-id",
        sources=(),
        output_directory=str(tmp_path / "out"),
    )
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported component: wrong-component-id" in result.reason


def test_output_collision_rejected(tmp_path: Path) -> None:
    """If output directory already exists, execution must fail closed."""
    existing_dir = tmp_path / "existing_output"
    existing_dir.mkdir()

    request = ComponentRequest(
        request_id="test-collision",
        component_id="review-camera",
        sources=(),
        output_directory=str(existing_dir.relative_to(tmp_path)),
        config={"geometry_bounds": [0.0, 10.0, 0.0, 10.0]},
    )

    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "output collision" in result.reason


def test_corrupt_or_nonfinite_inputs(tmp_path: Path) -> None:
    """Non-finite geometry bounds or dimensions must fail closed."""
    out_dir = tmp_path / "out_nonfinite"
    request = ComponentRequest(
        request_id="test-nan",
        component_id="review-camera",
        sources=(),
        output_directory=str(out_dir.relative_to(tmp_path)),
        config={
            "geometry_bounds": [0.0, float("nan"), 0.0, 10.0],
        },
    )
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "finite" in result.reason


def test_deterministic_runs_produce_identical_digests(tmp_path: Path) -> None:
    """Repeated runs on the same input must produce identical artifact digests."""
    source_data = _sample_source_scene()
    source_file = tmp_path / "source.json"
    source_file.write_text(json.dumps(source_data), encoding="utf-8")

    out1 = tmp_path / "run_1"
    out2 = tmp_path / "run_2"

    req1 = ComponentRequest(
        request_id="req-det",
        component_id="review-camera",
        sources=(
            SourceRef(
                artifact_id="s",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=str(out1.relative_to(tmp_path)),
        config={"camera_mode": "full_scene", "presentation": {"width": 320, "height": 180}},
    )
    req2 = ComponentRequest(
        request_id="req-det",
        component_id="review-camera",
        sources=(
            SourceRef(
                artifact_id="s",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=str(out2.relative_to(tmp_path)),
        config={"camera_mode": "full_scene", "presentation": {"width": 320, "height": 180}},
    )

    res1 = run(req1, base=tmp_path)
    res2 = run(req2, base=tmp_path)

    assert res1.status == "complete"
    assert res2.status == "complete"

    digests1 = {a["artifact_id"]: a["sha256"] for a in res1.artifacts}
    digests2 = {a["artifact_id"]: a["sha256"] for a in res2.artifacts}
    assert digests1 == digests2


def test_cli_execution() -> None:
    """Test CLI invoking --descriptor and normal execution."""
    # Test --descriptor flag
    assert main(["--descriptor"]) == 0

    # Test full CLI run with fixture files
    out_rel = "output/test_cli_execution_out"
    out_path = Path(out_rel)
    if out_path.exists():
        shutil.rmtree(out_path)

    try:
        ret = main(
            [
                "--input",
                str(FIXTURE_DIR / "request.json"),
                "--config",
                str(FIXTURE_DIR / "config.json"),
                "--output",
                out_rel,
            ]
        )
        assert ret == 0
        assert (out_path / f"{CAMERA_LAYOUT_SPEC_SCHEMA_VERSION}.json").exists()
        assert (out_path / f"{VISUALIZATION_SPEC_SCHEMA_VERSION}.json").exists()
    finally:
        if out_path.exists():
            shutil.rmtree(out_path)
