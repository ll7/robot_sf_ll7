"""Contract tests for review telemetry and annotation overlays (SREV-12, issue #9281)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    ReviewContractsValidationError,
    SourceRef,
)
from robot_sf.render.review_overlays import (
    ALL_LAYERS,
    COMPONENT_ID,
    COMPONENT_VERSION,
    DESCRIPTOR_SCHEMA_VERSION,
    OPTIONAL_CAPABILITIES,
    OUTPUT_TYPES,
    OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION,
    REQUIRED_CAPABILITIES,
    _boxes_overlap,
    _compute_nonoverlapping_labels,
    _parse_bounds_value,
    _validate_presentation_dimensions,
    descriptor_document,
    main,
    run,
)

FIXTURE_DIR = Path("tests/fixtures/scenario_review/review_overlays")


def _sample_source_scene() -> dict:
    return {
        "schema_version": "threejs-viewer.v1",
        "episode_id": 101,
        "map": {
            "width": 10.0,
            "height": 8.0,
            "bounds": [
                [0.0, 10.0, 0.0, 0.0],
                [0.0, 10.0, 8.0, 8.0],
                [0.0, 0.0, 0.0, 8.0],
                [10.0, 10.0, 0.0, 8.0],
            ],
            "obstacles": [{"vertices": [[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]}],
        },
        "frames": [
            {
                "frame_idx": 0,
                "t_s": 0.0,
                "robot": {
                    "position": [1.0, 1.5],
                    "speed": 0.5,
                    "action": [0.5, 0.0],
                    "clearance_m": 1.5,
                },
                "pedestrians": [{"id": "ped-01", "position": [2.5, 1.8]}],
            },
            {
                "frame_idx": 1,
                "t_s": 0.1,
                "robot": {
                    "position": [2.0, 2.0],
                    "speed": 0.7,
                    "action": [0.7, 0.2],
                    "clearance_m": 1.2,
                },
                "pedestrians": [{"id": "ped-01", "position": [2.8, 2.2]}],
            },
            {
                "frame_idx": 2,
                "t_s": 0.2,
                "robot": {
                    "position": [3.5, 3.0],
                    "speed": 0.9,
                    "action": [0.9, 0.4],
                    "clearance_m": 0.85,
                },
                "pedestrians": [{"id": "ped-01", "position": [3.2, 2.8]}],
            },
            {
                "frame_idx": 3,
                "t_s": 0.3,
                "robot": {
                    "position": [5.0, 4.0],
                    "speed": 1.0,
                    "action": [1.0, 0.0],
                    "clearance_m": 1.4,
                },
                "pedestrians": [],
            },
        ],
        "trajectory": [
            [1.0, 1.5],
            [2.0, 2.0],
            [3.5, 3.0],
            [5.0, 4.0],
        ],
        "events": [
            {"event_id": "evt-001", "type": "checkpoint", "t_s": 0.2, "position": [3.5, 3.0]}
        ],
    }


def test_descriptor_contract() -> None:
    """The descriptor document must match declared version and capabilities."""
    doc = descriptor_document()
    assert doc["schema_version"] == DESCRIPTOR_SCHEMA_VERSION
    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    for cap in REQUIRED_CAPABILITIES:
        assert cap in doc["required_capabilities"]
    for cap in OPTIONAL_CAPABILITIES:
        assert cap in doc["optional_capabilities"]
    for out in OUTPUT_TYPES:
        assert out in doc["output_types"]


def test_dimension_validation() -> None:
    """Presentation dimensions must be positive integers; fps/speed must be positive floats."""
    w, h, fps, spd = _validate_presentation_dimensions(
        {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}
    )
    assert (w, h, fps, spd) == (320, 180, 10.0, 1.0)

    with pytest.raises(ReviewContractsValidationError, match="must be numeric"):
        _validate_presentation_dimensions({"width": "invalid"})

    with pytest.raises(ReviewContractsValidationError, match="must be > 0"):
        _validate_presentation_dimensions({"width": -10, "height": 100})

    with pytest.raises(ReviewContractsValidationError, match="must be finite and > 0"):
        _validate_presentation_dimensions({"fps": 0.0})

    with pytest.raises(ReviewContractsValidationError, match="must be finite and > 0"):
        _validate_presentation_dimensions({"speed": -1.0})


def test_bounds_validation() -> None:
    """Scenario geometry bounds must be finite, non-inverted rectangles."""
    assert _parse_bounds_value([0.0, 10.0, 0.0, 8.0]) == (0.0, 10.0, 0.0, 8.0)
    assert _parse_bounds_value({"min_x": 1.0, "max_x": 5.0, "min_y": 2.0, "max_y": 6.0}) == (
        1.0,
        5.0,
        2.0,
        6.0,
    )

    with pytest.raises(ReviewContractsValidationError, match="must be finite"):
        _parse_bounds_value([float("nan"), 10.0, 0.0, 8.0])

    with pytest.raises(ReviewContractsValidationError, match="inverted or flat"):
        _parse_bounds_value([10.0, 2.0, 0.0, 8.0])

    with pytest.raises(ReviewContractsValidationError, match="unrecognized geometry_bounds format"):
        _parse_bounds_value("invalid-bounds")


def test_box_overlap_logic() -> None:
    """Bounding box overlap helper detects intersections correctly."""
    b1 = (10, 10, 50, 30)
    b2 = (40, 20, 80, 40)
    b3 = (60, 40, 100, 60)
    assert _boxes_overlap(b1, b2) is True
    assert _boxes_overlap(b1, b3) is False


def test_nonoverlapping_labels_placement() -> None:
    """Close actors produce non-overlapping label bounding boxes."""
    labels = [
        {"actor_id": "robot", "anchor_px": (100, 100), "text": "Robot (1.0, 1.0)m"},
        {"actor_id": "ped-01", "anchor_px": (102, 102), "text": "ped-01 (1.0, 1.0)m"},
    ]
    placed = _compute_nonoverlapping_labels(labels, width=320, height=180)
    assert len(placed) == 2
    b1 = placed[0]["box"]
    b2 = placed[1]["box"]
    assert _boxes_overlap(b1, b2) is False


def test_missing_mapping_disables_synchronized_layers(tmp_path: Path) -> None:
    """Missing presentation timestamp mapping disables synchronized layers with reason."""
    source_file = tmp_path / "scene.json"
    source_file.write_text(json.dumps(_sample_source_scene()), encoding="utf-8")

    out_rel = "out_no_map"
    req = ComponentRequest(
        request_id="req-test-no-map",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out_rel,
        config={"presentation": {"width": 320, "height": 180}},
    )

    result = run(req, base=tmp_path)
    assert result.status == "partial"
    assert "missing_presentation_timestamp_map" in str(result.reason)
    assert (tmp_path / out_rel / "layer_availability.json").exists()
    assert (tmp_path / out_rel / "mapping_receipt.json").exists()

    avail = json.loads((tmp_path / out_rel / "layer_availability.json").read_text(encoding="utf-8"))
    assert avail["mapping_status"] == "disabled"
    for lay in ALL_LAYERS:
        assert lay in avail["disabled_layers"]


def test_pause_repeats_source_state(tmp_path: Path) -> None:
    """Pause freezes source simulation timestamp and repeats telemetry."""
    source_file = tmp_path / "scene.json"
    source_file.write_text(json.dumps(_sample_source_scene()), encoding="utf-8")

    out_rel = "out_pause"
    req = ComponentRequest(
        request_id="req-test-pause",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out_rel,
        config={
            "presentation": {
                "width": 320,
                "height": 180,
                "fps": 10.0,
                "speed": 1.0,
                "presentation_timestamp_map": {
                    "policy": "explicit_table",
                    "mapping": [
                        {"frame_idx": 0, "presentation_t_s": 0.0, "source_t_s": 0.1},
                        {
                            "frame_idx": 1,
                            "presentation_t_s": 0.1,
                            "source_t_s": 0.1,
                            "is_pause": True,
                        },
                    ],
                },
            }
        },
    )

    result = run(req, base=tmp_path)
    assert result.status == "complete"

    receipt = json.loads((tmp_path / out_rel / "mapping_receipt.json").read_text(encoding="utf-8"))
    frames = receipt["frames"]
    assert len(frames) == 2
    assert frames[0]["source_t_s"] == 0.1
    assert frames[1]["source_t_s"] == 0.1
    assert frames[1]["is_pause"] is True


def test_dropped_telemetry_detected(tmp_path: Path) -> None:
    """Telemetry requests beyond available simulation data report dropped telemetry."""
    source_file = tmp_path / "scene.json"
    source_file.write_text(json.dumps(_sample_source_scene()), encoding="utf-8")

    out_rel = "out_drop"
    req = ComponentRequest(
        request_id="req-test-drop",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out_rel,
        config={
            "presentation": {
                "width": 320,
                "height": 180,
                "presentation_timestamp_map": {
                    "policy": "explicit_table",
                    "mapping": [
                        {"frame_idx": 0, "presentation_t_s": 0.0, "source_t_s": 0.0},
                        {"frame_idx": 1, "presentation_t_s": 0.1, "source_t_s": 50.0},
                    ],
                },
            }
        },
    )

    result = run(req, base=tmp_path)
    assert result.status == "partial"
    assert any("dropped telemetry" in d for d in result.diagnostics)

    receipt = json.loads((tmp_path / out_rel / "mapping_receipt.json").read_text(encoding="utf-8"))
    assert len(receipt["dropped_telemetry"]) == 1
    assert receipt["dropped_telemetry"][0]["frame_idx"] == 1


def test_actor_disappearance_tracked(tmp_path: Path) -> None:
    """When an actor leaves or despawns, disappearance lifecycle event is recorded."""
    source_file = tmp_path / "scene.json"
    source_file.write_text(json.dumps(_sample_source_scene()), encoding="utf-8")

    out_rel = "out_disappear"
    req = ComponentRequest(
        request_id="req-test-disappear",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out_rel,
        config={
            "presentation": {
                "width": 320,
                "height": 180,
                "presentation_timestamp_map": {
                    "policy": "explicit_table",
                    "mapping": [
                        {"frame_idx": 0, "presentation_t_s": 0.2, "source_t_s": 0.2},
                        {"frame_idx": 1, "presentation_t_s": 0.3, "source_t_s": 0.3},
                    ],
                },
            }
        },
    )

    result = run(req, base=tmp_path)
    assert result.status == "complete"

    receipt = json.loads((tmp_path / out_rel / "mapping_receipt.json").read_text(encoding="utf-8"))
    events = receipt["actor_lifecycle"]
    assert any(e["actor_id"] == "ped-01" and e["event"] == "disappeared" for e in events)


def test_output_collision_rejected(tmp_path: Path) -> None:
    """Existing output directories fail closed to protect prior artifacts."""
    source_file = tmp_path / "scene.json"
    source_file.write_text(json.dumps(_sample_source_scene()), encoding="utf-8")

    out_rel = "out_collide"
    (tmp_path / out_rel).mkdir(parents=True, exist_ok=True)

    req = ComponentRequest(
        request_id="req-test-collide",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out_rel,
    )

    with pytest.raises(ReviewContractsValidationError, match="output collision"):
        run(req, base=tmp_path)


def test_corrupt_source_json(tmp_path: Path) -> None:
    """Corrupted source files raise ReviewContractsValidationError."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{not-valid-json", encoding="utf-8")

    out_rel = "out_corrupt"
    req = ComponentRequest(
        request_id="req-test-corrupt",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="corrupt-scene",
                uri=str(corrupt_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out_rel,
    )

    with pytest.raises(ReviewContractsValidationError, match="corrupt JSON"):
        run(req, base=tmp_path)


def test_source_integrity_mismatch(tmp_path: Path) -> None:
    """Mismatch in declared SHA-256 raises ReviewContractsValidationError."""
    source_file = tmp_path / "scene.json"
    source_file.write_text(json.dumps(_sample_source_scene()), encoding="utf-8")

    out_rel = "out_sha_mismatch"
    req = ComponentRequest(
        request_id="req-test-sha-mismatch",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
                sha256="0000000000000000000000000000000000000000000000000000000000000000",
            ),
        ),
        output_directory=out_rel,
    )

    with pytest.raises(ReviewContractsValidationError, match="integrity mismatch"):
        run(req, base=tmp_path)


def test_unsupported_capabilities_unavailable(tmp_path: Path) -> None:
    """Requesting undeclared capabilities returns unavailable status."""
    out_rel = "out_unsupported"
    req = ComponentRequest(
        request_id="req-test-unsupported",
        component_id=COMPONENT_ID,
        required_capabilities=("quantum-rendering",),
        sources=(),
        output_directory=out_rel,
    )

    result = run(req, base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported required capabilities" in str(result.reason)


def test_deterministic_repeated_runs(tmp_path: Path) -> None:
    """Repeated runs on the same input generate identical artifact SHA-256 hashes."""
    source_file = tmp_path / "scene.json"
    source_file.write_text(json.dumps(_sample_source_scene()), encoding="utf-8")

    cfg = {
        "presentation": {
            "width": 320,
            "height": 180,
            "fps": 10.0,
            "speed": 1.0,
            "presentation_timestamp_map": {
                "policy": "explicit_table",
                "mapping": [
                    {"frame_idx": 0, "presentation_t_s": 0.0, "source_t_s": 0.0},
                    {"frame_idx": 1, "presentation_t_s": 0.1, "source_t_s": 0.1},
                ],
            },
        }
    }

    out1 = "run1"
    req1 = ComponentRequest(
        request_id="req-det-1",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out1,
        config=cfg,
    )
    res1 = run(req1, base=tmp_path)
    assert res1.status == "complete"

    out2 = "run2"
    req2 = ComponentRequest(
        request_id="req-det-2",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="scene",
                uri=str(source_file.relative_to(tmp_path)),
                format="threejs-viewer.v1",
                schema="threejs-viewer.v1",
            ),
        ),
        output_directory=out2,
        config=cfg,
    )
    res2 = run(req2, base=tmp_path)
    assert res2.status == "complete"

    receipt1 = json.loads((tmp_path / out1 / "mapping_receipt.json").read_text(encoding="utf-8"))
    receipt2 = json.loads((tmp_path / out2 / "mapping_receipt.json").read_text(encoding="utf-8"))

    sha1_f0 = receipt1["frames"][0]["sha256"]
    sha2_f0 = receipt2["frames"][0]["sha256"]
    assert sha1_f0 == sha2_f0

    sha1_f1 = receipt1["frames"][1]["sha256"]
    sha2_f1 = receipt2["frames"][1]["sha256"]
    assert sha1_f1 == sha2_f1


def test_cli_execution() -> None:
    """Test CLI invoking --descriptor and normal execution with fixture files."""
    assert main(["--descriptor"]) == 0

    out_rel = "output/test_review_overlays_cli_out"
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
        assert (out_path / "layer_availability.json").exists()
        assert (out_path / "mapping_receipt.json").exists()
        assert (out_path / "visualization_spec.json").exists()

        receipt = json.loads((out_path / "mapping_receipt.json").read_text(encoding="utf-8"))
        assert receipt["schema_version"] == OVERLAY_MAPPING_RECEIPT_SCHEMA_VERSION
        assert len(receipt["frames"]) == 5

        f0 = out_path / "frames" / "frame_0000.png"
        assert f0.exists()
        img = Image.open(f0)
        assert img.mode == "RGBA"
        assert img.size == (320, 180)
    finally:
        if out_path.exists():
            shutil.rmtree(out_path)


def test_cli_descriptor_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI --descriptor prints descriptor JSON and returns 0."""
    code = main(["--descriptor"])
    assert code == 0
    captured = capsys.readouterr()
    doc = json.loads(captured.out)
    assert doc["schema_version"] == "component-descriptor.v1"
    assert doc["component_id"] == COMPONENT_ID
