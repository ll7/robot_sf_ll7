"""Smoke tests for the trace-export Three.js viewer."""

import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import (
    load_simulation_trace_export,
)
from robot_sf.render.threejs_viewer import SCENE_SCHEMA_VERSION
from robot_sf.render.trace_viewer import (
    TRACE_VIEWER_SCENE_VERSION,
    TraceViewerResult,
    build_trace_scene,
    export_trace_viewer,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
)
MINIMAL_TRACE_PATH = FIXTURE_DIR / "minimal_trace.json"
MATERIALIZED_TRACE_PATH = FIXTURE_DIR / "planner_sanity_open_episode_0000.json"
ANNOTATION_FIXTURE_DIR = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "trace_annotation_set_v1"
)
ANNOTATION_PATH = ANNOTATION_FIXTURE_DIR / "issue_1962_planner_sanity_open_annotations.json"


def test_build_trace_scene_contains_minimal_trace_payload() -> None:
    """The minimal trace fixture should produce a valid scene payload."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)
    scene = build_trace_scene(trace, source=str(MINIMAL_TRACE_PATH))

    assert scene["schema_version"] == SCENE_SCHEMA_VERSION
    assert scene["trace_viewer_version"] == TRACE_VIEWER_SCENE_VERSION
    assert scene["trace_id"] == "fixture_trace_001"
    assert scene["metadata"]["source"]["scenario_id"] == "classic_bottleneck_medium"
    assert len(scene["frames"]) == 2
    assert scene["frames"][0]["robot"]["position"] == [0.0, 0.0]
    assert scene["frames"][0]["pedestrians"][0]["id"] == "ped_1"
    assert scene["frames"][0]["pedestrians"][0]["position"] == [1.0, 0.5]
    assert scene["frames"][0]["event"] == "start"
    assert scene["frames"][1]["event"] == "advance"
    assert scene["frames"][1]["planner_action"] == {
        "linear_velocity": 0.1,
        "angular_velocity": 0.0,
    }
    assert scene["frames"][1]["robot"]["velocity"] == [0.1, 0.0]


def test_build_trace_scene_contains_event_ids() -> None:
    """The materialized trace fixture should preserve event_id in scene frames."""
    trace = load_simulation_trace_export(MATERIALIZED_TRACE_PATH)
    scene = build_trace_scene(trace, source=str(MATERIALIZED_TRACE_PATH))

    assert scene["trace_id"] == "planner_sanity_open-ep0-seed7-source-aae87cf6"
    assert len(scene["frames"]) == 3
    assert scene["frames"][0]["event_id"] == (
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0000"
    )
    assert scene["frames"][2]["event_id"] == (
        "issue_1859_planner_sanity_open_trace_fixture_gen_7_ep0000-frame-0002"
    )
    assert scene["frames"][0]["event"] == "step"


def test_build_trace_scene_computes_map_bounds_from_positions() -> None:
    """Map bounds should be auto-computed from robot and pedestrian positions."""
    trace = load_simulation_trace_export(MATERIALIZED_TRACE_PATH)
    scene = build_trace_scene(trace)

    map_data = scene["map"]
    assert map_data["origin"][0] < scene["frames"][0]["robot"]["position"][0]
    assert map_data["origin"][1] < scene["frames"][0]["robot"]["position"][1]
    assert map_data["width"] > 0
    assert map_data["height"] > 0
    assert len(map_data["bounds"]) == 4
    assert map_data["obstacles"] == []
    assert map_data["robot_spawn_zones"] == []


def test_build_trace_scene_trajectory_from_robot_positions() -> None:
    """Trajectory should be computed from robot positions across frames."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)
    scene = build_trace_scene(trace)

    assert scene["trajectory"] == [[0.0, 0.0], [0.01, 0.0]]


def test_build_trace_scene_includes_source_metadata() -> None:
    """Source metadata should be embedded in the scene for traceability."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)
    scene = build_trace_scene(trace, source="test_source")

    meta = scene["metadata"]
    assert meta["schema_version"] == "simulation_trace_export.v1"
    assert meta["evidence_boundary"] == "analysis_workbench_only"
    assert meta["coordinate_frame"] == "world"
    assert meta["source"]["planner_id"] == "hybrid_rule_v0_minimal"
    assert meta["source"]["seed"] == 111
    assert "units" in meta


def test_build_trace_scene_with_annotations() -> None:
    """Optional annotations should be embedded in the scene payload."""
    trace = load_simulation_trace_export(MATERIALIZED_TRACE_PATH)

    from robot_sf.analysis_workbench.trace_annotation import (
        load_trace_annotation_set,
    )

    annotation_set = load_trace_annotation_set(ANNOTATION_PATH)
    annotations_payload = [
        {
            "annotation_id": a.annotation_id,
            "summary": a.summary,
            "category": a.category,
            "evidence_type": a.evidence_type,
            "anchor": {
                "frame_start": a.anchor.frame_start,
                "frame_end": a.anchor.frame_end,
                "event_ids": list(a.anchor.event_ids),
                "entities": [{"type": e.type, "id": e.id} for e in a.anchor.entities],
            },
            "details": a.details,
        }
        for a in annotation_set.annotations
    ]

    scene = build_trace_scene(trace, annotations=annotations_payload)

    assert "annotations" in scene
    assert len(scene["annotations"]) == 3
    assert scene["annotations"][0]["annotation_id"].startswith("issue_1962_step")
    ann = scene["annotations"][0]
    assert ann["category"] == "planner_action"
    assert ann["anchor"]["frame_start"] == 1
    assert ann["anchor"]["frame_end"] == 2
    assert len(ann["anchor"]["event_ids"]) == 2
    assert len(ann["anchor"]["entities"]) == 1
    assert ann["anchor"]["entities"][0]["type"] == "robot"
    assert ann["details"] is not None


def test_export_trace_viewer_writes_static_assets(tmp_path: Path) -> None:
    """A minimal trace should produce index.html, viewer.js, and scene.json."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    result = export_trace_viewer(trace, tmp_path / "trace_viewer")

    assert isinstance(result, TraceViewerResult)
    assert result.html_path.exists()
    assert (result.output_dir / "viewer.js").exists()
    assert result.scene_path.exists()

    scene = json.loads(result.scene_path.read_text(encoding="utf-8"))
    assert scene["trace_id"] == "fixture_trace_001"
    assert scene["trace_viewer_version"] == TRACE_VIEWER_SCENE_VERSION
    assert len(scene["frames"]) == 2


def test_build_trace_scene_annotation_spans_visualization_data() -> None:
    """Annotation anchors should carry frame ranges usable for timeline span rendering."""
    trace = load_simulation_trace_export(MATERIALIZED_TRACE_PATH)

    from robot_sf.analysis_workbench.trace_annotation import (
        load_trace_annotation_set,
    )

    annotation_set = load_trace_annotation_set(ANNOTATION_PATH)
    annotations_payload = [
        {
            "annotation_id": a.annotation_id,
            "summary": a.summary,
            "anchor": {
                "frame_start": a.anchor.frame_start,
                "frame_end": a.anchor.frame_end,
            },
        }
        for a in annotation_set.annotations
    ]

    scene = build_trace_scene(trace, annotations=annotations_payload)

    frame_steps = {f["timestep"] for f in scene["frames"]}
    for ann in scene["annotations"]:
        anchor = ann["anchor"]
        assert "frame_start" in anchor
        assert "frame_end" in anchor
        assert anchor["frame_end"] >= anchor["frame_start"]
        assert anchor["frame_start"] in frame_steps
        assert anchor["frame_end"] in frame_steps


def test_export_trace_viewer_uses_trace_viewer_version(tmp_path: Path) -> None:
    """The exported scene should carry the trace viewer version tag."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    result = export_trace_viewer(trace, tmp_path, source="test")
    scene = json.loads(result.scene_path.read_text(encoding="utf-8"))

    assert scene["trace_viewer_version"] == TRACE_VIEWER_SCENE_VERSION


def test_build_trace_scene_limitations_are_diagnostic_only(tmp_path: Path) -> None:
    """Limitations should state this is for qualitative review only."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)
    scene = build_trace_scene(trace)

    limitations = scene["limitations"]
    assert any("qualitative" in lim for lim in limitations)
    assert any("simulation_trace_export.v1" in lim for lim in limitations)
    assert any("diagnostic-only" in lim for lim in limitations)


def test_build_trace_scene_metadata_diagnostic_flag() -> None:
    """Scene metadata should carry diagnostic_only flag."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)
    scene = build_trace_scene(trace)

    assert scene["metadata"]["diagnostic_only"] is True


def test_build_trace_scene_with_empty_frames_raises() -> None:
    """An export with no frames should raise an error."""
    from robot_sf.analysis_workbench.simulation_trace_export import (
        SimulationTraceExport,
        SimulationTraceSource,
    )

    empty_trace = SimulationTraceExport(
        schema_version="simulation_trace_export.v1",
        trace_id="empty",
        source=SimulationTraceSource(
            scenario_id="test",
            seed=0,
            planner_id="test",
            episode_id="0",
            generated_by="test",
        ),
        evidence_boundary="analysis_workbench_only",
        coordinate_frame="world",
        units={"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        frames=[],
    )

    with pytest.raises(ValueError, match="at least one trace frame"):
        build_trace_scene(empty_trace)


def test_trace_viewer_cli_rejects_mismatched_annotation_trace(tmp_path: Path) -> None:
    """The CLI should fail closed when annotation anchors point at another trace."""
    from robot_sf.render.trace_viewer import main

    exit_code = main(
        [
            str(MINIMAL_TRACE_PATH),
            "--annotations",
            str(ANNOTATION_PATH),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert not (tmp_path / "scene.json").exists()


def test_export_trace_viewer_roundtrip_materialized_fixture(tmp_path: Path) -> None:
    """The materialized trace fixture should roundtrip through the export pipeline."""
    trace = load_simulation_trace_export(MATERIALIZED_TRACE_PATH)

    result = export_trace_viewer(trace, tmp_path, source=str(MATERIALIZED_TRACE_PATH))
    scene = json.loads(result.scene_path.read_text(encoding="utf-8"))

    assert scene["trace_id"] == trace.trace_id
    assert scene["episode_id"] == trace.source.episode_id
    assert len(scene["frames"]) == len(trace.frames)

    for scene_frame, trace_frame in zip(scene["frames"], trace.frames, strict=True):
        assert scene_frame["timestep"] == trace_frame.step
        assert abs(scene_frame["time_s"] - trace_frame.time_s) < 1e-9


def test_build_trace_scene_with_map_geometry_merges_zones_and_obstacles() -> None:
    """Supplied map geometry should replace empty obstacle/zone lists in computed map."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    geometry = {
        "obstacles": [
            {"vertices": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]], "lines": []},
        ],
        "robot_spawn_zones": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        "robot_goal_zones": [[[9.0, 9.0], [10.0, 9.0], [10.0, 10.0], [9.0, 10.0]]],
        "ped_spawn_zones": [],
        "ped_goal_zones": [],
        "ped_crowded_zones": [],
    }

    scene = build_trace_scene(trace, map_geometry=geometry)

    map_data = scene["map"]
    assert len(map_data["obstacles"]) == 1
    assert map_data["obstacles"][0]["vertices"] == geometry["obstacles"][0]["vertices"]
    assert map_data["robot_spawn_zones"] == geometry["robot_spawn_zones"]
    assert map_data["robot_goal_zones"] == geometry["robot_goal_zones"]
    assert map_data["ped_spawn_zones"] == []
    assert map_data["ped_goal_zones"] == []
    assert map_data["ped_crowded_zones"] == []

    auto = build_trace_scene(trace)
    auto_map = auto["map"]
    assert auto_map["obstacles"] == []
    assert auto_map["robot_spawn_zones"] == []


def test_build_trace_scene_map_geometry_updates_limitations() -> None:
    """Limitations should mention geometry overlay when map_geometry is supplied."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    scene = build_trace_scene(trace, map_geometry={"obstacles": [], "robot_spawn_zones": []})

    limitations = scene["limitations"]
    assert any("Map geometry overlaid" in lim for lim in limitations)
    assert any("not ground-truth map validation" in lim for lim in limitations)


def test_build_trace_scene_without_map_geometry_has_auto_bounds_limitation() -> None:
    """Without geometry, limitations should mention auto-computed bounds."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    scene = build_trace_scene(trace)

    limitations = scene["limitations"]
    assert any("auto-computed from trace positions" in lim for lim in limitations)
    assert any("no SVG map geometry" in lim for lim in limitations)


def test_build_trace_scene_map_geometry_rejects_unrecognised_key() -> None:
    """An unrecognised key in map_geometry should raise ValueError."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    with pytest.raises(ValueError, match="unrecognised key"):
        build_trace_scene(trace, map_geometry={"bogus": []})


def test_build_trace_scene_map_geometry_rejects_non_list_value() -> None:
    """A non-list value for a geometry key should raise ValueError."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    with pytest.raises(ValueError, match="expected a list"):
        build_trace_scene(trace, map_geometry={"obstacles": "not a list"})


def test_build_trace_scene_map_geometry_rejects_obstacle_without_vertices() -> None:
    """An obstacle dict missing a required field should raise ValueError."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    with pytest.raises(ValueError, match="missing 'vertices'"):
        build_trace_scene(trace, map_geometry={"obstacles": [{"lines": []}]})


def test_build_trace_scene_map_geometry_rejects_obstacle_without_lines() -> None:
    """An obstacle dict missing lines should raise ValueError."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    with pytest.raises(ValueError, match="missing 'lines'"):
        build_trace_scene(trace, map_geometry={"obstacles": [{"vertices": []}]})


def test_build_trace_scene_map_geometry_rejects_non_dict_obstacle() -> None:
    """A non-dict obstacle entry should raise ValueError."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    with pytest.raises(ValueError, match="expected a dict"):
        build_trace_scene(trace, map_geometry={"obstacles": ["string"]})


def test_build_trace_scene_map_geometry_rejects_malformed_zone() -> None:
    """A non-list zone entry should raise ValueError."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    with pytest.raises(ValueError, match="expected a list"):
        build_trace_scene(trace, map_geometry={"robot_spawn_zones": ["not_a_zone"]})


def test_build_trace_scene_map_geometry_rejects_bad_point_in_zone() -> None:
    """A non-[x, y] point in a zone should raise ValueError."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    with pytest.raises(ValueError, match="expected \\[x, y\\] point pair"):
        build_trace_scene(
            trace,
            map_geometry={"robot_spawn_zones": [[[1.0]]]},
        )


def test_export_trace_viewer_with_map_geometry(tmp_path: Path) -> None:
    """map_geometry should be passed through export_trace_viewer to scene.json."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    geometry = {
        "obstacles": [
            {
                "vertices": [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
                "lines": [[0.0, 0.0, 5.0, 0.0]],
            },
        ],
    }

    result = export_trace_viewer(trace, tmp_path / "geo_viewer", map_geometry=geometry)
    scene = json.loads(result.scene_path.read_text(encoding="utf-8"))

    assert len(scene["map"]["obstacles"]) == 1
    assert scene["map"]["obstacles"][0]["vertices"] == geometry["obstacles"][0]["vertices"]


def test_export_trace_viewer_without_map_geometry_preserves_empty_zones(
    tmp_path: Path,
) -> None:
    """Export without geometry should preserve auto-computed empty zones in scene.json."""
    trace = load_simulation_trace_export(MINIMAL_TRACE_PATH)

    result = export_trace_viewer(trace, tmp_path / "no_geo_viewer")
    scene = json.loads(result.scene_path.read_text(encoding="utf-8"))

    assert scene["map"]["obstacles"] == []
    assert scene["map"]["robot_spawn_zones"] == []
    assert scene["map"]["robot_goal_zones"] == []
    assert scene["map"]["ped_spawn_zones"] == []
    assert scene["map"]["ped_goal_zones"] == []
    assert scene["map"]["ped_crowded_zones"] == []
