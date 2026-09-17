"""Focused SREV-16 synchronization, provenance, and offline-control tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    SourceRef,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.render import review_panels

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "scenario_review" / "review_panels"
)


def _request(
    tmp_path: Path,
    *,
    config: dict[str, object] | None = None,
    output: str = "review",
    sources: list[dict[str, str]] | None = None,
    required_capabilities: list[str] | None = None,
) -> review_panels.ComponentRequest:
    source_rows = sources or [
        {"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"},
        {"artifact_id": "video", "uri": "video-map.json", "format": "media-mapping.v1"},
        {"artifact_id": "metrics", "uri": "metrics.json", "format": "metric-series.v1"},
        {"artifact_id": "events", "uri": "events.json", "format": "event-index.v1"},
    ]
    payload: dict[str, object] = {
        "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
        "request_id": "srev16-test",
        "component_id": review_panels.COMPONENT_ID,
        "sources": source_rows,
        "output_directory": output,
        "config": config or {},
    }
    if required_capabilities is not None:
        payload["required_capabilities"] = required_capabilities
    return component_request_from_dict(payload)


def _copy_fixtures(tmp_path: Path) -> None:
    for name in ("scene.json", "video-map.json", "metrics.json", "events.json"):
        shutil.copyfile(FIXTURE_ROOT / name, tmp_path / name)


def test_descriptor_is_shared_contract_valid() -> None:
    descriptor = component_descriptor_from_dict(review_panels.descriptor_document())

    assert descriptor.component_id == review_panels.COMPONENT_ID
    assert descriptor.supported_input_versions == (COMPONENT_REQUEST_SCHEMA_VERSION,)
    assert "review-panels.v1" in descriptor.output_types
    assert "source-time-cursor" in descriptor.required_capabilities


def test_fixture_builds_all_panels_with_source_time_and_provenance(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path)
    request = _request(
        tmp_path,
        output="out",
        config={
            "sample_resolution_s": 0.4,
            "cursor_time_s": 10.5,
            "intervals": [{"interval_id": "interaction", "start_s": 10.4, "end_s": 10.8}],
            "selected_interval": "interaction",
            "requested_metrics": ["clearance", "collision_count", "goal_distance"],
            "metric_visibility": {
                "clearance": True,
                "collision_count": False,
                "goal_distance": True,
            },
        },
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    assert len(result.artifacts) == 4
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["status"] == "complete"
    assert document["time"]["authority"] == "simulation_time"
    assert document["time"]["origin_s"] == 10.0
    assert document["time"]["terminal_s"] == 12.2
    assert document["time"]["rule"] == "nearest_sample_within_declared_resolution; no_interpolation"
    assert document["context"]["episode_id"] == "episode-17"
    assert document["context"]["execution_id"] == "exec-17"
    assert document["context"]["cursor"]["time_s"] == 10.5
    assert document["source_identity_status"]["scene"]["status"] == "available"
    assert document["goal_geometry"]["goal_point"] == [4.0, 1.0]
    assert document["goal_geometry"]["completion_boundary_status"] == "available"
    assert document["panel_status"] == {
        "scene": "available",
        "video": "available",
        "metrics": "available",
        "events": "available",
    }
    assert document["panels"]["scene"]["sample_time_s"] == 10.5
    assert document["panels"]["video"]["sample_time_s"] == 10.1
    assert document["panels"]["video"]["temporal_error_s"] == pytest.approx(0.4)
    assert document["metrics"]["clearance"]["unit"] == "m"
    assert document["metrics"]["clearance"]["missingness"]["missing_count"] == 1
    assert document["metrics"]["collision_count"]["visible"] is False
    assert document["metrics"]["goal_distance"]["derived"] is True
    assert document["panels"]["events"][0]["selected"] is True
    html = (tmp_path / "out" / "review-panels.v1.html").read_text()
    javascript = (
        tmp_path / "out" / "components" / "review_panels" / "review_panels.js"
    ).read_text()
    assert "unpkg" not in html + javascript
    assert "http://" not in html + javascript
    assert result.provenance["admission"] == "not_evaluated"


def test_nearest_sample_tie_prefers_earlier_and_missing_is_explicit() -> None:
    stream = {
        "status": "available",
        "resolution_s": 0.5,
        "samples": [
            {"time_s": 10.0, "value": "early", "source_index": 0},
            {"time_s": 11.0, "value": "late", "source_index": 1},
        ],
    }

    tie = review_panels.nearest_sample(stream, 10.5)
    missing = review_panels.nearest_sample(
        {
            **stream,
            "samples": [{"time_s": 10.0, "value": None, "missing": True}],
        },
        10.0,
    )

    assert tie.status == "available"
    assert tie.sample_time_s == 10.0
    assert tie.reason == ""
    assert missing.status == "unavailable"
    assert missing.reason == "sample_missing"
    assert missing.resolution_s == 0.5


def test_gaps_unequal_duration_and_resolution_do_not_hold_stale_values(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path)
    request = _request(
        tmp_path, config={"sample_resolution_s": 0.1, "cursor_time_s": 11.5}, output="out"
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["panels"]["scene"]["status"] == "unavailable"
    assert document["panels"]["scene"]["reason"] == "outside_declared_resolution"
    assert document["panels"]["video"]["status"] == "unavailable"
    assert document["panels"]["video"]["reason"] == "outside_declared_resolution"
    assert document["streams"]["scene"]["gaps"]
    assert document["streams"]["video"]["range"]["end_s"] == 12.2


def test_simulation_timeline_state_is_preserved_as_scene_geometry(tmp_path: Path) -> None:
    (tmp_path / "timeline.json").write_text(
        json.dumps(
            {
                "schema_version": "simulation_timeline.v1",
                "source_trace": {"source": {"episode_id": "timeline-episode"}},
                "frames": [
                    {
                        "frame_index": 0,
                        "step": 1,
                        "time_s": 7.5,
                        "state": {
                            "robot": {"position": [2.0, 3.0]},
                            "pedestrians": [],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    request = _request(
        tmp_path,
        output="out",
        sources=[
            {
                "artifact_id": "timeline",
                "uri": "timeline.json",
                "format": "simulation-timeline.v1",
            }
        ],
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["panels"]["scene"]["value"]["robot"]["position"] == [2.0, 3.0]


def test_video_requires_explicit_source_time_mapping(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path)
    (tmp_path / "video-map.json").write_text(
        json.dumps(
            {
                "schema_version": "media-mapping.v1",
                "fps_nominal": 30,
                "frames": [{"frame_index": 0, "pts_s": 0.0}],
            }
        ),
        encoding="utf-8",
    )
    request = _request(tmp_path, config={"sample_resolution_s": 0.2}, output="out")

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "partial"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["panel_status"]["video"] == "unavailable"
    assert document["streams"]["video"]["reason"] == "presentation_timestamp_map_missing"
    assert any(
        diagnostic["reason_code"] == "presentation_timestamp_map_missing"
        for diagnostic in result.diagnostics
    )


def test_request_config_timestamp_map_is_explicitly_consumed(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path)
    (tmp_path / "video-map.json").write_text("opaque media", encoding="utf-8")
    config = {
        "sample_resolution_s": 0.2,
        "presentation": {
            "presentation_timestamp_map": {
                "policy": "explicit_table",
                "mapping": [
                    {"frame_idx": 0, "presentation_t_s": 0.0, "source_t_s": 10.0},
                    {"frame_idx": 1, "presentation_t_s": 0.1, "source_t_s": 10.5},
                ],
            }
        },
    }

    result = review_panels.run(_request(tmp_path, config=config, output="out"), base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["panel_status"]["video"] == "available"
    assert document["streams"]["video"]["samples"][1]["value"]["frame_idx"] == 1


def test_explicit_video_pause_allows_repeated_source_time(tmp_path: Path) -> None:
    (tmp_path / "scene.json").write_text(
        json.dumps({"schema_version": "threejs-viewer.v1", "frames": [{"time_s": 10.0}]}),
        encoding="utf-8",
    )
    (tmp_path / "video.mp4").write_bytes(b"opaque media")
    request = _request(
        tmp_path,
        output="out",
        sources=[
            {"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"},
            {"artifact_id": "video", "uri": "video.mp4", "format": "video-mp4.v1"},
        ],
        config={
            "sample_resolution_s": 0.5,
            "media_uri": "video.mp4",
            "presentation": {
                "presentation_timestamp_map": {
                    "policy": "explicit_table",
                    "mapping": [
                        {"frame_idx": 0, "presentation_t_s": 0.0, "source_t_s": 10.0},
                        {
                            "frame_idx": 1,
                            "presentation_t_s": 0.0,
                            "source_t_s": 10.0,
                            "is_pause": True,
                        },
                        {"frame_idx": 2, "presentation_t_s": 0.2, "source_t_s": 10.5},
                    ],
                }
            },
        },
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["panel_status"]["video"] == "available"
    assert [row["time_s"] for row in document["streams"]["video"]["samples"]] == [10.0, 10.0, 10.5]
    assert document["streams"]["video"]["media_uri"] == "video.mp4"


def test_repeated_video_source_time_without_pause_is_unavailable(tmp_path: Path) -> None:
    (tmp_path / "video.mp4").write_bytes(b"opaque media")
    request = _request(
        tmp_path,
        output="out",
        sources=[{"artifact_id": "video", "uri": "video.mp4", "format": "video-mp4.v1"}],
        config={
            "presentation": {
                "presentation_timestamp_map": {
                    "mapping": [
                        {"frame_idx": 0, "presentation_t_s": 0.0, "source_t_s": 10.0},
                        {"frame_idx": 1, "presentation_t_s": 0.1, "source_t_s": 10.0},
                    ]
                }
            }
        },
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "partial"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["streams"]["video"]["reason"] == "source_time_not_nondecreasing"
    assert any(
        diagnostic["reason_code"] == "source_time_duplicate_without_pause"
        for diagnostic in result.diagnostics
    )


def test_missing_declared_source_is_partial_without_complete_artifacts(tmp_path: Path) -> None:
    (tmp_path / "scene.json").write_text(
        json.dumps({"schema_version": "threejs-viewer.v1", "frames": [{"time_s": 1.0}]}),
        encoding="utf-8",
    )
    request = _request(
        tmp_path,
        output="out",
        sources=[
            {"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"},
            {"artifact_id": "missing", "uri": "missing.json", "format": "metric-series.v1"},
        ],
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "partial"
    assert result.artifacts == ()
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["status"] == "partial"
    assert any(diagnostic["reason_code"] == "source_missing" for diagnostic in result.diagnostics)


def test_metric_scalar_envelope_and_actor_alias_are_unwrapped(tmp_path: Path) -> None:
    (tmp_path / "scene.json").write_text(
        json.dumps(
            {
                "schema_version": "threejs-viewer.v1",
                "source_identity": {"episode_id": "episode-1"},
                "frames": [
                    {
                        "time_s": 1.0,
                        "robot": {"actor": "ego-1", "position": [0.0, 0.0]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "schema_version": "metric-series.v1",
                "series": [
                    {
                        "metric_id": "clearance",
                        "units": "m",
                        "samples": [{"time_s": 1.0, "value": {"value": 0.75, "unit": "m"}}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    request = _request(
        tmp_path,
        output="out",
        sources=[
            {"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"},
            {"artifact_id": "metrics", "uri": "metrics.json", "format": "metric-series.v1"},
        ],
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["context"]["actor_id"] == "ego-1"
    assert document["source_identity"]["scene"]["actor_id"] == "ego-1"
    assert document["metrics"]["clearance"]["stream"]["samples"][0]["value"] == 0.75


def test_scene_and_video_surface_metadata_are_available_offline(tmp_path: Path) -> None:
    (tmp_path / "scene.json").write_text(
        json.dumps(
            {
                "schema_version": "threejs-viewer.v1",
                "map": {"width": 4.0, "height": 3.0, "origin": [0.0, 0.0]},
                "frames": [{"time_s": 1.0, "robot": {"position": [1.0, 1.0]}}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "clip.mp4").write_bytes(b"opaque media")
    request = _request(
        tmp_path,
        output="out",
        sources=[
            {"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"},
            {"artifact_id": "video", "uri": "clip.mp4", "format": "video-mp4.v1"},
        ],
        config={
            "media_uri": "clip.mp4",
            "presentation": {
                "presentation_timestamp_map": {
                    "mapping": [{"frame_idx": 0, "pts_s": 0.0, "source_t_s": 1.0}]
                }
            },
        },
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["scene_surface"]["map"]["width"] == 4.0
    assert document["streams"]["video"]["media_uri"] == "clip.mp4"
    assert document["panels"]["video"]["media_uri"] == "clip.mp4"
    assert document["surfaces"]["scene"]["mount"] == "mountSceneViewer"
    assert document["surfaces"]["video"]["renderer"] == "HTMLVideoElement"


def test_context_revision_changes_on_seek_and_interval() -> None:
    context = review_panels.ReviewContext(
        episode_id="ep-1", cursor=review_panels.SourceTimeCursor(2.0)
    )

    sought = context.seek(3.0, source="event")
    selected = sought.with_interval("interval-1")

    assert context.context_revision == 0
    assert sought.context_revision == 1
    assert sought.cursor.source == "event"
    assert selected.context_revision == 2
    assert selected.cursor.interval_id == "interval-1"


def test_missing_identity_and_goal_geometry_are_explicit(tmp_path: Path) -> None:
    (tmp_path / "scene.json").write_text(
        json.dumps(
            {
                "schema_version": "threejs-viewer.v1",
                "frames": [{"time_s": 5.0, "robot": {"position": [0, 0]}}],
            }
        ),
        encoding="utf-8",
    )
    request = _request(
        tmp_path,
        output="out",
        sources=[{"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"}],
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["source_identity_status"]["scene"]["status"] == "unavailable"
    assert document["goal_geometry"]["goal_point_status"] == "unavailable"
    assert document["goal_geometry"]["completion_boundary_status"] == "unavailable"


def test_requested_metric_without_recorded_samples_is_unavailable(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path)
    request = _request(
        tmp_path,
        output="out",
        sources=[{"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"}],
        config={"derive_metrics": False, "requested_metrics": ["clearance"]},
    )

    result = review_panels.run(request, base=tmp_path)

    assert result.status == "complete"
    document = json.loads((tmp_path / "out" / "review-panels.v1.json").read_text())
    assert document["panel_status"]["metrics"] == "unavailable"
    assert document["metrics"]["clearance"]["current"]["reason"] == "metric_not_recorded"


def test_bad_version_unsafe_path_collision_and_cancelled_are_truthful(tmp_path: Path) -> None:
    _copy_fixtures(tmp_path)
    bad_version = _request(tmp_path, config={"min_component_version": "9.0.0"}, output="version")
    assert review_panels.run(bad_version, base=tmp_path).status == "failed"

    unsafe = review_panels.ComponentRequest(
        request_id="unsafe",
        component_id=review_panels.COMPONENT_ID,
        sources=(SourceRef("scene", "../scene.json", "threejs-viewer.v1"),),
        output_directory="unsafe",
    )
    assert review_panels.run(unsafe, base=tmp_path).status == "failed"

    first = review_panels.run(_request(tmp_path, output="collision"), base=tmp_path)
    second = review_panels.run(_request(tmp_path, output="collision"), base=tmp_path)
    assert first.status == "complete"
    assert second.status == "failed"
    assert "output_collision" in second.reason

    cancelled = _request(tmp_path, config={"cancelled": True}, output="cancelled")
    assert review_panels.run(cancelled, base=tmp_path).status == "cancelled"


def test_cli_descriptor_and_fixture_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _copy_fixtures(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
                "request_id": "cli",
                "component_id": review_panels.COMPONENT_ID,
                "sources": [
                    {"artifact_id": "scene", "uri": "scene.json", "format": "threejs-viewer.v1"}
                ],
                "config": {"sample_resolution_s": 0.25},
            }
        ),
        encoding="utf-8",
    )

    assert review_panels.main(["--descriptor"]) == 0
    assert json.loads(capsys.readouterr().out)["component_id"] == review_panels.COMPONENT_ID
    assert (
        review_panels.main(
            ["--input", str(request_path), "--output", "cli-out", "--base", str(tmp_path)]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert (tmp_path / "cli-out" / "review-panels.v1.html").is_file()
