"""Tests for representative trajectory panel generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from robot_sf.analysis_workbench.simulation_trace_export import load_simulation_trace_export
from robot_sf.benchmark.figure_qa import check_figure_file
from robot_sf.benchmark.trajectory_panels import (
    _robot_displacement,
    generate_trajectory_panel_bundle,
    select_representative_episodes,
)
from scripts.tools.render_trajectory_panels import main

if TYPE_CHECKING:
    from pathlib import Path


def _trace_payload(
    *,
    trace_id: str,
    planner_id: str,
    scenario_id: str,
    episode_id: str,
    event: str,
    final_position: tuple[float, float],
    min_ttc: float | None = None,
) -> dict[str, Any]:
    """Build a tiny valid ``simulation_trace_export.v1`` fixture."""

    planner: dict[str, Any] = {
        "selected_action": {"linear_velocity": 0.1, "angular_velocity": 0.0},
        "event": "start",
    }
    final_planner: dict[str, Any] = {
        "selected_action": {"linear_velocity": 0.1, "angular_velocity": 0.0},
        "event": event,
    }
    if min_ttc is not None:
        final_planner["min_ttc"] = min_ttc
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": trace_id,
        "source": {
            "scenario_id": scenario_id,
            "seed": 7,
            "planner_id": planner_id,
            "episode_id": episode_id,
            "generated_by": "pytest fixture",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {
            "position": "m",
            "heading": "rad",
            "time": "s",
            "velocity": "m/s",
        },
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.1, 0.0]},
                "pedestrians": [{"id": "ped-1", "position": [0.5, 0.25], "velocity": [0.0, 0.0]}],
                "planner": planner,
            },
            {
                "step": 1,
                "time_s": 1.0,
                "robot": {
                    "position": [final_position[0], final_position[1]],
                    "heading": 0.0,
                    "velocity": [0.1, 0.0],
                },
                "pedestrians": [{"id": "ped-1", "position": [0.55, 0.25], "velocity": [0.0, 0.0]}],
                "planner": final_planner,
            },
        ],
    }


def _write_trace(path: Path, payload: dict[str, Any]) -> Path:
    """Write one trace fixture and return its path."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_select_representative_episodes_is_deterministic_by_category(tmp_path: Path) -> None:
    """Selection should choose one stable representative per planner/scenario/category."""

    success = _write_trace(
        tmp_path / "success.json",
        _trace_payload(
            trace_id="trace-success",
            planner_id="planner_a",
            scenario_id="crossing",
            episode_id="ep-success",
            event="goal_reached",
            final_position=(1.0, 0.0),
        ),
    )
    collision = _write_trace(
        tmp_path / "collision.json",
        _trace_payload(
            trace_id="trace-collision",
            planner_id="planner_a",
            scenario_id="crossing",
            episode_id="ep-collision",
            event="collision",
            final_position=(0.2, 0.0),
        ),
    )
    near_miss = _write_trace(
        tmp_path / "near_miss.json",
        _trace_payload(
            trace_id="trace-near-miss",
            planner_id="planner_b",
            scenario_id="overtake",
            episode_id="ep-near-miss",
            event="goal_reached",
            final_position=(1.0, 0.0),
            min_ttc=0.2,
        ),
    )

    selected = select_representative_episodes([near_miss, success, collision])

    assert [(row.category, row.trace.source.episode_id) for row in selected] == [
        ("collision", "ep-collision"),
        ("near_miss", "ep-near-miss"),
        ("success", "ep-success"),
    ]
    assert selected[0].artifact_id == "trajectory_panel_planner_a_crossing_collision_ep-collision"
    assert selected[0].diagnostic_only is True


def test_select_representative_episodes_uses_manual_override_csv(tmp_path: Path) -> None:
    """Manual selection CSVs should preserve reviewer-picked trace order and captions."""

    trace_path = _write_trace(
        tmp_path / "chosen.json",
        _trace_payload(
            trace_id="trace-chosen",
            planner_id="planner_z",
            scenario_id="stress",
            episode_id="ep-chosen",
            event="collision",
            final_position=(0.2, 0.0),
        ),
    )
    override = tmp_path / "selection.csv"
    override.write_text(
        "artifact_id,trace_path,panel_type,category,caption\n"
        "manual_artifact,chosen.json,failure_mosaic,reviewer_bucket,Reviewer selected failure\n",
        encoding="utf-8",
    )

    selected = select_representative_episodes([trace_path], override_csv=override)

    assert len(selected) == 1
    assert selected[0].artifact_id == "manual_artifact"
    assert selected[0].panel_type == "failure_mosaic"
    assert selected[0].category == "reviewer_bucket"
    assert selected[0].caption == "Reviewer selected failure"


def test_boolean_planner_values_do_not_trigger_near_miss(tmp_path: Path) -> None:
    """Boolean planner fields should not be treated as numeric minimum TTC values."""

    trace_path = _write_trace(
        tmp_path / "success.json",
        _trace_payload(
            trace_id="trace-success",
            planner_id="planner_a",
            scenario_id="crossing",
            episode_id="ep-success",
            event="goal_reached",
            final_position=(1.0, 0.0),
            min_ttc=False,
        ),
    )

    selected = select_representative_episodes([trace_path])

    assert selected[0].category == "success"


def test_malformed_robot_position_classifies_as_low_progress(tmp_path: Path) -> None:
    """Malformed robot positions should fail closed instead of raising IndexError/KeyError."""

    payload = _trace_payload(
        trace_id="trace-malformed",
        planner_id="planner_a",
        scenario_id="crossing",
        episode_id="ep-malformed",
        event="advance",
        final_position=(1.0, 0.0),
    )
    trace_path = _write_trace(tmp_path / "malformed.json", payload)
    trace = load_simulation_trace_export(trace_path)
    trace.frames[0].robot.pop("position")

    assert _robot_displacement(trace) == 0.0


def test_generate_trajectory_panel_bundle_writes_visual_artifacts(tmp_path: Path) -> None:
    """Bundle generation should write figures, captions, selection CSV, and manifest."""

    trace_path = _write_trace(
        tmp_path / "collision.json",
        _trace_payload(
            trace_id="trace-collision",
            planner_id="planner_a",
            scenario_id="crossing",
            episode_id="ep-collision",
            event="collision",
            final_position=(0.2, 0.0),
        ),
    )

    bundle = generate_trajectory_panel_bundle(
        [trace_path],
        output_dir=tmp_path / "panels",
        command="pytest fixture",
        commit="abc1234",
    )

    assert bundle.selection_csv.is_file()
    assert bundle.manifest_path.is_file()
    assert bundle.captions_path.is_file()
    assert bundle.artifacts[0].png_path.is_file()
    assert bundle.artifacts[0].pdf_path.is_file()
    assert (
        check_figure_file(
            bundle.artifacts[0].png_path,
            artifact_id=bundle.artifacts[0].artifact_id,
            caption_path=bundle.captions_path,
        )
        == []
    )
    assert (
        check_figure_file(
            bundle.artifacts[0].pdf_path,
            artifact_id=bundle.artifacts[0].artifact_id,
            expected_format="pdf",
            caption_path=bundle.captions_path,
        )
        == []
    )

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "trajectory_panel_manifest.v1"
    assert manifest["artifacts"][0]["source_files"][0]["path"] == str(trace_path)
    assert manifest["artifacts"][0]["claim_boundary"] == "diagnostic_only"
    assert "source_sha256" in manifest["artifacts"][0]["source_files"][0]


def test_cli_generates_bundle_from_trace_paths(tmp_path: Path) -> None:
    """CLI should expose the same bundle generator for automation."""

    trace_path = _write_trace(
        tmp_path / "success.json",
        _trace_payload(
            trace_id="trace-success",
            planner_id="planner_a",
            scenario_id="crossing",
            episode_id="ep-success",
            event="goal_reached",
            final_position=(1.0, 0.0),
        ),
    )
    output_dir = tmp_path / "cli-panels"

    exit_code = main(
        [
            "--trace",
            str(trace_path),
            "--output-dir",
            str(output_dir),
            "--commit",
            "abc1234",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "trajectory_panel_manifest.json").is_file()
    assert (output_dir / "representative_episode_selection.csv").is_file()
