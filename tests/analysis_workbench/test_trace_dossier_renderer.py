"""Tests for diagnostic multi-panel trace dossier rendering."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.trace_dossier_renderer import (
    TRACE_DOSSIER_MANIFEST_SCHEMA_FILE,
    TraceDossierRenderError,
    load_trace_dossier_manifest_schema,
    render_trace_dossier,
)
from robot_sf.benchmark.figure_qa import check_figure_file

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "simulation_trace_export_v1"
    / "minimal_trace.json"
)
CLI_PATH = Path(__file__).resolve().parents[2] / "scripts" / "tools" / "render_trace_dossier.py"


def test_trace_dossier_renderer_writes_four_panel_png_and_manifest(tmp_path: Path) -> None:
    """The pinned fixture should render to a diagnostic-only PNG and manifest."""

    png_path = tmp_path / "dossier.png"
    manifest_path = tmp_path / "trace_dossier_manifest.json"

    result = render_trace_dossier(
        FIXTURE_PATH,
        output_png=png_path,
        manifest_path=manifest_path,
        command="pytest fixture",
    )

    assert result.png_path == png_path
    assert check_figure_file(png_path, artifact_id="trace_dossier") == []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    Draft202012Validator(load_trace_dossier_manifest_schema()).validate(manifest)
    assert manifest["schema_version"] == "trace_dossier_manifest.v1"
    assert manifest["trace_schema_version"] == "simulation_trace_export.v1"
    assert manifest["trace_id"] == "fixture_trace_001"
    assert manifest["panels"] == [
        "trajectory",
        "speed_profile",
        "clearance_over_time",
        "event_timeline",
    ]
    assert manifest["clearance_semantics"]["mode"] == "center_distance_m"
    assert manifest["clearance_semantics"]["minimum_clearance"]["pedestrian_id"] == "ped_1"
    assert manifest["evidence_boundary"] == "diagnostic_only"
    assert "not body-edge clearance" in " ".join(manifest["limitations"])


def test_trace_dossier_renderer_repeat_is_byte_deterministic(tmp_path: Path) -> None:
    """Repeated renders of the same source should produce identical PNG and semantic manifest."""

    first_png = tmp_path / "first.png"
    second_png = tmp_path / "second.png"
    first_manifest = tmp_path / "first_manifest.json"
    second_manifest = tmp_path / "second_manifest.json"

    render_trace_dossier(
        FIXTURE_PATH,
        output_png=first_png,
        manifest_path=first_manifest,
        command="pytest fixture",
    )
    render_trace_dossier(
        FIXTURE_PATH,
        output_png=second_png,
        manifest_path=second_manifest,
        command="pytest fixture",
    )

    assert first_png.read_bytes() == second_png.read_bytes()
    first_payload = json.loads(first_manifest.read_text(encoding="utf-8"))
    second_payload = json.loads(second_manifest.read_text(encoding="utf-8"))
    first_payload["outputs"]["png"]["path"] = "<normalized>"
    second_payload["outputs"]["png"]["path"] = "<normalized>"
    assert first_payload == second_payload


def test_trace_dossier_renderer_fails_closed_without_events(tmp_path: Path) -> None:
    """The event timeline should not invent missing planner event annotations."""

    payload = _fixture_payload()
    del payload["frames"][0]["planner"]["event"]
    trace_path = tmp_path / "missing_event.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceDossierRenderError, match="planner.event"):
        render_trace_dossier(
            trace_path,
            output_png=tmp_path / "out.png",
            manifest_path=tmp_path / "manifest.json",
            command="pytest fixture",
        )


def test_trace_dossier_renderer_fails_closed_without_clearance_actor(tmp_path: Path) -> None:
    """The clearance panel should reject traces without pedestrian geometry."""

    payload = _fixture_payload()
    for frame in payload["frames"]:
        frame["pedestrians"] = []
    trace_path = tmp_path / "no_pedestrians.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceDossierRenderError, match="requires at least one pedestrian"):
        render_trace_dossier(
            trace_path,
            output_png=tmp_path / "out.png",
            manifest_path=tmp_path / "manifest.json",
            command="pytest fixture",
        )


def test_trace_dossier_renderer_rejects_mixed_radius_metadata(tmp_path: Path) -> None:
    """Body-edge clearance requires complete radius metadata for every actor."""

    payload = _fixture_payload()
    payload["frames"][0]["robot"]["radius"] = 0.2
    trace_path = tmp_path / "mixed_radius.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceDossierRenderError, match="cannot mix actor radius"):
        render_trace_dossier(
            trace_path,
            output_png=tmp_path / "out.png",
            manifest_path=tmp_path / "manifest.json",
            command="pytest fixture",
        )


def test_trace_dossier_renderer_rejects_out_of_range_numeric_value(tmp_path: Path) -> None:
    """Oversized finite JSON numbers fail closed instead of escaping as OverflowError."""

    payload = _fixture_payload()
    payload["frames"][0]["robot"]["position"][0] = 10**3000
    trace_path = tmp_path / "out_of_range.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceDossierRenderError, match="finite number|float range"):
        render_trace_dossier(
            trace_path,
            output_png=tmp_path / "out.png",
            manifest_path=tmp_path / "manifest.json",
            command="pytest fixture",
        )


def test_trace_dossier_renderer_rejects_out_of_range_selected_action(tmp_path: Path) -> None:
    """Oversized selected-action values fail closed during speed-panel preparation."""

    payload = _fixture_payload()
    payload["frames"][0]["planner"]["selected_action"]["linear_velocity"] = 10**3000
    trace_path = tmp_path / "out_of_range_action.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceDossierRenderError, match="finite number|float range"):
        render_trace_dossier(
            trace_path,
            output_png=tmp_path / "out.png",
            manifest_path=tmp_path / "manifest.json",
            command="pytest fixture",
        )


def test_trace_dossier_renderer_rejects_overflowed_edge_clearance(tmp_path: Path) -> None:
    """Overflowed radius subtraction cannot become a non-finite manifest value."""

    payload = _fixture_payload()
    for frame in payload["frames"]:
        frame["robot"]["radius"] = 1.0e308
        for pedestrian in frame["pedestrians"]:
            pedestrian["radius"] = 1.0e308
    trace_path = tmp_path / "overflowed_clearance.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TraceDossierRenderError, match="non-finite value"):
        render_trace_dossier(
            trace_path,
            output_png=tmp_path / "out.png",
            manifest_path=tmp_path / "manifest.json",
            command="pytest fixture",
        )


def test_trace_dossier_manifest_schema_rejects_benchmark_boundary() -> None:
    """The manifest schema should preserve the diagnostic-only boundary."""

    schema = json.loads(TRACE_DOSSIER_MANIFEST_SCHEMA_FILE.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "trace_dossier_manifest.v1",
        "trace_schema_version": "simulation_trace_export.v1",
        "trace_id": "fixture_trace_001",
        "source_trace": {"path": "trace.json", "sha256": "0" * 64},
        "evidence_boundary": "benchmark_evidence",
        "renderer": {
            "name": "trace_dossier_renderer",
            "version": "issue_7086.v1",
            "command": "pytest fixture",
        },
        "outputs": {"png": {"path": "dossier.png", "sha256": "1" * 64}},
        "panels": [
            "trajectory",
            "speed_profile",
            "clearance_over_time",
            "event_timeline",
        ],
        "clearance_semantics": {
            "mode": "center_distance_m",
            "units": "m",
            "minimum_clearance": {
                "time_s": 0.0,
                "step": 0,
                "pedestrian_id": "ped_1",
                "value_m": 1.0,
            },
        },
        "limitations": ["diagnostic-only"],
    }

    errors = list(Draft202012Validator(schema).iter_errors(manifest))

    assert any(
        list(error.absolute_path) == ["evidence_boundary"] and error.validator == "const"
        for error in errors
    )


def test_render_trace_dossier_cli_smoke(tmp_path: Path) -> None:
    """The CLI should render the pinned fixture through the public script path."""

    png_path = tmp_path / "cli_dossier.png"
    manifest_path = tmp_path / "cli_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CLI_PATH),
            "--trace",
            str(FIXTURE_PATH),
            "--output",
            str(png_path),
            "--manifest",
            str(manifest_path),
            "--command",
            "pytest cli fixture",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert png_path.exists()
    assert manifest_path.exists()


def _fixture_payload() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
