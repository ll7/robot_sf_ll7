"""Focused offline SREV-18 planner/pedestrian diagnostic tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_REQUEST_SCHEMA_VERSION,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.render import review_diagnostics

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "scenario_review" / "review_diagnostics"
)


def _source_ref(
    tmp_path: Path, artifact_id: str, filename: str, format_name: str
) -> dict[str, str]:
    path = tmp_path / filename
    return {
        "artifact_id": artifact_id,
        "uri": filename,
        "format": format_name,
        "schema": format_name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _request(
    tmp_path: Path,
    *,
    sources: list[dict[str, str]] | None = None,
    config: dict[str, object] | None = None,
    output: str = "out",
    required: list[str] | None = None,
) -> review_diagnostics.ComponentRequest:
    payload: dict[str, object] = {
        "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
        "request_id": "srev18-test",
        "component_id": review_diagnostics.COMPONENT_ID,
        "sources": sources
        or [_source_ref(tmp_path, "timeline", "timeline.json", "simulation-timeline.v1")],
        "output_directory": output,
        "config": config or {},
    }
    if required is not None:
        payload["required_capabilities"] = required
    return component_request_from_dict(payload)


def _stage(tmp_path: Path) -> None:
    for name in ("timeline.json", "analysis-trace.json", "diagnosis.json"):
        shutil.copyfile(FIXTURE_ROOT / name, tmp_path / name)


def test_descriptor_is_shared_contract_valid() -> None:
    descriptor = component_descriptor_from_dict(review_diagnostics.descriptor_document())

    assert descriptor.component_id == review_diagnostics.COMPONENT_ID
    assert descriptor.supported_input_versions == (COMPONENT_REQUEST_SCHEMA_VERSION,)
    assert "review-diagnostics.v1" in descriptor.output_types
    assert "evidence-reference-export.v1" in descriptor.output_types
    assert "source-time-cursor" in descriptor.required_capabilities


def test_timeline_fixture_exposes_recorded_values_and_missing_reasons(tmp_path: Path) -> None:
    _stage(tmp_path)
    request = _request(
        tmp_path,
        config={"cursor_time_s": 0.0, "actor_id": "ped-1", "context_revision": 4},
    )

    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    assert model["status"] == "complete"
    assert model["context"]["context_revision"] == 4
    planner = model["panels"]["planner"]
    assert planner["candidates"][1]["candidate_id"] == "fast"
    assert planner["costs"][0]["value"] == 1.2
    assert planner["constraints"]["collision_free"] is True
    controls = model["panels"]["controls"]
    assert controls["commanded"]["value"]["linear_m_s"] == 0.5
    assert controls["executed"]["value"]["linear_m_s"] == 0.4
    assert controls["comparison"]["value"]["linear_m_s"] == pytest.approx(-0.1)
    pedestrians = model["panels"]["pedestrians"]
    assert pedestrians["actors"][0]["actor_id"] == "ped-1"
    assert pedestrians["actors"][0]["diagnostics"]["clearance_m"] == 1.9
    assert pedestrians["diagnostics_state"]["value_origin"] == "simulator_ground_truth"
    assert model["selection_revision"] == 4
    assert all(reference["context_revision"] == 4 for reference in model["evidence_references"])
    assert model["provenance"]["admission"] == "not_evaluated"


def test_analysis_trace_inventory_preserves_amv_commanded_and_applied_controls(
    tmp_path: Path,
) -> None:
    _stage(tmp_path)
    request = _request(
        tmp_path,
        sources=[_source_ref(tmp_path, "analysis", "analysis-trace.json", "analysis-trace.v1")],
        config={"actor_id": "ped-a"},
    )

    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    assert model["inventories"][0]["kind"] == "analysis_trace"
    controls = model["panels"]["controls"]
    assert controls["commanded"]["value"]["linear_m_s"] == 0.25
    assert controls["executed"]["value"]["linear_m_s"] == 0.2
    assert controls["comparison"]["value"]["linear_m_s"] == pytest.approx(-0.05)
    assert controls["commanded"]["value_origin"] == "planner_visible"
    assert controls["executed"]["value_origin"] == "simulator_ground_truth"
    assert controls["comparison"]["value_origin"] == "post_hoc"


def test_optional_diagnosis_stream_is_independent(tmp_path: Path) -> None:
    _stage(tmp_path)
    request = _request(
        tmp_path,
        sources=[
            _source_ref(tmp_path, "timeline", "timeline.json", "simulation-timeline.v1"),
            _source_ref(tmp_path, "diagnosis", "diagnosis.json", "failure_diagnosis.v1"),
        ],
        config={"cursor_time_s": 0.0},
    )

    result = review_diagnostics.run(request, base=tmp_path)
    model = json.loads((tmp_path / "out" / review_diagnostics.OUTPUT_MODEL_FILENAME).read_text())

    assert result.status == "complete"
    assert model["panels"]["failure_diagnosis"]["status"] == "available"
    assert model["panels"]["planner"]["status"] == "available"
    assert (tmp_path / "out" / review_diagnostics.OUTPUT_REFERENCE_FILENAME).is_file()


def test_missing_control_dimensions_remain_unavailable_not_zero(tmp_path: Path) -> None:
    _stage(tmp_path)
    payload = json.loads((tmp_path / "timeline.json").read_text())
    payload["frames"][0]["state"]["controls"] = {"commanded": {"linear_m_s": 0.5}}
    (tmp_path / "timeline.json").write_text(json.dumps(payload))
    request = _request(tmp_path, config={"cursor_time_s": 0.0})
    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    comparison = model["panels"]["controls"]["comparison"]
    assert comparison["status"] == "unavailable"
    assert comparison["missing_reason"] == "commanded_or_executed_control_not_recorded"
    assert "turn_rate_rad_s" not in model["panels"]["controls"]["commanded"]["value"]


def test_control_comparison_requires_recorded_units(tmp_path: Path) -> None:
    _stage(tmp_path)
    payload = json.loads((tmp_path / "analysis-trace.json").read_text())
    payload.pop("units")
    (tmp_path / "analysis-trace.json").write_text(json.dumps(payload))
    request = _request(
        tmp_path,
        sources=[_source_ref(tmp_path, "analysis", "analysis-trace.json", "analysis-trace.v1")],
        config={"actor_id": "ped-a"},
    )
    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)
    comparison = model["panels"]["controls"]["comparison"]
    assert comparison["status"] == "unavailable"
    assert comparison["missing_reason"] == "control_units_not_recorded"


def test_selected_actor_disappearance_is_explicit(tmp_path: Path) -> None:
    _stage(tmp_path)
    request = _request(tmp_path, config={"cursor_time_s": 1.0, "actor_id": "ped-1"})
    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    panel = model["panels"]["pedestrians"]
    assert panel["status"] == "unavailable"
    assert panel["reason"] == "actor_disappeared"
    assert panel["missing_reason"] == "selected_actor_not_present_at_source_time"


@pytest.mark.parametrize("uri", ["/tmp/trace.json", "../trace.json", "trace\\evil.json"])
def test_unsafe_source_paths_fail_closed(tmp_path: Path, uri: str) -> None:
    payload = {
        "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
        "request_id": "unsafe",
        "component_id": review_diagnostics.COMPONENT_ID,
        "sources": [{"artifact_id": "trace", "uri": uri, "format": "analysis-trace.v1"}],
        "output_directory": "out",
    }
    if uri.startswith(("/", "..")):
        with pytest.raises(ReviewContractsValidationError, match="path traversal|absolute path"):
            component_request_from_dict(payload)
        return
    request = component_request_from_dict(payload)
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status in {"failed", "unavailable"}
    assert "unsafe" in result.reason or "source" in result.reason


def test_symlink_source_and_output_collision_fail_closed(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "link.json").symlink_to(tmp_path / "timeline.json")
    request = _request(
        tmp_path,
        sources=[_source_ref(tmp_path, "link", "link.json", "simulation-timeline.v1")],
        output="symlink-out",
    )
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "source" in result.reason

    request = _request(tmp_path, output="collision-out")
    (tmp_path / "collision-out").mkdir()
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason

    (tmp_path / "real-parent").mkdir()
    (tmp_path / "linked-parent").symlink_to(tmp_path / "real-parent", target_is_directory=True)
    request = _request(tmp_path, output="linked-parent/out")
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "unsafe_output_path" in result.reason


def test_unit_mismatch_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    source = _source_ref(tmp_path, "timeline", "timeline.json", "simulation-timeline.v1")
    source["units"] = "feet"
    request = _request(tmp_path, sources=[source])
    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)
    assert model["status"] == "unavailable"
    assert model["inventories"][0]["reason"] == "unit_source_mismatch"


def test_declared_inventory_mismatch_and_stale_revision_fail_closed(tmp_path: Path) -> None:
    _stage(tmp_path)
    source = _source_ref(tmp_path, "timeline", "timeline.json", "analysis-trace.v1")
    request = _request(tmp_path, sources=[source])
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_schema" in result.reason

    request = _request(
        tmp_path,
        config={
            "cursor_time_s": 0.0,
            "context_revision": 2,
            "selection_revision": 3,
        },
        output="stale-out",
    )
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "stale_context_revision" in result.reason


def test_corrupt_source_is_failed_and_diagnostic_only(tmp_path: Path) -> None:
    (tmp_path / "corrupt.json").write_bytes(b'{"schema_version":')
    request = _request(
        tmp_path,
        sources=[_source_ref(tmp_path, "corrupt", "corrupt.json", "analysis-trace.v1")],
        output="corrupt-out",
    )
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "source_corrupt" in result.reason
    assert result.artifacts == ()


def test_missing_source_is_unavailable_not_success(tmp_path: Path) -> None:
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "missing",
                "uri": "missing.json",
                "format": "analysis-trace.v1",
            }
        ],
        output="missing-out",
    )
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "source_missing" in result.reason
    assert result.artifacts == ()


def test_html_embedded_trace_strings_cannot_close_data_script(tmp_path: Path) -> None:
    _stage(tmp_path)
    payload = json.loads((tmp_path / "timeline.json").read_text())
    payload["frames"][0]["state"]["planner_decision"]["candidates"][0]["candidate_id"] = (
        "</script><img src=x onerror=alert(1)>"
    )
    (tmp_path / "timeline.json").write_text(json.dumps(payload))
    request = _request(tmp_path, output="xss-out")
    result = review_diagnostics.run(request, base=tmp_path)
    assert result.status == "complete"
    html = (tmp_path / "xss-out" / review_diagnostics.OUTPUT_HTML_FILENAME).read_text()
    assert "</script><img" not in html
    assert "<\\/script><img" in html


def test_cli_descriptor_and_node_runtime(tmp_path: Path) -> None:
    descriptor = subprocess.run(
        ["python", "-m", "robot_sf.render.review_diagnostics", "--descriptor"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(descriptor.stdout)["component_id"] == review_diagnostics.COMPONENT_ID
    _stage(tmp_path)
    request_path = tmp_path / "request.json"
    payload = json.loads((FIXTURE_ROOT / "request.json").read_text())
    payload["sources"] = [
        _source_ref(tmp_path, "timeline", "timeline.json", "simulation-timeline.v1")
    ]
    payload["output_directory"] = "cli-out"
    request_path.write_text(json.dumps(payload))
    completed = subprocess.run(
        [
            "python",
            "-m",
            "robot_sf.render.review_diagnostics",
            "--input",
            str(request_path),
            "--output",
            "cli-out",
            "--base",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["status"] == "complete"
