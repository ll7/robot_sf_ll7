"""Focused offline SREV-18 planner/pedestrian diagnostic tests."""

from __future__ import annotations

import hashlib
import json
import os
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
from robot_sf.benchmark.analysis_trace import trace_artifact_sha256
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
    diagnosis_panel = model["panels"]["failure_diagnosis"]
    assert diagnosis_panel["status"] == "unavailable"
    assert "source_time_not_recorded" in diagnosis_panel["missing_reasons"]
    assert model["selection_revision"] == 4
    assert all(reference["context_revision"] == 4 for reference in model["evidence_references"])
    for panel_name in ("planner", "controls", "pedestrians"):
        assert model["panels"][panel_name]["context_revision"] == 4
        assert model["panels"][panel_name]["selection_revision"] == 4
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
        config={"cursor_time_s": 0.5, "context_revision": 8},
    )

    result = review_diagnostics.run(request, base=tmp_path)
    model = json.loads((tmp_path / "out" / review_diagnostics.OUTPUT_MODEL_FILENAME).read_text())

    assert result.status == "complete"
    assert model["panels"]["failure_diagnosis"]["status"] == "available"
    assert model["panels"]["planner"]["status"] == "available"
    diagnosis_panel = model["panels"]["failure_diagnosis"]
    record = diagnosis_panel["records"][0]
    assert record["source_artifact_id"] == "diagnosis"
    assert record["source_time_s"] == 0.0
    assert record["actor_id"] == "ped-1"
    assert record["context_revision"] == 8
    assert "units_not_recorded" in record["missing_reasons"]
    assert diagnosis_panel["source_artifact_id"] == "diagnosis"
    assert diagnosis_panel["source_time_s"] == 0.0
    assert diagnosis_panel["actor_id"] == "ped-1"
    assert "units_not_recorded" in diagnosis_panel["missing_reasons"]
    diagnosis_reference = next(
        reference
        for reference in model["evidence_references"]
        if reference["kind"] == "failure_diagnosis"
    )
    assert diagnosis_reference["source_artifact_id"] == "diagnosis"
    assert diagnosis_reference["source_time_s"] == 0.0
    assert diagnosis_reference["actor_id"] == "ped-1"
    assert diagnosis_reference["context_revision"] == 8
    assert diagnosis_reference["missing_reason"] == "units_not_recorded"
    assert (tmp_path / "out" / review_diagnostics.OUTPUT_REFERENCE_FILENAME).is_file()


@pytest.mark.parametrize(
    ("execution_status", "drop_digest", "expected_source_status", "expected_reason"),
    [
        ("fallback", False, "unavailable", "source_execution_fallback"),
        ("degraded", False, "unavailable", "source_execution_degraded"),
        ("failed", False, "unavailable", "source_execution_failed"),
        ("unavailable", False, "unavailable", "source_execution_unavailable"),
        ("partial", False, "partial", "source_execution_partial"),
        ("incomplete", False, "partial", "source_execution_incomplete"),
        (None, True, "partial", "source_integrity_unbound"),
    ],
)
def test_unadmitted_diagnosis_source_is_unavailable_and_not_evidence(
    tmp_path: Path,
    execution_status: str | None,
    drop_digest: bool,
    expected_source_status: str,
    expected_reason: str,
) -> None:
    _stage(tmp_path)
    diagnosis_payload = json.loads((tmp_path / "diagnosis.json").read_text())
    if execution_status is not None:
        diagnosis_payload["execution_status"] = execution_status
    (tmp_path / "diagnosis.json").write_text(json.dumps(diagnosis_payload))
    diagnosis_source = _source_ref(tmp_path, "diagnosis", "diagnosis.json", "failure_diagnosis.v1")
    if drop_digest:
        diagnosis_source.pop("sha256")
    request = _request(
        tmp_path,
        sources=[
            _source_ref(tmp_path, "timeline", "timeline.json", "simulation-timeline.v1"),
            diagnosis_source,
        ],
        config={"cursor_time_s": 0.5},
    )

    result = review_diagnostics.run(request, base=tmp_path)
    model = json.loads((tmp_path / "out" / review_diagnostics.OUTPUT_MODEL_FILENAME).read_text())
    diagnosis_inventory = next(
        inventory for inventory in model["inventories"] if inventory["artifact_id"] == "diagnosis"
    )
    diagnosis_panel = model["panels"]["failure_diagnosis"]

    assert result.status == "partial"
    assert result.artifacts == ()
    assert model["status"] == "partial"
    assert diagnosis_inventory["status"] == expected_source_status
    assert diagnosis_inventory["reason"] == expected_reason
    assert diagnosis_panel["status"] == "unavailable"
    assert diagnosis_panel["reason"] == expected_reason
    assert diagnosis_panel["records"] == []
    assert not any(
        reference["kind"] == "failure_diagnosis" for reference in model["evidence_references"]
    )
    assert model["panels"]["planner"]["status"] == "available"


def test_unadmitted_analysis_trace_does_not_populate_panels_or_evidence(tmp_path: Path) -> None:
    _stage(tmp_path)
    payload = json.loads((tmp_path / "analysis-trace.json").read_text())
    payload["execution_status"] = " fallback "
    payload["artifact_sha256"] = trace_artifact_sha256(payload)
    (tmp_path / "analysis-trace.json").write_text(json.dumps(payload))
    request = _request(
        tmp_path,
        sources=[_source_ref(tmp_path, "analysis", "analysis-trace.json", "analysis-trace.v1")],
        config={"actor_id": "ped-a", "context_revision": 3},
    )

    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    inventory = model["inventories"][0]
    assert model["status"] == "unavailable"
    assert inventory["status"] == "unavailable"
    assert inventory["reason"] == "source_execution_fallback"
    assert model["evidence_references"] == []
    assert model["panels"]["planner"]["status"] == "unavailable"
    assert model["panels"]["planner"]["candidates"] == []
    assert model["panels"]["controls"]["status"] == "unavailable"
    assert model["panels"]["controls"]["commanded"]["status"] == "unavailable"
    assert model["panels"]["pedestrians"]["status"] == "unavailable"
    assert model["panels"]["pedestrians"]["actors"] == []


def test_out_of_range_cursor_does_not_hold_terminal_sample(tmp_path: Path) -> None:
    _stage(tmp_path)
    request = _request(tmp_path, config={"cursor_time_s": 100.0, "context_revision": 9})

    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    assert model["time"]["cursor"]["time_s"] == 100.0
    assert model["time"]["terminal_s"] < 100.0
    assert model["evidence_references"] == []
    for panel_name in ("planner", "controls", "pedestrians"):
        panel = model["panels"][panel_name]
        assert panel["status"] == "unavailable"
        assert panel["reason"] == "outside_declared_resolution"
        assert panel["source_index"] is None


def test_non_finite_source_number_is_rejected_before_build_output(tmp_path: Path) -> None:
    _stage(tmp_path)
    timeline_path = tmp_path / "timeline.json"
    raw = timeline_path.read_text().replace('"position": [0.0, 0.0]', '"position": [1e999, 0.0]')
    timeline_path.write_text(raw)
    request = _request(tmp_path)

    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    assert model["status"] == "failed"
    assert model["inventories"][0]["status"] == "failed"
    assert model["inventories"][0]["reason"] == "source_corrupt: source is not strict UTF-8 JSON"
    json.dumps(model, allow_nan=False)


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


def test_analysis_trace_without_units_is_unavailable_not_partial_controls(tmp_path: Path) -> None:
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
    inventory = model["inventories"][0]
    assert model["status"] == "failed"
    assert inventory["status"] == "failed"
    assert inventory["reason"].startswith("source_schema_invalid")
    assert model["panels"]["controls"]["comparison"]["status"] == "unavailable"


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


def test_retained_source_root_fd_survives_root_rename_and_symlink_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    outside = tmp_path.parent / "review-diagnostics-source-outside"
    outside.mkdir()
    original_root = tmp_path
    moved_root = tmp_path.parent / "review-diagnostics-source-moved"
    request = _request(tmp_path, output="out")
    original_load_sources = review_diagnostics._load_sources

    def move_root_then_load(
        component_request: review_diagnostics.ComponentRequest,
        root: Path,
        *,
        root_fd: int | None = None,
    ) -> list[review_diagnostics._LoadedSource]:
        os.rename(original_root, moved_root)
        original_root.symlink_to(outside, target_is_directory=True)
        return original_load_sources(component_request, root, root_fd=root_fd)

    monkeypatch.setattr(review_diagnostics, "_load_sources", move_root_then_load)
    result = review_diagnostics.run(request, base=original_root)

    assert result.status == "complete"
    assert not (outside / "out").exists()
    assert (moved_root / "out" / review_diagnostics.OUTPUT_MODEL_FILENAME).is_file()


def test_retained_output_fd_survives_root_replacement_without_outside_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    outside = tmp_path.parent / "review-diagnostics-output-outside"
    outside.mkdir()
    original_root = tmp_path
    moved_root = tmp_path.parent / "review-diagnostics-output-moved"
    request = _request(tmp_path, output="nested/out")
    original_build_document = review_diagnostics._build_document

    def replace_root_before_write(
        component_request: review_diagnostics.ComponentRequest,
        sources: list[review_diagnostics._LoadedSource],
        config: dict[str, object],
    ) -> tuple[dict[str, object], list[dict[str, object]], str]:
        os.rename(original_root, moved_root)
        original_root.symlink_to(outside, target_is_directory=True)
        return original_build_document(component_request, sources, config)

    monkeypatch.setattr(review_diagnostics, "_build_document", replace_root_before_write)
    result = review_diagnostics.run(request, base=original_root)

    assert result.status == "complete"
    assert not (outside / "nested").exists()
    assert (moved_root / "nested" / "out" / review_diagnostics.OUTPUT_HTML_FILENAME).is_file()


def test_retained_intermediate_source_fd_survives_rename_and_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    shutil.copyfile(tmp_path / "timeline.json", nested / "timeline.json")
    outside = tmp_path.parent / "review-diagnostics-intermediate-outside"
    outside.mkdir()
    moved_nested = tmp_path / "nested-moved"
    request = _request(
        tmp_path,
        sources=[
            _source_ref(tmp_path, "timeline", "nested/timeline.json", "simulation-timeline.v1")
        ],
    )
    original_load_sources = review_diagnostics._load_sources
    original_open = os.open
    raced = False

    def move_intermediate_after_open(
        component_request: review_diagnostics.ComponentRequest,
        root: Path,
        *,
        root_fd: int | None = None,
    ) -> list[review_diagnostics._LoadedSource]:
        nonlocal raced

        def open_with_race(file: object, flags: int, *args: object, **kwargs: object) -> int:
            nonlocal raced
            descriptor = original_open(file, flags, *args, **kwargs)
            if os.fsdecode(os.fspath(file)) == "nested" and not raced:
                raced = True
                os.rename(nested, moved_nested)
                nested.symlink_to(outside, target_is_directory=True)
            return descriptor

        monkeypatch.setattr(review_diagnostics.os, "open", open_with_race)
        monkeypatch.setattr(
            review_diagnostics.os,
            "supports_dir_fd",
            {*review_diagnostics.os.supports_dir_fd, open_with_race},
        )
        try:
            return original_load_sources(component_request, root, root_fd=root_fd)
        finally:
            monkeypatch.setattr(review_diagnostics.os, "open", original_open)

    monkeypatch.setattr(review_diagnostics, "_load_sources", move_intermediate_after_open)
    result = review_diagnostics.run(request, base=tmp_path)

    assert raced
    assert result.status == "complete"
    assert not (outside / "timeline.json").exists()
    assert (tmp_path / "out" / review_diagnostics.OUTPUT_MODEL_FILENAME).is_file()


def test_retained_output_parent_fd_survives_parent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    outside = tmp_path.parent / "review-diagnostics-output-parent-outside"
    outside.mkdir()
    original_root = tmp_path
    nested = original_root / "nested"
    moved_nested = original_root / "nested-moved"
    request = _request(tmp_path, output="nested/out")
    original_build_document = review_diagnostics._build_document

    def replace_parent_before_write(
        component_request: review_diagnostics.ComponentRequest,
        sources: list[review_diagnostics._LoadedSource],
        config: dict[str, object],
    ) -> tuple[dict[str, object], list[dict[str, object]], str]:
        os.rename(nested, moved_nested)
        nested.symlink_to(outside, target_is_directory=True)
        return original_build_document(component_request, sources, config)

    monkeypatch.setattr(review_diagnostics, "_build_document", replace_parent_before_write)
    result = review_diagnostics.run(request, base=original_root)

    assert result.status == "complete"
    assert not (outside / "out").exists()
    assert (moved_nested / "out" / review_diagnostics.OUTPUT_REFERENCE_FILENAME).is_file()


def test_output_component_boundary_symlink_cannot_escape_descriptor_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    outside = tmp_path.parent / "review-diagnostics-component-outside"
    outside.mkdir()
    request = _request(tmp_path, output="out")
    original_copy = review_diagnostics._copy_web_component

    def symlink_component_boundary(output_dir: review_diagnostics._DirectoryHandle) -> str:
        (output_dir.path / "components").symlink_to(outside, target_is_directory=True)
        return original_copy(output_dir)

    monkeypatch.setattr(review_diagnostics, "_copy_web_component", symlink_component_boundary)
    result = review_diagnostics.run(request, base=tmp_path)

    assert result.status == "failed"
    assert not (outside / "review_diagnostics" / "review_diagnostics.js").exists()


def test_missing_source_sha_is_non_complete_even_when_payload_is_valid(tmp_path: Path) -> None:
    _stage(tmp_path)
    request = _request(
        tmp_path,
        sources=[
            {
                "artifact_id": "timeline",
                "uri": "timeline.json",
                "format": "simulation-timeline.v1",
                "schema": "simulation-timeline.v1",
            }
        ],
    )

    result = review_diagnostics.run(request, base=tmp_path)
    model = json.loads((tmp_path / "out" / review_diagnostics.OUTPUT_MODEL_FILENAME).read_text())

    assert result.status == "partial"
    assert not result.artifacts
    assert model["inventories"][0]["integrity"] == "unbound"
    assert model["inventories"][0]["reason"] == "source_integrity_unbound"


def test_analysis_trace_requires_complete_owner_coverage(tmp_path: Path) -> None:
    _stage(tmp_path)
    payload = json.loads((tmp_path / "analysis-trace.json").read_text())
    payload.pop("artifact_sha256")
    (tmp_path / "analysis-trace.json").write_text(json.dumps(payload))
    request = _request(
        tmp_path,
        sources=[_source_ref(tmp_path, "analysis", "analysis-trace.json", "analysis-trace.v1")],
        config={"actor_id": "ped-a"},
    )

    model = review_diagnostics.build_diagnostic_model(request, base=tmp_path)

    assert model["status"] == "failed"
    assert model["inventories"][0]["status"] == "failed"
    assert "source_schema_invalid" in model["inventories"][0]["reason"]
    assert "artifact_hash" in model["inventories"][0]["reason"]


def test_descriptor_helpers_fail_closed_for_root_and_output_boundaries(tmp_path: Path) -> None:
    _stage(tmp_path)
    assert review_diagnostics._read_under(tmp_path, "timeline.json").startswith(b"{")

    root_handle = review_diagnostics._open_directory(tmp_path)
    try:
        (tmp_path / "directory").mkdir()
        with pytest.raises(ValueError, match="source_not_regular_file"):
            review_diagnostics._read_under(tmp_path, "directory", root_fd=root_handle.fd)
        with pytest.raises(ValueError, match="invalid output component"):
            review_diagnostics._ensure_directory(root_handle, ("nested/file",))
        with pytest.raises(ValueError, match="artifact name"):
            review_diagnostics._write_text(root_handle, "nested/file.txt", "x")
    finally:
        root_handle.close()
        root_handle.close()

    missing_root = tmp_path / "missing-root"
    with pytest.raises(ValueError, match="source_root_missing"):
        review_diagnostics._open_directory(missing_root)
    root_file = tmp_path / "root-file"
    root_file.write_text("x")
    with pytest.raises(ValueError, match="source_root_not_directory"):
        review_diagnostics._open_directory(root_file)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="source_root_not_directory|unsafe_source_path"):
        review_diagnostics._open_directory(root_link)


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


@pytest.mark.parametrize(
    "raw",
    [
        b'{"schema_version":"analysis-trace.v1","schema_version":"analysis-trace.v1"}',
        b'{"schema_version":NaN}',
    ],
)
def test_non_strict_source_json_is_failed_closed(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / "invalid.json").write_bytes(raw)
    request = _request(
        tmp_path,
        sources=[_source_ref(tmp_path, "invalid", "invalid.json", "analysis-trace.v1")],
        output="invalid-out",
    )

    result = review_diagnostics.run(request, base=tmp_path)

    assert result.status == "failed"
    assert result.artifacts == ()
    assert result.reason == "source_corrupt: source is not strict UTF-8 JSON"


@pytest.mark.parametrize("value", [True, "not-a-number", float("inf")])
def test_cursor_time_requires_finite_numeric_value(tmp_path: Path, value: object) -> None:
    _stage(tmp_path)
    request = _request(tmp_path, config={"cursor_time_s": value})

    with pytest.raises(ValueError, match="cursor_time_s must be finite numeric"):
        review_diagnostics.build_diagnostic_model(request, base=tmp_path)


def test_output_writer_rejects_collisions_and_non_strict_json(tmp_path: Path) -> None:
    output = review_diagnostics._open_directory(tmp_path)
    try:
        review_diagnostics._write_text(output, "artifact.txt", "first")
        with pytest.raises(ValueError, match="output_collision: artifact already exists"):
            review_diagnostics._write_text(output, "artifact.txt", "second")
        with pytest.raises(ValueError, match="invalid_output: model is not strict JSON"):
            review_diagnostics._write_json(output, "invalid.json", {"value": float("inf")})
    finally:
        output.close()


def test_source_path_helpers_reject_empty_and_control_paths() -> None:
    assert review_diagnostics._unsafe_path("")
    assert review_diagnostics._unsafe_path("trace\n.json")
    assert review_diagnostics._unsafe_path("C:\\trace.json")
    with pytest.raises(ValueError, match="unsafe_source_path: empty relative path"):
        review_diagnostics._path_parts(".")


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
