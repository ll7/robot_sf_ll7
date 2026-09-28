"""Focused tests for the SREV-08 review-alignment component (issue #9277)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench import review_alignment
from robot_sf.analysis_workbench.review_alignment import (
    COMPONENT_ID,
    COMPONENT_VERSION,
    descriptor,
    run,
)
from robot_sf.analysis_workbench.review_contracts import component_request_from_dict

FIXTURES = Path(__file__).parents[1] / "fixtures" / "scenario_review" / "review_alignment"


def _alignment(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "left_artifact_id": "trace-a",
        "right_artifact_id": "trace-b",
        "comparison_grain": "matched_planner_pair",
        "anchor": {"type": "event", "event_id": "evt-0001"},
    }
    doc.update(overrides)
    return doc


def _request_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "component-request.v1",
        "request_id": "srev08-test",
        "component_id": COMPONENT_ID,
        "sources": [
            {
                "artifact_id": "trace-a",
                "uri": "trace-a.json",
                "format": "simulation_trace_export.v1",
            },
            {
                "artifact_id": "trace-b",
                "uri": "trace-b.json",
                "format": "simulation_trace_export.v1",
            },
        ],
        "config": {"alignment": _alignment()},
        "output_directory": "out",
    }
    doc.update(overrides)
    return doc


def _trace_source(artifact_id: str, uri: str, file_format: str = "simulation_trace_export.v1"):
    return {"artifact_id": artifact_id, "uri": uri, "format": file_format}


def _run_request(doc: dict[str, Any], tmp_path: Path, output: str = "out") -> tuple[Any, Path]:
    request = component_request_from_dict({**doc, "output_directory": output})
    result = run(request, base=tmp_path, source_base=FIXTURES)
    return result, tmp_path / output


def test_fixture_smoke_produces_admissible_alignment(tmp_path: Path) -> None:
    """The checked-in fixture must yield a compatible, anchored comparison."""
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    request = component_request_from_dict({**payload, "output_directory": "smoke"})
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "complete", result.reason
    assert [artifact["artifact_id"] for artifact in result.artifacts] == [
        "alignment.json",
        "component-descriptor.json",
    ]
    alignment = json.loads((tmp_path / "smoke" / "alignment.json").read_text())
    assert alignment["compatibility"]["status"] == "available"
    assert alignment["anchor"] == {
        "type": "event",
        "event_id": "evt-0001",
        "status": "available",
    }
    assert alignment["interpretation"]["admissible"] is True
    assert alignment["durations_s"] == {"left": 1.0, "right": 1.0}
    assert len(alignment["compatibility"]["valid_common_event_anchors"]) == 2
    assert alignment["alignment_sha256"] == result.provenance["alignment_sha256"]
    assert descriptor()["component_id"] == COMPONENT_ID


def test_repeated_runs_compare_equal_alignment_bytes(tmp_path: Path) -> None:
    """Deterministic fixture runs must produce byte-identical alignments."""
    first, first_dir = _run_request(_request_doc(), tmp_path, output="run-a")
    second, second_dir = _run_request(_request_doc(), tmp_path, output="run-b")

    assert first.status == "complete", first.reason
    assert second.status == "complete", second.reason
    assert (first_dir / "alignment.json").read_bytes() == (
        second_dir / "alignment.json"
    ).read_bytes()


def test_incompatible_seeds_stay_visible_not_silent(tmp_path: Path) -> None:
    """Different seeds alone must not establish equivalence (no silent pass)."""
    doc = _request_doc(
        sources=[
            _trace_source("trace-a", "trace-a.json"),
            _trace_source("trace-c", "trace-c.json"),
        ],
        config={"alignment": _alignment(right_artifact_id="trace-c")},
    )
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "complete", result.reason
    alignment = json.loads((out_dir / "alignment.json").read_text(encoding="utf-8"))
    assert alignment["compatibility"]["status"] == "incompatible"
    assert alignment["interpretation"]["admissible"] is False
    assert alignment["interpretation"]["reasons"] != []


def test_missing_anchor_event_stays_visible(tmp_path: Path) -> None:
    """An anchor no side reports must be diagnosed, not invented."""
    doc = _request_doc()
    doc["config"] = {
        **doc["config"],
        "alignment": _alignment(anchor={"type": "event", "event_id": "evt-nope"}),
    }
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "complete", result.reason
    alignment = json.loads((out_dir / "alignment.json").read_text(encoding="utf-8"))
    assert alignment["anchor"]["status"] == "unavailable"
    assert alignment["interpretation"]["admissible"] is False


def test_out_of_range_absolute_time_anchor_stays_visible(tmp_path: Path) -> None:
    """An anchor outside either trace span must be diagnosed with spans."""
    doc = _request_doc()
    doc["config"] = {
        **doc["config"],
        "alignment": _alignment(anchor={"type": "absolute-time", "time_s": 99.0}),
    }
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "complete", result.reason
    alignment = json.loads((out_dir / "alignment.json").read_text(encoding="utf-8"))
    assert alignment["anchor"]["status"] == "unavailable"
    assert "spans_s" in alignment["anchor"]
    assert alignment["interpretation"]["admissible"] is False


def test_unequal_durations_are_reported_without_normalization(tmp_path: Path) -> None:
    """Duration mismatch must appear in the report, never be scaled away."""
    doc = _request_doc(
        sources=[
            _trace_source("trace-a", "trace-a.json"),
            _trace_source("trace-d", "trace-d.json"),
        ],
        config={"alignment": _alignment(right_artifact_id="trace-d")},
    )
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "complete", result.reason
    alignment = json.loads((out_dir / "alignment.json").read_text(encoding="utf-8"))
    assert alignment["durations_s"] == {"left": 1.0, "right": 1.5}
    assert any(
        "without normalization" in reason for reason in alignment["interpretation"]["reasons"]
    )
    assert alignment["interpretation"]["admissible"] is False


def test_unknown_grain_is_unavailable(tmp_path: Path) -> None:
    """Grains outside the owner vocabulary must not run."""
    doc = _request_doc()
    doc["config"] = {
        **doc["config"],
        "alignment": _alignment(comparison_grain="matched_vibes_pair"),
    }
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "unavailable"
    assert "unsupported-comparison-grain" in result.reason
    assert result.artifacts == ()


def test_same_artifact_both_sides_is_failed(tmp_path: Path) -> None:
    """Comparing a trace with itself is a corrupt request, not a comparison."""
    doc = _request_doc()
    doc["config"] = {
        **doc["config"],
        "alignment": _alignment(right_artifact_id="trace-a"),
    }
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "failed"
    assert "must differ" in result.reason


def test_wrong_component_is_unavailable(tmp_path: Path) -> None:
    """A request for another component must not produce artifacts."""
    result, _ = _run_request(_request_doc(component_id="srev99-other"), tmp_path)

    assert result.status == "unavailable"
    assert "unsupported component" in result.reason


def test_missing_capability_is_unavailable(tmp_path: Path) -> None:
    """Required capabilities outside the descriptor must stay unavailable."""
    result, _ = _run_request(_request_doc(required_capabilities=["video-overlay"]), tmp_path)

    assert result.status == "unavailable"
    assert "missing capabilities" in result.reason


def test_incompatible_required_version_is_unavailable(tmp_path: Path) -> None:
    """A newer required major version must not run against this component."""
    doc = _request_doc()
    doc["config"] = {**doc["config"], "required_component_version": "2.0.0"}
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "unavailable"
    assert "incompatible-required-version" in result.reason


def test_missing_named_side_is_failed(tmp_path: Path) -> None:
    """A named artifact with no usable source cannot be compared."""
    doc = _request_doc()
    doc["config"] = {
        **doc["config"],
        "alignment": _alignment(right_artifact_id="trace-missing"),
    }
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "failed"
    assert "missing-evidence" in result.reason


def test_unresolvable_named_sides_are_failed(tmp_path: Path) -> None:
    """Named sides that resolve to nothing cannot be compared."""
    doc = _request_doc(
        sources=[_trace_source("video-0000", "trace-a.json", "video/mp4")],
        config={
            "alignment": _alignment(left_artifact_id="video-0000", right_artifact_id="video-0001")
        },
    )
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "failed"
    assert "missing-evidence" in result.reason


def test_extra_skipped_sources_keep_partial_status(tmp_path: Path) -> None:
    """Usable pairs still align; skipped extra sources are diagnosed."""
    doc = _request_doc(
        sources=[
            _trace_source("trace-a", "trace-a.json"),
            _trace_source("trace-b", "trace-b.json"),
            _trace_source("video-0000", "trace-a.json", "video/mp4"),
        ]
    )
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "partial", result.reason
    assert (out_dir / "alignment.json").is_file()
    assert [item["artifact_id"] for item in result.diagnostics] == ["video-0000"]


def test_output_collision_is_failed(tmp_path: Path) -> None:
    """Alignments must never silently overwrite an existing directory."""
    (tmp_path / "out").mkdir()
    result, _ = _run_request(_request_doc(), tmp_path)

    assert result.status == "failed"
    assert "output-collision" in result.reason


def test_cli_smoke_reports_complete(tmp_path: Path, capsys: Any) -> None:
    """The standalone CLI must wire input/config/output through run()."""
    out_dir = tmp_path / "cli-out"
    exit_code = review_alignment.main(
        [
            "--input",
            str(FIXTURES / "request.json"),
            "--config",
            str(FIXTURES / "config.json"),
            "--output",
            "cli-out",
            "--base",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"status": "complete"' in captured.out
    assert (out_dir / "alignment.json").is_file()


def test_descriptor_matches_contract() -> None:
    """The shipped descriptor must declare this component's exact surface."""
    doc = descriptor()

    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    assert doc["required_capabilities"] == []
    assert doc["output_types"] == ["review-alignment.v1"]
