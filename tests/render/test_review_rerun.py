"""Focused tests for the SREV-19 review-rerun component (issue #9289)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    ReviewContractsValidationError,
    component_request_from_dict,
)
from robot_sf.analysis_workbench.simulation_timeline import validate_simulation_timeline
from robot_sf.render import review_rerun
from robot_sf.render.review_rerun import (
    COMPONENT_ID,
    COMPONENT_VERSION,
    descriptor,
    run,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "scenario_review" / "review_rerun"


class _FakeRerun:
    """Offline stand-in for the optional rerun SDK recording calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def init(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("init", args, kwargs))

    def set_time_seconds(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_time_seconds", args, kwargs))

    def set_time_sequence(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("set_time_sequence", args, kwargs))

    def log(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append(("log", args, kwargs))

    def AnyValues(self, **kwargs: Any) -> Any:
        self.calls.append(("AnyValues", (), kwargs))
        return ("any-values", kwargs)

    def Clear(self, *, recursive: bool) -> Any:
        self.calls.append(("Clear", (), {"recursive": recursive}))
        return ("clear", recursive)

    def Points2D(self, points: Any, radii: Any = None) -> Any:
        self.calls.append(("Points2D", (points,), {"radii": radii}))
        return ("points", points, radii)

    def save(self, path: str) -> None:
        self.calls.append(("save", (path,), {}))
        Path(path).write_bytes(b"fake-rrd-bytes")


class _FailingRerun(_FakeRerun):
    """Fake SDK that fails at one explicit adapter boundary."""

    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation

    def init(self, *args: Any, **kwargs: Any) -> None:
        if self.operation == "init":
            raise RuntimeError("init failed")
        super().init(*args, **kwargs)

    def log(self, *args: Any, **kwargs: Any) -> None:
        if self.operation == "log":
            raise RuntimeError("log failed")
        super().log(*args, **kwargs)

    def save(self, path: str) -> None:
        if self.operation == "save":
            raise RuntimeError("save failed")
        super().save(path)


def _request_doc(**overrides: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "component-request.v1",
        "request_id": "srev19-test",
        "component_id": COMPONENT_ID,
        "sources": [
            {
                "artifact_id": "trace-0001",
                "uri": "trace-export.json",
                "format": "simulation_trace_export.v1",
            }
        ],
        "config": {"recording": {"mode": "auto"}},
        "output_directory": "out",
    }
    doc.update(overrides)
    return doc


def _run_request(doc: dict[str, Any], tmp_path: Path, output: str = "out") -> tuple[Any, Path]:
    request = component_request_from_dict({**doc, "output_directory": output})
    result = run(request, base=tmp_path, source_base=FIXTURES)
    return result, tmp_path / output


def test_fixture_smoke_produces_timeline_report_and_descriptor(tmp_path: Path) -> None:
    """The checked-in fixture must yield an offline timeline plus a report."""
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    request = component_request_from_dict({**payload, "output_directory": "smoke"})
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "complete", result.reason
    assert [artifact["artifact_id"] for artifact in result.artifacts] == [
        "trace-0001.inspection-timeline.json",
        "prototype-report.json",
        "component-descriptor.json",
    ]
    timeline = json.loads((tmp_path / "smoke" / "trace-0001.inspection-timeline.json").read_text())
    assert timeline["trace_id"] == "trace-0001"
    assert timeline["counts"] == {"frames": 3, "pedestrian_points": 3}
    assert [frame["time_s"] for frame in timeline["frames"]] == [0.0, 0.5, 1.0]
    assert timeline["frames"][1]["event_id"] == "evt-0001"
    report = json.loads((tmp_path / "smoke" / "prototype-report.json").read_text())
    assert report["prototype"] is True
    assert report["report_sha256"] == result.provenance["report_sha256"]
    assert timeline["evidence_boundary"] == "analysis_workbench_only"
    assert timeline["diagnostic_only"] is True
    assert timeline["admission"] == "not_evaluated"
    assert timeline["source_trace"]["source"]["planner_id"] == "fixture-planner"
    assert (
        timeline["source"]["sha256"]
        == hashlib.sha256((FIXTURES / "trace-export.json").read_bytes()).hexdigest()
    )
    validate_simulation_timeline(timeline["canonical_timeline"])
    assert report["evidence_boundary"] == "analysis_workbench_only"
    assert report["diagnostic_only"] is True
    assert report["admission"] == "not_evaluated"
    assert report["provenance"]["config_sha256"]
    assert descriptor()["component_id"] == COMPONENT_ID


@pytest.mark.parametrize(
    "artifact_id",
    ["../escape", "/absolute/escape", "nested/name", r"nested\\name", "bad\x00id"],
)
def test_source_artifact_ids_are_filename_safe(artifact_id: str) -> None:
    """Source IDs must not become paths or contain filesystem control characters."""
    with pytest.raises(ReviewContractsValidationError, match="unsafe artifact id"):
        component_request_from_dict(
            _request_doc(
                sources=[
                    {
                        "artifact_id": artifact_id,
                        "uri": "trace-export.json",
                        "format": "simulation_trace_export.v1",
                    }
                ]
            )
        )


def test_duplicate_source_artifact_ids_are_rejected() -> None:
    """Two sources cannot claim one physical timeline filename."""
    source = {
        "artifact_id": "same",
        "uri": "trace-export.json",
        "format": "simulation_trace_export.v1",
    }
    with pytest.raises(ReviewContractsValidationError, match="duplicate scoped artifact id"):
        component_request_from_dict(_request_doc(sources=[source, dict(source, uri="other.json")]))


def test_repeated_runs_compare_equal_timeline_bytes(tmp_path: Path) -> None:
    """Deterministic fixture runs must produce byte-identical timelines."""
    first, first_dir = _run_request(_request_doc(), tmp_path, output="run-a")
    second, second_dir = _run_request(_request_doc(), tmp_path, output="run-b")

    assert first.status == "complete", first.reason
    assert second.status == "complete", second.reason
    assert (first_dir / "trace-0001.inspection-timeline.json").read_bytes() == (
        second_dir / "trace-0001.inspection-timeline.json"
    ).read_bytes()
    assert first.provenance["report_sha256"] == second.provenance["report_sha256"]


def test_wrong_component_is_unavailable(tmp_path: Path) -> None:
    """A request for another component must not produce artifacts."""
    result, _ = _run_request(_request_doc(component_id="srev99-other"), tmp_path)

    assert result.status == "unavailable"
    assert "unsupported component" in result.reason
    assert result.artifacts == ()


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


def test_corrupt_recording_mode_is_failed(tmp_path: Path) -> None:
    """Unknown recording modes must fail closed."""
    doc = _request_doc()
    doc["config"] = {"recording": {"mode": "hologram"}}
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "failed"
    assert "corrupt-recording" in result.reason
    assert result.artifacts == ()


def test_explicit_rerun_mode_without_sdk_is_unavailable(tmp_path: Path) -> None:
    """An explicit rerun request cannot be satisfied without the SDK."""
    assert "rerun" not in sys.modules
    doc = _request_doc()
    doc["config"] = {"recording": {"mode": "rerun"}}
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "unavailable"
    assert "rerun-sdk-missing" in result.reason
    assert result.artifacts == ()


def test_rerun_mode_records_with_simulation_time_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK path must log simulation time, geometry, and save a recording."""
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    doc = _request_doc()
    doc["config"] = {"recording": {"mode": "rerun"}}
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "complete", result.reason
    assert (out_dir / "trace-0001.inspection-recording.rrd").is_file()
    time_calls = [call for call in fake.calls if call[0] == "set_time_seconds"]
    assert [call[1] for call in time_calls] == [
        ("time", 0.0),
        ("time", 0.5),
        ("time", 1.0),
    ]
    assert any(call[0] == "save" for call in fake.calls)


def test_rerun_preserves_empty_frames_and_actor_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The optional stream must clear disappeared actors and key actors stably."""
    fake = _FakeRerun()
    monkeypatch.setitem(sys.modules, "rerun", fake)
    source_root = tmp_path / "source"
    source_root.mkdir()
    payload = json.loads((FIXTURES / "trace-export.json").read_text(encoding="utf-8"))
    payload["frames"][1]["pedestrians"] = []
    (source_root / "trace.json").write_text(json.dumps(payload), encoding="utf-8")
    doc = _request_doc(
        sources=[
            {
                "artifact_id": "trace-empty",
                "uri": "trace.json",
                "format": "simulation_trace_export.v1",
            }
        ],
        config={"recording": {"mode": "rerun"}},
    )
    request = component_request_from_dict(doc)

    result = run(request, base=tmp_path, source_base=source_root)

    assert result.status == "complete", result.reason
    timeline = json.loads(
        (tmp_path / "out" / "trace-empty.inspection-timeline.json").read_text(encoding="utf-8")
    )
    assert timeline["frames"][1]["pedestrians"] == []
    log_paths = [call[1][0] for call in fake.calls if call[0] == "log"]
    assert "episode-0001/pedestrians/ped-0000" in log_paths
    clear_logs = [call for call in fake.calls if call[0] == "log" and call[1][1][0] == "clear"]
    assert [call[1][0] for call in clear_logs] == ["episode-0001/pedestrians/ped-0000"]
    actor_metadata = [call for call in fake.calls if call[0] == "AnyValues"]
    assert any(call[2].get("actor_id") == "ped-0000" for call in actor_metadata)


@pytest.mark.parametrize("operation", ["init", "log", "save"])
def test_rerun_sdk_failures_return_failed_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Installed-but-broken SDKs must not escape or publish partial output."""
    monkeypatch.setitem(sys.modules, "rerun", _FailingRerun(operation))
    doc = _request_doc(config={"recording": {"mode": "rerun"}})

    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "failed"
    assert "rerun-recording-failed" in result.reason
    assert result.artifacts == ()
    assert not out_dir.exists()


def test_empty_sources_are_failed(tmp_path: Path) -> None:
    """A request with no evidence cannot be recorded."""
    from robot_sf.analysis_workbench.review_contracts import ComponentRequest

    request = ComponentRequest(
        request_id="srev19-test",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory="out",
        config={"recording": {"mode": "json"}},
    )
    result = run(request, base=tmp_path, source_base=FIXTURES)

    assert result.status == "failed"
    assert "missing-evidence" in result.reason


def test_unusable_sources_are_unavailable(tmp_path: Path) -> None:
    """Sources in unsupported formats must not block, but yield no recording."""
    doc = _request_doc(
        sources=[
            {
                "artifact_id": "video-0000",
                "uri": "trace-export.json",
                "format": "video/mp4",
            }
        ]
    )
    result, _ = _run_request(doc, tmp_path)

    assert result.status == "unavailable"
    assert "unsupported-evidence" in result.reason
    assert result.artifacts == ()


def test_partially_usable_sources_stay_complete_with_diagnostic(
    tmp_path: Path,
) -> None:
    """Usable traces still record; skipped sources are diagnosed, not fatal."""
    doc = _request_doc(
        sources=[
            {
                "artifact_id": "trace-0001",
                "uri": "trace-export.json",
                "format": "simulation_trace_export.v1",
            },
            {
                "artifact_id": "video-0000",
                "uri": "trace-export.json",
                "format": "video/mp4",
            },
        ]
    )
    result, out_dir = _run_request(doc, tmp_path)

    assert result.status == "complete", result.reason
    assert (out_dir / "trace-0001.inspection-timeline.json").is_file()
    diagnosed = [item["artifact_id"] for item in result.diagnostics]
    assert "video-0000" in diagnosed


def test_output_collision_is_failed(tmp_path: Path) -> None:
    """Recordings must never silently overwrite an existing directory."""
    (tmp_path / "out").mkdir()
    result, _ = _run_request(_request_doc(), tmp_path)

    assert result.status == "failed"
    assert "output-collision" in result.reason


def test_cli_smoke_reports_complete(tmp_path: Path, capsys: Any) -> None:
    """The standalone CLI must wire input/config/output through run()."""
    out_dir = tmp_path / "cli-out"
    exit_code = review_rerun.main(
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
    assert '"schema_version": "component-result.v1"' in captured.out
    assert '"status": "complete"' in captured.out
    assert (out_dir / "trace-0001.inspection-timeline.json").is_file()


def test_cli_rejects_non_object_config_without_output(tmp_path: Path) -> None:
    """A parsed list/scalar config must not be silently ignored."""
    config = tmp_path / "bad-config.json"
    config.write_text('["json"]', encoding="utf-8")
    output = tmp_path / "bad-output"

    with pytest.raises(ReviewContractsValidationError, match="config must be a JSON object"):
        review_rerun.main(
            [
                "--input",
                str(FIXTURES / "request.json"),
                "--config",
                str(config),
                "--output",
                "bad-output",
                "--base",
                str(tmp_path),
            ]
        )

    assert not output.exists()


def test_descriptor_matches_contract() -> None:
    """The shipped descriptor must declare this component's exact surface."""
    doc = descriptor()

    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    assert doc["required_capabilities"] == []
    assert doc["output_types"] == ["review-rerun.v1"]
