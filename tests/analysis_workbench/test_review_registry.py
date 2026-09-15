"""Focused tests for the SREV-29 review registry (issue #9290)."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    component_descriptor_from_dict,
    component_request_from_dict,
    component_result_from_dict,
)
from robot_sf.analysis_workbench.review_registry import (
    COMPONENT_ID,
    COMPONENT_VERSION,
    check_component_conformance,
    descriptor,
    discover_components,
    run,
    validate_registry_config,
)

FIXTURES = Path("tests/fixtures/scenario_review/review_registry")
TRACE_URI = "tests/fixtures/scenario_review/review_registry/trace.json"


def _stage_fixture_tree(tmp_path: Path) -> Path:
    """Copy the fixture tree under tmp preserving repo-relative layout."""
    target = tmp_path / TRACE_URI
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES / "trace.json", target)
    return tmp_path


def _fixture_request(**config_overrides: Any) -> Any:
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    for key, value in config_overrides.items():
        if value is None:
            config.pop(key, None)
        else:
            config[key] = value
    payload["config"] = config
    request = component_request_from_dict(payload)
    assert request.component_id == COMPONENT_ID
    return request


def _conformance_config(target: str) -> dict[str, Any]:
    return {
        "mode": "conformance",
        "target_component_id": target,
        "conformance_probe": {
            "sources": [
                {
                    "artifact_id": "trace-srev29-smoke",
                    "uri": TRACE_URI,
                    "format": "episode-trace.fixture",
                }
            ],
            "config": {},
        },
    }


def test_descriptor_validates_against_contract_schema() -> None:
    described = component_descriptor_from_dict(descriptor())
    assert described.component_id == COMPONENT_ID
    assert described.component_version == COMPONENT_VERSION
    assert described.required_capabilities == ("bounded-execution",)


def test_validate_registry_config_rejects_unknown_keys() -> None:
    from robot_sf.analysis_workbench.review_registry import ReviewRegistryError

    with pytest.raises(ReviewRegistryError, match="unknown config keys"):
        validate_registry_config({"mode": "execute", "arbitrary_code": "x"})
    with pytest.raises(ReviewRegistryError, match="config must be a mapping"):
        validate_registry_config([])
    with pytest.raises(ReviewRegistryError, match="mode must be one of"):
        validate_registry_config({"mode": "teleport"})
    with pytest.raises(ReviewRegistryError, match="target_component_id"):
        validate_registry_config({"mode": "execute"})
    with pytest.raises(ReviewRegistryError, match="target_config"):
        validate_registry_config({"mode": "discover", "target_config": [], "conformance_probe": {}})
    with pytest.raises(ReviewRegistryError, match="conformance_probe"):
        validate_registry_config({"mode": "discover", "conformance_probe": []})
    with pytest.raises(ReviewRegistryError, match="required_component_version"):
        validate_registry_config({"mode": "discover", "required_component_version": 3})


def test_discovery_lists_both_fixture_components() -> None:
    components, rows = discover_components()
    assert {"srev29-example-analyzer", "srev29-example-renderer"} <= set(components)
    by_id = {row["component_id"]: row for row in rows if row.get("status") == "available"}
    assert by_id["srev29-example-analyzer"]["entry_point"] == "srev29-example-analyzer"


def test_discovery_rejects_conflicting_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from robot_sf.analysis_workbench import review_registry as registry_module
    from robot_sf.analysis_workbench.review_registry import ReviewRegistryError

    entries = [
        SimpleNamespace(
            name="alpha",
            value="examples.scenario_review.components.episode_analyzer:DESCRIPTOR",
        ),
        SimpleNamespace(
            name="beta",
            value="examples.scenario_review.components.episode_analyzer:DESCRIPTOR",
        ),
    ]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    with pytest.raises(ReviewRegistryError, match="conflicting_component_id"):
        discover_components()


def test_discovery_marks_untrusted_and_broken_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from robot_sf.analysis_workbench import review_registry as registry_module

    entries = [
        SimpleNamespace(name="evil", value="os:system"),
        SimpleNamespace(
            name="missing", value="examples.scenario_review.components.nope:DESCRIPTOR"
        ),
    ]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    components, rows = discover_components()
    assert components == {}
    reasons = {row["entry_point"]: row["reason"] for row in rows}
    assert "untrusted" in reasons["evil"]
    assert "unavailable" in reasons["missing"]


def test_execute_analyzer_end_to_end(tmp_path: Path) -> None:
    root = _stage_fixture_tree(tmp_path)
    result = run(_fixture_request(), base=root)
    assert result.status == "complete"
    envelope = component_result_from_dict(
        {
            "schema_version": "component-result.v1",
            "request_id": result.request_id,
            "component_id": result.component_id,
            "status": result.status,
            "artifacts": [dict(entry) for entry in result.artifacts],
            "diagnostics": [dict(entry) for entry in result.diagnostics],
            "provenance": dict(result.provenance),
            "reason": result.reason,
        }
    )
    assert envelope.status == "complete"
    assert [entry["artifact_id"] for entry in result.artifacts] == ["analyzer-summary.json"]
    summary_path = root / result.artifacts[0]["uri"]
    assert summary_path.is_file()
    content = summary_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == result.artifacts[0]["sha256"]
    summary = json.loads(content.decode("utf-8"))
    assert summary["episode_count"] == 2
    assert summary["episodes"][0]["episode_id"] == "episode-0000"
    assert result.provenance["target"]["routed_by"] == COMPONENT_ID


def test_execute_renderer_reports_figure_content(tmp_path: Path) -> None:
    root = _stage_fixture_tree(tmp_path)
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    config["target_component_id"] = "srev29-example-renderer"
    payload["config"] = config
    result = run(component_request_from_dict(payload), base=root)
    assert result.status == "complete"
    by_id = {entry["artifact_id"]: entry for entry in result.artifacts}
    caption = json.loads((root / by_id["renderer-caption.json"]["uri"]).read_bytes())
    assert caption["episode_count"] == 2
    assert caption["figure"]["width"] == 320
    assert caption["figure"]["height"] == 180
    figure_path = root / by_id["telemetry-figure.png"]["uri"]
    assert figure_path.is_file()
    assert (
        hashlib.sha256(figure_path.read_bytes()).hexdigest()
        == by_id["telemetry-figure.png"]["sha256"]
    )


def test_discover_mode_lists_index_artifact(tmp_path: Path) -> None:
    request = _fixture_request(mode="discover", target_component_id=None)
    result = run(request, base=tmp_path)
    assert result.status == "complete"
    assert [entry["artifact_id"] for entry in result.artifacts] == ["registry-index.json"]
    index = json.loads((tmp_path / result.artifacts[0]["uri"]).read_bytes())
    ids = {row.get("component_id") for row in index["components"]}
    assert {"srev29-example-analyzer", "srev29-example-renderer"} <= ids


def test_conformance_mode_passes_for_fixture_components(tmp_path: Path) -> None:
    root = _stage_fixture_tree(tmp_path)
    for target in ("srev29-example-analyzer", "srev29-example-renderer"):
        payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
        payload["config"] = _conformance_config(target)
        payload["output_directory"] = f"srev-29-conformance-{target}"
        result = run(component_request_from_dict(payload), base=root)
        assert result.status == "complete", result.reason
        assert [entry["artifact_id"] for entry in result.artifacts] == ["conformance-report.json"]
        document = json.loads((root / result.artifacts[0]["uri"]).read_bytes())
        assert document["passed"] is True
        assert {check["name"] for check in document["checks"]} >= {
            "descriptor_validates",
            "version_supported",
            "missing_capability_refused",
            "probe_completes",
            "envelope_valid",
            "namespace_contained",
            "output_collision_refused",
            "deterministic_rerun",
        }


def test_conformance_helper_directly(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    root = _stage_fixture_tree(tmp_path)
    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload=dict(episode_analyzer.DESCRIPTOR),
        run_callable=episode_analyzer.run,
        probe_sources=[
            {
                "artifact_id": "trace-srev29-smoke",
                "uri": TRACE_URI,
                "format": "episode-trace.fixture",
            }
        ],
        probe_config={},
        base_dir=root,
        case_name="direct",
        source_root=root,
    )
    assert report.passed is True


def test_conformance_fails_closed_for_broken_descriptor(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload={"schema_version": "component-descriptor.v1"},
        run_callable=episode_analyzer.run,
        probe_sources=[],
        probe_config={},
        base_dir=tmp_path,
        case_name="broken",
    )
    assert report.passed is False
    assert report.checks[-1].name == "descriptor_validates"


def test_unknown_target_is_unavailable(tmp_path: Path) -> None:
    request = _fixture_request(target_component_id="no-such-component")
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "unknown component" in result.reason
    assert result.artifacts == ()


def test_missing_capability_is_unavailable(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    payload["config"] = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    payload["required_capabilities"] = ["video-frames"]
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "unavailable"
    assert "video-frames" in result.reason


def test_incompatible_required_version_is_unavailable(tmp_path: Path) -> None:
    request = _fixture_request(required_component_version="99.0.0")
    result = run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "incompatible_version" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    payload["config"] = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    payload["component_id"] = "no-such-component"
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported component" in result.reason


def test_corrupt_inputs_fail_without_artifacts(tmp_path: Path) -> None:
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    payload["config"] = {"mode": "execute"}
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "failed"
    assert "target_component_id" in result.reason
    assert result.artifacts == ()


def test_output_collision_is_failed(tmp_path: Path) -> None:
    root = _stage_fixture_tree(tmp_path)
    first = run(_fixture_request(), base=root)
    assert first.status == "complete"
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    payload["config"] = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    second = run(component_request_from_dict(copy.deepcopy(payload)), base=root)
    assert second.status == "failed"
    assert "output_collision" in second.reason
    assert second.artifacts == ()


def test_renderer_without_matplotlib_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from examples.scenario_review.components import telemetry_renderer

    root = _stage_fixture_tree(tmp_path)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    request = component_request_from_dict(
        {
            "schema_version": "component-request.v1",
            "request_id": "mpl-missing",
            "component_id": telemetry_renderer.COMPONENT_ID,
            "sources": [
                {
                    "artifact_id": "trace-srev29-smoke",
                    "uri": TRACE_URI,
                    "format": "episode-trace.fixture",
                }
            ],
            "output_directory": "mpl-missing",
        }
    )
    result = telemetry_renderer.run(request, base=root)
    assert result.status == "unavailable"
    assert "matplotlib" in result.reason


def test_cli_rejects_invalid_config_without_execution(tmp_path: Path) -> None:
    from robot_sf.analysis_workbench.review_registry import main

    request_path = tmp_path / "request.json"
    config_path = tmp_path / "config.json"
    request_path.write_text(
        json.dumps(json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    bad_config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
    bad_config["unknown_key"] = True
    config_path.write_text(json.dumps(bad_config), encoding="utf-8")
    code = main(
        [
            "--input",
            str(request_path),
            "--config",
            str(config_path),
            "--output",
            "out",
            "--base",
            str(tmp_path),
        ]
    )
    assert code == 1


def _stub_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    descriptor_payload: dict[str, Any],
    run_impl: Any = None,
) -> Any:
    """Inject a stub component module under an allowlisted entry-point name."""
    import sys
    import types

    module = types.ModuleType(name)
    module.DESCRIPTOR = descriptor_payload
    if run_impl is not None:
        module.run = run_impl
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub_entry(name: str, module: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(name=name, value=f"{module}:DESCRIPTOR")


def _good_descriptor(component_id: str, **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "component-descriptor.v1",
        "component_id": component_id,
        "component_version": "1.0.0",
        "supported_input_versions": ["component-request.v1"],
        "required_capabilities": ["bounded-execution"],
        "optional_capabilities": [],
        "output_types": ["stub.v1"],
    }
    payload.update(overrides)
    return payload


def test_discover_config_target_must_be_string() -> None:
    from robot_sf.analysis_workbench.review_registry import ReviewRegistryError

    with pytest.raises(ReviewRegistryError, match="target_component_id must be a string"):
        validate_registry_config({"mode": "discover", "target_component_id": 3})


def test_commit_provenance_unknown_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as subprocess_module

    from robot_sf.analysis_workbench.review_registry import _commit_provenance

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise OSError("no git today")

    monkeypatch.setattr(subprocess_module, "run", _raise)
    assert _commit_provenance() == {"commit": "unknown"}


def test_entry_point_target_rejects_shapes() -> None:
    from types import SimpleNamespace

    from robot_sf.analysis_workbench.review_registry import _entry_point_target

    assert _entry_point_target(SimpleNamespace(value=123)) is None
    assert _entry_point_target(SimpleNamespace(value="os:system")) is None
    assert _entry_point_target(SimpleNamespace(value="mod:attr:extra")) is None


def test_descriptor_missing_attribute_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_sf.analysis_workbench import review_registry as registry_module

    entries = [
        __import__("types").SimpleNamespace(
            name="bad-attr",
            value="examples.scenario_review.components.episode_analyzer:NONEXISTENT",
        )
    ]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    components, rows = registry_module.discover_components()
    assert components == {}
    assert "no attribute" in rows[0]["reason"]


def test_run_callable_missing_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from robot_sf.analysis_workbench import review_registry as registry_module

    _stub_module(
        monkeypatch, "examples.scenario_review.components.norun", _good_descriptor("stub-norun")
    )
    entries = [_stub_entry("norun", "examples.scenario_review.components.norun")]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    components, rows = registry_module.discover_components()
    assert components == {}
    assert "no run(request) callable" in rows[0]["reason"]


def test_battery_stage_failure_is_reported(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload=dict(episode_analyzer.DESCRIPTOR),
        run_callable=episode_analyzer.run,
        probe_sources=[{"artifact_id": "gone", "uri": "missing.json", "format": "f"}],
        probe_config={},
        base_dir=tmp_path,
        case_name="nostage",
    )
    assert report.passed is False
    assert report.checks[-1].name == "probe_inputs_staged"


def test_battery_reports_raising_component(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    def _raise(request: Any, **kwargs: Any) -> Any:
        refused = _refuse_never(request)
        if refused is not None:
            return refused
        raise RuntimeError("boom")

    root = _stage_fixture_tree(tmp_path)
    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload=dict(episode_analyzer.DESCRIPTOR),
        run_callable=_raise,
        probe_sources=[
            {
                "artifact_id": "trace-srev29-smoke",
                "uri": TRACE_URI,
                "format": "episode-trace.fixture",
            }
        ],
        probe_config={},
        base_dir=root,
        case_name="raiser",
        source_root=root,
    )
    assert report.passed is False
    assert any("raised RuntimeError" in check.detail for check in report.checks)


def test_battery_rejects_unsupported_version(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload=_good_descriptor(
            episode_analyzer.COMPONENT_ID, supported_input_versions=["other-v9"]
        ),
        run_callable=episode_analyzer.run,
        probe_sources=[],
        probe_config={},
        base_dir=tmp_path,
        case_name="oldversion",
    )
    assert report.passed is False
    assert report.checks[-1].name == "version_supported"


def _refuse_never(request: Any) -> Any:
    from robot_sf.analysis_workbench.review_contracts import ComponentResult

    if "__never_supported_capability__" in request.required_capabilities:
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="unavailable",
            reason="missing capabilities: __never_supported_capability__",
        )
    return None


def _envelope_breaker(request: Any, **kwargs: Any) -> Any:
    from robot_sf.analysis_workbench.review_contracts import ComponentResult

    refused = _refuse_never(request)
    if refused is not None:
        return refused
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="complete",
        artifacts=({"artifact_id": "x", "uri": "x.json"},),
    )


def test_battery_rejects_invalid_envelope(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    root = _stage_fixture_tree(tmp_path)
    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload=dict(episode_analyzer.DESCRIPTOR),
        run_callable=_envelope_breaker,
        probe_sources=[
            {
                "artifact_id": "trace-srev29-smoke",
                "uri": TRACE_URI,
                "format": "episode-trace.fixture",
            }
        ],
        probe_config={},
        base_dir=root,
        case_name="badenvelope",
        source_root=root,
    )
    assert report.passed is False
    assert report.checks[-1].name == "envelope_valid"


def _escape_artist(request: Any, **kwargs: Any) -> Any:
    from robot_sf.analysis_workbench.review_contracts import ComponentResult

    refused = _refuse_never(request)
    if refused is not None:
        return refused
    base = Path(kwargs.get("base", "."))
    output_dir = base / str(request.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    outside = output_dir.parent / "escape-target"
    outside.mkdir(parents=True, exist_ok=True)
    (output_dir / "exfil-link").symlink_to(outside, target_is_directory=True)
    return ComponentResult(
        request_id=request.request_id,
        component_id=request.component_id,
        status="complete",
        artifacts=({"artifact_id": "x", "uri": "exfil-link/evil.json", "sha256": "a" * 64},),
    )


def test_battery_rejects_namespace_escape(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    root = _stage_fixture_tree(tmp_path)
    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload=dict(episode_analyzer.DESCRIPTOR),
        run_callable=_escape_artist,
        probe_sources=[
            {
                "artifact_id": "trace-srev29-smoke",
                "uri": TRACE_URI,
                "format": "episode-trace.fixture",
            }
        ],
        probe_config={},
        base_dir=root,
        case_name="escaper",
        source_root=root,
    )
    assert report.passed is False
    assert report.checks[-1].name == "namespace_contained"


def test_battery_detects_divergent_rerun(tmp_path: Path) -> None:
    from examples.scenario_review.components import episode_analyzer

    calls: list[str] = []

    seen: dict[str, int] = {}

    def _flaky(request: Any, **kwargs: Any) -> Any:
        from robot_sf.analysis_workbench.review_contracts import ComponentResult

        refused = _refuse_never(request)
        if refused is not None:
            return refused
        calls.append(str(request.output_directory))
        directory = str(request.output_directory)
        seen[directory] = seen.get(directory, 0) + 1
        if seen[directory] > 1:
            return ComponentResult(
                request_id=request.request_id,
                component_id=request.component_id,
                status="failed",
                reason="output collision, already exists",
            )
        digest = "c" * 64 if "run-b" in directory else "b" * 64
        return ComponentResult(
            request_id=request.request_id,
            component_id=request.component_id,
            status="complete",
            artifacts=({"artifact_id": "a", "uri": "a.json", "sha256": digest},),
        )

    root = _stage_fixture_tree(tmp_path)
    report = check_component_conformance(
        component_id=episode_analyzer.COMPONENT_ID,
        descriptor_payload=dict(episode_analyzer.DESCRIPTOR),
        run_callable=_flaky,
        probe_sources=[
            {
                "artifact_id": "trace-srev29-smoke",
                "uri": TRACE_URI,
                "format": "episode-trace.fixture",
            }
        ],
        probe_config={},
        base_dir=root,
        case_name="flaky",
        source_root=root,
    )
    assert report.passed is False
    assert report.checks[-1].name == "deterministic_rerun"


def test_execute_reports_target_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from robot_sf.analysis_workbench import review_registry as registry_module

    def _raise(request: Any, **kwargs: Any) -> Any:
        raise RuntimeError("target exploded")

    _stub_module(
        monkeypatch,
        "examples.scenario_review.components.exploder",
        _good_descriptor("stub-exploder"),
        _raise,
    )
    entries = [_stub_entry("exploder", "examples.scenario_review.components.exploder")]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    request = _fixture_request(target_component_id="stub-exploder")
    result = registry_module.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "execution_error" in result.reason


def test_execute_rejects_non_result_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from robot_sf.analysis_workbench import review_registry as registry_module

    def _junk(request: Any, **kwargs: Any) -> Any:
        return {"not": "a result"}

    _stub_module(
        monkeypatch,
        "examples.scenario_review.components.junk",
        _good_descriptor("stub-junk"),
        _junk,
    )
    entries = [_stub_entry("junk", "examples.scenario_review.components.junk")]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    request = _fixture_request(target_component_id="stub-junk")
    result = registry_module.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "non-result payload" in result.reason


def test_execute_rejects_incompatible_target_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from robot_sf.analysis_workbench import review_registry as registry_module

    def _ok(request: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not be invoked")

    _stub_module(
        monkeypatch,
        "examples.scenario_review.components.oldie",
        _good_descriptor("stub-oldie", supported_input_versions=["other-v9"]),
        _ok,
    )
    entries = [_stub_entry("oldie", "examples.scenario_review.components.oldie")]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    request = _fixture_request(target_component_id="stub-oldie")
    result = registry_module.run(request, base=tmp_path)
    assert result.status == "unavailable"
    assert "incompatible_version" in result.reason


def test_discover_reports_conflicts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from robot_sf.analysis_workbench import review_registry as registry_module

    entries = [
        SimpleNamespace(
            name="alpha",
            value="examples.scenario_review.components.episode_analyzer:DESCRIPTOR",
        ),
        SimpleNamespace(
            name="beta",
            value="examples.scenario_review.components.episode_analyzer:DESCRIPTOR",
        ),
    ]
    monkeypatch.setattr(registry_module, "_installed_entry_points", lambda: entries)
    request = _fixture_request(mode="discover", target_component_id=None)
    result = registry_module.run(request, base=tmp_path)
    assert result.status == "failed"
    assert "conflicting_component_id" in result.reason


def test_main_read_errors_are_reported(tmp_path: Path) -> None:
    from robot_sf.analysis_workbench.review_registry import ReviewRegistryError, main

    with pytest.raises(ReviewRegistryError, match="cannot read request"):
        main(
            ["--input", str(tmp_path / "missing.json"), "--output", "out", "--base", str(tmp_path)]
        )
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    bad_config = tmp_path / "config.json"
    bad_config.write_text("{nope", encoding="utf-8")
    with pytest.raises(ReviewRegistryError, match="cannot read config"):
        main(
            [
                "--input",
                str(request_path),
                "--config",
                str(bad_config),
                "--output",
                "out",
                "--base",
                str(tmp_path),
            ]
        )
