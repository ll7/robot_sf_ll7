"""Focused tests for the SREV-05 review-events component (issue #9274)."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.analysis_workbench import review_events
from robot_sf.analysis_workbench.review_contracts import (
    component_descriptor_from_dict,
    component_result_from_dict,
)
from robot_sf.analysis_workbench.review_events import (
    COMPONENT_ID,
    descriptor,
    main,
    run,
)

FIXTURE_DIR = "tests/fixtures/scenario_review/review_events"

EVENTS = {
    "intervals": [
        {
            "interval_id": "ev-a",
            "start_s": 0.0,
            "end_s": 0.5,
            "actor_ids": ["robot"],
            "category": "near-miss",
            "metric_value": 0.42,
            "precursor_ids": [],
            "recovery_ids": ["ev-b"],
        },
        {
            "interval_id": "ev-b",
            "start_s": 0.3,
            "end_s": 0.8,
            "actor_ids": ["robot"],
            "category": "recovery",
            "metric_value": 0.91,
            "precursor_ids": ["ev-a"],
            "recovery_ids": [],
        },
    ]
}

PHASES = {
    "intervals": [
        {
            "interval_id": "ph-1",
            "start_s": 0.0,
            "end_s": 1.0,
            "actor_ids": [],
            "category": "nominal",
            "precursor_ids": ["ev-b"],
            "recovery_ids": [],
        }
    ]
}

CONFIG = {"t0_s": 0.0, "terminal_s": 1.0}


def _stage(tmp_path: Path) -> None:
    _write_source(tmp_path / "events.json", EVENTS, "event-list.v1")
    _write_source(tmp_path / "phases.json", PHASES, "phase-list.v1")


def _write_source(path: Path, payload: dict, schema_version: str) -> None:
    document = {**payload, "schema_version": schema_version}
    path.write_text(json.dumps(document), encoding="utf-8")


def _source_ref(artifact_id: str, uri: str, format_name: str, base: Path) -> dict:
    path = base / uri
    reference = {"artifact_id": artifact_id, "uri": uri, "format": format_name}
    if path.is_file():
        reference.update(
            {
                "schema": f"{format_name}.v1",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return reference


def _request(
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict | None = None,
    output: str = "out",
    sources: list[dict] | None = None,
    base: Path | None = None,
) -> dict:
    from robot_sf.analysis_workbench.review_events import (
        component_request_from_dict,
    )

    config = dict(CONFIG)
    config.update(config_extra or {})
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": component_id,
        "sources": (
            sources
            if sources is not None
            else [
                _source_ref("events", "events.json", "event-list", base or Path.cwd()),
                _source_ref("phases", "phases.json", "phase-list", base or Path.cwd()),
            ]
        ),
        "output_directory": output,
        "config": config,
        "required_capabilities": list(required),
    }
    return component_request_from_dict(payload)


def test_success_indexes_overlapping_intervals_with_links(tmp_path: Path) -> None:
    _stage(tmp_path)
    source_before = (tmp_path / "events.json").read_bytes()
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    assert (tmp_path / "events.json").read_bytes() == source_before
    index = json.loads((tmp_path / "out" / "event-index.json").read_text())
    assert [e["interval_id"] for e in index["intervals"]] == ["ev-a", "ph-1", "ev-b"]
    by_id = {e["interval_id"]: e for e in index["intervals"]}
    assert by_id["ev-a"]["recovery_ids"] == ["ev-b"]
    assert by_id["ev-b"]["precursor_ids"] == ["ev-a"]
    assert by_id["ev-a"]["metric_value"] == 0.42
    assert by_id["ev-a"]["category"] == "near-miss"
    assert len(result.artifacts) == 2
    assert index["schema_version"] == "event-index.v1"
    assert result.provenance["source_integrity"] == "digest_and_schema_verified"
    assert (
        result.provenance["sources"][0]["sha256_declared"]
        == result.provenance["sources"][0]["sha256_observed"]
    )
    capability = json.loads((tmp_path / "out" / "missing-capability-report.json").read_text())
    assert capability["missing_capabilities"] == ["predicate-report"]
    assert capability["skipped_optional_streams"] == []
    for artifact in result.artifacts:
        artifact_path = tmp_path / artifact["uri"]
        assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]


def test_deterministic_repeat_runs_match_bytes(tmp_path: Path) -> None:
    _stage(tmp_path)
    first = run(_request(output="out-a", base=tmp_path), base=tmp_path)
    second = run(_request(output="out-b", base=tmp_path), base=tmp_path)
    assert first.status == second.status == "complete"
    assert (tmp_path / "out-a" / "event-index.json").read_bytes() == (
        tmp_path / "out-b" / "event-index.json"
    ).read_bytes()


def test_missing_links_are_unavailable_not_invented(tmp_path: Path) -> None:
    events = {"intervals": [dict(EVENTS["intervals"][0], precursor_ids=[], recovery_ids=[])]}
    _write_source(tmp_path / "events.json", events, "event-list.v1")
    _write_source(tmp_path / "phases.json", {"intervals": []}, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "partial"
    assert "links_unavailable" in result.reason
    assert result.artifacts == ()


def test_interval_id_cannot_hide_blocking_link_diagnostic(tmp_path: Path) -> None:
    events = {
        "intervals": [
            dict(
                EVENTS["intervals"][0],
                interval_id="optional_stream_skipped",
                precursor_ids=[],
                recovery_ids=[],
            )
        ]
    }
    _write_source(tmp_path / "events.json", events, "event-list.v1")
    _write_source(tmp_path / "phases.json", {"intervals": []}, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "partial"
    assert "optional_stream_skipped: links_unavailable" in result.reason
    assert result.artifacts == ()


def test_dangling_link_is_partial(tmp_path: Path) -> None:
    events = {
        "intervals": [dict(EVENTS["intervals"][0], recovery_ids=["ghost"])],
    }
    _write_source(tmp_path / "events.json", events, "event-list.v1")
    _write_source(tmp_path / "phases.json", PHASES, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "partial"
    assert "dangling_links" in result.reason


def test_out_of_range_interval_is_partial(tmp_path: Path) -> None:
    events = {
        "intervals": [
            {
                "interval_id": "ev-far",
                "start_s": 9.0,
                "end_s": 10.0,
                "precursor_ids": [],
                "recovery_ids": [],
            }
        ]
    }
    _write_source(tmp_path / "events.json", events, "event-list.v1")
    _write_source(tmp_path / "phases.json", PHASES, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "partial"
    assert "outside_range" in result.reason


def test_corrupt_required_source_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "events.json").write_text("{not json", encoding="utf-8")
    _write_source(tmp_path / "phases.json", PHASES, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "source_not_json" in result.reason


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(required=("rvo2-binary",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(config_extra={"min_component_version": "2.0.0"}), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_minor_component_version_requirement_is_enforced(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(
        _request(config_extra={"min_component_version": "1.0.1"}, base=tmp_path),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_declared_required_optional_capability_must_be_present(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(required=("predicate-report",), base=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "required_source_family_missing:predicate-report" in result.reason
    assert not (tmp_path / "out").exists()


def test_optional_source_is_reported_as_skipped_when_not_required(tmp_path: Path) -> None:
    _stage(tmp_path)
    _write_source(tmp_path / "predicates.json", {"intervals": []}, "predicate-report.v1")
    sources = [
        _source_ref("events", "events.json", "event-list", tmp_path),
        _source_ref("phases", "phases.json", "phase-list", tmp_path),
        _source_ref("predicates", "predicates.json", "predicate-report", tmp_path),
    ]
    result = run(_request(sources=sources, base=tmp_path), base=tmp_path)
    assert result.status == "complete"
    capability = json.loads((tmp_path / "out" / "missing-capability-report.json").read_text())
    assert capability["missing_capabilities"] == []
    assert capability["skipped_optional_streams"] == ["predicate-report"]


def test_source_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    _stage(tmp_path)
    sources = [
        _source_ref("events", "events.json", "event-list", tmp_path),
        _source_ref("phases", "phases.json", "phase-list", tmp_path),
    ]
    sources[0]["sha256"] = "0" * 64
    result = run(_request(sources=sources, base=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "source_digest_mismatch" in result.reason
    assert not (tmp_path / "out").exists()


def test_declared_source_provenance_is_preserved(tmp_path: Path) -> None:
    _stage(tmp_path)
    sources = [
        _source_ref("events", "events.json", "event-list", tmp_path),
        _source_ref("phases", "phases.json", "phase-list", tmp_path),
    ]
    sources[0].update(
        {
            "source_commit": "a" * 40,
            "config_identity": "fixture-config-v1",
            "units": "seconds",
            "coordinate_frame": "world",
        }
    )
    result = run(_request(sources=sources, base=tmp_path), base=tmp_path)
    assert result.status == "complete"
    source = result.provenance["sources"][0]
    assert source["source_commit"] == "a" * 40
    assert source["config_identity"] == "fixture-config-v1"
    assert source["units"] == "seconds"
    assert source["coordinate_frame"] == "world"


def test_json_array_source_returns_a_failure_result(tmp_path: Path) -> None:
    (tmp_path / "events.json").write_text("[]", encoding="utf-8")
    _write_source(tmp_path / "phases.json", PHASES, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "source_not_json_object" in result.reason
    assert result.diagnostics
    assert not (tmp_path / "out").exists()


def test_source_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside-events.json"
    _write_source(outside, EVENTS, "event-list.v1")
    (root / "events.json").symlink_to(outside)
    _write_source(root / "phases.json", PHASES, "phase-list.v1")
    result = run(_request(base=root), base=root)
    assert result.status == "failed"
    assert "source_unreadable_or_unsafe" in result.reason
    assert not (tmp_path / "outside-output").exists()


def test_output_parent_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _stage(root)
    outside = tmp_path / "outside-output"
    outside.mkdir()
    (root / "redirect").symlink_to(outside, target_is_directory=True)
    result = run(_request(output="redirect/out", base=root), base=root)
    assert result.status == "failed"
    assert "symlink" in result.reason
    assert not (outside / "out").exists()


def test_typed_identity_is_rejected_instead_of_coerced(tmp_path: Path) -> None:
    events = {"intervals": [dict(EVENTS["intervals"][0], actor_ids=[1])]}
    _write_source(tmp_path / "events.json", events, "event-list.v1")
    _write_source(tmp_path / "phases.json", PHASES, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "partial"
    assert "typed_actor_id" in result.reason


def test_conflicting_duplicate_interval_is_not_silently_deduplicated(tmp_path: Path) -> None:
    events = {
        "intervals": [
            *EVENTS["intervals"],
            dict(EVENTS["intervals"][0], category="conflicting-category"),
        ]
    }
    _write_source(tmp_path / "events.json", events, "event-list.v1")
    _write_source(tmp_path / "phases.json", PHASES, "phase-list.v1")
    result = run(_request(base=tmp_path), base=tmp_path)
    assert result.status == "partial"
    assert "conflicting_duplicate_interval_id:ev-a" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_final_artifact_collision_does_not_replace_raced_file(tmp_path: Path, monkeypatch) -> None:
    _stage(tmp_path)
    original_link = review_events.os.link
    target = tmp_path / "out" / "event-index.json"
    raced = False

    def create_raced_target(source, destination, *args, **kwargs):
        nonlocal raced
        if not raced and Path(destination) == target:
            target.write_text("raced-in", encoding="utf-8")
            raced = True
        return original_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(review_events.os, "link", create_raced_target)
    result = run(_request(base=tmp_path), base=tmp_path)
    assert raced
    assert result.status == "failed"
    assert "output_collision" in result.reason
    assert target.read_text(encoding="utf-8") == "raced-in"


def test_source_read_rejects_swap_to_symlink_after_resolution(tmp_path: Path, monkeypatch) -> None:
    _stage(tmp_path)
    outside = tmp_path / "outside-events.json"
    _write_source(outside, EVENTS, "event-list.v1")
    source = tmp_path / "events.json"
    original_open = review_events.os.open
    swapped = False

    def swap_before_root_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if not swapped and Path(path) == tmp_path and dir_fd is None:
            source.unlink()
            source.symlink_to(outside)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(review_events.os, "open", swap_before_root_open)
    result = run(_request(base=tmp_path), base=tmp_path)
    assert swapped
    assert result.status == "failed"
    assert "source_unreadable_or_unsafe" in result.reason
    assert "source_digest_mismatch" not in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_declares_capabilities() -> None:
    info = descriptor()
    component_descriptor_from_dict(info)
    assert info["schema_version"] == "component-descriptor.v1"
    assert info["component_id"] == COMPONENT_ID
    assert set(info["required_capabilities"]) == {"event-list", "phase-list"}
    assert "predicate-report" in info["optional_capabilities"]


def test_raw_malformed_request_returns_failure_result() -> None:
    result = run({})
    assert result.status == "failed"
    assert result.request_id == "unknown"
    assert result.component_id == COMPONENT_ID
    assert result.reason == "invalid_request: expected validated ComponentRequest"


def test_module_touches_no_simulator_paths() -> None:
    """The adapter must stay observational: no sim/planner imports."""
    tree = ast.parse(
        (
            Path(__file__).resolve().parents[2] / "robot_sf/analysis_workbench/review_events.py"
        ).read_bytes()
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    imported -= {"robot_sf", "__future__", "typing"}
    assert not (imported & {"sim", "planner", "training", "torch"}), imported


def test_cli_produces_index_from_fixture_request() -> None:
    import shutil

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / FIXTURE_DIR
    assert (fixture / "request.json").exists()
    output_rel = "output/scenario_review/srev-05-cli-test"
    shutil.rmtree(repo / output_rel, ignore_errors=True)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_sf.analysis_workbench.review_events",
                "--input",
                str(fixture / "request.json"),
                "--config",
                str(fixture / "config.json"),
                "--output",
                output_rel,
            ],
            capture_output=True,
            text=True,
            cwd=repo,
            check=False,
        )
        # The gap-exercising fixture yields partial with the explicit code.
        assert completed.returncode == 1, completed.stderr[-2000:]
        payload = json.loads(completed.stdout)
        component_result_from_dict(payload)
        assert payload["schema_version"] == "component-result.v1"
        assert payload["status"] == "partial"
        assert "links_unavailable" in payload["reason"]
        index = json.loads((repo / output_rel / "event-index.json").read_text())
        assert len(index["intervals"]) == 3
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)


def test_cli_malformed_json_emits_component_result_envelope(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{not json", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.analysis_workbench.review_events",
            "--input",
            str(request_path),
            "--output",
            "out",
            "--base",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
    )
    assert completed.returncode == 1, completed.stderr[-2000:]
    payload = json.loads(completed.stdout)
    component_result_from_dict(payload)
    assert payload["schema_version"] == "component-result.v1"
    assert payload["status"] == "failed"
    assert "JSONDecodeError" in payload["reason"]


def test_cli_main_fixture_request_in_process(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The in-process fixture run mirrors the subprocess CLI contract."""
    import shutil

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / FIXTURE_DIR
    output_rel = "output/scenario_review/srev-05-cli-in-process-test"
    shutil.rmtree(repo / output_rel, ignore_errors=True)
    try:
        assert (
            main(
                [
                    "--input",
                    str(fixture / "request.json"),
                    "--config",
                    str(fixture / "config.json"),
                    "--output",
                    output_rel,
                ]
            )
            == 1
        )
        payload = json.loads(capsys.readouterr().out)
        component_result_from_dict(payload)
        assert payload["status"] == "partial"
        assert "links_unavailable" in payload["reason"]
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)


def test_cli_main_malformed_json_in_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Malformed request JSON fails closed through the in-process entry."""
    request_path = tmp_path / "request.json"
    request_path.write_text("{not json", encoding="utf-8")
    assert main(["--input", str(request_path), "--output", "out", "--base", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    component_result_from_dict(payload)
    assert payload["status"] == "failed"
    assert "JSONDecodeError" in payload["reason"]


def test_cli_main_non_object_request_in_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A JSON-array request fails closed through the in-process entry."""
    request_path = tmp_path / "request.json"
    request_path.write_text("[]", encoding="utf-8")
    assert main(["--input", str(request_path), "--output", "out", "--base", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    component_result_from_dict(payload)
    assert payload["status"] == "failed"
    assert "request must be a JSON object" in payload["reason"]
