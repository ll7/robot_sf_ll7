"""Focused tests for the SREV-05 review-events component (issue #9274)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from robot_sf.analysis_workbench.review_events import (
    COMPONENT_ID,
    descriptor,
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
    (tmp_path / "events.json").write_text(json.dumps(EVENTS), encoding="utf-8")
    (tmp_path / "phases.json").write_text(json.dumps(PHASES), encoding="utf-8")


def _request(
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict | None = None,
    output: str = "out",
    sources: list[dict] | None = None,
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
                {"artifact_id": "events", "uri": "events.json", "format": "event-list"},
                {"artifact_id": "phases", "uri": "phases.json", "format": "phase-list"},
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
    result = run(_request(), base=tmp_path)
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
    assert len(result.artifacts) == 1


def test_deterministic_repeat_runs_match_bytes(tmp_path: Path) -> None:
    _stage(tmp_path)
    first = run(_request(output="out-a"), base=tmp_path)
    second = run(_request(output="out-b"), base=tmp_path)
    assert first.status == second.status == "complete"
    assert (tmp_path / "out-a" / "event-index.json").read_bytes() == (
        tmp_path / "out-b" / "event-index.json"
    ).read_bytes()


def test_missing_links_are_unavailable_not_invented(tmp_path: Path) -> None:
    events = {"intervals": [dict(EVENTS["intervals"][0], precursor_ids=[], recovery_ids=[])]}
    (tmp_path / "events.json").write_text(json.dumps(events), encoding="utf-8")
    (tmp_path / "phases.json").write_text(json.dumps({"intervals": []}), encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status == "partial"
    assert "links_unavailable" in result.reason
    assert result.artifacts == ()


def test_dangling_link_is_partial(tmp_path: Path) -> None:
    events = {
        "intervals": [dict(EVENTS["intervals"][0], recovery_ids=["ghost"])],
    }
    (tmp_path / "events.json").write_text(json.dumps(events), encoding="utf-8")
    (tmp_path / "phases.json").write_text(json.dumps(PHASES), encoding="utf-8")
    result = run(_request(), base=tmp_path)
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
    (tmp_path / "events.json").write_text(json.dumps(events), encoding="utf-8")
    (tmp_path / "phases.json").write_text(json.dumps(PHASES), encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status == "partial"
    assert "outside_range" in result.reason


def test_corrupt_source_is_partial(tmp_path: Path) -> None:
    (tmp_path / "events.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "phases.json").write_text(json.dumps(PHASES), encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status == "partial"
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


def test_output_collision_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_descriptor_declares_capabilities() -> None:
    info = descriptor()
    assert info["component_id"] == COMPONENT_ID
    assert set(info["required_capabilities"]) == {"event-list", "phase-list"}
    assert "predicate-report" in info["optional_capabilities"]


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
        assert payload["status"] == "partial"
        assert "links_unavailable" in payload["reason"]
        index = json.loads((repo / output_rel / "event-index.json").read_text())
        assert len(index["intervals"]) == 3
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)
