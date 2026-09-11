"""Contract tests for the conservative compute-window consumer graph (#8860)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import compute_window_artifact_consumers as tool

FIXTURE = Path(__file__).parent / "fixtures" / "compute_window_artifact_consumers.json"


def test_requested_fixture_graph_and_classes_are_deterministic() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    first = tool.build_graph(payload)
    second = tool.build_graph(payload)
    assert first == second
    assert first["status"] == "failed"
    assert {finding["code"] for finding in first["findings"]} >= {
        "dangling_logical_id",
        "conflicting_identity",
        "supersession_cycle",
        "unknown_dissertation_reference",
        "unknown_release_reference",
        "closed_issue_active_runtime_consumer",
    }
    classes = {row["logical_id"]: row["classification"] for row in first["classifications"]}
    expected = {
        "active-model": "active_required",
        "historical-table": "historical_required",
        "replacement-old": "replacement_verified",
        "regenerable": "regenerable_verified",
        "unknown": "consumer_unknown",
        "orphan": "orphan_candidate",
        "conflict": "unresolved_conflict",
        "cycle-a": "replacement_verified",
        "cycle-b": "replacement_verified",
    }
    assert {key: classes[key] for key in expected} == expected
    assert {edge["type"] for edge in first["edges"]} == set(tool.EDGE_TYPES)
    assert "private-node" in json.dumps(first)


def test_private_projection_is_rejected_without_echoing_locator() -> None:
    payload = {
        "schema": tool.SCHEMA,
        "artifacts": [{"logical_id": "a", "sha256": "a" * 64}],
        "private_projection": [
            {"consumer_id": "private", "locator": "/secret/node", "refs": ["a"]}
        ],
    }
    report = tool.build_graph(payload)
    assert report["ok"] is False
    assert "unsanitized_private_reference" in {item["code"] for item in report["findings"]}
    assert "/secret/node" not in json.dumps(report)


def test_cli_formats_and_check_exit_code(capsys) -> None:
    assert tool.main(["--input", str(FIXTURE), "--check", "--format", "json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == tool.SCHEMA
    assert "digraph consumer_graph" in tool.render_dot(report)
    assert "# Compute-window artifact consumer graph" in tool.render_markdown(report)
