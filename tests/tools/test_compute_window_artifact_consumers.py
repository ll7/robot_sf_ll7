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
        "replacement-old": "unresolved_conflict",
        "regenerable": "regenerable_verified",
        "unknown": "consumer_unknown",
        "orphan": "orphan_candidate",
        "conflict": "unresolved_conflict",
        "cycle-a": "unresolved_conflict",
        "cycle-b": "unresolved_conflict",
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


def test_private_artifact_metadata_is_redacted_from_all_formats() -> None:
    private_values = {
        "path": "/private/locator",
        "command": "srun --secret token",
        "host": "gpu.internal.example.com",
        "url": "https://private.example.com/object",
        "environment": "PRIVATE_VALUE",
    }
    report = tool.build_graph(
        {"schema": tool.SCHEMA, "artifacts": [{"logical_id": "a", **private_values}]}
    )
    rendered = tool.render_json(report) + tool.render_dot(report) + tool.render_markdown(report)
    assert report["ok"] is False
    assert "redacted_private_field" in {item["code"] for item in report["findings"]}
    assert all(value not in rendered for value in private_values.values())
    assert report["artifacts"][0]["path"] is None


def test_malformed_refs_and_record_metadata_are_structured_and_deterministic() -> None:
    payload = {
        "schema": tool.SCHEMA,
        "artifacts": [{"logical_id": "a", "replacement_for": 3, "path": "../escape"}],
        "consumers": [
            {"id": "bad-edge", "kind": "config", "refs": [{"logical_id": "a", "edge": []}]},
            {"id": "bad-ref", "kind": "config", "refs": [None, {"logical_id": 4}]},
            {"id": "bad-state", "kind": "config", "state": {"nested": True}, "refs": "a"},
        ],
    }
    first = tool.build_graph(payload)
    second = tool.build_graph(payload)
    assert first == second and first["ok"] is False
    assert {item["code"] for item in first["findings"]} >= {
        "invalid_replacement_for",
        "invalid_path",
        "invalid_edge_type",
        "invalid_ref",
        "invalid_reference_id",
        "invalid_record_metadata",
    }


def test_cycles_and_multiple_superseders_are_unresolved_conflicts() -> None:
    payload = {
        "schema": tool.SCHEMA,
        "artifacts": [
            {"logical_id": "old"},
            {"logical_id": "one", "replacement_for": "old"},
            {"logical_id": "two", "replacement_for": "old"},
            {"logical_id": "cycle-a", "replacement_for": "cycle-b"},
            {"logical_id": "cycle-b", "replacement_for": "cycle-a"},
        ],
    }
    report = tool.build_graph(payload)
    classes = {row["logical_id"]: row["classification"] for row in report["classifications"]}
    assert classes["old"] == "unresolved_conflict"
    assert classes["cycle-a"] == classes["cycle-b"] == "unresolved_conflict"
    assert {item["code"] for item in report["findings"]} >= {
        "ambiguous_superseders",
        "supersession_cycle",
    }


def test_non_check_render_returns_zero_for_findings(capsys) -> None:
    assert tool.main(["--input", str(FIXTURE), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
