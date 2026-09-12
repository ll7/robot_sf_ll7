"""Contract tests for the conservative compute-window consumer graph (#8860)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.tools import compute_window_artifact_consumers as tool

FIXTURE = Path(__file__).parent / "fixtures" / "compute_window_artifact_consumers.json"


def _graph(artifacts: list[dict], *, root: Path | None = None, **groups: object) -> dict:
    return tool.build_graph({"schema": tool.SCHEMA, "artifacts": artifacts, **groups}, root=root)


def _classes(report: dict) -> dict[str, str]:
    return {row["logical_id"]: row["classification"] for row in report["classifications"]}


def test_fixture_graph_is_deterministic_and_preserves_classes() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["active_tasks"].append({"id": "resume-task", "kind": "zzz", "state": "zzz"})
    permuted = json.loads(json.dumps(payload))
    permuted["active_tasks"].reverse()
    permuted["active_tasks"][1]["refs"].reverse()
    first = tool.build_graph(payload)
    assert first == tool.build_graph(payload) == tool.build_graph(permuted)
    assert first["status"] == "failed"
    assert set(
        "dangling_logical_id conflicting_identity supersession_cycle "
        "unknown_dissertation_reference unknown_release_reference "
        "closed_issue_active_runtime_consumer".split()
    ) <= {item["code"] for item in first["findings"]}
    assert _classes(first) == dict(
        zip(
            "active-model historical-table replacement-old regenerable replacement-new "
            "unverified-replacement unverified-regeneration unknown orphan conflict cycle-a cycle-b".split(),
            "active_required historical_required unresolved_conflict regenerable_verified "
            "replacement_verified consumer_unknown historical_required consumer_unknown "
            "orphan_candidate unresolved_conflict unresolved_conflict unresolved_conflict".split(),
            strict=True,
        )
    )
    assert first["verification_contract"].startswith("replacement_verified requires")
    assert {edge["type"] for edge in first["edges"]} == set(tool.EDGE_TYPES)
    assert "private-node" in json.dumps(first)


def test_privacy_and_cli_semantics_are_preserved(capsys) -> None:
    private_values = dict.fromkeys("path command host url environment".split(), "/private/locator")
    projection = _graph(
        [{"logical_id": "a", "sha256": "a" * 64}],
        private_projection=[{"consumer_id": "private", "locator": "/secret/node", "refs": ["a"]}],
    )
    artifact = _graph([{"logical_id": "a", **private_values}])
    assert not projection["ok"] and "/secret/node" not in json.dumps(projection)
    assert not artifact["ok"]
    for render in (tool.render_json, tool.render_dot, tool.render_markdown):
        assert all(value not in render(artifact) for value in private_values.values())
    assert artifact["artifacts"][0]["path"] is None
    assert tool.main(["--input", str(FIXTURE), "--check", "--format", "json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert tool.main(["--input", str(FIXTURE), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert "digraph consumer_graph" in tool.render_dot(report)
    assert "# Compute-window artifact consumer graph" in tool.render_markdown(report)


# robot_sf-artifact-ref: explicit-marker
def test_root_markers_and_line_boundaries(monkeypatch, tmp_path: Path) -> None:
    payload = {"schema": tool.SCHEMA, "artifacts": [{"logical_id": "explicit-marker"}]}
    (tmp_path / "marker.txt").write_text(
        " \t#\trobot_sf-artifact-ref:\texplicit-marker\t\nordinary-marker\n", encoding="utf-8"
    )
    monkeypatch.setattr(tool.subprocess, "check_output", lambda *args, **kwargs: b"marker.txt\0")
    report = tool.build_graph(payload, root=tmp_path)
    assert report["ok"] and len(report["edges"]) == 1
    assert report["edges"][0]["source"].startswith("tracked:")
    assert report["edges"][0]["target"] == "explicit-marker"
    split_lines = (
        "robot_sf-artifact-ref:\nexplicit-marker\n",
        "robot_sf-artifact-ref:\t\nexplicit-marker\n",
    )
    assert all(tool.PUBLIC_REF_RE.findall(text) == [] for text in split_lines)


def test_malformed_inputs_and_strict_verification_remain_fail_closed() -> None:
    payload = {
        "schema": tool.SCHEMA,
        "artifacts": [
            {"logical_id": "old", "replacement_for": 3},
            {"logical_id": "target", "replacement_verified": True},
            {"logical_id": "new", "replacement_for": "old", "replacement_verified": "yes"},
            {"logical_id": "trace", "regenerable": 1, "regeneration_verified": "true"},
            {"id": "orphan", "path": "..", "replacement_for": "x", "orphan_candidate": True},
        ],
        "consumers": [
            {"id": "bad-edge", "kind": "config", "refs": [{"logical_id": "old", "edge": []}]},
            {"id": "bad-ref", "kind": "config", "refs": [None, {"logical_id": 4}]},
            {"id": "bad-state", "kind": "config", "state": {"nested": True}, "refs": "old"},
        ],
    }
    report = tool.build_graph(payload)
    codes = {item["code"] for item in report["findings"]}
    required_codes = (
        "invalid_edge_type invalid_ref invalid_reference_id "
        "invalid_replacement_for invalid_path invalid_boolean invalid_record_metadata"
    )
    assert not report["ok"] and set(required_codes.split()) <= codes
    assert _classes(report)["orphan"] == "unresolved_conflict"
    assert sum(item["code"] == "invalid_boolean" for item in report["findings"]) == 3
    for group in (*tool.CONSUMER_GROUPS, "private_projection"):
        assert not _graph([{"logical_id": "a"}], **{group: [None]})["ok"]


def test_invalid_artifacts_unknown_states_and_missing_ids_never_look_clean() -> None:
    artifacts = [
        {"orphan_candidate": True},
        {"logical_id": "", "orphan_candidate": True},
        {"logical_id": "/private/locator", "orphan_candidate": True},
        {"logical_id": "unknown-state", "orphan_candidate": True},
        {"logical_id": "list-target", "orphan_candidate": True},
        {"logical_id": "mapping-target", "orphan_candidate": True},
    ]
    base = {"schema": tool.SCHEMA, "artifacts": artifacts}
    state_report = tool.build_graph(
        {**base, "consumers": [{"id": "unknown", "refs": ["unknown-state"]}]}
    )
    assert _classes(state_report)["unknown-state"] == "consumer_unknown"
    assert sum(item["code"] == "invalid_artifact_id" for item in state_report["findings"]) == 3
    assert "/private/locator" not in json.dumps(state_report)
    collections = [
        ("consumers", [{"refs": ["list-target"]}, {"refs": ["mapping-target"]}]),
        (
            "configs",
            {"first-key": {"refs": ["list-target"]}, "second-key": {"refs": ["mapping-target"]}},
        ),
    ]
    for field, collection in collections:
        reverse = (
            collection[::-1] if isinstance(collection, list) else dict(reversed(collection.items()))
        )
        first = tool.build_graph({**base, field: collection})
        assert first == tool.build_graph({**base, field: reverse})
        assert not first["ok"]
        assert sum(item["code"] == "missing_consumer_id" for item in first["findings"]) == 2
        assert all(f"{field}:{i}" not in {c["id"] for c in first["consumers"]} for i in range(2))
        classes = _classes(first)
        assert classes["list-target"] == classes["mapping-target"] == "consumer_unknown"


@pytest.mark.parametrize(
    ("records", "finding", "classification"),
    [
        (
            [{"id": "unknown", "state": "zzz", "refs": ["a"]}],
            "unknown_consumer_state",
            "consumer_unknown",
        ),
        ([None], "invalid_records", "unresolved_conflict"),
        ([{"id": "malformed", "refs": [None]}], "invalid_ref", "unresolved_conflict"),
    ],
)
def test_malformed_consumer_evidence_never_proves_orphan(
    records: list[dict | None], finding: str, classification: str
) -> None:
    report = _graph([{"logical_id": "a", "orphan_candidate": True}], consumers=records)
    assert not report["ok"]
    assert finding in {item["code"] for item in report["findings"]}
    assert _classes(report)["a"] == classification
    if finding == "unknown_consumer_state":
        verified = _graph(
            [{"logical_id": "a", "regenerable": True, "regeneration_verified": True}],
            consumers=[
                {
                    "id": "verified",
                    "state": "zzz",
                    "refs": [{"logical_id": "a", "edge": "requires_for_resume"}],
                }
            ],
        )
        assert _classes(verified)["a"] == "consumer_unknown"


def test_unknown_consumer_state_fails_check(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": tool.SCHEMA,
                "artifacts": [{"logical_id": "a", "orphan_candidate": True}],
                "consumers": [{"id": "c", "state": "zzz", "refs": ["a"]}],
            }
        ),
        encoding="utf-8",
    )
    assert tool.main(["--input", str(manifest), "--check"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert _graph(
        [{"logical_id": "clean"}], consumers=[{"id": "c", "state": "active", "refs": ["clean"]}]
    )["ok"]


def test_duplicate_artifacts_are_canonical_and_report_identity_conflict() -> None:
    declarations = [
        {"logical_id": "a", "sha256": "a" * 64},
        {"logical_id": "a", "sha256": "b" * 64},
    ]
    first = _graph(declarations)
    second = _graph(list(reversed(declarations)))
    assert first == second
    assert first["artifacts"][0]["sha256"] == "a" * 64
    assert sum(item["code"] == "conflicting_identity" for item in first["findings"]) == 1


def test_dot_qualifies_reserved_namespace_ids() -> None:
    report = _graph(
        [
            {"logical_id": "old"},
            {"logical_id": "artifact:new", "replacement_for": "old"},
            {"logical_id": "consumer:artifact", "replacement_for": "old"},
        ],
        consumers=[
            {"id": "consumer:reader", "refs": ["artifact:new"]},
            {"id": "artifact:reader", "refs": ["consumer:artifact"]},
        ],
    )
    dot = tool.render_dot(report)
    assert '"artifact:artifact:new" -> "artifact:old"' in dot
    assert '"artifact:consumer:artifact" -> "artifact:old"' in dot
    assert '"consumer:consumer:reader" -> "artifact:artifact:new"' in dot
    assert '"consumer:artifact:reader" -> "artifact:consumer:artifact"' in dot
    assert '"artifact:new" ->' not in dot and '"consumer:artifact" ->' not in dot


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("scan", "tracked_scan_failed"),
        ("read", "tracked_read_failed"),
        ("file_decode", "tracked_decode_failed"),
    ],
)
def test_tracked_collection_failures_are_structured_and_conservative(
    monkeypatch, tmp_path: Path, mode: str, expected: str
) -> None:
    if mode == "scan":
        monkeypatch.setattr(tool.subprocess, "check_output", Mock(side_effect=OSError()))
    else:
        monkeypatch.setattr(tool.subprocess, "check_output", Mock(return_value=b"tracked.txt\0"))
        error = OSError() if mode == "read" else UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        monkeypatch.setattr(Path, "read_text", Mock(side_effect=error))
    report = _graph([{"logical_id": "unverified", "orphan_candidate": True}], root=tmp_path)
    assert not report["ok"] and expected in {item["code"] for item in report["findings"]}
    assert _classes(report)["unverified"] == "unresolved_conflict"
