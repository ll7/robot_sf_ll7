"""Focused tests for the checkpoint preservation custody checker (issue #8831)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation import check_checkpoint_preservation as tool

FIXTURES = Path(__file__).parent / "fixtures" / "checkpoint_preservation"
CASE_STATES = {
    "missing_companion": tool.STATE_NO_COMPANION,
    "digest_mismatch": tool.STATE_DIGEST,
    "stale_alias": tool.STATE_AMBIGUOUS,
    "missing_lineage": tool.STATE_LINEAGE,
    "partial_copy": tool.STATE_PARTIAL,
    "contract_mismatch": tool.STATE_CONTRACT,
    "loadability_failed": tool.STATE_LOAD,
    "incomplete_training": tool.STATE_TRAINING,
    "unsafe_destination": tool.STATE_DESTINATION,
    "uncleared_publication": tool.STATE_PUBLICATION,
    "missing_artifact": tool.STATE_NO_ARTIFACT,
}


def _rows(report):
    return {row["artifact_id"]: row for row in report["artifacts"]}


def _write(tmp_path, payload, name="case.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_complete_fixture_is_ready_and_records_inventory():
    report = tool.build_report(FIXTURES / "complete.json")
    row = _rows(report)["checkpoint_complete"]
    assert report["status"] == "ready" and report["findings"] == []
    assert row["state"] == tool.STATE_READY and row["load_status"] == tool.LOAD_VERIFIED
    assert row["producer"]["data_identity"] == "fixture_dataset_v1"
    assert row["observation_contract"]["shape"] == [4] and row["seed"] == 7
    assert {item["role"] for item in row["companions"]} == {"normalizer", "vecnormalize"}
    assert row["downstream_consumers"] == ["configs/train.yaml"]
    assert row["byte_size"] == row["byte_size_observed"] == 25


@pytest.mark.parametrize(("artifact_id", "state"), sorted(CASE_STATES.items()))
def test_failing_fixture_reports_each_required_case(artifact_id, state):
    report = tool.build_report(FIXTURES / "failing.json")
    assert report["status"] == "blocked"
    assert _rows(report)[artifact_id]["state"] == state
    assert tool.render_json(report) == tool.render_json(
        tool.build_report(FIXTURES / "failing.json")
    )


def test_loadability_unavailable_is_explicit_and_not_a_failure():
    row = _rows(tool.build_report(FIXTURES / "failing.json"))["loadability_unavailable"]
    assert row["load_status"] == tool.LOAD_UNAVAILABLE and row["state"] == tool.STATE_READY


def test_cli_exit_codes_and_json_are_deterministic(capsys):
    assert tool.main(["--check", "--fixture", str(FIXTURES / "complete.json")]) == 0
    complete = capsys.readouterr().out
    assert tool.main(["--check", "--fixture", str(FIXTURES / "failing.json")]) == 1
    failing = capsys.readouterr().out
    assert json.loads(complete)["status"] == "ready"
    assert json.loads(failing)["status"] == "blocked"
    assert complete == tool.render_json(json.loads(complete))


def test_text_format_is_compact(capsys):
    tool.main(["--fixture", str(FIXTURES / "complete.json"), "--format", "text"])
    out = capsys.readouterr().out
    assert out.startswith("Checkpoint preservation: READY") and "checkpoint_complete" in out


def test_unreadable_fixture_is_unknown(capsys):
    assert tool.main(["--check", "--fixture", str(FIXTURES / "absent.json")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unknown"


def test_private_paths_and_uncovered_workload_refs_fail_closed(tmp_path):
    secret = "/home/private-host/checkpoint.bin"
    path = _write(
        tmp_path,
        {
            "schema": tool.FIXTURE_SCHEMA,
            "root": ".",
            "workloads": [{"workload_id": "job-x", "checkpoint_refs": ["leaky", "absent"]}],
            "artifacts": [
                {
                    "artifact_id": "leaky",
                    "artifact_path": secret,
                    "artifact_sha256": "a" * 64,
                    "byte_size": 1,
                }
            ],
        },
    )
    report = tool.build_report(path)
    codes = {item["code"] for item in report["findings"]}
    assert report["status"] == "blocked" and "workload_reference_uncovered" in codes
    assert "unsafe_input_path" in _rows(report)["leaky"]["reason_codes"]
    assert secret not in tool.render_json(report)
