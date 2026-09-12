"""Focused tests for the checkpoint preservation custody checker (issue #8831)."""

from __future__ import annotations

import hashlib
import json
import shutil
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
    assert row["downstream_consumers"] == ["fixture_config.yaml"]
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


def _fixture_copy(tmp_path: Path) -> Path:
    """Copy the shared fixtures into a mutable root for one test."""
    root = tmp_path / "root"
    shutil.copytree(FIXTURES, root)
    return root


def _mutate_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _audit_row(root: Path, model_id: str) -> dict:
    payload = json.loads((root / "compatibility_audit.json").read_text(encoding="utf-8"))
    return next(row for row in payload["models"] if row["model_id"] == model_id)


@pytest.mark.parametrize("code", ["invalid_artifact", "missing_data_member"])
def test_fatal_loadability_codes_block_regardless_of_named_code(tmp_path: Path, code: str) -> None:
    """Every fatal loadability reason reaches the blocked load state."""
    root = _fixture_copy(tmp_path)
    audit = json.loads((root / "compatibility_audit.json").read_text(encoding="utf-8"))
    for row in audit["models"]:
        if row["model_id"] == "fixture_fatal_v1":
            row["reason_codes"] = [code]
    (root / "compatibility_audit.json").write_text(json.dumps(audit), encoding="utf-8")

    row = _rows(tool.build_report(root / "failing.json"))["loadability_failed"]

    assert row["state"] == tool.STATE_LOAD
    assert "loadability_failed" in row["reason_codes"]
    assert code in row["reason_codes"]


def test_verified_receipt_requires_declared_digest_binding(tmp_path: Path) -> None:
    """A verified receipt without the artifact digest is not loadability evidence."""
    root = _fixture_copy(tmp_path)
    audit = json.loads((root / "compatibility_audit.json").read_text(encoding="utf-8"))
    for row in audit["models"]:
        if row["model_id"] == "fixture_complete_v1":
            row["artifact"].pop("sha256", None)
    (root / "compatibility_audit.json").write_text(json.dumps(audit), encoding="utf-8")

    row = _rows(tool.build_report(root / "complete.json"))["checkpoint_complete"]

    assert row["state"] == tool.STATE_DIGEST
    assert "receipt_digest_mismatch" in row["reason_codes"]


def test_verified_receipt_requires_declared_framework_binding(tmp_path: Path) -> None:
    """A verified receipt naming another loader is a contract mismatch."""
    root = _fixture_copy(tmp_path)
    audit = json.loads((root / "compatibility_audit.json").read_text(encoding="utf-8"))
    for row in audit["models"]:
        if row["model_id"] == "fixture_complete_v1":
            row["probe"]["loader"] = "jax"
    (root / "compatibility_audit.json").write_text(json.dumps(audit), encoding="utf-8")

    row = _rows(tool.build_report(root / "complete.json"))["checkpoint_complete"]

    assert row["state"] == tool.STATE_CONTRACT
    assert "loadability_contract_mismatch" in row["reason_codes"]


def test_trace_registry_placeholder_is_not_resolvable(tmp_path: Path) -> None:
    """A non-digest placeholder must never resolve as a dataset identity."""
    root = _fixture_copy(tmp_path)
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    complete["artifacts"][0]["producer"]["data_identity"] = "pending"
    (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")

    row = _rows(tool.build_report(root / "complete.json"))["checkpoint_complete"]

    assert row["state"] == tool.STATE_LINEAGE
    assert "data_identity_unresolved" in row["reason_codes"]


def test_malformed_trace_entry_fails_closed(tmp_path: Path) -> None:
    """A malformed trace identity is a blocking registry finding."""
    root = _fixture_copy(tmp_path)
    trace = root / "trace_registry.yaml"
    trace.write_text(
        trace.read_text(encoding="utf-8").replace(
            "trace_id: fixture_train", "trace_id: 123-bad id"
        ),
        encoding="utf-8",
    )

    report = tool.build_report(root / "complete.json")

    assert report["status"] == "blocked"
    assert "unsupported_receipt" in {item["code"] for item in report["findings"]}


def test_free_text_absolute_paths_never_render(tmp_path: Path) -> None:
    """Absolute private paths in free-text fields are rejected and never echoed."""
    root = _fixture_copy(tmp_path)
    secret = "/home/private-host/secret-checkpoint.bin"
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    complete["artifacts"][0]["downstream_consumers"] = [secret]
    (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")

    report = tool.build_report(root / "complete.json")
    row = _rows(report)["checkpoint_complete"]

    assert row["state"] == tool.STATE_INVENTORY
    assert "inventory_field_missing" in row["reason_codes"]
    assert secret not in tool.render_json(report)


def test_companion_symlink_escape_fails_closed(tmp_path: Path) -> None:
    """A companion symlink escaping the fixture root is refused even with a match."""
    root = _fixture_copy(tmp_path)
    content = b"outside companion bytes\n"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(content)
    (root / "artifacts" / "escaped.bin").symlink_to(outside)
    digest = hashlib.sha256(content).hexdigest()

    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    for companion in complete["artifacts"][0]["companions"]:
        companion.update(path="artifacts/escaped.bin", sha256=digest, byte_size=len(content))
    (root / "complete.json").write_text(json.dumps(complete), encoding="utf-8")

    row = _rows(tool.build_report(root / "complete.json"))["checkpoint_complete"]

    assert row["state"] == tool.STATE_NO_COMPANION
    assert "companion_unsafe" in row["reason_codes"]
