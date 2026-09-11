"""Focused tests for the terminal-job harvest helper (issues #8824 and #9033)."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_harvest_terminal_job", REPO / "scripts/validation/harvest_terminal_job.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
COMMIT, CONFIG = "a" * 40, "c" * 64
PATHS = {role: [role] for role in MOD.INVENTORY_ROLES}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fixture(tmp_path: Path, state: str = "completed"):
    root = tmp_path / "fixture/artifact_root"
    for role in MOD.INVENTORY_ROLES:
        if role != "rows":
            _write(root / role, role)
    _write(root / "environment", json.dumps({"source_commit": COMMIT, "config_sha256": CONFIG}))
    rows = []
    for index, (mode, status) in enumerate((("native", "completed"), ("degraded", "completed"))):
        row_id = f"row-{index + 1:04d}"
        record = {"row_id": row_id, "mode": mode, "status": status}
        path = _write(root / f"rows/{row_id}.json", json.dumps(record))
        rows.append({"row_id": row_id, "sha256": MOD._sha256_file(path)})
    request = {"schema_version": MOD.REQUEST_SCHEMA, "issue": 8824}
    request["issue_url"] = "https://github.com/ll7/robot_sf_ll7/issues/8824"
    request["job"] = {"job_id": "fixture-job", "state": state}
    request["ownership"] = {"owner": "ll7", "campaign_id": "fixture"}
    request["source"] = {"commit": COMMIT, "config_sha256": CONFIG}
    request["rows"] = rows
    request["inventory"] = dict(PATHS)
    request["destination"] = {"capacity_bytes": 1 << 20}
    _write(tmp_path / "fixture/request.json", json.dumps(request))
    return request, root


def test_scheduler_and_artifact_axes_stay_separate(tmp_path):
    for state in ("completed", "failed", "cancelled", "timeout"):
        request, root = _fixture(tmp_path / state, state)
        report = MOD.build_report(request, root)
        assert report["status"] == "ready" and report["artifact_status"] == "complete"
        assert report["scheduler"]["terminal"] is True
        assert report["scheduler"]["scientific_status"] == "not_evaluated"
    request, root = _fixture(tmp_path / "running", "running")
    report = MOD.build_report(request, root)
    assert report["status"] == "blocked" and "job_not_terminal" in report["reason_codes"]
    assert report["artifact_status"] == "complete"

    request, root = _fixture(tmp_path / "unknown", "unknown")
    request["job"]["terminal"] = True
    report = MOD.build_report(request, root)
    assert report["status"] == "blocked"
    assert report["scheduler"]["terminal"] is False
    assert "unsupported_scheduler_state" in report["reason_codes"]
    assert report["cleanup_eligibility"]["eligible"] is False


def test_row_dispositions_roles_and_partial_artifacts(tmp_path):
    request, root = _fixture(tmp_path / "partial")
    (root / "rows/row-0002.json").unlink()
    report = MOD.build_report(request, root)
    assert report["status"] == "ready" and report["artifact_status"] == "partial"
    assert report["row_reconciliation"]["disposition_counts"]["missing"] == 1
    assert "artifact_incomplete" in report["cleanup_eligibility"]["reason_codes"]
    roles = {role for item in report["inventory"] for role in item["roles"]}
    assert roles == set(MOD.INVENTORY_ROLES)
    assert report["byte_counts"]["member_count"] == 8
    for status, disposition in (("failed", "failed"), ("unavailable", "unavailable")):
        request, root = _fixture(tmp_path / status)
        record = {"row_id": "row-0002", "mode": "degraded", "status": status}
        path = _write(root / "rows/row-0002.json", json.dumps(record))
        request["rows"][1]["sha256"] = MOD._sha256_file(path)
        counts = MOD.build_report(request, root)["row_reconciliation"]
        assert counts["disposition_counts"][disposition] == 1
        assert counts["mode_counts"]["degraded"] == 1


def test_conflicts_and_missing_surfaces_fail_closed(tmp_path):
    cases = {
        "duplicate": "duplicate_row",
        "checksum": "row_checksum_mismatch",
        "unclassified": "unclassified_member",
        "unexpected": "unexpected_row",
        "stale": "stale_source",
        "no_manifest": "missing_required_surface",
    }
    for case, code in cases.items():
        request, root = _fixture(tmp_path / case)
        if case == "duplicate":
            _write(root / "rows/attempt2/row-0001.json", (root / "rows/row-0001.json").read_text())
        elif case == "checksum":
            _write(root / "rows/row-0001.json", (root / "rows/row-0001.json").read_text() + " ")
        elif case == "unclassified":
            _write(root / "scratch.tmp", "extra")
        elif case == "unexpected":
            _write(root / "rows/row-9999.json", '{"row_id":"row-9999"}')
        elif case == "stale":
            _write(
                root / "environment",
                json.dumps({"source_commit": "b" * 40, "config_sha256": CONFIG}),
            )
        else:
            (root / "manifest").unlink()
        report = MOD.build_report(request, root)
        assert report["status"] == "blocked" and code in report["reason_codes"], case


def test_missing_identity_contract_and_capacity_fail_closed(tmp_path):
    for key, code in (
        ("job", "missing_job_identity"),
        ("rows", "missing_row_contract"),
        ("ownership", "missing_artifact_ownership"),
        ("source", "missing_source_identity"),
        ("inventory", "missing_field"),
    ):
        request, root = _fixture(tmp_path / key)
        request.pop(key)
        assert code in MOD.build_report(request, root)["reason_codes"]
    request, root = _fixture(tmp_path / "capacity")
    request["destination"]["capacity_bytes"] = 1
    assert "destination_capacity_exceeded" in MOD.build_report(request, root)["reason_codes"]
    request["destination"].pop("capacity_bytes")
    assert "missing_capacity_receipt" in MOD.build_report(request, root)["reason_codes"]
    request, root = _fixture(tmp_path / "missing_role")
    request["inventory"].pop("manifest")
    assert "missing_required_role" in MOD.build_report(request, root)["reason_codes"]


def test_empty_required_roles_cannot_report_complete(tmp_path):
    for role in MOD.REQUIRED_ROLES:
        request, root = _fixture(tmp_path / role)
        request["inventory"][role] = []
        if role == "manifest":
            (root / role).unlink()
        elif role == "rows":
            shutil.rmtree(root / role)
        else:
            (root / role).unlink()
        report = MOD.build_report(request, root)
        assert report["status"] == "blocked"
        assert report["artifact_status"] != "complete"
        assert "empty_required_role" in report["reason_codes"]
        assert report["cleanup_eligibility"]["eligible"] is False


def test_conflicting_environment_records_are_not_lexically_selected(tmp_path):
    request, root = _fixture(tmp_path)
    alternate = _write(
        root / "environment-alternate.json",
        json.dumps({"source_commit": "b" * 40, "config_sha256": CONFIG}),
    )
    request["inventory"]["environment"] = [
        *request["inventory"]["environment"],
        alternate.name,
    ]
    report = MOD.build_report(request, root)
    assert report["status"] == "blocked"
    assert "conflicting_environment_records" in report["reason_codes"]
    assert report["environment"]["record"] is None
    assert report["cleanup_eligibility"]["eligible"] is False


def test_manifest_destination_and_idempotency(tmp_path):
    request, root = _fixture(tmp_path)
    report = MOD.build_report(request, root)
    written = MOD.write_receipts(report, tmp_path / "a")[1]
    before = {path.name: path.read_bytes() for path in (tmp_path / "a").iterdir()}
    MOD.write_receipts(report, tmp_path / "b")
    MOD.write_receipts(report, tmp_path / "a")
    assert all((tmp_path / "a" / name).read_bytes() == data for name, data in before.items())
    for name in written:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    sums = (tmp_path / "a" / MOD.SUMS_FILENAME).read_text().splitlines()
    present = [item for item in report["inventory"] if item["status"] == "present"]
    assert len(sums) == len(present) == 9
    for line, item in zip(sums, present, strict=True):
        assert line == f"{item['sha256']}  {item['relative_path']}"
    text = (tmp_path / "a" / MOD.RECEIPT_FILENAME).read_text()
    assert str(tmp_path) not in text
    assert all(not item["relative_path"].startswith("/") for item in report["inventory"])
    destination = tmp_path / "destination"
    shutil.copytree(root, destination)
    verified = MOD.build_report(request, root, destination_root=destination)
    assert verified["destination"]["verification"] == "verified"
    assert verified["destination"]["unexpected_members"] == 0
    assert verified["cleanup_eligibility"]["eligible"] is True
    _write(destination / "untracked.txt", "not in the source inventory")
    extra = MOD.build_report(request, root, destination_root=destination)
    assert extra["status"] == "blocked"
    assert extra["destination"]["verification"] == "unexpected_members"
    assert extra["destination"]["unexpected_members"] == 1
    assert "destination_unexpected_member" in extra["reason_codes"]
    assert extra["cleanup_eligibility"]["eligible"] is False
    (destination / "rows/row-0002.json").unlink()
    blocked = MOD.build_report(request, root, destination_root=destination)
    assert blocked["status"] == "blocked"
    assert blocked["destination"]["verification"] == "incomplete"
    assert blocked["cleanup_eligibility"]["eligible"] is False
    (tmp_path / "a" / MOD.SUMS_FILENAME).write_text("tampered\n", encoding="utf-8")
    conflict, written = MOD.write_receipts(report, tmp_path / "a")
    assert written == [] and "immutable_output_conflict" in conflict["reason_codes"]


def test_cli_check_write_malformed_and_issue_guard(tmp_path, capsys):
    _, root = _fixture(tmp_path)
    base = ["--check", "--fixture", str(tmp_path / "fixture")]
    assert MOD.main([*base, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == MOD.RECEIPT_SCHEMA
    assert MOD.main([*base, "--issue", "1"]) == 2
    assert "issue_mismatch" in capsys.readouterr().out
    assert (
        MOD.main(["--output-root", str(tmp_path / "out"), "--fixture", str(tmp_path / "fixture")])
        == 0
    )
    malformed = _write(tmp_path / "bad.json", "{not json")
    assert MOD.main(["--check", "--request", str(malformed), "--artifact-root", str(root)]) == 3
