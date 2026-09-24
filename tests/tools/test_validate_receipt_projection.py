"""Focused contract tests for private-to-public receipt projection validation (#8844)."""

from __future__ import annotations

import contextlib
import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from scripts.tools import validate_receipt_projection as tool

FIXTURES = Path(__file__).parent / "fixtures" / "receipt_projection"
SCHEDULER, HARVEST, UNSUPPORTED = (
    FIXTURES / "scheduler_allocation.private.json",
    FIXTURES / "harvest.private.json",
    FIXTURES / "unsupported.private.json",
)
PRIVATE_MARKS = (
    "custody-0007 camera-ready-2026-09 gpu-node-17 researcher42 /scratch TOKENVALUE "
    "proj-rm gpu-a100 slurm-987654"
).split()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _public(private: dict[str, Any]) -> dict[str, Any]:
    projection = tool.project_receipt(private)
    assert not projection.issues, projection.issues
    assert projection.public_receipt is not None
    return projection.public_receipt


def _codes(report: dict[str, Any]) -> set[str]:
    return {issue["code"] for issue in report["issues"]}


@pytest.mark.parametrize(
    ("path", "schema"),
    [(SCHEDULER, "robot_sf.scheduler_allocation_receipt.v1"), (HARVEST, "terminal_job_harvest.v1")],
)
def test_supported_classes_project_valid_and_preserve_identity(path: Path, schema: str) -> None:
    """Both supported classes project to a valid public receipt without private marks."""
    private = _load(path)
    projection = tool.project_receipt(private)
    public = projection.public_receipt
    assert public is not None and not projection.issues
    assert public["source_binding"]["receipt_sha256"] == projection.private_receipt_sha256
    report = tool.check_receipt_projection(private, public)
    assert report["status"] == "projection_valid" and report["issue_count"] == 0
    assert report["public_schema"] == schema and report["check_only"] is True
    assert not [mark for mark in PRIVATE_MARKS if mark in tool.render_json(public)]
    # Required 64-hex digests in the fixture are not mistaken for secrets.
    assert tool._private_content_reason("a" * 64) == ""


def test_aliases_are_stable_path_scoped_and_byte_stable() -> None:
    """Aliases are deterministic, key-order independent, and never ad hoc string masking."""
    private = _load(SCHEDULER)
    public = _public(private)
    assert public["job_alias"] == _public(json.loads(json.dumps(private)))["job_alias"]
    assert public["job_alias"] != private["job_alias"]
    assert public["job_alias"].startswith("job-") and len(public["job_alias"]) == 20
    assert public["campaign_id"].startswith("campaign-")
    assert public["artifacts"]["result_root_identity"].startswith("artifact-root-")
    assert public["submission"]["intent_digest"] == private["submission"]["intent_digest"]
    reordered = dict(reversed(list(json.loads(json.dumps(private)).items())))
    assert tool.render_json(_public(reordered)) == tool.render_json(public)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_id", "redacted"),
        ("scheduler_state", "unavailable"),
        ("job_alias", "job-REDACTED"),
    ],
)
def test_over_redaction_or_masking_of_required_identity_fails(field: str, value: str) -> None:
    """Redacted, unavailable, or hand-masked required identity fails the check."""
    private = _load(SCHEDULER)
    public = _public(private)
    public[field] = value
    report = tool.check_receipt_projection(private, public)
    assert report["status"] == "projection_invalid"
    assert _codes(report) & {"over_redacted_identity", "invalid_required_identity"}


def test_omitted_required_identity_fails() -> None:
    """Dropping a required digest or row count is reported, not silently accepted."""
    private = _load(HARVEST)
    public = _public(private)
    del public["source"]["config_sha256"]
    del public["row_reconciliation"]["expected_count"]
    assert {"missing_required_identity", "projection_mismatch"} <= _codes(
        tool.check_receipt_projection(private, public)
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda p: p["environment"].update(record="/home/researcher42/output/env.json"),
            id="private-path",
        ),
        pytest.param(lambda p: p.update(hostname="gpu-node-17.cluster.invalid"), id="field"),
        pytest.param(
            lambda p: p.update(
                claim_boundary=p["claim_boundary"]
                + " at 10.1.2.3 https://cluster.invalid/run?sig=SECRETSIG"
            ),
            id="ip-and-signed-url",
        ),
    ],
)
def test_under_redaction_paths_hosts_ips_and_urls_fail(mutate: Any) -> None:
    """Private paths, forbidden fields, IPs, and signed URLs never pass as public."""
    private = _load(HARVEST)
    public = _public(private)
    mutate(public)
    assert "under_redacted_value" in _codes(tool.check_receipt_projection(private, public))


def test_digest_only_policy_keeps_proof_without_publishing_message() -> None:
    """The digest policy records proof of a private message without its bytes."""
    private = _load(HARVEST)
    private["problems"] = [
        {"code": "unclassified_member", "location": "aux/note.txt", "message": "token=abc123"}
    ]
    private.update({"problem_count": 1, "reason_codes": ["unclassified_member"]})
    private.update({"status": "blocked", "harvest_state": "harvest_blocked"})
    public = _public(private)
    assert public["problems"][0]["message"].startswith("sha256:")
    assert "abc123" not in tool.render_json(public)
    assert tool.check_receipt_projection(private, public)["status"] == "projection_valid"


def test_rejected_and_unknown_private_fields_fail_closed() -> None:
    """Credential-named and policy-less private fields are refused, not partially projected."""
    private = _load(SCHEDULER)
    private.update({"password": "hunter2", "custom_field": {"value": 1}})
    codes = {issue.code for issue in tool.project_receipt(private).issues}
    assert {"rejected_private_field", "unsupported_field"} <= codes
    assert (
        tool.check_receipt_projection(private, _public(_load(SCHEDULER)))["status"]
        == "projection_invalid"
    )


def test_binding_and_reused_scheduler_semantics_fail_closed() -> None:
    """The binding digest must match, and illegal scheduler transitions are rejected."""
    private = _load(HARVEST)
    public = _public(private)
    public["source_binding"]["receipt_sha256"] = "0" * 64
    assert {"source_binding_mismatch", "projection_mismatch"} <= _codes(
        tool.check_receipt_projection(private, public)
    )
    scheduler = _load(SCHEDULER)
    scheduler_public = _public(scheduler)
    scheduler_public["state_history"] = ["completed", "running"]
    assert "public_schema_illegal_transition" in _codes(
        tool.check_receipt_projection(scheduler, scheduler_public)
    )


def test_unsupported_class_is_reported_not_partially_projected() -> None:
    """A class without a contract is reported unsupported with no projection digest."""
    private = _load(UNSUPPORTED)
    assert tool.project_receipt(private).public_receipt is None
    report = tool.check_receipt_projection(private, {"schema_version": "synthetic.public.v1"})
    assert report["status"] == "unsupported_receipt_class"
    assert report["projected_receipt_sha256"] is None
    assert _codes(report) == {"unsupported_receipt_class"}


def test_cli_check_json_reports_valid_leaked_and_malformed(tmp_path: Path) -> None:
    """The check-only CLI is JSON-deterministic and uses 0/2/3 exit codes."""
    private = _load(HARVEST)
    private_path, public_path = tmp_path / "private.json", tmp_path / "public.json"
    private_path.write_text(json.dumps(private), encoding="utf-8")
    public_path.write_text(tool.render_json(_public(private)), encoding="utf-8")
    args = [
        "--check",
        "--private",
        str(private_path),
        "--public",
        str(public_path),
        "--format",
        "json",
    ]
    out, err = StringIO(), StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        assert tool.main(args) == 0
    assert json.loads(out.getvalue())["status"] == "projection_valid"
    leaked = _public(private)
    leaked["inventory"][0]["relative_path"] = "/home/researcher42/rows/row-0001.json"
    public_path.write_text(json.dumps(leaked), encoding="utf-8")
    out = StringIO()
    with contextlib.redirect_stdout(out):
        assert tool.main(args) == 2
    assert "under_redacted_value" in {item["code"] for item in json.loads(out.getvalue())["issues"]}
    assert tool.main(["--check", "--private", str(UNSUPPORTED), "--public", str(public_path)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert tool.main(["--check", "--private", str(bad), "--public", str(public_path)]) == 3
