"""Focused contract tests for the cleanup-eligibility guard (#8906)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validation import check_cleanup_eligibility as tool

CASE_FILE = Path(__file__).parent / "fixtures" / "cleanup_eligibility" / "cases.json"
TOOL = (
    Path(__file__).resolve().parents[2] / "scripts" / "validation" / "check_cleanup_eligibility.py"
)
DIGEST = "a" * 64
CASES = [
    (
        "active-job",
        "protected_active",
        "active_output_owner,active_writer_or_lease",
        "owner:issue-8819,writer:job-alpha",
    ),
    (
        "one-copy-cluster-result",
        "protected_only_copy",
        "no_verified_durable_copy,only_expiring_host_copy",
        "copy:cluster-copy",
    ),
    ("two-copy-verified-result", "eligible", "", ""),
    ("stale-pointer", "conflict", "stale_pointer", "pointer:stale-pointer"),
    ("changed-digest", "protected_unverified", "destination_digest_mismatch", "copy:archive-copy"),
    (
        "active-dissertation-consumer",
        "protected_referenced",
        "retention_hold,unresolved_consumer",
        "retention:active-dissertation-consumer,consumer:diss-chapter-4",
    ),
    ("superseded-diagnostic", "eligible", "", ""),
    (
        "unknown-private-projection",
        "unknown",
        "private_projection_unresolved",
        "projection:unknown-private-projection",
    ),
    ("concurrent-writer", "conflict", "concurrent_writer", "writer:writer-a,writer:writer-b"),
    ("scheduler-completion-claim", "protected_unverified", "non_evidence_basis", "copy:claim-copy"),
]


@pytest.mark.parametrize(("artifact_id", "outcome", "codes", "owners"), CASES)
def test_fixture_outcomes_match_contract(
    artifact_id: str, outcome: str, codes: str, owners: str
) -> None:
    """All nine required fixtures plus the non-evidence-basis case match the contract."""
    report = tool.check_artifact_file(CASE_FILE, artifact_id)
    assert report.outcome == outcome
    assert set(report.reason_codes) == set(filter(None, codes.split(",")))
    assert set(report.blocking_owners) == set(filter(None, owners.split(",")))
    assert (report.semantic_digest == DIGEST) or outcome != "eligible"


def test_cli_exit_codes_json_and_determinism() -> None:
    """Eligible exits 0, protected exits 2, and repeated runs are byte-identical."""

    def run(artifact_id: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(TOOL),
                "--record",
                str(CASE_FILE),
                "--artifact",
                artifact_id,
                "--check",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    eligible, protected = run("two-copy-verified-result"), run("one-copy-cluster-result")
    assert eligible.returncode == 0
    assert protected.returncode == 2
    assert json.loads(eligible.stdout)["outcome"] == "eligible"
    assert json.loads(protected.stdout)["outcome"] == "protected_only_copy"
    assert run("two-copy-verified-result").stdout == eligible.stdout


def test_private_locator_values_are_rejected_and_never_echoed(tmp_path: Path) -> None:
    """A locator-like copy ID fails closed and never appears in the public report."""
    secret = "/srv/private-host/cluster-a"
    payload = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    artifact = next(
        a for a in payload["artifacts"] if a["artifact_id"] == "two-copy-verified-result"
    )
    artifact["artifact_id"] = "leaky-copy"
    artifact["copies"] = [artifact["copies"][0] | {"copy_id": secret}]
    record = tmp_path / "leaky.json"
    record.write_text(json.dumps(payload), encoding="utf-8")
    report = tool.check_artifact_file(record, "leaky-copy")
    assert report.outcome == "unknown"
    assert "private_locator_rejected" in report.reason_codes
    assert secret not in json.dumps(report.to_dict())
    assert secret not in tool.render_text(report)


def test_missing_and_unreadable_records_fail_closed(tmp_path: Path) -> None:
    """Absent identities are unknown and unreadable records raise RecordError."""
    assert tool.check_artifact_file(CASE_FILE, "not-recorded").reason_codes == (
        "artifact_not_recorded",
    )
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(tool.RecordError):
        tool.check_artifact_file(bad, "two-copy-verified-result")
    with pytest.raises(tool.RecordError):
        tool.check_artifact_file(CASE_FILE, "../escape")


def test_cleanup_command_checks_eligibility_in_check_only_mode() -> None:
    """The existing cleanup command gates on the guard and deletes nothing in --check mode."""
    from scripts.dev.clean_generated_output import main as cleanup_main

    blocked = cleanup_main(
        ["--eligibility-record", str(CASE_FILE), "--artifact", "active-job", "--check"]
    )
    allowed = cleanup_main(
        [
            "--eligibility-record",
            str(CASE_FILE),
            "--artifact",
            "two-copy-verified-result",
            "--check",
        ]
    )
    assert blocked == 2
    assert allowed == 0
