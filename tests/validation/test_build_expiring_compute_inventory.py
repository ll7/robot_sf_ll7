"""Focused contract tests for the expiring-compute inventory (#8822).

GitHub and scheduler states are mocked from JSON fixtures; no live API, scheduler, credential,
or scientific state is contacted or mutated.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts.validation import build_expiring_compute_inventory as tool

# Dense layout keeps issue #8822's 800 net-line cap; ruff lint stays active, format is scoped out.
# fmt: off
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "expiring_compute_inventory" / "cases.json"
TOOL = ROOT / "scripts" / "validation" / "build_expiring_compute_inventory.py"
CASES = tool.load_case_inventories(FIXTURE)
CASE_CONTRACT = [
    ("resource_dependent", "resource_dependent", "submit_now", ["window_open"]),
    ("locally_runnable_later", "locally_runnable_later", "safe_to_defer", []),
    ("unknown_evidence", "unknown", "unknown",
     ["missing_local_feasibility_evidence", "insufficient_structured_evidence"]),
    ("owner_disagreement", "unknown", "unknown", ["owner_disagreement"]),
    ("keyword_only", "unknown", "unknown",
     ["keyword_mentions_ignored", "insufficient_structured_evidence"])]
WINDOW_ACTIONS = {"job_running_carla": "harvest_now", "job_harvested_gpu": "transfer_now",
                  "job_blocked_gpu": "blocked_do_not_submit", "job_routine_gpu": "prestage_now",
                  "job_completed_cpu": "safe_to_defer"}


def _report(case: str) -> dict[str, Any]:
    return tool.build_inventory([deepcopy(CASES[case])])


def _row(case: str) -> dict[str, Any]:
    rows = _report(case)["rows"]
    assert len(rows) == 1
    return rows[0]


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True,
                          check=False, cwd=ROOT)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return _run_tool("--inputs", str(FIXTURE), "--check", *args)


@pytest.mark.parametrize(("name", "dependence", "action", "codes"), CASE_CONTRACT)
def test_fixture_cases_match_contract(name: str, dependence: str, action: str,
                                      codes: list[str]) -> None:
    report = _report(name)
    row = report["rows"][0]
    assert (row["dependence"], row["recommended_action"]) == (dependence, action)
    assert set(row["reason_codes"]) == set(codes)
    assert report["schema"] == tool.REPORT_SCHEMA and report["check_only"] is True


@pytest.mark.parametrize("name", ["resource_dependent", "unknown_evidence", "keyword_only"])
def test_cli_reports_cases_as_compact_json(name: str) -> None:
    result = _run_cli("--case", name, "--format", "json")
    assert result.returncode == 0
    assert len(result.stdout.strip().splitlines()) == 1
    payload = json.loads(result.stdout)
    assert payload["summary"]["row_count"] == 1
    assert payload["rows"][0]["work_id"]


def test_cli_repeat_runs_are_byte_stable_and_text_is_concise() -> None:
    first = _run_cli("--case", "window_actions", "--format", "json")
    second = _run_cli("--case", "window_actions", "--format", "json")
    assert first.stdout == second.stdout
    text = _run_cli("--case", "resource_dependent", "--format", "text")
    assert "inventory status=complete rows=1" in text.stdout and "submit_now" in text.stdout


def test_keyword_mentions_never_establish_dependence() -> None:
    row = _row("keyword_only")
    assert row["keyword_mentions"] == ["carla", "gpu", "slurm"]
    assert row["evidence_fields"] == []
    assert row["dependence"] == "unknown"
    assert row["evidence_refs"] == []


def test_owner_disagreement_and_missing_evidence_stop_at_unknown() -> None:
    assert _row("owner_disagreement")["reason_codes"] == ["owner_disagreement"]
    unknown = _row("unknown_evidence")
    assert "insufficient_structured_evidence" in unknown["reason_codes"]
    assert unknown["evidence_fields"] == ["dependence.basis",
                                          "dependence.required_capabilities",
                                          "dependence.resource_class"]


def test_missing_exact_identity_stays_unknown() -> None:
    document = deepcopy(CASES["resource_dependent"])
    del document["items"][0]["identities"]["checkpoint"]
    row = tool.build_inventory([document])["rows"][0]
    assert row["dependence"] == "unknown"
    assert {"missing_identity_checkpoint", "missing_exact_identity"} <= set(row["reason_codes"])


def test_window_actions_follow_the_documented_precedence() -> None:
    report = _report("window_actions")
    actions = {row["work_id"]: row["recommended_action"] for row in report["rows"]}
    assert actions == WINDOW_ACTIONS
    assert report["submission_order"] == ["job_running_carla", "job_harvested_gpu",
                                          "job_routine_gpu"]
    assert len(report["follow_ups"]) == 5


def test_explicit_exclusion_is_reported_without_classification() -> None:
    row = _row("excluded_duplicate")
    assert row["excluded"] is True
    assert row["exclusion_reason"] == "duplicate_of_issue_8823"
    assert row["recommended_action"] == "unknown"
    assert row["reason_codes"] == ["explicitly_excluded"]


def test_duplicate_work_items_fail_closed() -> None:
    document = deepcopy(CASES["resource_dependent"])
    document["items"].append(deepcopy(document["items"][0]))
    report = tool.build_inventory([document])
    assert [row["dependence"] for row in report["rows"]] == ["unknown", "unknown"]
    assert all("duplicate_work_item" in row["reason_codes"] for row in report["rows"])
    assert report["follow_ups"] == []


def test_unsanitized_values_never_echo_into_public_output() -> None:
    secret = "/home/private-owner/campaign/run.json"
    document = deepcopy(CASES["resource_dependent"])
    document["items"][0]["evidence_refs"] = [secret]
    report = tool.build_inventory([document])
    assert "unsanitized_input" in report["rows"][0]["reason_codes"]
    assert report["rows"][0]["dependence"] == "unknown"
    assert secret not in json.dumps(report)


def test_private_companion_is_bound_by_digest_only(tmp_path: Path) -> None:
    private = tmp_path / "private_inventory.json"
    private.write_text('{"private": "do-not-publish"}', encoding="utf-8")
    expected = hashlib.sha256(private.read_bytes()).hexdigest()
    result = _run_tool("--inputs", str(FIXTURE), "--case", "resource_dependent", "--check",
                       "--format", "json", "--private-inventory", str(private),
                       "--private-owner", "private_ops_inventory")
    assert result.returncode == 0
    assert json.loads(result.stdout)["private_binding"] == {
        "owner": "private_ops_inventory", "class": "external_private_inventory",
        "sha256": expected, "content_published": False}
    assert "do-not-publish" not in result.stdout


def test_missing_as_of_fails_closed_unless_overridden(tmp_path: Path) -> None:
    document = deepcopy(CASES["resource_dependent"])
    document.pop("as_of")
    path = tmp_path / "no-as-of.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert _run_tool("--inputs", str(path), "--check", "--format", "json").returncode == 2
    overridden = _run_tool("--inputs", str(path), "--check", "--format", "json",
                           "--as-of", "2026-09-12T08:00:00Z")
    assert overridden.returncode == 0
    assert "missing_as_of" in json.loads(overridden.stdout)["rows"][0]["reason_codes"]


def test_access_window_elapsed_blocks_new_submission() -> None:
    document = deepcopy(CASES["resource_dependent"])
    document["as_of"] = "2026-09-13T08:00:00Z"
    document["items"][0]["access_deadline_utc"] = "2026-09-12T20:00:00Z"
    row = tool.build_inventory([document])["rows"][0]
    assert row["recommended_action"] == "blocked_do_not_submit"
    assert "access_window_elapsed" in row["reason_codes"]


def test_item_order_does_not_change_the_report() -> None:
    document = deepcopy(CASES["window_actions"])
    shuffled = deepcopy(document)
    shuffled["items"] = list(reversed(shuffled["items"]))
    assert tool.build_inventory([document]) == tool.build_inventory([shuffled])


def test_malformed_schema_and_missing_check_flag_exit_two(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text('{"schema": "wrong"}', encoding="utf-8")
    result = _run_tool("--inputs", str(path), "--check", "--format", "json")
    assert result.returncode == 2
    assert "FAIL invalid_input" in result.stderr
    unchecked = _run_tool("--inputs", str(FIXTURE), "--case", "resource_dependent")
    assert unchecked.returncode == 2 and "--check is required" in unchecked.stderr
