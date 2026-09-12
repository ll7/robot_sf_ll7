"""Tests for generate_post_access_handoff (#8830)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.tools.generate_post_access_handoff import (
    WORKLOAD_STATES,
    generate_handoff,
    main,
    sanitize_text,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "post_access_handoff"
CASES_FILE = FIXTURES_DIR / "cases.json"
CASES_DATA = json.loads(CASES_FILE.read_text(encoding="utf-8")) if CASES_FILE.is_file() else {}


@pytest.fixture
def base_inventory_data() -> dict:
    inv_path = FIXTURES_DIR / "canonical_inventory.json"
    return json.loads(inv_path.read_text(encoding="utf-8"))


def test_canonical_fixture_standalone() -> None:
    inv_path = FIXTURES_DIR / "canonical_inventory.json"
    report = generate_handoff(inv_path)
    assert report["verdict"] == "pass"
    assert report["summary"]["total_workloads"] == 2
    assert report["summary"]["total_artifacts"] == 1
    assert report["summary"]["total_environments"] == 1
    assert not report["discrepancies"]
    assert not report["reasons"]


def test_cli_invocation_and_formats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    inv_path = FIXTURES_DIR / "canonical_inventory.json"

    assert main(["--inventory", str(inv_path), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "pass"

    assert main(["--inventory", str(inv_path), "--format", "markdown"]) == 0
    out_md = capsys.readouterr().out
    assert "# Compute Window Post-Access Handoff" in out_md
    assert "#5416" in out_md

    assert main(["--inventory", str(inv_path), "--format", "text"]) == 0
    assert "Verdict: PASS" in capsys.readouterr().out

    out_file = tmp_path / "summary.md"
    assert (
        main(["--inventory", str(inv_path), "--format", "markdown", "--output", str(out_file)]) == 0
    )
    assert out_file.is_file()
    assert "# Compute Window Post-Access Handoff" in out_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("state", WORKLOAD_STATES)
def test_all_terminal_states(tmp_path: Path, base_inventory_data: dict, state: str) -> None:
    data = copy.deepcopy(base_inventory_data)
    data["workloads"][0]["current_state"] = state
    if state == "safe_to_defer":
        data["workloads"][0]["next_command"] = None
    elif state == "not_submitted":
        data["workloads"][0]["scheduler_job"] = None
        data["workloads"][0]["next_command"] = "sbatch run.sh"
        data["scheduler_receipts"] = [data["scheduler_receipts"][1]]
    else:
        data["workloads"][0]["next_command"] = f"check {state}"

    p = tmp_path / f"state_{state}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    report = generate_handoff(p)
    assert report["verdict"] == "pass"
    assert report["summary"]["states"][state] >= 1


def _mutate_state(data: dict, key: str) -> None:
    if key == "missing_next_command":
        data["workloads"][1]["next_command"] = None
    elif key == "contradictory_restore":
        data["workloads"][0]["restore_status"] = "blocked"
    elif key == "contradictory_row_count":
        data["workloads"][0]["observed_rows"] = 50
    elif key == "contradictory_not_submitted_job":
        data["workloads"][0]["current_state"] = "not_submitted"
        data["workloads"][0]["next_command"] = "sbatch job.sh"
    elif key == "orphan_job":
        data["scheduler_receipts"].append({"job_id": 9999, "workload_ref": 9999})
    elif key == "orphan_artifact":
        data["artifacts"].append(
            {
                "artifact_id": "art-orphan-999",
                "retention_class": "disposable",
                "size_bytes": 100,
                "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "custody_state": "missing",
                "consumer": "9999",
            }
        )


def _mutate_identity_and_leaks(data: dict, key: str) -> None:
    if key == "duplicate_identity":
        data["workloads"].append(copy.deepcopy(data["workloads"][0]))
    elif key == "invalid_state":
        data["workloads"][0]["current_state"] = "undefined_mystery_state"
    elif key == "missing_issue":
        data["workloads"][0]["public_issue"] = None
    elif key == "private_path_leak":
        data["workloads"][0]["identities"]["config_path"] = "/home/luttkule/secret.json"
    elif key == "credential_leak":
        data["workloads"][0]["purpose"] = "test AWS_SECRET_ACCESS_KEY=abcd1234efgh"
    elif key == "signed_url_leak":
        data["workloads"][0]["purpose"] = "download https://s3.local/b?X-Amz-Signature=1234"
    elif key == "private_host_leak":
        data["workloads"][0]["purpose"] = "run on worker1.cluster"


def _apply_case_mutation(data: dict, key: str) -> None:
    _mutate_state(data, key)
    _mutate_identity_and_leaks(data, key)


@pytest.mark.parametrize("mutation_key", list(CASES_DATA.keys()))
def test_validation_and_redaction_cases(
    tmp_path: Path,
    base_inventory_data: dict,
    mutation_key: str,
) -> None:
    expected = CASES_DATA[mutation_key]
    data = copy.deepcopy(base_inventory_data)
    _apply_case_mutation(data, mutation_key)
    p = tmp_path / f"{mutation_key}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    report = generate_handoff(p)
    assert report["verdict"] == expected["verdict"]
    for reason in expected["reasons"]:
        assert reason in report["reasons"]


def test_missing_inventory_and_malformed_json(tmp_path: Path) -> None:
    rep_missing = generate_handoff(tmp_path / "nonexistent.json")
    assert rep_missing["verdict"] == "blocked"
    assert "inventory_not_found" in rep_missing["reasons"]

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{bad json", encoding="utf-8")
    rep_malformed = generate_handoff(malformed)
    assert rep_malformed["verdict"] == "fail"
    assert "unsupported_schema" in rep_malformed["reasons"]


def test_sanitize_text() -> None:
    raw = "Path /home/user/file key AWS_SECRET_ACCESS_KEY url https://x?token=123 host node.cluster"
    sanitized = sanitize_text(raw)
    assert "/home/user" not in sanitized
    assert "AWS_SECRET_ACCESS_KEY" not in sanitized
    assert "token=123" not in sanitized
    assert "node.cluster" not in sanitized
