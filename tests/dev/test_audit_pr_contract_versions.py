"""Tests for the PR contract-version inventory tool (issue #7892 Part A)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.dev.audit_pr_contract_versions import (
    REPORT_SCHEMA,
    _check_inventory,
    _classify_pr,
    build_inventory,
)
from scripts.dev.pr_contract_v2 import parse_pr_contract_v2

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "dev" / "audit_pr_contract_versions.py"

V2_TOOLING_PAYLOAD = """change_class: tooling
linked_issues:
  closes: []
  relates: []
deferred_work:
  status: none
  issues: []
evidence:
  applicability: na
  tier: null
  result: na
domain_approval:
  required: false
  status: not_required
performance:
  claimed: false"""


def _v2_body(payload: str = V2_TOOLING_PAYLOAD) -> str:
    return f"""## Summary
Human-authored summary for a v2 PR.

<!-- pr-contract:v2
{payload}
-->
"""


def _v1_body() -> str:
    return """## Summary
A v1-compatibility PR body without any pr-contract:v2 marker.
"""


def _record(
    number: int,
    body: str | None,
    *,
    url: str = "https://github.com/ll7/robot_sf_ll7/pull/1",
    author: str = "ll7",
    is_draft: bool = False,
    head_sha: str = "a" * 40,
) -> dict:
    return {
        "number": number,
        "url": url,
        "author": author,
        "isDraft": is_draft,
        "headRefOid": head_sha,
        "body": body,
    }


def test_script_is_python_and_importable() -> None:
    assert SCRIPT.exists()
    assert SCRIPT.suffix == ".py"
    import scripts.dev.audit_pr_contract_versions  # noqa: F401


def test_classify_valid_v2_marker() -> None:
    row = _classify_pr(_record(1, _v2_body()))
    assert row["contract_class"] == "v2_valid"
    assert row["contract_status"] == "ok"
    assert row["errors"] == []
    assert row["draft"] is False
    assert row["head_sha"] == "a" * 40


def test_classify_malformed_v2_marker_with_no_v1_fallback() -> None:
    body = _v2_body("change_class: tooling\n  bad_indent: [unclosed")
    row = _classify_pr(_record(2, body))
    assert row["contract_class"] == "v2_invalid"
    assert row["contract_status"] == "malformed"
    assert row["errors"]
    # The canonical parser never falls back to v1 for a present marker.
    assert parse_pr_contract_v2(body, source="fixture").status == "malformed"


def test_classify_v1_compatibility_body() -> None:
    row = _classify_pr(_record(3, _v1_body()))
    assert row["contract_class"] == "v1_compatibility"
    assert row["contract_status"] == "absent"


def test_classify_missing_body() -> None:
    row = _classify_pr(_record(4, None))
    assert row["contract_class"] == "body_missing"
    assert row["contract_status"] == "absent"


def test_classify_duplicate_pr_numbers_are_reported_independently() -> None:
    first = _classify_pr(_record(5, _v1_body()))
    second = _classify_pr(_record(5, _v2_body()))
    assert first["contract_class"] == "v1_compatibility"
    assert second["contract_class"] == "v2_valid"
    assert first["number"] == second["number"] == 5


def test_inventory_counts_and_schema() -> None:
    report = build_inventory(
        [
            _record(1, _v2_body()),
            _record(2, _v1_body()),
            _record(3, _v1_body()),
            _record(4, None),
        ],
        source="fixture",
    )
    assert report["schema"] == REPORT_SCHEMA
    assert report["source"] == "fixture"
    assert report["counts"] == {"v2_valid": 1, "v1_compatibility": 2, "body_missing": 1}
    assert report["retirement_policy"] == "compatibility"
    assert len(report["prs"]) == 4


def test_check_passes_for_valid_and_v1_compatible_prs() -> None:
    report = build_inventory([_record(1, _v2_body()), _record(2, _v1_body())], source="fixture")
    exit_code, problems = _check_inventory(report)
    assert exit_code == 0
    assert problems == []


def test_check_fails_closed_on_malformed_v2_marker() -> None:
    report = build_inventory(
        [_record(1, _v2_body("change_class: bogus_class")), _record(2, _v1_body())],
        source="fixture",
    )
    exit_code, problems = _check_inventory(report)
    assert exit_code == 1
    assert len(problems) == 1
    assert "PR 1" in problems[0]
    assert "change_class" in problems[0]


def test_credential_safe_output(tmp_path: Path) -> None:
    body = _v1_body() + "\nhttps://hooks.example.com/secret-token-abc\n"
    report = build_inventory([_record(1, body)], source="fixture")
    text = json.dumps(report)
    assert "secret-token-abc" not in text
    assert "hooks.example.com" not in text
    assert "token" not in text.lower()


def test_cli_fixture_mode_is_deterministic(tmp_path: Path) -> None:
    fixture = tmp_path / "prs.json"
    records = [
        _record(1, _v2_body()),
        _record(2, _v1_body()),
        _record(3, None),
    ]
    fixture.write_text(json.dumps(records), encoding="utf-8")
    first = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(fixture)],
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(fixture)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0
    assert first.stdout == second.stdout
    report = json.loads(first.stdout)
    assert report["counts"] == {"v2_valid": 1, "v1_compatibility": 1, "body_missing": 1}


def test_cli_check_mode_fails_on_malformed_marker(tmp_path: Path) -> None:
    fixture = tmp_path / "prs.json"
    fixture.write_text(
        json.dumps([_record(1, _v2_body("change_class: not_a_class"))]),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(fixture), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "check failed" in proc.stderr


def test_cli_output_file(tmp_path: Path) -> None:
    fixture = tmp_path / "prs.json"
    out = tmp_path / "report.json"
    fixture.write_text(json.dumps([_record(1, _v2_body())]), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(fixture), "--output", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["counts"] == {"v2_valid": 1}


def test_cli_requires_repo_or_fixture() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert proc.returncode != 0
    assert "one of --repo or --fixture is required" in proc.stderr
