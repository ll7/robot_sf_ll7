"""Focused contract tests for the sanitized compute-window readiness dashboard (#8911)."""

from __future__ import annotations

import contextlib
import json
from io import StringIO
from pathlib import Path

import pytest

from scripts.tools import compute_window_readiness_dashboard as tool

FIX = Path(__file__).parent / "fixtures" / "compute_window_dashboard"
PASSING, FAILING = FIX / "inputs", FIX / "failing"
PRIVATE_PATH = "/scratch/private-user-42/campaign/results"
PRIVATE_USER = "researcher42"
PRIVATE_HOST = "gpu-node-17.cluster.invalid"
BASE = json.loads((PASSING / "passing_packet.json").read_text(encoding="utf-8"))
ROW = BASE["rows"][0]
REPORT = {**BASE, "report_id": "canonical-test-0001"}


def _row(**overrides):
    return {**ROW, **overrides}


def _report(**overrides):
    return {**REPORT, **overrides}


def _write(directory: Path, name: str, payload) -> None:
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def _run(argv):
    out = StringIO()
    with contextlib.redirect_stdout(out):
        code = tool.main(argv)
    return code, out.getvalue()


def _load(source):
    code, stdout = _run(["--inputs", str(source), "--check", "--format", "json"])
    return code, json.loads(stdout)


CASES = (
    ("passing-ready", PASSING, "c-6561-ped-speed", "scheduler", "not_submitted"),
    ("passing-running", PASSING, "c-7049-recovery", "scheduler", "running"),
    ("passing-terminal-unharvested", PASSING, "c-5409-h500", "artifact", "harvest_pending"),
    ("passing-preserved", PASSING, "c-6700-preserved", "artifact", "restore_verified"),
    ("passing-cluster-only", PASSING, "c-6127-cluster-only", "artifact", "cluster_only"),
    ("passing-blocked-gate", PASSING, "c-6561-scientific-gate", "evidence", "blocked"),
    ("failing-stale", FAILING, "c-stale-row", "display", "unavailable"),
    ("failing-conflicting", FAILING, "c-conflict-row", "display", "unavailable"),
)


@pytest.mark.parametrize(("case", "directory", "campaign_id", "field", "expected"), CASES)
def test_eight_fixture_cases_end_to_end(case, directory, campaign_id, field, expected):
    """Each required fixture case is exercised through the CLI and asserted."""
    code, payload = _load(directory)
    assert code == (0 if case.startswith("passing") else 2)
    entry = next(e for e in payload["campaigns"] if e["campaign_id"] == campaign_id)
    assert entry["display"] == ("available" if case.startswith("passing") else "unavailable")
    if field != "display":
        assert entry["states"][field] == expected
    if case.startswith("passing"):
        assert entry["reason_codes"] == []


def test_passing_packet_separates_axes_orders_rows_and_keeps_fields():
    """Seven axes stay separate; urgency, priority, and stable identity order rows."""
    code, payload = _load(PASSING)
    assert code == 0 and payload["ok"] and payload["status"] == "available"
    assert payload["summary"]["available_count"] == 6 and payload["summary"]["issue_count"] == 0
    assert [e["campaign_id"] for e in payload["campaigns"]] == [
        "c-6561-scientific-gate",
        "c-5409-h500",
        "c-6561-ped-speed",
        "c-7049-recovery",
        "c-6127-cluster-only",
        "c-6700-preserved",
    ]
    preserved = payload["campaigns"][-1]
    assert set(preserved["states"]) == set(tool.AXES)
    assert (preserved["states"]["artifact"], preserved["job_state"]) == (
        "restore_verified",
        "succeeded",
    )
    assert all(key in preserved for key in tool._ROW_KEYS)


def test_stale_and_conflicting_inputs_mask_rows_without_positive_display():
    """Stale and contradictory rows show explicit unavailable plus stable codes."""
    code, payload = _load(FAILING)
    assert code == 2 and payload["ok"] is False and payload["status"] == "unavailable"
    entries = {e["campaign_id"]: e for e in payload["campaigns"]}
    assert entries["c-stale-row"]["reason_codes"] == ["stale_input"]
    assert "conflicting_state" in entries["c-conflict-row"]["reason_codes"]
    for entry in (entries["c-stale-row"], entries["c-conflict-row"]):
        assert set(entry["states"].values()) == {tool.UNAVAILABLE}
        assert entry["job_state"] == tool.UNAVAILABLE
    inputs = {item["report_id"]: item for item in payload["inputs"]}
    stale = inputs["canonical-stale-0001"]
    assert stale["freshness"] == "stale" and stale["generated_at"] == "2026-09-01T00:00:00Z"
    assert stale["age_seconds"] > inputs["canonical-conflict-0003"]["age_seconds"]


def test_schema_duplicate_conflicting_and_private_violations_fail_closed(tmp_path, capsys):
    """Wrong schema, bad rows, duplicates, conflicts, and private values all reject."""
    for name in ("schema", "rows", "conflict", "dupes", "leak"):
        (tmp_path / name).mkdir()
    _write(tmp_path / "schema", "bad.json", _report(schema="wrong.v9", hostname=PRIVATE_HOST))
    code, payload = _load(tmp_path / "schema")
    codes = {issue["code"] for issue in payload["issues"]}
    assert code == 2 and {"invalid_schema", "unknown_field", "forbidden_field"} <= codes
    assert PRIVATE_HOST not in json.dumps(payload)

    row = _row()
    del row["priority"]
    row["not_a_field"] = "x"
    _write(tmp_path / "rows", "rows.json", _report(rows=[row]))
    code, payload = _load(tmp_path / "rows")
    assert code == 2 and {"missing_field", "unknown_field"} <= {
        i["code"] for i in payload["issues"]
    }

    both = {**ROW["states"], "scheduler": "running"}
    conflict = _row(states=both, harvest_state="harvested")
    _write(tmp_path / "conflict", "conflict.json", _report(rows=[conflict]))
    code, payload = _load(tmp_path / "conflict")
    assert code == 2 and "conflicting_state" in {i["code"] for i in payload["issues"]}

    _write(tmp_path / "dupes", "a.json", _report())
    _write(tmp_path / "dupes", "b.json", _report())
    code, payload = _load(tmp_path / "dupes")
    codes = {issue["code"] for issue in payload["issues"]}
    assert {"duplicate_report", "duplicate_campaign"} <= codes
    assert all(e["display"] == "unavailable" for e in payload["campaigns"])

    leaked = _row(
        public_owner=PRIVATE_HOST, next_owner=f"{PRIVATE_USER}@x", source_status=PRIVATE_PATH
    )
    _write(tmp_path / "leak", "leak.json", _report(rows=[leaked]))
    code, payload = _load(tmp_path / "leak")
    errors = capsys.readouterr().err
    assert code == 2 and "forbidden_value" in {i["code"] for i in payload["issues"]}
    combined = json.dumps(payload) + errors
    assert all(private not in combined for private in (PRIVATE_PATH, PRIVATE_USER, PRIVATE_HOST))


def test_repeated_renders_are_byte_stable_and_emit_no_score():
    """Unchanged inputs render identically and no scientific score is emitted."""
    first = tool.render_dashboard_json(tool.build_dashboard([PASSING]))
    second = tool.render_dashboard_json(tool.build_dashboard([PASSING]))
    assert first == second
    payload = json.loads(first)
    assert payload["claim_boundary"] == tool.CLAIM_BOUNDARY
    assert set(payload) == {
        "schema",
        "claim_boundary",
        "status",
        "ok",
        "as_of",
        "max_age_seconds",
        "summary",
        "inputs",
        "issues",
        "campaigns",
    }
    markdown = tool.render_dashboard_markdown(payload)
    for token in ("scheduler:", "artifact:", "evidence:", "review:", "claim:", "next_owner:"):
        assert token in markdown


def test_write_mode_as_of_override_and_empty_inputs_fail_closed(tmp_path):
    """Write mode emits JSON plus Markdown; as-of and empty input sets fail closed."""
    output = tmp_path / "dash.json"
    code, _ = _run(["--inputs", str(PASSING), "--output", str(output)])
    assert code == 0 and output.is_file() and output.with_suffix(".md").is_file()
    code, _ = _run(["--inputs", str(FAILING), "--output", str(tmp_path / "bad.json")])
    assert code == 2
    args = [
        "--inputs",
        str(PASSING),
        "--check",
        "--format",
        "json",
        "--as-of",
        "2026-09-01T00:00:00Z",
    ]
    code, payload = _run(args)
    payload = json.loads(payload)
    assert code == 2 and {i["code"] for i in payload["issues"]} == {"stale_input"}
    assert all(e["display"] == "unavailable" for e in payload["campaigns"])
    empty = tmp_path / "empty"
    empty.mkdir()
    code, payload = _load(empty)
    assert code == 2 and {i["code"] for i in payload["issues"]} == {"missing_inputs"}
    assert tool.main(["--inputs", str(PASSING), "--as-of", "not-a-time"]) == 2
