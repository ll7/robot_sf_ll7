"""Focused contract tests for the sanitized lineage index (#8897)."""

from __future__ import annotations

import contextlib
import copy
import json
from io import StringIO
from pathlib import Path

import pytest

from scripts.tools import lineage_index as tool

FIX = Path(__file__).parent / "fixtures" / "lineage_index"
COMPLETE = FIX / "complete.json"
BASE = json.loads(COMPLETE.read_text(encoding="utf-8"))
EXPECTED_MD = (FIX / "complete.expected.md").read_text(encoding="utf-8")


def _sha(char):
    return char * 64


def _run(argv):
    out = StringIO()
    with contextlib.redirect_stdout(out):
        code = tool.main(argv)
    return code, out.getvalue()


def _check(path):
    code, stdout = _run(["--input", str(path), "--check", "--format", "json"])
    return code, json.loads(stdout)


def _write(tmp_path, records=None, projection=None, name="case.json"):
    doc = copy.deepcopy(BASE)
    if records is not None:
        doc["records"] = records
    if projection is not None:
        doc["private_projection"] = projection
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _rows(payload):
    return {row["lineage_key"]: row for row in payload["rows"]}


def _codes(payload):
    return {finding["code"] for finding in payload["findings"]}


def _missing(row):
    return {entry["category"]: entry["class"] for entry in row["missing_links"]}


def _node(kind, ident, **fields):
    return {"kind": kind, "id": ident, **fields}


def _artifact(ident, char, kind="raw", locator="public_release", **refs):
    return _node(
        "artifact", ident, digest=_sha(char), artifact_kind=kind, locator_class=locator, refs=refs
    )


def _job(ident, relation, refs):
    return _node(
        "job", ident, attempt_index=2, relation=relation, predecessor_job_id="job-1", refs=refs
    )


def _case(case):
    records = copy.deepcopy(BASE["records"])
    projection = []
    if case == "retry":
        retry_job = _job(
            "job-2", "retry", {"campaign_ids": ["camp-newline"], "artifact_ids": ["art-raw-2"]}
        )
        retry_job["not_applicable"] = ["checkpoint", "environment"]
        records += [retry_job, _artifact("art-raw-2", "c", job_ids=["job-2"])]
    elif case == "resume":
        records += [
            _job("job-b2", "resume", {"config_ids": ["cfg-newline"], "artifact_ids": ["art-c2"]}),
            _artifact("art-c2", "d", kind="compact", locator="tracked_path", job_ids=["job-b2"]),
        ]
    elif case == "private_locator":
        records += [_artifact("art-ckpt-2", "e", locator="private_overlay", job_ids=["job-1"])]
        projection = [{"target": "artifact:art-ckpt-2", "digest": _sha("e"), "withheld": True}]
    elif case == "missing_receipt":
        next(item for item in records if item["id"] == "job-1")["submission_receipt"] = (
            "not_recorded"
        )
    elif case == "digest_conflict":
        records += [_node("config", "cfg-newline", digest=_sha("f"), owner="benchmark")]
    elif case == "deterministic":
        records = list(reversed(records))
    return records, projection


CASES = (
    "complete retry resume private_locator missing_receipt digest_conflict deterministic".split()
)


@pytest.mark.parametrize("case", CASES)
def test_seven_fixture_cases_end_to_end(case, tmp_path):
    records, projection = _case(case)
    code, payload = _check(_write(tmp_path, records, projection))
    rows = _rows(payload)
    assert code == (2 if case == "digest_conflict" else 0)
    if case == "complete":
        assert payload["ok"] and payload["summary"]["record_count"] == 14
        row = rows["job:job-1"]
        assert row["attempt_index"] == 1 and row["relation"] == "initial"
        assert row["predecessor_job_id"] is None
        assert row["missing_links"] == [] and row["reason_codes"] == []
        assert {field for field, values in row["records"].items() if not values} == set()
        assert row["records"]["issue_ids"] == ["8897"] and row["records"]["claim_ids"] == [
            "claim-1"
        ]
        assert row["artifact_digests"] == [_sha("a"), _sha("b")]
        assert row["owners"]["claim"] == ["paper"]
        assert payload["reverse_lookup"]["issue"]["8897"] == ["job:job-1"]
        assert payload["reverse_lookup"]["artifact_digest"][_sha("a")] == ["job:job-1"]
        assert payload["reverse_lookup"]["commit"]["c0ffee1"] == ["job:job-1"]
        _, markdown = _run(["--input", str(COMPLETE), "--check", "--format", "markdown"])
        assert markdown == EXPECTED_MD
    elif case == "retry":
        assert set(rows) == {"job:job-1", "job:job-2"}
        retry = rows["job:job-2"]
        assert retry["attempt_index"] == 2 and retry["relation"] == "retry"
        assert retry["predecessor_job_id"] == "job-1"
        assert retry["artifact_digests"] == [_sha("c")]
        assert _missing(retry)["checkpoint"] == "not_applicable"
        assert rows["job:job-1"]["artifact_digests"] == [_sha("a"), _sha("b")]
        assert payload["reverse_lookup"]["campaign"]["camp-newline"] == ["job:job-1", "job:job-2"]
    elif case == "resume":
        assert set(rows) == {"job:job-1", "job:job-b2"}
        row = rows["job:job-b2"]
        assert row["relation"] == "resume" and row["predecessor_job_id"] == "job-1"
        assert row["records"]["config_ids"] == ["cfg-newline"]
    elif case == "private_locator":
        assert _missing(rows["job:job-1"])["artifact_locator"] == "private_unavailable"
        blob = json.dumps(payload)
        assert all(token not in blob for token in ("/scratch/", "@", "gpu-node", "https://"))
    elif case == "missing_receipt":
        row = rows["job:job-1"]
        assert _missing(row)["submission_receipt"] == "not_recorded"
        assert row["reason_codes"] == ["not_recorded"]
    elif case == "digest_conflict":
        assert {"duplicate_semantic_id", "source_identity_conflict"} <= _codes(payload)
        assert _missing(rows["job:job-1"])["config"] == "conflict"
    else:
        _, canonical = _check(_write(tmp_path, None, name="canonical.json"))
        assert payload == canonical


def test_dangling_orphan_and_projection_drift_fail_closed(tmp_path):
    records = copy.deepcopy(BASE["records"]) + [
        {"kind": "environment", "id": "env-2", "refs": {"job_ids": ["job-404"]}},
        {"kind": "analysis", "id": "analysis-2", "refs": {"artifact_ids": ["art-404"]}},
    ]
    drift = [{"target": "artifact:art-raw-1", "digest": _sha("f"), "withheld": True}]
    code, payload = _check(_write(tmp_path, records, drift))
    assert code == 2
    assert {"dangling_reference", "orphaned_artifact_pointer", "projection_drift"} <= _codes(
        payload
    )
    rows = _rows(payload)
    assert _missing(rows["environment:env-2"])["job"] == "dangling"
    assert _missing(rows["analysis:analysis-2"])["artifact"] == "dangling"


def test_forbidden_private_value_fails_closed_without_echo(tmp_path):
    secret = "/scratch/private-user-42/results"
    records = copy.deepcopy(BASE["records"])
    records[0]["owner"] = secret
    code, payload = _check(_write(tmp_path, records))
    assert code == 2 and "forbidden_value" in _codes(payload)
    assert secret not in json.dumps(payload)


def test_query_modes_resolve_stable_identities_and_fail_on_no_match(tmp_path):
    path = _write(tmp_path)
    selectors = (
        ("--issue", "8897"),
        ("--job", "job-1"),
        ("--campaign", "camp-newline"),
        ("--commit", "c0ffee1"),
        ("--config", "cfg-newline"),
        ("--artifact-digest", _sha("a")),
    )
    for flag, value in selectors:
        code, stdout = _run(["query", "--input", str(path), flag, value, "--format", "json"])
        assert code == 0 and json.loads(stdout)["match_count"] == 1
    code, _ = _run(["query", "--input", str(path), "--issue", "9999", "--format", "json"])
    assert code == 2
