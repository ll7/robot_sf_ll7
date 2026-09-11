"""Focused contract tests for the checkpoint compatibility audit (#8896)."""

from __future__ import annotations

import contextlib
import json
from io import StringIO
from pathlib import Path

import pytest

from scripts.models import audit_checkpoint_compatibility as tool

FIXTURES = Path(__file__).parent / "fixtures" / "checkpoint_audit"
ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "fixture_missing_v1": {"artifact_missing"},
    "fixture_wrong_digest_v1": {"digest_mismatch"},
    "fixture_wrong_normalizer_v1": {"normalizer_missing"},
    "fixture_normalizer_digest_v1": {"normalizer_digest_mismatch"},
    "fixture_wrong_observation_space_v1": {"observation_contract_mismatch"},
    "fixture_mutable_alias_v1": {"mutable_alias", "digest_not_pinned"},
    "fixture_unavailable_dependency_v1": {"dependency_unavailable"},
    "fixture_rights_blocked_v1": {"rights_blocked"},
    "fixture_nonfinite_v1": {"non_finite_parameters", "normalizer_missing"},
    "fixture_missing_custom_object_v1": {"missing_custom_object"},
    "fixture_lineage_unresolved_v1": {"lineage_unresolved"},
    "fixture_duplicate_v1": {"duplicate_model_id"},
}


def _run(argv):
    out = StringIO()
    with contextlib.redirect_stdout(out):
        code = tool.main(argv)
    return code, out.getvalue()


def _check(path, *, root=ROOT):
    argv = ["--input", str(path), "--check", "--format", "json", "--root", str(root)]
    code, stdout = _run(argv)
    return code, json.loads(stdout)


def _rows(payload):
    return {row["model_id"]: row for row in payload["models"]}


def _codes(payload):
    return {item["code"] for item in payload["findings"]}


def _document(models, consumers=()):
    return {"schema": tool.INPUT_SCHEMA, "models": list(models), "consumers": list(consumers)}


def _write(tmp_path, document, name="case.json"):
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_compatible_fixture_passes_and_records_public_state():
    code, payload = _check(FIXTURES / "compatible.json")
    row = _rows(payload)["fixture_compatible_v1"]
    assert (code, payload["status"]) == (0, "pass")
    assert row["state"] == "public" and row["load_status"] == "verified"
    assert row["reason_codes"] == [] and row["observation_contract"]["shape"] == [4]


@pytest.mark.parametrize(("model_id", "expected"), sorted(CASES.items()))
def test_failing_fixture_reports_each_required_case(model_id, expected):
    code, payload = _check(FIXTURES / "failing.json")
    codes = set(_rows(payload)[model_id]["reason_codes"])
    assert code == 1 and payload["status"] == "fail"
    assert expected <= codes and expected <= _codes(payload)


def test_consumer_gate_fails_unstaged_recoverable_model(tmp_path):
    models = [
        {
            "model_id": "unstaged_v1",
            "artifact_version": "v1",
            "artifact_sha256": "a" * 64,
            "locator_class": "cloud_durable",
            "algorithm_class": "unknown",
        }
    ]
    consumer = {"consumer_id": "configs/baselines/ppo.yaml", "required_model_ids": ["unstaged_v1"]}
    code, payload = _check(_write(tmp_path, _document(models, [consumer])))
    assert code == 1 and "consumer_model_load_unverified" in _codes(payload)


def test_output_is_deterministic_and_rows_are_sorted(tmp_path):
    first = _check(FIXTURES / "failing.json")[1]
    assert tool.render_json(first) == tool.render_json(_check(FIXTURES / "failing.json")[1])
    identifiers = [row["model_id"] for row in first["models"]]
    assert identifiers == sorted(identifiers)
    shuffled = json.loads((FIXTURES / "failing.json").read_text(encoding="utf-8"))
    shuffled["models"].reverse()
    reordered = _check(_write(tmp_path, shuffled, "r.json"))[1]
    assert tool.render_json(reordered) == tool.render_json(first)


def test_private_input_is_rejected_and_never_rendered(tmp_path):
    secret = "/home/private-host/model.zip"
    leaky = {"model_id": "leaky_v1", "artifact_path": secret, "locator_class": "unknown"}
    document = _document([leaky])
    code, stdout = _run(["--input", str(_write(tmp_path, document)), "--check", "--format", "json"])
    assert code == 2 and secret not in stdout
    assert "forbidden_value" in json.loads(stdout)["findings"][0]["code"]


def test_unknown_input_exits_two_and_markdown_is_concise(tmp_path):
    code, stdout = _run(["--input", str(tmp_path / "missing.json"), "--check", "--format", "json"])
    assert code == 2 and json.loads(stdout)["status"] == "unknown"
    markdown = [
        "--input",
        str(FIXTURES / "compatible.json"),
        "--format",
        "markdown",
        "--root",
        str(ROOT),
    ]
    code, stdout = _run(markdown)
    assert code == 0 and stdout.startswith("# Checkpoint Compatibility Audit")
    assert "fixture_compatible_v1" in stdout
