"""Focused tests for the cross-host conformance capsule and comparator (issue #8914)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation import cross_host_conformance_capsule as capsule

FIXTURE = Path(__file__).parent / "fixtures" / "conformance_capsule" / "classic_crossing_low.json"


def _fields() -> dict:
    return {
        "identity/source_commit": {"class": "exact", "value": "a" * 40},
        "identity/scenario_id": {"class": "exact", "value": "classic_crossing_low"},
        "identity/seed": {"class": "exact", "value": 42},
        "structure/step_count": {"class": "exact", "value": 8},
        "structure/pedestrian_ids": {
            "class": "set",
            "value": ["simulator-slot-1", "simulator-slot-0"],
        },
        "checksums/config_hash": {"class": "exact", "value": "fa8d02918c1a01cc"},
        "numeric/avg_speed_m_s": {
            "class": "tolerance",
            "value": 0.25,
            "abs_tol": 1e-9,
            "rel_tol": 1e-6,
        },
        "informational/run_wall_time_sec": {"class": "informational", "value": 4.2},
    }


def _receipt(host_alias: str = "host_a", fields: dict | None = None, **overrides) -> dict:
    receipt = {
        "schema_version": capsule.RECEIPT_SCHEMA_VERSION,
        "capsule_id": "classic_crossing_low_goal_seed42_h8",
        "capsule_digest": "d" * 64,
        "host_alias": host_alias,
        "environment_class": {"os_class": "linux", "architecture_class": "x86_64"},
        "execution": {
            "status": "completed",
            "failure_reason": None,
            "fallback_degraded_status": "native",
            "degraded": False,
            "degraded_reasons": [],
            "fallback_markers": [],
        },
        "fields": fields if fields is not None else _fields(),
    }
    return {**receipt, **overrides}


def _mutated(path: str, value) -> dict:
    fields = _fields()
    return {**fields, path: {**fields[path], "value": value}}


def test_exact_match_with_reordered_collection_and_field_order_is_conformant():
    fields_b = _fields()
    fields_b["structure/pedestrian_ids"] = {
        "class": "set",
        "value": ["simulator-slot-0", "simulator-slot-1"],
    }
    shuffled = dict(reversed(list(fields_b.items())))
    report = capsule.compare_receipts(_receipt("host_b", fields=shuffled), _receipt("host_a"))
    assert report["status"] == "conformant"
    assert report["reason_codes"] == []


@pytest.mark.parametrize(
    "delta, expected", [(1e-9, "conformant"), (1e-3, "numeric_out_of_tolerance")]
)
def test_declared_numeric_tolerance_boundary(delta, expected):
    fields = _fields()
    fields["numeric/avg_speed_m_s"]["value"] += delta
    report = capsule.compare_receipts(_receipt("host_a", fields=fields), _receipt("host_b"))
    assert report["status"] == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.update({"capsule_id": "other"}),
        lambda r: r.update({"capsule_digest": "e" * 64}),
        lambda r: r["fields"].update(_mutated("identity/source_commit", "f" * 40)),
        lambda r: r["fields"].update(_mutated("identity/seed", 43)),
    ],
)
def test_source_config_and_seed_mismatch_is_identity_mismatch(mutate):
    first, second = _receipt("host_a"), _receipt("host_b")
    mutate(first)
    report = capsule.compare_receipts(first, second)
    assert report["status"] == "identity_mismatch" and report["reason_codes"]


@pytest.mark.parametrize(
    "path, value, reason",
    [
        ("structure/step_count", 9, "exact_value_mismatch"),
        ("structure/pedestrian_ids", ["simulator-slot-0"], "set_value_mismatch"),
        ("checksums/config_hash", "deadbeefdeadbeef", "exact_value_mismatch"),
    ],
)
def test_structural_mismatch_fields(path, value, reason):
    report = capsule.compare_receipts(
        _receipt("host_a", fields=_mutated(path, value)), _receipt("host_b")
    )
    assert report["status"] == "structural_mismatch" and reason in report["reason_codes"]


def test_unavailable_metric_is_unknown():
    fields = _fields()
    fields["numeric/avg_speed_m_s"] = {"class": "unavailable", "reason": "metric_not_emitted"}
    report = capsule.compare_receipts(_receipt("host_a", fields=fields), _receipt("host_b"))
    assert report["status"] == "unknown" and report["reason_codes"] == ["field_unavailable"]


@pytest.mark.parametrize(
    "update",
    [
        {"status": "fallback"},
        {"fallback_degraded_status": "fallback"},
        {"degraded": True, "degraded_reasons": ["degenerate_planner_view"]},
        {"fallback_markers": ["planner=fallback"]},
        {"failure_reason": "structure_contract_violation: step_count_out_of_contract"},
    ],
)
def test_hidden_fallback_or_degraded_fails_comparison(update):
    receipt = _receipt("host_b")
    receipt["execution"].update(update)
    report = capsule.compare_receipts(_receipt("host_a"), receipt)
    assert report["status"] == "environment_incompatible"
    assert "execution_integrity" in report["reason_codes"]


def test_report_is_symmetric_and_deterministically_ordered():
    fields_a = _mutated("structure/step_count", 9)
    fields_a["checksums/config_hash"] = {"class": "exact", "value": "deadbeefdeadbeef"}
    first, second = _receipt("host_a", fields=fields_a), _receipt("host_b")
    forward = capsule.compare_receipts(first, second)
    backward = capsule.compare_receipts(second, first)
    assert forward == backward
    issue_codes = [issue["code"] for issue in forward["issues"]]
    assert issue_codes == sorted(issue_codes)
    assert forward["reason_codes"] == sorted(set(forward["reason_codes"]))


def test_environment_class_policy_is_informational_by_default():
    receipt = _receipt("host_b")
    receipt["environment_class"]["os_class"] = "darwin"
    first = _receipt("host_a")
    assert capsule.compare_receipts(first, receipt)["status"] == "conformant"
    report = capsule.compare_receipts(first, receipt, require_same_environment_class=True)
    assert report["status"] == "environment_incompatible"


def test_fixture_spec_loads_and_deterministic_controls_fail_closed(tmp_path):
    spec = capsule.load_capsule_spec(FIXTURE)
    assert spec["horizon"] == 8 and spec["scenario"]["id"] == "classic_crossing_low"
    spec["deterministic_controls"]["workers"] = 2
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(capsule.CapsuleContractError):
        capsule.load_capsule_spec(drifted)


def test_compare_cli_emits_deterministic_json(tmp_path, capsys):
    path_a, path_b = tmp_path / "a.json", tmp_path / "b.json"
    path_a.write_text(json.dumps(_receipt("host_a")), encoding="utf-8")
    path_b.write_text(json.dumps(_receipt("host_b")), encoding="utf-8")
    assert capsule.main(["compare", str(path_a), str(path_b), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "conformant"


def test_receipt_sanitizes_private_execution_details(monkeypatch):
    def _boom(spec):
        raise RuntimeError("failed under /home/secretuser at 10.1.2.3 for alice@private-host")

    monkeypatch.setattr(capsule, "_execute_bounded_episode", _boom)
    receipt = capsule.run_capsule(capsule.load_capsule_spec(FIXTURE), "host_a")
    text = json.dumps(receipt)
    assert "/home/secretuser" not in text
    assert "10.1.2.3" not in text and "alice@private-host" not in text
    assert receipt["execution"]["status"] == "failed"
