"""Protect portable acquired-anchor evidence without resetting or stepping anything."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/context/evidence/2026-10-04_freeze008_calibration"
SPEC = importlib.util.spec_from_file_location(
    "determinism_receipt", ROOT / "scripts/dev/build_snqi_v2_determinism_receipt.py"
)
assert SPEC and SPEC.loader
RECEIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIPT)


def sample_row():
    """Return a minimal recorded row, with a fixed independent scalar example."""
    return {
        "metrics": {"robot_force_impulse_total": 2.0, "jerk_mean": 3.0, "curvature_mean": 4.0},
        "metric_values": {"clearance": float("nan")},
        "steps": 12,
        "status": "success",
        "wall_time": 1.0,
    }


def test_acquisition_proof_has_guard_counters_and_inert_source_audit():
    """The delivered proof must expose safety arbitration, not just mixed execution mode."""
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    comparison = proof["rehearsal_comparison"]
    assert "changed_runtime_config_paths" not in comparison
    assert comparison["changed_source_paths_behaviourally_inert_for_this_grid"] is True
    assert comparison["changed_source_paths_behaviourally_inert_reason"]
    assert comparison["changed_source_paths"]
    counters = proof["guard_arbitration_counts"]["guarded_ppo"]
    assert counters["fallback_safe"] > 0
    assert counters["ppo_clear"] > 0
    assert all(type(value) is int and value >= 0 for value in counters.values())


def test_delivered_determinism_receipt_binds_all_paired_rows():
    """The public claim must bind the actual complete comparison, not a summary-only receipt."""
    proof = json.loads((EVIDENCE / "acquisition-proof.json").read_text())
    binding = proof["determinism_receipt"]
    raw = (ROOT / binding["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
    receipt = json.loads(raw)
    comparison = receipt["original_vs_repeat"]
    pairs = comparison["row_hashes"]
    assert len(pairs) == comparison["rows"] == 1344
    assert len({(row["arm"], row["scenario"], row["seed"]) for row in pairs}) == 1344
    assert {row["seed"] for row in pairs} == {1001, 1002}
    different = sum(row["left_sha256"] != row["right_sha256"] for row in pairs)
    assert different == comparison["different_rows"] == len(comparison["differences"])
    same_environment = (
        receipt["execution_contexts"]["original"] == receipt["execution_contexts"]["repeat"]
    )
    assert receipt["same_recorded_environment"] == same_environment
    if receipt["classification"] == "a":
        assert same_environment and different == 0
    elif receipt["classification"] == "b":
        assert same_environment and different > 0
    else:
        assert not same_environment


def test_metric_row_hash_includes_metrics_steps_status_and_excludes_wall_time():
    """A trajectory change must affect the receipt; bookkeeping time must not."""
    row = sample_row()
    before = RECEIPT.metric_row_sha256(row)
    row["wall_time"] = 99.0
    assert RECEIPT.metric_row_sha256(row) == before
    for field, value in (("steps", 13), ("status", "collision")):
        changed = {**row, field: value}
        assert RECEIPT.metric_row_sha256(changed) != before
    row["metrics"]["jerk_mean"] = 3.0000000000000004
    assert RECEIPT.metric_row_sha256(row) != before


def test_signed_zero_and_nonfinite_sentinels_have_stable_hashes():
    """No approximate equality or JSON NaN extension can conceal a metric-byte change."""
    row = sample_row()
    row["metric_values"] = {"zero": -0.0, "nonfinite": float("inf")}
    expected_payload = {
        "metrics": {
            "robot_force_impulse_total": {"float64_hex": "0x1.0000000000000p+1"},
            "jerk_mean": {"float64_hex": "0x1.8000000000000p+1"},
            "curvature_mean": {"float64_hex": "0x1.0000000000000p+2"},
        },
        "metric_values": {
            "zero": {"float64_hex": "-0x0.0p+0"},
            "nonfinite": {"float64_hex": "inf"},
        },
        "steps": 12,
        "status": "success",
    }
    expected = hashlib.sha256(
        json.dumps(expected_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert RECEIPT.metric_row_sha256(row) == expected
    row["metric_values"]["zero"] = 0.0
    assert RECEIPT.metric_row_sha256(row) != expected


def test_comparison_reports_changed_columns_steps_and_status():
    """Every divergent row remains actionable, including a missing/null column change."""
    key = ("ppo", "classic_doorway_high", 1001)
    left, right = sample_row(), sample_row()
    right["steps"], right["status"] = 13, "collision"
    right["metric_values"]["extra"] = None
    right["metrics"]["jerk_mean"] = 5.0
    result = RECEIPT.compare_rows({key: left}, {key: right})
    assert (
        result["different_rows"] == result["step_differences"] == result["status_differences"] == 1
    )
    difference = result["differences"][0]
    assert difference["left"] == {"steps": 12, "status": "success"}
    assert difference["right"] == {"steps": 13, "status": "collision"}
    assert difference["changed_metric_columns"] == ["metrics.jerk_mean", "metric_values.extra"]


def test_comparison_refuses_missing_row():
    """An incomplete repeat cannot be described as reproducible."""
    with pytest.raises(ValueError, match="comparison grids differ"):
        RECEIPT.compare_rows({("ppo", "doorway", 1001): sample_row()}, {})


@pytest.mark.parametrize(
    ("different_rows", "same_environment", "expected"),
    [
        (0, True, "a"),
        (1, True, "b"),
        (0, False, "unresolved_environment"),
        (1, False, "unresolved_environment"),
    ],
)
def test_repeat_classification_requires_matching_environment(
    different_rows, same_environment, expected
):
    """Neither a reproducibility pass nor same-environment nondeterminism can mask an env change."""
    assert RECEIPT.classify_repeat(different_rows, same_environment) == expected


def test_interpolated_p95_need_not_be_a_raw_sample():
    """The interpolation itself explains an absent raw p95 value without implying lost rows."""
    left, right = sample_row(), sample_row()
    left["metrics"]["jerk_mean"], right["metrics"]["jerk_mean"] = 1.0, 3.0
    data = {("ppo", "doorway", 1001): left, ("ppo", "doorway", 1002): right}
    detail = RECEIPT.jerk_percentile_details(data, 2.0)
    assert detail["zero_based_index"] == 0.95
    assert detail["p95"] == 2.9
    assert detail["rows_above_original_p95"] == 1
