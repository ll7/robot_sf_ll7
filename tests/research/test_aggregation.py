"""Unit tests for metric aggregation module."""

import json
from pathlib import Path

import pytest

from robot_sf.research.aggregation import (
    _load_manifest_payload,
    aggregate_metrics,
    bootstrap_ci,
    compute_completeness_score,
    export_metrics_csv,
    export_metrics_json,
    extract_seed_metrics,
)


def test_aggregate_metrics_basic():
    """Test basic metric aggregation with minimal data."""
    metric_records = [
        {
            "seed": 42,
            "policy_type": "baseline",
            "variant_id": 1,
            "success_rate": 0.7,
            "timesteps_to_convergence": 500000,
        },
        {
            "seed": 123,
            "policy_type": "baseline",
            "variant_id": 2,
            "success_rate": 0.75,
            "timesteps_to_convergence": 480000,
        },
        {
            "seed": 42,
            "policy_type": "pretrained",
            "variant_id": 3,
            "success_rate": 0.85,
            "timesteps_to_convergence": 280000,
        },
        {
            "seed": 123,
            "policy_type": "pretrained",
            "variant_id": 4,
            "success_rate": 0.88,
            "timesteps_to_convergence": 270000,
        },
    ]

    result = aggregate_metrics(metric_records, group_by="policy_type", ci_samples=100, seed=42)

    # Check that we have results for both conditions
    conditions = {r["condition"] for r in result}
    assert "baseline" in conditions
    assert "pretrained" in conditions

    # Check metrics are present
    metrics = {r["metric_name"] for r in result}
    assert "success_rate" in metrics
    assert "timesteps_to_convergence" in metrics

    # Assert non-metric/metadata columns are NOT aggregated as metrics
    assert "seed" not in metrics
    assert "policy_type" not in metrics
    assert "variant_id" not in metrics

    # Check baseline success_rate aggregation
    baseline_success = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "success_rate"
    )
    assert baseline_success["mean"] == pytest.approx(0.725, abs=1e-6)
    assert baseline_success["median"] == pytest.approx(0.725, abs=1e-6)
    assert baseline_success["p95"] == pytest.approx(0.7475, abs=1e-6)
    assert baseline_success["std"] == pytest.approx(0.035355, abs=1e-5)
    assert baseline_success["sample_size"] == 2
    assert baseline_success["ci_low"] == pytest.approx(0.7, abs=1e-6)
    assert baseline_success["ci_high"] == pytest.approx(0.75, abs=1e-6)
    assert baseline_success["ci_confidence"] == 0.95
    assert baseline_success["effect_size"] is None


def test_aggregate_metrics_single_value():
    """Test aggregation with single value per condition (no CI)."""
    metric_records = [
        {"seed": 42, "policy_type": "baseline", "success_rate": 0.7},
    ]

    result = aggregate_metrics(metric_records, group_by="policy_type", ci_samples=100, seed=42)

    baseline_success = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "success_rate"
    )
    assert baseline_success["mean"] == 0.7
    assert baseline_success["median"] == 0.7
    assert baseline_success["p95"] == 0.7
    assert baseline_success["std"] == 0.0
    assert baseline_success["sample_size"] == 1
    assert baseline_success["ci_low"] is None  # No CI for single value
    assert baseline_success["ci_high"] is None


def test_bootstrap_ci_basic():
    """Test bootstrap CI computation."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    ci_low, ci_high = bootstrap_ci(values, ci_samples=1000, ci_confidence=0.95, seed=42)

    assert ci_low == pytest.approx(1.8, abs=1e-6)
    assert ci_high == pytest.approx(4.2, abs=1e-6)


def test_bootstrap_ci_insufficient_data():
    """Test bootstrap CI with insufficient data."""
    values = [1.0]
    ci_low, ci_high = bootstrap_ci(values, ci_samples=1000, seed=42)

    assert ci_low is None
    assert ci_high is None


def test_aggregate_metrics_empty():
    """Test aggregation with empty records."""
    result = aggregate_metrics([], group_by="policy_type")
    assert result == []


def test_aggregate_metrics_missing_values():
    """Test aggregation with missing/None values."""
    metric_records = [
        {
            "seed": 42,
            "policy_type": "baseline",
            "collision_rate": None,
            "success_rate": 0.7,
        },
        {
            "seed": 123,
            "policy_type": "baseline",
            "collision_rate": 0.1,
            "success_rate": 0.75,
        },
    ]

    result = aggregate_metrics(metric_records, group_by="policy_type")

    # success_rate should have 2 samples
    baseline_success = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "success_rate"
    )
    assert baseline_success["sample_size"] == 2

    # collision_rate should have 1 sample (None dropped)
    baseline_collision = next(
        r for r in result if r["condition"] == "baseline" and r["metric_name"] == "collision_rate"
    )
    assert baseline_collision["sample_size"] == 1
    assert baseline_collision["mean"] == 0.1


def test_completeness_score():
    """Completeness scoring tracks missing and failed seeds."""
    completeness = compute_completeness_score(
        expected_seeds=[1, 2, 3], completed_seeds=[1, 2], failed_seeds=[3]
    )

    assert completeness["score"] == 66.7
    assert completeness["missing_seeds"] == []
    assert completeness["failed_seeds"] == ["3"]
    assert completeness["status"] == "PARTIAL"
    assert completeness["expected"] == 3
    assert completeness["completed"] == 2


def test_aggregate_metrics_missing_groupby_field():
    """Verify that aggregate_metrics returns [] if group_by field is missing from df."""
    metric_records = [{"seed": 42, "success_rate": 0.7}]
    result = aggregate_metrics(metric_records, group_by="policy_type")
    assert result == []


def test_aggregate_metrics_seed_reproducibility():
    """Verify that seed parameter ensures reproducible CIs, and different seeds differ."""
    metric_records = [
        {"seed": i, "policy_type": "baseline", "success_rate": float(i) / 50.0} for i in range(50)
    ]
    res1 = aggregate_metrics(metric_records, seed=42)
    res2 = aggregate_metrics(metric_records, seed=42)
    res3 = aggregate_metrics(metric_records, seed=123)

    ci1 = (res1[0]["ci_low"], res1[0]["ci_high"])
    ci2 = (res2[0]["ci_low"], res2[0]["ci_high"])
    ci3 = (res3[0]["ci_low"], res3[0]["ci_high"])

    assert ci1 == ci2
    assert ci1 != ci3
    assert res1[0]["ci_low"] == pytest.approx(0.40916, abs=1e-5)
    assert res1[0]["ci_high"] == pytest.approx(0.56323, abs=1e-5)


def test_aggregate_metrics_entirely_none_metric():
    """Verify that a metric column that is entirely None is skipped, without breaking the loop."""
    metric_records = [
        {
            "seed": 1,
            "policy_type": "baseline",
            "collision_rate": None,
            "success_rate": 0.7,
        },
        {
            "seed": 2,
            "policy_type": "baseline",
            "collision_rate": None,
            "success_rate": 0.8,
        },
        {
            "seed": 3,
            "policy_type": "pretrained",
            "collision_rate": 0.1,
            "success_rate": 0.9,
        },
    ]
    result = aggregate_metrics(metric_records)
    baseline_metrics = {r["metric_name"] for r in result if r["condition"] == "baseline"}
    assert baseline_metrics == {"success_rate"}


def test_bootstrap_ci_exactly_two_values():
    """Verify bootstrap_ci computes a valid CI when values has exactly 2 elements."""
    values = [1.0, 2.0]
    ci_low, ci_high = bootstrap_ci(values, seed=42)
    assert ci_low is not None
    assert ci_high is not None
    assert ci_low <= ci_high


def test_bootstrap_ci_uses_lower_percentile_boundary():
    """Verify the lower confidence bound uses the declared percentile boundary."""
    values = [0.01, 0.11, 0.29, 0.73, 0.97]
    ci_low, _ = bootstrap_ci(values, ci_samples=200, seed=42)
    assert ci_low == pytest.approx(0.0856, abs=1e-6)


def test_bootstrap_ci_defaults():
    """Verify bootstrap_ci works correctly with default arguments (no confidence or samples specified)."""
    values = [0.03, 0.17, 0.31, 0.58, 0.91]
    ci_low, ci_high = bootstrap_ci(values, seed=42)
    assert ci_low == pytest.approx(0.142, abs=1e-6)
    assert ci_high == pytest.approx(0.66805, abs=1e-6)


def test_completeness_score_empty_expected():
    """Verify completeness score handles empty expected_seeds by returning score 0.0."""
    completeness = compute_completeness_score(expected_seeds=[], completed_seeds=[])
    assert completeness["score"] == 0.0
    assert completeness["status"] == "PASS"


def test_completeness_score_sorting():
    """Verify completeness score correctly sorts missing and failed seeds."""
    completeness = compute_completeness_score(
        expected_seeds=[10, 2, "abc", 1, "1a", 11, 3],
        completed_seeds=[1],
        failed_seeds=[10, 2, "abc", "1a"],
    )
    assert completeness["missing_seeds"] == ["3", "11"]
    # failed seeds should sort numeric first: 2, 10, 1a, abc
    assert completeness["failed_seeds"] == ["2", "10", "1a", "abc"]


def test_completeness_score_fail():
    """Verify completeness score status is FAIL when there are no completed seeds."""
    completeness = compute_completeness_score(expected_seeds=[1, 2], completed_seeds=[])
    assert completeness["status"] == "FAIL"
    assert completeness["score"] == 0.0
    assert completeness["expected"] == 2
    assert completeness["completed"] == 0


# ---------------------------------------------------------------------------
# Mutation-coverage slice: export / manifest / extraction paths (issue #5508)
# ---------------------------------------------------------------------------


def test_export_metrics_json_roundtrip(tmp_path: Path):
    """Exported JSON carries schema_version and the full metrics payload."""
    aggregated = [
        {"metric_name": "success_rate", "condition": "baseline", "mean": 0.7, "median": 0.7},
    ]
    out = tmp_path / "metrics.json"
    export_metrics_json(aggregated, str(out))

    assert out.exists()
    payload = tmp_path.joinpath("metrics.json").read_text(encoding="utf-8")
    loaded = json.loads(payload)
    assert loaded["schema_version"] == "1.0.0"
    assert loaded["metrics"] == aggregated


def test_export_metrics_json_creates_parent_dirs(tmp_path: Path):
    """export_metrics_json creates missing parent directories."""
    out = tmp_path / "nested" / "dir" / "metrics.json"
    export_metrics_json([], str(out))
    assert out.exists()


def test_export_metrics_csv_roundtrip(tmp_path: Path):
    """Exported CSV is tab-delimited and round-trips through pandas."""
    import pandas as pd

    aggregated = [
        {"metric_name": "success_rate", "condition": "baseline", "mean": 0.7},
        {"metric_name": "collision_rate", "condition": "baseline", "mean": 0.1},
    ]
    out = tmp_path / "metrics.csv"
    export_metrics_csv(aggregated, str(out))

    assert out.exists()
    df = pd.read_csv(out, sep="\t")
    assert list(df.columns) == ["metric_name", "condition", "mean"]
    assert len(df) == 2
    assert df["mean"].iloc[0] == 0.7


def test_export_metrics_csv_creates_parent_dirs(tmp_path: Path):
    """export_metrics_csv creates missing parent directories."""
    out = tmp_path / "deep" / "metrics.csv"
    export_metrics_csv([], str(out))
    assert out.exists()


def test_load_manifest_payload_json(tmp_path: Path):
    """JSON manifests load as a single payload."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"seed": 1, "policy_type": "baseline"}), encoding="utf-8")
    payload = _load_manifest_payload(path)
    assert payload == {"seed": 1, "policy_type": "baseline"}


def test_load_manifest_payload_jsonl_takes_last_line(tmp_path: Path):
    """JSONL manifests use the last non-empty line."""
    path = tmp_path / "m.jsonl"
    path.write_text('\n{"seed": 1}\n\n{"seed": 2, "policy_type": "x"}\n', encoding="utf-8")
    payload = _load_manifest_payload(path)
    assert payload == {"seed": 2, "policy_type": "x"}


def test_load_manifest_payload_empty_jsonl_raises(tmp_path: Path):
    """Empty JSONL manifests raise ValueError."""
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Empty manifest"):
        _load_manifest_payload(path)


def test_extract_seed_metrics_basic(tmp_path: Path):
    """extract_seed_metrics reads per-seed metrics and summary aliases."""
    m1 = tmp_path / "a.json"
    m1.write_text(
        json.dumps(
            {
                "seed": 42,
                "policy_type": "baseline",
                "variant_id": 1,
                "metrics": {"success_rate": 0.8, "collision_rate": 0.1},
            }
        ),
        encoding="utf-8",
    )
    m2 = tmp_path / "b.json"
    m2.write_text(
        json.dumps(
            {
                "seed": 7,
                "policy_type": "pretrained",
                "summary": {"metrics": {"success_rate": 0.9, "timesteps_to_convergence": 300.0}},
            }
        ),
        encoding="utf-8",
    )

    records, failures = extract_seed_metrics([str(m1), str(m2)])
    assert failures == []
    assert len(records) == 2

    by_seed = {r["seed"]: r for r in records}
    assert by_seed[42]["success_rate"] == 0.8
    assert by_seed[42]["collision_rate"] == 0.1
    assert by_seed[42]["policy_type"] == "baseline"
    assert by_seed[42]["variant_id"] == 1
    assert by_seed[7]["timesteps_to_convergence"] == 300.0
    assert by_seed[7]["policy_type"] == "pretrained"


def test_extract_seed_metrics_missing_metrics_fails(tmp_path: Path):
    """Manifests without a metrics block are reported as failures."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"seed": 3, "policy_type": "baseline"}), encoding="utf-8")

    records, failures = extract_seed_metrics([str(bad)])
    assert records == []
    assert len(failures) == 1
    assert failures[0]["path"] == str(bad)
    assert failures[0]["seed"] == 3
    assert "metrics not found" in failures[0]["reason"]


def test_extract_seed_metrics_no_numeric_metrics_fails(tmp_path: Path):
    """Manifests with non-numeric metrics only are failures."""
    bad = tmp_path / "meta.json"
    bad.write_text(
        json.dumps({"seed": 5, "policy_type": "baseline", "metrics": {"policy_type": "baseline"}}),
        encoding="utf-8",
    )

    records, failures = extract_seed_metrics([str(bad)])
    assert records == []
    assert failures[0]["reason"] == "no numeric metrics found"


def test_extract_seed_metrics_unparseable_jsonl_fails(tmp_path: Path):
    """Malformed JSON lines are captured as parse failures."""
    bad = tmp_path / "broken.jsonl"
    bad.write_text("{not valid json}", encoding="utf-8")

    records, failures = extract_seed_metrics([str(bad)])
    assert records == []
    assert len(failures) == 1
    assert failures[0]["seed"] is None
    assert "Expecting" in failures[0]["reason"]


def test_extract_seed_metrics_missing_file_fails(tmp_path: Path):
    """A missing manifest path is captured as a failure, not a crash."""
    missing = tmp_path / "does_not_exist.json"
    records, failures = extract_seed_metrics([str(missing)])
    assert records == []
    assert len(failures) == 1
    assert failures[0]["reason"]
