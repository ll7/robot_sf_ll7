"""Contract tests for diagnostic social-compliance metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/benchmarks/social_compliance_metric_contract_v1.yaml"
EXPECTED_FAMILIES = {
    "pedestrian_deviation",
    "flow_disruption",
    "comfort_exposure",
    "legibility_progress",
    "distributional_inconvenience",
}
REQUIRED_METRIC_FIELDS = {
    "id",
    "family",
    "direction",
    "units",
    "denominator",
    "claim_class",
    "required_signals",
    "definition",
    "aggregation",
    "missing_data_behavior",
    "fixture_expectation",
}


def _load_contract() -> dict[str, Any]:
    """Load the checked-in social-compliance metric contract."""
    payload = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_social_compliance_metric_contract_covers_issue_3061_scope() -> None:
    """The contract should cover every metric family named in issue #3061."""
    contract = _load_contract()
    metrics = contract["metrics"]
    families = {metric["family"] for metric in metrics}

    assert contract["schema_version"] == "social-compliance-metric-contract.v1"
    assert contract["issue"] == 3061
    assert contract["status"] == "diagnostic_contract"
    assert set(contract["required_metric_families"]) == EXPECTED_FAMILIES
    assert EXPECTED_FAMILIES <= families
    assert len(metrics) >= len(EXPECTED_FAMILIES)


def test_social_compliance_metric_contract_is_diagnostic_not_claim_grade() -> None:
    """Metric definitions must not overclaim simulation proxies as social-validity evidence."""
    contract = _load_contract()

    assert "simulation-only diagnostic proxies" in contract["claim_boundary"]
    assert "do not establish real-world" in contract["claim_boundary"]
    assert "SNQI weights and headline score definitions are unchanged" in contract[
        "report_surfaces"
    ]["snqi_relationship"]
    assert set(contract["missing_data_policy"]["status_values"]) == {
        "available",
        "unavailable",
        "not_applicable",
    }

    for metric in contract["metrics"]:
        assert REQUIRED_METRIC_FIELDS <= set(metric)
        assert metric["claim_class"] == "diagnostic_proxy"
        assert metric["direction"] in {"lower_is_better", "higher_is_better"}
        assert metric["required_signals"]
        assert "support" in contract["missing_data_policy"]["rule"]


def test_social_compliance_metric_contract_fixture_expectations_are_concrete() -> None:
    """Fixture examples should encode units, denominators, and missing-data-free expectations."""
    metrics = {metric["id"]: metric for metric in _load_contract()["metrics"]}

    deviation = metrics["pedestrian_deviation_mean_m"]["fixture_expectation"]
    assert math.isclose(
        sum(deviation["sample_distances_m"]) / len(deviation["sample_distances_m"]),
        deviation["expected_mean_m"],
    )

    delay = metrics["flow_disruption_delay_s"]["fixture_expectation"]
    delays = [
        max(0.0, actual - reference)
        for actual, reference in zip(
            delay["arrival_time_s"], delay["reference_arrival_time_s"], strict=True
        )
    ]
    assert math.isclose(sum(delays) / len(delays), delay["expected_mean_delay_s"])

    exposure = metrics["comfort_exposure_person_s"]["fixture_expectation"]
    assert math.isclose(
        sum(bool(step) for step in exposure["exposed_steps"]) * exposure["timestep_seconds"],
        exposure["expected_person_s"],
    )

    progress = metrics["legibility_progress_deficit_m"]["fixture_expectation"]
    deficits = [
        max(0.0, actual - reference)
        for actual, reference in zip(
            progress["goal_distance_m"], progress["reference_goal_distance_m"], strict=True
        )
    ]
    assert math.isclose(sum(deficits) / len(deficits), progress["expected_mean_deficit_m"])

    distribution = metrics["distributional_inconvenience_p90_p50_gap"]["fixture_expectation"]
    samples = np.asarray(distribution["per_pedestrian_delay_s"], dtype=float)
    gap = float(np.quantile(samples, 0.9) - np.quantile(samples, 0.5))
    assert math.isclose(gap, distribution["expected_gap_s"])
