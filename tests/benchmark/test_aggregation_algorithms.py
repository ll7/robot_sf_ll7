"""Contract tests for algorithm-aware aggregation behaviour."""

from __future__ import annotations

import pytest
from loguru import logger

from robot_sf.benchmark.aggregate import compute_aggregates_with_ci
from robot_sf.benchmark.errors import AggregationInputError, AggregationMetadataError


def _make_record(
    algo: str | None,
    *,
    include_nested: bool = True,
    scenario_id: str = "scenario-1",
    success_rate: float = 1.0,
) -> dict[str, object]:
    """TODO docstring. Document this function.

    Args:
        algo: TODO docstring.
        include_nested: TODO docstring.
        scenario_id: TODO docstring.
        success_rate: TODO docstring.

    Returns:
        TODO docstring.
    """
    record: dict[str, object] = {
        "episode_id": f"{scenario_id}-{algo or 'none'}",
        "scenario_id": scenario_id,
        "metrics": {"success_rate": success_rate},
    }
    if include_nested:
        record["scenario_params"] = {"algo": algo}
    else:
        record["scenario_params"] = {}
    if algo is not None:
        record["algo"] = algo
    return record


def test_grouping_prefers_nested_algo():
    """Nested algorithm metadata should be preferred, with top-level used as fallback."""

    records = [
        _make_record("sf", include_nested=True, scenario_id="scenario-sf", success_rate=0.9),
        _make_record("ppo", include_nested=False, scenario_id="scenario-ppo", success_rate=0.4),
    ]

    result = compute_aggregates_with_ci(records, return_ci=False)

    groups = {key for key in result if key != "_meta"}
    assert {"sf", "ppo"} <= groups
    assert result["sf"]["success_rate"]["mean"] == pytest.approx(0.9)
    assert result["ppo"]["success_rate"]["mean"] == pytest.approx(0.4)


def test_warns_and_flags_missing_algorithms():
    """Warnings should surface when expected algorithms are absent."""

    records = [_make_record("sf", include_nested=True, scenario_id="scenario-sf")]
    captured: list = []

    def capture_message(message):
        """Collect Loguru warning records for assertion."""
        captured.append(message)

    handle = logger.add(capture_message, level="WARNING")
    try:
        result = compute_aggregates_with_ci(
            records,
            expected_algorithms={"sf", "ppo"},
            return_ci=False,
        )
    finally:
        logger.remove(handle)

    assert result["sf"]["success_rate"]["mean"] == pytest.approx(1.0)
    assert result["_meta"]["missing_algorithms"] == ["ppo"]
    assert any(
        msg.record["extra"].get("event") == "aggregation_missing_algorithms" for msg in captured
    )
    assert "effective_group_key" in result["_meta"]


def test_missing_algo_fields_raise():
    """Records missing both nested and top-level algorithm identifiers should fail fast."""

    records = [_make_record(None, include_nested=False, scenario_id="scenario-missing")]

    with pytest.raises(AggregationMetadataError):
        compute_aggregates_with_ci(records, return_ci=False)


@pytest.mark.parametrize(
    "confidence",
    [0.0, 1.0, -0.5, 1.5, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_bootstrap_confidence_fails_closed_when_ci_enabled(confidence: float) -> None:
    """Bootstrap CI controls must be finite probabilities in the open unit interval."""
    records = [_make_record("sf", scenario_id="scenario-sf")]

    with pytest.raises(AggregationInputError, match="bootstrap_confidence"):
        compute_aggregates_with_ci(
            records,
            bootstrap_samples=16,
            bootstrap_confidence=confidence,
        )


@pytest.mark.parametrize(
    ("return_ci", "bootstrap_samples", "confidence"),
    [(True, 0, 0.0), (False, 16, 1.5)],
)
def test_invalid_bootstrap_confidence_is_ignored_when_ci_disabled(
    return_ci: bool,
    bootstrap_samples: int,
    confidence: float,
) -> None:
    """Explicit no-CI modes preserve aggregation without validating unused controls."""
    records = [_make_record("sf", scenario_id="scenario-sf")]

    result = compute_aggregates_with_ci(
        records,
        return_ci=return_ci,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=confidence,
    )

    assert result["sf"]["success_rate"]["mean"] == pytest.approx(1.0)
    assert "mean_ci" not in result["sf"]["success_rate"]
