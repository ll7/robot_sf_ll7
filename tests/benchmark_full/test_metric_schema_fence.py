"""Full-classic summaries must not pool historical and current metric meanings."""

from types import SimpleNamespace

import pytest

from robot_sf.benchmark.full_classic.aggregation import aggregate_metrics
from robot_sf.benchmark.metric_definitions import (
    LEGACY_METRIC_SCHEMA_VERSION,
    METRIC_SCHEMA_VERSION,
)


def _record(jerk: float, *, version: str | None, archetype: str = "crossing") -> dict:
    """Build a saved episode with an explicit or historical metric definition."""
    record = {"archetype": archetype, "density": "low", "metrics": {"jerk_mean": jerk}}
    if version is not None:
        record["metric_schema_version"] = version
    return record


def _config() -> SimpleNamespace:
    """Disable resampling to isolate definition admission from uncertainty estimates."""
    return SimpleNamespace(bootstrap_samples=0, master_seed=100000)


@pytest.mark.parametrize("second_archetype", ["crossing", "overtaking"])
def test_full_classic_rejects_mixed_metric_definitions(second_archetype: str) -> None:
    """Neither one group nor separate groups may share a mixed-definition report."""
    records = [
        _record(0.042, version=None),
        _record(0.417, version=METRIC_SCHEMA_VERSION, archetype=second_archetype),
    ]

    with pytest.raises(ValueError, match="incompatible metric definitions"):
        aggregate_metrics(iter(records), _config())


@pytest.mark.parametrize("version", [None, LEGACY_METRIC_SCHEMA_VERSION, METRIC_SCHEMA_VERSION])
def test_full_classic_accepts_uniform_definition_generators(version: str | None) -> None:
    """A version fence must preserve legacy/current input and one-pass iterables."""
    records = (_record(jerk, version=version) for jerk in (0.125, 0.375))

    groups = aggregate_metrics(records, _config())

    assert len(groups) == 1
    assert groups[0].count == 2
    assert groups[0].metrics["jerk_mean"].mean == pytest.approx(0.25)
