"""Contract test T009 for `aggregate_metrics`.

Expectations:
  - Returns list of AggregateMetricsGroup objects (at least one) grouped by archetype & density.
  - Each group contains metric keys present in input episode records.
  - mean_ci for rate metrics (collision_rate, success_rate) uses Wilson interval within [0,1].
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.aggregation import aggregate_metrics


def test_aggregate_metrics_structure(synthetic_episode_record):
    """TODO docstring. Document this function.

    Args:
        synthetic_episode_record: TODO docstring.
    """
    records = [
        synthetic_episode_record(
            episode_id="ep1",
            scenario_id="scenario_a",
            seed=1,
            archetype="crossing",
            density="low",
        ),
        synthetic_episode_record(
            episode_id="ep2",
            scenario_id="scenario_a",
            seed=2,
            archetype="crossing",
            density="low",
        ),
    ]

    class _Cfg:  # minimal config stub
        """TODO docstring. Document this class."""

        bootstrap_samples = 50
        bootstrap_confidence = 0.95
        master_seed = 123
        smoke = True

    groups = aggregate_metrics(records, _Cfg())
    assert groups and len(groups) == 1
    g = groups[0]
    assert g.archetype == "crossing" and g.density == "low"
    # Required placeholder metrics present
    for key in [
        "collision_rate",
        "success_rate",
        "time_to_goal",
        "path_efficiency",
        "avg_speed",
        "snqi",
    ]:
        assert key in g.metrics
        m = g.metrics[key]
        assert m.mean is not None
        if key in {"collision_rate", "success_rate"}:
            assert m.mean_ci is not None
            low, high = m.mean_ci
            assert 0.0 <= low <= high <= 1.0
