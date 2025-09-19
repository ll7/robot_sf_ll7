"""Contract test T009 for `aggregate_metrics`.

Expectations:
  - Returns list of AggregateMetricsGroup objects (at least one) grouped by archetype & density.
  - Each group contains required metric keys: collision_rate, success_rate, time_to_goal, path_efficiency, snqi.
  - With fixed cfg.master_seed/bootstrap seed (implied later) results deterministic (cannot test numeric values yet, but structure).

Current state: aggregate_metrics not implemented -> NotImplementedError expected.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.aggregation import aggregate_metrics


def test_aggregate_metrics_structure(synthetic_episode_record):
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
        bootstrap_samples = 100
        bootstrap_confidence = 0.95
        smoke = True
        snqi_weights_path = None

    cfg = _Cfg()

    with pytest.raises(NotImplementedError):  # until T030
        aggregate_metrics(records, cfg)
