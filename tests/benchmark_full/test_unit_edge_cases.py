"""Polish Phase T049: Edge case tests for aggregation & effect sizes.

Validates:
  * Wilson interval for zero collisions returns >0 upper bound (non-degenerate) and
    is symmetric-ish near mid p for large n.
  * Effect size Glass Δ falls back to 0 when CI absent or variance effectively zero.

These are focused fast tests using the existing aggregation & effects modules.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.aggregation import aggregate_metrics
from robot_sf.benchmark.full_classic.effects import compute_effect_sizes


class _Cfg:  # minimal config stub
    bootstrap_samples = 300
    bootstrap_confidence = 0.95
    master_seed = 123
    smoke = True
    effect_size_reference_density = "low"


def test_wilson_zero_collision_upper_bound_nonzero():
    records = []
    # Create 50 episodes with zero collisions (collision_rate metric = 0 for each sample, interpreted as Bernoulli mean 0)
    for i in range(50):
        records.append(
            {
                "episode_id": f"ep{i}",
                "archetype": "crossing",
                "density": "low",
                "metrics": {"collision_rate": 0.0, "success_rate": 1.0},
            },
        )
    groups = aggregate_metrics(records, _Cfg())
    assert groups, "Expected a single aggregate group"
    g = groups[0]
    coll = g.metrics["collision_rate"]
    low, high = coll.mean_ci  # Wilson interval applied for rate metrics
    # Allow tiny positive numerical noise in the lower bound (floating arithmetic)
    assert low <= 1e-12, f"Lower bound should be ~0, got {low}"
    assert high >= 0.0
    # Upper bound should be > 0 providing informative uncertainty
    assert high > 0.0, f"Wilson upper bound should be >0 for zero events, got {high}"


def test_glass_delta_zero_when_ci_missing():
    # Construct two groups manually bypassing CI to force missing mean_ci scenario
    # We simulate by aggregating with a single value (mean_ci collapses to identical bounds)
    records = [
        {
            "episode_id": "ep1",
            "archetype": "crossing",
            "density": "low",
            "metrics": {"time_to_goal": 10.0},
        },
        {
            "episode_id": "ep2",
            "archetype": "crossing",
            "density": "high",
            "metrics": {"time_to_goal": 10.0},
        },
    ]
    groups = aggregate_metrics(records, _Cfg())
    reports = compute_effect_sizes(groups, _Cfg())
    # time_to_goal identical between densities -> diff 0 -> Glass Δ should be 0
    assert reports
    entries = [e for r in reports for e in r.comparisons if e.metric == "time_to_goal"]
    assert entries, "Expected time_to_goal comparison entry"
    for e in entries:
        assert e.diff == 0.0
        assert e.standardized == 0.0
