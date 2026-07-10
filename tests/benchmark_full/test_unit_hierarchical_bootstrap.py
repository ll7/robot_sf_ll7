"""Unit tests for hierarchical (scenario-then-episode) bootstrap (issue #5139).

Covers:
  * ``hierarchical_bootstrap_ci`` two-stage cluster bootstrap: determinism,
    degenerate inputs, and the anti-conservatism property (hierarchical CIs are
    at least as wide as flat CIs when episodes are positively intra-cluster
    correlated).
  * ``cluster_robust_interval`` for binary endpoints: boundary behaviour,
    single-cluster fallback to the flat Wilson interval, and degenerate cases.
  * ``aggregate_metrics`` end-to-end in ``bootstrap_mode="hierarchical"``:
    structure preserved, rate metrics use the cluster-robust interval, and the
    ``bootstrap_cluster`` option selects scenario vs seed cells.
  * Backward compatibility: the default mode is ``flat`` and reproduces the
    original i.i.d. bootstrap.

These are focused CPU tests; no campaign, training, or benchmark claim is made.
The width comparisons characterize the implemented resampling procedures on
synthetic structured fixtures, not on campaign evidence.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy

import pytest

from robot_sf.benchmark.full_classic.aggregation import (
    _bootstrap_ci,  # type: ignore[import-private-names]
    _stable_group_seed,  # type: ignore[import-private-names]
    _wilson_interval,  # type: ignore[import-private-names]
    aggregate_metrics,
    cluster_robust_interval,
    hierarchical_bootstrap_ci,
)


class _Cfg:
    """Minimal config stub matching the BenchmarkConfig surface."""

    def __init__(self, **overrides):
        self.bootstrap_samples = 300
        self.bootstrap_confidence = 0.95
        self.master_seed = 123
        self.smoke = True
        self.bootstrap_mode = "flat"
        self.bootstrap_cluster = "scenario"
        for k, v in overrides.items():
            setattr(self, k, v)


def _clustered(rate_per_cell: list[float], episodes_per_cell: int = 10):
    """Build per-cluster Bernoulli outcome lists from per-cell success rates."""
    return [
        [1.0 if (i % 100) / 100 < rate else 0.0 for i in range(episodes_per_cell)]
        for rate in rate_per_cell
    ]


# --------------------------------------------------------------------------- #
# hierarchical_bootstrap_ci
# --------------------------------------------------------------------------- #


def test_hierarchical_ci_empty_returns_nan():
    """Empty cluster input returns NaN CIs."""
    mean_ci, median_ci = hierarchical_bootstrap_ci([], 50, 0.95, random.Random(1))
    assert all(math.isnan(v) for v in mean_ci)
    assert all(math.isnan(v) for v in median_ci)


def test_hierarchical_ci_single_value_collapses():
    """A single overall episode collapses the CI to that value."""
    mean_ci, median_ci = hierarchical_bootstrap_ci([[5.0]], 50, 0.95, random.Random(1))
    assert mean_ci == (5.0, 5.0)
    assert median_ci == (5.0, 5.0)


def test_hierarchical_ci_deterministic_same_seed():
    """Same seed reproduces identical hierarchical CIs."""
    cells = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    m1, _ = hierarchical_bootstrap_ci(deepcopy(cells), 200, 0.95, random.Random(42))
    m2, _ = hierarchical_bootstrap_ci(deepcopy(cells), 200, 0.95, random.Random(42))
    assert m1 == m2


def test_hierarchical_ci_different_seed_differs():
    """Different seeds produce different hierarchical CIs."""
    cells = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    m1, _ = hierarchical_bootstrap_ci(deepcopy(cells), 300, 0.95, random.Random(42))
    m2, _ = hierarchical_bootstrap_ci(deepcopy(cells), 300, 0.95, random.Random(7))
    assert m1 != m2


def test_hierarchical_ci_wider_than_flat_under_intracluster_correlation():
    """When episodes within a cell are identical (perfect intra-cluster
    correlation), the flat i.i.d. bootstrap treats them as independent and
    understates uncertainty. The hierarchical two-stage bootstrap must report a
    wider interval than the flat bootstrap."""
    # 6 cells, each cell has 10 identical values (perfect within-cell correlation).
    cell_means = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    cells = [[m] * 10 for m in cell_means]
    flat_values = [v for cell in cells for v in cell]
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    flat_mean_ci, _ = _bootstrap_ci(flat_values, 500, 0.95, rng_a)
    hier_mean_ci, _ = hierarchical_bootstrap_ci(cells, 500, 0.95, rng_b)
    flat_width = flat_mean_ci[1] - flat_mean_ci[0]
    hier_width = hier_mean_ci[1] - hier_mean_ci[0]
    assert hier_width > flat_width, (
        f"Hierarchical CI ({hier_width}) should be wider than flat ({flat_width}) "
        "under perfect intra-cluster correlation."
    )


# --------------------------------------------------------------------------- #
# cluster_robust_interval
# --------------------------------------------------------------------------- #


def test_cluster_robust_interval_empty_returns_nan():
    """Empty input returns NaN bounds."""
    low, high = cluster_robust_interval([], 0.95)
    assert math.isnan(low) and math.isnan(high)


def test_cluster_robust_interval_single_episode_collapses():
    """A single episode collapses the cluster-robust interval to its value."""
    low, high = cluster_robust_interval([[0.5]], 0.95)
    assert low == 0.5 and high == 0.5


def test_cluster_robust_interval_single_cluster_falls_back_to_wilson():
    """With one cluster the interval falls back to the flat Wilson interval."""
    cells = [[0.0, 1.0, 0.0, 1.0]]  # one cluster, 4 trials, p=0.5
    low, high = cluster_robust_interval(cells, 0.95)
    wilson_low, wilson_high = _wilson_interval(0.5, 4, 0.95)
    assert (low, high) == (wilson_low, wilson_high)


def test_cluster_robust_interval_clipped_to_unit_interval():
    """Extreme per-cell dispersion still clips the interval to [0, 1]."""
    # Extreme per-cell dispersion around p=0.5 should still be clipped to [0,1].
    cells = [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    low, high = cluster_robust_interval(cells, 0.95)
    assert 0.0 <= low <= high <= 1.0


def test_cluster_robust_interval_wider_than_flat_wilson_under_clustering():
    """With strong between-cell heterogeneity but identical pooled p, the
    cluster-robust interval must be wider than the flat Wilson interval that
    assumes i.i.d. Bernoulli trials."""
    # 6 cells of 10 episodes; pooled p = 0.5 but cells are all-0 or all-1.
    cells = [[1.0] * 10, [1.0] * 10, [1.0] * 10, [0.0] * 10, [0.0] * 10, [0.0] * 10]
    pooled = [v for c in cells for v in c]
    p_hat = sum(pooled) / len(pooled)
    wilson_low, wilson_high = _wilson_interval(p_hat, len(pooled), 0.95)
    cr_low, cr_high = cluster_robust_interval(cells, 0.95)
    assert (cr_high - cr_low) > (wilson_high - wilson_low)


# --------------------------------------------------------------------------- #
# aggregate_metrics end-to-end (hierarchical mode)
# --------------------------------------------------------------------------- #


def _nested_records(n_cells: int = 6, episodes_per_cell: int = 8):
    """Build records with scenario/seed nesting and intra-cluster correlation.

    Each scenario cell shares a cell-level success probability so episodes
    within a cell are correlated, which is the regime where flat resampling is
    anti-conservative.
    """
    rng = random.Random(2024)
    cell_rates = [0.05, 0.15, 0.4, 0.6, 0.85, 0.95]
    records = []
    ep = 0
    for cell_idx in range(n_cells):
        rate = cell_rates[cell_idx % len(cell_rates)]
        scenario_id = f"scenario_{cell_idx}"
        for seed in range(episodes_per_cell):
            collision = 1.0 if rng.random() < rate else 0.0
            success = 1.0 - collision
            records.append(
                {
                    "episode_id": f"ep{ep}",
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "archetype": "crossing",
                    "density": "low",
                    "metrics": {
                        "collision_rate": collision,
                        "success_rate": success,
                        "time_to_goal": 10.0 + cell_idx + rng.random(),
                    },
                }
            )
            ep += 1
    return records


def test_aggregate_hierarchical_preserves_structure():
    """Hierarchical mode preserves the (archetype, density) group structure."""
    recs = _nested_records()
    groups = aggregate_metrics(recs, _Cfg(bootstrap_mode="hierarchical"))
    assert len(groups) == 1
    g = groups[0]
    assert g.archetype == "crossing" and g.density == "low"
    assert g.count == len(recs)
    for key in ("collision_rate", "success_rate", "time_to_goal"):
        assert key in g.metrics
        m = g.metrics[key]
        assert m.mean_ci is not None
        low, high = m.mean_ci
        assert low <= high


def test_aggregate_hierarchical_rate_metrics_in_unit_interval():
    """Rate-metric CIs in hierarchical mode stay within [0, 1]."""
    recs = _nested_records()
    groups = aggregate_metrics(recs, _Cfg(bootstrap_mode="hierarchical"))
    g = groups[0]
    for key in ("collision_rate", "success_rate"):
        low, high = g.metrics[key].mean_ci
        assert 0.0 <= low <= high <= 1.0


def test_aggregate_hierarchical_wider_than_flat_for_rate():
    """The cluster-robust interval for collision_rate must be wider than the
    flat Wilson interval on the same clustered records, documenting the
    anti-conservatism magnitude the issue asks to surface."""
    recs = _nested_records()
    flat = aggregate_metrics(deepcopy(recs), _Cfg(bootstrap_mode="flat"))[0]
    hier = aggregate_metrics(deepcopy(recs), _Cfg(bootstrap_mode="hierarchical"))[0]
    flat_w = _ci_width(flat.metrics["collision_rate"].mean_ci)
    hier_w = _ci_width(hier.metrics["collision_rate"].mean_ci)
    assert hier_w > flat_w, (
        f"Hierarchical collision_rate CI width ({hier_w}) should exceed flat "
        f"Wilson width ({flat_w}) on clustered records."
    )


def test_aggregate_hierarchical_seed_cluster_uses_seed_field():
    """bootstrap_cluster='seed' must run without error and produce a valid
    interval, exercising the seed-level cluster bootstrap path."""
    recs = _nested_records()
    groups = aggregate_metrics(recs, _Cfg(bootstrap_mode="hierarchical", bootstrap_cluster="seed"))
    g = groups[0]
    m = g.metrics["time_to_goal"]
    assert m.mean_ci is not None
    low, high = m.mean_ci
    assert low <= high


def test_aggregate_default_mode_is_flat():
    """Omitting bootstrap_mode must reproduce the original flat behaviour."""
    recs = _nested_records()
    cfg = _Cfg()  # bootstrap_mode defaults to "flat"
    assert cfg.bootstrap_mode == "flat"
    groups = aggregate_metrics(recs, cfg)
    flat_default = groups[0].metrics["time_to_goal"].mean_ci
    flat_explicit = (
        aggregate_metrics(recs, _Cfg(bootstrap_mode="flat"))[0].metrics["time_to_goal"].mean_ci
    )
    assert flat_default == flat_explicit


def test_aggregate_unknown_mode_falls_back_to_flat():
    """An unrecognised bootstrap_mode logs and falls back to flat resampling."""
    recs = _nested_records()
    groups = aggregate_metrics(recs, _Cfg(bootstrap_mode="nonsense"))
    assert len(groups) == 1
    m = groups[0].metrics["time_to_goal"]
    assert m.mean_ci is not None


def test_aggregate_hierarchical_deterministic_same_seed():
    """Hierarchical aggregation is reproducible for a fixed master seed."""
    recs = _nested_records()
    a = aggregate_metrics(deepcopy(recs), _Cfg(bootstrap_mode="hierarchical", master_seed=7))
    b = aggregate_metrics(deepcopy(recs), _Cfg(bootstrap_mode="hierarchical", master_seed=7))
    assert a[0].metrics["time_to_goal"].mean_ci == b[0].metrics["time_to_goal"].mean_ci
    assert a[0].metrics["time_to_goal"].median_ci == b[0].metrics["time_to_goal"].median_ci


def test_group_seed_is_stable_and_order_sensitive() -> None:
    """Bootstrap seeds must not depend on Python's randomized hash salt."""
    seed = _stable_group_seed(123, "crossing", "low", "time_to_goal")
    assert seed == _stable_group_seed(123, "crossing", "low", "time_to_goal")
    assert seed != _stable_group_seed(123, "crossing", "high", "time_to_goal")


def test_aggregate_hierarchical_missing_cluster_field_degrades_gracefully():
    """Records without the cluster field must not crash; they are bucketed into
    a synthetic cell and resampled."""
    recs = [
        {
            "episode_id": f"ep{i}",
            "archetype": "crossing",
            "density": "low",
            "metrics": {"time_to_goal": 10.0 + i},
        }
        for i in range(12)
    ]
    groups = aggregate_metrics(recs, _Cfg(bootstrap_mode="hierarchical"))
    m = groups[0].metrics["time_to_goal"]
    assert m.mean_ci is not None
    low, high = m.mean_ci
    assert low <= high


def _ci_width(ci):
    """Width of a (low, high) CI tuple, treating NaN as zero width."""
    if ci is None or any(math.isnan(v) for v in ci):
        return 0.0
    return ci[1] - ci[0]


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_cluster_robust_interval_monotone_in_confidence(conf):
    """Higher confidence must not shrink the cluster-robust interval."""
    cells = _clustered([0.1, 0.3, 0.5, 0.7, 0.9])
    widths = []
    for c in sorted({0.90, 0.95, 0.99, conf}):
        low, high = cluster_robust_interval(cells, c)
        widths.append((c, high - low))
    # Verify monotonicity across the full sorted set when we tested all three.
    if {0.90, 0.95, 0.99} <= {c for c, _ in widths}:
        w = dict(widths)
        assert w[0.90] <= w[0.95] <= w[0.99]
