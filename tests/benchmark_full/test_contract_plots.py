"""Contract test T012 for `generate_plots`.

Expectation (final):
  - In smoke mode produces a subset of PDF plots (e.g., distributions_*.pdf) in output dir.
  - Returns list of PlotArtifact objects with status 'generated' or 'skipped'.

Current state: NotImplementedError expected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic.plots import generate_plots


def test_generate_plots_smoke(temp_results_dir, synthetic_episode_record):
    out_dir = Path(temp_results_dir) / "plots"
    records = [
        synthetic_episode_record(
            episode_id="ep1",
            scenario_id="scenario_a",
            seed=1,
        )
    ]

    class _Metric:
        def __init__(self, name, mean):
            self.name = name
            self.mean = mean
            self.median = mean
            self.p95 = mean
            self.mean_ci = None
            self.median_ci = None

    class _Group:
        def __init__(self):
            self.archetype = "crossing"
            self.density = "low"
            self.count = 1
            self.metrics = {"collision_rate": _Metric("collision_rate", 0.0)}

    groups = [_Group()]

    class _Cfg:
        smoke = True

    with pytest.raises(NotImplementedError):  # until T035
        generate_plots(groups, records, str(out_dir), _Cfg())
