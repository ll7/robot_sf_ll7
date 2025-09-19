"""Integration test T019: effect size report presence.

Expectation (future): After full run, reports/effect_sizes.json exists with
entries per archetype; each comparison has standardized field.

Current: NotImplementedError expected.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_effect_sizes_presence(config_factory):
    cfg = config_factory(smoke=True)
    with pytest.raises(NotImplementedError):  # until T032 integrated
        run_full_benchmark(cfg)
