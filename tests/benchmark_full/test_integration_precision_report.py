"""Integration test T020: precision report structure.

Expectation (future): statistical_sufficiency.json contains final_pass key and
evaluations array with scenario entries.

Current: NotImplementedError expected.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_precision_report_structure(config_factory):
    cfg = config_factory(smoke=True)
    with pytest.raises(NotImplementedError):  # until T033/T034
        run_full_benchmark(cfg)
