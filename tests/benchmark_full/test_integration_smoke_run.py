"""Contract integration test T016 for `run_full_benchmark` smoke path.

Expectation (final):
  - Creates directory tree (episodes, aggregates, reports, plots) and manifest.json.
  - Videos skipped in smoke mode.

Current state: NotImplementedError expected.
"""

from __future__ import annotations

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_smoke_run_creates_structure(config_factory):
    cfg = config_factory(smoke=True, workers=1)
    with pytest.raises(NotImplementedError):  # until T029
        run_full_benchmark(cfg)
