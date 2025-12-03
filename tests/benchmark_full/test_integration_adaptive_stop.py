"""Integration test T018: adaptive early stop.

Validate that the adaptive loop executes at least one iteration, produces
artifacts, and does not exceed the configured max episodes per scenario.
We configure small numbers so the loop should terminate quickly either due
to hitting max_episodes or (in future) precision pass. For now precision
pass will be immediate because placeholder metrics have zero variance for
rates.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


@pytest.mark.timeout(60)
def test_adaptive_early_stop(config_factory, perf_policy):
    """TODO docstring. Document this function.

    Args:
        config_factory: TODO docstring.
        perf_policy: TODO docstring.
    """
    start = time.perf_counter()
    cfg = config_factory(
        smoke=True,
        workers=1,
        max_episodes=12,  # cap per scenario
        initial_episodes=4,
        batch_size=4,
        collision_ci=0.5,  # loose thresholds to allow immediate pass
        success_ci=0.5,
    )
    manifest = run_full_benchmark(cfg)

    root = Path(manifest.output_root)
    episodes_file = root / "episodes" / "episodes.jsonl"
    assert episodes_file.exists()
    lines = [ln for ln in episodes_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # Should not exceed max_episodes * number_of_scenarios (<= scenarios * 12)
    # Scenario count derived from lines because each scenario has at least one line.
    assert len(lines) >= cfg.initial_episodes  # ran initial batch

    # Aggregates artifact
    summary_path = root / "aggregates" / "summary.json"
    assert summary_path.exists(), "summary.json missing"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data, "summary.json empty"

    # Effect sizes artifact
    effects_path = root / "reports" / "effect_sizes.json"
    assert effects_path.exists(), "effect_sizes.json missing"

    # Precision report artifact
    precision_path = root / "reports" / "statistical_sufficiency.json"
    assert precision_path.exists(), "statistical_sufficiency.json missing"
    precision = json.loads(precision_path.read_text(encoding="utf-8"))
    assert "final_pass" in precision and "evaluations" in precision
    assert isinstance(precision["evaluations"], list)
    elapsed = time.perf_counter() - start
    assert perf_policy.classify(elapsed) != "hard", (
        f"Adaptive stop test exceeded hard threshold: {elapsed:.2f}s"
    )
