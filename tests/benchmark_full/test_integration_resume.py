"""Integration test T017: resume behavior.

Expectation (eventual):
  - First run creates some episodes.
  - Second run with same config should skip previously completed episodes (executed_jobs smaller on second invocation).

Current state: run_full_benchmark implemented with resume + aggregation.
"""

from __future__ import annotations

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


def test_resume_skips_existing(config_factory):
    cfg = config_factory(smoke=True, workers=1)
    manifest_first = run_full_benchmark(cfg)
    episodes_file = manifest_first.output_root / "episodes" / "episodes.jsonl"
    with episodes_file.open("r", encoding="utf-8") as f:
        first_lines = [ln for ln in f.read().splitlines() if ln.strip()]
    # Second run should not add more lines
    run_full_benchmark(cfg)
    with episodes_file.open("r", encoding="utf-8") as f:
        second_lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(second_lines) == len(first_lines) >= 1
