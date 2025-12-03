"""Contract integration test T016 for `run_full_benchmark` smoke path.

Expectation (current skeleton T029):
  - Creates base directory tree: episodes/, aggregates/, reports/, plots/ under output_root.
  - Writes manifest.json with required keys.
  - Generates an episodes.jsonl file with at least initial episode batch (>=1 line).
  - In smoke mode videos are skipped (no videos/ directory required yet).

Later tasks (T035+) will add plots content; for now only directory creation asserted.
"""

from __future__ import annotations

import time

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


@pytest.mark.timeout(60)
def test_smoke_run_creates_structure(config_factory, perf_policy):
    """TODO docstring. Document this function.

    Args:
        config_factory: TODO docstring.
        perf_policy: TODO docstring.
    """
    start = time.perf_counter()
    cfg = config_factory(smoke=True, workers=1)
    manifest = run_full_benchmark(cfg)
    # Root path
    root = manifest.output_root
    # Required subdirectories
    expected_dirs = [
        root / "episodes",
        root / "aggregates",
        root / "reports",
        root / "plots",
    ]
    for d in expected_dirs:
        assert d.exists() and d.is_dir(), f"Missing expected directory: {d}"
    # Episodes file exists with >=1 line
    episodes_file = root / "episodes" / "episodes.jsonl"
    assert episodes_file.exists()
    with episodes_file.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) >= 1
    # Manifest file present
    manifest_path = root / "manifest.json"
    assert manifest_path.exists()
    # Basic attribute checks on returned manifest object
    for attr in ["git_hash", "scenario_matrix_hash", "config"]:
        assert hasattr(manifest, attr)
    elapsed = time.perf_counter() - start
    # Use perf_policy to ensure we don't breach hard threshold
    assert perf_policy.classify(elapsed) != "hard", (
        f"Smoke run unexpectedly exceeded hard threshold: {elapsed:.2f}s"
    )
