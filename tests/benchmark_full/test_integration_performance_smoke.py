"""Performance smoke test T044.

Objectives:
  - Execute a minimal benchmark run in non-smoke mode with tiny episode counts.
  - Assert overall runtime stays within soft threshold (e.g., < 3 seconds) for synthetic placeholder implementation.
  - Validate manifest contains scaling efficiency metrics introduced in T041.

Rationale:
  The current implementation uses synthetic episode execution (no heavy simulation), so a tight
  runtime threshold helps catch accidental performance regressions (e.g., excessive loops or IO).
"""

from __future__ import annotations

import time
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark

SOFT_RUNTIME_SEC = 8.0  # generous for CI; synthetic episodes should be << 1s


def test_performance_smoke(config_factory):
    # Configure very small run: 1 initial, max 2 episodes per scenario, batch size 1
    cfg = config_factory(
        smoke=False,
        initial_episodes=1,
        max_episodes=2,
        batch_size=1,
        workers=1,
    )
    start = time.time()
    run_full_benchmark(cfg)
    elapsed = time.time() - start
    assert elapsed < SOFT_RUNTIME_SEC, f"Performance smoke exceeded soft threshold: {elapsed:.3f}s"

    # Manifest scaling efficiency metrics
    manifest_path = Path(cfg.output_root) / "manifest.json"
    assert manifest_path.exists(), "Manifest file missing"
    data = manifest_path.read_text(encoding="utf-8")
    # Basic key presence checks without full JSON parse dependency here
    for key in [
        "runtime_sec",
        "episodes_per_second",
        "scaling_efficiency",
    ]:
        assert key in data, f"Missing '{key}' in manifest JSON content"
