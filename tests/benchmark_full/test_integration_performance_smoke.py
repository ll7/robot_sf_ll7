"""Performance smoke test T044.

Objectives:
  - Execute a minimal benchmark run in non-smoke mode with tiny episode counts.
  - Assert overall runtime stays within soft threshold for the synthetic smoke implementation.
  - Validate manifest contains scaling efficiency metrics introduced in T041.

Rationale:
  The current implementation uses synthetic episode execution (no heavy simulation), so a tight
  runtime threshold helps catch accidental performance regressions (e.g., excessive loops or IO).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import (
    BenchmarkManifest,
    _update_scaling_efficiency,
    run_full_benchmark,
)

SOFT_RUNTIME_SEC = 20.0  # generous for CI now that real simulation runs


def test_performance_smoke(config_factory):
    # Configure very small run: 1 initial, max 2 episodes per scenario, batch size 1
    """TODO docstring. Document this function.

    Args:
        config_factory: TODO docstring.
    """
    cfg = config_factory(
        smoke=True,
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
    manifest = json.loads(data)
    scaling = manifest["scaling_efficiency"]
    assert scaling["throughput_per_worker"] >= 0.0
    assert scaling["parallel_efficiency"] == "not_available"
    assert scaling["parallel_efficiency_basis"] == "requires measured sequential baseline"
    assert scaling["evidence_status"] == "smoke_only_non_evidence"
    assert scaling["parallel_efficiency_placeholder_deprecated"] is True


def test_scaling_compatibility_alias_is_zero_without_throughput(tmp_path: Path) -> None:
    """Deprecated efficiency alias should not report nonzero efficiency for zero throughput."""

    class _Cfg:
        """Minimal config stub for scaling diagnostics."""

        workers = 4
        smoke = True

    manifest = BenchmarkManifest(
        output_root=tmp_path,
        git_hash="test",
        scenario_matrix_hash="test",
        config=_Cfg(),
        episodes_path=str(tmp_path / "episodes.jsonl"),
    )
    manifest.created_at -= 10.0
    manifest.executed_jobs = 0

    scaling = _update_scaling_efficiency(manifest, _Cfg())

    assert scaling["episodes_per_second"] == 0.0
    assert scaling["throughput_per_worker"] == 0.0
    assert scaling["parallel_efficiency_placeholder"] == 0.0
