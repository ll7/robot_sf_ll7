"""Performance smoke test T044.

Objectives:
  - Execute a minimal benchmark run in non-smoke mode with tiny episode counts.
  - Assert overall runtime stays within a soft threshold for the smoke implementation.
  - Validate manifest contains scaling efficiency metrics introduced in T041.

Rationale:
  The current implementation runs a real (lightweight) benchmark, so a soft runtime threshold
  helps catch accidental performance regressions (e.g., excessive loops or IO). Issue #5876 showed
  the smoke breaches the 20s soft threshold when an unrelated parallel readiness lane (16 xdist
  workers plus other repository test work) contends for the same host: the deterministic isolated
  baseline is ~1.2s, but shared-host CPU contention pushed wall time to 21.479s.

  To stay a reliable readiness signal we measure the smoke's own CPU time (``time.process_time``)
  instead of wall-clock. CPU time counts only this process's own consumption, so transient
  contention from *other* processes does not inflate it, while a genuine representative-path
  regression (more CPU work in our code) still grows it proportionally and breaches the threshold.
  The contract is fully deterministic across host load and is documented in
  docs/context/issue_1436_reproducibility_flaky_acceptance.md.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import (
    BenchmarkManifest,
    _update_scaling_efficiency,
    run_full_benchmark,
)

# Soft CPU-time budget for the smoke. The isolated deterministic baseline is ~1.2s
# wall / well under 1s CPU, so this is generous enough for quiet hosts yet tight
# enough that a real regression (e.g. excessive loops or IO) still fails.
SOFT_RUNTIME_SEC = 20.0


def test_performance_smoke(config_factory):
    # Configure very small run: 1 initial, max 2 episodes per scenario, batch size 1
    """Execute the smoke and assert its CPU time stays within the soft threshold.

    Args:
        config_factory: Factory producing a BenchmarkConfig test double.
    """
    cfg = config_factory(
        smoke=True,
        initial_episodes=1,
        max_episodes=2,
        batch_size=1,
        workers=1,
    )
    start = time.process_time()
    run_full_benchmark(cfg)
    elapsed = time.process_time() - start
    assert elapsed < SOFT_RUNTIME_SEC, (
        f"Performance smoke exceeded soft CPU-time threshold: {elapsed:.3f}s"
    )

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


def test_cpu_time_metric_ignores_external_host_contention() -> None:
    """External (cross-process) CPU contention must not inflate this process's CPU time.

    The smoke relies on ``time.process_time`` so that an unrelated parallel readiness lane
    on the same host cannot push the measurement over threshold. This test confirms the
    metric only counts our own CPU consumption: a background process burning cores does not
    raise a short CPU-time sample above the same value measured on a quiet host.
    """

    def _measure() -> float:
        start = time.process_time()
        # representative in-process busy work
        total = 0
        for _ in range(200_000):
            total += 1
        return time.process_time() - start

    quiet = _measure()

    import subprocess

    # Spawn external CPU burners in separate processes (mimics the parallel readiness lane).
    burners = []
    cores = max(1, (os.cpu_count() or 4) - 1)
    for _ in range(cores):
        burners.append(
            subprocess.Popen(
                [sys.executable, "-c", "while True:\n pass"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    try:
        loaded = _measure()
    finally:
        for proc in burners:
            proc.kill()
    # Our process's own CPU time is essentially unchanged by external contention.
    assert loaded <= quiet * 2.0 + 0.05


def test_cpu_time_contract_rejects_deliberate_regression(config_factory) -> None:
    """A deliberately CPU-heavy benchmark must still breach the soft threshold."""

    cfg = config_factory(
        smoke=True,
        initial_episodes=1,
        max_episodes=2,
        batch_size=1,
        workers=1,
    )

    # Inject a deliberate CPU-bound slowdown into the smoke path. Because the
    # regression itself consumes our process's CPU, CPU-time measurement exposes it.
    import robot_sf.benchmark.full_classic.orchestrator as orch

    original = orch.run_full_benchmark

    def _slow(*args, **kwargs):
        busy_until = time.process_time() + (SOFT_RUNTIME_SEC * 2.0)
        while time.process_time() < busy_until:
            pass
        return original(*args, **kwargs)

    orch.run_full_benchmark = _slow
    try:
        start = time.process_time()
        orch.run_full_benchmark(cfg)
        elapsed = time.process_time() - start
    finally:
        orch.run_full_benchmark = original
    assert elapsed >= SOFT_RUNTIME_SEC, (
        f"Deliberate regression below threshold: {elapsed:.3f}s < {SOFT_RUNTIME_SEC:.3f}s"
    )


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
