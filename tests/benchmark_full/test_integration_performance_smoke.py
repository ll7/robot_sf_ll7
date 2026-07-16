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

  To stay a reliable readiness signal we use the smoke's own CPU time (``time.process_time``) for
  the soft budget. CPU time counts only this process's consumption, so transient contention from
  *other* processes does not inflate it, while extra CPU work in our code still breaches the
  threshold. A separate, deliberately generous wall-time fail-safe retains coverage for blocking
  and I/O regressions without restoring the load-sensitive 20-second wall-clock assertion. The
  contract is documented in docs/context/issue_1436_reproducibility_flaky_acceptance.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic.orchestrator import (
    BenchmarkManifest,
    _update_scaling_efficiency,
    run_full_benchmark,
)

# The isolated deterministic baseline is ~1.2s wall / well under 1s CPU. The soft CPU budget
# detects excess computation without charging this process for unrelated host load. The wall
# fail-safe is intentionally above the observed 21.479s loaded run and catches blocking or I/O
# regressions that process CPU time cannot see.
SOFT_CPU_RUNTIME_SEC = 20.0
HARD_WALL_RUNTIME_SEC = 60.0


def _assert_runtime_within_budget(
    *,
    cpu_elapsed: float,
    wall_elapsed: float,
    cpu_limit: float = SOFT_CPU_RUNTIME_SEC,
    wall_limit: float = HARD_WALL_RUNTIME_SEC,
) -> None:
    """Fail when computation or end-to-end runtime breaches its respective budget."""
    assert cpu_elapsed < cpu_limit, (
        f"Performance smoke exceeded soft CPU-time threshold: {cpu_elapsed:.3f}s"
    )
    assert wall_elapsed < wall_limit, (
        f"Performance smoke exceeded hard wall-time threshold: {wall_elapsed:.3f}s"
    )


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
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    run_full_benchmark(cfg)
    _assert_runtime_within_budget(
        cpu_elapsed=time.process_time() - cpu_start,
        wall_elapsed=time.perf_counter() - wall_start,
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


def test_runtime_contract_rejects_controlled_threshold_breaches() -> None:
    """Controlled CPU and wall-time regressions must fail without consuming test resources."""
    with pytest.raises(AssertionError, match="soft CPU-time threshold"):
        _assert_runtime_within_budget(
            cpu_elapsed=0.02,
            wall_elapsed=0.0,
            cpu_limit=0.01,
            wall_limit=1.0,
        )

    with pytest.raises(AssertionError, match="hard wall-time threshold"):
        _assert_runtime_within_budget(
            cpu_elapsed=0.0,
            wall_elapsed=0.02,
            cpu_limit=1.0,
            wall_limit=0.01,
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
