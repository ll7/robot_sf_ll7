"""Integration test T017: resume behavior (accelerated minimal matrix variant).

Purpose:
    Verify that running the full benchmark twice with identical configuration
    (same output_root) does not duplicate episodes — the second invocation must
    detect existing work and *not* append lines to ``episodes.jsonl``.

Acceleration Strategy (mirrors reproducibility test):
    - Inject a single-scenario minimal matrix (instead of loading the full classic matrix)
    - Force tiny workload: initial_episodes = batch_size = max_episodes = 2
    - Disable bootstrap (bootstrap_samples = 0)
    - horizon_override = 12 (short synthetic episode generation)
    - workers = 1 (avoid multi‑process overhead and ordering variance)

Performance & Safety:
    - Hard timeout via pytest marker + signal alarm (60s absolute ceiling)
    - Expected total episodes generated: exactly 2
    - Soft timing guidance: <2s local / <5s CI (warn if slower, fail >20s)
    - Guard against unexpected heavy artifacts (videos) in smoke mode

Failure Guidance:
    If this test slows down, ensure the minimal matrix is still used and that
    adaptive loop exit condition (max episodes budget) is reached after the first
    iteration. The budget is ``max_episodes * len(scenarios)``; with one scenario
    and max_episodes=2 the adaptive loop should terminate immediately after the
    initial two jobs.

Minimization Rationale (T026):
    Uses the same minimal matrix + 2 episode budget as the reproducibility
    test to keep runtime low while targeting a distinct property (resume
    idempotency). Increasing episode count, horizon, or enabling bootstrap
    would not improve confidence in the resume behavior and would slow the
    suite. Keep this file focused; create a separate stress test if higher
    load resume scenarios need validation.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from tests.perf_utils.minimal_matrix import write_minimal_matrix  # Shared helper (T011)


def _tune_config(cfg):  # consolidated minimal tuning (T011 refactor)
    """TODO docstring. Document this function.

    Args:
        cfg: TODO docstring.
    """
    for attr, value in [
        ("workers", 1),
        ("bootstrap_samples", 0),
        ("initial_episodes", 2),
        ("batch_size", 2),
        ("max_episodes", 2),
        ("horizon_override", 12),
        ("collision_ci", 1.0),  # lenient precision targets for early exit
        ("success_ci", 1.0),
    ]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, value)
    return cfg


def _inject_minimal_matrix(cfg):  # delegate to shared writer
    """TODO docstring. Document this function.

    Args:
        cfg: TODO docstring.
    """
    root = Path(cfg.output_root)
    root.mkdir(parents=True, exist_ok=True)
    mini_path = write_minimal_matrix(root)
    if hasattr(cfg, "scenario_matrix_path"):
        cfg.scenario_matrix_path = str(mini_path)  # type: ignore[attr-defined]
    return cfg


def _read_lines(path: Path):
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
    """
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _assert_no_video_artifacts(root: Path):
    """TODO docstring. Document this function.

    Args:
        root: TODO docstring.
    """
    vids = list(root.rglob("*.mp4")) + list(root.rglob("*.gif"))
    assert not vids, f"Unexpected video artifacts produced: {vids}"


@pytest.mark.timeout(60)
def test_resume_skips_existing(config_factory, perf_policy):
    """TODO docstring. Document this function.

    Args:
        config_factory: TODO docstring.
        perf_policy: TODO docstring.
    """
    start = time.perf_counter()
    hard_timeout_sec = 60

    def _timeout_handler(signum, frame):  # pragma: no cover
        """TODO docstring. Document this function.

        Args:
            signum: TODO docstring.
            frame: TODO docstring.
        """
        raise TimeoutError("Resume integration test exceeded hard timeout (60s)")

    try:
        prev = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(hard_timeout_sec)
    except AttributeError:
        prev = None

    try:
        cfg = config_factory(smoke=True, workers=1)
        cfg = _inject_minimal_matrix(cfg)
        cfg = _tune_config(cfg)
        manifest_first = run_full_benchmark(cfg)
        episodes_file = manifest_first.output_root / "episodes" / "episodes.jsonl"
        assert episodes_file.is_file(), "episodes.jsonl missing after first run"
        first = _read_lines(episodes_file)
        expected_eps = 2
        assert len(first) == expected_eps, (
            f"Expected {expected_eps} episodes (minimal matrix), got {len(first)}"
        )

        # Second run (resume) should not add lines
        run_full_benchmark(cfg)
        second = _read_lines(episodes_file)
        assert len(second) == len(first) == expected_eps, (
            f"Resume failed: second run line count changed from {len(first)} to {len(second)}"
        )

        _assert_no_video_artifacts(Path(manifest_first.output_root))

        elapsed = time.perf_counter() - start
        classification = perf_policy.classify(elapsed)
        assert classification != "hard", (
            f"Elapsed {elapsed:.2f}s breached hard threshold {perf_policy.hard_timeout_seconds}s."
        )
        # Maintain broad expectation (<20s) using policy soft threshold (currently 20s)
        assert elapsed < perf_policy.soft_threshold_seconds + 1e-6, (
            f"Resume test too slow: {elapsed:.2f}s (>{perf_policy.soft_threshold_seconds}s)"
        )
        ci = bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))
        soft = 5.0 if ci else 2.0
        if elapsed > soft:
            print(
                f"[WARN] Resume test elapsed {elapsed:.2f}s exceeded soft limit {soft:.2f}s. "
                "Check for unintended workload increase.",
            )
    finally:
        try:
            signal.alarm(0)
            if "prev" in locals() and prev is not None:
                signal.signal(signal.SIGALRM, prev)
        except Exception:
            pass
