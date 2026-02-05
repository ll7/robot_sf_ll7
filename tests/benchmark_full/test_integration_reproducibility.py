"""Integration test (accelerated) – reproducibility of episode ordering & hash.

Optimization Header (Spec: specs/123-reduce-runtime-of/):
This accelerated variant intentionally minimizes workload while preserving core
determinism guarantees (episode ordering + scenario hash). It replaces a slower
earlier version by:
    * Reducing workload to 2 episodes total (2 seeds * 1 scenario)
    * Disabling heavy artifacts (videos, plots, bootstrap)
    * Using workers=1 to avoid process pool overhead & ordering variance
    * Adding a timing assertion (<8s local, <16s CI) to detect regressions
    * Deterministic planning replay (no second full benchmark run)

If this test starts failing due to timing on a legitimately slower CI node,
set STRICT_REPRO_TEST=0 (future enhancement) or adjust thresholds in spec.
For functional determinism failures, inspect the first differing episode_id
and scenario planning logic. See research & contract docs for rationale.

Minimization Rationale (T026):
    This test deliberately constrains workload to the bare minimum needed to
    validate determinism (2 episodes total). Any increase (extra scenarios,
    seeds, horizon, bootstrap sampling, multi-worker fan-out, artifact
    generation) would add latency without improving signal for the contract
    assertions. If future determinism regressions require broader coverage,
    add a separate extended test rather than expanding this accelerated one.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic.orchestrator import (
    _episode_id_from_job,
    run_full_benchmark,
)
from robot_sf.benchmark.full_classic.planning import (
    expand_episode_jobs,
    load_scenario_matrix,
    plan_scenarios,
)

# New shared helper for minimal scenario matrix (performance refactor T010)
from tests.perf_utils.minimal_matrix import write_minimal_matrix


def _read_episode_records(root: Path) -> list[dict]:
    """Read all episode JSON records from a benchmark output root."""
    p = root / "episodes" / "episodes.jsonl"
    records: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _run_minimal_benchmark(
    config_factory,
    seeds: list[int],
    run_label: str,
    base_root_override: Path | None = None,
) -> tuple[list[str], str, object]:  # T010/T020/T031
    """Execute a minimal benchmark run and return ordered episode_ids and scenario_matrix_hash.

    Refactored (T010): now delegates minimal matrix creation to shared helper
    tests.perf_utils.minimal_matrix.write_minimal_matrix to avoid duplicate YAML
    logic across benchmark tests. This keeps semantic focus (ordering + hash)
    while minimizing runtime.
    """
    cfg = config_factory(smoke=False, master_seed=seeds[0])
    base_root = Path(cfg.output_root) if base_root_override is None else base_root_override
    run_root = base_root / f"repro_{run_label}"
    run_root.mkdir(parents=True, exist_ok=True)
    cfg.output_root = str(run_root)
    # Deterministic & minimal settings
    for attr, value in [
        ("seeds", seeds),
        ("workers", 1),
        ("initial_episodes", 2),
        ("batch_size", 2),
        ("max_episodes", 2),
        ("disable_videos", True),
        ("bootstrap_samples", 0),
        ("horizon_override", 12),
        ("collision_ci", 1.0),
        ("success_ci", 1.0),
    ]:
        if hasattr(cfg, attr):
            setattr(cfg, attr, value)  # type: ignore[arg-type]

    # Create & point to minimal matrix
    mini_matrix_path = write_minimal_matrix(run_root)
    if hasattr(cfg, "scenario_matrix_path"):
        cfg.scenario_matrix_path = str(mini_matrix_path)  # type: ignore[attr-defined]

    manifest = run_full_benchmark(cfg)
    records = _read_episode_records(Path(manifest.output_root))
    ordered_ids = [r["episode_id"] for r in records]
    scenario_hash = getattr(manifest, "scenario_matrix_hash", "")
    return ordered_ids, scenario_hash, cfg


@pytest.mark.timeout(60)  # marker declared in pytest.ini options; enforcement done manually below
def test_reproducibility_same_seed(
    config_factory,
    perf_policy,
):  # T011-T016 T020-T023 T030-T032 + timeout requirement (policy integrated)
    """Accelerated reproducibility test.

    Contract (specs/123-reduce-runtime-of/):
        1. One minimal run yields episode_id ordering that matches deterministic planning replay.
        2. scenario_matrix_hash matches deterministic hash from matrix contents.
        3. No duplicate episode_ids within a run.
        4. At least one episode generated (sanity / non-empty).
        5. Wall-clock runtime below soft threshold (<8s local, <16s CI). Exceeding threshold = test failure (can relax later).
    """

    seeds = [123, 456]  # Minimal deterministic seed list (length=2 per spec guidance)
    expected_episode_count = 2
    start = time.perf_counter()

    # Install hard timeout (60s) using signal (Unix only). Falls back silently if unavailable.
    hard_timeout_sec = 60

    def _timeout_handler(signum, frame):  # pragma: no cover - timeout path rarely triggered
        """TODO docstring. Document this function.

        Args:
            signum: TODO docstring.
            frame: TODO docstring.
        """
        raise TimeoutError(
            f"Reproducibility test exceeded hard timeout of {hard_timeout_sec}s (signal).",
        )

    # Cross-platform guard: Windows and some environments may not expose SIGALRM.
    # We gracefully skip the alarm setup if unavailable (the @pytest.mark.timeout provides a fallback).
    sigalrm = getattr(signal, "SIGALRM", None)
    orig_handler = None
    if sigalrm is not None:  # Only attempt alarm logic when supported by the platform
        try:
            orig_handler = signal.getsignal(sigalrm)
            signal.signal(sigalrm, _timeout_handler)
            signal.alarm(hard_timeout_sec)
        except Exception:
            try:
                signal.alarm(0)  # type: ignore[attr-defined]
            except Exception:
                pass

    try:
        ids1, hash1, cfg = _run_minimal_benchmark(config_factory, seeds, run_label="a")
    finally:
        # Best effort cleanup - only if SIGALRM existed
        if sigalrm is not None:
            try:
                signal.alarm(0)
                if orig_handler is not None:
                    signal.signal(sigalrm, orig_handler)
            except Exception:
                pass

    elapsed = time.perf_counter() - start

    raw = load_scenario_matrix(cfg.scenario_matrix_path)
    matrix_bytes = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_hash = hashlib.sha1(matrix_bytes).hexdigest()[:12]
    rng = random.Random(int(getattr(cfg, "master_seed", 123)))
    scenarios = plan_scenarios(raw, cfg, rng=rng)
    jobs = expand_episode_jobs(scenarios, cfg)
    planned_ids = [_episode_id_from_job(jb) for jb in jobs]

    # Determinism assertions
    assert ids1 == planned_ids, "Episode ordering differs from deterministic planning."
    assert hash1 == expected_hash, "Scenario matrix hash mismatch."
    assert len(ids1) >= 1, "No episodes generated in minimal benchmark run."
    assert len(ids1) == expected_episode_count, (
        f"Expected {expected_episode_count} episodes, got {len(ids1)}"
    )
    assert len(ids1) == len(set(ids1)), "Duplicate episode_ids encountered within a run."

    # Soft timing guard (currently enforced as hard assert per spec performance envelope)
    ci = os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")
    limit = 16.0 if ci else 8.0
    # Additional broad requirement: total completion should not breach hard policy threshold
    classification = perf_policy.classify(elapsed)
    assert classification != "hard", (
        f"Elapsed {elapsed:.2f}s breached hard threshold {perf_policy.hard_timeout_seconds}s."
    )
    # (Soft classification is acceptable here because this test has its own tighter envelopes below.)

    if elapsed >= limit:  # stricter accelerated envelope
        # T042: richer guidance message
        guidance = (
            "Exceeded time budget. Mitigations: verify smoke=True, bootstrap_samples=0, "
            "workers=1, only 2 seeds, horizon_override minimal. If environment is legitimately slow, "
            "consider raising CI threshold (specs/123-reduce-runtime-of/spec.md)."
        )
        raise AssertionError(
            f"Repro test exceeded time budget: {elapsed:.2f}s (limit {limit}s). {guidance}",
        )

    # T022: assert absence of heavy artifact types (videos) across both run roots
    # (Videos not produced in smoke mode; this guards against regressions.)
    base_root = Path(cfg.output_root)
    produced_video_files = list(base_root.rglob("*.mp4")) + list(base_root.rglob("*.gif"))
    assert not produced_video_files, f"Unexpected video artifacts generated: {produced_video_files}"

    # Horizon note (T023): horizon_override set to 12 to shorten synthetic episode generation.

    # Future extension (T020+): consider multi-worker negative control & mutated seed set.


# Negative control scaffold (T030-T032):
# def _negative_control_example(config_factory):
#     """Illustrative (not executed) negative control showing differing seeds produce different ordering.
#     Kept commented to avoid doubling runtime; serves as guidance for a future extended determinism suite.
#     """
#     seeds_a = [111, 222]
#     seeds_b = [333, 444]
#     ids_a, _ = _run_minimal_benchmark(config_factory, seeds_a, run_label="neg_a")
#     ids_b, _ = _run_minimal_benchmark(config_factory, seeds_b, run_label="neg_b")
#     if ids_a == ids_b:  # pragma: no cover - diagnostic path
#         raise AssertionError("Negative control seeds unexpectedly produced identical ordering")

# Temp dir independence note: each helper call nests under distinct run_root (repro_<label>) inside
# the fixture-provided base output_root ensuring no resume collision (addresses T031 rationale).
