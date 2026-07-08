# Issue #4792: Scenario Criticality Real-Runner Close-Out

**Status**: Core real-runner implementation complete; residuals remain tracked on
issue #4792. This note documents the landed implementation, policy decisions, and
remaining residuals.

**Related**: [#4362](https://github.com/ll7/robot_sf_ll7/issues/4362), [#4792](https://github.com/ll7/robot_sf_ll7/issues/4792)

**Last Updated**: 2026-07-08

## Overview

Issue #4792 was opened to complete the criticality optimizer by connecting it to the real
simulator runner (replacing mock placeholder metrics). The core real-runner path has been
implemented through a series of PRs. This note documents the implementation state, policy
decisions, and remaining follow-up work still tracked on the parent issue.

## PR Ledger

The following PRs landed the core functionality:

| PR | Date | Description |
|----|------|-------------|
| [#4804](https://github.com/ll7/robot_sf_ll7/pull/4804) | 2026-07-07 | Real runner integration - `_evaluate_candidate` now runs actual episodes via `run_map_batch` instead of mock placeholder metrics |
| [#4812](https://github.com/ll7/robot_sf_ll7/pull/4812) | 2026-07-07 | Canonical collision-key handling (`CANONICAL_COLLISION_KEY` with fallbacks) and optional parallel candidate execution (`max_workers`) |
| [#4819](https://github.com/ll7/robot_sf_ll7/pull/4819) | 2026-07-07 | Shared planner resolution - pre-resolve planner spec once in main process, workers reuse it |
| [#4821](https://github.com/ll7/robot_sf_ll7/pull/4821) | 2026-07-08 | `differential_evolution` optimizer type behind the shared interface |
| [#4822](https://github.com/ll7/robot_sf_ll7/pull/4822) | 2026-07-08 | CPU validation confirming objective responsiveness and real-runner integration |
| [#4823](https://github.com/ll7/robot_sf_ll7/pull/4823) | 2026-07-08 | `cma_es` optimizer type (CMA-ES) with dependency and formatting fixes |
| [#4835](https://github.com/ll7/robot_sf_ll7/pull/4835) | 2026-07-08 | Close-out policy note documenting implementation state and residuals |
| [#4839](https://github.com/ll7/robot_sf_ll7/pull/4839) | 2026-07-08 | Per-candidate timing breakdown profiling (patch_s, simulation_s, score_s) |

## Canonical Evaluator Policy

The canonical evaluator path for #4792/#4362 criticality search is:

**`metrics_source == simulator_run_map_batch`**

This means:
- Baseline-vs-best comparisons must be generated from real simulator episode runs
- Analytical or fixture evaluators are allowed only for tests or debug smoke, never as
  criticality-search evidence
- Any alternative evaluator (e.g., deterministic analytical) must use a separate backend
  name and must not replace `simulator_run_map_batch` for evidence generation

### Evidence Path Requirements

All criticality-search evidence must satisfy:
1. Real simulator episodes executed through `run_map_batch`
2. Metrics extracted via `compute_criticality_score` from actual episode data
3. `metrics_source: simulator_run_map_batch` recorded in the manifest
4. `collision_key_used` and `collision_key_status` recorded for provenance
5. Fail-closed `not_evaluable`/`invalid_candidate` handling for missing metrics

### Alternative Evaluators (Allowed Scope)

Analytical or fixture-based evaluators are allowed only for:
- Unit tests (`test_scenario_criticality_objective.py`)
- Debug smoke runs (explicit `--diagnostic` flag)
- Contract validation (checking the objective responds to perturbations)

They must never be used for:
- Baseline-vs-best comparisons in published results
- Criticality-search optimization sweeps
- Any claims about scenario difficulty or stress-test efficacy

## Current Implementation State

### Core Components

1. **Objective Function** (`robot_sf/benchmark/scenario_criticality_objective.py`)
   - `compute_criticality_score()`: Scalar score for optimization
   - `compute_criticality_decomposed()`: Detailed metric breakdown
   - `apply_criticality_parameters()`: Scenario parameter patching with deep-copy safety
   - `_resolve_collision_key()`: Canonical key with explicit fallback chain
   - Fail-closed `not_evaluable` status for missing metrics

2. **Optimization Script** (`scripts/benchmark/run_scenario_criticality_optimization.py`)
   - Random search baseline (`random_search`)
   - Differential evolution optimizer (`differential_evolution`)
   - CMA-ES optimizer (`cma_es`)
   - Parallel candidate execution (`max_workers`)
   - Shared planner resolution (pre-resolve before worker dispatch)
   - Per-candidate result isolation with deterministic ordering

3. **Validation** (`tests/benchmark/test_scenario_criticality_*.py`)
   - Objective responds to collision/near-miss/clearance perturbations
   - `apply_criticality_parameters` deep-copy safety verified
   - Baseline vs perturbed candidates both evaluate with `metrics_source == simulator_run_map_batch`
   - Sequential and parallel execution are semantically equivalent
   - DE/CMA-ES optimizers produce finite scores on bounded fixtures

### Optimizer Types

| Type | Status | Notes |
|------|--------|-------|
| `random_search` | ✅ Implemented | Baseline, deterministic, no extra dependency |
| `differential_evolution` | ✅ Implemented | SciPy-based, deterministic under configured seed |
| `cma_es` | ✅ Implemented | CMA package dependency, deterministic under seed |
| `bayesian` | ❌ Not implemented | Post-submission stretch if needed |

## Objective Responsiveness Evidence

From PR #4822 CPU validation:

- The criticality score increases monotonically with:
  - Collision count (weighted by `collision_weight`, default 10.0)
  - Near-miss events (weighted by `near_miss_weight`, default 2.0)
  - Low clearance (below `clearance_margin`, default 0.5m)
  - Progress failures (weighted by `progress_failure_weight`, default 5.0)
  - Stalled time (weighted by `stalled_time_weight`, default 0.5)

- Unit tests confirm:
  - Score responds to collision count (baseline vs 1-collision vs multi-collision)
  - Score responds to near-misses independently
  - Score responds to clearance violations
  - `apply_criticality_parameters` does not mutate the original scenario
  - Parameter→scenario mapping covers both `pedestrians` and `single_pedestrians` groups

- Integration tests confirm:
  - Baseline and perturbed candidates both evaluate with `metrics_source == simulator_run_map_batch`
  - DE optimizer produces `de_optimum`, `de_trial_*`, and baseline with finite scores
  - Sequential and parallel execution produce semantically equivalent results

## Parallel Execution Policy

### Current Implementation

- `max_workers: 1` → sequential execution
- `max_workers: N` → up to N parallel candidates via `ProcessPoolExecutor`
- `max_workers: 0` → auto-detect CPU count with `os.cpu_count()` and fallback to 1

### Determinism Guarantees

- Candidate results are sorted by `candidate_id` before output
- Worker completion order does not affect artifact order
- `ProcessPoolExecutor` with deterministic seed ensures reproducibility

### Current Limitations

- No batch-mode execution (each candidate runs full `run_map_batch` independently)
- No shared planner batch optimization beyond the pre-resolution step

## Collision-Key Schema Policy

### Canonical Key

```python
CANONICAL_COLLISION_KEY = "collision_count"
COLLISION_KEY_FALLBACKS = ("agent_collision_count", "total_collision_count", "collisions")
```

### Resolution Behavior

1. If `collision_count` exists → use it, record `collision_key_used="collision_count"`
2. Else if a fallback exists → use it, record `collision_key_used=<fallback>` and
   `collision_key_status="legacy_fallback"`
3. Else if no collision key exists → `not_evaluable` status, never score as zero collisions

### Schema Status

- The episode-metrics schema has not yet stabilized on `collision_count` as the sole key
- Fallback acceptance remains necessary for legacy fixtures
- A future schema-finalization pass should:
  - Migrate all fixtures to use `collision_count`
  - Remove fallback acceptance or keep it only behind an explicit legacy flag
  - Close out the `collision_key_status="legacy_fallback"` residual

## Remaining Residuals

### Post-Submission Follow-ups (Out of Scope for #4792)

1. **Parallel-batch tuning**
   - Timing/profiling fields per candidate implemented (PR #4839: patch_s, simulation_s, score_s)
   - Batch-mode execution not yet implemented (would require grouping candidates and sharing more state)
   - Current parallelization (ProcessPoolExecutor with shared planner resolution) is sufficient for bounded optimization sweeps
   - Batch-mode should be implemented only if profiling data shows clear bottlenecks

2. **Collision-key schema finalization**
   - Migrate all fixtures to use `collision_count`
   - Remove fallback acceptance or add explicit legacy flag
   - Depends on external episode-metrics schema settlement

3. **Bayesian optimizer** (stretch)
   - Not required for #4792 close-out
   - Can be added post-submission if needed for optimizer comparison

## Claim Boundary

**This implementation is diagnostic/research-only.**

It does NOT provide:
- Validated benchmark evidence
- Stress-test coverage guarantees
- Scenario difficulty quantification
- Planner quality assessment

It DOES provide:
- A bounded classical-optimization baseline for learned/adaptive approaches
- Exploratory diagnostic capability for scenario parameter search
- Reproducible artifact generation with provenance metadata

Any use of this implementation for publication-facing claims requires separate validation
per the repository's evidence grading ladder (diagnostic-only → smoke → nominal benchmark →
paper-grade).

## Verification Commands

```bash
# Check code quality
uv run ruff check \
  scripts/benchmark/run_scenario_criticality_optimization.py \
  robot_sf/benchmark/scenario_criticality_objective.py \
  tests/benchmark/test_scenario_criticality_optimization.py \
  tests/benchmark/test_scenario_criticality_objective.py

# Run tests
uv run pytest \
  tests/benchmark/test_scenario_criticality_objective.py \
  tests/benchmark/test_scenario_criticality_optimization.py -q

# Run smoke (sequential)
uv run python scripts/benchmark/run_scenario_criticality_optimization.py \
  --config configs/benchmarks/issue_4362_scenario_criticality_smoke.yaml \
  --output-dir output/issue_4792_real_runner_closeout

# Run smoke (parallel, if max_workers > 1 in config)
uv run python scripts/benchmark/run_scenario_criticality_optimization.py \
  --config configs/benchmarks/issue_4362_scenario_criticality_smoke.yaml \
  --output-dir output/issue_4792_real_runner_parallel
```

## Acceptance Status

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Real-runner path canonical | ✅ | PR #4804; `metrics_source: simulator_run_map_batch` in manifest |
| Objective responsive | ✅ | PR #4822; unit tests confirm collision/near-miss/clearance response |
| Baseline-vs-best uses real metrics | ✅ | PR #4804; mock placeholder metrics replaced |
| Parallel execution deterministic | ✅ | PR #4812/#4819; sequential/parallel equivalent |
| Collision-key handling fail-closed | ✅ | PR #4812; `not_evaluable` when no collision key present |
| Multiple optimizers | ✅ | PR #4821 (DE), #4823 (CMA-ES) |
| Canonical evaluator policy documented | ✅ | This note |
| PR ledger complete | ✅ | All landed PRs tracked above |
| Remaining residuals documented | ✅ | See "Remaining Residuals" section |

## Issue State

This note is a state and policy record, not an automatic closure claim for issue #4792.
The parent issue should remain open while the residuals above are still tracked there, or
until maintainers explicitly accept those items as non-blocking and move any durable work
to child follow-up issues.

The core functionality (real-runner integration, objective responsiveness, and
baseline-vs-best comparison from simulator rows) is complete for the diagnostic lane, but
parallel-batch tuning, collision-key schema finalization, and the Bayesian optimizer stretch
remain residual work.
