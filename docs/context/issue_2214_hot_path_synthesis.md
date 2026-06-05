# Issue #2214 Hot-Path Optimization Synthesis

Date: 2026-06-04

Status: diagnostic synthesis; no paper-facing speed claim.

## Scope

Issue #2214 asked for a synthesis of the simulator hot-path optimization wave
after the allocation and snapshot-reuse PRs. The issue-specified wave starts
after `5eead086fceac0bbdd00bb10ec133a612dfc5b25` and runs through `c125d5ae`.
The local measurement pass compared that baseline against current main at
`5bd87d58d4869e8420943301e45de1a8dc6513a1`, which also includes the later
`#2219` Social Force velocity-diff reuse PR.

## Commands

The synthesis used the same command shapes on the baseline and current
worktrees:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/performance_smoke_test.py --num-resets 5
```

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/validation/run_policy_search_candidate.py \
    --candidate hybrid_rule_v3_fast_progress \
    --stage smoke \
    --workers 1
```

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
ROBOT_SF_SIM_STEPS_SOFT=2.0 ROBOT_SF_SIM_STEPS_HARD=0.5 ROBOT_SF_PERF_ENFORCE=0 \
  uv run pytest tests/perf/test_simulation_speed_perf.py \
    -k test_simulation_step_throughput -q
```

## Results

| Surface | Baseline | Current | Delta | Interpretation |
|---|---:|---:|---:|---|
| Environment creation | 3.3084 sec | 3.2473 sec | -1.8% | Inconclusive; both runs retained the smoke suite warning status. |
| Reset throughput | 830.49 resets/sec | 790.54 resets/sec | -4.8% | No measured speedup on this short smoke surface. |
| Policy-search smoke runtime | 9.8605 sec | 10.0573 sec | +2.0% | No measured speedup; one-episode smoke is startup dominated. |
| Simulation throughput pytest | skipped | skipped | n/a | Both revisions skipped under local thresholds, so the result was excluded from the quantitative synthesis. |

The compact derived evidence lives in
`docs/context/evidence/issue_2214_hot_path_synthesis_2026-06-04/summary.json`.

## Interpretation

This local evidence does not show a broad end-to-end speedup from the hot-path
wave on single-row smoke surfaces. The most defensible conclusion is narrower:
the merged PRs likely reduced allocation pressure and clarified hot-path data
ownership, but those effects were not visible above cold startup and first-run
noise in these smoke commands.

The strongest recent wall-clock speed evidence remains issue #2172, where the
18-job h80 nominal-sanity benchmark runner improved from 86.314 seconds with one
worker to 48.140 seconds with two workers, a 1.793x speedup. That comparison
suggests near-term speed work should prefer representative multi-episode or
worker-scaling profiles over more single-row cold smoke measurements.

## Follow-Up Direction

1. Before another hot-path micro-optimization wave, add or run a semantic
   equivalence guard for optimized simulator paths so speed edits do not become
   behavior edits by accident.
2. Use a representative multi-episode benchmark profile when making future
   wall-clock speed claims.
3. Treat the short performance smoke and one-candidate policy smoke commands as
   regression smoke only, not as speedup proof.
