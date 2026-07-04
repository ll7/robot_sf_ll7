# Issue #4404 Trace-Capable H600 Runtime-Scaling Context

This note records the workload surface for the trace-capable horizon-600
(`h600`) re-run config before any private queue submission or benchmark
interpretation.

## Scope

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4404>
- Predecessor pull request: <https://github.com/ll7/robot_sf_ll7/pull/4417>
- Runnable config:
  `configs/benchmarks/paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml`
- Pre-registration contract:
  `configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml`
- Downstream blocked consumer:
  `configs/analysis/issue_4206_policy_structure_mechanism_crosscut.yaml`
- Source h600 context jobs: `13268` confirm run and `13273` extended-roster run.
- Claim boundary: context and dispatch readiness only. This note does not report
  campaign results, promote an F-C4(ii) policy-structure causal-upgrade claim,
  edit paper/dissertation claims, submit a Slurm (Simple Linux Utility for
  Resource Management) job, or estimate wall-clock runtime without measured
  evidence.

## Why This Re-Run Exists

Issue #4206 needs failure-mechanism labels for retained h600 rows, but source
jobs `13268` and `13273` predate the trace-capable episode exporter. Their
retained episode rows cannot provide trace-verified mechanism labels, and a
sidecar cannot reconstruct traces that were never recorded. PR #4417 added the
runnable campaign config that turns on both trace exporters for a fresh h600 run.

The runnable config keeps the re-run tied to the existing issue #4206 contract:

- scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`,
  with expected hash `c10df617a87c`;
- horizon: `600`;
- seeds: fixed list `[20, 21, 22, 23, 24]`;
- planner roster: 12 keys in the order validated by
  `tests/validation/test_issue_4206_trace_capable_h600_rerun_preregistration.py`;
- trace capture: `record_planner_decision_trace: true` and
  `record_simulation_step_trace: true`;
- `guarded_ppo`: dependency-gated and fail-closed until checkpoint/config
  availability is verified before launch.

## Static Workload Surface

The repository scenario loader resolves
`configs/scenarios/classic_interactions_francis2023.yaml` to 48 scenarios. The
checked-in issue #4404 config therefore describes:

| Factor | Value | Provenance |
| --- | ---: | --- |
| Scenario count | 48 | `robot_sf.training.scenario_loader.load_scenarios` |
| Seed count | 5 | `seed_policy.seeds` in the runnable config |
| Planner arms | 12 | `planners` in the runnable config |
| Planned episode rows | 2,880 | `48 scenarios * 5 seeds * 12 planners` |
| Horizon | 600 steps | `horizon: 600` |
| Config worker setting | 1 | `workers: 1` in the runnable config |

Relative to the earlier Issue #4230 h600 hybrid-roster pre-registration surface
of 48 scenarios, 3 seeds, and 4 planner arms, this trace-capable re-run is a
`5x` planned-episode expansion before trace export overhead, dependency-gated
planners, queue placement, retries, or fallback exclusions. This ratio is static
config arithmetic only, not a measured wall-clock runtime prediction.

## No-Submit Boundary

This PR records no transient queue-routing state: no target host, packet lineage
pointer, private queue ID, Slurm job ID, or submission timestamp. That state
belongs in the append-only issue state surface or private operations state, not
in this tracked context note.

The next valid empirical action is an explicitly authorized private submission
using the runnable config after launch-time dependency checks pass. Until that
happens, the issue #4206 F-C4(ii) policy-structure causal-upgrade path remains
blocked on trace-capable h600 outputs and must not be treated as benchmark,
paper, or dissertation evidence.

## Validation Path

Focused validation for this note should remain docs/config only:

```bash
scripts/dev/run_worktree_shared_venv.sh -- pytest \
  tests/validation/test_issue_4206_trace_capable_h600_rerun_preregistration.py
```

No full benchmark campaign, Slurm/GPU submission, or private queue action is
part of this context-note slice.
