# Issue #6464 — BRNE Corridor Diagnostic

Status: **diagnostic complete under declared single-thread isolation; no planner-ranking or benchmark claim**.

BRNE (Bayesian Recursive Nash Equilibrium) now has a fail-closed map-runner
adapter for the approved corridor-only preflight. The bounded run below proves
that the pinned upstream core can execute through the Robot SF adapter and emit
non-degenerate native-core actions. It does not establish safety, realism,
matched-compute parity, planner superiority, or paper evidence.

## Frozen contract

- Scenario: `classic_head_on_corridor_low` from
  `configs/scenarios/issue_6464_brne_corridor_diagnostic.yaml`.
- Seeds: `111`, `112`, and `113`; horizon `500`; timestep `0.1` seconds.
- Geometry: one constant-width corridor only. Doorways, T-junctions,
  arbitrary static geometry, and crowds above the upstream cap are excluded.
- BRNE source: `MurpheyLab/brne` at commit `633a5cd`, GPL-3.0, staged locally
  and not vendored or redistributed.
- The upstream request is `49` samples; its pinned grid produces `42` actual
  samples, frozen as `expected_effective_num_samples: 42`. The adapter records
  the effective count and handles the upstream plan-step-first tensor layout.
- The upstream random generator is rebound from each declared episode seed at
  planner reset, so repeated runs use the same BRNE sampling stream.
- Fallback is disabled. Fallback, degraded, unknown, failed, over-cap, or
  degenerate rows are unavailable rather than success evidence. Solver stop
  actions carry runtime failure provenance and are excluded even when the
  trace has non-zero displacement.
- The completed rerun used `NUMBA_NUM_THREADS=1` to isolate the native BRNE
  timing path. A default-thread repeat previously exposed the known
  load-sensitive seed-112 budget failure and remains incomplete rather than
  success evidence; the source-smoke guard itself was stabilized by merged PR
  [#6931](https://github.com/ll7/robot_sf_ll7/pull/6931) for [#6924](https://github.com/ll7/robot_sf_ll7/issues/6924).

## Observed result

All three arms covered the exact three seed pairs. The comparator rows provide
paired diagnostic coverage only; they are not a ranking or benchmark result.

| Arm | Exact runs | Native/execution | Eligible | Goal reached | Non-degenerate | Corridor violations | Fallback/degraded | Over cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BRNE | 3/3 | 3/3 | 3/3 | 0/3 | 3/3 | 0 | 0 | 0 |
| ORCA comparator | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 0 | 0 | 0 |
| Social-force comparator | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 0 | 0 | 0 |

The principal finding is therefore bounded and mixed: the native upstream BRNE
core executed through the Robot SF adapter with non-zero motion and stayed
inside the declared corridor, but reached the
goal in none of the three predeclared episodes. This is a hypothesis signal,
not evidence to widen the scenario class or promote the planner.

## Implementation findings

1. Occupancy-grid map observations expose flattened keys such as
   `robot_position`, `robot_velocity_xy`, `goal_current`, and
   `pedestrians_positions`. The BRNE bridge now normalizes those keys into the
   nested state contract before constructing the upstream payload. Without
   this bridge, the first campaign produced all-zero planner inputs and was
   correctly classified as incomplete.
2. The upstream sample generator can return fewer samples than requested and
   returns plan-step-first action tensors. The source wrapper now sizes its
   arrays from the effective upstream count and normalizes both the pinned
   plan-step-first layout and the legacy samples-first fixture layout fail-closed.
3. The SocNav observation contract stores pedestrian velocities in robot-ego
   coordinates. The BRNE bridge rotates them back to world coordinates before
   prediction and preserves the declared robot heading when it is available.
4. Report completion requires unique exact scenario/seed pairs. Goal-reaching
   counts are restricted to eligible rows; unavailable rows are retained only
   as explicit diagnostics.

## Reproduction and evidence

From the dedicated linked worktree, stage the external source and run:

```bash
uv run python scripts/tools/manage_external_repos.py stage brne
uv run pytest -q tests/baselines/test_brne_source_smoke.py tests/baselines/test_brne_planner.py
NUMBA_NUM_THREADS=1 uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6464_brne_diagnostic_<timestamp>
```

The clean follow-up report is
`output/benchmarks/issue_6464_brne_diagnostic_followup_20260812T031500Z/`.
Its report JSON SHA-256 is
`b76c1fc9ab08e1272b67fe78c85d341f56e56b62a31b3299c75c293f00a6e62f` and its
Markdown SHA-256 is
`b30a2b7ba441f4b86c558b837c8198cf8bb57f7e0bf2e811938cf36b6d8125e0`.
Every episode records implementation commit
`cb5ee5700ebe262733dd29e8e97c74deaff03cc7`; the staged upstream source is
validated at pinned commit
`633a5cdcb39ab27f18b596cb8cb1968644f82391` with clean tracked source.
Raw episode files and the staged GPL source remain ignored, worktree-local
artifacts. The compact evidence handoff is tracked in
`docs/context/evidence/issue_6464_brne_corridor_diagnostic_summary.json`.

## Next decision

Open follow-up issue [#6923](https://github.com/ll7/robot_sf_ll7/issues/6923)
for a hypothesis-driven diagnostic of the `0/3` BRNE goal-reaching result before
any broader campaign. Candidate checks are goal/heading frame alignment,
progress-versus-interaction weighting, and whether the pinned upstream action
aggregation is appropriate for this control loop. The follow-up preserves the
same native-core-via-adapter, corridor-only, fail-closed boundary.

## Issue #6923 trace follow-up (2026-08-12)

Status: **diagnostic-only retention; no adapter revision, negative-result closure,
ranking, or paper claim**.

The exact current-head run was executed from implementation head
`ed9bd4dcc01d3af8d498cbba74f4bc5f62bea227` with the frozen config, seeds,
fallback policy, and pinned upstream source:

```bash
NUMBA_NUM_THREADS=4 LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6923_brne_trace_20260812T0625Z
```

The run returned `diagnostic_complete`: all three arms have exact `3/3`
scenario/seed coverage and `3/3` trace tables. BRNE is `3/3` native,
execution-admissible, non-degenerate, and pinned to effective sample count
`42`, but reaches the goal in `0/3` rows and collides in all three. ORCA
reaches `3/3`; Social Force reaches `2/3`. These are paired diagnostic
comparators, not a planner ranking. The direct configured corridor band from
the merged #6932/#6933 contract is used; all arms have zero corridor
violations.

The trace extractor now preserves the initial and changing route waypoints,
declared heading, velocity-derived heading, world-frame pedestrian positions
and velocities, goal-bearing error, selected commands and command deltas,
route-switch markers, phase progress, exposure, clearance, termination,
collision/goal steps, runtime failures, effective samples, source provenance,
and the BRNE `weighted_first_command` / `plan_step_first` aggregation layout.
Progress is reset at a waypoint switch so the distance to one waypoint is not
silently compared with the next waypoint's distance.

Observed BRNE signals from the durable trace tables:

- Seed 111: 142 steps, collision at step 141, maximum absolute heading-goal
  error `2.419 rad`, first opposing-heading row 87, and zero nonzero angular
  commands in the remaining `55` rows; linear command stayed at `2.0 m/s`.
- Seed 112: 173 steps, collision at step 172, maximum absolute heading-goal
  error `2.608 rad`, first opposing-heading row 131, and zero nonzero angular
  commands in the remaining `42` rows; linear command stayed at `2.0 m/s`.
- Seed 113: 184 steps, collision at step 183, maximum absolute heading-goal
  error `2.853 rad`, first opposing-heading row 131, and `4/53` nonzero
  angular commands after that threshold; linear command stayed at `2.0 m/s`.

The maximum absolute difference between declared and velocity-derived heading
was `0.0413`, `0.0370`, and `0.0369 rad` for seeds 111–113. This provides no
trace evidence of a heading/velocity-frame mismatch in the recorded state.
Seeds 112 and 113 had zero interaction-zone exposure but still collided, so
pedestrian interaction is not a necessary explanation. The sparse angular
correction after opposing-heading rows is a supported diagnostic signal, not a
proven upstream cause. The trace confirms the aggregation layout but does not
expose per-ensemble-member headings or weights; the internal aggregation
hypothesis therefore remains unresolved and does not justify an adapter change.

The clean run required `NUMBA_NUM_THREADS=4` in this environment because
default-thread attempts exposed load-sensitive BRNE step-budget failures. The
thread setting is recorded as an execution caveat, not a scientific result;
runtime determinism needs a separate bounded follow-up. Raw JSONL and trace
tables remain ignored, worktree-local artifacts. The durable machine-readable
handoff is
`docs/context/evidence/issue_6923_brne_trace_diagnostic_summary.json`, with
report and trace-table hashes plus the exact provenance boundary.

The next proof is already tracked in [#6934](https://github.com/ll7/robot_sf_ll7/issues/6934): isolate the upstream weight/action-unit contract and raw-command clipping with a hand-checkable ensemble fixture. The load-sensitive timing caveat is related to the completed support issue [#6924](https://github.com/ll7/robot_sf_ll7/issues/6924); it remains an execution caveat for this diagnostic and is not promoted to a scientific result.
