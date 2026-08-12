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

## Historical pre-#6923 handoff

The original handoff below was captured before the exact-head follow-up. It is
retained for provenance, but its eligible-row counts must not be treated as the
current BRNE runtime result.

## Historical observed result

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

## Follow-up decision

Follow-up issue [#6923](https://github.com/ll7/robot_sf_ll7/issues/6923) completed
the hypothesis-driven mechanism diagnosis before any broader campaign. It
preserved the same native-core-via-adapter, corridor-only, fail-closed boundary.

## Final exact-head #6923 reproduction

The final exact-head reproduction was run from commit
`4ee87d61bf8f8b2627f316b3de06f4190c397b57` with single-thread isolation and the
frozen command:

```bash
NUMBA_NUM_THREADS=1 uv run python scripts/benchmark/run_brne_corridor_diagnostic_issue_6464.py \
  --config configs/benchmarks/issue_6464_brne_corridor_diagnostic.yaml \
  --output-dir output/benchmarks/issue_6923_brne_trace_diagnosis_20260812T061500Z
```

Pair coverage was exact for all three arms (`3/3` seeds each), with no
fallback/degraded rows and the pinned BRNE source recorded as clean at
`633a5cdcb39ab27f18b596cb8cb1968644f82391`. The BRNE mechanism trace was
complete and schema-valid for all three rows, with all three rows runtime
eligible, non-degenerate, and inside the approved direct corridor band.
Effective samples remained `42` from the requested `49`.

| Arm | Exact pairs | Mechanism trace | Runtime-eligible | Eligible goal reached | Runtime failure | Corridor violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BRNE | 3/3 | 3/3 | 3/3 | 0/3 | 0/3 | 0 |
| ORCA comparator | 3/3 | common trace only | 3/3 | 3/3 | 0/3 | 0 |
| Social-force comparator | 3/3 | common trace only | 3/3 | 2/3 | 0/3 | 0 |

The compact mechanism table records finite declared heading, goal bearing,
angular difference, world-frame adapter pedestrians, effective sample count,
ensemble layout, per-step runtime status, signed goal-distance progress,
interaction exposure, radius-aware clearance, and terminal events. The BRNE
heading/goal fields do not show a gross frame mismatch in this slice. All three
BRNE rows use the pinned `plan_step_first` aggregation with shape `[25, 42, 2]`
and requested/effective samples `49/42`. Their signed goal-distance progress
has the same broad pattern—early worsening, middle improvement, then late
worsening—and each row terminates in collision before reaching the goal at
steps `141`, `172`, and `183` for seeds `111`, `112`, and `113`.

Decision: **diagnostic-only retention; no bounded adapter revision and no
broader campaign from this result**. If BRNE is pursued, a separately scoped
experiment should test the observed progress reversal and collision mechanism
on the same frozen matrix. This run does not support an objective, planner,
safety, ranking, or paper claim. The durable machine-readable handoff is
recorded in
`docs/context/evidence/issue_6464_brne_corridor_diagnostic_summary.json` under
`follow_up_6923`.

The final report is
`output/benchmarks/issue_6923_brne_trace_diagnosis_20260812T061500Z/`.
Its JSON SHA-256 is
`38afc5acbee0b40cdb049d895b2ac12dee5ae3e5f08a4fb1c6138a803f09ef3c` and its
Markdown SHA-256 is
`f312c0d4b1147a900446619ee81618edfe47146d4d84743e907493c8b6532a13`.
Raw episode files remain ignored and worktree-local; the tracked summary is a
compact handoff, not a raw episode archive.
