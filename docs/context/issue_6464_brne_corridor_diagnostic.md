# Issue #6464 — BRNE Corridor Diagnostic

Status: **diagnostic and mechanism trace complete under declared single-thread isolation; no planner-ranking or benchmark claim**.

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

## Mechanism trace result (#6923)

The provenance-bearing rerun used commit `2f617879125c2113bfb35914a63fb3739fe6ec10`,
the pinned upstream source at `633a5cdcb39ab27f18b596cb8cb1968644f82391`, and
the exact three BRNE seed pairs. All BRNE rows were native, eligible, and
runtime-OK; all recorded 42 effective samples, the
`plan_step_sample_command` layout, and no failure reasons. The initial
velocity-derived heading was unavailable in all three rows because the robot
was stationary at the first decision, so the declared heading is the valid
initial orientation signal.

| Seed | Initial heading-goal Δ (rad) | Progress early / middle / final (m) | Min clearance (m) | Interaction ≤5 m | Action changes | Raw v range → final v (m/s) | Raw angular max → final max | Terminal |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 111 | -0.118 | +1.356 / -5.748 / -11.124 | 0.251 | 0.261 | 13 | 4.2–89.9 → 2.0–2.0 | 41.4 → 1.0 | collision, step 141 |
| 112 | -0.112 | -0.032 / -11.401 / -18.551 | 2.580 | 0.110 | 13 | 4.2–87.4 → 2.0–2.0 | 15.1 → 1.0 | collision, step 172 |
| 113 | +0.058 | +0.176 / -12.011 / -19.678 | 1.465 | 0.147 | 21 | 4.2–88.5 → 2.0–2.0 | 19.3 → 1.0 | collision, step 183 |

The adapter trace also preserves the selected pedestrian world-frame positions
and velocities, the effective sample count, tensor shapes, and the exact
weighted aggregation expression. These observations reject a remaining
runtime/provenance explanation and show that the initial declared heading and
goal bearing are broadly aligned, while supporting a bounded action-scale
hypothesis: every BRNE step produced a raw command outside at least one
configured limit and was clamped before application. This is not a causal
intervention, so it does not establish that aggregation scale caused the
negative outcome.

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

The mechanism-trace report for #6923 is
`output/benchmarks/issue_6923_brne_mechanism_trace_final_20260812T061200Z/`.
Its diagnostic report JSON SHA-256 is
`cb235855d142b1d59f8f54acb0f9ada2ae34d8a301a9ad76c43610e173604151`, its
Markdown SHA-256 is
`13129a70e9bab718e018ed1ec2d6773569a7dc328313b2588563c69275990d01`, and
the trace table SHA-256 is
`5fcb7493b7136636cc55491d14c00976896f1adb1292498d7cf6b64739cf7d3c`.
The report status is `complete`, with exact native BRNE coverage and no
fallback/degraded rows. Raw traces and the staged GPL source remain ignored,
worktree-local artifacts; the compact JSON summary below is the durable pointer.

## Next decision

The explicit decision for #6923 is **diagnostic-only retention**: the trace is
complete, but no bounded planner behavior change is justified without an
intervention. Follow-up [#6934](https://github.com/ll7/robot_sf_ll7/issues/6934)
is the next smallest proof: isolate whether the pinned mean-normalized weights
and the adapter weighted-sum control contract use compatible units, then either
add a bounded contract fix with a repeat of this exact diagnostic or close the
hypothesis without widening the campaign.
